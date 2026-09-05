//! # Mamba-3 — Double-Pass SSD Forward
//!
//! This module provides the [`Mamba3::forward_double_ssd`](crate::mamba3::mamba3::Mamba3::forward_double_ssd) method:
//! The burn-mamba implementation of the [`VikramLex/mamba3-minimal`](https://github.com/VikramLex/mamba3-minimal) decomposition:
//!
//! ```text
//!   hₚ = αₚ hₚ₋₁ + βₚ Bₚ₋ₗₐ₉ ⊗ xₚ₋ₗₐ₉ + γₚ Bₚ ⊗ xₚ   (original double-ssd trapezoidal)
//!
//!   forward:    h = SSD(γ-scaled V, B)   +   SSD(β-scaled V_shifted, B_shifted)
//! ```
//!
//! The shift is [`Trapezoid::tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag)
//! folded positions (`1` by default, `u` for
//! [`Trapezoid::Vertical`](crate::mamba3::trapezoid::Trapezoid::Vertical)), and
//! `β` carries the transport across that whole gap.
//!
//! This is simple to derive and to verify (everything reuses the standard SSD)
//! but increases the intra-chunk and chunk-state memory during training.
//!
//! See also: [`crate::mamba3::mamba3`] and [`crate::mamba3::single_ssd::single_ssd`].

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::{rotate_bc_forward, rotate_bc_step};
use burn_stack::modules::Silu;
use burn_stack::modules::sanity as san;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba3::forward  (chunkwise double-SSD — training / prefill)
// ---------------------------------------------------------------------------

impl Mamba3 {
    /// Process a full input sequence using the (double-ssd) trapezoidal algorithm.
    ///
    /// For SISO (mimo_rank=1), this is the standard double-SSD decomposition.
    /// For MIMO (mimo_rank>1), B/C have mimo_rank parallel rank channels.
    /// The hidden state is shared across mimo ranks; each mimo rank contributes independently.
    ///
    /// # Shapes
    /// - `input_bsm` : `[batch, sequence, d_model]`
    /// - output      : `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward_double_ssd(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3DoubleSsdCache>,
        ssd_path: &Mamba3SsdPath,
    ) -> (Tensor<3>, Mamba3DoubleSsdCache) {
        let [batch, tokens, _d_model] = input_bsm.dims();
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let micro_steps = self.micro_steps;
        let device = input_bsm.device();

        // MambaProduct: from the split below down to the readout, one sequence
        // position *is* one micro-step — the `s` in every shape suffix counts
        // micro-steps, and `tokens` is the only name still at token resolution.
        // See [`crate::mamba3::product`].
        let sequence = tokens * micro_steps;

        assert!(tokens > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        let mut cache = cache.unwrap_or_else(|| {
            let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
            let (k_state_bumhr, v_state_buhp) = self.zero_tap_slots(batch, &device);
            let rotation = self.zero_rotation_state(batch, &device);
            Mamba3DoubleSsdCache {
                ssm_bhpr,
                k_state_bumhr,
                v_state_buhp,
                rotation,
            }
        });

        // ── Step 1: In-projection ─────────────────────────────────────────────
        let proj_bsd = self.in_proj.forward(input_bsm);
        let bc_size = ngroups * state_rank * mimo_rank;

        // [batch, tokens, *] split along channel dim; `u` = micro_steps widens
        // every per-micro-step segment, and `unfold` reinterprets each of them
        // as `u` consecutive sequence positions. `z` (per-token gate) and `C`
        // (per-token read) do not widen — `C` is instead broadcast across the
        // group so its *last* copy carries the right cumulative rotation.
        // b_raw_bsMGR / c_raw_bsMGR have channel size `mimo_rank * ngroups * state_rank`.
        // The two optional segments come off the tail first — `Real1D` projects no
        // rotation and `Trapezoid::None` no `λ`, and a zero-width segment would
        // silently vanish from the fixed-arity split below.
        let u = micro_steps;
        let lambda_channels = self.lambda_channels_total();
        let (proj_bsd, rot_btA) =
            helpers::split_trailing(proj_bsd, self.rotation_channels_total(), 2);
        let (proj_bsd, lambda_raw_btH) = helpers::split_trailing(proj_bsd, lambda_channels, 2);
        #[rustfmt::skip]
        let [
                z_bsi, x_btI,
                b_raw_btMGRU, c_raw_btMGR,
                dd_dt_btH, dd_A_raw_btH,
        ] = burn_stack::modules::split_into(
            proj_bsd,
            [
                d_inner, u * d_inner,
                u * bc_size, bc_size,
                u * nheads, u * nheads,
            ],
            2,
        );

        use crate::mamba3::product::{repeat_micro_bs, unfold_micro_bs};
        let x_bsi = unfold_micro_bs(x_btI, u);
        let b_raw_bsMGR = unfold_micro_bs(b_raw_btMGRU, u);
        let c_raw_bsMGR = repeat_micro_bs(c_raw_btMGR, u);
        let dd_dt_bsh = unfold_micro_bs(dd_dt_btH, u);
        let dd_A_raw_bsh = unfold_micro_bs(dd_A_raw_btH, u);
        let lambda_raw_bsh = lambda_raw_btH.map(|t| unfold_micro_bs(t, u));
        let rot_bsa = rot_btA.map(|t| unfold_micro_bs(t, u));

        san(&z_bsi);
        san(&x_bsi);
        san(&dd_dt_bsh);

        // ── Step 2: Discretisation + trapezoidal coefficients ─────────────────
        let helpers::TrapezoidCoeffs {
            dt: dt_bsh,
            da: da_bsh,
            alpha: _alpha_bsh,
            beta: beta_bsh,
            gamma: gamma_bsh,
        } = helpers::trapezoidal_coefficients(
            dd_dt_bsh,
            dd_A_raw_bsh,
            lambda_raw_bsh,
            self.dt_bias_h.val(),
            self.dt_limit,
            self.a_floor,
        );

        san(&dt_bsh);
        san(&da_bsh);
        if let Some(beta_bsh) = &beta_bsh {
            san(beta_bsh);
        }
        san(&gamma_bsh);

        // ── Step 3: Reshape x ─────────────────────────────────────────────────
        let x_bshp = x_bsi.reshape([batch, sequence, nheads, per_head_dim]);

        // ── Step 4: QK-Norm on B and C  ───────────────────────────────────────
        // QK-Norm over state_rank, then expand ngroups→nheads, then add per-(head,
        // mimo-rank) bias [nheads, mimo_rank, state_rank]. Group dim is axis 3 of
        // `_bsmgr` (D = 5).
        let b_bsmhr = helpers::qk_norm_expand_bias::<5, 6>(
            b_raw_bsMGR.reshape([batch, sequence, mimo_rank, ngroups, state_rank]),
            &self.b_norm,
            self.b_bias_hmr.val(),
            3,
            nheads,
        );
        let c_bsmhr = helpers::qk_norm_expand_bias::<5, 6>(
            c_raw_bsMGR.reshape([batch, sequence, mimo_rank, ngroups, state_rank]),
            &self.c_norm,
            self.c_bias_hmr.val(),
            3,
            nheads,
        );
        assert_eq!(
            [batch, sequence, mimo_rank, nheads, state_rank],
            b_bsmhr.dims()
        );
        assert_eq!(
            [batch, sequence, mimo_rank, nheads, state_rank],
            c_bsmhr.dims()
        );

        // ── Step 5: Data-dependent transition rotation of B and C ─────────────
        // Complex2D: abelian RoPE (cumulative angle). Quaternion4D: cumulative
        // unit quaternion. The new cache accumulator is returned for Step (cache
        // update) below. See [`rotate_bc_forward`].
        let (b_bsmhr, c_bsmhr, new_rotation) = rotate_bc_forward(
            rot_bsa,
            dt_bsh.clone(),
            cache.rotation.clone(),
            b_bsmhr,
            c_bsmhr,
            self.rotation_spec(),
        );
        san(&b_bsmhr);
        san(&c_bsmhr);

        // ── Steps 6–7: the β term's shifted, β-scaled inputs ──────────────────
        //
        // "Shift-Before-Chunking": prepend the cached xₜ₋₁ / Bₜ₋₁ at the
        // sequence level (before SSD chunking) so the β term at t=0 sees the
        // prior token from a continued cache. For a fresh (zero) cache this is
        // equivalent to zero-padding.
        //
        // The shift is [`Trapezoid::tap_lag`] *folded* positions — 1 for the
        // default [`Trapezoid::HorizontalCarryOver`], `u` for
        // [`Trapezoid::Vertical`] — matching `step`'s FIFO depth exactly.
        //
        // A lag-`L` tap must be transported across *its own* gap (§9), i.e. by
        // `Πᵈ⁼⁰..ᴸ⁻¹ αₚ₋ᵈ` rather than `αₚ`. `β` carries the `d = 0` factor and
        // `interior_gap_decay` the rest; for the first `L` positions the missing
        // factors are the ones the cache's `v` slots already carry.
        //
        // Under [`Trapezoid::None`] there is no left endpoint at all: no shift,
        // no β-scaled copy of `x`, and (below) no second SSD call — `forward`
        // becomes one standard SSD pass whose keys are scaled by `γ = Δ`.
        let lag = self.tap_lag();
        let beta_side: Option<(Tensor<4>, Tensor<5>)> = beta_bsh.map(|beta_bsh| {
            let v_state_buhp = cache
                .v_state_buhp
                .clone()
                .expect("a β tap keeps its (B, x) cache slots");
            let k_state_bumhr = cache
                .k_state_bumhr
                .clone()
                .expect("a β tap keeps its (B, x) cache slots");
            let x_prev_bshp = helpers::shift_stream(x_bshp.clone(), v_state_buhp, lag);
            let b_prev_bsmhr = helpers::shift_stream(b_bsmhr.clone(), k_state_bumhr, lag);
            let beta_bsh = match helpers::interior_gap_decay(da_bsh.clone(), lag) {
                Some(gap_bsh) => beta_bsh * gap_bsh,
                None => beta_bsh,
            };
            // β is a per-head scalar, broadcast over mimo_rank and per_head_dim.
            let beta_bsh1 = beta_bsh.unsqueeze_dim::<4>(3);
            (x_prev_bshp * beta_bsh1, b_prev_bsmhr) // βₚ · xₚ₋ₗₐ₉
        });

        // ── Step 7b: Scale the current-token input by γ ───────────────────────
        let gamma_bsh1 = gamma_bsh.unsqueeze_dim::<4>(3);
        let x_gamma_bshp = x_bshp.clone() * gamma_bsh1; // γₜ · xₜ

        // ── Save the last `lag` positions' B and x for the cache ──────────────
        // The last `lag` *micro-steps* — exactly the positions whose taps the
        // next call has to pay, so the trapezoid continues across a split
        // prefill unchanged (at `lag = u` that window is precisely the last
        // token). With no β tap there is nothing to continue and the slots stay
        // empty. See [`Self::save_tap_slots`].
        let (b_last_bumhr, x_last_buhp) =
            self.save_tap_slots(&b_bsmhr, &x_bshp, &da_bsh, lag);

        // ── Step 8: Pad sequence to multiple of chunk_len ─────────────────────
        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        // The zero blocks are built at most once each and shared by the streams
        // that need them (a `Tensor` clone is a handle, not a copy).
        let pads = (pad > 0).then(|| {
            (
                Tensor::<4>::zeros([batch, pad, nheads, per_head_dim], &device),
                Tensor::<3>::zeros([batch, pad, nheads], &device),
                Tensor::<5>::zeros([batch, pad, mimo_rank, nheads, state_rank], &device),
            )
        });
        let pad_hp = |t: Tensor<4>| match &pads {
            Some((p, _, _)) => Tensor::cat(vec![t, p.clone()], 1),
            None => t,
        };
        let pad_h = |t: Tensor<3>| match &pads {
            Some((_, p, _)) => Tensor::cat(vec![t, p.clone()], 1),
            None => t,
        };
        let pad_mhr = |t: Tensor<5>| match &pads {
            Some((_, _, p)) => Tensor::cat(vec![t, p.clone()], 1),
            None => t,
        };

        let x_gamma_bShp = pad_hp(x_gamma_bshp);
        let da_bSh = pad_h(da_bsh);
        let b_bSmhr = pad_mhr(b_bsmhr);
        let c_bSmhr = pad_mhr(c_bsmhr);
        let beta_side =
            beta_side.map(|(x_beta, b_prev)| (pad_hp(x_beta), pad_mhr(b_prev)));

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let x_gamma_bnlhp = x_gamma_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bnlmhr = c_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

        // ── Step 9: the MIMO-SSD call(s) ─────────────────────────────────────────
        // Build V tensors — insert the mimo_rank axis at position 3 of `_bnlhp`.
        // With a β tap this is the pair of standard SSD passes the pathway is
        // named for (γ-SSM + β-SSM, summed); under [`Trapezoid::None`] the second
        // one does not exist and a "double"-SSD forward is a single pass.
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let v_gamma_bnlmhp =
            helpers::build_v_with_mimo::<5, 6>(x_gamma_bnlhp.clone(), mimo_x_hmp.as_ref(), 3);

        let input_gamma = Mamba3DoubleSsdInput {
            v_bnlmhp: v_gamma_bnlmhp,
            da_bnlh: da_bnlh.clone(),
            b_bnlmhr: b_bnlmhr.clone(),
            c_bnlmhr: c_bnlmhr.clone(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        };
        let (y_bnlmhp, final_state_bhpr) = input_gamma.run(ssd_path);

        let (y_bnlmhp, final_state_bhpr) = match beta_side {
            None => (y_bnlmhp, final_state_bhpr),
            Some((x_beta_bShp, b_prev_bSmhr)) => {
                let x_beta_bnlhp =
                    x_beta_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
                let b_prev_bnlmhr =
                    b_prev_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
                let v_beta_bnlmhp =
                    helpers::build_v_with_mimo::<5, 6>(x_beta_bnlhp, mimo_x_hmp.as_ref(), 3);
                let input_beta = Mamba3DoubleSsdInput {
                    v_bnlmhp: v_beta_bnlmhp,
                    da_bnlh,
                    b_bnlmhr: b_prev_bnlmhr,
                    c_bnlmhr,
                    initial_state_bhpr: Tensor::zeros(
                        [batch, nheads, per_head_dim, state_rank],
                        &device,
                    ),
                    init_state_hpr: None,
                };
                let (y_beta_bnlmhp, final_state_beta_bhpr) = input_beta.run(ssd_path);
                (
                    y_bnlmhp + y_beta_bnlmhp,
                    final_state_bhpr + final_state_beta_bhpr,
                )
            }
        };

        san(&y_bnlmhp);
        san(&final_state_bhpr);

        cache.ssm_bhpr = final_state_bhpr;

        // ── Step 10: Unpad ────────────────────────────────────────────────────
        let y_bSmhp = y_bnlmhp.reshape([batch, sequence_padded, mimo_rank, nheads, per_head_dim]);
        let y_bsmhp = if pad == 0 {
            y_bSmhp
        } else {
            y_bSmhp.narrow(1, 0, sequence)
        };

        // ── Step 10b: back to token resolution (MambaProduct) ─────────────────
        // The readout happens after all `u` writes, so only each token's last
        // micro-step survives; the intervening positions computed a `y` from a
        // repeated `C` and it is dropped here. From this point `t` (`tokens`)
        // is the sequence axis again.
        let y_btmhp = crate::mamba3::product::last_micro5(y_bsmhp, micro_steps);
        let x_bthp = crate::mamba3::product::last_micro4(x_bshp.clone(), micro_steps);

        // ── Step 11: D skip + gate + aggregate ranks ──────────────────────────
        // D skip uses raw x * mimo_x_hmp (not gamma-scaled), at the micro-step
        // the readout is contemporaneous with.
        // Insert the mimo_rank axis at position 2 of `_bthp`.
        let v_raw_bsmhp = helpers::build_v_with_mimo::<4, 5>(x_bthp, mimo_x_hmp.as_ref(), 2);
        let y_bsmhp = y_btmhp;
        let sequence = tokens;

        let d_111h1 = self.d_h.val().unsqueeze_dims::<5>(&[0, 1, 2, 4]);
        let y_bsmhp = y_bsmhp + d_111h1 * v_raw_bsmhp.clone();

        // ── Gate (or gated norm) and rank aggregation ─────────────────────────
        // When `out_norm` is set, the SiLU gate is replaced by a per-head
        // gated RMSNorm: `RmsNormGated(y, z) = norm(y) * silu(z)`.
        let y_bsi = if mimo_rank > 1 {
            let mimo_z_hmp = self.mimo_z_hmp.as_ref().map(|p| p.val()).unwrap();
            let mimo_o_hmp = self.mimo_o_hmp.as_ref().map(|p| p.val()).unwrap();

            let z_bshp = z_bsi
                .clone()
                .reshape([batch, sequence, nheads, per_head_dim]);
            let z_bsmhp = {
                let z_bsmhp = z_bshp
                    .unsqueeze_dim::<5>(2) // z_bs1hp
                    .expand([batch, sequence, mimo_rank, nheads, per_head_dim]); // z_bsmhp
                let mimo_z_bsmhp = mimo_z_hmp
                    .swap_dims(0, 1) // mimo_z_mhp
                    .unsqueeze_dims::<5>(&[0, 1]) // mimo_z_11mhp
                    .expand([batch, sequence, mimo_rank, nheads, per_head_dim]); // mimo_z_bsmhp
                z_bsmhp * mimo_z_bsmhp
            };

            // gate or gated norm:
            //   without out_norm: y_r * silu(z_r)
            //   with    out_norm: norm(y_r) * silu(z_r)  (norm over per_head_dim)
            let y_combined_bsmhp = match &self.out_norm {
                Some(norm) => norm.forward(y_bsmhp, z_bsmhp),
                None => y_bsmhp * Silu::new().forward(z_bsmhp),
            };

            // Down-project with mimoₒ_hmp: out = sumₘ mimoₒ_hmp[h, r, p] * yᵣ
            let mimo_o_bsmhp = mimo_o_hmp
                .swap_dims(0, 1) // mimo_o_mhp
                .unsqueeze_dims::<5>(&[0, 1]) // mimo_o_11mhp
                .expand([batch, sequence, mimo_rank, nheads, per_head_dim]); // mimo_o_bsmhp
            // sum over mimo rank dim
            let y_bshp: Tensor<4> = (y_combined_bsmhp * mimo_o_bsmhp)
                .sum_dim(2) // y_bs1hp
                .squeeze_dim(2); // y_bshp
            y_bshp.reshape([batch, sequence, d_inner])
        } else {
            // SISO: squeeze rank dim, apply gate (or gated norm) over per_head_dim.
            let y_bshp: Tensor<4> = y_bsmhp.squeeze_dim(2); // mimo_rank == 1
            let z_bshp = z_bsi.reshape([batch, sequence, nheads, per_head_dim]);
            let y_combined_bshp = match &self.out_norm {
                Some(norm) => norm.forward(y_bshp, z_bshp),
                None => y_bshp * Silu::new().forward(z_bshp),
            };
            y_combined_bshp.reshape([batch, sequence, d_inner])
        };
        san(&y_bsi);

        // ── Out-projection ────────────────────────────────────────────────────
        let out_bsm = self.out_proj.forward(y_bsi);
        san(&out_bsm);

        // ── Update remaining cache fields ─────────────────────────────────────
        // k_state / v_state = the tap FIFO's last `lag` positions (both `None`
        // when the pattern has no β tap).
        cache.k_state_bumhr = b_last_bumhr;
        cache.v_state_buhp = x_last_buhp;

        // Cumulative rotation at the last micro-step (angle wrapped to [−π, π], or
        // the cumulative quaternion), to continue a longer sequence.
        cache.rotation = new_rotation;

        (out_bsm, cache)
    }
}

// ---------------------------------------------------------------------------
// Mamba3::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    /// One token's in-projection unpacked into the step-shaped pieces
    /// [`Mamba3::step_double_ssd`] works from: the gate/value streams, the
    /// **pre-rotation** QK-normed B/C, the raw rotation channels, and the
    /// trapezoid coefficients.
    ///
    /// Every per-micro-step stream carries a `u` axis (MambaProduct; `u = 1` for
    /// stock Mamba-3). [`Self::micro`] peels one micro-step off it.
    pub(crate) struct StepProjection {
        /// Recurrence micro-steps per token — the size of the `u` axis below.
        pub micro_steps: usize,
        /// Per-token gate stream `[batch, d_inner]`.
        pub z_bi: Tensor<2>,
        /// Per-token QK-normed, GQA-expanded, biased C — **before** the
        /// rotation. The read happens once, after all `u` writes.
        pub c_bmhr: Tensor<4>,
        /// Value stream `[batch, u, nheads, per_head_dim]`.
        pub x_buhp: Tensor<4>,
        /// QK-normed, GQA-expanded, biased B — **before** the rotation.
        /// `[batch, u, mimo_rank, nheads, state_rank]`.
        pub b_bumhr: Tensor<5>,
        /// Raw rotation channels `[batch, u, num_rotation_channels]`; `None` for
        /// [`RotationKind::Real1D`](crate::mamba3::rotation::RotationKind::Real1D),
        /// which projects none.
        pub rot_bua: Option<Tensor<3>>,
        /// `Δ` `[batch, u, nheads]`.
        pub dt_buh: Tensor<3>,
        /// `α = exp(Δ·A)` `[batch, u, nheads]`.
        pub alpha_buh: Tensor<3>,
        /// `β = (1−λ)·Δ·α` `[batch, u, nheads]`; `None` under
        /// [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None).
        pub beta_buh: Option<Tensor<3>>,
        /// `γ = λ·Δ` `[batch, u, nheads]` (`= Δ` when there is no `λ`).
        pub gamma_buh: Tensor<3>,
    }

    /// One micro-step of a [`StepProjection`], with the `u` axis peeled off.
    pub(crate) struct MicroProjection {
        /// Value stream `[batch, nheads, per_head_dim]`.
        pub x_bhp: Tensor<3>,
        /// Pre-rotation B `[batch, mimo_rank, nheads, state_rank]`.
        pub b_bmhr: Tensor<4>,
        /// Raw rotation channels `[batch, num_rotation_channels]`.
        pub rot_ba: Option<Tensor<2>>,
        /// `Δ` `[batch, nheads]`.
        pub dt_bh: Tensor<2>,
        /// `α = exp(Δ·A)` `[batch, nheads]`.
        pub alpha_bh: Tensor<2>,
        /// `β = (1−λ)·Δ·α` `[batch, nheads]`; `None` under
        /// [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None).
        pub beta_bh: Option<Tensor<2>>,
        /// `γ = λ·Δ` `[batch, nheads]` (`= Δ` when there is no `λ`).
        pub gamma_bh: Tensor<2>,
    }

    impl StepProjection {
        /// Micro-step `j` (`0 ≤ j < micro_steps`), in execution order.
        pub fn micro(&self, j: usize) -> MicroProjection {
            let pick2 = |t: Tensor<3>| t.narrow(1, j, 1).squeeze_dim::<2>(1);
            MicroProjection {
                x_bhp: self.x_buhp.clone().narrow(1, j, 1).squeeze_dim(1),
                b_bmhr: self.b_bumhr.clone().narrow(1, j, 1).squeeze_dim(1),
                rot_ba: self.rot_bua.clone().map(pick2),
                dt_bh: pick2(self.dt_buh.clone()),
                alpha_bh: pick2(self.alpha_buh.clone()),
                beta_bh: self.beta_buh.clone().map(pick2),
                gamma_bh: pick2(self.gamma_buh.clone()),
            }
        }
    }

    impl Mamba3 {
        /// In-projection → split → trapezoid coefficients → QK-norm for a
        /// single token, **stopping before** the rotation (which
        /// needs the cache's cumulative rotation).
        ///
        /// The per-micro-step streams keep a `u` axis; see [`StepProjection`].
        #[allow(non_snake_case)]
        pub(crate) fn step_project(&self, input_bd: Tensor<2>) -> StepProjection {
            let [batch, _d_model] = input_bd.dims();
            let d_inner = self.d_inner();
            let nheads = self.nheads();
            let ngroups = self.ngroups;
            let per_head_dim = self.per_head_dim();
            let state_rank = self.state_rank;
            let mimo_rank = self.mimo_rank;
            let u = self.micro_steps;

            assert_eq!(nheads % ngroups, 0);
            san(&input_bd);

            // ── In-projection ─────────────────────────────────────────────────
            let proj_bd = self.in_proj.forward(input_bd);
            san(&proj_bd);
            let bc_size = ngroups * state_rank * mimo_rank;
            // [batch, *] split along channel dim; the per-micro-step segments
            // are `u` times as wide and split onto a `u` axis of their own,
            // matching `forward`'s fold into the sequence.
            // b_raw_bMGR / c_raw_bMGR have channel size `mimo_rank * ngroups * state_rank`.
            // See the note in `forward`: the two trailing segments are the
            // optional ones (`Real1D` projects no rotation, `Trapezoid::None` no `λ`).
            let (proj_bd, rot_bA) =
                helpers::split_trailing(proj_bd, self.rotation_channels_total(), 1);
            let (proj_bd, lambda_raw_bH) =
                helpers::split_trailing(proj_bd, self.lambda_channels_total(), 1);
            #[rustfmt::skip]
            let [
                    z_bi, x_bI,
                    b_raw_bMGRU, c_raw_bMGR,
                    dd_dt_bH, dd_a_raw_bH,
            ] = burn_stack::modules::split_into(
                proj_bd,
                [
                    d_inner, u * d_inner,
                    u * bc_size, bc_size,
                    u * nheads, u * nheads,
                ],
                1,
            );

            use crate::mamba3::product::unfold_micro_b;
            let rot_bua = rot_bA.map(|t| unfold_micro_b(t, u));

            // ── Reshape x ─────────────────────────────────────────────────────
            let x_buhp = x_bI.reshape([batch, u, nheads, per_head_dim]);

            // ── Discretisation + trapezoidal coefficients ─────────────────────
            let helpers::TrapezoidCoeffs {
                dt: dt_buh,
                da: _da_buh,
                alpha: alpha_buh,
                beta: beta_buh,
                gamma: gamma_buh,
            } = helpers::trapezoidal_coefficients(
                unfold_micro_b(dd_dt_bH, u),
                unfold_micro_b(dd_a_raw_bH, u),
                lambda_raw_bH.map(|t| unfold_micro_b(t, u)),
                self.dt_bias_h.val(),
                self.dt_limit,
                self.a_floor,
            );
            san(&dt_buh);
            san(&alpha_buh);
            if let Some(beta_buh) = &beta_buh {
                san(beta_buh);
            }
            san(&gamma_buh);

            // ── QK-Norm on B and C ────────────────────────────────────────────
            // B carries the `u` axis, so its group dim is axis 3 of `_bumgr`
            // (D = 5); C is per token, group dim axis 2 of `_bmgr` (D = 4).
            let b_bumhr = helpers::qk_norm_expand_bias::<5, 6>(
                b_raw_bMGRU.reshape([batch, u, mimo_rank, ngroups, state_rank]),
                &self.b_norm,
                self.b_bias_hmr.val(),
                3,
                nheads,
            );
            let c_bmhr = helpers::qk_norm_expand_bias::<4, 5>(
                c_raw_bMGR.reshape([batch, mimo_rank, ngroups, state_rank]),
                &self.c_norm,
                self.c_bias_hmr.val(),
                2,
                nheads,
            );
            assert_eq!([batch, u, mimo_rank, nheads, state_rank], b_bumhr.dims());
            san(&b_bumhr);
            san(&c_bmhr);

            StepProjection {
                micro_steps: u,
                z_bi,
                c_bmhr,
                x_buhp,
                b_bumhr,
                rot_bua,
                dt_buh,
                alpha_buh,
                beta_buh,
                gamma_buh,
            }
        }

        /// State→output contraction:
        /// `out[b, m, h, p] = Σᵣ C[b, m, h, r] · state[b, h, p, r]`
        /// (`einsum('bhpr,bmhr->bmhp', state, C)`).
        ///
        /// At `mimo_rank == 1` the output axis of the GEMM is 1, so the matmul
        /// is a matrix–vector product; [`step_readout_siso`](Mamba3::step_readout_siso)
        /// writes it as a broadcast multiply plus a `state_rank` reduction.
        /// Selected by
        /// [`Mamba3Config::siso_specialization_decode`](crate::mamba3::mamba3::Mamba3Config::siso_specialization_decode)
        /// — both branches compute the same values and gradients.
        pub(crate) fn step_readout(
            state_bhpr: Tensor<4>,
            c_bmhr: Tensor<4>,
            siso_specialization: bool,
        ) -> Tensor<4> {
            let [_batch, mimo_rank, _nheads, _state_rank] = c_bmhr.dims();
            if mimo_rank == 1 && siso_specialization {
                Self::step_readout_siso(state_bhpr, c_bmhr)
            } else {
                Self::step_readout_mimo(state_bhpr, c_bmhr)
            }
        }

        /// SISO (`mimo_rank == 1`) state→output contraction: broadcast `C` over
        /// `per_head_dim` and reduce `state_rank`.
        pub(crate) fn step_readout_siso(state_bhpr: Tensor<4>, c_bmhr: Tensor<4>) -> Tensor<4> {
            let c_bh1r: Tensor<4> = c_bmhr.squeeze_dim::<3>(1).unsqueeze_dim(2);
            let out_bhp1: Tensor<4> = (state_bhpr * c_bh1r).sum_dim(3);
            out_bhp1.squeeze_dim::<3>(3).unsqueeze_dim(1) // out_b1hp
        }

        /// General MIMO state→output contraction: one matmul over `state_rank`.
        pub(crate) fn step_readout_mimo(state_bhpr: Tensor<4>, c_bmhr: Tensor<4>) -> Tensor<4> {
            let c_bhrm = c_bmhr.permute([0, 2, 3, 1]);
            let out_bhpm = state_bhpr.matmul(c_bhrm);
            out_bhpm.permute([0, 3, 1, 2])
        }

        /// Shared block tail: `D` skip, gate (or gated RMSNorm), MIMO rank
        /// aggregation, and the output projection.
        ///
        /// `out_m_bmhp` is the raw SSM readout (see [`Mamba3::step_readout`]);
        /// `x_vals_bmhp` the MIMO-expanded values; `z_bi` the gate stream.
        pub(crate) fn step_finish(
            &self,
            out_m_bmhp: Tensor<4>,
            x_vals_bmhp: Tensor<4>,
            z_bi: Tensor<2>,
        ) -> Tensor<2> {
            let [batch, mimo_rank, nheads, per_head_dim] = x_vals_bmhp.dims();
            let d_inner = self.d_inner();

            // D skip
            let d_bmhp = self
                .d_h
                .val()
                .unsqueeze_dims::<4>(&[0, 1, 3]) // d_11h1
                .expand([batch, mimo_rank, nheads, per_head_dim]); // d_bmhp
            let out_m_bmhp = out_m_bmhp + d_bmhp * x_vals_bmhp;
            san(&out_m_bmhp);

            // ── Gate (or gated norm) and rank aggregation ─────────────────────
            // When `out_norm` is set, the SiLU gate is replaced by a per-head
            // gated RMSNorm: `RmsNormGated(y, z) = norm(y) * silu(z)`.
            let z_bhp = z_bi.reshape([batch, nheads, per_head_dim]);
            let y_bi = if mimo_rank > 1 {
                let mimo_z_hmp = self.mimo_z_hmp.as_ref().map(|p| p.val()).unwrap();
                let mimo_o_hmp = self.mimo_o_hmp.as_ref().map(|p| p.val()).unwrap();

                // zₘ = z * mimo_z_hmp[m]
                let z_bmhp = z_bhp
                    .unsqueeze_dim::<4>(1) // z_b1hp
                    .expand([batch, mimo_rank, nheads, per_head_dim]); // z_bmhp
                // mimo_z_hmp
                let mimo_z_bmhp = mimo_z_hmp
                    .swap_dims(0, 1) // mimo_z_mhp
                    .unsqueeze_dim::<4>(0) // mimo_z_1mhp
                    .expand([batch, mimo_rank, nheads, per_head_dim]); // mimo_z_bmhp
                let z_bmhp = z_bmhp * mimo_z_bmhp;
                san(&z_bmhp);

                // Per-rank gate or gated norm.
                let combined_bmhp = match &self.out_norm {
                    Some(norm) => norm.forward(out_m_bmhp, z_bmhp),
                    None => out_m_bmhp * Silu::new().forward(z_bmhp),
                };
                san(&combined_bmhp);

                // Project down: out = sumₘ mimo_o_hmp[m] * combined_bmhp[m]
                let mimo_o_bmhp = mimo_o_hmp
                    .swap_dims(0, 1) // mimo_o_mhp
                    .unsqueeze_dim::<4>(0) // mimo_o_1mhp
                    .expand([batch, mimo_rank, nheads, per_head_dim]); // mimo_o_bmhp
                let out_bhp: Tensor<3> = (combined_bmhp * mimo_o_bmhp)
                    .sum_dim(1) // out_b1hp
                    .squeeze_dim(1); // out_bhp
                san(&out_bhp);
                out_bhp.reshape([batch, d_inner]) // y_bi
            } else {
                // SISO: squeeze rank dim, gate (or gated norm) over per_head_dim.
                let y_bhp: Tensor<3> = out_m_bmhp.squeeze_dim(1);
                let combined = match &self.out_norm {
                    Some(norm) => norm.forward(y_bhp, z_bhp),
                    None => y_bhp * Silu::new().forward(z_bhp),
                };
                san(&combined);
                combined.reshape([batch, d_inner])
            };

            // ── Out-projection ────────────────────────────────────────────────
            let out_bm = self.out_proj.forward(y_bi);
            san(&out_bm);
            out_bm
        }

        /// Process a **single token** using the pure recurrent form.
        ///
        /// For SISO (mimo_rank=1):
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ
        ///   yₜ = Cₜᵀ hₜ + D xₜ
        /// ```
        ///
        /// For MIMO (mimo_rank>1):
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + Σₘ βₜ Bₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_x_hmp[m]) + Σₘ γₜ Bₜ[m] ⊗ (xₜ ⊙ mimo_x_hmp[m])
        ///   yₜ[r] = Cₜ[r]ᵀ hₜ + D xₜ ⊙ mimo_x_hmp[r]
        ///   outₜ = Σₘ mimo_o_hmp[m] ⊙ silu(zₜ ⊙ mimo_z_hmp[m]) ⊙ yₜ[m]
        /// ```
        ///
        /// # Shapes
        /// - `input_bd` : `[batch, d_model]`
        /// - output     : `[batch, d_model]`
        #[allow(non_snake_case)]
        pub fn step_double_ssd(
            &self,
            input_bd: Tensor<2>,
            cache: Option<Mamba3DoubleSsdCache>,
        ) -> (Tensor<2>, Mamba3DoubleSsdCache) {
            let [batch, _d_model] = input_bd.dims();
            let nheads = self.nheads();
            let per_head_dim = self.per_head_dim();
            let state_rank = self.state_rank;
            let device = &input_bd.device();
            let ssm_shape = [batch, nheads, per_head_dim, state_rank];

            let mut cache = cache.unwrap_or_else(|| {
                let ssm_bhpr = Tensor::zeros(ssm_shape, device);
                let (k_state_bumhr, v_state_buhp) = self.zero_tap_slots(batch, device);
                let rotation = self.zero_rotation_state(batch, device);
                Mamba3DoubleSsdCache {
                    ssm_bhpr,
                    k_state_bumhr,
                    v_state_buhp,
                    rotation,
                }
            });

            // ── In-projection → coefficients → QK-norm ────────────────────────
            let proj = self.step_project(input_bd);
            let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
            let siso = self.use_siso_decode_kernels();

            // ── `u` micro-steps of the recurrence, then one readout ────────────
            // MambaProduct: each pass is one plain Mamba-3 step, so the token's
            // transition is the product of the `u` of them and the trapezoid's
            // two taps straddle *micro-steps*. Identical to `forward` running
            // over the folded sequence, which is what the parity tests assert.
            // `u = 1` executes the loop once and is stock Mamba-3.
            //
            // The taps below are a FIFO [`Trapezoid::tap_lag`] deep, oldest
            // first: lag 1 is [`Trapezoid::HorizontalCarryOver`] (the tap
            // crosses a token only at `j = 0`), lag `u` is
            // [`Trapezoid::Vertical`] (every tap crosses, reading the same
            // micro-step of the previous token). Per micro-step the FIFO is
            // **tapped, then the survivors are decayed, then the new position is
            // pushed** — so a slot's `x` accumulates exactly the `α`s between
            // its own position and the one that taps it, which is the gap
            // transport a lag-`L` tap needs (§9). At lag 1 nothing survives a
            // tap, so no decay is ever applied and `β = (1−λ)Δα` is the whole
            // coefficient, as before.
            let lag = self.tap_lag();
            let mut state_bhpr = cache.ssm_bhpr.clone();
            let mut rotation = cache.rotation.clone();
            let mut tap_b: Vec<Tensor<4>> = Vec::new();
            let mut tap_x: Vec<Tensor<3>> = Vec::new();
            if let (Some(k_state_bumhr), Some(v_state_buhp)) =
                (cache.k_state_bumhr.clone(), cache.v_state_buhp.clone())
            {
                for slot in 0..lag {
                    tap_b.push(k_state_bumhr.clone().narrow(1, slot, 1).squeeze_dim(1));
                    tap_x.push(v_state_buhp.clone().narrow(1, slot, 1).squeeze_dim(1));
                }
            }
            // Set on the last pass; the readout uses only that one.
            let mut last: Option<(Tensor<4>, Tensor<4>)> = None;

            for j in 0..proj.micro_steps {
                let m = proj.micro(j);

                // ── Update cumulative rotation, rotate B and C ─────────────
                // Complex2D: abelian RoPE angle. Quaternion4D: cumulative
                // quaternion. See [`rotate_bc_step`]. `C` is the same tensor at
                // every micro-step — matching `forward`'s repeat — so the copy
                // that reaches the readout carries the cumulative rotation of
                // the *last* micro-step.
                let (b_bmhr, c_bmhr, new_rotation) = rotate_bc_step(
                    m.rot_ba,
                    m.dt_bh,
                    rotation,
                    m.b_bmhr,
                    proj.c_bmhr.clone(),
                    self.rotation_spec(),
                );
                san(&b_bmhr);
                san(&c_bmhr);
                new_rotation.sanity();

                // ── Build MIMO value tensors ───────────────────────────────
                // Insert the mimo_rank axis at position 1 of `_bhp`.
                let x_vals_bmhp =
                    helpers::build_v_with_mimo::<3, 4>(m.x_bhp.clone(), mimo_x_hmp.as_ref(), 1);
                san(&x_vals_bmhp);

                // ── SSM state update ───────────────────────────────────────
                // new_state[b, h, p, r] = alpha * state
                //   + sumₘ gamma * x_vals[m] ⊗ B_cur[m]
                //   + sumₘ beta  * xs_vals[m] ⊗ B_state[m]   (β tap only)
                //
                // For the outer product sum:
                //   xBt[b, h, p, r] = sumₘ coeff[m, h, p] * B[m, h, n]
                //   = einsum('bmhp,bmhr->bhpr', coeff*x_vals, B)
                //   = matmul over m: [b, h, p, m] @ [b, h, m, r]
                // x_vals_bmhp * gamma_b1h1
                // Need gamma as [b, 1, h, 1] to broadcast over m and p:
                let gamma_b1h1 = m.gamma_bh.unsqueeze_dims::<4>(&[1, 3]);
                let x_gamma_bmhp = x_vals_bmhp.clone() * gamma_b1h1;
                san(&x_gamma_bmhp);

                // einsum('bmhp,bmhr->bhpr', x_gamma, B_cur):
                let xbt_state_bhpr = helpers::mimo_outer_sum(x_gamma_bmhp, b_bmhr.clone(), siso);
                san(&xbt_state_bhpr);

                // The tapped step's write — the FIFO's oldest slot, i.e. the
                // position `lag` back. Under `Trapezoid::None` there is no
                // second tap: no previous value tensor, no second outer product,
                // and one fewer term in the state update.
                let xbt_prev_bhpr = m.beta_bh.map(|beta_bh| {
                    let x_prev_bhp = tap_x.remove(0);
                    let b_prev_bmhr = tap_b.remove(0);
                    let xs_vals_bmhp =
                        helpers::build_v_with_mimo::<3, 4>(x_prev_bhp, mimo_x_hmp.as_ref(), 1);
                    san(&xs_vals_bmhp);
                    let x_beta_bmhp = xs_vals_bmhp * beta_bh.unsqueeze_dims::<4>(&[1, 3]);
                    san(&x_beta_bmhp);
                    let xbt_prev_bhpr =
                        helpers::mimo_outer_sum(x_beta_bmhp, b_prev_bmhr, siso);
                    san(&xbt_prev_bhpr);
                    xbt_prev_bhpr
                });

                let alpha_bh11 = m.alpha_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
                let new_state_bhpr = alpha_bh11 * state_bhpr + xbt_state_bhpr;
                let new_state_bhpr = match xbt_prev_bhpr {
                    Some(xbt_prev_bhpr) => new_state_bhpr + xbt_prev_bhpr,
                    None => new_state_bhpr,
                };
                san(&new_state_bhpr);

                state_bhpr = new_state_bhpr;
                if lag > 0 {
                    // Decay the slots that survived this step's tap, then push
                    // this position: each slot's `x` ends up carrying `Πα` from
                    // its own position to the current one.
                    let alpha_bh1 = m.alpha_bh.clone().unsqueeze_dim::<3>(2);
                    for x_bhp in tap_x.iter_mut() {
                        *x_bhp = x_bhp.clone() * alpha_bh1.clone();
                    }
                    tap_b.push(b_bmhr);
                    tap_x.push(m.x_bhp);
                }
                rotation = new_rotation;
                last = Some((c_bmhr, x_vals_bmhp));
            }

            let (c_bmhr, x_vals_bmhp) = last.expect("micro_steps ≥ 1");

            // ── Output ────────────────────────────────────────────────────────
            // outₘ[b, m, h, p] = sumᵣ C[b, m, h, r] * state[b, h, p, r] + D * x_vals[b, m, h, p]
            let out_m_bmhp = Self::step_readout(state_bhpr.clone(), c_bmhr, siso);
            san(&out_m_bmhp);

            // ── D skip, gate (or gated norm), rank aggregation, out-projection ─
            let out_bm = self.step_finish(out_m_bmhp, x_vals_bmhp, proj.z_bi);

            // ── Update cache ──────────────────────────────────────────────────
            // The FIFO, re-stacked onto its slot axis (oldest first) — the same
            // layout, and the same decayed-`x` convention, `forward` writes.
            cache.ssm_bhpr = state_bhpr;
            cache.k_state_bumhr = (lag > 0).then(|| {
                Tensor::cat(
                    tap_b.into_iter().map(|t| t.unsqueeze_dim::<5>(1)).collect(),
                    1,
                )
            });
            cache.v_state_buhp = (lag > 0).then(|| {
                Tensor::cat(
                    tap_x.into_iter().map(|t| t.unsqueeze_dim::<4>(1)).collect(),
                    1,
                )
            });
            cache.rotation = rotation;

            (out_bm, cache)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
