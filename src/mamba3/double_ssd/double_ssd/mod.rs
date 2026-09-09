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
//! `β = ν·α` carries the transport across that whole gap.
//!
//! A two-tap pattern adds a second `β` side at lag 1. The two shifts differ, so
//! the passes cannot be fused and this pathway runs **three** SSD calls — the
//! single-SSD one collapses them into its key scale and stays at one, which is
//! where such a pattern wants to run.
//!
//! This is simple to derive and to verify (everything reuses the standard SSD)
//! but increases the intra-chunk and chunk-state memory during training.
//!
//! See also: [`crate::mamba3::mamba3`] and [`crate::mamba3::single_ssd::single_ssd`].

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::rotate_bc_forward;
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
        // The three optional segments come off the tail first — `Real1D` projects
        // no rotation, a one-tap pattern no `μ` and `Trapezoid::None` no `λ` —
        // since a zero-width segment would silently vanish from the fixed-arity
        // split below.
        let u = micro_steps;
        let (proj_bsd, rot_btA) =
            helpers::split_trailing(proj_bsd, self.rotation_channels_total(), 2);
        let (proj_bsd, mu_raw_btH) = helpers::split_trailing(proj_bsd, self.mu_channels_total(), 2);
        let (proj_bsd, lambda_raw_btH) =
            helpers::split_trailing(proj_bsd, self.lambda_channels_total(), 2);
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

        use crate::mamba3::product::unfold_micro_bs;
        let x_bsi = unfold_micro_bs(x_btI, u);
        let b_raw_bsMGR = unfold_micro_bs(b_raw_btMGRU, u);
        // `C` stays at token resolution: the readout happens once per token, at
        // its last micro-step. See [`Mamba3DoubleSsdInput::c_bntmhr`].
        let dd_dt_bsh = unfold_micro_bs(dd_dt_btH, u);
        let dd_A_raw_bsh = unfold_micro_bs(dd_A_raw_btH, u);
        let lambda_raw_bsh = lambda_raw_btH.map(|t| unfold_micro_bs(t, u));
        let mu_raw_bsh = mu_raw_btH.map(|t| unfold_micro_bs(t, u));
        let rot_bsa = rot_btA.map(|t| unfold_micro_bs(t, u));

        san(&z_bsi);
        san(&x_bsi);
        san(&dd_dt_bsh);

        // ── Step 2: Discretisation + trapezoidal coefficients ─────────────────
        let helpers::TrapezoidCoeffs {
            dt: dt_bsh,
            da: da_bsh,
            alpha: alpha_bsh,
            nu: nu_bsh,
            nu_interior: nu_interior_bsh,
            gamma: gamma_bsh,
        } = helpers::trapezoidal_coefficients(
            dd_dt_bsh,
            dd_A_raw_bsh,
            lambda_raw_bsh,
            mu_raw_bsh,
            self.dt_bias_h.val(),
            self.trapezoid_spec(),
        );

        san(&dt_bsh);
        san(&da_bsh);
        for nu_bsh in [&nu_bsh, &nu_interior_bsh].into_iter().flatten() {
            san(nu_bsh);
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
        let c_btmhr = helpers::qk_norm_expand_bias::<5, 6>(
            c_raw_btMGR.reshape([batch, tokens, mimo_rank, ngroups, state_rank]),
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
            [batch, tokens, mimo_rank, nheads, state_rank],
            c_btmhr.dims()
        );

        // ── Step 5: Data-dependent transition rotation of B and C ─────────────
        // Complex2D: abelian RoPE (cumulative angle). Quaternion4D: cumulative
        // unit quaternion. The new cache accumulator is returned for Step (cache
        // update) below. See [`rotate_bc_forward`].
        // `C` is rotated at token resolution too: it needs the cumulative
        // rotation of the micro-step it is read at.
        let (b_bsmhr, c_btmhr, new_rotation) = rotate_bc_forward(
            rot_bsa,
            dt_bsh.clone(),
            cache.rotation.clone(),
            b_bsmhr,
            c_btmhr,
            u,
            self.rotation_spec(),
        );
        san(&b_bsmhr);
        san(&c_btmhr);

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
        // `Πᵈ⁼⁰..ᴸ⁻¹ αₚ₋ᵈ` rather than `αₚ`. `β = ν·α` carries the `d = 0` factor
        // and `interior_gap_decay` the rest; for the first `L` positions the
        // missing factors are the ones the cache's `v` slots already carry.
        //
        // A two-tap pattern adds a second, lag-1 side. The two shifts differ, so
        // they cannot share a pass and this pathway runs three SSD calls; the
        // single-SSD one fuses them into its key scale and stays at one. The
        // interior tap's prefix is the FIFO's newest slot — which *is* the lag-1
        // slot, undecayed — and is inert under
        // [`Trapezoid::VerticalPlusHorizontalReset`], whose `νⁱⁿᵗ` is zero at
        // every position that would read it.
        //
        // Under [`Trapezoid::None`] there is no left endpoint at all: no shift,
        // no β-scaled copy of `x`, and (below) no second SSD call — `forward`
        // becomes one standard SSD pass whose keys are scaled by `γ = Δ`.
        let lag = self.tap_lag();
        let beta_side = |nu_bsh: Tensor<3>, lag: usize| -> (Tensor<4>, Tensor<5>) {
            let slots_buhp = cache
                .v_state_buhp
                .clone()
                .expect("a β tap keeps its (B, x) cache slots");
            let slots_bumhr = cache
                .k_state_bumhr
                .clone()
                .expect("a β tap keeps its (B, x) cache slots");
            // A lag-`L` tap's prefix is the FIFO's newest `L` slots — all of it
            // for the pattern's own tap, the last one alone for an interior tap.
            let newest = slots_buhp.dims()[1] - lag;
            let x_prev_bshp =
                helpers::shift_stream(x_bshp.clone(), slots_buhp.narrow(1, newest, lag), lag);
            let b_prev_bsmhr =
                helpers::shift_stream(b_bsmhr.clone(), slots_bumhr.narrow(1, newest, lag), lag);
            let beta_bsh = nu_bsh * alpha_bsh.clone();
            let beta_bsh = match helpers::interior_gap_decay(da_bsh.clone(), lag) {
                Some(gap_bsh) => beta_bsh * gap_bsh,
                None => beta_bsh,
            };
            // β is a per-head scalar, broadcast over mimo_rank and per_head_dim.
            let beta_bsh1 = beta_bsh.unsqueeze_dim::<4>(3);
            (x_prev_bshp * beta_bsh1, b_prev_bsmhr) // βₚ · xₚ₋ₗₐ₉
        };
        let beta_sides: Vec<(Tensor<4>, Tensor<5>)> = nu_bsh
            .map(|nu_bsh| beta_side(nu_bsh, lag))
            .into_iter()
            .chain(nu_interior_bsh.map(|nu_bsh| beta_side(nu_bsh, 1)))
            .collect();

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
        let chunk_len = ssd_path.chunk_len_or_optimal(self);
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

        // `C` rides the chunk's read axis, so it pads by whole tokens.
        // `chunk_len` is a multiple of `u`, hence so is `pad`.
        let tokens_padded = sequence_padded / u;
        let pad_tmhr = |t: Tensor<5>| match pad {
            0 => t,
            _ => Tensor::cat(
                vec![
                    t,
                    Tensor::zeros(
                        [batch, tokens_padded - tokens, mimo_rank, nheads, state_rank],
                        &device,
                    ),
                ],
                1,
            ),
        };

        let x_gamma_bShp = pad_hp(x_gamma_bshp);
        let da_bSh = pad_h(da_bsh);
        let b_bSmhr = pad_mhr(b_bsmhr);
        let c_bTmhr = pad_tmhr(c_btmhr);
        let beta_sides: Vec<_> = beta_sides
            .into_iter()
            .map(|(x_beta, b_prev)| (pad_hp(x_beta), pad_mhr(b_prev)))
            .collect();

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let chunk_tokens = Mamba3SsdPath::chunk_tokens(chunk_len, u);
        let x_gamma_bnlhp = x_gamma_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bntmhr =
            c_bTmhr.reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]);

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
            c_bntmhr: c_bntmhr.clone(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
            read_stride: u,
        };
        let (y_bntmhp, final_state_bhpr) = input_gamma.run(ssd_path);

        let (y_bntmhp, final_state_bhpr) = beta_sides.into_iter().fold(
            (y_bntmhp, final_state_bhpr),
            |(y_bntmhp, final_state_bhpr), (x_beta_bShp, b_prev_bSmhr)| {
                let x_beta_bnlhp =
                    x_beta_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
                let b_prev_bnlmhr =
                    b_prev_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
                let v_beta_bnlmhp =
                    helpers::build_v_with_mimo::<5, 6>(x_beta_bnlhp, mimo_x_hmp.as_ref(), 3);
                let input_beta = Mamba3DoubleSsdInput {
                    v_bnlmhp: v_beta_bnlmhp,
                    da_bnlh: da_bnlh.clone(),
                    b_bnlmhr: b_prev_bnlmhr,
                    c_bntmhr: c_bntmhr.clone(),
                    initial_state_bhpr: Tensor::zeros(
                        [batch, nheads, per_head_dim, state_rank],
                        &device,
                    ),
                    init_state_hpr: None,
                    read_stride: u,
                };
                let (y_beta_bntmhp, final_state_beta_bhpr) = input_beta.run(ssd_path);
                (
                    y_bntmhp + y_beta_bntmhp,
                    final_state_bhpr + final_state_beta_bhpr,
                )
            },
        );

        san(&y_bntmhp);
        san(&final_state_bhpr);

        cache.ssm_bhpr = final_state_bhpr;

        // ── Step 10: Unpad ────────────────────────────────────────────────────
        // The SSD returns token resolution: the readout happens after all `u`
        // writes, and the kernel only ever computed the row it happens at. From
        // this point `t` (`tokens`) is the sequence axis again.
        let y_bTmhp = y_bntmhp.reshape([batch, tokens_padded, mimo_rank, nheads, per_head_dim]);
        let y_btmhp = if pad == 0 {
            y_bTmhp
        } else {
            y_bTmhp.narrow(1, 0, tokens)
        };
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
    /// stock Mamba-3) — the token's whole folded block, which
    /// [`Mamba3::step_double_ssd`] consumes at once rather than a position at a
    /// time.
    pub(crate) struct StepProjection {
        /// Per-token gate stream `[batch, d_inner]`.
        pub z_bi: Tensor<2>,
        /// Per-token QK-normed, GQA-expanded, biased C — **before** the
        /// rotation. The read happens once, after all `u` writes, so this is
        /// the chunk's read axis with a single row: `[batch, 1, mimo_rank,
        /// nheads, state_rank]`.
        pub c_b1mhr: Tensor<5>,
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
        /// `Δ·A` `[batch, u, nheads]`, the log-decay — the block's transport
        /// (both within it and into the tap slots it leaves behind) is a
        /// product of `α`s, i.e. a sum of these.
        pub da_buh: Tensor<3>,
        /// `α = exp(Δ·A)` `[batch, u, nheads]`.
        pub alpha_buh: Tensor<3>,
        /// The tap mass `ν` `[batch, u, nheads]`; `None` under
        /// [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None).
        /// `β = ν·α` is formed where it is spent.
        pub nu_buh: Option<Tensor<3>>,
        /// The interior (lag-1) tap's mass `[batch, u, nheads]`; `None` unless
        /// the pattern
        /// [`has_interior_tap`](crate::mamba3::trapezoid::Trapezoid::has_interior_tap).
        pub nu_interior_buh: Option<Tensor<3>>,
        /// `γ = λ·Δ` `[batch, u, nheads]` (`= Δ` when there is no `λ`).
        pub gamma_buh: Tensor<3>,
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
            // See the note in `forward`: the three trailing segments are the
            // optional ones (`Real1D` projects no rotation, a one-tap pattern no
            // `μ`, `Trapezoid::None` no `λ`).
            let (proj_bd, rot_bA) =
                helpers::split_trailing(proj_bd, self.rotation_channels_total(), 1);
            let (proj_bd, mu_raw_bH) =
                helpers::split_trailing(proj_bd, self.mu_channels_total(), 1);
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
                da: da_buh,
                alpha: alpha_buh,
                nu: nu_buh,
                nu_interior: nu_interior_buh,
                gamma: gamma_buh,
            } = helpers::trapezoidal_coefficients(
                unfold_micro_b(dd_dt_bH, u),
                unfold_micro_b(dd_a_raw_bH, u),
                lambda_raw_bH.map(|t| unfold_micro_b(t, u)),
                mu_raw_bH.map(|t| unfold_micro_b(t, u)),
                self.dt_bias_h.val(),
                self.trapezoid_spec(),
            );
            san(&dt_buh);
            san(&alpha_buh);
            for nu_buh in [&nu_buh, &nu_interior_buh].into_iter().flatten() {
                san(nu_buh);
            }
            san(&gamma_buh);

            // ── QK-Norm on B and C ────────────────────────────────────────────
            // Both carry a leading axis — `u` writes for B, the token's one read
            // for C — so this is `forward`'s pair of calls at `sequence = u`,
            // `tokens = 1`, group dim 3.
            let b_bumhr = helpers::qk_norm_expand_bias::<5, 6>(
                b_raw_bMGRU.reshape([batch, u, mimo_rank, ngroups, state_rank]),
                &self.b_norm,
                self.b_bias_hmr.val(),
                3,
                nheads,
            );
            let c_b1mhr = helpers::qk_norm_expand_bias::<5, 6>(
                c_raw_bMGR.reshape([batch, 1, mimo_rank, ngroups, state_rank]),
                &self.c_norm,
                self.c_bias_hmr.val(),
                3,
                nheads,
            );
            assert_eq!([batch, u, mimo_rank, nheads, state_rank], b_bumhr.dims());
            san(&b_bumhr);
            san(&c_b1mhr);

            StepProjection {
                z_bi,
                c_b1mhr,
                x_buhp,
                b_bumhr,
                rot_bua,
                dt_buh,
                da_buh,
                alpha_buh,
                nu_buh,
                nu_interior_buh,
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
        /// At `micro_steps = u > 1` the token is `u` of those steps
        /// ([`crate::mamba3::product`]) and they are evaluated **together**, not
        /// one at a time — see the body.
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
            let mimo_rank = self.mimo_rank;
            let u = self.micro_steps;
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

            // ── The token's `u` micro-steps at once, then one readout ────────────
            // MambaProduct evaluates a token as `u` consecutive positions of the
            // ordinary recurrence, so a `step` call is a **folded block** of
            // length `u` — and one is `forward` on a one-token sequence, which
            // is what the parity tests assert. It is not walked a position at a
            // time, because it need not be: after the RoPE factoring the
            // transition inside the block is the *scalar* `α`, so the block has
            // a closed form,
            //
            //     h = (∏ⱼ αⱼ)·h₋₁ + Σⱼ wⱼ · writeⱼ ,    wⱼ = ∏_{r>j} αᵣ
            //
            // whose transport `wⱼ` — a write's decay to the end of the block —
            // is the very quantity [`crate::mamba3::helpers::tail_decay`]
            // already computes for the tap slots the call leaves behind, over
            // the whole block instead of its last `lag`.
            //
            // Every `writeⱼ` is an outer product into one shared state, so
            // transporting them and fusing `(u, mimo_rank)` into a single
            // contracted axis makes each **side** of the recurrence one
            // [`helpers::mimo_outer_sum`] — the collapse the single-SSD boundary
            // seed already makes over its slots. The op count is therefore
            // independent of `u`, and at `u = 1` the fused axis is `mimo_rank`
            // and this is stock Mamba-3's op graph unchanged.
            //
            // The trapezoid is untouched by any of it: a tap at
            // [`Trapezoid::tap_lag`] reads folded position `p − lag`, which is
            // `forward`'s [`helpers::shift_stream`] over this block with the
            // cache's FIFO as the prefix, and `β = ν·α` times
            // [`helpers::interior_gap_decay`] is the same gap transport (§9). At
            // lag 1 that gap has no interior and the FIFO is one slot deep; at
            // lag `u` the block is exactly as long as the lag, so every tap
            // reads the FIFO and none of them reads within the block.
            let lag = self.tap_lag();

            // ── Cumulative rotation, applied to B at every write and to C ─────
            // at the one read: `forward`'s routine at `sequence = u`,
            // `tokens = 1`, `read_stride = u`. The read row is the last
            // micro-step, so `C` picks up exactly the cumulative rotation the
            // readout happens at.
            let (b_bumhr, c_b1mhr, new_rotation) = rotate_bc_forward(
                proj.rot_bua,
                proj.dt_buh,
                cache.rotation.clone(),
                proj.b_bumhr,
                proj.c_b1mhr,
                u,
                self.rotation_spec(),
            );
            san(&b_bumhr);
            san(&c_b1mhr);
            new_rotation.sanity();

            // ── The block's two transports ────────────────────────────────────
            // `w` carries a write to the end of the block (`None` at `u = 1`,
            // where that product is empty); `α_total` is the token's whole
            // transition, the one factor the carried state takes.
            let w_buh = helpers::tail_decay(proj.da_buh.clone(), u);
            let alpha_total_bh11 = proj
                .da_buh
                .clone()
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .exp()
                .unsqueeze_dims::<4>(&[2, 3]);

            // One side of the recurrence, transported and fused:
            //
            //   write[b, h, p, r] = Σ_{j,m} massⱼ·wⱼ · v[j, m, h, p] · K[j, m, h, r]
            //
            // i.e. `einsum('bfhp,bfhr->bhpr', mass·w·v, K)` over the fused
            // `f = u · mimo_rank` axis — one matmul, whatever `u` is.
            let write = |mass_buh: Tensor<3>, x_buhp: Tensor<4>, k_bumhr: Tensor<5>| {
                let mass_buh = match &w_buh {
                    Some(w_buh) => mass_buh * w_buh.clone(),
                    None => mass_buh,
                };
                let v_bumhp = helpers::build_v_with_mimo::<4, 5>(x_buhp, mimo_x_hmp.as_ref(), 2)
                    * mass_buh.unsqueeze_dims::<5>(&[2, 4]);
                san(&v_bumhp);
                let write_bhpr = helpers::mimo_outer_sum(
                    v_bumhp.reshape([batch, u * mimo_rank, nheads, per_head_dim]),
                    k_bumhr.reshape([batch, u * mimo_rank, nheads, state_rank]),
                    siso,
                );
                san(&write_bhpr);
                write_bhpr
            };

            // A tapped side at its own lag — `forward`'s `beta_side`, over this
            // block. The prefix is the FIFO's newest `lag` slots: all of it for
            // the pattern's own tap, the newest slot alone for a two-tap
            // pattern's lag-1 one, which by the FIFO's convention carries the
            // empty decay product. Under [`Trapezoid::None`] there is no tapped
            // side at all — no shifted stream, no second outer product, one
            // fewer term in the state update.
            let tap_write = |nu_buh: Tensor<3>, lag: usize| {
                let slots_buhp = cache
                    .v_state_buhp
                    .clone()
                    .expect("a β tap keeps its (B, x) cache slots");
                let slots_bumhr = cache
                    .k_state_bumhr
                    .clone()
                    .expect("a β tap keeps its (B, x) cache slots");
                let newest = slots_buhp.dims()[1] - lag;
                let x_prev_buhp = helpers::shift_stream(
                    proj.x_buhp.clone(),
                    slots_buhp.narrow(1, newest, lag),
                    lag,
                );
                let b_prev_bumhr =
                    helpers::shift_stream(b_bumhr.clone(), slots_bumhr.narrow(1, newest, lag), lag);
                let beta_buh = nu_buh * proj.alpha_buh.clone();
                let beta_buh = match helpers::interior_gap_decay(proj.da_buh.clone(), lag) {
                    Some(gap_buh) => beta_buh * gap_buh,
                    None => beta_buh,
                };
                write(beta_buh, x_prev_buhp, b_prev_bumhr)
            };

            // ── SSM state update ──────────────────────────────────────────────
            let state_bhpr = alpha_total_bh11 * cache.ssm_bhpr.clone()
                + write(proj.gamma_buh, proj.x_buhp.clone(), b_bumhr.clone());
            let state_bhpr = [
                proj.nu_buh.map(|nu_buh| tap_write(nu_buh, lag)),
                proj.nu_interior_buh.map(|nu_buh| tap_write(nu_buh, 1)),
            ]
            .into_iter()
            .flatten()
            .fold(state_bhpr, |state, write| state + write);
            san(&state_bhpr);

            // The readout's own inputs: `C` at the block's one read row, and the
            // `D` skip's value at the micro-step that read is contemporaneous
            // with — the last one.
            let c_bmhr = c_b1mhr.squeeze_dim::<4>(1);
            let x_last_bhp = proj.x_buhp.clone().narrow(1, u - 1, 1).squeeze_dim::<3>(1);
            let x_vals_bmhp =
                helpers::build_v_with_mimo::<3, 4>(x_last_bhp, mimo_x_hmp.as_ref(), 1);
            san(&x_vals_bmhp);

            // ── Output ────────────────────────────────────────────────────────
            // outₘ[b, m, h, p] = sumᵣ C[b, m, h, r] * state[b, h, p, r] + D * x_vals[b, m, h, p]
            let out_m_bmhp = Self::step_readout(state_bhpr.clone(), c_bmhr, siso);
            san(&out_m_bmhp);

            // ── D skip, gate (or gated norm), rank aggregation, out-projection ─
            let out_bm = self.step_finish(out_m_bmhp, x_vals_bmhp, proj.z_bi);

            // ── Update cache ──────────────────────────────────────────────────
            // The block's last `lag` positions, `x` pre-scaled by the decay
            // since its own position — `forward`'s own helper, so the two write
            // the same slot layout under the same convention.
            let (k_state_bumhr, v_state_buhp) =
                self.save_tap_slots(&b_bumhr, &proj.x_buhp, &proj.da_buh, lag);
            cache.ssm_bhpr = state_bhpr;
            cache.k_state_bumhr = k_state_bumhr;
            cache.v_state_buhp = v_state_buhp;
            cache.rotation = new_rotation;

            (out_bm, cache)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
