//! # Mamba-3 — Double-Pass SSD Forward
//!
//! This module provides the [`Mamba3::forward_double_ssd`](crate::mamba3::mamba3::Mamba3::forward_double_ssd) method:
//! The burn-mamba implementation of the [`VikramLex/mamba3-minimal`](https://github.com/VikramLex/mamba3-minimal) decomposition:
//!
//! ```text
//!   hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ      (original double-ssd trapezoidal)
//!
//!   forward:    h = SSD(γ-scaled V, B)   +   SSD(β-scaled V_shifted, B_shifted)
//! ```
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
            let k_state_bmhr = Tensor::zeros([batch, mimo_rank, nheads, state_rank], &device);
            let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], &device);
            let rotation = self.zero_rotation_state(batch, &device);
            Mamba3DoubleSsdCache {
                ssm_bhpr,
                k_state_bmhr,
                v_state_bhp,
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
        // The rotation channels come off first: `Real1D` projects none, and a
        // zero-width segment would silently vanish from the split below.
        let u = micro_steps;
        let (proj_bsd, rot_btA) =
            helpers::split_rotation_channels(proj_bsd, self.rotation_channels_total(), 2);
        #[rustfmt::skip]
        let [
                z_bsi, x_btI,
                b_raw_btMGRU, c_raw_btMGR,
                dd_dt_btH, dd_A_raw_btH, lambda_raw_btH,
        ] = burn_stack::modules::split_into(
            proj_bsd,
            [
                d_inner, u * d_inner,
                u * bc_size, bc_size,
                u * nheads, u * nheads, u * nheads,
            ],
            2,
        );

        use crate::mamba3::product::{repeat_micro_bs, unfold_micro_bs};
        let x_bsi = unfold_micro_bs(x_btI, u);
        let b_raw_bsMGR = unfold_micro_bs(b_raw_btMGRU, u);
        let c_raw_bsMGR = repeat_micro_bs(c_raw_btMGR, u);
        let dd_dt_bsh = unfold_micro_bs(dd_dt_btH, u);
        let dd_A_raw_bsh = unfold_micro_bs(dd_A_raw_btH, u);
        let lambda_raw_bsh = unfold_micro_bs(lambda_raw_btH, u);
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
        san(&beta_bsh);
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

        // ── Step 6: Build shifted inputs for β term ───────────────────────────
        //
        // "Shift-Before-Chunking": prepend the cached xₜ₋₁ / Bₜ₋₁ at the
        // sequence level (before SSD chunking) so the β term at t=0 sees the
        // prior token from a continued cache. For a fresh (zero) cache this is
        // equivalent to zero-padding.
        //
        // The shift is one *folded* position — [`Trapezoid::HorizontalCarryOver`],
        // matching `step`'s per-micro-step carry. A lag-`u` pattern shifts by `u`
        // and seeds from a `u`-slot cache instead; nothing else here changes.
        let x_prev_first_b1hp = cache.v_state_bhp.clone().unsqueeze_dim::<4>(1);
        let x_prev_bshp = if sequence == 1 {
            x_prev_first_b1hp
        } else {
            Tensor::cat(
                vec![x_prev_first_b1hp, x_bshp.clone().narrow(1, 0, sequence - 1)],
                1,
            )
        };
        let b_prev_first_b1mhr = cache.k_state_bmhr.clone().unsqueeze_dim::<5>(1);
        let b_prev_bsmhr = if sequence == 1 {
            b_prev_first_b1mhr
        } else {
            Tensor::cat(
                vec![
                    b_prev_first_b1mhr,
                    b_bsmhr.clone().narrow(1, 0, sequence - 1),
                ],
                1,
            )
        };

        // ── Step 7: Scale inputs by trapezoidal coefficients ──────────────────
        // gamma and beta are per-head scalars, broadcast over mimo_rank and per_head_dim:
        let gamma_bsh1 = gamma_bsh.unsqueeze_dim::<4>(3);
        let beta_bsh1 = beta_bsh.unsqueeze_dim::<4>(3);
        let x_gamma_bshp = x_bshp.clone() * gamma_bsh1; // γₜ · xₜ
        let x_beta_bshp = x_prev_bshp * beta_bsh1; // βₜ · xₜ₋₁

        // ── Save the last position's B and x for the cache ────────────────────
        // "Last position" is the last *micro-step* of the last token — exactly
        // the step the next call's first micro-step follows, so the trapezoid's
        // previous-step taps continue across a split prefill unchanged.
        let b_last_bmhr = b_bsmhr
            .clone()
            .narrow(1, sequence - 1, 1)
            .reshape([batch, mimo_rank, nheads, state_rank]);
        let x_last_bhp = x_bshp
            .clone()
            .narrow(1, sequence - 1, 1) // x_b1hp
            .squeeze_dim::<3>(1); // x_bhp

        // ── Step 8: Pad sequence to multiple of chunk_len ─────────────────────
        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        #[rustfmt::skip]
        let (x_gamma_bShp, x_beta_bShp, da_bSh, b_bSmhr, b_prev_bSmhr, c_bSmhr) = if pad == 0 {
            (x_gamma_bshp, x_beta_bshp, da_bsh, b_bsmhr, b_prev_bsmhr, c_bsmhr)
        } else {
            let pad_bShp = Tensor::zeros([batch, pad, nheads, per_head_dim], &device);
            let pad_bSh = Tensor::zeros([batch, pad, nheads], &device);
            let pad_bSmhr = Tensor::zeros([batch, pad, mimo_rank, nheads, state_rank], &device);
            (
                Tensor::cat(vec![x_gamma_bshp, pad_bShp.clone()], 1),
                Tensor::cat(vec![x_beta_bshp, pad_bShp], 1),
                Tensor::cat(vec![da_bsh, pad_bSh], 1),
                Tensor::cat(vec![b_bsmhr, pad_bSmhr.clone()], 1),
                Tensor::cat(vec![b_prev_bsmhr, pad_bSmhr.clone()], 1),
                Tensor::cat(vec![c_bsmhr, pad_bSmhr], 1),
            )
        };

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let x_gamma_bnlhp = x_gamma_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let x_beta_bnlhp = x_beta_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let b_prev_bnlmhr =
            b_prev_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bnlmhr = c_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

        // ── Step 9: Double MIMO-SSD calls ────────────────────────────────────────
        // Build V tensors — insert the mimo_rank axis at position 3 of `_bnlhp`.
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let v_gamma_bnlmhp =
            helpers::build_v_with_mimo::<5, 6>(x_gamma_bnlhp.clone(), mimo_x_hmp.as_ref(), 3);
        let v_beta_bnlmhp =
            helpers::build_v_with_mimo::<5, 6>(x_beta_bnlhp, mimo_x_hmp.as_ref(), 3);

        let input_gamma = Mamba3DoubleSsdInput {
            v_bnlmhp: v_gamma_bnlmhp,
            da_bnlh: da_bnlh.clone(),
            b_bnlmhr: b_bnlmhr.clone(),
            c_bnlmhr: c_bnlmhr.clone(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        };
        let (y_gamma_bnlmhp, final_state_gamma_bhpr) = input_gamma.run(ssd_path);

        let input_beta = Mamba3DoubleSsdInput {
            v_bnlmhp: v_beta_bnlmhp,
            da_bnlh,
            b_bnlmhr: b_prev_bnlmhr,
            c_bnlmhr,
            initial_state_bhpr: Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device),
            init_state_hpr: None,
        };
        let (y_beta_bnlmhp, final_state_beta_bhpr) = input_beta.run(ssd_path);

        let y_bnlmhp = y_gamma_bnlmhp + y_beta_bnlmhp;
        let final_state_bhpr = final_state_gamma_bhpr + final_state_beta_bhpr;

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
        // k_state / v_state = B / x at the last micro-step of the last token
        cache.k_state_bmhr = b_last_bmhr;
        cache.v_state_bhp = x_last_bhp;

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

    /// One token's in-projection unpacked into the step-shaped pieces shared by
    /// [`Mamba3::step_double_ssd`] and the constant-input shortcut
    /// [`Mamba3::step_infinite`]: the gate/value streams, the **pre-rotation**
    /// QK-normed B/C, the raw rotation channels, and the trapezoid
    /// coefficients.
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
        /// `β = (1−λ)·Δ·α` `[batch, u, nheads]`.
        pub beta_buh: Tensor<3>,
        /// `γ = λ·Δ` `[batch, u, nheads]`.
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
        /// `β = (1−λ)·Δ·α` `[batch, nheads]`.
        pub beta_bh: Tensor<2>,
        /// `γ = λ·Δ` `[batch, nheads]`.
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
                beta_bh: pick2(self.beta_buh.clone()),
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
            // See the note in `forward`: `Real1D` projects no rotation channels.
            let (proj_bd, rot_bA) =
                helpers::split_rotation_channels(proj_bd, self.rotation_channels_total(), 1);
            #[rustfmt::skip]
            let [
                    z_bi, x_bI,
                    b_raw_bMGRU, c_raw_bMGR,
                    dd_dt_bH, dd_a_raw_bH, lambda_raw_bH,
            ] = burn_stack::modules::split_into(
                proj_bd,
                [
                    d_inner, u * d_inner,
                    u * bc_size, bc_size,
                    u * nheads, u * nheads, u * nheads,
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
                unfold_micro_b(lambda_raw_bH, u),
                self.dt_bias_h.val(),
                self.dt_limit,
                self.a_floor,
            );
            san(&dt_buh);
            san(&alpha_buh);
            san(&beta_buh);
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
            let mimo_rank = self.mimo_rank;
            let device = &input_bd.device();
            let ssm_shape = [batch, nheads, per_head_dim, state_rank];

            let mut cache = cache.unwrap_or_else(|| {
                let ssm_bhpr = Tensor::zeros(ssm_shape, device);
                let k_state_bmhr = Tensor::zeros([batch, mimo_rank, nheads, state_rank], device);
                let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], device);
                let rotation = self.zero_rotation_state(batch, device);
                Mamba3DoubleSsdCache {
                    ssm_bhpr,
                    k_state_bmhr,
                    v_state_bhp,
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
            // The `prev_*` carries below are re-assigned every micro-step, which
            // is [`Trapezoid::HorizontalCarryOver`] — lag 1 on the folded
            // sequence, so the tap crosses a token only at `j = 0`. A lag-`u`
            // pattern would carry `u` slots and read the one `u` back.
            let mut state_bhpr = cache.ssm_bhpr.clone();
            let mut prev_b_bmhr = cache.k_state_bmhr.clone();
            let mut prev_x_bhp = cache.v_state_bhp.clone();
            let mut rotation = cache.rotation.clone();
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
                let xs_vals_bmhp =
                    helpers::build_v_with_mimo::<3, 4>(prev_x_bhp, mimo_x_hmp.as_ref(), 1);
                san(&xs_vals_bmhp);

                // ── SSM state update ───────────────────────────────────────
                // new_state[b, h, p, r] = alpha * state
                //   + sumₘ gamma * x_vals[m] ⊗ B_cur[m]
                //   + sumₘ beta  * xs_vals[m] ⊗ B_state[m]
                //
                // For the outer product sum:
                //   xBt[b, h, p, r] = sumₘ coeff[m, h, p] * B[m, h, n]
                //   = einsum('bmhp,bmhr->bhpr', coeff*x_vals, B)
                //   = matmul over m: [b, h, p, m] @ [b, h, m, r]
                // x_vals_bmhp * gamma_b1h1
                // Need gamma as [b, 1, h, 1] to broadcast over m and p:
                let gamma_b1h1 = m.gamma_bh.unsqueeze_dims::<4>(&[1, 3]);
                let beta_b1h1 = m.beta_bh.unsqueeze_dims::<4>(&[1, 3]);

                let x_gamma_bmhp = x_vals_bmhp.clone() * gamma_b1h1;
                san(&x_gamma_bmhp);
                let x_beta_bmhp = xs_vals_bmhp * beta_b1h1;
                san(&x_beta_bmhp);

                // einsum('bmhp,bmhr->bhpr', x_gamma, B_cur):
                let xbt_state_bhpr = helpers::mimo_outer_sum(x_gamma_bmhp, b_bmhr.clone(), siso);
                san(&xbt_state_bhpr);
                let xbt_prev_bhpr = helpers::mimo_outer_sum(x_beta_bmhp, prev_b_bmhr, siso);
                san(&xbt_prev_bhpr);

                let alpha_bh11 = m.alpha_bh.unsqueeze_dims::<4>(&[2, 3]);
                let new_state_bhpr = alpha_bh11 * state_bhpr + xbt_state_bhpr + xbt_prev_bhpr;
                san(&new_state_bhpr);

                state_bhpr = new_state_bhpr;
                prev_b_bmhr = b_bmhr;
                prev_x_bhp = m.x_bhp;
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
            cache.ssm_bhpr = state_bhpr;
            cache.k_state_bmhr = prev_b_bmhr;
            cache.v_state_bhp = prev_x_bhp;
            cache.rotation = rotation;

            (out_bm, cache)
        }
    }
}

pub(crate) use step::MicroProjection;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
