//! # Mamba-3 — Double-Pass SSD Forward
//!
//! This module provides the [`Mamba3::forward_double_ssd`] method:
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
use crate::mamba3::rotation::{RotationState, rotate_bc_forward, rotate_bc_step};
use crate::modules::{Silu, StateMoments};
use crate::modules::sanity as san;
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
        let (out_bsm, cache, _) = self.forward_double_ssd_impl(input_bsm, cache, ssd_path, None);
        (out_bsm, cache)
    }

    /// [`Self::forward_double_ssd`] optionally computing the physical-frame
    /// state moments from the pre-SSD seam (`None` — no moments; `Some(detach)`
    /// — compute them, detached or attached). See `mamba3/moments.rs`.
    #[allow(non_snake_case)]
    pub(crate) fn forward_double_ssd_impl(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3DoubleSsdCache>,
        ssd_path: &Mamba3SsdPath,
        with_moments: Option<bool>,
    ) -> (Tensor<3>, Mamba3DoubleSsdCache, Option<StateMoments>) {
        let [batch, sequence, _d_model] = input_bsm.dims();
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let num_rope_angles = self.num_rope_angles;
        let mimo_rank = self.mimo_rank;
        let device = input_bsm.device();

        assert!(sequence > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        let mut cache = cache.unwrap_or_else(|| {
            let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
            let k_state_bmhr = Tensor::zeros([batch, mimo_rank, nheads, state_rank], &device);
            let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], &device);
            let rotation = match self.rotation {
                RotationKind::Quaternion4D => {
                    RotationState::identity_quaternion(batch, nheads, self.num_quat_blocks, &device)
                }
                RotationKind::Complex2D => {
                    RotationState::zeros_angle(batch, nheads, num_rope_angles, &device)
                }
            };
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

        // [batch, sequence, *] split along channel dim.
        // b_raw_bsMGR / c_raw_bsMGR have channel size `mimo_rank * ngroups * state_rank`.
        #[rustfmt::skip]
        let [
                z_bsi, x_bsi,
                b_raw_bsMGR, c_raw_bsMGR,
                dd_dt_bsh, dd_A_raw_bsh, lambda_raw_bsh,
                rot_bsa
        ] = crate::modules::split_into(
            proj_bsd,
            [
                d_inner, d_inner,
                bc_size, bc_size,
                nheads, nheads, nheads,
                self.num_rotation_channels,
            ],
            2,
        );

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

        // ── Step 5: Data-dependent positional rotation of B and C ─────────────
        // Complex2D: abelian RoPE (cumulative angle). Quaternion4D: cumulative
        // unit quaternion. The new cache accumulator is returned for Step (cache
        // update) below. See [`rotate_bc_forward`].
        let (b_bsmhr, c_bsmhr, new_rotation, rotation_seq) = rotate_bc_forward(
            rot_bsa,
            dt_bsh.clone(),
            cache.rotation.clone(),
            b_bsmhr,
            c_bsmhr,
            self.rotation_kind(),
            self.rope_dim,
        );
        san(&b_bsmhr);
        san(&c_bsmhr);

        // ── Step 6: Build shifted inputs for β term ───────────────────────────
        //
        // "Shift-Before-Chunking": prepend the cached xₜ₋₁ / Bₜ₋₁ at the
        // sequence level (before SSD chunking) so the β term at t=0 sees the
        // prior token from a continued cache. For a fresh (zero) cache this is
        // equivalent to zero-padding.
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
        let gamma_bsh1 = gamma_bsh.clone().unsqueeze_dim::<4>(3);
        let beta_bsh1 = beta_bsh.clone().unsqueeze_dim::<4>(3);
        let x_gamma_bshp = x_bshp.clone() * gamma_bsh1; // γₜ · xₜ
        let x_beta_bshp = x_prev_bshp * beta_bsh1; // βₜ · xₜ₋₁

        // ── Save last-token B for cache ───────────────────────────────────────
        let b_last_bmhr = b_bsmhr
            .clone()
            .narrow(1, sequence - 1, 1)
            .reshape([batch, mimo_rank, nheads, state_rank]);

        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);

        // ── Physical-frame state moments (optional; pre-SSD seam) ─────────────
        // Built from the same sequence-level pieces the SSD consumes below;
        // the initial state is the cache's `ssm_bhpr` (counted once — the β
        // stream's first element carries the boundary write).
        let moments = with_moments.map(|detach| {
            let input = self.build_moments_input(
                x_bshp.clone(),
                b_bsmhr.clone(),
                gamma_bsh.clone(),
                beta_bsh.clone(),
                da_bsh.clone(),
                cache.v_state_bhp.clone(),
                cache.k_state_bmhr.clone(),
                cache.ssm_bhpr.clone(),
                rotation_seq,
                chunk_len,
            );
            let input = if detach { input.detached() } else { input };
            input.state_moments_phys_recalculated(sequence)
        });

        // ── Step 8: Pad sequence to multiple of chunk_len ─────────────────────
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

        // ── Step 11: D skip + gate + aggregate ranks ──────────────────────────
        // D skip uses raw x * mimo_x_hmp (not gamma-scaled)
        // Insert the mimo_rank axis at position 2 of `_bshp`.
        let v_raw_bsmhp =
            helpers::build_v_with_mimo::<4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);

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
                    .permute([1, 0, 2]) // mimo_z_mhp
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
                .permute([1, 0, 2]) // mimo_o_mhp
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
        // k_state = B at last token
        cache.k_state_bmhr = b_last_bmhr;

        // v_state = x at last token
        cache.v_state_bhp = x_bshp
            .narrow(1, sequence - 1, 1) // x_b1hp
            .squeeze_dim::<3>(1); // x_bhp

        // Cumulative rotation at the last token (angle wrapped to [−π, π], or
        // the cumulative quaternion), to continue a longer sequence.
        cache.rotation = new_rotation;

        (out_bsm, cache, moments)
    }
}

// ---------------------------------------------------------------------------
// Mamba3::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    /// One token's in-projection unpacked into the step-shaped pieces shared by
    /// [`Mamba3::step_double_ssd`] and the constant-input shortcuts
    /// ([`Mamba3::step_n_approx`] / [`Mamba3::step_infinite`]): the gate/value
    /// streams, the **pre-rotation** QK-normed B/C, the raw rotation channels,
    /// and the trapezoid coefficients.
    pub(crate) struct StepProjection {
        /// Gate stream `[batch, d_inner]`.
        pub z_bi: Tensor<2>,
        /// Value stream `[batch, nheads, per_head_dim]`.
        pub x_bhp: Tensor<3>,
        /// QK-normed, GQA-expanded, biased B — **before** the positional rotation.
        pub b_bmhr: Tensor<4>,
        /// QK-normed, GQA-expanded, biased C — **before** the positional rotation.
        pub c_bmhr: Tensor<4>,
        /// Raw rotation channels `[batch, num_rotation_channels]`.
        pub rot_ba: Tensor<2>,
        /// `Δ` `[batch, nheads]`.
        pub dt_bh: Tensor<2>,
        /// `Δ·A` (negative; the log-decay) `[batch, nheads]`.
        pub da_bh: Tensor<2>,
        /// `α = exp(Δ·A)` `[batch, nheads]`.
        pub alpha_bh: Tensor<2>,
        /// `β = (1−λ)·Δ·α` `[batch, nheads]`.
        pub beta_bh: Tensor<2>,
        /// `γ = λ·Δ` `[batch, nheads]`.
        pub gamma_bh: Tensor<2>,
    }

    impl Mamba3 {
        /// In-projection → split → trapezoid coefficients → QK-norm for a
        /// single token, **stopping before** the positional rotation (which
        /// needs the cache's cumulative rotation).
        #[allow(non_snake_case)]
        pub(crate) fn step_project(&self, input_bd: Tensor<2>) -> StepProjection {
            let [batch, _d_model] = input_bd.dims();
            let d_inner = self.d_inner();
            let nheads = self.nheads();
            let ngroups = self.ngroups;
            let per_head_dim = self.per_head_dim();
            let state_rank = self.state_rank;
            let mimo_rank = self.mimo_rank;

            assert_eq!(nheads % ngroups, 0);
            san(&input_bd);

            // ── In-projection ─────────────────────────────────────────────────
            let proj_bd = self.in_proj.forward(input_bd);
            san(&proj_bd);
            let bc_size = ngroups * state_rank * mimo_rank;
            // [batch, *] split along channel dim.
            // b_raw_bMGR / c_raw_bMGR have channel size `mimo_rank * ngroups * state_rank`.
            #[rustfmt::skip]
            let [
                    z_bi, x_bi,
                    b_raw_bMGR, c_raw_bMGR,
                    dd_dt_bh, dd_a_raw_bh, lambda_raw_bh,
                    rot_ba,
            ] = crate::modules::split_into(
                proj_bd,
                [
                    d_inner, d_inner,
                    bc_size, bc_size,
                    nheads, nheads, nheads,
                    self.num_rotation_channels,
                ],
                1,
            );

            // ── Reshape x ─────────────────────────────────────────────────────
            let x_bhp = x_bi.reshape([batch, nheads, per_head_dim]);

            // ── Discretisation + trapezoidal coefficients ─────────────────────
            let helpers::TrapezoidCoeffs {
                dt: dt_bh,
                da: da_bh,
                alpha: alpha_bh,
                beta: beta_bh,
                gamma: gamma_bh,
            } = helpers::trapezoidal_coefficients(
                dd_dt_bh,
                dd_a_raw_bh,
                lambda_raw_bh,
                self.dt_bias_h.val(),
                self.dt_limit,
                self.a_floor,
            );
            san(&dt_bh);
            san(&alpha_bh);
            san(&beta_bh);
            san(&gamma_bh);

            // ── QK-Norm on B and C ────────────────────────────────────────────
            // Group dim is axis 2 of `_bmgr` (D = 4).
            let b_bmhr = helpers::qk_norm_expand_bias::<4, 5>(
                b_raw_bMGR.reshape([batch, mimo_rank, ngroups, state_rank]),
                &self.b_norm,
                self.b_bias_hmr.val(),
                2,
                nheads,
            );
            let c_bmhr = helpers::qk_norm_expand_bias::<4, 5>(
                c_raw_bMGR.reshape([batch, mimo_rank, ngroups, state_rank]),
                &self.c_norm,
                self.c_bias_hmr.val(),
                2,
                nheads,
            );
            assert_eq!([batch, mimo_rank, nheads, state_rank], b_bmhr.dims());
            san(&b_bmhr);
            san(&c_bmhr);

            StepProjection {
                z_bi,
                x_bhp,
                b_bmhr,
                c_bmhr,
                rot_ba,
                dt_bh,
                da_bh,
                alpha_bh,
                beta_bh,
                gamma_bh,
            }
        }

        /// State→output contraction:
        /// `out[b, m, h, p] = Σᵣ C[b, m, h, r] · state[b, h, p, r]`
        /// (`einsum('bhpr,bmhr->bmhp', state, C)`).
        pub(crate) fn step_readout(state_bhpr: Tensor<4>, c_bmhr: Tensor<4>) -> Tensor<4> {
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
                    .permute([1, 0, 2]) // mimo_z_mhp
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
                    .permute([1, 0, 2]) // mimo_o_mhp
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
            let num_rope_angles = self.num_rope_angles;
            let mimo_rank = self.mimo_rank;
            let device = &input_bd.device();
            let ssm_shape = [batch, nheads, per_head_dim, state_rank];

            let mut cache = cache.unwrap_or_else(|| {
                let ssm_bhpr = Tensor::zeros(ssm_shape, device);
                let k_state_bmhr = Tensor::zeros([batch, mimo_rank, nheads, state_rank], device);
                let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], device);
                let rotation = match self.rotation {
                    RotationKind::Quaternion4D => RotationState::identity_quaternion(
                        batch,
                        nheads,
                        self.num_quat_blocks,
                        device,
                    ),
                    RotationKind::Complex2D => {
                        RotationState::zeros_angle(batch, nheads, num_rope_angles, device)
                    }
                };
                Mamba3DoubleSsdCache {
                    ssm_bhpr,
                    k_state_bmhr,
                    v_state_bhp,
                    rotation,
                }
            });

            // ── In-projection → coefficients → QK-norm ────────────────────────
            let StepProjection {
                z_bi,
                x_bhp,
                b_bmhr,
                c_bmhr,
                rot_ba,
                dt_bh,
                da_bh: _,
                alpha_bh,
                beta_bh,
                gamma_bh,
            } = self.step_project(input_bd);

            // ── Update cumulative rotation, rotate B and C ─────────────────────
            // Complex2D: abelian RoPE angle. Quaternion4D: cumulative quaternion.
            // See [`rotate_bc_step`].
            let (b_bmhr, c_bmhr, new_rotation) = rotate_bc_step(
                rot_ba,
                dt_bh.clone(),
                cache.rotation.clone(),
                b_bmhr,
                c_bmhr,
                self.rotation_kind(),
                self.rope_dim,
            );
            san(&b_bmhr);
            san(&c_bmhr);
            new_rotation.sanity();

            // ── Build MIMO value tensors ───────────────────────────────────────
            // Insert the mimo_rank axis at position 1 of `_bhp`.
            let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
            let x_vals_bmhp =
                helpers::build_v_with_mimo::<3, 4>(x_bhp.clone(), mimo_x_hmp.as_ref(), 1);
            san(&x_vals_bmhp);
            let xs_vals_bmhp = helpers::build_v_with_mimo::<3, 4>(
                cache.v_state_bhp.clone(),
                mimo_x_hmp.as_ref(),
                1,
            );
            san(&xs_vals_bmhp);

            // ── SSM state update ───────────────────────────────────────────────
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
            let gamma_b1h1 = gamma_bh.clone().unsqueeze_dims::<4>(&[1, 3]);
            let beta_b1h1 = beta_bh.clone().unsqueeze_dims::<4>(&[1, 3]);

            let x_gamma_bmhp = x_vals_bmhp.clone() * gamma_b1h1;
            san(&x_gamma_bmhp);
            let x_beta_bmhp = xs_vals_bmhp * beta_b1h1;
            san(&x_beta_bmhp);

            // einsum('bmhp,bmhr->bhpr', x_gamma, B_cur):
            let xbt_state_bhpr = {
                let b_bhmr = b_bmhr.clone().permute([0, 2, 1, 3]);
                let xg_bhpm = x_gamma_bmhp.permute([0, 2, 3, 1]);
                xg_bhpm.matmul(b_bhmr)
            };
            san(&xbt_state_bhpr);
            let xbt_prev_bhpr = {
                let b_state_bhmr = cache.k_state_bmhr.clone().permute([0, 2, 1, 3]);
                let xb_bhpm = x_beta_bmhp.permute([0, 2, 3, 1]);
                xb_bhpm.matmul(b_state_bhmr)
            };
            san(&xbt_prev_bhpr);

            let alpha_bh11 = alpha_bh.unsqueeze_dims::<4>(&[2, 3]);
            let new_state_bhpr =
                alpha_bh11 * cache.ssm_bhpr.clone() + xbt_state_bhpr + xbt_prev_bhpr;
            san(&new_state_bhpr);

            // ── Output ────────────────────────────────────────────────────────
            // outₘ[b, m, h, p] = sumᵣ C[b, m, h, r] * state[b, h, p, r] + D * x_vals[b, m, h, p]
            let out_m_bmhp = Self::step_readout(new_state_bhpr.clone(), c_bmhr);
            san(&out_m_bmhp);

            // ── D skip, gate (or gated norm), rank aggregation, out-projection ─
            let out_bm = self.step_finish(out_m_bmhp, x_vals_bmhp, z_bi);

            // ── Update cache ──────────────────────────────────────────────────
            cache.ssm_bhpr = new_state_bhpr;
            cache.k_state_bmhr = b_bmhr;
            cache.v_state_bhp = x_bhp;
            cache.rotation = new_rotation;

            (out_bm, cache)
        }
    }
}

pub(crate) use step::StepProjection;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
