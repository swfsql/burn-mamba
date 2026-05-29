//! # Mamba-3 — Single-Pass SSD Forward
//!
//! This module provides the `forward_single_ssd` method on [`Mamba3`]:
//! The burn-mamba implementation of the **official Mamba-3 algorithm**
//! as shipped in Triton (SISO) and Tilelang (MIMO):
//!
//! ```text
//!   scaleₜ = γₜ + (1 − λₜ₊₁) · Δₜ₊₁
//!
//!   forward_single_ssd:    h' = SSD(V_raw, K_scaled = scaleₜ B) with:
//!                               * strict lower-triangular intra-chunk mask
//!                               * additive γ-weighted same-step correction
//!                               * boundary β seed (1−λ₀) Δ₀ Kₜ₋₁ ⊗ xₜ₋₁
//! ```
//!
//! References:
//! - [`mamba3_siso_fwd.py`](https://github.com/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py),
//! - [`mamba3_mimo_fwd.py`](https://github.com/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py).
//!
//! See also: [`crate::mamba3::mamba3`] and [`crate::mamba3::double_ssd::double_ssd`].

use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCache;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::rotate_bc_forward;
use crate::mamba3::single_ssd::prelude::*;
use crate::utils::sanity::sanity as san;
use crate::utils::silu::Silu;
use burn::backend::Backend;
use burn::prelude::*;

impl Mamba3 {
    /// Process a full input sequence using the **single-ssd form (single-pass)**
    /// trapezoidal algorithm.
    ///
    /// Functionally equivalent to [`Self::forward`] but uses approximately half
    /// the SSD memory during training. Cache is a separate type
    /// ([`Mamba3SingleSsdCache`]) because the stored hidden state has different
    /// semantics than the original-form cache used by [`Self::forward`].
    ///
    /// # Shapes
    /// - `input_bsm`: `[batch, sequence, d_model]`
    /// - output: `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward_single_ssd(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3SingleSsdCache>,
        ssd_path: &Mamba3SsdPath,
    ) -> (Tensor<3>, Mamba3SingleSsdCache) {
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
            Mamba3SingleSsdCache {
                ssm_bhpr,
                k_state_bmhr,
                v_state_bhp,
                rotation,
            }
        });

        // ── Step 1: In-projection ─────────────────────────────────────────────
        let proj_bsd = self.in_proj.forward(input_bsm);
        let bc_size = ngroups * state_rank * mimo_rank;

        #[rustfmt::skip]
        let [
                z_bsi, x_bsi,
                b_raw_bsMGR, c_raw_bsMGR,
                dd_dt_bsh, dd_A_raw_bsh, lambda_raw_bsh,
                rot_bsa
        ] = crate::utils::split::split_into(
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
            beta: _beta_bsh,
            gamma: gamma_bsh,
        } = helpers::trapezoidal_coefficients(
            dd_dt_bsh,
            dd_A_raw_bsh,
            lambda_raw_bsh.clone(),
            self.dt_bias_h.val(),
            self.dt_limit,
            self.a_floor,
        );
        san(&dt_bsh);
        san(&da_bsh);
        san(&gamma_bsh);

        // ── Compute scaleₜ = γₜ + (1 − λₜ₊₁) · Δₜ₊₁ ──────────────────────────
        //
        // The shifted term is zero at the very last sequence position (no future
        // token). Out-of-bounds Δ_{t+1} is zero by construction (we pad with
        // zeros), and (1 − λ) is bounded, so the multiplication safely yields 0.
        let lambda_bsh = burn::tensor::activation::sigmoid(lambda_raw_bsh);
        let shifted_gamma_bsh = {
            let zero_b1h = Tensor::zeros([batch, 1, nheads], &device);
            if sequence == 1 {
                zero_b1h.clone()
            } else {
                let dt_next_bsh = Tensor::cat(
                    vec![dt_bsh.clone().narrow(1, 1, sequence - 1), zero_b1h.clone()],
                    1,
                );
                let lambda_next_bsh =
                    Tensor::cat(vec![lambda_bsh.narrow(1, 1, sequence - 1), zero_b1h], 1);
                dt_next_bsh * (-lambda_next_bsh + 1.0)
            }
        };
        let scale_bsh = gamma_bsh.clone() + shifted_gamma_bsh;
        san(&scale_bsh);

        // ── Step 3: Reshape x ─────────────────────────────────────────────────
        let x_bshp = x_bsi.reshape([batch, sequence, nheads, per_head_dim]);

        // ── Step 4: QK-Norm on B and C ────────────────────────────────────────
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

        // ── Step 5: Data-dependent positional rotation of B and C ─────────────
        // Complex2D: abelian RoPE (cumulative angle). Quaternion4D: cumulative
        // unit quaternion. Shared with the double-ssd pathway via
        // [`rotate_bc_forward`]; the single-pass SSD core below is
        // rotation-agnostic — it only ever consumes the rotated B̄/C̄ (the RoPE
        // factoring `C̄ₜᵀB̄ᵢ = Cₜᵀ·Rel(t,i)·Bᵢ` holds for either algebra).
        let (b_bsmhr, c_bsmhr, new_rotation) = rotate_bc_forward(
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

        // ── Save last-token B and last-token x (raw, no MIMO_V) for cache ─────
        let b_last_bmhr = b_bsmhr
            .clone()
            .narrow(1, sequence - 1, 1)
            .reshape([batch, mimo_rank, nheads, state_rank]);
        let x_last_bhp = x_bshp
            .clone()
            .narrow(1, sequence - 1, 1)
            .squeeze_dim::<3>(1);

        // ── Boundary β seed for initial state ─────────────────────────────────
        // Add (1 − λ₀) · Δ₀ · Σₘ Kₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_xₘ) to the carried
        // single-ssd SSM state. λ₀, Δ₀ are taken from the current call's first
        // token; Kₜ₋₁ and xₜ₋₁ come from the cache (zeros on fresh start).
        //
        // γₜ = λₜ·Δₜ, so (1−λ₀)·Δ₀ = Δ₀ − γ₀.
        let boundary_factor_bh = dt_bsh.clone().narrow(1, 0, 1).squeeze_dim::<2>(1)
            - gamma_bsh.clone().narrow(1, 0, 1).squeeze_dim::<2>(1);

        // Σₘ Kₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_xₘ)  → [batch, nheads, per_head_dim, state_rank]
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let v_prev_mimo_bmhp =
            helpers::build_v_with_mimo::<3, 4>(cache.v_state_bhp.clone(), mimo_x_hmp.as_ref(), 1); // [batch, mimo_rank, nheads, per_head_dim]
        let boundary_seed_bhpr = {
            // einsum: bmhr, bmhp -> bhpr  (contract over m)
            let k_prev_bhmr = cache.k_state_bmhr.clone().permute([0, 2, 1, 3]);
            let v_prev_bhpm = v_prev_mimo_bmhp.permute([0, 2, 3, 1]);
            v_prev_bhpm.matmul(k_prev_bhmr)
        };
        let initial_state_bhpr = cache.ssm_bhpr.clone()
            + boundary_seed_bhpr
                * boundary_factor_bh.unsqueeze_dims::<4>(&[2, 3]).expand([
                    batch,
                    nheads,
                    per_head_dim,
                    state_rank,
                ]);
        san(&initial_state_bhpr);

        // ── Step 6: Pad sequence to multiple of chunk_len ─────────────────────
        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        // V passed to SSD is raw x with MIMO_V applied (not γ-scaled).
        let v_bshmp = helpers::build_v_with_mimo::<4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);
        // v_bshmp has axis order [b, s, m, h, p] (insert_dim=2 onto [b,s,h,p]).

        #[rustfmt::skip]
        let (v_bShmp, da_bSh, gamma_bSh, scale_bSh, b_bSmhr, c_bSmhr) = if pad == 0 {
            (v_bshmp, da_bsh, gamma_bsh, scale_bsh, b_bsmhr, c_bsmhr)
        } else {
            let pad_bShmp = Tensor::zeros([batch, pad, mimo_rank, nheads, per_head_dim], &device);
            let pad_bSh = Tensor::zeros([batch, pad, nheads], &device);
            let pad_bSmhr = Tensor::zeros([batch, pad, mimo_rank, nheads, state_rank], &device);
            (
                Tensor::cat(vec![v_bshmp, pad_bShmp], 1),
                Tensor::cat(vec![da_bsh, pad_bSh.clone()], 1),
                Tensor::cat(vec![gamma_bsh, pad_bSh.clone()], 1),
                Tensor::cat(vec![scale_bsh, pad_bSh], 1),
                Tensor::cat(vec![b_bsmhr, pad_bSmhr.clone()], 1),
                Tensor::cat(vec![c_bsmhr, pad_bSmhr], 1),
            )
        };

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let v_bnlmhp =
            v_bShmp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let gamma_bnlh = gamma_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let scale_bnlh = scale_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bnlmhr = c_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

        // ── Step 7: Run single-pass form SSD ───────────────────────────────────────
        let ssd_input = Mamba3SingleSsdInput {
            v_bnlmhp,
            b_bnlmhr,
            c_bnlmhr,
            da_bnlh,
            gamma_bnlh,
            scale_bnlh,
            initial_state_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        };
        let (y_bnlmhp, final_state_bhpr) = ssd_input.run(ssd_path);

        san(&y_bnlmhp);
        san(&final_state_bhpr);
        cache.ssm_bhpr = final_state_bhpr;

        // ── Step 8: Unpad ─────────────────────────────────────────────────────
        let y_bSmhp = y_bnlmhp.reshape([batch, sequence_padded, mimo_rank, nheads, per_head_dim]);
        let y_bsmhp = if pad == 0 {
            y_bSmhp
        } else {
            y_bSmhp.narrow(1, 0, sequence)
        };

        // ── Step 9: D skip + gate + MIMO_O down-projection ────────────────────
        // D skip uses raw x ⊙ mimo_x (not γ-scaled, matching forward).
        let v_raw_bsmhp =
            helpers::build_v_with_mimo::<4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);
        let d_111h1 = self.d_h.val().unsqueeze_dims::<5>(&[0, 1, 2, 4]);
        let y_bsmhp = y_bsmhp + d_111h1 * v_raw_bsmhp;

        let y_bsi = if mimo_rank > 1 {
            let mimo_z_hmp = self.mimo_z_hmp.as_ref().map(|p| p.val()).unwrap();
            let mimo_o_hmp = self.mimo_o_hmp.as_ref().map(|p| p.val()).unwrap();

            let z_bshp = z_bsi
                .clone()
                .reshape([batch, sequence, nheads, per_head_dim]);
            let z_bsmhp = {
                let z_bsmhp = z_bshp.unsqueeze_dim::<5>(2).expand([
                    batch,
                    sequence,
                    mimo_rank,
                    nheads,
                    per_head_dim,
                ]);
                let mimo_z_bsmhp = mimo_z_hmp
                    .permute([1, 0, 2])
                    .unsqueeze_dims::<5>(&[0, 1])
                    .expand([batch, sequence, mimo_rank, nheads, per_head_dim]);
                z_bsmhp * mimo_z_bsmhp
            };

            let y_combined_bsmhp = match &self.out_norm {
                Some(norm) => norm.forward(y_bsmhp, z_bsmhp),
                None => y_bsmhp * Silu::new().forward(z_bsmhp),
            };

            let mimo_o_bsmhp = mimo_o_hmp
                .permute([1, 0, 2])
                .unsqueeze_dims::<5>(&[0, 1])
                .expand([batch, sequence, mimo_rank, nheads, per_head_dim]);
            let y_bshp: Tensor<4> = (y_combined_bsmhp * mimo_o_bsmhp).sum_dim(2).squeeze_dim(2);
            y_bshp.reshape([batch, sequence, d_inner])
        } else {
            let y_bshp: Tensor<4> = y_bsmhp.squeeze_dim(2);
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
        cache.k_state_bmhr = b_last_bmhr;
        cache.v_state_bhp = x_last_bhp;
        // The new cumulative rotation (Complex2D: angle wrapped to [−π, π];
        // Quaternion4D: the cumulative quaternion), from [`rotate_bc_forward`] —
        // matches the double-ssd cache convention so the two inter-convert.
        cache.rotation = new_rotation;

        (out_bsm, cache)
    }
}

// ---------------------------------------------------------------------------
// Mamba3::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    impl Mamba3 {
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
        pub fn step_single_ssd(
            &self,
            input_bd: Tensor<2>,
            cache: Option<Mamba3SingleSsdCache>,
        ) -> (Tensor<2>, Mamba3SingleSsdCache) {
            // Token-by-token decoding always uses the recurrent (double-ssd)
            // form. A single-ssd cache holds the trapezoid state at a sequence
            // boundary, where the single- and double-ssd accumulators coincide
            // (see the `From` impls in `crate::mamba3::cache`), so converting in
            // and back out is lossless. The single recurrence step is itself a
            // boundary-to-boundary transition, so the round-trip stays exact.
            let cache = cache.map(Mamba3DoubleSsdCache::from);
            let (out_bd, cache) = self.step_double_ssd(input_bd, cache);
            (out_bd, cache.into())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — forward_single_ssd parity with forward_double_ssd, step, and split-prefill
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
