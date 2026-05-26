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

use crate::mamba3::double_ssd::double_ssd::apply_rope_partial;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::single_ssd::prelude::*;
use crate::utils::sanity::sanity as san;
use crate::utils::silu::Silu;
use burn::prelude::*;

impl<B: Backend + Mamba3SingleSsdBackendExt> Mamba3<B> {
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
        input_bsm: Tensor<B, 3>,
        cache: Option<Mamba3SingleSsdCache<B>>,
        ssd_path: Mamba3SingleSsdPath,
    ) -> (Tensor<B, 3>, Mamba3SingleSsdCache<B>) {
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
            let cum_angle_bha = Tensor::zeros([batch, nheads, num_rope_angles], &device);
            Mamba3SingleSsdCache {
                ssm_bhpr,
                k_state_bmhr,
                v_state_bhp,
                cum_angle_bha,
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
                theta_bsa
        ] = crate::utils::split::split_into(
            proj_bsd,
            [
                d_inner, d_inner,
                bc_size, bc_size,
                nheads, nheads, nheads,
                num_rope_angles,
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
        let b_bsmhr = helpers::qk_norm_expand_bias::<_, 5, 6>(
            b_raw_bsMGR.reshape([batch, sequence, mimo_rank, ngroups, state_rank]),
            &self.b_norm,
            self.b_bias_hmr.val(),
            3,
            nheads,
        );
        let c_bsmhr = helpers::qk_norm_expand_bias::<_, 5, 6>(
            c_raw_bsMGR.reshape([batch, sequence, mimo_rank, ngroups, state_rank]),
            &self.c_norm,
            self.c_bias_hmr.val(),
            3,
            nheads,
        );

        // ── Step 5: Data-dependent cumulative RoPE angles ─────────────────────
        let theta_scaled_bsa = theta_bsa.tanh() * std::f32::consts::PI;
        let raw_angles_bsha =
            dt_bsh.clone().unsqueeze_dim::<4>(3) * theta_scaled_bsa.unsqueeze_dim::<4>(2);
        let cumsum_bsha = raw_angles_bsha.cumsum(1);
        let cum_angles_bsha = cache.cum_angle_bha.clone().unsqueeze_dim::<4>(1) + cumsum_bsha;
        san(&cum_angles_bsha);

        let cum_angles_bsmha = cum_angles_bsha.clone().unsqueeze_dim::<5>(2).expand([
            batch,
            sequence,
            mimo_rank,
            nheads,
            num_rope_angles,
        ]);

        let rotate_pairwise = mimo_rank == 1;
        let rope_dim = self.rope_dim;
        let b_bsmhr = apply_rope_partial::<B, 5>(
            b_bsmhr,
            cum_angles_bsmha.clone(),
            rope_dim,
            rotate_pairwise,
        );
        let c_bsmhr =
            apply_rope_partial::<B, 5>(c_bsmhr, cum_angles_bsmha, rope_dim, rotate_pairwise);
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
        let v_prev_mimo_bmhp = helpers::build_v_with_mimo::<_, 3, 4>(
            cache.v_state_bhp.clone(),
            mimo_x_hmp.as_ref(),
            1,
        ); // [batch, mimo_rank, nheads, per_head_dim]
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
        let v_bshmp = helpers::build_v_with_mimo::<_, 4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);
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
        let (y_bnlmhp, final_state_bhpr) = ssd_path.run(ssd_input);

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
            helpers::build_v_with_mimo::<_, 4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);
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
            let y_bshp: Tensor<B, 4> = (y_combined_bsmhp * mimo_o_bsmhp).sum_dim(2).squeeze_dim(2);
            y_bshp.reshape([batch, sequence, d_inner])
        } else {
            let y_bshp: Tensor<B, 4> = y_bsmhp.squeeze_dim(2);
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
        cache.cum_angle_bha = cum_angles_bsha
            .narrow(1, sequence - 1, 1)
            .squeeze_dim::<3>(1);

        (out_bsm, cache)
    }
}

// ---------------------------------------------------------------------------
// Mamba3::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    impl<B: Backend> Mamba3<B> {
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
            _input_bd: Tensor<B, 2>,
            _cache: Option<Mamba3SingleSsdCache<B>>,
        ) -> (Tensor<B, 2>, Mamba3SingleSsdCache<B>) {
            // currently not changed from the double_ssd
            todo!("step method for single_ssd form is not yet implemented")

            // Hint:
            // Token-by-token decoding always uses the recurrent form (double-ssd cache).
            // When running a step that uses a single-ssd cache, the single-ssd cache
            // would first need converting into the double-ssd form.
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — forward_single_ssd parity with forward_double_ssd, step, and split-prefill
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
    use crate::mamba3::double_ssd::prelude::*;
    use crate::mamba3::mamba3::Mamba3Config;
    use burn::backend::{Autodiff, Flex};
    use burn::module::Param;
    use burn::tensor::Distribution;

    type InnerB = Flex;
    type B = Autodiff<InnerB>;
    type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

    fn small_config() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
    }

    fn small_config_mimo() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_mimo_rank(2)
    }

    fn cfg_ngroups2() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_ngroups(2)
    }

    fn cfg_mimo_ngroups2() -> Mamba3Config {
        cfg_ngroups2().with_mimo_rank(2)
    }

    /// Build a matched pair of initial caches for cross-algorithm parity
    /// (`forward_double_ssd`/`step_double_ssd` use [`Mamba3DoubleSsdCache`];
    /// `forward_single_ssd` uses [`Mamba3SingleSsdCache`]).
    /// With `random = true` the SSM state and cumulative RoPE angle are random while
    /// the previous-token K/V history is **zero** — so the single-ssd form's
    /// boundary-β seed is zero and both forms share the exact same logical initial state.
    /// With `random = false` everything is zero.
    fn build_cross_caches(
        cfg: &Mamba3Config,
        batch: usize,
        random: bool,
    ) -> (Mamba3DoubleSsdCache<B>, Mamba3SingleSsdCache<B>) {
        let device: Device = Default::default();
        let nheads = cfg.nheads();
        let per_head_dim = cfg.per_head_dim;
        let state_rank = cfg.state_rank;
        let mimo_rank = cfg.mimo_rank;
        let num_rope_angles = cfg.num_rope_angles();
        let dist = Distribution::Normal(0.0, 1.0);
        let ssm = if random {
            Tensor::<InnerB, 4>::random([batch, nheads, per_head_dim, state_rank], dist, &device)
        } else {
            Tensor::<InnerB, 4>::zeros([batch, nheads, per_head_dim, state_rank], &device)
        };
        let angle = if random {
            Tensor::<InnerB, 3>::random([batch, nheads, num_rope_angles], dist, &device)
        } else {
            Tensor::<InnerB, 3>::zeros([batch, nheads, num_rope_angles], &device)
        };
        // Zero previous-token history so the two cache forms agree logically.
        let k = Tensor::<InnerB, 4>::zeros([batch, mimo_rank, nheads, state_rank], &device);
        let v = Tensor::<InnerB, 3>::zeros([batch, nheads, per_head_dim], &device);
        let c3 = Mamba3DoubleSsdCache {
            ssm_bhpr: Tensor::from_inner(ssm.clone()),
            k_state_bmhr: Tensor::from_inner(k.clone()),
            v_state_bhp: Tensor::from_inner(v.clone()),
            cum_angle_bha: Tensor::from_inner(angle.clone()),
        };
        let cm = Mamba3SingleSsdCache {
            ssm_bhpr: Tensor::from_inner(ssm),
            k_state_bmhr: Tensor::from_inner(k),
            v_state_bhp: Tensor::from_inner(v),
            cum_angle_bha: Tensor::from_inner(angle),
        };
        (c3, cm)
    }

    /// Build an initial [`Mamba3SingleSsdCache`] for the single-ssd form continuity test.
    /// With `random = true` *every* field (including the previous-token K/V
    /// history) is random, exercising forward_single_ssd continuation from an arbitrary
    /// single-ssd form state.
    fn build_single_ssd_cache(
        cfg: &Mamba3Config,
        batch: usize,
        random: bool,
    ) -> Mamba3SingleSsdCache<B> {
        let device: Device = Default::default();
        let nheads = cfg.nheads();
        let per_head_dim = cfg.per_head_dim;
        let state_rank = cfg.state_rank;
        let mimo_rank = cfg.mimo_rank;
        let num_rope_angles = cfg.num_rope_angles();
        let dist = Distribution::Normal(0.0, 1.0);
        let mk4 = |shape: [usize; 4]| {
            let t = if random {
                Tensor::<InnerB, 4>::random(shape, dist, &device)
            } else {
                Tensor::<InnerB, 4>::zeros(shape, &device)
            };
            Tensor::from_inner(t)
        };
        let mk3 = |shape: [usize; 3]| {
            let t = if random {
                Tensor::<InnerB, 3>::random(shape, dist, &device)
            } else {
                Tensor::<InnerB, 3>::zeros(shape, &device)
            };
            Tensor::from_inner(t)
        };
        Mamba3SingleSsdCache {
            ssm_bhpr: mk4([batch, nheads, per_head_dim, state_rank]),
            k_state_bmhr: mk4([batch, mimo_rank, nheads, state_rank]),
            v_state_bhp: mk3([batch, nheads, per_head_dim]),
            cum_angle_bha: mk3([batch, nheads, num_rope_angles]),
        }
    }

    /// Per-run gradient bundle (subset of params; mirrors the equivalent struct
    /// in `mamba3::tests` but kept local to avoid cross-module visibility).
    struct RunGrads {
        out: Tensor<InnerB, 3>,
        d_input: Tensor<InnerB, 3>,
        d_in_proj_w: Tensor<InnerB, 2>,
        d_dt_bias: Tensor<InnerB, 1>,
        d_d: Tensor<InnerB, 1>,
        d_b_norm_gamma: Tensor<InnerB, 1>,
        d_c_norm_gamma: Tensor<InnerB, 1>,
        d_b_bias: Tensor<InnerB, 3>,
        d_c_bias: Tensor<InnerB, 3>,
        d_out_proj_w: Tensor<InnerB, 2>,
    }

    fn run_with_grads(
        model: &Mamba3<B>,
        input: &Param<Tensor<B, 3>>,
        head: &Tensor<InnerB, 3>,
        forward: impl FnOnce(&Mamba3<B>, Tensor<B, 3>) -> Tensor<B, 3>,
    ) -> RunGrads {
        let out = forward(model, input.val());
        let out_inner = out.clone().inner();
        let head = Tensor::from_inner(head.clone());
        let loss = (out * head).sum();
        let grads = loss.backward();
        RunGrads {
            out: out_inner,
            d_input: input.val().grad(&grads).expect("grad input"),
            d_in_proj_w: model
                .in_proj
                .weight
                .val()
                .grad(&grads)
                .expect("in_proj.weight"),
            d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("dt_bias_h"),
            d_d: model.d_h.val().grad(&grads).expect("d_h"),
            d_b_norm_gamma: model.b_norm.gamma.val().grad(&grads).expect("b_norm.gamma"),
            d_c_norm_gamma: model.c_norm.gamma.val().grad(&grads).expect("c_norm.gamma"),
            d_b_bias: model.b_bias_hmr.val().grad(&grads).expect("b_bias_hmr"),
            d_c_bias: model.c_bias_hmr.val().grad(&grads).expect("c_bias_hmr"),
            d_out_proj_w: model
                .out_proj
                .weight
                .val()
                .grad(&grads)
                .expect("out_proj.weight"),
        }
    }

    fn check_grads_match(label: &str, a: &RunGrads, b: &RunGrads, grad_tol: f32) {
        let mut failures: Vec<String> = Vec::new();
        macro_rules! check {
            ($field:ident, $name:expr) => {{
                let d = (a.$field.clone() - b.$field.clone())
                    .abs()
                    .max()
                    .into_scalar();
                eprintln!("{:>40} {:>16} | max abs diff = {:>10.6}", label, $name, d);
                if d >= grad_tol {
                    failures.push(format!(
                        "{}: grad of {} max abs diff = {:.6} (tol {})",
                        label, $name, d, grad_tol
                    ));
                }
            }};
        }
        check!(d_input, "input");
        check!(d_in_proj_w, "in_proj.weight");
        check!(d_dt_bias, "dt_bias_h");
        check!(d_d, "d_h");
        check!(d_b_norm_gamma, "b_norm.gamma");
        check!(d_c_norm_gamma, "c_norm.gamma");
        check!(d_b_bias, "b_bias_hmr");
        check!(d_c_bias, "c_bias_hmr");
        check!(d_out_proj_w, "out_proj.weight");
        assert!(
            failures.is_empty(),
            "gradient mismatches:\n  {}",
            failures.join("\n  ")
        );
    }

    fn param_input(input: &Tensor<InnerB, 3>) -> Param<Tensor<B, 3>> {
        Param::from_tensor(Tensor::from_inner(input.clone()))
    }

    /// Random downstream heads for the single-ssd form continuity loss (output plus
    /// every single-ssd cache field).
    struct Heads {
        out: Tensor<InnerB, 3>,
        ssm: Tensor<InnerB, 4>,
        k: Tensor<InnerB, 4>,
        v: Tensor<InnerB, 3>,
        angle: Tensor<InnerB, 3>,
    }

    /// A [`RunGrads`] plus the final single-ssd cache fields, for the continuity test.
    struct SingleSsdRun {
        rg: RunGrads,
        final_ssm: Tensor<InnerB, 4>,
        final_k: Tensor<InnerB, 4>,
        final_v: Tensor<InnerB, 3>,
        final_angle: Tensor<InnerB, 3>,
    }

    /// Like [`run_with_grads`] but the loss couples the output with every final
    /// single-ssd cache field, and the final cache is returned for comparison. Both
    /// runs being compared use `forward_single_ssd`, so the single-ssd cache semantics match.
    fn run_with_grads_single_ssd(
        model: &Mamba3<B>,
        input: &Param<Tensor<B, 3>>,
        heads: &Heads,
        runner: impl FnOnce(&Mamba3<B>, Tensor<B, 3>) -> (Tensor<B, 3>, Mamba3SingleSsdCache<B>),
    ) -> SingleSsdRun {
        let (out, cache) = runner(model, input.val());
        let out_inner = out.clone().inner();
        let ssm = cache.ssm_bhpr;
        let k = cache.k_state_bmhr;
        let v = cache.v_state_bhp;
        let angle = cache.cum_angle_bha;
        let final_ssm = ssm.clone().inner();
        let final_k = k.clone().inner();
        let final_v = v.clone().inner();
        let final_angle = angle.clone().inner();

        let out_head = Tensor::from_inner(heads.out.clone());
        let ssm_head = Tensor::from_inner(heads.ssm.clone());
        let k_head = Tensor::from_inner(heads.k.clone());
        let v_head = Tensor::from_inner(heads.v.clone());
        let angle_head = Tensor::from_inner(heads.angle.clone());
        let loss = (out * out_head).sum()
            + (ssm * ssm_head).sum()
            + (k * k_head).sum()
            + (v * v_head).sum()
            + (angle * angle_head).sum();
        let grads = loss.backward();

        let rg = RunGrads {
            out: out_inner,
            d_input: input.val().grad(&grads).expect("grad input"),
            d_in_proj_w: model
                .in_proj
                .weight
                .val()
                .grad(&grads)
                .expect("in_proj.weight"),
            d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("dt_bias_h"),
            d_d: model.d_h.val().grad(&grads).expect("d_h"),
            d_b_norm_gamma: model.b_norm.gamma.val().grad(&grads).expect("b_norm.gamma"),
            d_c_norm_gamma: model.c_norm.gamma.val().grad(&grads).expect("c_norm.gamma"),
            d_b_bias: model.b_bias_hmr.val().grad(&grads).expect("b_bias_hmr"),
            d_c_bias: model.c_bias_hmr.val().grad(&grads).expect("c_bias_hmr"),
            d_out_proj_w: model
                .out_proj
                .weight
                .val()
                .grad(&grads)
                .expect("out_proj.weight"),
        };
        SingleSsdRun {
            rg,
            final_ssm,
            final_k,
            final_v,
            final_angle,
        }
    }

    /// Compare output, every final single-ssd cache field, and parameter gradients.
    fn check_single_ssd_match(
        label: &str,
        a: &SingleSsdRun,
        b: &SingleSsdRun,
        val_tol: f32,
        grad_tol: f32,
    ) {
        use crate::utils::test_helpers::max_abs_diff;
        let vals = [
            ("output", max_abs_diff(a.rg.out.clone(), b.rg.out.clone())),
            (
                "final ssm",
                max_abs_diff(a.final_ssm.clone(), b.final_ssm.clone()),
            ),
            (
                "final k_state",
                max_abs_diff(a.final_k.clone(), b.final_k.clone()),
            ),
            (
                "final v_state",
                max_abs_diff(a.final_v.clone(), b.final_v.clone()),
            ),
            (
                "final cum_angle",
                max_abs_diff(a.final_angle.clone(), b.final_angle.clone()),
            ),
        ];
        for (name, d) in vals {
            assert!(
                d < val_tol,
                "{label}: {name} max abs diff = {d:.6} (tol {val_tol})"
            );
        }
        check_grads_match(label, &a.rg, &b.rg, grad_tol);
    }

    /// Guard: a random initial state must actually change the forward_single_ssd output
    /// (vs a *zero* single-ssd cache). Otherwise the initial state is being silently
    /// ignored, which would make the parity comparisons pass trivially.
    fn guard_random_init_consumed(
        random_init: bool,
        model: &Mamba3<B>,
        cfg: &Mamba3Config,
        batch: usize,
        input: &Tensor<InnerB, 3>,
        ssd_path: &Mamba3SsdPath,
        random_out: &Tensor<InnerB, 3>,
    ) {
        if !random_init {
            return;
        }
        use crate::utils::test_helpers::max_abs_diff;
        let (out_zero, _) = model.forward_single_ssd(
            Tensor::from_inner(input.clone()),
            Some(build_single_ssd_cache(cfg, batch, false)),
            Mamba3SingleSsdPath::from(ssd_path.clone()),
        );
        let d = max_abs_diff(random_out.clone(), out_zero.inner());
        assert!(
            d > 1e-3,
            "random initial state appears ignored: random-init vs zero-init \
             output max abs diff = {d:.6} (expected a clear difference)"
        );
    }

    /// forward_single_ssd ≡ forward_double_ssd on values and gradients, from the same
    /// initial state. With `random_init = true` the shared logical initial state
    /// is random (random SSM state + cumulative RoPE angle; zero previous-token
    /// history so the single-ssd and double-ssd forms coincide). The output and all
    /// parameter gradients must agree. The single-ssd cache SSM accumulator itself is
    /// not compared here (different semantics from the double-form state); the
    /// single-ssd cache is compared in `run_forward_single_ssd_split_matches_full`.
    fn forward_match(cfg: Mamba3Config, ssd_path: Mamba3SsdPath, random_init: bool) {
        let device: Device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 5;
        let d_model = cfg.d_model;
        let normal = Distribution::Normal(0.0, 1.0);

        let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
        let head = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);

        let (c3, cm) = build_cross_caches(&cfg, batch, random_init);

        let input_a = param_input(&input);
        let c3c = c3;
        let path_a = Mamba3DoubleSsdPath::from(ssd_path.clone());
        let r_fwd_double_ssd = run_with_grads(&model, &input_a, &head, |m, x| {
            let (out, _) = m.forward_double_ssd(x, Some(c3c), path_a);
            out
        });

        let input_b = param_input(&input);
        let cmc = cm;
        let single_ssd_b = Mamba3SingleSsdPath::from(ssd_path.clone());
        let r_fwd_single_ssd = run_with_grads(&model, &input_b, &head, |m, x| {
            let (out, _) = m.forward_single_ssd(x, Some(cmc), single_ssd_b);
            out
        });

        let diff = (r_fwd_double_ssd.out.clone() - r_fwd_single_ssd.out.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            diff < 1e-4,
            "forward_double_ssd vs forward_single_ssd max absolute difference = {diff:.6} (expected < 1e-4)"
        );
        check_grads_match(
            "forward_single_ssd vs forward_double_ssd",
            &r_fwd_double_ssd,
            &r_fwd_single_ssd,
            1e-3,
        );

        guard_random_init_consumed(
            random_init,
            &model,
            &cfg,
            batch,
            &input,
            &ssd_path,
            &r_fwd_single_ssd.out,
        );
    }

    #[test]
    fn forward_match_simple() {
        forward_match(small_config(), Mamba3SsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn forward_match_random_init() {
        forward_match(small_config(), Mamba3SsdPath::Minimal(Some(4)), true);
    }

    #[test]
    fn forward_match_ngroups2() {
        forward_match(cfg_ngroups2(), Mamba3SsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn forward_match_ngroups2_random_init() {
        forward_match(cfg_ngroups2(), Mamba3SsdPath::Minimal(Some(4)), true);
    }

    #[test]
    fn forward_match_mimo() {
        forward_match(small_config_mimo(), Mamba3SsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn forward_match_mimo_random_init() {
        forward_match(small_config_mimo(), Mamba3SsdPath::Minimal(Some(4)), true);
    }

    #[test]
    fn forward_match_mimo_ngroups2() {
        forward_match(cfg_mimo_ngroups2(), Mamba3SsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn forward_match_mimo_ngroups2_random_init() {
        forward_match(cfg_mimo_ngroups2(), Mamba3SsdPath::Minimal(Some(4)), true);
    }

    #[test]
    fn forward_match_serial() {
        forward_match(small_config(), Mamba3SsdPath::Serial(Some(4)), false);
    }

    #[test]
    fn forward_match_serial_mimo() {
        forward_match(small_config_mimo(), Mamba3SsdPath::Serial(Some(4)), false);
    }

    #[test]
    fn forward_match_recalc() {
        forward_match(
            small_config(),
            Mamba3SsdPath::SerialRecalculated(Some(4)),
            false,
        );
    }

    #[test]
    fn forward_match_recalc_mimo() {
        forward_match(
            small_config_mimo(),
            Mamba3SsdPath::SerialRecalculated(Some(4)),
            false,
        );
    }

    /// forward_single_ssd ≡ token-by-token step on values and gradients, from the same
    /// initial state (random when `random_init = true`, with zero previous-token
    /// history so the single-ssd and recurrent forms coincide).
    fn run_forward_single_ssd_matches_step(
        cfg: Mamba3Config,
        single_ssd_path: Mamba3SingleSsdPath,
        random_init: bool,
    ) {
        let device: Device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 5;
        let d_model = cfg.d_model;
        let normal = Distribution::Normal(0.0, 1.0);

        let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
        let head = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);

        let (_c3, cm) = build_cross_caches(&cfg, batch, random_init);

        let input_a = param_input(&input);
        let cmc = cm.clone();
        let single_ssd_a = single_ssd_path.clone();
        let r_fwd_single_ssd = run_with_grads(&model, &input_a, &head, |m, x| {
            let (out, _) = m.forward_single_ssd(x, Some(cmc), single_ssd_a);
            out
        });

        let input_b = param_input(&input);
        let cmc = cm;
        let r_step = run_with_grads(&model, &input_b, &head, |m, x| {
            let mut cache: Option<Mamba3SingleSsdCache<B>> = Some(cmc);
            let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
                let (out_t, new_cache) = m.step_single_ssd(token, cache);
                cache = Some(new_cache);
                outs.push(out_t);
            }
            Tensor::stack(outs, 1)
        });

        let diff = (r_fwd_single_ssd.out.clone() - r_step.out.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            diff < 1e-4,
            "forward_single_ssd vs step max absolute difference = {diff:.6} (expected < 1e-4)"
        );
        check_grads_match(
            "forward_single_ssd vs step",
            &r_fwd_single_ssd,
            &r_step,
            1e-3,
        );

        guard_random_init_consumed(
            random_init,
            &model,
            &cfg,
            batch,
            &input,
            &(single_ssd_path.into()),
            &r_fwd_single_ssd.out,
        );
    }

    #[test]
    fn forward_single_ssd_matches_step() {
        run_forward_single_ssd_matches_step(
            small_config(),
            Mamba3SingleSsdPath::Minimal(Some(4)),
            false,
        );
    }

    #[test]
    fn forward_single_ssd_matches_step_random_init() {
        run_forward_single_ssd_matches_step(
            small_config(),
            Mamba3SingleSsdPath::Minimal(Some(4)),
            true,
        );
    }

    #[test]
    fn forward_single_ssd_matches_step_mimo() {
        run_forward_single_ssd_matches_step(
            small_config_mimo(),
            Mamba3SingleSsdPath::Minimal(Some(4)),
            false,
        );
    }

    #[test]
    fn forward_single_ssd_matches_step_mimo_random_init() {
        run_forward_single_ssd_matches_step(
            small_config_mimo(),
            Mamba3SingleSsdPath::Minimal(Some(4)),
            true,
        );
    }

    #[test]
    fn forward_single_ssd_matches_step_serial() {
        run_forward_single_ssd_matches_step(
            small_config(),
            Mamba3SingleSsdPath::Serial(Some(4)),
            false,
        );
    }

    #[test]
    fn forward_single_ssd_matches_step_serial_mimo() {
        run_forward_single_ssd_matches_step(
            small_config_mimo(),
            Mamba3SingleSsdPath::Serial(Some(4)),
            false,
        );
    }

    #[test]
    fn forward_single_ssd_matches_step_recalc() {
        run_forward_single_ssd_matches_step(
            small_config(),
            Mamba3SingleSsdPath::SerialRecalculated(Some(4)),
            false,
        );
    }

    #[test]
    fn forward_single_ssd_matches_step_recalc_mimo() {
        run_forward_single_ssd_matches_step(
            small_config_mimo(),
            Mamba3SingleSsdPath::SerialRecalculated(Some(4)),
            false,
        );
    }

    /// forward_single_ssd continuation from a **random** initial single-ssd cache:
    /// `forward_single_ssd(full, cache) ≡ forward_single_ssd(prefix, cache)` then
    /// `forward_single_ssd(suffix, mid_cache)`. Compares outputs, the final single-ssd cache,
    /// and gradients. This replaces the old zero-init split-vs-full test: a
    /// random initial cache subsumes the chunked-prefill continuity guarantee
    /// from an arbitrary starting state, and the guard at the end confirms the
    /// initial cache is actually consumed (not silently ignored).
    fn run_forward_single_ssd_split_matches_full(
        cfg: Mamba3Config,
        single_ssd_path: Mamba3SingleSsdPath,
    ) {
        let device: Device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 6;
        let split = 2;
        let d_model = cfg.d_model;
        let nheads = cfg.nheads();
        let per_head_dim = cfg.per_head_dim;
        let state_rank = cfg.state_rank;
        let mimo_rank = cfg.mimo_rank;
        let num_rope_angles = cfg.num_rope_angles();
        let normal = Distribution::Normal(0.0, 1.0);

        let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
        let heads = Heads {
            out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
            ssm: Tensor::<InnerB, 4>::random(
                [batch, nheads, per_head_dim, state_rank],
                normal,
                &device,
            ),
            k: Tensor::<InnerB, 4>::random([batch, mimo_rank, nheads, state_rank], normal, &device),
            v: Tensor::<InnerB, 3>::random([batch, nheads, per_head_dim], normal, &device),
            angle: Tensor::<InnerB, 3>::random([batch, nheads, num_rope_angles], normal, &device),
        };

        let init_cache = build_single_ssd_cache(&cfg, batch, true);

        let input_full = param_input(&input);
        let cache_full = init_cache.clone();
        let single_ssd_f = single_ssd_path.clone();
        let r_full = run_with_grads_single_ssd(&model, &input_full, &heads, |m, x| {
            m.forward_single_ssd(x, Some(cache_full), single_ssd_f)
        });

        let input_split = param_input(&input);
        let cache_split = init_cache;
        let single_ssd_s = single_ssd_path.clone();
        let r_split = run_with_grads_single_ssd(&model, &input_split, &heads, |m, x| {
            let prefix = x.clone().narrow(1, 0, split);
            let suffix = x.narrow(1, split, seq_len - split);
            let (out_prefix, mid) =
                m.forward_single_ssd(prefix, Some(cache_split), single_ssd_s.clone());
            let (out_suffix, last) = m.forward_single_ssd(suffix, Some(mid), single_ssd_s);
            (Tensor::cat(vec![out_prefix, out_suffix], 1), last)
        });

        check_single_ssd_match(
            "forward_single_ssd split vs full",
            &r_full,
            &r_split,
            1e-4,
            1e-3,
        );

        // Guard: the random initial single_ssd cache must change the full output.
        {
            use crate::utils::test_helpers::max_abs_diff;
            let (out_zero, _) = model.forward_single_ssd(
                Tensor::from_inner(input.clone()),
                Some(build_single_ssd_cache(&cfg, batch, false)),
                single_ssd_path.clone(),
            );
            let d = max_abs_diff(r_full.rg.out.clone(), out_zero.inner());
            assert!(
                d > 1e-3,
                "random initial state appears ignored: random-init vs zero-init \
                 output max abs diff = {d:.6} (expected a clear difference)"
            );
        }
    }

    #[test]
    fn forward_single_ssd_split_matches_full() {
        run_forward_single_ssd_split_matches_full(
            small_config(),
            Mamba3SingleSsdPath::Minimal(Some(4)),
        );
    }

    #[test]
    fn forward_single_ssd_split_matches_full_mimo() {
        run_forward_single_ssd_split_matches_full(
            small_config_mimo(),
            Mamba3SingleSsdPath::Minimal(Some(4)),
        );
    }

    #[test]
    fn forward_single_ssd_split_matches_full_serial() {
        run_forward_single_ssd_split_matches_full(
            small_config(),
            Mamba3SingleSsdPath::Serial(Some(4)),
        );
    }

    #[test]
    fn forward_single_ssd_split_matches_full_serial_mimo() {
        run_forward_single_ssd_split_matches_full(
            small_config_mimo(),
            Mamba3SingleSsdPath::Serial(Some(4)),
        );
    }

    #[test]
    fn forward_single_ssd_split_matches_full_recalc() {
        run_forward_single_ssd_split_matches_full(
            small_config(),
            Mamba3SingleSsdPath::SerialRecalculated(Some(4)),
        );
    }

    #[test]
    fn forward_single_ssd_split_matches_full_recalc_mimo() {
        run_forward_single_ssd_split_matches_full(
            small_config_mimo(),
            Mamba3SingleSsdPath::SerialRecalculated(Some(4)),
        );
    }
}
