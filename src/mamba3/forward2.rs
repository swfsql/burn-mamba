//! # `Mamba3::forward2` — Single-Pass (Merged-Form) Trapezoidal Forward
//!
//! This module provides the `forward2` method on [`Mamba3`]: an alternative to
//! [`Mamba3::forward`] that computes the trapezoidal recurrence with **one**
//! chunkwise SSD pass instead of two.
//!
//! ## Why two paths?
//!
//! The existing [`Mamba3::forward`] is the burn-mamba implementation of the
//! `VikramLex/mamba3-minimal` decomposition:
//!
//! ```text
//!   hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ      (original trapezoidal)
//!
//!   forward:    h = SSD(γ-scaled V, B)   +   SSD(β-scaled V_shifted, B_shifted)
//! ```
//!
//! This is simple to derive and to verify (everything reuses the standard SSD)
//! but doubles the intra-chunk and chunk-state memory during training.
//!
//! [`Mamba3::forward2`] is the burn-mamba implementation of the **official
//! Mamba-3 algorithm** as shipped in Triton (SISO) and Tilelang (MIMO):
//!
//! ```text
//!   scaleₜ = γₜ + (1 − λₜ₊₁) · Δₜ₊₁
//!
//!   forward2:    h' = SSD(V_raw, K_scaled = scaleₜ B) with:
//!                 * strict lower-triangular intra-chunk mask
//!                 * additive γ-weighted same-step correction
//!                 * boundary β seed (1−λ₀) Δ₀ Kₜ₋₁ ⊗ xₜ₋₁
//! ```
//!
//! The two formulations are mathematically equivalent; the parity tests in
//! `mamba3.rs` and elsewhere assert this on small configurations.
//!
//! Reference: `refs/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`,
//! `refs/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`.

use crate::mamba3::helpers;
use crate::mamba3::mamba3::apply_rope_partial;
use crate::mamba3::prelude::*;
use crate::utils::sanity::sanity as san;
use crate::utils::silu::Silu;
use burn::prelude::*;

impl<B: Backend + Mamba3TrapBackendExt> Mamba3<B> {
    /// Process a full input sequence using the **merged-form (single-pass)**
    /// trapezoidal algorithm.
    ///
    /// Functionally equivalent to [`Self::forward`] but uses approximately half
    /// the SSD memory during training. Cache is a separate type
    /// ([`Mamba3MergedCache`]) because the stored hidden state has different
    /// semantics than the original-form cache used by [`Self::forward`].
    ///
    /// # Shapes
    /// - `input_bsm`: `[batch, sequence, d_model]`
    /// - output: `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward2(
        &self,
        input_bsm: Tensor<B, 3>,
        cache: Option<Mamba3MergedCache<B>>,
        ssd_path: Mamba3TrapSsdPath,
    ) -> (Tensor<B, 3>, Mamba3MergedCache<B>) {
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
            Mamba3MergedCache {
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
        let helpers::TrapCoeffs {
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
        // merged SSM state. λ₀, Δ₀ are taken from the current call's first
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

        // ── Step 7: Run merged-form SSD ───────────────────────────────────────
        let ssd_input = Mamba3TrapSsdInput {
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
// Tests — forward2 parity with forward (double-SSD), step, and split-prefill
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
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
    /// (`forward`/`step` use [`Mamba3Cache`]; `forward2` uses
    /// [`Mamba3MergedCache`]). With `random = true` the SSM state and cumulative
    /// RoPE angle are random while the previous-token K/V history is **zero** — so
    /// the merged form's boundary-β seed is zero and both forms share the exact
    /// same logical initial state. With `random = false` everything is zero.
    fn build_cross_caches(
        cfg: &Mamba3Config,
        batch: usize,
        random: bool,
    ) -> (Mamba3Cache<B>, Mamba3MergedCache<B>) {
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
        let c3 = Mamba3Cache {
            ssm_bhpr: Tensor::from_inner(ssm.clone()),
            k_state_bmhr: Tensor::from_inner(k.clone()),
            v_state_bhp: Tensor::from_inner(v.clone()),
            cum_angle_bha: Tensor::from_inner(angle.clone()),
        };
        let cm = Mamba3MergedCache {
            ssm_bhpr: Tensor::from_inner(ssm),
            k_state_bmhr: Tensor::from_inner(k),
            v_state_bhp: Tensor::from_inner(v),
            cum_angle_bha: Tensor::from_inner(angle),
        };
        (c3, cm)
    }

    /// Build an initial [`Mamba3MergedCache`] for the merged-form continuity test.
    /// With `random = true` *every* field (including the previous-token K/V
    /// history) is random, exercising forward2 continuation from an arbitrary
    /// merged-form state.
    fn build_merged_cache(cfg: &Mamba3Config, batch: usize, random: bool) -> Mamba3MergedCache<B> {
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
        Mamba3MergedCache {
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

    /// Random downstream heads for the merged-form continuity loss (output plus
    /// every merged-cache field).
    struct Heads {
        out: Tensor<InnerB, 3>,
        ssm: Tensor<InnerB, 4>,
        k: Tensor<InnerB, 4>,
        v: Tensor<InnerB, 3>,
        angle: Tensor<InnerB, 3>,
    }

    /// A [`RunGrads`] plus the final merged-cache fields, for the continuity test.
    struct MergedRun {
        rg: RunGrads,
        final_ssm: Tensor<InnerB, 4>,
        final_k: Tensor<InnerB, 4>,
        final_v: Tensor<InnerB, 3>,
        final_angle: Tensor<InnerB, 3>,
    }

    /// Like [`run_with_grads`] but the loss couples the output with every final
    /// merged-cache field, and the final cache is returned for comparison. Both
    /// runs being compared use `forward2`, so the merged-cache semantics match.
    fn run_with_grads_merged(
        model: &Mamba3<B>,
        input: &Param<Tensor<B, 3>>,
        heads: &Heads,
        runner: impl FnOnce(&Mamba3<B>, Tensor<B, 3>) -> (Tensor<B, 3>, Mamba3MergedCache<B>),
    ) -> MergedRun {
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
        MergedRun {
            rg,
            final_ssm,
            final_k,
            final_v,
            final_angle,
        }
    }

    /// Compare output, every final merged-cache field, and parameter gradients.
    fn check_merged_match(label: &str, a: &MergedRun, b: &MergedRun, val_tol: f32, grad_tol: f32) {
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

    /// Guard: a random initial state must actually change the forward2 output
    /// (vs a *zero* merged cache). Otherwise the initial state is being silently
    /// ignored, which would make the parity comparisons pass trivially.
    fn guard_random_init_consumed(
        random_init: bool,
        model: &Mamba3<B>,
        cfg: &Mamba3Config,
        batch: usize,
        input: &Tensor<InnerB, 3>,
        trap_path: &Mamba3TrapSsdPath,
        random_out: &Tensor<InnerB, 3>,
    ) {
        if !random_init {
            return;
        }
        use crate::utils::test_helpers::max_abs_diff;
        let (out_zero, _) = model.forward2(
            Tensor::from_inner(input.clone()),
            Some(build_merged_cache(cfg, batch, false)),
            trap_path.clone(),
        );
        let d = max_abs_diff(random_out.clone(), out_zero.inner());
        assert!(
            d > 1e-3,
            "random initial state appears ignored: random-init vs zero-init \
             output max abs diff = {d:.6} (expected a clear difference)"
        );
    }

    /// forward2 ≡ forward (double-SSD) on values and gradients, from the same
    /// initial state. With `random_init = true` the shared logical initial state
    /// is random (random SSM state + cumulative RoPE angle; zero previous-token
    /// history so the merged and double forms coincide). The output and all
    /// parameter gradients must agree. The merged-cache SSM accumulator itself is
    /// not compared here (different semantics from the double-form state); the
    /// merged cache is compared in `run_forward2_split_matches_full`.
    fn run_forward2_matches_forward(
        cfg: Mamba3Config,
        trap_path: Mamba3TrapSsdPath,
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

        let ssd_path = Mamba3SsdPath::Minimal(Some(4));
        let (c3, cm) = build_cross_caches(&cfg, batch, random_init);

        let input_a = param_input(&input);
        let c3c = c3;
        let path_a = ssd_path;
        let r_fwd = run_with_grads(&model, &input_a, &head, |m, x| {
            let (out, _) = m.forward(x, Some(c3c), path_a);
            out
        });

        let input_b = param_input(&input);
        let cmc = cm;
        let trap_b = trap_path.clone();
        let r_fwd2 = run_with_grads(&model, &input_b, &head, |m, x| {
            let (out, _) = m.forward2(x, Some(cmc), trap_b);
            out
        });

        let diff = (r_fwd.out.clone() - r_fwd2.out.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            diff < 1e-4,
            "forward vs forward2 max absolute difference = {diff:.6} (expected < 1e-4)"
        );
        check_grads_match("forward2 vs forward", &r_fwd, &r_fwd2, 1e-3);

        guard_random_init_consumed(
            random_init,
            &model,
            &cfg,
            batch,
            &input,
            &trap_path,
            &r_fwd2.out,
        );
    }

    #[test]
    fn forward2_matches_forward() {
        run_forward2_matches_forward(small_config(), Mamba3TrapSsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn forward2_matches_forward_random_init() {
        run_forward2_matches_forward(small_config(), Mamba3TrapSsdPath::Minimal(Some(4)), true);
    }

    #[test]
    fn forward2_matches_forward_ngroups2() {
        run_forward2_matches_forward(cfg_ngroups2(), Mamba3TrapSsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn forward2_matches_forward_ngroups2_random_init() {
        run_forward2_matches_forward(cfg_ngroups2(), Mamba3TrapSsdPath::Minimal(Some(4)), true);
    }

    #[test]
    fn forward2_matches_forward_mimo() {
        run_forward2_matches_forward(
            small_config_mimo(),
            Mamba3TrapSsdPath::Minimal(Some(4)),
            false,
        );
    }

    #[test]
    fn forward2_matches_forward_mimo_random_init() {
        run_forward2_matches_forward(
            small_config_mimo(),
            Mamba3TrapSsdPath::Minimal(Some(4)),
            true,
        );
    }

    #[test]
    fn forward2_matches_forward_mimo_ngroups2() {
        run_forward2_matches_forward(
            cfg_mimo_ngroups2(),
            Mamba3TrapSsdPath::Minimal(Some(4)),
            false,
        );
    }

    #[test]
    fn forward2_matches_forward_mimo_ngroups2_random_init() {
        run_forward2_matches_forward(
            cfg_mimo_ngroups2(),
            Mamba3TrapSsdPath::Minimal(Some(4)),
            true,
        );
    }

    #[test]
    fn forward2_matches_forward_serial() {
        run_forward2_matches_forward(small_config(), Mamba3TrapSsdPath::Serial(Some(4)), false);
    }

    #[test]
    fn forward2_matches_forward_serial_mimo() {
        run_forward2_matches_forward(
            small_config_mimo(),
            Mamba3TrapSsdPath::Serial(Some(4)),
            false,
        );
    }

    #[test]
    fn forward2_matches_forward_recalc() {
        run_forward2_matches_forward(
            small_config(),
            Mamba3TrapSsdPath::SerialRecalculated(Some(4)),
            false,
        );
    }

    #[test]
    fn forward2_matches_forward_recalc_mimo() {
        run_forward2_matches_forward(
            small_config_mimo(),
            Mamba3TrapSsdPath::SerialRecalculated(Some(4)),
            false,
        );
    }

    /// forward2 ≡ token-by-token step on values and gradients, from the same
    /// initial state (random when `random_init = true`, with zero previous-token
    /// history so the merged and recurrent forms coincide).
    fn run_forward2_matches_step(
        cfg: Mamba3Config,
        trap_path: Mamba3TrapSsdPath,
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

        let (c3, cm) = build_cross_caches(&cfg, batch, random_init);

        let input_a = param_input(&input);
        let cmc = cm;
        let trap_a = trap_path.clone();
        let r_fwd2 = run_with_grads(&model, &input_a, &head, |m, x| {
            let (out, _) = m.forward2(x, Some(cmc), trap_a);
            out
        });

        let input_b = param_input(&input);
        let c3c = c3;
        let r_step = run_with_grads(&model, &input_b, &head, |m, x| {
            let mut cache: Option<Mamba3Cache<B>> = Some(c3c);
            let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
                let (out_t, new_cache) = m.step(token, cache);
                cache = Some(new_cache);
                outs.push(out_t);
            }
            Tensor::stack(outs, 1)
        });

        let diff = (r_fwd2.out.clone() - r_step.out.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            diff < 1e-4,
            "forward2 vs step max absolute difference = {diff:.6} (expected < 1e-4)"
        );
        check_grads_match("forward2 vs step", &r_fwd2, &r_step, 1e-3);

        guard_random_init_consumed(
            random_init,
            &model,
            &cfg,
            batch,
            &input,
            &trap_path,
            &r_fwd2.out,
        );
    }

    #[test]
    fn forward2_matches_step() {
        run_forward2_matches_step(small_config(), Mamba3TrapSsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn forward2_matches_step_random_init() {
        run_forward2_matches_step(small_config(), Mamba3TrapSsdPath::Minimal(Some(4)), true);
    }

    #[test]
    fn forward2_matches_step_mimo() {
        run_forward2_matches_step(
            small_config_mimo(),
            Mamba3TrapSsdPath::Minimal(Some(4)),
            false,
        );
    }

    #[test]
    fn forward2_matches_step_mimo_random_init() {
        run_forward2_matches_step(
            small_config_mimo(),
            Mamba3TrapSsdPath::Minimal(Some(4)),
            true,
        );
    }

    #[test]
    fn forward2_matches_step_serial() {
        run_forward2_matches_step(small_config(), Mamba3TrapSsdPath::Serial(Some(4)), false);
    }

    #[test]
    fn forward2_matches_step_serial_mimo() {
        run_forward2_matches_step(
            small_config_mimo(),
            Mamba3TrapSsdPath::Serial(Some(4)),
            false,
        );
    }

    #[test]
    fn forward2_matches_step_recalc() {
        run_forward2_matches_step(
            small_config(),
            Mamba3TrapSsdPath::SerialRecalculated(Some(4)),
            false,
        );
    }

    #[test]
    fn forward2_matches_step_recalc_mimo() {
        run_forward2_matches_step(
            small_config_mimo(),
            Mamba3TrapSsdPath::SerialRecalculated(Some(4)),
            false,
        );
    }

    /// forward2 continuation from a **random** initial merged cache:
    /// `forward2(full, cache) ≡ forward2(prefix, cache)` then
    /// `forward2(suffix, mid_cache)`. Compares outputs, the final merged cache,
    /// and gradients. This replaces the old zero-init split-vs-full test: a
    /// random initial cache subsumes the chunked-prefill continuity guarantee
    /// from an arbitrary starting state, and the guard at the end confirms the
    /// initial cache is actually consumed (not silently ignored).
    fn run_forward2_split_matches_full(cfg: Mamba3Config, trap_path: Mamba3TrapSsdPath) {
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

        let init_cache = build_merged_cache(&cfg, batch, true);

        let input_full = param_input(&input);
        let cache_full = init_cache.clone();
        let trap_f = trap_path.clone();
        let r_full = run_with_grads_merged(&model, &input_full, &heads, |m, x| {
            m.forward2(x, Some(cache_full), trap_f)
        });

        let input_split = param_input(&input);
        let cache_split = init_cache;
        let trap_s = trap_path.clone();
        let r_split = run_with_grads_merged(&model, &input_split, &heads, |m, x| {
            let prefix = x.clone().narrow(1, 0, split);
            let suffix = x.narrow(1, split, seq_len - split);
            let (out_prefix, mid) = m.forward2(prefix, Some(cache_split), trap_s.clone());
            let (out_suffix, last) = m.forward2(suffix, Some(mid), trap_s);
            (Tensor::cat(vec![out_prefix, out_suffix], 1), last)
        });

        check_merged_match("forward2 split vs full", &r_full, &r_split, 1e-4, 1e-3);

        // Guard: the random initial merged cache must change the full output.
        {
            use crate::utils::test_helpers::max_abs_diff;
            let (out_zero, _) = model.forward2(
                Tensor::from_inner(input.clone()),
                Some(build_merged_cache(&cfg, batch, false)),
                trap_path.clone(),
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
    fn forward2_split_matches_full() {
        run_forward2_split_matches_full(small_config(), Mamba3TrapSsdPath::Minimal(Some(4)));
    }

    #[test]
    fn forward2_split_matches_full_mimo() {
        run_forward2_split_matches_full(small_config_mimo(), Mamba3TrapSsdPath::Minimal(Some(4)));
    }

    #[test]
    fn forward2_split_matches_full_serial() {
        run_forward2_split_matches_full(small_config(), Mamba3TrapSsdPath::Serial(Some(4)));
    }

    #[test]
    fn forward2_split_matches_full_serial_mimo() {
        run_forward2_split_matches_full(small_config_mimo(), Mamba3TrapSsdPath::Serial(Some(4)));
    }

    #[test]
    fn forward2_split_matches_full_recalc() {
        run_forward2_split_matches_full(
            small_config(),
            Mamba3TrapSsdPath::SerialRecalculated(Some(4)),
        );
    }

    #[test]
    fn forward2_split_matches_full_recalc_mimo() {
        run_forward2_split_matches_full(
            small_config_mimo(),
            Mamba3TrapSsdPath::SerialRecalculated(Some(4)),
        );
    }
}
