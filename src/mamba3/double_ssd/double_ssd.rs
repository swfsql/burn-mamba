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
use crate::utils::sanity::sanity as san;
use crate::utils::silu::Silu;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// RoPE utility
// ---------------------------------------------------------------------------

/// Apply rotary position embeddings to `x` along its last dimension.
///
/// Two pairing conventions are supported, selected by `rotate_pairwise`:
///
/// - `rotate_pairwise = true` — **interleaved** (NeoX / Triton style): adjacent
///   pairs `(0,1)`, `(2,3)`, … are rotated together. Used by the SISO Triton
///   kernel (`mamba3_siso_*.py`).
/// - `rotate_pairwise = false` — **half-and-half** (GPT-J style): position `n`
///   is paired with `n + state_rank/2`. Used by the MIMO Tilelang kernel
///   (`mamba3_mimo_fwd.py`).
///
/// Reference: `mamba3.py:335` sets `rotate_pairwise = not self.is_mimo`.
///
/// # Shapes
/// - `x`:      `[..., state_rank]` where `state_rank` is even
/// - `angles`: `[..., state_rank / 2]`  (one angle per pair)
/// - output:   same shape as `x`
pub fn apply_rope<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    angles: Tensor<B, D>,
    rotate_pairwise: bool,
) -> Tensor<B, D> {
    let dims = x.dims();
    let n = dims[D - 1];
    let n2 = n / 2;
    let leading: usize = dims[..D - 1].iter().product();

    let angles_flat = angles.reshape([leading, n2]);
    let cos = angles_flat.clone().cos();
    let sin = angles_flat.sin();

    if rotate_pairwise {
        // Interleaved: reshape to [leading, n2, 2], pairs along last axis.
        let x_pairs = x.reshape([leading, n2, 2]);
        let x0 = x_pairs.clone().narrow(2, 0, 1).squeeze_dim(2);
        let x1 = x_pairs.narrow(2, 1, 1).squeeze_dim(2);

        let x0r = cos.clone() * x0.clone() - sin.clone() * x1.clone();
        let x1r = sin * x0 + cos * x1;

        Tensor::cat(
            vec![x0r.unsqueeze_dim::<3>(2), x1r.unsqueeze_dim::<3>(2)],
            2,
        )
        .reshape(dims)
    } else {
        // Half-and-half: reshape to [leading, 2, n2], halves along middle axis.
        let x_halves = x.reshape([leading, 2, n2]);
        let x0 = x_halves.clone().narrow(1, 0, 1).squeeze_dim(1);
        let x1 = x_halves.narrow(1, 1, 1).squeeze_dim(1);

        let x0r = cos.clone() * x0.clone() - sin.clone() * x1.clone();
        let x1r = sin * x0 + cos * x1;

        Tensor::cat(
            vec![x0r.unsqueeze_dim::<3>(1), x1r.unsqueeze_dim::<3>(1)],
            1,
        )
        .reshape(dims)
    }
}

/// Apply RoPE to only the rotation-active entries of the last dimension; the
/// remainder passes through unchanged. Falls back to [`apply_rope`] when
/// `rope_dim == state_rank` (full RoPE).
///
/// Pairing scheme (must match the reference kernels — see Section
/// "Data-Dependent RoPE" in the paper, and `mamba3_siso_fwd.py` /
/// `mamba3_mimo_fwd.py`):
///
/// - `rotate_pairwise = true` (SISO, interleaved/NeoX): pairs `(0,1), (2,3), …`.
///   Only pairs `0..num_rope_angles` are rotated; pairs beyond are passed
///   through. Equivalent to slicing the first `rope_dim` entries and rotating
///   them.
/// - `rotate_pairwise = false` (MIMO, half-and-half/GPT-J): pair distance is
///   always `state_rank/2`, i.e. element `n` is paired with element
///   `state_rank/2 + n`. With partial RoPE only the first `num_rope_angles`
///   pairs are rotated; the remaining elements in both halves pass through.
pub(crate) fn apply_rope_partial<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    angles: Tensor<B, D>,
    rope_dim: usize,
    rotate_pairwise: bool,
) -> Tensor<B, D> {
    let state_rank = x.dims()[D - 1];
    if rope_dim == state_rank {
        return apply_rope::<B, D>(x, angles, rotate_pairwise);
    }

    if rotate_pairwise {
        // Pairs are local — slicing the first rope_dim entries gives the same
        // result as the reference (which rotates the whole headdim but with
        // identity cos/sin for the tail pairs).
        let x_rope = x.clone().narrow(D - 1, 0, rope_dim);
        let x_rest = x.narrow(D - 1, rope_dim, state_rank - rope_dim);
        let x_rope_rotated = apply_rope::<B, D>(x_rope, angles, true);
        return Tensor::cat(vec![x_rope_rotated, x_rest], D - 1);
    }

    // Half-and-half partial RoPE: pair distance must be `state_rank/2`, not
    // `rope_dim/2`. Slicing the first `rope_dim` entries and calling
    // `apply_rope` would pair within the slice and produce the wrong rotation.
    let half = state_rank / 2;
    let num_rope_angles = rope_dim / 2;
    debug_assert!(
        num_rope_angles < half,
        "partial RoPE requires rope_dim < state_rank here"
    );

    // Split x into the two halves, then within each half separate the
    // rotation-active prefix from the pass-through suffix.
    let x_h1 = x.clone().narrow(D - 1, 0, half);
    let x_h2 = x.narrow(D - 1, half, half);
    let x_h1_rope = x_h1.clone().narrow(D - 1, 0, num_rope_angles);
    let x_h1_pass = x_h1.narrow(D - 1, num_rope_angles, half - num_rope_angles);
    let x_h2_rope = x_h2.clone().narrow(D - 1, 0, num_rope_angles);
    let x_h2_pass = x_h2.narrow(D - 1, num_rope_angles, half - num_rope_angles);

    // angles: [..., num_rope_angles] — broadcasts element-wise against the rope-active slices.
    let cos = angles.clone().cos();
    let sin = angles.sin();
    let x_h1_rot = cos.clone() * x_h1_rope.clone() - sin.clone() * x_h2_rope.clone();
    let x_h2_rot = sin * x_h1_rope + cos * x_h2_rope;

    // Reassemble: [ first-half-rotated | first-half-passthrough | second-half-rotated | second-half-passthrough ]
    let x_h1_out = Tensor::cat(vec![x_h1_rot, x_h1_pass], D - 1);
    let x_h2_out = Tensor::cat(vec![x_h2_rot, x_h2_pass], D - 1);
    Tensor::cat(vec![x_h1_out, x_h2_out], D - 1)
}

// ---------------------------------------------------------------------------
// Mamba3::forward  (chunkwise double-SSD — training / prefill)
// ---------------------------------------------------------------------------

impl<B: Backend + Mamba3BackendExt> Mamba3<B> {
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
        input_bsm: Tensor<B, 3>,
        cache: Option<Mamba3DoubleSsdCache<B>>,
        ssd_path: Mamba3DoubleSsdPath,
    ) -> (Tensor<B, 3>, Mamba3DoubleSsdCache<B>) {
        println!("double-ssd forward");
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
            Mamba3DoubleSsdCache {
                ssm_bhpr,
                k_state_bmhr,
                v_state_bhp,
                cum_angle_bha,
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
        assert_eq!(
            [batch, sequence, mimo_rank, nheads, state_rank],
            b_bsmhr.dims()
        );
        assert_eq!(
            [batch, sequence, mimo_rank, nheads, state_rank],
            c_bsmhr.dims()
        );

        // ── Step 5: Data-dependent cumulative RoPE angles ─────────────────────
        let theta_scaled_bsa = theta_bsa.tanh() * std::f32::consts::PI;
        let raw_angles_bsha =
            dt_bsh.clone().unsqueeze_dim::<4>(3) // dt_bsh1
            *
            theta_scaled_bsa.unsqueeze_dim::<4>(2) // theta_scaled_bs1a
            ;

        let cumsum_bsha = raw_angles_bsha.cumsum(1);
        let cum_angles_bsha = cache.cum_angle_bha.clone()
            .unsqueeze_dim::<4>(1) // cum_angle_b1ha
            + cumsum_bsha;
        assert_eq!(
            [batch, sequence, nheads, num_rope_angles],
            cum_angles_bsha.dims()
        );
        san(&cum_angles_bsha);

        // Apply RoPE to B and C: angles broadcast over the mimo_rank dim.
        let cum_angles_bsmha = cum_angles_bsha
            .clone()
            .unsqueeze_dim::<5>(2) // cum_angles_bs1ha
            .expand([batch, sequence, mimo_rank, nheads, num_rope_angles]); // cum_angles_bsmha
        // SISO uses interleaved (pairwise) pairing; MIMO uses half-and-half.
        // Partial RoPE: rotate only the first `rope_dim` entries of B/C.
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
        let gamma_bsh1 = gamma_bsh.unsqueeze_dim::<4>(3);
        let beta_bsh1 = beta_bsh.unsqueeze_dim::<4>(3);
        let x_gamma_bshp = x_bshp.clone() * gamma_bsh1; // γₜ · xₜ
        let x_beta_bshp = x_prev_bshp * beta_bsh1; // βₜ · xₜ₋₁

        // ── Save last-token B for cache ───────────────────────────────────────
        let b_last_bmhr = b_bsmhr
            .clone()
            .narrow(1, sequence - 1, 1)
            .reshape([batch, mimo_rank, nheads, state_rank]);

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
            helpers::build_v_with_mimo::<_, 5, 6>(x_gamma_bnlhp.clone(), mimo_x_hmp.as_ref(), 3);
        let v_beta_bnlmhp =
            helpers::build_v_with_mimo::<_, 5, 6>(x_beta_bnlhp, mimo_x_hmp.as_ref(), 3);

        let input_gamma = Mamba3DoubleSsdInput {
            v_bnlmhp: v_gamma_bnlmhp,
            da_bnlh: da_bnlh.clone(),
            b_bnlmhr: b_bnlmhr.clone(),
            c_bnlmhr: c_bnlmhr.clone(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        };
        let (y_gamma_bnlmhp, final_state_gamma_bhpr) = ssd_path.clone().run(input_gamma);

        let input_beta = Mamba3DoubleSsdInput {
            v_bnlmhp: v_beta_bnlmhp,
            da_bnlh,
            b_bnlmhr: b_prev_bnlmhr,
            c_bnlmhr,
            initial_state_bhpr: Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device),
            init_state_hpr: None,
        };
        let (y_beta_bnlmhp, final_state_beta_bhpr) = ssd_path.run(input_beta);

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
            helpers::build_v_with_mimo::<_, 4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);

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
            let y_bshp: Tensor<B, 4> = (y_combined_bsmhp * mimo_o_bsmhp)
                .sum_dim(2) // y_bs1hp
                .squeeze_dim(2); // y_bshp
            y_bshp.reshape([batch, sequence, d_inner])
        } else {
            // SISO: squeeze rank dim, apply gate (or gated norm) over per_head_dim.
            let y_bshp: Tensor<B, 4> = y_bsmhp.squeeze_dim(2); // mimo_rank == 1
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

        // cum_angle at last token
        cache.cum_angle_bha = cum_angles_bsha
            .narrow(1, sequence - 1, 1) // cum_angles_b1ha
            .squeeze_dim::<3>(1); // cum_angles_bha

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
        pub fn step_double_ssd(
            &self,
            input_bd: Tensor<B, 2>,
            cache: Option<Mamba3DoubleSsdCache<B>>,
        ) -> (Tensor<B, 2>, Mamba3DoubleSsdCache<B>) {
            let [batch, d_model] = input_bd.dims();
            let d_inner = self.d_inner();
            let nheads = self.nheads();
            let ngroups = self.ngroups;
            let per_head_dim = self.per_head_dim();
            let state_rank = self.state_rank;
            let num_rope_angles = self.num_rope_angles;
            let mimo_rank = self.mimo_rank;
            let device = &input_bd.device();
            let ssm_shape = [batch, nheads, per_head_dim, state_rank];

            assert_eq!(nheads % ngroups, 0);

            let mut cache = cache.unwrap_or_else(|| {
                let ssm_bhpr = Tensor::zeros(ssm_shape, device);
                let k_state_bmhr = Tensor::zeros([batch, mimo_rank, nheads, state_rank], device);
                let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], device);
                let cum_angle_bha = Tensor::zeros([batch, nheads, num_rope_angles], device);
                Mamba3DoubleSsdCache {
                    ssm_bhpr,
                    k_state_bmhr,
                    v_state_bhp,
                    cum_angle_bha,
                }
            });

            // ── In-projection ─────────────────────────────────────────────────
            let proj_bd = self.in_proj.forward(input_bd);
            let bc_size = ngroups * state_rank * mimo_rank;
            // [batch, *] split along channel dim.
            // b_raw_bMGR / c_raw_bMGR have channel size `mimo_rank * ngroups * state_rank`.
            #[rustfmt::skip]
            let [
                    z_bi, x_bi,
                    b_raw_bMGR, c_raw_bMGR,
                    dd_dt_bh, dd_a_raw_bh, lambda_raw_bh,
                    theta_ba,
            ] = crate::utils::split::split_into(
                proj_bd,
                [
                    d_inner, d_inner,
                    bc_size, bc_size,
                    nheads, nheads, nheads,
                    num_rope_angles,
                ],
                1,
            );

            // ── Reshape x ─────────────────────────────────────────────────────
            let x_bhp = x_bi.reshape([batch, nheads, per_head_dim]);

            // ── Discretisation + trapezoidal coefficients ─────────────────────
            let helpers::TrapezoidCoeffs {
                dt: dt_bh,
                da: _da_bh,
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

            // ── QK-Norm on B and C ────────────────────────────────────────────
            // Group dim is axis 2 of `_bmgr` (D = 4).
            let b_bmhr = helpers::qk_norm_expand_bias::<_, 4, 5>(
                b_raw_bMGR.reshape([batch, mimo_rank, ngroups, state_rank]),
                &self.b_norm,
                self.b_bias_hmr.val(),
                2,
                nheads,
            );
            let c_bmhr = helpers::qk_norm_expand_bias::<_, 4, 5>(
                c_raw_bMGR.reshape([batch, mimo_rank, ngroups, state_rank]),
                &self.c_norm,
                self.c_bias_hmr.val(),
                2,
                nheads,
            );
            assert_eq!([batch, mimo_rank, nheads, state_rank], b_bmhr.dims());

            // ── RoPE: update cumulative angle, rotate B and C ──────────────────
            let theta_scaled_ba = theta_ba.tanh() * std::f32::consts::PI;
            let raw_angle_bha = dt_bh.unsqueeze_dim::<3>(2) * theta_scaled_ba.unsqueeze_dim::<3>(1);
            let new_cum_angle_bha = cache.cum_angle_bha.clone() + raw_angle_bha;

            // Broadcast angles over mimo_rank
            let new_cum_angle_bmha = new_cum_angle_bha
                .clone()
                .unsqueeze_dim::<4>(1) // new_cum_angle_b1ha
                .expand([batch, mimo_rank, nheads, num_rope_angles]); // new_cum_angle_bmha
            // SISO uses interleaved (pairwise) pairing; MIMO uses half-and-half.
            // Partial RoPE: rotate only the first `rope_dim` entries of B/C.
            let rotate_pairwise = mimo_rank == 1;
            let rope_dim = self.rope_dim;
            let b_bmhr = apply_rope_partial::<B, 4>(
                b_bmhr,
                new_cum_angle_bmha.clone(),
                rope_dim,
                rotate_pairwise,
            );
            let c_bmhr =
                apply_rope_partial::<B, 4>(c_bmhr, new_cum_angle_bmha, rope_dim, rotate_pairwise);

            // ── Build MIMO value tensors ───────────────────────────────────────
            // Insert the mimo_rank axis at position 1 of `_bhp`.
            let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
            let x_vals_bmhp =
                helpers::build_v_with_mimo::<_, 3, 4>(x_bhp.clone(), mimo_x_hmp.as_ref(), 1);
            let xs_vals_bmhp = helpers::build_v_with_mimo::<_, 3, 4>(
                cache.v_state_bhp.clone(),
                mimo_x_hmp.as_ref(),
                1,
            );

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
            let x_beta_bmhp = xs_vals_bmhp * beta_b1h1;

            // einsum('bmhp,bmhr->bhpr', x_gamma, B_cur):
            let xbt_state_bhpr = {
                let b_bhmr = b_bmhr.clone().permute([0, 2, 1, 3]);
                let xg_bhpm = x_gamma_bmhp.permute([0, 2, 3, 1]);
                xg_bhpm.matmul(b_bhmr)
            };
            let xbt_prev_bhpr = {
                let b_state_bhmr = cache.k_state_bmhr.clone().permute([0, 2, 1, 3]);
                let xb_bhpm = x_beta_bmhp.permute([0, 2, 3, 1]);
                xb_bhpm.matmul(b_state_bhmr)
            };

            let alpha_bh11 = alpha_bh.unsqueeze_dims::<4>(&[2, 3]);
            let new_state_bhpr =
                alpha_bh11 * cache.ssm_bhpr.clone() + xbt_state_bhpr + xbt_prev_bhpr;

            // ── Output ────────────────────────────────────────────────────────
            // outₘ[b, m, h, p] = sumᵣ C[b, m, h, r] * state[b, h, p, r] + D * x_vals[b, m, h, p]
            // = einsum('bhpr,bmhr->bmhp', state, C)
            let out_m_bmhp = {
                let c_bhrm = c_bmhr.permute([0, 2, 3, 1]);
                let out_bhpm = new_state_bhpr.clone().matmul(c_bhrm);
                out_bhpm.permute([0, 3, 1, 2])
            };

            // D skip
            let d_bmhp = self
                .d_h
                .val()
                .unsqueeze_dims::<4>(&[0, 1, 3]) // d_11h1
                .expand([batch, mimo_rank, nheads, per_head_dim]); // d_bmhp
            let out_m_bmhp = out_m_bmhp + d_bmhp * x_vals_bmhp;

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

                // Per-rank gate or gated norm.
                let combined_bmhp = match &self.out_norm {
                    Some(norm) => norm.forward(out_m_bmhp, z_bmhp),
                    None => out_m_bmhp * Silu::new().forward(z_bmhp),
                };

                // Project down: out = sumₘ mimo_o_hmp[m] * combined_bmhp[m]
                let mimo_o_bmhp = mimo_o_hmp
                    .permute([1, 0, 2]) // mimo_o_mhp
                    .unsqueeze_dim::<4>(0) // mimo_o_1mhp
                    .expand([batch, mimo_rank, nheads, per_head_dim]); // mimo_o_bmhp
                let out_bhp: Tensor<B, 3> = (combined_bmhp * mimo_o_bmhp)
                    .sum_dim(1) // out_b1hp
                    .squeeze_dim(1); // out_bhp
                out_bhp.reshape([batch, d_inner]) // y_bi
            } else {
                // SISO: squeeze rank dim, gate (or gated norm) over per_head_dim.
                let y_bhp: Tensor<B, 3> = out_m_bmhp.squeeze_dim(1);
                let combined = match &self.out_norm {
                    Some(norm) => norm.forward(y_bhp, z_bhp),
                    None => y_bhp * Silu::new().forward(z_bhp),
                };
                combined.reshape([batch, d_inner])
            };

            // ── Out-projection ────────────────────────────────────────────────
            let out_bm = self.out_proj.forward(y_bi);
            assert_eq!([batch, d_model], out_bm.dims());

            // ── Update cache ──────────────────────────────────────────────────
            cache.ssm_bhpr = new_state_bhpr;
            cache.k_state_bmhr = b_bmhr;
            cache.v_state_bhp = x_bhp;
            cache.cum_angle_bha = new_cum_angle_bha;

            (out_bm, cache)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Flex};
    use burn::module::Param;
    use burn::tensor::Distribution;

    /// Inner (non-autodiff) backend used for materialising values and
    /// extracted gradients.
    type InnerB = Flex;
    /// Autodiff-wrapped backend used to drive `.backward()`.
    type B = Autodiff<InnerB>;

    type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

    fn small_config() -> Mamba3Config {
        Mamba3Config::new(32) // d_model = 32
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

    /// A bundle of input + model-parameter gradients extracted from one
    /// forward+backward run.  Each `check_grads_match` call compares these
    /// across two runs that should be mathematically equivalent.
    struct RunGrads {
        out: Tensor<InnerB, 3>,
        /// Final SSM hidden state from the returned cache.
        final_ssm: Tensor<InnerB, 4>,
        /// Final previous-token B state from the returned cache.
        final_k: Tensor<InnerB, 4>,
        /// Final previous-token x state from the returned cache.
        final_v: Tensor<InnerB, 3>,
        /// Final cumulative RoPE angle from the returned cache.
        final_angle: Tensor<InnerB, 3>,
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

    /// Fixed (non-tracked) random "downstream heads" used to form a scalar loss
    /// from the output **and** every final cache field, so the backward pass
    /// exercises both the output and the state path.
    struct Heads {
        out: Tensor<InnerB, 3>,
        ssm: Tensor<InnerB, 4>,
        k: Tensor<InnerB, 4>,
        v: Tensor<InnerB, 3>,
        angle: Tensor<InnerB, 3>,
    }

    /// Build the initial cache passed to both `forward` and the `step`
    /// unrolling. With `random = false` it is zero (the standard fresh start);
    /// with `random = true` every field (SSM state, previous-token B/x, and
    /// cumulative RoPE angle) holds random values, exercising parity from an
    /// arbitrary initial state.
    fn build_init_cache(cfg: &Mamba3Config, batch: usize, random: bool) -> Mamba3DoubleSsdCache<B> {
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
        Mamba3DoubleSsdCache {
            ssm_bhpr: mk4([batch, nheads, per_head_dim, state_rank]),
            k_state_bmhr: mk4([batch, mimo_rank, nheads, state_rank]),
            v_state_bhp: mk3([batch, nheads, per_head_dim]),
            cum_angle_bha: mk3([batch, nheads, num_rope_angles]),
        }
    }

    /// Compare the output and every final cache field of two runs.
    fn assert_outputs_match(label: &str, a: &RunGrads, b: &RunGrads, tol: f32) {
        use crate::utils::test_helpers::max_abs_diff;
        let checks = [
            ("output", max_abs_diff(a.out.clone(), b.out.clone())),
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
        for (name, d) in checks {
            assert!(d < tol, "{label}: {name} max abs diff = {d:.6} (tol {tol})");
        }
    }

    /// Run a closure that produces an output tensor from a model and an input
    /// (wrapped as a `Param` so it has its own autodiff leaf), then derive a
    /// scalar loss with a fixed (non-tracked) random "head" and return the
    /// gradients of the input and a representative set of model parameters.
    fn run_with_grads(
        model: &Mamba3<B>,
        input: &Param<Tensor<B, 3>>,
        heads: &Heads,
        forward: impl FnOnce(&Mamba3<B>, Tensor<B, 3>) -> (Tensor<B, 3>, Mamba3DoubleSsdCache<B>),
    ) -> RunGrads {
        let (out, cache) = forward(model, input.val());
        let out_inner = out.clone().inner();
        let ssm = cache.ssm_bhpr;
        let k = cache.k_state_bmhr;
        let v = cache.v_state_bhp;
        let angle = cache.cum_angle_bha;
        let final_ssm = ssm.clone().inner();
        let final_k = k.clone().inner();
        let final_v = v.clone().inner();
        let final_angle = angle.clone().inner();

        // Loss couples the output and every final cache field (each via its own
        // random head) so parameter gradients reflect both output and state.
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

        RunGrads {
            out: out_inner,
            final_ssm,
            final_k,
            final_v,
            final_angle,
            d_input: input.val().grad(&grads).expect("grad input"),
            d_in_proj_w: model
                .in_proj
                .weight
                .val()
                .grad(&grads)
                .expect("grad in_proj.weight"),
            d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("grad dt_bias_h"),
            d_d: model.d_h.val().grad(&grads).expect("grad d_h"),
            d_b_norm_gamma: model
                .b_norm
                .gamma
                .val()
                .grad(&grads)
                .expect("grad b_norm.gamma"),
            d_c_norm_gamma: model
                .c_norm
                .gamma
                .val()
                .grad(&grads)
                .expect("grad c_norm.gamma"),
            d_b_bias: model
                .b_bias_hmr
                .val()
                .grad(&grads)
                .expect("grad b_bias_hmr"),
            d_c_bias: model
                .c_bias_hmr
                .val()
                .grad(&grads)
                .expect("grad c_bias_hmr"),
            d_out_proj_w: model
                .out_proj
                .weight
                .val()
                .grad(&grads)
                .expect("grad out_proj.weight"),
        }
    }

    /// Assert that every entry in `a` and `b` agrees to within `grad_tol`,
    /// printing every comparison so a failure dump shows the full picture
    /// (instead of stopping at the first mismatch).
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

    /// Build a fresh `Param<Tensor>` from a stable inner tensor.
    /// A new Param is needed per run so that the autodiff leaf has a fresh
    /// node, isolating each backward pass to its own forward graph.
    fn param_input(input: &Tensor<InnerB, 3>) -> Param<Tensor<B, 3>> {
        Param::from_tensor(Tensor::from_inner(input.clone()))
    }

    /// `forward(x)` is mathematically equivalent to repeatedly calling `step`
    /// token-by-token from the **same** initial cache. Outputs, every final
    /// cache field (SSM state, previous-token B/x, cumulative RoPE angle), and
    /// parameter gradients must all agree up to float-summation-order noise.
    ///
    /// With `random_init = true` the shared initial cache is random rather than
    /// zero. Parity from an arbitrary initial state subsumes the chunked-prefill
    /// (split-vs-full) guarantee: if `forward` from any state matches the
    /// recurrent unrolling from that same state — outputs *and* final cache —
    /// then feeding a `forward`-produced cache back in continues correctly.
    fn run_step_matches_forward(cfg: Mamba3Config, random_init: bool) {
        let device: Device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 5;
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

        let ssd_path = Mamba3DoubleSsdPath::Minimal(Some(4));
        let init_cache = build_init_cache(&cfg, batch, random_init);

        let input_fwd = param_input(&input);
        let cache_fwd = init_cache.clone();
        let path_fwd = ssd_path.clone();
        let r_fwd = run_with_grads(&model, &input_fwd, &heads, |m, x| {
            m.forward_double_ssd(x, Some(cache_fwd), path_fwd)
        });

        let input_step = param_input(&input);
        let cache_step = init_cache;
        let r_step = run_with_grads(&model, &input_step, &heads, |m, x| {
            let mut cache: Option<Mamba3DoubleSsdCache<B>> = Some(cache_step);
            let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
                let (out_t, new_cache) = m.step_double_ssd(token, cache);
                cache = Some(new_cache);
                outs.push(out_t);
            }
            (Tensor::stack(outs, 1), cache.unwrap())
        });

        assert_outputs_match("step vs forward", &r_fwd, &r_step, 1e-4);
        check_grads_match("step vs forward", &r_fwd, &r_step, 1e-3);

        // ── Guard: the random initial state must actually be consumed ─────
        // Re-run forward from a *zero* initial cache; its output must differ
        // from the random-init output. Otherwise the initial state is being
        // silently ignored and forward/step would match trivially.
        if random_init {
            use crate::utils::test_helpers::max_abs_diff;
            let (out_zero, _) = model.forward_double_ssd(
                Tensor::from_inner(input.clone()),
                Some(build_init_cache(&cfg, batch, false)),
                ssd_path.clone(),
            );
            let d = max_abs_diff(r_fwd.out.clone(), out_zero.inner());
            assert!(
                d > 1e-3,
                "random initial state appears ignored: random-init vs zero-init \
                 output max abs diff = {d:.6} (expected a clear difference)"
            );
        }
    }

    // Config variants exercised by the parity tests below.
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
    fn cfg_rope_half() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_rope_fraction(0.5)
    }
    fn cfg_rope_half_mimo() -> Mamba3Config {
        cfg_rope_half().with_mimo_rank(2)
    }
    fn cfg_outproj_norm() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_has_outproj_norm(true)
    }
    fn cfg_outproj_norm_mimo() -> Mamba3Config {
        cfg_outproj_norm().with_mimo_rank(2)
    }
    fn cfg_rope_half_outproj_norm_mimo() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_mimo_rank(2)
            .with_rope_fraction(0.5)
            .with_has_outproj_norm(true)
    }

    #[test]
    fn step_matches_forward() {
        run_step_matches_forward(small_config(), false);
    }

    #[test]
    fn step_matches_forward_random_init() {
        run_step_matches_forward(small_config(), true);
    }

    #[test]
    fn step_matches_forward_ngroups2() {
        run_step_matches_forward(cfg_ngroups2(), false);
    }

    #[test]
    fn step_matches_forward_ngroups2_random_init() {
        run_step_matches_forward(cfg_ngroups2(), true);
    }

    #[test]
    fn step_matches_forward_mimo() {
        run_step_matches_forward(small_config_mimo(), false);
    }

    #[test]
    fn step_matches_forward_mimo_random_init() {
        run_step_matches_forward(small_config_mimo(), true);
    }

    #[test]
    fn step_matches_forward_mimo_ngroups2() {
        run_step_matches_forward(cfg_mimo_ngroups2(), false);
    }

    #[test]
    fn step_matches_forward_mimo_ngroups2_random_init() {
        run_step_matches_forward(cfg_mimo_ngroups2(), true);
    }

    // ── rope_fraction = 0.5 (partial RoPE) ──────────────────────────────────

    #[test]
    fn step_matches_forward_rope_half() {
        run_step_matches_forward(cfg_rope_half(), false);
    }

    #[test]
    fn step_matches_forward_rope_half_random_init() {
        run_step_matches_forward(cfg_rope_half(), true);
    }

    #[test]
    fn step_matches_forward_rope_half_mimo() {
        run_step_matches_forward(cfg_rope_half_mimo(), false);
    }

    #[test]
    fn step_matches_forward_rope_half_mimo_random_init() {
        run_step_matches_forward(cfg_rope_half_mimo(), true);
    }

    // ── has_outproj_norm = true (gated RMSNorm) ─────────────────────────────

    #[test]
    fn step_matches_forward_outproj_norm() {
        run_step_matches_forward(cfg_outproj_norm(), false);
    }

    #[test]
    fn step_matches_forward_outproj_norm_random_init() {
        run_step_matches_forward(cfg_outproj_norm(), true);
    }

    #[test]
    fn step_matches_forward_outproj_norm_mimo() {
        run_step_matches_forward(cfg_outproj_norm_mimo(), false);
    }

    #[test]
    fn step_matches_forward_outproj_norm_mimo_random_init() {
        run_step_matches_forward(cfg_outproj_norm_mimo(), true);
    }

    // ── Both features combined ──────────────────────────────────────────────

    #[test]
    fn step_matches_forward_rope_half_outproj_norm_mimo() {
        run_step_matches_forward(cfg_rope_half_outproj_norm_mimo(), false);
    }

    #[test]
    fn step_matches_forward_rope_half_outproj_norm_mimo_random_init() {
        run_step_matches_forward(cfg_rope_half_outproj_norm_mimo(), true);
    }
}
