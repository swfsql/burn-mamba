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
use crate::mamba3::rotation::{rotate_bc_forward, rotate_bc_step, RotationState};
use crate::utils::sanity::sanity as san;
use crate::utils::silu::Silu;
use burn::backend::Backend;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// RoPE utility
// ---------------------------------------------------------------------------

/// Reduce angles modulo `2π` into `[−π, π]`, leaving the autodiff graph intact.
///
/// `sin`/`cos` are `2π`-periodic, so subtracting an integer multiple of `2π` is
/// value-exact. Keeping `|angle| ≤ π` preserves precision in low-bit floats —
/// roughly half of `f16`'s representable values lie in `|x| ≤ 1`, and the
/// periodic `sin`/`cos` only lose accuracy when the argument is allowed to drift
/// to large magnitudes. The same applies to the cumulative angle accumulator,
/// which would otherwise grow without bound across a long sequence / many decode
/// steps.
///
/// The integer multiple `k` is `detach`ed, so it is a constant with respect to
/// autodiff: `d/dx (x − k·2π) = 1`, i.e. the backward pass is identical to the
/// un-wrapped angle. This mirrors the detached `max` rescaling in
/// [`RmsNormGated`](crate::utils::rms_norm_gated::RmsNormGated).
pub(crate) fn wrap_angle<const D: usize>(angles: Tensor<D>) -> Tensor<D> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let k = (angles.clone().detach() * (1.0f32 / two_pi)).round();
    angles - k * two_pi
}

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
pub fn apply_rope<const D: usize>(
    x: Tensor<D>,
    angles: Tensor<D>,
    rotate_pairwise: bool,
) -> Tensor<D> {
    let dims = x.dims();
    let n = dims[D - 1];
    let n2 = n / 2;
    let leading: usize = dims[..D - 1].iter().product();

    let angles_flat = wrap_angle(angles.reshape([leading, n2]));
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
/// `rope_dim == state_rank` (full RoPE), and is the **identity** when
/// `rope_dim == 0` (RoPE disabled, `rope_fraction = 0`) — `angles` is ignored.
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
pub(crate) fn apply_rope_partial<const D: usize>(
    x: Tensor<D>,
    angles: Tensor<D>,
    rope_dim: usize,
    rotate_pairwise: bool,
) -> Tensor<D> {
    if rope_dim == 0 {
        // RoPE disabled (rope_fraction = 0): identity. The upstream angle data
        // flow is still computed and cached, but no rotation is applied. This
        // also avoids zero-width narrows below (Burn has no zero-width tensors).
        return x;
    }

    let state_rank = x.dims()[D - 1];
    if rope_dim == state_rank {
        return apply_rope::<D>(x, angles, rotate_pairwise);
    }

    if rotate_pairwise {
        // Pairs are local — slicing the first rope_dim entries gives the same
        // result as the reference (which rotates the whole headdim but with
        // identity cos/sin for the tail pairs).
        let x_rope = x.clone().narrow(D - 1, 0, rope_dim);
        let x_rest = x.narrow(D - 1, rope_dim, state_rank - rope_dim);
        let x_rope_rotated = apply_rope::<D>(x_rope, angles, true);
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
    let angles = wrap_angle(angles);
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
        pub fn step_double_ssd(
            &self,
            input_bd: Tensor<2>,
            cache: Option<Mamba3DoubleSsdCache>,
        ) -> (Tensor<2>, Mamba3DoubleSsdCache) {
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
                let rotation = match self.rotation {
                    RotationKind::Quaternion4D => {
                        RotationState::identity_quaternion(batch, nheads, self.num_quat_blocks, device)
                    }
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
                    rot_ba,
            ] = crate::utils::split::split_into(
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

            // ── Build MIMO value tensors ───────────────────────────────────────
            // Insert the mimo_rank axis at position 1 of `_bhp`.
            let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
            let x_vals_bmhp =
                helpers::build_v_with_mimo::<3, 4>(x_bhp.clone(), mimo_x_hmp.as_ref(), 1);
            let xs_vals_bmhp = helpers::build_v_with_mimo::<3, 4>(
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
                let out_bhp: Tensor<3> = (combined_bmhp * mimo_o_bmhp)
                    .sum_dim(1) // out_b1hp
                    .squeeze_dim(1); // out_bhp
                out_bhp.reshape([batch, d_inner]) // y_bi
            } else {
                // SISO: squeeze rank dim, gate (or gated norm) over per_head_dim.
                let y_bhp: Tensor<3> = out_m_bmhp.squeeze_dim(1);
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
