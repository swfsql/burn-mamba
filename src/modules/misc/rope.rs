use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::{RotationState, rotate_bc_forward, rotate_bc_step};
use crate::modules::Silu;
use crate::modules::sanity as san;
use burn::prelude::*;

// TODO: move to mamba3/rotation/mod.rs

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
pub fn wrap_angle<const D: usize>(angles: Tensor<D>) -> Tensor<D> {
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
pub fn apply_rope_partial<const D: usize>(
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
