//! The RoPE trick for the abelian rotation kind
//! ([`Complex2D`](crate::mamba3::rotation::RotationKind::Complex2D)).
//!
//! Purely mechanical: given per-pair cumulative angles, rotate the paired
//! channels of B/C. See the [module header](crate::mamba3::rotation) for why
//! these angles are a *state transition*, not a positional encoding.

use burn::prelude::*;

// ---------------------------------------------------------------------------
// RoPE trick
// ---------------------------------------------------------------------------

/// Reduce angles modulo `2π` into `[−π, π]`, leaving the autodiff graph intact.
///
/// `sin`/`cos` are `2π`-periodic, so a subtraction of an integer multiple of
/// `2π` does not change the rotation. In f16 and bf16, the dtype cannot hold
/// `2π`, so the subtraction is in two parts (`helpers::sub_periods`).
/// Otherwise, each wrap of the decode cache would lose `1.9·10⁻³` per turn.
/// `|angle| ≤ π` keeps precision in low-bit floats: about
/// half of the representable `f16` values are in `|x| ≤ 1`, and the periodic
/// `sin`/`cos` lose accuracy only when the argument drifts to large
/// magnitudes. The same applies to the cumulative angle accumulator. Without
/// this, it would grow without bound over a long sequence or many decode
/// steps.
///
/// The wrap does not remove the rounding of the angle itself. In f16, the
/// spacing at `|angle| ≈ π` is `2⁻⁹ ≈ 2·10⁻³`, and each decode step rounds the
/// sum again. So a long f16 decode drifts: in a numpy simulation, 4096
/// micro-steps of 1–2.5 rad drift by 0.18–0.82 rad. Only an f32 angle scan and an f32
/// angle cache remove this drift.
///
/// The integer multiple `k` is `detach`ed, so autodiff sees it as a constant:
/// `d/dx (x − k·2π) = 1`, and the backward pass is the same as for the
/// unwrapped angle.
pub fn wrap_angle<const D: usize>(angles: Tensor<D>) -> Tensor<D> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let k = (angles.clone().detach() * (1.0f32 / two_pi)).round();
    crate::mamba3::helpers::sub_periods(angles, k, two_pi)
}

/// Rotate the pairs of `x` along its last dimension (the RoPE trick).
///
/// Purely the mechanical pairwise rotation. In Mamba-3 the `angles` are the
/// cumulative rotation of the *state transition* (see [`crate::mamba3::mamba3`]),
/// not a position index.
///
/// The matrix of a pair is `R(+angle)`: `x₀' = cos·x₀ − sin·x₁` and
/// `x₁' = sin·x₀ + cos·x₁`. Section 2 of that header compares this direction with
/// the sources.
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

/// Apply the RoPE trick to only the rotation-active entries of the last
/// dimension. The rest is unchanged. When `rotation_dim == state_rank` (a full
/// rotation), this is [`apply_rope`]. `rotation_dim` must be positive: a block
/// that rotates nothing is
/// [`RotationKind::Real1D`](crate::mamba3::rotation::RotationKind::Real1D),
/// which never comes here.
///
/// Pairing scheme (it must match the reference kernels: see Section
/// "Data-Dependent RoPE" in the paper, and `mamba3_siso_fwd.py` /
/// `mamba3_mimo_fwd.py`):
///
/// - `rotate_pairwise = true` (SISO, interleaved/NeoX): pairs `(0,1), (2,3), …`.
///   Only pairs `0..num_rotation_angles` turn, and the pairs after them are
///   unchanged. This is the same as a rotation of the first `rotation_dim`
///   entries.
/// - `rotate_pairwise = false` (MIMO, half-and-half/GPT-J): the pair distance
///   is always `state_rank/2`: element `n` pairs with element
///   `state_rank/2 + n`. With a partial rotation, only the first
///   `num_rotation_angles` pairs turn. The other elements in both halves are
///   unchanged.
pub fn apply_rope_partial<const D: usize>(
    x: Tensor<D>,
    angles: Tensor<D>,
    rotation_dim: usize,
    rotate_pairwise: bool,
) -> Tensor<D> {
    debug_assert!(rotation_dim > 0, "a partial rotation requires rotation_dim > 0");

    let state_rank = x.dims()[D - 1];
    if rotation_dim == state_rank {
        return apply_rope::<D>(x, angles, rotate_pairwise);
    }

    if rotate_pairwise {
        // Pairs are local. A slice of the first rotation_dim entries gives the
        // same result as the reference (which rotates the whole headdim, with
        // identity cos/sin for the tail pairs).
        let x_rope = x.clone().narrow(D - 1, 0, rotation_dim);
        let x_rest = x.narrow(D - 1, rotation_dim, state_rank - rotation_dim);
        let x_rope_rotated = apply_rope::<D>(x_rope, angles, true);
        return Tensor::cat(vec![x_rope_rotated, x_rest], D - 1);
    }

    // Half-and-half partial rotation: the pair distance must be `state_rank/2`,
    // not `rotation_dim/2`. `apply_rope` on a slice of the first `rotation_dim`
    // entries would pair within the slice and give the wrong rotation.
    let half = state_rank / 2;
    let num_rotation_angles = rotation_dim / 2;
    debug_assert!(
        num_rotation_angles < half,
        "a partial rotation requires rotation_dim < state_rank here"
    );

    // Split x into the two halves, then within each half separate the
    // rotation-active prefix from the pass-through suffix.
    let x_h1 = x.clone().narrow(D - 1, 0, half);
    let x_h2 = x.narrow(D - 1, half, half);
    let x_h1_rope = x_h1.clone().narrow(D - 1, 0, num_rotation_angles);
    let x_h1_pass = x_h1.narrow(D - 1, num_rotation_angles, half - num_rotation_angles);
    let x_h2_rope = x_h2.clone().narrow(D - 1, 0, num_rotation_angles);
    let x_h2_pass = x_h2.narrow(D - 1, num_rotation_angles, half - num_rotation_angles);

    // angles: [..., num_rotation_angles] — broadcasts element-wise against the
    // rotation-active slices.
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
