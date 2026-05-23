use crate::utils::sanity::sanity as san;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// segsum  (stable segment sum for the 1-SS mask)
// ---------------------------------------------------------------------------

/// Compute stable segment sums for constructing the 1-semiseparable mask.
///
/// Given a tensor `x` of shape `[..., sequence]`, returns a tensor of shape `[..., sequence, sequence]` where:
///
/// ```text
///   out[..., i, j] = Σ_{k=j+1}^{i} x[..., k]   for i ≥ j  (lower triangle)
///   out[..., i, j] = -∞                        for i < j  (upper triangle)
/// ```
///
/// ## Implementation
///
/// A naive computation of all pairwise products `A[j+1]·...·A[i]` would
/// suffer from underflow for long sequences (e.g. `0.9^1000 ≈ 2.6×10⁻⁴⁶`).
/// Working in log-space and computing differences of prefix sums avoids this:
///
/// ```text
///   segsum(x)[i, j] = cumsum(x)[i] - cumsum(x)[j]
/// ```
///
/// The upper triangle is masked to -∞ so that `exp(segsum(...))` gives 0
/// for non-causal positions (the strict upper triangle of L must be zero).
pub fn segsum<B: Backend, const D: usize, const D2: usize>(x: Tensor<B, D>) -> Tensor<B, D2> {
    assert_eq!(D + 1, D2);

    let x_cumsum = x.cumsum(D - 1);
    san(&x_cumsum);
    let x_cumsum_row = x_cumsum.clone().unsqueeze_dim(D); // [..., sequence, 1]
    let x_cumsum_col = x_cumsum.unsqueeze_dim(D - 1); //     [..., 1, sequence]

    let diff = x_cumsum_row - x_cumsum_col; // [..., sequence, sequence]
    san(&diff);
    let neg_inf_mask = Tensor::full_like(&diff, f32::NEG_INFINITY).triu(1);
    diff + neg_inf_mask
}
