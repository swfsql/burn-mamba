//! The RMSNorm-then-dot **score** — a parameter-free RMS normalisation folded
//! into a dot product against a learnable query.
//!
//! Both users score a set of parallel items and turn the scores into mixing
//! weights: [`MultiGateResidual`](crate::modules::MultiGateResidual) over the
//! depth-*stream* axis, and
//! [`MimoGateArm`](crate::mamba3::mimo_gate::MimoGateArm) over Mamba-3's MIMO
//! *rank* axis. One definition serves both.
//!
//! The RMS denominator is constant along the feature axis, so it is folded
//! *out* of the reduction — [`normed_score`](crate::modules::normed_score)
//! never materialises the full-width normalised tensor, while computing
//! exactly `Σ_feat(rms_norm(x) · w) · scale`.

use crate::utils::div_eps;
use burn::prelude::*;
use burn::tensor::{DType, f16};

/// The parameter-free RMS denominator `d(x) ∈ [‥, 1]` such that the RMSNorm
/// (matching [`RmsNorm`] math with `γ ≡ 1`) is `x / d(x)`, shape `[‥, 1]`.
///
/// The fp16 path keeps the same overflow-safe max-rescale as [`RmsNorm`],
/// folded into the same scalar denominator.
///
/// [`RmsNorm`]: crate::modules::RmsNorm
pub fn rms_denom<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    match x.dtype() {
        DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
            let eps = div_eps(x.dtype());
            // eps *inside* the root (matches `RmsNorm`): the `sqrt` backward
            // is otherwise singular for a zero-norm slice.
            ((x.clone() * x).mean_dim(D - 1) + eps).sqrt()
        }
        DType::F16 => {
            use burn::tensor::ElementConversion;
            let eps: f16 = f16::from_elem(div_eps(x.dtype())) * f16::from_f32(2.);
            // Single global scalar `max`, reshaped to `[1; D]` so it
            // broadcasts against the `[‥, 1]` partial RMS.
            let max = x.clone().no_grad().detach().abs().max().reshape([1; D]);
            let x_ = x.clone() / (max.clone() + eps); // x_.abs() <= 1
            // eps inside the root (matches `RmsNorm`'s F16 branch).
            let rms_partial = ((x.clone() * x_).mean_dim(D - 1) + eps).sqrt();
            // `max` is detached (no backward), but floor it too so an
            // all-zero tensor (`max = 0`) yields a nonzero denominator
            // rather than `0/0` in the caller — matching the F32 branch.
            rms_partial * (max + eps).sqrt()
        }
        _ => unreachable!("rms_denom expects a float dtype"),
    }
}

/// The RMSNorm-then-dot score `scale · Σ_feat(x · w) / (rms(x)+eps)`, shape
/// `[‥, 1]`.
///
/// `w` broadcasts against `x` on every axis but the feature one (`D-1`), where
/// it must be full width. `scale` is the query's `1/√width` temperature.
pub fn normed_score<const D: usize>(x: Tensor<D>, w: Tensor<D>, scale: f64) -> Tensor<D> {
    let dot = (x.clone() * w).sum_dim(D - 1);
    dot * scale / rms_denom(x)
}

/// `1/√width` — the temperature keeping a `width`-wide dot product `O(1)`.
pub fn score_scale(width: usize) -> f64 {
    (width as f64).powf(-0.5)
}
