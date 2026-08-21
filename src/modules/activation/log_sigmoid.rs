//! Numerically-stable log-sigmoid: `log σ(x) = −log(1 + e^−x)`.
//!
//! Evaluated as `−softplus(−x)` on every float format:
//! [`softplus`](crate::modules::softplus) already branches to the identity past
//! a per-dtype threshold, which is what keeps the large-negative tail
//! (`log σ(x) → x`) finite — computing `log(1 / (1 + e^−x))` directly instead
//! overflows `e^−x` there and collapses to `−∞` with a `NaN` gradient.

use burn::prelude::*;

/// Applies the log-sigmoid function element-wise: `log(1 / (1 + e^−x))`.
///
/// Panics on non-float element types.
pub fn log_sigmoid<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    // log_sigmoid(x) = -log(1 + e^-x) = -softplus(-x)
    -crate::modules::softplus(x.neg())
}
