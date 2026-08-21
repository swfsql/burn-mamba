//! Softplus activation: `softplus(x) = log(1 + eˣ)`, a smooth ReLU.
//!
//! Used to produce the strictly-positive discretisation step `Δ` (and, in
//! Mamba-3, the data-dependent `A`).  Above a per-dtype threshold `log(1 + eˣ)`
//! is indistinguishable from `x` in that format (in the value *and* in the
//! derivative, which has already saturated at `1`), so the tail is evaluated as
//! the identity.  The `log1p(eˣ)` branch is then only ever fed inputs clamped to
//! that threshold — far below the `eˣ` overflow point — so the usual
//! `max(x, 0) + log(1 + e^−|x|)` rewrite is not needed.

use burn::prelude::*;
use burn::tensor::DType;

/// Applies the softplus function element-wise: `log(1 + eˣ)`.
///
/// Panics on non-float element types.
pub fn softplus<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    // Forward + backward precision limit, ie. the smallest `x` from which both
    // `softplus(x)` and its derivative round to `x` and `1` in that format.
    // Well under the `eˣ` overflow points (f16: 11.09, bf16/f32: 88.72, f64: 710).
    let threshold = match x.dtype() {
        DType::F64 => 38.,
        DType::F32 | DType::Flex32 => 18.,
        DType::BF16 => 7.,
        DType::F16 => 9.,
        DType::I64
        | DType::I32
        | DType::I16
        | DType::I8
        | DType::U64
        | DType::U32
        | DType::U16
        | DType::U8 => {
            unreachable!()
        }
        DType::Bool(_) => {
            unreachable!()
        }
        DType::QFloat(_) => {
            unimplemented!()
        }
    };

    // softplus = log(e^x + 1)  below the threshold,  x  above it.
    let above = x.clone().greater_elem(threshold);
    let below = x.clone().clamp_max(threshold);
    below.exp().log1p().mask_where(above, x)
}
