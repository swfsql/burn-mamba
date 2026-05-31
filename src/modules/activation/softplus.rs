//! Softplus activation: `softplus(x) = log(1 + eˣ)`, a smooth ReLU.
//!
//! Used to produce the strictly-positive discretisation step `Δ` (and, in
//! Mamba-3, the data-dependent `A`).  The fp16 path uses the numerically-stable
//! identity `softplus(x) = max(x, 0) + log(1 + e^−|x|)` to avoid overflow in
//! `eˣ`; the wider formats use `log1p(eˣ)` directly.

use burn::backend::Backend;
use burn::prelude::*;
use burn::tensor::DType;

/// Applies the softplus function element-wise: `log(1 + eˣ)`.
///
/// Panics on non-float element types.
pub fn softplus<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    match x.dtype() {
        DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
            // softplus = log(e^x + 1)
            x.exp().log1p()
        }
        DType::F16 => {
            // (x.exp() + 1.).log()

            // max(a,b) = (a + b + |a-b|)/2
            // softplus = max(x, 0) + log(e^-|x| + 1)
            //          = (x + |x|) / 2 + log(e^-|x| + 1)
            let xabs = x.clone().abs();
            (x + xabs.clone()) / 2. + xabs.neg().exp().log1p()
        }
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
    }
}
