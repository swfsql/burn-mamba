//! Numerically-stable log-sigmoid: `log σ(x) = −log(1 + e^−x)`.
//!
//! The wider float formats evaluate `log(1 / (1 + e^−x))` directly; the fp16
//! path uses the stable identity `log σ(x) = −softplus(−x)` (see
//! [`softplus`](crate::utils::softplus)) to avoid overflow.

use burn::prelude::*;
use burn::tensor::DType;

/// Applies the log-sigmoid function element-wise: `log(1 / (1 + e^−x))`.
///
/// Panics on non-float element types.
pub fn log_sigmoid<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    match x.dtype() {
        DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
            // log_sigmoid(x) = log(1 / (1 + exp(-x)))
            (x.neg().exp() + 1.).recip().log()
        }
        DType::F16 => {
            // log_sigmoid(x) = -softplus(-x)
            -crate::utils::softplus::softplus(x.neg())
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
