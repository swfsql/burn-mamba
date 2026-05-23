//! Helpers for converting between `FloatTensor` primitives and `Tensor<B, D>`.

use burn::prelude::*;
use burn::tensor::{TensorPrimitive, ops::FloatTensor};

/// `FloatTensor<B>` → `Tensor<B, D>`.
pub(crate) fn mk<B: Backend, const D: usize>(p: FloatTensor<B>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(p))
}
