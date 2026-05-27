//! Helpers for converting between `FloatTensor` primitives and `Tensor<D>`.

use burn::prelude::*;
use burn::backend::{TensorPrimitive, tensor::FloatTensor};
use burn::backend::Backend;

/// `FloatTensor<B>` → `Tensor<D>`.
pub(crate) fn mk<B: Backend, const D: usize>(p: FloatTensor<B>) -> Tensor<D> {
    Tensor::from_primitive(TensorPrimitive::Float(p))
}
