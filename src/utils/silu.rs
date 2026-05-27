//! SiLU (a.k.a. swish) activation: `silu(x) = x · sigmoid(x)`.
//!
//! Implemented as `x / (1 + exp(−x))`, which is fp16-aware (no separate
//! `sigmoid` op) and used for the gating branches throughout the Mamba blocks.

use burn::prelude::*;
use burn::backend::Backend;

/// SiLU activation module: `silu(x) = x · sigmoid(x) = x / (1 + exp(−x))`.
#[derive(Module, Debug, Default)]
pub struct Silu;

impl Silu {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        input.clone() / ((-input).exp() + 1.0)
    }
}
