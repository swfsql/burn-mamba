use burn::prelude::*;

// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
#[derive(Module, Clone, Debug, Default)]
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
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        input.clone() / ((-input).exp() + 1.0)
    }
}
