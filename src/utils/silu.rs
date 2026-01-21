use crate::utils::contains_nan_or_inf;
use burn::prelude::*;

// silu activation for x is x * sigmoid(x)
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
        if contains_nan_or_inf(&input) {
            panic!();
        }

        let sigmoid = nn::Sigmoid::new().forward(input.clone());
        if contains_nan_or_inf(&sigmoid) {
            panic!();
        }

        let res = input * sigmoid;

        if contains_nan_or_inf(&res) {
            panic!();
        }

        res
    }
}
