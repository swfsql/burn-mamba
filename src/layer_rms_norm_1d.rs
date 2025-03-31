// Implementation references:
// - https://github.com/johnma2006/mamba-minimal/blob/03de542a36d873f6e6c4057ad687278cc6ae944d/model.py#L328
// - https://github.com/kroggen/mamba.c/blob/7387f49e352f86a0c22041c0f66fd2a40b58a207/mamba.c#L222

use burn::{module::Param, nn::Initializer, prelude::*};

/// Applies RMS Layer Normalization over an input tensor as described in the paper [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
///
/// `Y = rms_norm(X) * γ + β`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `γ` is the learnable weight
/// - `β` is the learnable bias
///
/// Should be created using [LayerRmsNorm1DConfig](LayerRmsNorm1DConfig).
#[derive(Module, Debug)]
pub struct LayerRmsNorm1D<B: Backend> {
    /// The learnable weight.
    pub gamma: Param<Tensor<B, 1>>,
    /// The learnable bias.
    pub beta: Param<Tensor<B, 1>>,
    /// A value required for numerical stability.
    epsilon: f64,
}

/// Configuration to create a [LayerRmsNorm1D](LayerRmsNorm1D) layer using the [init function](LayerRmsNorm1D::init).
#[derive(Debug, Config)]
pub struct LayerRmsNorm1DConfig {
    /// The size of the input features.
    pub d_model: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl LayerRmsNorm1DConfig {
    /// Initialize a new [layer rms norm](LayerRmsNorm1D) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerRmsNorm1D<B> {
        let gamma = Initializer::Ones.init([self.d_model], device);
        let beta = Initializer::Zeros.init([self.d_model], device);

        LayerRmsNorm1D {
            gamma,
            beta,
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> LayerRmsNorm1D<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See the [LayerRmsNorm1D](LayerRmsNorm1D) documentation for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // x rms_norm
        let shape = x.dims();
        let sq = x.clone() * x.clone();
        let sq_mean = sq.mean_dim(D - 1);
        let rsqrt = (sq_mean + self.epsilon).sqrt().recip().expand(shape);
        let x_normalized = x * rsqrt;

        (x_normalized * self.gamma.val().unsqueeze()) + self.beta.val().unsqueeze()
    }
}
