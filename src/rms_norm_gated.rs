use crate::silu::Silu;
use burn::module::Param;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::DType;

/// Configuration to create a [RmsNormGated](RmsNormGated) layer.
#[derive(Config, Debug)]
pub struct RmsNormGatedConfig {
    /// The size of the input features.
    pub d_model: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// Whether to apply normalization before gating. Default: true
    #[config(default = true)]
    pub norm_before_gate: bool,
}

impl RmsNormGatedConfig {
    /// Initialize a new [RmsNormGated](RmsNormGated) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNormGated<B> {
        assert!(self.epsilon > 0.0, "epsilon must be positive.");

        let gamma = Initializer::Ones.init([self.d_model], device);

        RmsNormGated {
            gamma,
            epsilon: self.epsilon,
            norm_before_gate: self.norm_before_gate,
        }
    }
}

/// Applies Gated Rms Normalization over an input tensor along the last dimension.
///
/// - If `norm_before_gate=true`: `Y = (X / sqrt(mean(X^2) + eps) * gamma) * SiLU(z)`
/// - If `norm_before_gate=false`: `Y = (X * SiLU(z)) / sqrt(mean((X * SiLU(z))^2) + eps) * gamma`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `z` is the gating tensor
/// - `gamma` is the learnable weight
/// - `mean` is the mean operation
/// - `eps` is a small value to avoid division by zero.
///
/// Should be created using the [RmsNormGatedConfig](RmsNormGatedConfig) configuration.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct RmsNormGated<B: Backend> {
    /// The learnable parameter to scale the normalized tensor.
    pub gamma: Param<Tensor<B, 1>>,
    /// A value required for numerical stability.
    pub epsilon: f64,
    /// Whether to normalize before applying the gating.
    pub norm_before_gate: bool,
}

impl<B: Backend> RmsNormGated<B> {
    /// Applies the forward pass on the input tensor with gating.
    ///
    /// # Shapes
    /// - input `x`: `[..., any, d_model]`
    /// - input `z`: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>, z: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let silu = Silu::new();

        let x = if self.norm_before_gate {
            // gate will be applied later
            x
        } else {
            // gate before norm
            x * silu.forward(z.clone())
        };

        let rms =
            (x.clone().cast(DType::F32).powf_scalar(2.0).mean_dim(D - 1) + self.epsilon).sqrt();
        let normalized = (x / rms.cast(dtype)) * self.gamma.val().unsqueeze();

        if self.norm_before_gate {
            // gate gets applied late (now)
            normalized * silu.forward(z)
        } else {
            // gate already got applied before
            normalized
        }
    }
}

impl<B: Backend> ModuleDisplay for RmsNormGated<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_model] = self.gamma.shape().dims();
        content
            .add("d_model", &d_model)
            .add("epsilon", &self.epsilon)
            .optional()
    }
}
