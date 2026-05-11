use crate::utils::div_eps;
use crate::utils::silu::Silu;
use burn::module::{Content, DisplaySettings, ModuleDisplay, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::{DType, Element, f16};

/// Configuration to create a [RmsNormGated](RmsNormGated) layer.
#[derive(Config, Debug)]
pub struct RmsNormGatedConfig {
    /// The size of the input features.
    pub d_model: usize,
    // // TODO: config epsilon is no longer used.
    // /// A value required for numerical stability. Default: 1e-5
    // #[config(default = 1e-5)]
    // pub epsilon: f64,
    /// Whether to apply normalization before gating. Default: true
    #[config(default = true)]
    pub norm_before_gate: bool,
}

impl RmsNormGatedConfig {
    /// Initialize a new [RmsNormGated](RmsNormGated) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNormGated<B> {
        // assert!(self.epsilon > 0.0, "epsilon must be positive.");

        let gamma = Initializer::Ones.init([self.d_model], device);

        RmsNormGated {
            gamma,
            // epsilon: self.epsilon,
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
    // // TODO: config epsilon is no longer used.
    // /// A value required for numerical stability.
    // pub epsilon: f64,
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
        let silu = Silu::new();

        let x = if self.norm_before_gate {
            // gate will be applied later
            x
        } else {
            // gate before norm
            x * silu.forward(z.clone())
            // x * burn::tensor::activation::leaky_relu(z.clone(), 0.01)
        };

        let normalized = match <B::FloatElem as Element>::dtype() {
            DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
                let div_eps = div_eps::<B>();

                let rms = (x.clone() * x.clone()).mean_dim(D - 1).sqrt();
                let normalized = (x / (rms + div_eps)) * self.gamma.val().unsqueeze();
                normalized
            }
            DType::F16 => {
                use burn::tensor::ElementConversion;
                let div_eps: f16 = f16::from_elem(div_eps::<B>()) * f16::from_f32(2.);

                // avoid calculating x² directly (due to overflow e.g. on 256 * 256)
                let max = x.clone().no_grad().detach().abs().max().expand(x.shape());
                let x_ = x.clone() / (max.clone() + div_eps); // |x_| <= 1
                let rms_partial = (x.clone() * x_).mean_dim(D - 1).sqrt(); // √(x²) = √(x²/max) * √max
                let normalized =
                    (x / (rms_partial + div_eps)) / max.sqrt() * self.gamma.val().unsqueeze();
                normalized
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
            DType::Bool => {
                unreachable!()
            }
            DType::QFloat(_) => {
                unimplemented!()
            }
        };

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
            // .add("epsilon", &self.epsilon)
            .optional()
    }
}
