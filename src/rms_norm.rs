use crate::utils::div_eps;
use burn::module::{Content, DisplaySettings, ModuleDisplay, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::{DType, Element, f16};

/// Configuration to create a [RmsNorm](RmsNorm) layer.
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// The size of the input features.
    pub d_model: usize,
}

impl RmsNormConfig {
    /// Initialize a new [RmsNorm](RmsNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        let gamma = Initializer::Ones.init([self.d_model], device);
        RmsNorm { gamma }
    }
}

/// Applies Rms Normalization over an input tensor along the last dimension.
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `z` is the gating tensor
/// - `gamma` is the learnable weight
/// - `mean` is the mean operation
///
/// Should be created using the [RmsNormConfig](RmsNormConfig) configuration.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct RmsNorm<B: Backend> {
    /// The learnable parameter to scale the normalized tensor.
    pub gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend> RmsNorm<B> {
    /// Applies the forward pass on the input tensor with gating.
    ///
    /// # Shapes
    /// - input `x`: `[..., any, d_model]`
    /// - input `z`: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
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
                let x_ = x.clone() / (max.clone() + div_eps); // x_.abs() <= 1
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
        normalized
    }
}

impl<B: Backend> ModuleDisplay for RmsNorm<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_model] = self.gamma.shape().dims();
        content.add("d_model", &d_model).optional()
    }
}
