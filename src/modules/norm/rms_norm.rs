//! Root-mean-square normalisation over the last dimension.
//!
//! `RMSNorm(x) = x / rms(x) · γ` where `rms(x) = √(mean(x²))`.  Unlike
//! LayerNorm there is no mean-subtraction or bias — only a learnable per-channel
//! scale `γ`.  Used both as the Pre-LN of every residual block and, in Mamba-3,
//! as the **QK-Norm** applied to the B/C projections.
//!
//! The fp16 path avoids forming `x²` directly (which overflows for moderately
//! large activations, e.g. 256·256): it first normalises against `max(|x|)` so
//! the squared values stay `≤ 1`, then rescales.  See [`rms_norm_gated`] for the
//! SiLU-gated variant.
//!
//! [`rms_norm_gated`]: crate::utils::rms_norm_gated

use crate::utils::div_eps;
use burn::module::{Content, DisplaySettings, ModuleDisplay, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::{DType, f16};

/// Configuration to create a [`RmsNorm`] layer.
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    /// The size of the input features.
    pub d_model: usize,
}

impl RmsNormConfig {
    /// Initialize a new [`RmsNorm`] module.
    pub fn init(&self, device: &Device) -> RmsNorm {
        let gamma = Initializer::Ones.init([self.d_model], device);
        RmsNorm { gamma }
    }
}

/// Applies RMS normalisation over an input tensor along the last dimension:
/// `y = x / √(mean(x²)) · γ`.
///
/// Should be created using the [`RmsNormConfig`] configuration.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct RmsNorm {
    /// The learnable per-channel scale `γ`, shape `[d_model]`.
    pub gamma: Param<Tensor<1>>,
}

impl RmsNorm {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    /// - input `x`: `[..., d_model]`
    /// - output: `[..., d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let normalized = match x.dtype() {
            DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
                let div_eps = div_eps(x.dtype());
                let rms = (x.clone() * x.clone()).mean_dim(D - 1).sqrt();
                let normalized = (x / (rms + div_eps)) * self.gamma.val().unsqueeze();
                normalized
            }
            DType::F16 => {
                use burn::tensor::ElementConversion;
                let div_eps: f16 = f16::from_elem(div_eps(x.dtype())) * f16::from_f32(2.);
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
            DType::Bool(_) => {
                unreachable!()
            }
            DType::QFloat(_) => {
                unimplemented!()
            }
        };
        normalized
    }
}

impl ModuleDisplay for RmsNorm {
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
