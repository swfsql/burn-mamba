//! RMS normalisation fused with a SiLU(z) gate — the Mamba-2 output norm.
//!
//! `norm_before_gate` selects the order of the two operations:
//! - `true`  — normalise, then gate:   `y = (x / rms(x) · γ) · SiLU(z)`
//! - `false` — gate, then normalise:   `y = rms(x · SiLU(z)) · γ` applied to
//!   `x · SiLU(z)`
//!
//! The numerical-stability epsilon is the per-dtype [`div_eps`] (so there is no
//! configurable epsilon); the fp16 path uses the same `max(|x|)`-rescaling
//! trick as [`RmsNorm`](crate::utils::rms_norm::RmsNorm).

use crate::utils::div_eps;
use crate::utils::silu::Silu;
use burn::backend::Backend;
use burn::module::{Content, DisplaySettings, ModuleDisplay, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::{DType, f16};

/// Configuration to create a [`RmsNormGated`] layer.
#[derive(Config, Debug)]
pub struct RmsNormGatedConfig {
    /// The size of the input features.
    pub d_model: usize,
    /// Whether to apply normalization before gating. Default: true
    #[config(default = true)]
    pub norm_before_gate: bool,
}

impl RmsNormGatedConfig {
    /// Initialize a new [`RmsNormGated`] module.
    pub fn init(&self, device: &Device) -> RmsNormGated {
        let gamma = Initializer::Ones.init([self.d_model], device);
        RmsNormGated {
            gamma,
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
/// Should be created using the [`RmsNormGatedConfig`] configuration.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct RmsNormGated {
    /// The learnable per-channel scale `γ`, shape `[d_model]`.
    pub gamma: Param<Tensor<1>>,
    /// Whether to normalize before applying the gating.
    pub norm_before_gate: bool,
}

impl RmsNormGated {
    /// Applies the forward pass on the input tensor with gating.
    ///
    /// # Shapes
    /// - input `x`: `[..., any, d_model]`
    /// - input `z`: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<D>, z: Tensor<D>) -> Tensor<D> {
        let silu = Silu::new();

        let x = if self.norm_before_gate {
            // gate will be applied later
            x
        } else {
            // gate before norm
            x * silu.forward(z.clone())
        };

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
            DType::Bool(_) => {
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

impl ModuleDisplay for RmsNormGated {
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
