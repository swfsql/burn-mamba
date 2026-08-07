//! The SwiGLU feed-forward block that some checkpoints interleave between SSM
//! mixers (`mamba_ssm`'s `GatedMLP`, selected there by `d_intermediate > 0`).
//!
//! ```text
//!   [v | g] = fc1(x)            split of the `2·hidden` output down the last dim
//!   y       = fc2(v ⊙ silu(g))
//! ```
//!
//! The value half comes first and the gate half second — the order the reference
//! `y.chunk(2, dim=-1)` produces, and therefore the order the checkpoint's single
//! fused `fc1` matrix is stored in.
//!
//! `hidden` is rounded **up** to a multiple of [`GatedMlpConfig::multiple_of`], so
//! the config figure and the checkpoint shape can legitimately disagree: the
//! Mamba-3 MIMO 187m `config.json` says `d_intermediate: 1264`, while its
//! `fc1.weight` is `[2·1280, 768]`.
//!
//! Point-wise in the sequence, so `forward` and `step` are the same map applied at
//! different ranks — hence the rank-generic body.

use crate::modules::Silu;
use crate::modules::split_into;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Configuration to create a [`GatedMlp`].
#[derive(Config, Debug)]
pub struct GatedMlpConfig {
    /// Input and output width (the residual stream), `d_model`.
    pub d_model: usize,
    /// Inner width **before** rounding — `d_intermediate` in the reference
    /// configs. Rounded up to a multiple of [`Self::multiple_of`] by
    /// [`Self::hidden`].
    pub d_intermediate: usize,
    /// Round [`Self::d_intermediate`] up to a multiple of this.
    #[config(default = 128)]
    pub multiple_of: usize,
    /// Whether `fc1`/`fc2` carry a bias term.
    #[config(default = false)]
    pub has_proj_bias: bool,
}

impl GatedMlpConfig {
    /// The realised inner width: [`Self::d_intermediate`] rounded up to a
    /// multiple of [`Self::multiple_of`].
    pub fn hidden(&self) -> usize {
        self.d_intermediate.div_ceil(self.multiple_of) * self.multiple_of
    }

    /// Initialize a new [`GatedMlp`] module.
    pub fn init(&self, device: &Device) -> GatedMlp {
        let hidden = self.hidden();
        GatedMlp {
            fc1: LinearConfig::new(self.d_model, 2 * hidden)
                .with_bias(self.has_proj_bias)
                .init(device),
            fc2: LinearConfig::new(hidden, self.d_model)
                .with_bias(self.has_proj_bias)
                .init(device),
            activation: Silu::new(),
        }
    }
}

/// A SwiGLU feed-forward block: `fc2(v ⊙ silu(g))` where `[v | g] = fc1(x)`.
///
/// Should be created using the [`GatedMlpConfig`] configuration.
#[derive(Module, Debug)]
pub struct GatedMlp {
    /// Fused value+gate up-projection, `d_model → 2·hidden`.
    pub fc1: Linear,
    /// Down-projection, `hidden → d_model`.
    pub fc2: Linear,
    /// The gate activation (SiLU).
    pub activation: Silu,
}

impl GatedMlp {
    /// The inner width `hidden` (inferred from `fc2`).
    pub fn hidden(&self) -> usize {
        let [hidden, _d_model] = self.fc2.weight.dims();
        hidden
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    /// - input `x`: `[..., d_model]`
    /// - output: `[..., d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let hidden = self.hidden();
        let [value, gate] = split_into(self.fc1.forward(x), [hidden, hidden], D - 1);
        self.fc2.forward(value * self.activation.forward(gate))
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
