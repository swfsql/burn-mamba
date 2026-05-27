//! Strategies for merging the straight (→) and reverse (←) passes of a
//! bidirectional Mamba-3 layer pair into a single `[batch, sequence, d_model]`
//! output.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::backend::Backend;

/// A zero-parameter placeholder used where the `Module` derive expects a value
/// (the `Mean` merge carries no weights).
#[derive(Module, Debug)]
pub struct NoOp;

/// How the two directions of a bidirectional pair are combined.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum OutputMerge {
    /// Element-wise average of the two directions (no parameters).
    Mean(NoOp),
    /// Concatenate along the feature axis and project back down with a learnable
    /// `[2 · d_model, d_model]` linear layer.
    CatLinear(Linear),
}

impl OutputMerge {
    /// Merge the two directional outputs.
    ///
    /// # Shapes
    ///   - Input straight `[batch, sequence, d_model]`
    ///   - Input reverse `[batch, sequence, d_model]`
    ///   - Output `[batch, sequence, d_model]`
    pub fn forward(&self, straight: Tensor<3>, reverse: Tensor<3>) -> Tensor<3> {
        let [batch_size, sequence_len, d_model] = straight.dims();
        assert_eq!(straight.dims(), reverse.dims());
        match self {
            OutputMerge::Mean(_noop) => (straight + reverse) * 0.5,
            OutputMerge::CatLinear(proj) => {
                let cat = Tensor::cat([straight, reverse].to_vec(), 2);
                assert_eq!([batch_size, sequence_len, 2 * d_model], cat.dims());
                let merged = proj.forward(cat);
                assert_eq!([batch_size, sequence_len, d_model], merged.dims());
                merged
            }
        }
    }
}

/// Configuration / factory for [`OutputMerge`].
#[derive(Config, Debug)]
pub enum OutputMergeConfig {
    /// Build an [`OutputMerge::Mean`].
    Mean,
    /// Build an [`OutputMerge::CatLinear`].
    CatLinear,
}

impl OutputMergeConfig {
    /// A vector of `n_real_layers / 2` [`Self::Mean`] configs (one per pair).
    pub fn mean(n_real_layers: usize) -> Vec<Self> {
        vec![Self::Mean; n_real_layers / 2]
    }
    /// A vector of `n_real_layers / 2` [`Self::CatLinear`] configs (one per pair).
    pub fn cat_linear(n_real_layers: usize) -> Vec<Self> {
        vec![Self::CatLinear; n_real_layers / 2]
    }

    /// Allocate the merge module on `device` for the given `d_model`.
    pub fn init(&self, d_model: usize, device: &Device) -> OutputMerge {
        match self {
            OutputMergeConfig::Mean => OutputMerge::Mean(NoOp),
            OutputMergeConfig::CatLinear => {
                let cat_linear = LinearConfig::new(d_model * 2, d_model).init(device);
                OutputMerge::CatLinear(cat_linear)
            }
        }
    }
}
