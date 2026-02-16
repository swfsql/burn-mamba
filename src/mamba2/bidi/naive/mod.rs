//! Naive implementation where the Mamba2 block is not adapted.
//!
//! Two independent layers are executed as a bidi pair,
//! where the input flip-split happens before the layer normalization,
//! and they are merged (by a ) after the block output,
//! before the layer-pair skip connection.

mod layer;
mod output_merge;

pub use layer::{
    Mamba2BidiLayerPair, Mamba2BidiLayerPairConfig, Mamba2BidiLayers, Mamba2BidiLayersConfig,
};
pub use output_merge::{OutputMerge, OutputMergeConfig};
