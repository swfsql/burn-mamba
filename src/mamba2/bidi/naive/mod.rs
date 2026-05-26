//! Naive bidirectional Mamba-2: the block itself is **not** adapted.
//!
//! Each `Mamba2BidiLayerPair` runs two independent layers — a straight (→) pass
//! and a reversed (←) pass over the sequence axis. The reversal happens before
//! the layer normalization, and the two directions are merged (by an
//! [`OutputMerge`]) after the block output, before the layer-pair skip
//! connection.

mod layer;
mod output_merge;

pub use layer::{
    Mamba2BidiLayerPair, Mamba2BidiLayerPairConfig, Mamba2BidiLayers, Mamba2BidiLayersConfig,
};
pub use output_merge::{OutputMerge, OutputMergeConfig};
