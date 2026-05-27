//! The example networks (`in_proj → Layers → out_proj`) plus the config
//! builders, and the [`ModelConfigExt`] factory trait that lets the generic
//! training loop build any model config into a module.

use burn::prelude::*;
use burn::backend::Backend;

/// Bidirectional wrapper networks for the examples.
pub mod bidi;
mod model;

pub use model::{
    mamba2::{MyMamba2Network, MyMamba2NetworkConfig, mamba2_block_config, mamba2_layers_config},
    mamba3::{MyMamba3Network, MyMamba3NetworkConfig, mamba3_block_config, mamba3_layers_config},
};

/// A model config that can build its module on a device — the seam the generic
/// training loop uses to stay model-agnostic.
pub trait ModelConfigExt<B: Backend>: Config {
    /// The module type this config builds.
    type Model: Module<B>;
    /// Allocate and initialise the model on `device`.
    fn init(&self, device: &Device) -> Self::Model;
}
