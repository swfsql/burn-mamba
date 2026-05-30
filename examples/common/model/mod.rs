//! The example model surface: just the [`ModelConfigExt`] factory trait (the
//! seam the generic training loop uses to stay model-agnostic) plus its impl for
//! the library's unified [`MambaLatentNetConfig`].
//!
//! The examples no longer define their own networks. They build directly from
//! the library's family-generic types (`in_proj → Layers → out_proj`, exposed as
//! the runtime-selectable [`MambaLatentNet`]); each example just picks the family
//! variant in its `model_config()`.

use burn::prelude::*;
use burn_mamba::prelude::{MambaLatentNet, MambaLatentNetConfig};

/// A model config that can build its module on a device — the seam the generic
/// training loop uses to stay model-agnostic.
pub trait ModelConfigExt: Config {
    /// The module type this config builds.
    type Model: Module;
    /// Allocate and initialise the model on `device`.
    fn init(&self, device: &Device) -> Self::Model;
}

impl ModelConfigExt for MambaLatentNetConfig {
    type Model = MambaLatentNet;
    fn init(&self, device: &Device) -> Self::Model {
        // `self.init(..)` resolves to the inherent `MambaLatentNetConfig::init`
        // (inherent methods win over trait methods in method-call syntax), so
        // this delegates to the library builder rather than recursing.
        self.init(device)
    }
}
