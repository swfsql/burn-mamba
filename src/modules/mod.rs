use burn::config::Config;
use burn::prelude::*;

/// Custom activations (fp16-stable `silu` / `softplus` / `log_sigmoid`).
pub mod activation;
/// Bidirectional layer stacks (straight + reversed passes, merged per pair).
pub mod bidi;
/// The per-network cache collection trait ([`CacheStack`]) + [`MambaCaches`].
pub mod cache;
/// A single Pre-LN residual layer wrapping one SSM block ([`Layer`]).
pub mod layer;
/// The (virtual-)layer stack over real weight sets ([`Layers`]).
pub mod layers;
/// Loss functions (binary cross-entropy, cross-entropy, mean squared error).
pub mod loss;
/// Tensor helpers: `segsum`, `gqa`, typed `split`, and `sanity` guards.
pub mod misc;
/// Multi-Gate Residuals: multi-stream gated depth-wise residuals ([`Residuals`]).
pub mod multi_gate;
/// Family-generic networks ([`MambaLatentNet`] / [`MambaVocabNet`]).
pub mod network;
/// RMS norms ([`RmsNorm`] QK-norm + [`RmsNormGated`]), fp16-safe.
pub mod norm;

pub use activation::log_sigmoid::log_sigmoid;
pub use activation::silu::Silu;
pub use activation::softplus::softplus;
pub use misc::gqa::gqa_expand_to_heads;
pub use misc::sanity::sanity;
pub use misc::segsum::segsum;
pub use misc::split::split_into;
pub use norm::rms_norm::{RmsNorm, RmsNormConfig};
pub use norm::rms_norm_gated::{RmsNormGated, RmsNormGatedConfig};

pub use bidi::{MambaBidiLayers, MambaBidiLayersConfig};
pub use cache::{CacheStack, MambaCaches};
pub use layer::Layer;
pub use layers::{Layers, LayersBuilder};
pub use multi_gate::{
    MultiGate, MultiGateResidual, MultiGateResidualConfig, Residuals, ResidualsConfig,
};
pub use network::{MambaLatentNet, MambaLatentNetConfig, MambaVocabNet, MambaVocabNetConfig};

/// Per-family block interface the generic [`Layer`]/[`Layers`] delegate to.
pub trait MambaBlock: Module {
    /// Per-block streaming cache (one layer's worth of state).
    type Cache;
    /// The per-network cache collection for this family.
    type Caches: CacheStack<Cache = Self::Cache>;
    /// SSD algorithm / chunk-length selector. `()` for families without one.
    type SsdPath;

    /// Full-sequence (chunked) pass — training / prefill.
    fn block_forward(
        &self,
        x: Tensor<3>,
        cache: Option<Self::Cache>,
        ssd_path: Self::SsdPath,
    ) -> (Tensor<3>, Self::Cache);

    /// Single-token recurrent step — decoding.
    fn block_step(&self, x: Tensor<2>, cache: Option<Self::Cache>) -> (Tensor<2>, Self::Cache);

    /// Build `n_virtual` zero caches sized for a `[batch, sequence, d_model]` input.
    fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Self::Caches;
    /// Build `n_virtual` zero caches sized for a `[batch, d_model]` input.
    fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Self::Caches;
}

/// A block *config* that knows its `d_model` and how to build its [`MambaBlock`].
/// Lets the generic builders construct `Layers<M>` without knowing the family.
pub trait MambaBlockConfig: Config {
    /// The block this config builds.
    type Block: MambaBlock;
    /// Model width, used to size each layer's pre-norm.
    fn d_model(&self) -> usize;
    /// Allocate and initialise the block on `device`.
    fn init_block(&self, device: &Device) -> Self::Block;
}

// ===========================================================================
// Unifying enums: one runtime + one serializable Config across all families
// ===========================================================================
//
// The generic `LatentNetwork<M>` above is family-typed (`M` is fixed at the type
// level). To let an example (or a user) choose the family at *runtime* — and to
// serialize that choice for docs/config round-trips — we wrap the three
// monomorphisations in enums. `#[derive(Module)]` and `#[derive(Config)]` both
// support enums (verified), so this stays first-class Burn.

/// An explicit, family-tagged SSD-path selector for the unified API.
///
/// Each variant carries the concrete per-family path so callers can choose the
/// algorithm/chunk explicitly; the `*_default` constructors offer the common
/// "ride along the family default" path without making it the *only* option.
#[derive(Debug, Clone)]
pub enum MambaSsdPath {
    /// Mamba-1 has no SSD chunking (path is the unit type).
    #[cfg(feature = "mamba1")]
    Mamba1,
    /// Mamba-2 SSD path.
    #[cfg(feature = "mamba2")]
    Mamba2(crate::mamba2::prelude::Mamba2SsdPath),
    /// Mamba-3 SSD path.
    #[cfg(feature = "mamba3")]
    Mamba3(crate::mamba3::prelude::Mamba3SsdPath),
}

impl MambaSsdPath {
    /// The Mamba-2 default path (`SerialRecalculated`, optimal chunk).
    #[cfg(feature = "mamba2")]
    pub fn mamba2_default() -> Self {
        Self::Mamba2(Default::default())
    }
    /// The Mamba-3 default path (`SerialRecalculated`, optimal chunk).
    #[cfg(feature = "mamba3")]
    pub fn mamba3_default() -> Self {
        Self::Mamba3(Default::default())
    }
}
