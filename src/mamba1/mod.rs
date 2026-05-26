//! # Mamba-1
//!
//! The original selective state space model.  Mamba-1 runs a **sequential
//! selective scan** (no SSD, no backend-extension trait, no bidirectional
//! wrappers); it does share the [`layer`]-stack virtual scheduling and the
//! cache-threaded [`network`] with Mamba-2/3.  See [`mamba1`] for the block.
//!
//! - [`mamba1`] / [`layer`] / [`network`] — the block, the residual layer
//!   stack, and the full language model.
//! - [`cache`] — the conv-window + SSM-state carried between calls.

pub mod cache;
pub mod layer;
pub mod mamba1;
pub mod network;

/// Public re-exports for Mamba-1.
pub mod prelude {
    use super::*;
    pub use cache::{Mamba1Cache, Mamba1CacheConfig, Mamba1Caches, Mamba1CachesConfig};
    pub use layer::{Mamba1Layer, Mamba1LayerConfig, Mamba1Layers, Mamba1LayersConfig};
    pub use mamba1::{Mamba1, Mamba1Config};
    pub use network::{Mamba1Network, Mamba1NetworkConfig};
}
