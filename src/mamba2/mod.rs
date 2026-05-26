//! # Mamba-2
//!
//! Structured State Space Duality (SSD).  Mamba-2 recasts the selective SSM
//! recurrence as a chunkwise algorithm built from batched GEMMs, making it
//! tensor-core friendly for training while remaining exactly equivalent to the
//! recurrent form used for decoding.  See [`mamba2`] for the full SSD math.
//!
//! - [`mamba2`] / [`layer`] / [`network`] — the block, the residual layer
//!   stack, and the full language model.
//! - [`cache`] — the conv-window + SSM-state carried between calls.
//! - [`ssd`] — the pluggable chunkwise SSD algorithms (Minimal / Serial /
//!   SerialRecalculated) and the backend extension trait.
//! - [`bidi`] — bidirectional wrappers for non-autoregressive tasks.

pub mod bidi;
pub mod cache;
pub mod layer;
pub mod mamba2;
pub mod network;
pub mod ssd;

/// Public re-exports for Mamba-2.
pub mod prelude {
    use super::*;
    pub use cache::{Mamba2Cache, Mamba2CacheConfig, Mamba2Caches, Mamba2CachesConfig};
    pub use layer::{Mamba2Layer, Mamba2LayerConfig, Mamba2Layers, Mamba2LayersConfig};
    pub use mamba2::{Mamba2, Mamba2Config};
    pub use network::{Mamba2Network, Mamba2NetworkConfig};
    #[cfg(feature = "autodiff")]
    pub use ssd::Mamba2AutodiffBackendExt;
    pub use ssd::Mamba2BackendExt;
    pub use ssd::{Mamba2SsdInput, Mamba2SsdPath};
}
