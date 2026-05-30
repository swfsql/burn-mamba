//! # Mamba-1
//!
//! The original selective state space model.  Mamba-1 runs a **sequential
//! selective scan** (no SSD, no backend-extension trait).  See [`mamba1`] for
//! the block; the residual layer stack, the full language model, and the
//! bidirectional wrappers are the family-generic types in [`crate::generic`]
//! (e.g. `MambaLatentNet` / `MambaVocabNet` / `MambaBidiLayers`).
//!
//! - [`mamba1`] — the selective-SSM block.
//! - [`cache`] — the conv-window + SSM-state carried between calls.

pub mod cache;
pub mod mamba1;

/// Public re-exports for Mamba-1.
pub mod prelude {
    use super::*;
    pub use cache::{Mamba1Cache, Mamba1CacheConfig, Mamba1Caches, Mamba1CachesConfig};
    pub use mamba1::{Mamba1, Mamba1Config};
}
