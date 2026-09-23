//! # Mamba-2
//!
//! Structured State Space Duality (SSD). Mamba-2 recasts the selective SSM
//! recurrence as a chunkwise algorithm made of batched matrix multiplications.
//! This form uses tensor cores in training, and it stays exactly equal to the
//! recurrent form that decoding uses. See [`mamba2`](crate::mamba2::mamba2) for
//! the full SSD math.
//!
//! - [`mamba2`](crate::mamba2::mamba2) — the SSD block.
//! - [`cache`](crate::mamba2::cache) — the conv window and the SSM state that
//!   the block carries between calls.
//! - [`ssd`](crate::mamba2::ssd) — the pluggable chunkwise SSD algorithms
//!   (Minimal / Serial / SerialRecalculated) and the backend extension trait.
//!
//! The residual layers, the networks and the bidirectional stacks are the
//! family-generic types in [`burn_stack::modules`]. [`crate::unified`] wraps
//! them in runtime-selectable enums.

pub mod cache;
pub mod mamba2;
pub mod ssd;

/// Public re-exports for Mamba-2.
pub mod prelude {
    use super::*;
    pub use cache::{Mamba2Cache, Mamba2CacheConfig, Mamba2Caches, Mamba2CachesConfig};
    pub use mamba2::{Mamba2, Mamba2Config, Mamba2Untied};
    #[cfg(feature = "autodiff")]
    pub use ssd::Mamba2AutodiffBackendExt;
    pub use ssd::Mamba2BackendExt;
    pub use ssd::{Mamba2SsdInput, Mamba2SsdPath};
}
