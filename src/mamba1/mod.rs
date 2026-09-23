//! # Mamba-1
//!
//! The original selective state space model. Mamba-1 runs a **sequential
//! selective scan**. It has no SSD and no backend-extension trait.
//!
//! - [`mamba1`](crate::mamba1::mamba1) — the selective-SSM block.
//! - [`cache`](crate::mamba1::cache) — the conv window and the SSM state that
//!   the block carries between calls.
//!
//! The residual layers, the networks and the bidirectional stacks are the
//! family-generic types in [`burn_stack::modules`]. [`crate::unified`] wraps
//! them in runtime-selectable enums.

pub mod cache;
pub mod mamba1;

/// Public re-exports for Mamba-1.
pub mod prelude {
    use super::*;
    pub use cache::{Mamba1Cache, Mamba1CacheConfig, Mamba1Caches, Mamba1CachesConfig};
    pub use mamba1::{Mamba1, Mamba1Config, Mamba1Untied};
}
