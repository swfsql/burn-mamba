//! Serial SSD with a custom, memory-efficient backward.
//!
//! The forward and the [`Mamba2BackendExt`] trait are in `serial_recalculated`.
//! The `RecomputeOp` impl ([`backward`](crate::mamba2::ssd::serial_recalculated::backward))
//! and the gradient math
//! ([`combined_backward`](crate::mamba2::ssd::serial_recalculated::combined_backward))
//! call the [standard SSD scan](crate::standard_ssd). Together they use ~⅓ less
//! training memory than a backward that stores every intermediate.

/// The `RecomputeOp` impl of the autodiff backend.
#[cfg(feature = "autodiff")]
pub mod backward;
/// Recompute-based gradient math (the memory-efficient backward).
pub mod combined_backward;
mod serial_recalculated;

pub use serial_recalculated::Mamba2BackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba2AutodiffBackendExt;
