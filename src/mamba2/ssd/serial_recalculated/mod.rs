//! Serial SSD with a custom, memory-efficient backward.
//!
//! The forward and the [`Mamba2BackendExt`] trait are in `serial_recalculated`.
//! The registered autodiff node ([`backward`](crate::mamba2::ssd::serial_recalculated::backward))
//! and the recompute-based gradient math
//! ([`combined_backward`](crate::mamba2::ssd::serial_recalculated::combined_backward))
//! mirror the official `ssd_combined.py`. Together they use ~⅓ less training
//! memory than a backward that stores every intermediate.

/// The registered custom `Backward` node (autodiff op).
#[cfg(feature = "autodiff")]
pub mod backward;
/// Recompute-based gradient math (the memory-efficient backward).
pub mod combined_backward;
mod serial_recalculated;

pub use serial_recalculated::Mamba2BackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba2AutodiffBackendExt;
