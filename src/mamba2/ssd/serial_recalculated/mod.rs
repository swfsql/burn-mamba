//! Serial SSD with a custom, memory-efficient backward.
//!
//! The forward + [`Mamba2BackendExt`] impl live in `serial_recalculated`; the
//! registered autodiff [`backward`] node and the recompute-based gradient math
//! in [`combined_backward`] together mirror the official `ssd_combined.py`,
//! saving ~⅓ of the training memory versus storing every intermediate.

/// The registered custom `Backward` node (autodiff op).
#[cfg(feature = "autodiff")]
pub mod backward;
/// Recompute-based gradient math (the memory-efficient backward).
pub mod combined_backward;
mod serial_recalculated;

pub use serial_recalculated::Mamba2BackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba2AutodiffBackendExt;
