//! Single-SSD serial scan with a custom, memory-efficient backward.
//!
//! The forward + [`Mamba3SingleSsdBackendExt`] impl live in
//! `serial_recalculated`; the registered autodiff [`backward`] node and the
//! recompute-based gradient math in [`combined_backward`] save training memory
//! by recomputing intermediates instead of storing them.

/// The registered custom `Backward` node (autodiff op).
#[cfg(feature = "autodiff")]
pub mod backward;
/// Recompute-based gradient math (the memory-efficient backward).
pub mod combined_backward;
mod serial_recalculated;

pub use serial_recalculated::Mamba3SingleSsdBackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba3SingleSsdAutodiffBackendExt;
