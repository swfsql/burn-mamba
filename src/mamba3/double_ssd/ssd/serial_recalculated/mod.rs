//! Double-SSD serial scan with a custom, memory-efficient backward.
//!
//! The forward + [`Mamba3DoubleSsdBackendExt`] impl live in
//! `serial_recalculated`; the registered autodiff [`backward`] node and the
//! recompute-based gradient math in [`combined_backward`] save training memory
//! by recomputing intermediates instead of storing them.

/// The registered custom `Backward` node (autodiff op).
#[cfg(feature = "autodiff")]
pub mod backward;
/// Recompute-based gradient math (the memory-efficient backward).
pub mod combined_backward;
mod serial_recalculated;

pub use serial_recalculated::Mamba3DoubleSsdBackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba3DoubleSsdAutodiffBackendExt;

// Primitive forward kernels reused by the recompute backward and by the
// single-SSD pathway's backward (which shares the standard SSD kernels).
pub(crate) use serial_recalculated::{
    k1_ssd_chunk_cumsum, k2_ssd_bmm, k3_ssd_chunk_state, k4_ssd_state_passing,
};
