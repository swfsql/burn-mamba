//! Double-SSD serial scan with a custom, memory-efficient backward.
//!
//! The forward and the [`Mamba3DoubleSsdBackendExt`] trait are in
//! `serial_recalculated`. The registered autodiff node
//! ([`backward`](crate::mamba3::double_ssd::ssd::serial_recalculated::backward))
//! and the recompute-based gradient math
//! ([`combined_backward`](crate::mamba3::double_ssd::ssd::serial_recalculated::combined_backward))
//! save training memory: they recompute intermediates instead of storing them.

/// The registered custom `Backward` node (autodiff op).
#[cfg(feature = "autodiff")]
pub mod backward;
/// Recompute-based gradient math (the memory-efficient backward).
pub mod combined_backward;
mod serial_recalculated;

pub use serial_recalculated::Mamba3DoubleSsdBackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba3DoubleSsdAutodiffBackendExt;

// Primitive forward kernels that the recompute backward uses again, and that
// the backward of the single-SSD pathway also uses (it shares the standard
// SSD kernels).
pub(crate) use serial_recalculated::{
    cat_chunk_groups, k1_ssd_chunk_cumsum, k2_ssd_bmm, k3_ssd_chunk_state, k4_ssd_state_passing,
    k4_ssd_state_passing_backward,
};
