//! Single-SSD serial scan with a custom, memory-efficient backward.
//!
//! The forward and the [`Mamba3SingleSsdBackendExt`] trait are in
//! `serial_recalculated`. The `RecomputeOp` impl
//! ([`backward`](crate::mamba3::single_ssd::ssd::serial_recalculated::backward))
//! and the recompute-based gradient math
//! ([`combined_backward`](crate::mamba3::single_ssd::ssd::serial_recalculated::combined_backward))
//! save training memory: they recompute intermediates instead of storing them.

/// The `RecomputeOp` impl of the autodiff backend.
#[cfg(feature = "autodiff")]
pub mod backward;
/// The assembly of the memory-efficient backward. Its kernels are in the
/// [standard SSD scan](crate::standard_ssd::kernels).
pub mod combined_backward;
/// The diagonal γ-correction on primitives — forward and analytic backward.
pub mod diag;
mod serial_recalculated;

pub use serial_recalculated::Mamba3SingleSsdBackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba3SingleSsdAutodiffBackendExt;
