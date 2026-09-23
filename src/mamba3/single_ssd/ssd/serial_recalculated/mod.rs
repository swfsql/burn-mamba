//! Single-SSD serial scan with a custom, memory-efficient backward.
//!
//! The forward and the [`Mamba3SingleSsdBackendExt`] trait are in
//! `serial_recalculated`. The registered autodiff node
//! ([`backward`](crate::mamba3::single_ssd::ssd::serial_recalculated::backward))
//! and the recompute-based gradient math
//! ([`combined_backward`](crate::mamba3::single_ssd::ssd::serial_recalculated::combined_backward))
//! save training memory: they recompute intermediates instead of storing them.

/// The registered custom `Backward` node (autodiff op).
#[cfg(feature = "autodiff")]
pub mod backward;
/// Recompute-based gradient math (the memory-efficient backward).
pub mod combined_backward;
/// The same-step γ-correction on primitives — forward and analytic backward.
pub mod diag;
mod serial_recalculated;

pub use serial_recalculated::Mamba3SingleSsdBackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba3SingleSsdAutodiffBackendExt;
