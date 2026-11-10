//! Double-SSD serial scan with a custom, memory-efficient backward.
//!
//! The forward and the [`Mamba3DoubleSsdBackendExt`] trait are in
//! `serial_recalculated`. The `RecomputeOp` impl
//! ([`backward`](crate::mamba3::double_ssd::ssd::serial_recalculated::backward))
//! calls the backward of the [standard SSD scan](crate::standard_ssd). It saves
//! training memory: it recomputes intermediates instead of storing them.

/// The `RecomputeOp` impl of the autodiff backend.
#[cfg(feature = "autodiff")]
pub mod backward;
mod serial_recalculated;

pub use serial_recalculated::Mamba3DoubleSsdBackendExt;
