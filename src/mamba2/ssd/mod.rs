//! # Mamba-2 chunkwise SSD algorithms
//!
//! Three exact reformulations of the same Structured State Space Duality scan.
//! They agree on values **and** gradients (the `ssd_path` tests assert it):
//!
//! - [`minimal`] — mostly batched matmuls and a `segsum` mask, with the plain
//!   autodiff backward.
//! - [`serial`] — a serial loop over chunks (it mirrors the Triton kernels
//!   K1–K5), with the plain autodiff backward.
//! - [`serial_recalculated`] — the same serial loop with a **custom backward**
//!   that recomputes intermediates. It uses ~⅓ less training memory.
//!
//! [`ssd_path`] holds the [`Mamba2SsdPath`] selector and the [`Mamba2SsdInput`]
//! bundle whose `run()` dispatches to one of the three.

/// Matmul/`segsum` SSD with plain autodiff backward.
pub mod minimal;
/// Serial-over-chunks SSD with plain autodiff backward.
pub mod serial;
/// Serial-over-chunks SSD with a custom recompute backward.
pub mod serial_recalculated;
/// The [`Mamba2SsdPath`] selector and [`Mamba2SsdInput`] dispatch bundle.
pub mod ssd_path;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba2AutodiffBackendExt;
pub use serial_recalculated::Mamba2BackendExt;
pub use ssd_path::{Mamba2SsdInput, Mamba2SsdPath};
