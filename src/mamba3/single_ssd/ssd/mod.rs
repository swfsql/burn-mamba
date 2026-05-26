//! Standard MIMO-first SSD kernels specialised to the single-pass form.
//!
//! Three exact reformulations (Minimal / Serial / SerialRecalculated) that
//! agree on values and gradients; selected via [`Mamba3SsdPath`] and dispatched
//! by [`Mamba3SingleSsdInput::run`].  These take the raw `v` plus `gamma`/`scale`
//! so the kernel applies the trapezoid weights and boundary seed internally.
//!
//! [`Mamba3SsdPath`]: crate::mamba3::ssd_path::Mamba3SsdPath

/// Matmul/`segsum` MIMO-first SSD with plain autodiff backward.
pub mod minimal;
/// Serial-over-chunks SSD with plain autodiff backward.
pub mod serial;
/// Serial-over-chunks SSD with a custom recompute backward.
pub mod serial_recalculated;
/// The [`Mamba3SingleSsdInput`] bundle and its `run()` dispatch.
pub mod ssd_path;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba3SingleSsdAutodiffBackendExt;
pub use serial_recalculated::Mamba3SingleSsdBackendExt;
pub use ssd_path::Mamba3SingleSsdInput;
