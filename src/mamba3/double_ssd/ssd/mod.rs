//! Standard MIMO-first SSD kernels reused by the double-SSD γ and β passes.
//!
//! Three exact reformulations (Minimal / Serial / SerialRecalculated) that
//! agree on values and gradients; selected via [`Mamba3SsdPath`] and dispatched
//! by [`Mamba3DoubleSsdInput::run`].
//!
//! [`Mamba3SsdPath`]: crate::mamba3::ssd_path::Mamba3SsdPath

/// Matmul/`segsum` MIMO-first SSD with plain autodiff backward.
pub mod minimal;
/// Serial-over-chunks SSD with plain autodiff backward.
pub mod serial;
/// Serial-over-chunks SSD with a custom recompute backward.
pub mod serial_recalculated;
/// The [`Mamba3DoubleSsdInput`] bundle and its `run()` dispatch.
pub mod ssd_path;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba3DoubleSsdAutodiffBackendExt;
pub use serial_recalculated::Mamba3DoubleSsdBackendExt;
pub use ssd_path::Mamba3DoubleSsdInput;
