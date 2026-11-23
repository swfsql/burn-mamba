//! Standard MIMO-first SSD kernels reused by the double-SSD γ and β passes.
//!
//! Three exact reformulations (Minimal / Serial / SerialRecalculated) agree on
//! values and gradients. [`SsdPath`] selects one, and
//! [`Mamba3DoubleSsdInput::run`](crate::mamba3::double_ssd::ssd::ssd_path::Mamba3DoubleSsdInput::run)
//! dispatches it.
//!
//! [`SsdPath`]: crate::ssd_path::SsdPath

/// Matmul/`segsum` MIMO-first SSD with plain autodiff backward.
pub mod minimal;
/// Serial-over-chunks SSD with plain autodiff backward.
pub mod serial;
/// Serial-over-chunks SSD with a custom recompute backward.
pub mod serial_recalculated;
/// The [`Mamba3DoubleSsdInput`] bundle and its `run()` dispatch.
pub mod ssd_path;

pub use serial_recalculated::Mamba3DoubleSsdBackendExt;
pub use ssd_path::Mamba3DoubleSsdInput;
