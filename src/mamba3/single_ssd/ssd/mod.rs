//! Standard MIMO-first SSD kernels specialised to the single-pass form.
//!
//! Three exact reformulations (Minimal / Serial / SerialRecalculated) agree on
//! values and gradients. [`SsdPath`] selects one, and
//! [`Mamba3SingleSsdInput::run`](crate::mamba3::single_ssd::ssd::ssd_path::Mamba3SingleSsdInput::run)
//! dispatches it. These kernels take the raw `v` and `gamma`/`scale`, so the
//! kernel applies the trapezoid weights and the boundary seed internally.
//!
//! [`SsdPath`]: crate::ssd_path::SsdPath

/// The diagonal γ-correction shared by all three SSD paths.
pub mod diag;
/// Matmul/`segsum` MIMO-first SSD with plain autodiff backward.
pub mod minimal;
/// Serial-over-chunks SSD with plain autodiff backward.
pub mod serial;
/// Serial-over-chunks SSD with a custom recompute backward.
pub mod serial_recalculated;
/// The [`Mamba3SingleSsdInput`] bundle and its `run()` dispatch.
pub mod ssd_path;

pub use serial_recalculated::Mamba3SingleSsdBackendExt;
pub use ssd_path::Mamba3SingleSsdInput;
