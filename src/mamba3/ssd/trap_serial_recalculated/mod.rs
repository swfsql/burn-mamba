#[cfg(feature = "autodiff")]
pub mod backward;
pub mod combined_backward;
mod trap_serial_recalculated;

pub use trap_serial_recalculated::Mamba3TrapBackendExt;

#[cfg(feature = "autodiff")]
pub use trap_serial_recalculated::Mamba3TrapAutodiffBackendExt;
