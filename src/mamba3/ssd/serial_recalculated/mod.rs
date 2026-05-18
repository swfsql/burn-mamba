#[cfg(feature = "autodiff")]
pub mod backward;
pub mod combined_backward;
mod serial_recalculated;

pub use serial_recalculated::Mamba3BackendExt;

#[cfg(feature = "autodiff")]
pub use serial_recalculated::Mamba3AutodiffBackendExt;
