//! Inter-chunk state passing shared by the serial Mamba-3 SSD paths.
//!
//! The recurrence is exposed as a backend extension so CubeCL backends can run
//! the whole chunk scan as one operation.  The autodiff implementation uses a
//! single custom node with an exact reverse recurrence.

/// Backend extension, primitive reference implementation and high-level wrapper.
pub mod state_passing;

/// Exact custom backward for the state-passing recurrence.
#[cfg(feature = "autodiff")]
pub mod backward;

/// Fused forward and backward kernels for raw CubeCL backends.
#[cfg(feature = "cubecl")]
mod cube;

/// Fusion custom-operation registration around the CubeCL backend operation.
#[cfg(feature = "fusion")]
mod fusion;

pub use state_passing::{Mamba3StatePassingBackendExt, state_passing};

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
