//! Small scan primitives shared by the serial Mamba-3 SSD paths.
//!
//! The intra-chunk cumulative sum (K1) and inter-chunk state recurrence (K4)
//! are exposed as backend extensions so CubeCL backends can run each scan as
//! one operation. Their autodiff implementations use exact custom reverse
//! scans.

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

pub use state_passing::{Mamba3StatePassingBackendExt, chunk_cumsum, state_passing};

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
