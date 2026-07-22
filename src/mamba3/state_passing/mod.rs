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

pub use state_passing::{Mamba3StatePassingBackendExt, state_passing};

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
