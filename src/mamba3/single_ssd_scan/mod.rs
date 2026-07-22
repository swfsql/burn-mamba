//! Fused recurrent single-SSD scan for the production MIMO-rank-one case.
//!
//! The forward kernel replaces the five materialized serial SSD stages with the
//! exact token recurrence
//!
//! `pre = exp(da) * state; y = C * (pre + gamma * B * v);`
//! `state = pre + scale * B * v`.
//!
//! Its custom backward reconstructs the preceding state while scanning in
//! reverse. Checkpoints every eight tokens cap inverse-decay reconstruction at
//! a numerically stable interval, while an `O(tokens * state_rank)` reduction
//! buffer and one-eighth checkpoint history replace the full state history.
//! CubeCL backends use the operation by default; set
//! `BURN_MAMBA_FUSED_SINGLE_SCAN=0` to select the five-stage tensor reference.

pub(super) const RECONSTRUCTION_INTERVAL: usize = 8;

mod single_ssd_scan;

#[cfg(feature = "autodiff")]
mod backward;

#[cfg(feature = "cubecl")]
mod cube;

#[cfg(feature = "fusion")]
mod fusion;

pub use single_ssd_scan::{Mamba3SingleSsdScanBackendExt, single_ssd_scan};

#[cfg(all(test, feature = "_dev-test"))]
pub(crate) use single_ssd_scan::single_ssd_scan_reference;
