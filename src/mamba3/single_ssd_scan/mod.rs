//! Fused recurrent single-SSD scan for the production MIMO-rank-one case.
//!
//! The forward kernel replaces the five materialized serial SSD stages with the
//! exact token recurrence
//!
//! `pre = exp(da) * state; y = C * (pre + gamma * B * v);`
//! `state = pre + scale * B * v`.
//!
//! Its custom backward reconstructs the preceding state while scanning in
//! reverse, so it retains only the final state and an `O(tokens * state_rank)`
//! reduction buffer instead of an `O(tokens * per_head_dim * state_rank)` state
//! history. The operation is currently opt-in through
//! `BURN_MAMBA_FUSED_SINGLE_SCAN=1` while target-GPU performance is evaluated.

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
