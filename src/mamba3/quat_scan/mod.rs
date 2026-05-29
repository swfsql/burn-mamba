//! # Memory-efficient quaternion cumulative-product scan (custom backward)
//!
//! The Quaternion4D forward composes its per-step rotations with a cumulative
//! product over the sequence ([`crate::mamba3::rotation::quat_cumprod`], a
//! Hillis–Steele parallel scan). Expressed in plain autodiff that scan retains
//! `O(log seq)` *full-sequence* intermediates for the backward pass — fast, but
//! memory-hungry.
//!
//! This module provides the **recompute** alternative, mirroring the SSD
//! `SerialRecalculated` design: a [`Mamba3QuatScanBackendExt`] trait whose
//! `Autodiff` impl ([`backward`]) registers a single custom
//! [`Backward`](burn::backend::autodiff::ops::Backward) node that saves only the
//! leaf inputs (the per-step quaternions and the carry) and recomputes the scan
//! during backprop. The gradient is the exact quaternion VJP of the cumulative
//! product, evaluated with parallel ops (a prefix product + a reverse-cumsum) —
//! no token loop, so the saved memory does not buy back a slow backward.
//!
//! [`quat_cumprod_recalculated`] is the high-level drop-in for `quat_cumprod`
//! used by the Quaternion4D `forward` path; the plain-autodiff `quat_cumprod`
//! stays as the verified reference (the tests assert the two agree on values
//! **and** gradients).

/// The backend-extension trait, its primitive default body, the per-backend
/// impls, and the [`quat_cumprod_recalculated`] high-level wrapper.
pub mod quat_scan;

/// The registered custom `Backward` node (autodiff op) + the recompute gradient
/// math.
#[cfg(feature = "autodiff")]
pub mod backward;

pub use quat_scan::{Mamba3QuatScanBackendExt, quat_cumprod_recalculated};

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
