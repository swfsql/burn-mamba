//! # Memory-efficient quaternion cumulative-product scan (custom backward)
//!
//! The quaternion kinds (`Quaternion4D`, `Rotor4D`) compose their per-step
//! rotations with a cumulative product over the sequence
//! ([`crate::mamba3::rotation::quat_cumprod`], a Hillis–Steele parallel scan).
//! In plain autodiff, that scan keeps `O(log seq)` *full-sequence*
//! intermediates for the backward pass: fast, but memory-hungry.
//!
//! This module gives the **recompute** alternative, like the SSD
//! `SerialRecalculated` design. The `Autodiff` impl ([`backward`]) of the
//! [`Mamba3QuatScanBackendExt`] trait registers one custom
//! [`Backward`](burn::backend::autodiff::ops::Backward) node. The node saves
//! only the leaf inputs (the per-step quaternions and the carry) and
//! recomputes the scan during backprop. The gradient is the exact quaternion
//! VJP of the cumulative product, computed with parallel ops (a prefix product
//! and a reverse cumsum). There is no token loop, so the memory saving does
//! not cost a slow backward.
//!
//! [`quat_cumprod_recalculated`] replaces `quat_cumprod` in
//! [`rotate_bc_forward`](crate::mamba3::rotation::rotate_bc_forward). The
//! plain-autodiff `quat_cumprod` stays as the verified reference: the tests
//! assert that the two agree on values **and** gradients.

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
