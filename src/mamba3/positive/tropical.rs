//! The tropical register `cₜ = lse(cₜ₋₁ + aₜ, bₜ)` (see the module header one
//! level up), from the `(a, b)` channels of the in-projection.

use super::LOG_ZERO;
use super::scan::{self, Affine};
use burn::prelude::*;

/// The register at every position of a folded run, continued from `carry_bh`
/// ([`LOG_ZERO`] for a fresh sequence: the max of nothing).
///
/// # At `u > 1`
///
/// The register advances once per **micro-step**, like the recurrence. A token
/// applies its `u` maps `(aⱼ, bⱼ)` in order, and their composition is again one
/// affine element: `a = Σⱼ aⱼ`, `b = lse over j of (bⱼ + Σ_{r>j} aᵣ)`. So a
/// token still applies one map of the same kind. But its `b` is a soft maximum
/// of `u` affine reads of the token, not one, and one token can both reset and
/// count.
///
/// The readout sees the value after the last micro-step of the token (the
/// read axis, `helpers::read_rows`), and the cache holds the same value. The
/// `u − 1` values between are computed and never read. A composition of the
/// maps of each token first (`u − 1` elementwise [`Affine`] composes), then a
/// scan over tokens, would give the same reads and carry at `1/u` of the
/// scanned length. This function scans the folded axis instead: the axis that
/// the Kalman gate needs, because every micro-step reads its decay.
///
/// # Shapes
/// - `a_bsh`, `b_bsh` : `[batch, len, nheads]`
/// - `carry_bh`       : `[batch, nheads]`
/// - out              : `[batch, len, nheads]`
pub fn register(a_bsh: Tensor<3>, b_bsh: Tensor<3>, carry_bh: Tensor<2>) -> Tensor<3> {
    let elements = Affine {
        a: a_bsh.clamp_min(LOG_ZERO),
        b: b_bsh.clamp_min(LOG_ZERO),
    };
    scan::prefix(elements).apply(carry_bh)
}
