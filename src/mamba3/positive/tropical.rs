//! The tropical register `cₜ = lse(cₜ₋₁ + aₜ, bₜ)` (see the module header one
//! level up), from the in-projection's `(a, b)` channels.

use super::LOG_ZERO;
use super::scan::{self, Affine};
use burn::prelude::*;

/// The register at every position of a folded run, continued from `carry_bh`
/// ([`LOG_ZERO`] for a fresh sequence: the max of nothing).
///
/// # At `u > 1`
///
/// The register advances once per **micro-step**, like the recurrence: a token
/// applies its `u` maps `(aⱼ, bⱼ)` in order, and their composition is again one
/// affine element — `a = Σⱼ aⱼ`, `b = lse over j of (bⱼ + Σ_{r>j} aᵣ)`. So a
/// token still applies one map of the same kind, but its `b` is a soft maximum
/// of `u` affine reads of the token rather than one, and a single token can
/// both reset and count. What the readout sees is the value after the token's
/// last micro-step (the read axis, `helpers::read_rows`), and the cache carries
/// the same one; the `u − 1` values in between are computed and never read.
/// Composing each token's maps first (`u − 1` elementwise
/// [`Affine`] composes) and scanning over tokens would give the same reads and
/// carry at `1/u` of the scanned length. This scans the folded axis instead —
/// the axis the Kalman gate needs, whose decay every micro-step does read.
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
