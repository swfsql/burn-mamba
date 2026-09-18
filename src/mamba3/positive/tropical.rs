//! The tropical register `cₜ = lse(cₜ₋₁ + aₜ, bₜ)` (see the module header one
//! level up), from the in-projection's `(a, b)` channels.

use super::LOG_ZERO;
use super::scan::{self, Affine};
use burn::prelude::*;

/// The register at every position of a folded run, continued from `carry_bh`
/// ([`LOG_ZERO`] for a fresh sequence: the max of nothing).
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
