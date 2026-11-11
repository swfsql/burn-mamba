//! # Mamba-3 Caches (double-SSD pathway)
//!
//! The cache that the block keeps between calls:
//!
//! 1. **State**: `hₜ ∈ ℝ^{per_head_dim×state_rank}` per head, the
//!    compressed context.
//! 2. **Tap FIFO** ([`TapFifo`](crate::mamba3::cache::TapFifo)): the `B` and
//!    the `x` of the last `lag` positions, oldest first. With them, the β term
//!    of the trapezoidal recurrence rebuilds `β Bₚ₋ₗₐ₉ ⊗ xₚ₋ₗₐ₉`. The FIFO
//!    exists only when the
//!    [`TapPattern`](crate::mamba3::tap_pattern::TapPattern) of the block has a
//!    β tap. It is as deep as its
//!    [`tap_lag`](crate::mamba3::tap_pattern::TapPattern::tap_lag) (`1` for the
//!    default, `u` for
//!    [`TapPattern::Vertical`](crate::mamba3::tap_pattern::TapPattern::Vertical)).
//! 3. **Cumulative rotation**
//!    ([`RotationState`](crate::mamba3::rotation::RotationState)): the rotation accumulated up
//!    to position `t`, to continue the data-dependent rotation.
//! 4. **Positive-system fields**: `ln Λ` of a Kalman gain and `c` of a tropical
//!    register, when the block has them.
//!
//! Mamba-3 has **no conv cache**. It removes the short 1-D convolution of
//! Mamba-2: the trapezoidal discretisation and the learnable B/C biases take
//! its role (`info/mamba-3/architecture-deltas.md`).

use crate::mamba3::prelude::*;
use burn_stack::modules::sanity as san;
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba3DoubleSsdCaches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer caches for a complete Mamba-3 network.
#[derive(Module, Debug)]
pub struct Mamba3DoubleSsdCaches {
    /// Per-layer caches. The length is the number of virtual layers.
    pub caches: Vec<Mamba3DoubleSsdCache>,
}

// ---------------------------------------------------------------------------
// Mamba3DoubleSsdCache  (cache for a single block)
// ---------------------------------------------------------------------------

/// The cache that a **single** Mamba-3 block carries between calls.
///
/// Each call to [`Mamba3::step`](crate::mamba3::mamba3::Mamba3::step) or
/// [`Mamba3::forward`](crate::mamba3::mamba3::Mamba3::forward) returns a new
/// cache with all tensors updated.
#[derive(Module, Debug)]
pub struct Mamba3DoubleSsdCache {
    /// **State** `hₜ`.
    ///
    /// The (double-ssd) trapezoidal recurrence updates it:
    /// `hₜ = αₜ hₜ₋₁ + βₜ (sumₘ Kₜ₋₁[m] ⊗ (Vₜ₋₁ * mimo_x[m])) + γₜ (sumₘ Bₜ[m] ⊗ (xₜ * mimo_x[m]))`
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<4>,

    /// **The tap FIFO** ([`TapFifo`]), which rebuilds the β term of the next
    /// call: `β * sumₘ Bₚ₋ₗₐ₉[m] ⊗ (xₚ₋ₗₐ₉ * mimo_x[m])`.
    ///
    /// `None` (not allocated, not zeroed) under [`TapPattern::None`], which has
    /// no β term.
    pub tap_fifo: Option<TapFifo>,

    /// **Cumulative data-dependent rotation** up to the current position
    /// ([`RotationState`]): nothing for `Real1D`, the abelian rotation angle
    /// for [`Complex2D`](crate::mamba3::rotation::RotationKind::Complex2D)
    /// (each micro-step `cum_angleₜ = cum_angleₜ₋₁ + Δₜ · range · π · tanh(θₜ)`), or
    /// the cumulative unit quaternion(s) for the quaternion kinds.
    ///
    /// The identity for fresh sequences. It continues across calls for
    /// streaming.
    pub rotation: RotationState,

    /// **The log-precision of the Kalman gain**, `ln Λ` after the last
    /// position ([`crate::mamba3::positive::kalman`]): the evidence that the
    /// head holds, from which the gain computes the decay of the next position.
    ///
    /// [`LOG_ZERO`](crate::mamba3::positive::LOG_ZERO) (`Λ = 0`) for a fresh
    /// sequence. `None` under [`Gain::Projected`].
    ///
    /// Shape: `[batch, nheads]`
    pub log_precision_bh: Option<Tensor<2>>,

    /// **The tropical register** `c` after the last position
    /// ([`crate::mamba3::positive::tropical`]).
    ///
    /// [`LOG_ZERO`](crate::mamba3::positive::LOG_ZERO) (the max of nothing) for
    /// a fresh sequence. `None` under [`Tropical::None`].
    ///
    /// Shape: `[batch, nheads]`
    pub tropical_bh: Option<Tensor<2>>,
}

impl Mamba3DoubleSsdCache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        if let Some(tap_fifo) = &self.tap_fifo {
            tap_fifo.sanity();
        }
        self.rotation.sanity();
        for field in [&self.log_precision_bh, &self.tropical_bh].into_iter().flatten() {
            san(field);
        }
    }
}
