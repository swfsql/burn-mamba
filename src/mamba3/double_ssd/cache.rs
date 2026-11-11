//! # Mamba-3 Caches (double-SSD pathway)
//!
//! The cache that the block keeps between calls:
//!
//! 1. **State**: `hₜ ∈ ℝ^{per_head_dim×state_rank}` per head, the
//!    compressed context.
//! 2. **Previous K state**: the `B` of the last `lag` positions, oldest first,
//!    `[batch, lag, mimo_rank, nheads, state_rank]`. The β term of the
//!    trapezoidal recurrence needs it.
//! 3. **Previous V state**: the matching `x`, `[batch, lag, nheads,
//!    per_head_dim]`. With k_state, it rebuilds `β Bₚ₋ₗₐ₉ ⊗ xₚ₋ₗₐ₉`.
//!    Items 2 and 3 are the **tap slots** of the trapezoid. They exist only
//!    when the [`TapPattern`](crate::mamba3::tap_pattern::TapPattern) of the block
//!    has a β tap. The FIFO is as deep as its
//!    [`tap_lag`](crate::mamba3::tap_pattern::TapPattern::tap_lag) (`1` for the
//!    default, `u` for
//!    [`TapPattern::Vertical`](crate::mamba3::tap_pattern::TapPattern::Vertical)).
//! 4. **Cumulative rotation**
//!    ([`RotationState`](crate::mamba3::rotation::RotationState)): the rotation accumulated up
//!    to position `t`, to continue the data-dependent rotation.
//! 5. **Positive-system fields**: `ln Λ` of a Kalman gain and `c` of a tropical
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

    /// **The B of the tap FIFO**, one slot per lagged position, **oldest
    /// first**.
    ///
    /// It rebuilds the β term: `β * sumₘ Bₚ₋ₗₐ₉[m] ⊗ (xₚ₋ₗₐ₉ * mimo_x[m])`.
    /// Stored **as rotated**, at its own position: the relative-rotation
    /// factoring `C̄ₜᵀB̄ₛ` then rebuilds the transport, so the next call applies
    /// no rotation again.
    ///
    /// The tap slots of the trapezoid are `None` (not allocated, not zeroed)
    /// under [`TapPattern::None`], which has no β term. Present exactly when
    /// [`Self::v_state_buhp`] is (see [`TapPattern::tap_slots`]).
    ///
    /// Shape: `[batch, tap_slots, mimo_rank, nheads, state_rank]`
    pub k_state_bumhr: Option<Tensor<5>>,

    /// **The x of the tap FIFO**, matching [`Self::k_state_bumhr`] slot for
    /// slot, and `None` with it.
    ///
    /// Each slot is **pre-scaled by the decay accumulated since its own
    /// position** (`helpers::tail_decay`). This carries the gap transport of a
    /// lag-`u` tap across the call boundary. At `lag = 1` that product is
    /// empty, so the slot is the plain `xₜ₋₁`.
    ///
    /// Shape: `[batch, tap_slots, nheads, per_head_dim]`
    pub v_state_buhp: Option<Tensor<4>>,

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
        assert_eq!(
            self.k_state_bumhr.is_some(),
            self.v_state_buhp.is_some(),
            "the trapezoid's tap slots are present or absent together"
        );
        if let Some(k_state_bumhr) = &self.k_state_bumhr {
            san(k_state_bumhr);
        }
        if let Some(v_state_buhp) = &self.v_state_buhp {
            san(v_state_buhp);
        }
        self.rotation.sanity();
        for field in [&self.log_precision_bh, &self.tropical_bh].into_iter().flatten() {
            san(field);
        }
    }
}
