//! # Mamba-3 Caches (double-SSD pathway)
//!
//! The cache that the block keeps between calls:
//!
//! 1. **State**: `hₜ ∈ ℝ^{per_head_dim×state_rank}` per head, the
//!    compressed context.
//! 2. **Frame cache**
//!    ([`Mamba3FrameCache`](crate::mamba3::cache::Mamba3FrameCache)): the
//!    fields that the plant frame writes, with the same meaning in both
//!    pathways.
//!    - **Tap FIFO** ([`TapFifo`](crate::mamba3::cache::TapFifo)): the `B` and
//!      the `x` of the last `lag` positions, oldest first. With them, the β
//!      term of the trapezoidal recurrence rebuilds `β Bₚ₋ₗₐ₉ ⊗ xₚ₋ₗₐ₉`. The
//!      FIFO exists only when the
//!      [`TapPattern`](crate::mamba3::tap_pattern::TapPattern) of the block
//!      has a β tap.
//!    - **Cumulative rotation**
//!      ([`RotationState`](crate::mamba3::rotation::RotationState)): the
//!      rotation accumulated up to position `t`, to continue the
//!      data-dependent rotation.
//!    - **Positive-system fields**: `ln Λ` of a Kalman gain and `c` of a
//!      tropical register, when the block has them.
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

    /// **The frame cache** ([`Mamba3FrameCache`]): the tap FIFO, the
    /// cumulative rotation and the registers of the positive systems. The
    /// plant frame writes it, and it has the same meaning in both pathways.
    pub frame: Mamba3FrameCache,
}

impl Mamba3DoubleSsdCache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        self.frame.sanity();
    }
}
