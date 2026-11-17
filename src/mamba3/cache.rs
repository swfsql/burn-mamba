//! # The Mamba-3 cache
//!
//! [`Mamba3Cache`] is the cache that one block keeps between calls. The caches
//! of a layer stack are a `Vec<Mamba3Cache>`, with one cache for each virtual
//! layer. [`Mamba3::zero_cache`] builds the cache of a fresh sequence.
//!
//! A cache holds:
//!
//! 1. **State** (`ssm_bhpr`): `hₜ ∈ ℝ^{per_head_dim×state_rank}` for each
//!    head, the compressed context.
//! 2. **Frame cache** ([`Mamba3FrameCache`]): the fields that the plant frame
//!    writes.
//!    - **Tap FIFO** ([`TapFifo`]): the `B` and the `x` of the last `lag`
//!      write positions, oldest first. With them, the β term of the next call
//!      rebuilds `β Bₚ₋ₗₐ₉ ⊗ xₚ₋ₗₐ₉`. The FIFO exists only when the
//!      [`TapPattern`](crate::mamba3::tap_pattern::TapPattern) of the block
//!      has a β tap. It owns its layout: its zeros, and the docs of the slot
//!      order and of the decay pre-scale of `x`.
//!    - **Cumulative rotation**
//!      ([`RotationState`](crate::mamba3::rotation::RotationState)): the
//!      rotation up to the last position, to continue the data-dependent
//!      rotation.
//!    - **Positive-system fields**: `ln Λ` of a Kalman gain and `c` of a
//!      tropical register, when the block has them.
//!
//! Mamba-3 has **no conv cache**. It removes the short 1-D convolution of
//! Mamba-2: the trapezoidal discretisation and the learnable B/C biases take
//! its role (`info/mamba-3/architecture-deltas.md`).
//!
//! ## One cache for both pathways
//!
//! One cache type serves both pathways (ADR-0003). In the middle of a
//! sequence, single-SSD carries its accumulator `h'`, which is different from
//! the state `h` of double-SSD. But a cache exists only at a call boundary,
//! and there the two are equal:
//!
//! - At a call boundary, the look-ahead term `νₜ₊ₗₐ₉` is zero, so `scaleₜ = γₜ`
//!   for the last `lag` positions. The next call pays their second
//!   installment, from the tap FIFO.
//! - With this substitution, the accumulator
//!   `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ⊗xₜ` is exactly the state
//!   `hₜ = αₜ hₜ₋₁ + Σ_taps βₜ Bₜ₋ₗ⊗xₜ₋ₗ + γₜ Bₜ⊗xₜ`.
//! - The next call rebuilds the deferred β term from the tap FIFO, in the
//!   same way on both pathways.
//!
//! The frame cache has the same meaning on both pathways. So a cache from a
//! forward of one pathway continues its sequence on either pathway, and on
//! `step`.

use crate::mamba3::mamba3::Mamba3;
use burn::prelude::*;
use burn_stack::modules::sanity as san;

// ---------------------------------------------------------------------------
// The cache of one block
// ---------------------------------------------------------------------------

/// The cache that one Mamba-3 block carries between calls, on both pathways.
///
/// Each call to [`Mamba3::step`] or [`Mamba3::forward`] returns a new cache
/// with all tensors updated.
#[derive(Module, Debug)]
pub struct Mamba3Cache {
    /// **State** `hₜ`, at a call boundary.
    ///
    /// The trapezoidal recurrence of double-SSD and of `step` updates it, here
    /// at the default lag 1:
    /// `hₜ = αₜ hₜ₋₁ + βₜ (sumₘ Bₜ₋₁[m] ⊗ (xₜ₋₁ * mimo_x[m])) + γₜ (sumₘ Bₜ[m] ⊗ (xₜ * mimo_x[m]))`
    ///
    /// Single-SSD writes its accumulator `h'` here. At a call boundary, the
    /// accumulator equals the state (see the module header).
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<4>,

    /// **The frame cache** ([`Mamba3FrameCache`]): the tap FIFO, the
    /// cumulative rotation and the registers of the positive systems.
    pub frame: Mamba3FrameCache,
}

impl Mamba3Cache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        self.frame.sanity();
    }
}

// ---------------------------------------------------------------------------
// The tap FIFO
// ---------------------------------------------------------------------------

/// The tap FIFO of the trapezoid: the `B` and the `x` of the last write
/// positions. A cache keeps them for the `β` taps of the next call. One slot
/// holds one write position, and the slots are **oldest first**.
///
/// The frame cache holds it as an option ([`Mamba3FrameCache::tap_fifo`]).
/// It has [`tap_slots`](crate::mamba3::tap_pattern::TapPattern::tap_slots)
/// slots: `1` for the default, `u` for
/// [`TapPattern::Vertical`](crate::mamba3::tap_pattern::TapPattern::Vertical).
#[derive(Module, Debug)]
pub struct TapFifo {
    /// **The `B` of each slot**, after its bias and its rotation.
    ///
    /// It is stored **as rotated**, at its own position. The relative-rotation
    /// factoring `C̄ₜᵀB̄ₛ` then rebuilds the transport, so the next call
    /// applies no rotation again.
    ///
    /// Shape: `[batch, tap_slots, mimo_rank, nheads, state_rank]`
    pub b_bumhr: Tensor<5>,

    /// **The `x` of each slot**, pre-scaled by the decay accumulated since its
    /// own position (`helpers::tail_decay`). This carries the gap transport of
    /// a lag-`u` tap across the call boundary. At `lag = 1` that product is
    /// empty, so the slot is the plain `xₜ₋₁`.
    ///
    /// Shape: `[batch, tap_slots, nheads, per_head_dim]`
    pub x_buhp: Tensor<4>,
}

impl TapFifo {
    /// The FIFO of a fresh sequence: `slots` zero slots. A zero slot adds
    /// nothing to a tap.
    pub(crate) fn zeros(
        batch: usize,
        slots: usize,
        mimo_rank: usize,
        nheads: usize,
        state_rank: usize,
        per_head_dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            b_bumhr: Tensor::zeros([batch, slots, mimo_rank, nheads, state_rank], device),
            x_buhp: Tensor::zeros([batch, slots, nheads, per_head_dim], device),
        }
    }

    /// The newest `slots` slots, oldest first: the prefix of a tap that reads
    /// `slots` write positions back. The tap at
    /// [`tap_lag`](crate::mamba3::tap_pattern::TapPattern::tap_lag) reads the
    /// whole FIFO. The lag-1 tap of a two-tap pattern reads only the newest
    /// slot, which has the empty decay product. So a two-tap pattern reads one
    /// FIFO twice.
    pub(crate) fn newest(&self, slots: usize) -> Self {
        let start = self.x_buhp.dims()[1] - slots;
        Self {
            b_bumhr: self.b_bumhr.clone().narrow(1, start, slots),
            x_buhp: self.x_buhp.clone().narrow(1, start, slots),
        }
    }

    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on both
    /// tensors.
    pub fn sanity(&self) {
        san(&self.b_bumhr);
        san(&self.x_buhp);
    }
}

// ---------------------------------------------------------------------------
// The frame cache
// ---------------------------------------------------------------------------

/// The frame cache: the part of a Mamba-3 cache that the plant frame writes.
/// It has the same meaning on both pathways. [`Mamba3Cache`] holds it beside
/// `ssm_bhpr`.
#[derive(Module, Debug)]
pub struct Mamba3FrameCache {
    /// **The tap FIFO** ([`TapFifo`]). The `β` taps of the next call read it.
    /// Double-SSD rebuilds its β term
    /// `β * sumₘ Bₚ₋ₗₐ₉[m] ⊗ (xₚ₋ₗₐ₉ * mimo_x[m])` from it. Single-SSD seeds
    /// the deferred boundary β contributions
    /// `Σⱼ (1 − λⱼ) · Δⱼ · Bₚ₋ₗₐ₉₊ⱼ ⊗ xₚ₋ₗₐ₉₊ⱼ` from it. The previous call
    /// could not add them, because it did not know the `λ, Δ` of the `lag`
    /// positions that pay them.
    ///
    /// `None` (not allocated, not zeroed) under
    /// [`TapPattern::None`](crate::mamba3::tap_pattern::TapPattern::None),
    /// which has no β term.
    pub tap_fifo: Option<TapFifo>,

    /// **Cumulative data-dependent rotation** up to the current position
    /// ([`RotationState`](crate::mamba3::rotation::RotationState)). It holds
    /// nothing for `Real1D`, and the abelian rotation angle for
    /// [`Complex2D`](crate::mamba3::rotation::RotationKind::Complex2D) (each
    /// micro-step `cum_angleₜ = cum_angleₜ₋₁ + Δₜ · range · π · tanh(θₜ)`). It
    /// holds the cumulative unit quaternion(s) for the quaternion kinds.
    ///
    /// The identity for fresh sequences. It continues across calls for
    /// streaming.
    pub rotation: crate::mamba3::rotation::RotationState,

    /// **The log-precision of the Kalman gain**, `ln Λ` after the last
    /// position ([`crate::mamba3::positive::kalman`]). It is the evidence that
    /// the head holds. The gain computes the decay of the next position from
    /// it.
    ///
    /// [`LOG_ZERO`](crate::mamba3::positive::LOG_ZERO) (`Λ = 0`) for a fresh
    /// sequence. `None` under
    /// [`Gain::Projected`](crate::mamba3::positive::Gain::Projected).
    ///
    /// Shape: `[batch, nheads]`
    pub log_precision_bh: Option<Tensor<2>>,

    /// **The tropical register** `c` after the last position
    /// ([`crate::mamba3::positive::tropical`]).
    ///
    /// [`LOG_ZERO`](crate::mamba3::positive::LOG_ZERO) (the max of nothing) for
    /// a fresh sequence. `None` under
    /// [`Tropical::None`](crate::mamba3::positive::Tropical::None).
    ///
    /// Shape: `[batch, nheads]`
    pub tropical_bh: Option<Tensor<2>>,
}

impl Mamba3FrameCache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every
    /// tensor of the frame cache.
    pub fn sanity(&self) {
        if let Some(tap_fifo) = &self.tap_fifo {
            tap_fifo.sanity();
        }
        self.rotation.sanity();
        for field in [&self.log_precision_bh, &self.tropical_bh].into_iter().flatten() {
            san(field);
        }
    }
}

// ---------------------------------------------------------------------------
// The fresh cache
// ---------------------------------------------------------------------------

impl Mamba3 {
    /// The cache of a fresh sequence, for `batch` rows. It holds a zero state,
    /// a zero tap FIFO, the identity rotation and the fresh registers of the
    /// positive systems. A `forward` or a `step` without a cache starts from it, and
    /// so does each segment of a packed call without an origin. The shapes come
    /// from the block.
    pub fn zero_cache(&self, batch: usize, device: &Device) -> Mamba3Cache {
        let ssm_bhpr = Tensor::zeros(
            [batch, self.nheads(), self.per_head_dim(), self.state_rank],
            device,
        );
        let tap_fifo = self.zero_tap_fifo(batch, device);
        let rotation = self.zero_rotation_state(batch, device);
        let (log_precision_bh, tropical_bh) = self.zero_positive_state(batch, device);
        let frame = Mamba3FrameCache { tap_fifo, rotation, log_precision_bh, tropical_bh };
        Mamba3Cache { ssm_bhpr, frame }
    }
}
