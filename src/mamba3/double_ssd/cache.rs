//! # Mamba-3 Caches (double-SSD pathway)
//!
//! The state that the block keeps between calls:
//!
//! 1. **SSM hidden state**: `hₜ ∈ ℝ^{per_head_dim×state_rank}` per head, the
//!    compressed context.
//! 2. **Previous K state**: the `B` of the last `lag` positions, oldest first,
//!    `[batch, lag, mimo_rank, nheads, state_rank]`. The β term of the
//!    trapezoidal recurrence needs it.
//! 3. **Previous V state**: the matching `x`, `[batch, lag, nheads,
//!    per_head_dim]`. With k_state, it rebuilds `β Bₚ₋ₗₐ₉ ⊗ xₚ₋ₗₐ₉`.
//!    Items 2 and 3 are the **tap slots** of the trapezoid. They exist only
//!    when the [`Trapezoid`](crate::mamba3::trapezoid::Trapezoid) of the block
//!    has a β tap. The FIFO is as deep as its
//!    [`tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag) (`1` for the
//!    default, `u` for
//!    [`Trapezoid::Vertical`](crate::mamba3::trapezoid::Trapezoid::Vertical)).
//! 4. **Cumulative rotation**
//!    ([`RotationState`](crate::mamba3::rotation::RotationState)): the rotation accumulated up
//!    to position `t`, to continue the data-dependent rotation.
//! 5. **Positive-system slots**: `ln Λ` of a Kalman gain and `c` of a tropical
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

/// Configuration / factory for [`Mamba3DoubleSsdCaches`].
#[derive(Config, Debug)]
pub struct Mamba3DoubleSsdCachesConfig {
    /// Number of cache slots (= number of virtual layers).
    pub n_caches: usize,

    /// Shared configuration that determines the shape of each cache.
    pub cache: Mamba3DoubleSsdCacheConfig,
}

impl Mamba3DoubleSsdCachesConfig {
    /// Convenience constructor from a block config.
    pub fn new_from_block_config(
        n_caches: usize,
        batch: usize,
        block_config: Mamba3Config,
    ) -> Self {
        Self {
            n_caches,
            cache: Mamba3DoubleSsdCacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init(&self, device: &Device) -> Mamba3DoubleSsdCaches {
        let caches = (0..self.n_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba3DoubleSsdCaches { caches }
    }
}

// ---------------------------------------------------------------------------
// Mamba3DoubleSsdCache  (state for a single layer)
// ---------------------------------------------------------------------------

/// The state carried between calls for a **single** Mamba-3 layer.
///
/// Each call to [`Mamba3::step`](crate::mamba3::mamba3::Mamba3::step) or
/// [`Mamba3::forward`](crate::mamba3::mamba3::Mamba3::forward) returns a new
/// cache with all tensors updated.
#[derive(Module, Debug)]
pub struct Mamba3DoubleSsdCache {
    /// **SSM hidden state** `hₜ`.
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
    /// under [`Trapezoid::None`], which has no β term. Present exactly when
    /// [`Self::v_state_buhp`] is (see [`Trapezoid::tap_slots`]).
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
    /// ([`RotationState`]): nothing for `Real1D`, the abelian RoPE angle for
    /// [`Complex2D`](crate::mamba3::rotation::RotationKind::Complex2D) (each
    /// step `cum_angleₜ = cum_angleₜ₋₁ + Δₜ · range · π · tanh(θₜ)`), or the
    /// cumulative unit quaternion(s) for the quaternion kinds.
    ///
    /// The identity for fresh sequences. It continues across calls for
    /// streaming.
    pub rotation: RotationState,

    /// **The log-precision of the Kalman gate**, `ln Λ` after the last
    /// position ([`crate::mamba3::positive::kalman`]): the evidence that the
    /// head holds, from which the gate computes the decay of the next position.
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
        for slot in [&self.log_precision_bh, &self.tropical_bh].into_iter().flatten() {
            san(slot);
        }
    }
}

/// Configuration / factory for a single [`Mamba3DoubleSsdCache`].
#[derive(Config, Debug)]
pub struct Mamba3DoubleSsdCacheConfig {
    /// Batch size.
    pub batch: usize,

    /// State rank.
    #[config(default = 128)]
    pub state_rank: usize,

    /// Head dimension per_head_dim.
    #[config(default = 64)]
    pub per_head_dim: usize,

    /// Number of SSM heads.
    pub nheads: usize,

    /// MIMO rank.  1 = SISO.
    #[config(default = 1)]
    pub mimo_rank: usize,

    /// Number of RoPE angle pairs = `rope_dim / 2` = `(state_rank * rope_fraction) / 2`
    /// (rounded down to even by `Mamba3Config::rope_dim`).
    pub num_rope_angles: usize,

    /// Which transition rotation the block uses ([`RotationKind`]). It selects
    /// the accumulator variant (see [`RotationState::identity`]).
    #[config(default = "crate::mamba3::rotation::RotationKind::Complex2D")]
    pub rotation: RotationKind,

    /// Number of quaternion blocks (`rope_dim / 4`). Only
    /// [`RotationKind::Quaternion4D`] / [`RotationKind::Rotor4D`] use it.
    #[config(default = 1)]
    pub num_quat_blocks: usize,

    /// The tap pattern of the block ([`Trapezoid`]). It decides whether the
    /// tap slots exist, and, with [`Self::micro_steps`], how many there are
    /// (see [`Trapezoid::tap_slots`]).
    #[config(default = "crate::mamba3::trapezoid::Trapezoid::HorizontalCarryOver")]
    pub trapezoid: Trapezoid,

    /// Recurrence micro-steps per token (`u`). Only the lag-`u` tap patterns
    /// read it (see [`Trapezoid::tap_lag`]).
    #[config(default = 1)]
    pub micro_steps: usize,

    /// The gain of the block. A Kalman gain keeps a `ln Λ` slot.
    #[config(default = "crate::mamba3::positive::Gain::Projected")]
    pub gain: crate::mamba3::positive::Gain,

    /// The tropical register of the block. `MaxPlus` keeps a `c` slot.
    #[config(default = "crate::mamba3::positive::Tropical::None")]
    pub tropical: crate::mamba3::positive::Tropical,
}

impl Mamba3DoubleSsdCacheConfig {
    /// Derive cache shapes from a Mamba-3 block configuration plus a batch size.
    pub fn new_from_block_config(batch: usize, block_config: Mamba3Config) -> Self {
        Self {
            batch,
            state_rank: block_config.state_rank,
            per_head_dim: block_config.per_head_dim,
            nheads: block_config.nheads(),
            mimo_rank: block_config.mimo_rank,
            num_rope_angles: block_config.num_rope_angles(),
            rotation: block_config.rotation,
            num_quat_blocks: block_config.num_quat_blocks(),
            trapezoid: block_config.trapezoid,
            micro_steps: block_config.micro_steps,
            gain: block_config.gain,
            tropical: block_config.tropical,
        }
    }

    /// Allocate zero/identity-initialised cache tensors on `device`.
    pub fn init(&self, device: &Device) -> Mamba3DoubleSsdCache {
        let ssm_bhpr = Tensor::zeros(
            [self.batch, self.nheads, self.per_head_dim, self.state_rank],
            device,
        );
        let slots = self.trapezoid.tap_slots(self.micro_steps);
        let tap = slots > 0;
        let k_state_bumhr = tap.then(|| {
            Tensor::zeros(
                [
                    self.batch,
                    slots,
                    self.mimo_rank,
                    self.nheads,
                    self.state_rank,
                ],
                device,
            )
        });
        let v_state_buhp = tap.then(|| {
            Tensor::zeros(
                [self.batch, slots, self.nheads, self.per_head_dim],
                device,
            )
        });
        let rotation = RotationState::identity(
            self.rotation,
            self.batch,
            self.nheads,
            self.num_rope_angles,
            self.num_quat_blocks,
            device,
        );
        let (log_precision_bh, tropical_bh) = crate::mamba3::positive::fresh_slots(
            self.gain,
            self.tropical,
            self.batch,
            self.nheads,
            device,
        );
        Mamba3DoubleSsdCache {
            ssm_bhpr,
            k_state_bumhr,
            v_state_buhp,
            rotation,
            log_precision_bh,
            tropical_bh,
        }
    }
}
