//! # Mamba-3 Inference Caches
//!
//! During autoregressive (token-by-token) generation, three pieces of state
//! must be preserved between calls:
//!
//! 1. **SSM hidden state** — `hₜ ∈ ℝ^{per_head_dim×state_rank}` per head, compressed context.
//! 2. **Previous K state** — the last `lag` positions' `B`, oldest first
//!    `[batch, lag, mimo_rank, nheads, state_rank]`, needed for the β term of
//!    the (double-ssd) trapezoidal recurrence.
//! 3. **Previous V state** — the matching `x`
//!    `[batch, lag, nheads, per_head_dim]`, paired with k_state to reconstruct
//!    `β Bₚ₋ₗₐ₉ ⊗ xₚ₋ₗₐ₉`.
//!    Both are the trapezoid's **tap slots** and exist only when the block's
//!    [`Trapezoid`](crate::mamba3::trapezoid::Trapezoid) has a β tap; the FIFO is
//!    as deep as its [`tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag)
//!    (`1` for the default, `u` for
//!    [`Trapezoid::Vertical`](crate::mamba3::trapezoid::Trapezoid::Vertical)).
//! 4. **Cumulative RoPE angle** — the accumulated rotation angle up to position
//!    `t`, needed to correctly continue data-dependent rotary embeddings.
//!
//! Note: Mamba-3 has **no conv cache** (the short 1-dimensional convolution present in
//! Mamba-3 is removed; its role is absorbed by the trapezoidal discretization
//! and the learnable B/C biases).

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
    /// Per-layer caches.  Length equals the number of virtual layers.
    pub caches: Vec<Mamba3DoubleSsdCache>,
}

/// Configuration / factory for [`Mamba3DoubleSsdCaches`].
#[derive(Config, Debug)]
pub struct Mamba3DoubleSsdCachesConfig {
    /// Number of cache slots (= number of virtual layers).
    pub n_real_caches: usize,

    /// Shared configuration that determines the shape of each cache.
    pub cache: Mamba3DoubleSsdCacheConfig,
}

impl Mamba3DoubleSsdCachesConfig {
    /// Convenience constructor from a block config.
    pub fn new_from_block_config(
        n_real_caches: usize,
        batch: usize,
        block_config: Mamba3Config,
    ) -> Self {
        Self {
            n_real_caches,
            cache: Mamba3DoubleSsdCacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init(&self, device: &Device) -> Mamba3DoubleSsdCaches {
        let caches = (0..self.n_real_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba3DoubleSsdCaches { caches }
    }
}

// ---------------------------------------------------------------------------
// Mamba3DoubleSsdCache  (state for a single layer)
// ---------------------------------------------------------------------------

/// The mutable state carried between decoding steps for a **single** Mamba-3 layer.
///
/// All tensors are updated at every call to [`crate::mamba3::mamba3::Mamba3::step`].
#[derive(Module, Debug)]
pub struct Mamba3DoubleSsdCache {
    /// **SSM hidden state** `hₜ`.
    ///
    /// Updated via the (double-ssd) trapezoidal recurrence:
    /// `hₜ = αₜ hₜ₋₁ + βₜ (sumₘ Kₜ₋₁[m] ⊗ (Vₜ₋₁ * mimo_x[m])) + γₜ (sumₘ Bₜ[m] ⊗ (xₜ * mimo_x[m]))`
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<4>,

    /// **The tap FIFO's B**, one slot per lagged position, **oldest first**.
    ///
    /// Used to reconstruct the β term: `β * sumₘ Bₚ₋ₗₐ₉[m] ⊗ (xₚ₋ₗₐ₉ * mimo_x[m])`.
    /// Stored **as rotated**, at its own position: the relative-rotation
    /// factoring `C̄ₜᵀB̄ₛ` then reconstructs the transport, so no rotation is
    /// re-applied on the next call.
    ///
    /// The trapezoid's tap slots: `None` — allocated nowhere, not zeroed — under
    /// [`Trapezoid::None`], which has no β term. Present exactly when
    /// [`Self::v_state_buhp`] is (see [`Trapezoid::tap_slots`]).
    ///
    /// Shape: `[batch, tap_slots, mimo_rank, nheads, state_rank]`
    pub k_state_bumhr: Option<Tensor<5>>,

    /// **The tap FIFO's x**, matching [`Self::k_state_bumhr`] slot for slot, and
    /// `None` with it.
    ///
    /// Each slot is **pre-scaled by the decay it has accumulated since its own
    /// position** (`helpers::tail_decay`), which is what carries a lag-`u` tap's
    /// gap transport across the call boundary. At `lag = 1` that product is
    /// empty, so the slot is the plain `xₜ₋₁`.
    ///
    /// Shape: `[batch, tap_slots, nheads, per_head_dim]`
    pub v_state_buhp: Option<Tensor<4>>,

    /// **Cumulative data-dependent rotation** up to the current position
    /// ([`RotationState`]): the abelian RoPE angle for
    /// [`Complex2D`](crate::mamba3::rotation::RotationKind::Complex2D) (each step
    /// `cum_angleₜ = cum_angleₜ₋₁ + Δₜ · tanh(θₜ) · π`), or the cumulative unit
    /// quaternion for [`Quaternion4D`](crate::mamba3::rotation::RotationKind::Quaternion4D).
    ///
    /// Starts at the identity for fresh sequences; continued across calls for
    /// streaming.
    pub rotation: RotationState,
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
    /// (rounded down to even via `Mamba3Config::rope_dim`).
    pub num_rope_angles: usize,

    /// Which transition rotation the block uses ([`RotationKind`]); selects the
    /// accumulator variant — [`RotationState::Quaternion`] for
    /// [`RotationKind::Quaternion4D`], [`RotationState::Rotor`] for
    /// [`RotationKind::Rotor4D`], else [`RotationState::Angle`].
    #[config(default = "crate::mamba3::rotation::RotationKind::Complex2D")]
    pub rotation: RotationKind,

    /// Number of quaternion blocks (`rope_dim / 4`); only used for
    /// [`RotationKind::Quaternion4D`] / [`RotationKind::Rotor4D`].
    #[config(default = 1)]
    pub num_quat_blocks: usize,

    /// The block's tap pattern ([`Trapezoid`]) — it decides whether the tap
    /// slots exist at all, and with [`Self::micro_steps`] how many there are
    /// (see [`Trapezoid::tap_slots`]).
    #[config(default = "crate::mamba3::trapezoid::Trapezoid::HorizontalCarryOver")]
    pub trapezoid: Trapezoid,

    /// Recurrence micro-steps per token (`u`); only the lag-`u` tap patterns
    /// read it (see [`Trapezoid::tap_lag`]).
    #[config(default = 1)]
    pub micro_steps: usize,
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
        Mamba3DoubleSsdCache {
            ssm_bhpr,
            k_state_bumhr,
            v_state_buhp,
            rotation,
        }
    }
}
