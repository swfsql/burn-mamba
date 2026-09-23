//! # Mamba-3 Caches (single-SSD pathway)
//!
//! The cache of [`crate::mamba3::mamba3::Mamba3::forward_single_ssd`] (the
//! single-pass SSD algorithm of the Triton SISO and Tilelang MIMO reference
//! kernels). Its fields mirror those of
//! [`Mamba3DoubleSsdCache`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache),
//! but the **SSM accumulator has different semantics** mid-sequence:
//!
//! - double-SSD: `ssm_bhpr` holds the trapezoidal hidden state
//!   `hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ` (at the default lag 1).
//! - single-SSD: `ssm_bhpr` holds the **trapezoid accumulator**
//!   `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ ⊗ xₜ`, with
//!   `scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)`. With it, `yₜ = Cₜᵀ h'ₜ` is correct at
//!   all positions except the `lag`-wide band before a tap is paid. The kernel
//!   patches the diagonal (lag 1) with an explicit `γₜ · (Cₜᵀ Bₜ) · xₜ` term,
//!   and `token_band` patches the wider band (lag `u`).
//!
//! The two states are equal at call boundaries, where caches are produced and
//! consumed (see [`crate::mamba3::cache`]). Under
//! [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None) the difference
//! disappears everywhere: `scaleₜ = γₜ`, so `h' ≡ h` at every position, and
//! neither cache has tap slots.
//!
//! The distinct type prevents a `forward_double_ssd` cache from going into
//! `forward_single_ssd` (or the reverse) mid-sequence, which would silently
//! corrupt the state.

use crate::mamba3::prelude::*;
use burn_stack::modules::sanity as san;
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba3SingleSsdCaches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer single-ssd caches for a complete Mamba-3 network.
#[derive(Module, Debug)]
pub struct Mamba3SingleSsdCaches {
    /// Per-layer caches. The length is the number of virtual layers.
    pub caches: Vec<Mamba3SingleSsdCache>,
}

/// Configuration / factory for [`Mamba3SingleSsdCaches`].
#[derive(Config, Debug)]
pub struct Mamba3SingleSsdCachesConfig {
    /// Number of cache slots (= number of virtual layers).
    pub n_caches: usize,

    /// Shared configuration that determines the shape of each cache.
    pub cache: Mamba3SingleSsdCacheConfig,
}

impl Mamba3SingleSsdCachesConfig {
    /// Convenience constructor from a block config.
    pub fn new_from_block_config(
        n_caches: usize,
        batch: usize,
        block_config: Mamba3Config,
    ) -> Self {
        Self {
            n_caches,
            cache: Mamba3SingleSsdCacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init(&self, device: &Device) -> Mamba3SingleSsdCaches {
        let caches = (0..self.n_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba3SingleSsdCaches { caches }
    }
}

// ---------------------------------------------------------------------------
// Mamba3SingleSsdCache  (state for a single layer)
// ---------------------------------------------------------------------------

/// The state of one Mamba-3 layer that runs the single-ssd algorithm.
///
/// The fields and shapes match
/// [`Mamba3DoubleSsdCache`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache).
/// The semantic difference is only in `ssm_bhpr` (see the module header).
#[derive(Module, Debug)]
pub struct Mamba3SingleSsdCache {
    /// **Single-SSD SSM accumulator** `h'ₜ`.
    ///
    /// Update rule: `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ · sumₘ Bₜ[m] ⊗ (xₜ ⊙ mimo_xₘ)`.
    /// Mid-sequence, it is different from the double-ssd `ssm_bhpr`.
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<4>,

    /// **The K of the tap FIFO** (post-RoPE, post-bias `B`), one slot per
    /// lagged position, **oldest first**.
    ///
    /// At the start of the next `forward_single_ssd` call, it seeds the
    /// deferred boundary β contributions
    /// `Σⱼ (1 − λⱼ) · Δⱼ · Bₚ₋ₗₐ₉₊ⱼ ⊗ xₚ₋ₗₐ₉₊ⱼ`. The previous call could not add
    /// them, because it did not know the `λ, Δ` of the `lag` positions that pay
    /// them.
    ///
    /// `None` under [`Trapezoid::None`], which has no boundary β term to defer.
    ///
    /// Shape: `[batch, tap_slots, mimo_rank, nheads, state_rank]`
    pub k_state_bumhr: Option<Tensor<5>>,

    /// **The x of the tap FIFO**, matching [`Self::k_state_bumhr`] slot for
    /// slot, and `None` with it. Pre-scaled by the decay accumulated since its
    /// own position, exactly as in the double-SSD cache.
    ///
    /// Shape: `[batch, tap_slots, nheads, per_head_dim]`
    pub v_state_buhp: Option<Tensor<4>>,

    /// **Cumulative data-dependent rotation** up to the current position
    /// ([`RotationState`]).
    ///
    /// The same value as the field of the double-ssd cache (the `From` impls
    /// move it across), so the two caches convert by field identity.
    pub rotation: RotationState,

    /// **The log-precision of the Kalman gate**, `ln Λ` after the last position
    /// (the same value as
    /// [`Mamba3DoubleSsdCache::log_precision_bh`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache::log_precision_bh)).
    ///
    /// Shape: `[batch, nheads]`
    pub log_precision_bh: Option<Tensor<2>>,

    /// **The tropical register** `c` after the last position (see
    /// [`Mamba3DoubleSsdCache::tropical_bh`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache::tropical_bh)).
    ///
    /// Shape: `[batch, nheads]`
    pub tropical_bh: Option<Tensor<2>>,
}

impl Mamba3SingleSsdCache {
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

/// Configuration / factory for a single [`Mamba3SingleSsdCache`].
#[derive(Config, Debug)]
pub struct Mamba3SingleSsdCacheConfig {
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

    /// MIMO rank. 1 = SISO.
    #[config(default = 1)]
    pub mimo_rank: usize,

    /// Number of RoPE angle pairs
    /// (see [`crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCacheConfig::num_rope_angles`]).
    pub num_rope_angles: usize,

    /// Which transition rotation the block uses (see
    /// [`crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCacheConfig::rotation`]).
    #[config(default = "crate::mamba3::rotation::RotationKind::Complex2D")]
    pub rotation: RotationKind,

    /// Number of quaternion blocks (`rope_dim / 4`). Only
    /// [`RotationKind::Quaternion4D`] / [`RotationKind::Rotor4D`] use it.
    #[config(default = 1)]
    pub num_quat_blocks: usize,

    /// The tap pattern of the block (see
    /// [`Mamba3DoubleSsdCacheConfig::trapezoid`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCacheConfig::trapezoid)).
    #[config(default = "crate::mamba3::trapezoid::Trapezoid::HorizontalCarryOver")]
    pub trapezoid: Trapezoid,

    /// Recurrence micro-steps per token (`u`). With [`Self::trapezoid`], it
    /// sets the depth of the tap FIFO (see [`Trapezoid::tap_lag`]).
    #[config(default = 1)]
    pub micro_steps: usize,

    /// The gain of the block. A Kalman gain keeps a `ln Λ` slot.
    #[config(default = "crate::mamba3::positive::Gain::Projected")]
    pub gain: crate::mamba3::positive::Gain,

    /// The tropical register of the block. `MaxPlus` keeps a `c` slot.
    #[config(default = "crate::mamba3::positive::Tropical::None")]
    pub tropical: crate::mamba3::positive::Tropical,
}

impl Mamba3SingleSsdCacheConfig {
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
    pub fn init(&self, device: &Device) -> Mamba3SingleSsdCache {
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
        Mamba3SingleSsdCache {
            ssm_bhpr,
            k_state_bumhr,
            v_state_buhp,
            rotation,
            log_precision_bh,
            tropical_bh,
        }
    }
}
