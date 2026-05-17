//! # Mamba-3 Inference Caches
//!
//! During autoregressive (token-by-token) generation, three pieces of state
//! must be preserved between calls:
//!
//! 1. **SSM hidden state** — `hₜ ∈ ℝ^{P×N}` per head, compressed context.
//! 2. **Previous Bx product** — `B_{t-1} xₜ₋₁ᵀ`, needed for the β term of
//!    the trapezoidal recurrence.
//! 3. **Cumulative RoPE angle** — the accumulated rotation angle up to position
//!    `t`, needed to correctly continue data-dependent rotary embeddings.
//!
//! Note: Mamba-3 has **no conv cache** (the short 1-D convolution present in
//! Mamba-2 is removed; its role is absorbed by the trapezoidal discretization
//! and the learnable B/C biases).

use crate::mamba3::prelude::*;
use crate::utils::sanity::sanity as san;
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba3Caches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer caches for a complete Mamba-3 network.
#[derive(Module, Debug)]
pub struct Mamba3Caches<B: Backend> {
    /// Per-layer caches.  Length equals the number of virtual layers.
    pub caches: Vec<Mamba3Cache<B>>,
}

/// Configuration / factory for [`Mamba3Caches`].
#[derive(Config, Debug)]
pub struct Mamba3CachesConfig {
    /// Number of cache slots (= number of virtual layers).
    pub n_real_caches: usize,

    /// Shared configuration that determines the shape of each cache.
    pub cache: Mamba3CacheConfig,
}

impl Mamba3CachesConfig {
    /// Convenience constructor from a block config.
    pub fn new_from_block_config(
        n_real_caches: usize,
        batch: usize,
        block_config: Mamba3Config,
    ) -> Self {
        Self {
            n_real_caches,
            cache: Mamba3CacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba3Caches<B> {
        let caches = (0..self.n_real_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba3Caches { caches }
    }
}

// ---------------------------------------------------------------------------
// Mamba3Cache  (state for a single layer)
// ---------------------------------------------------------------------------

/// The mutable state carried between decoding steps for a **single** Mamba-3 layer.
///
/// All three tensors are updated at every call to [`crate::mamba3::mamba3::Mamba3::step`].
#[derive(Module, Debug)]
pub struct Mamba3Cache<B: Backend> {
    /// **SSM hidden state** `hₜ`.
    ///
    /// Updated via the trapezoidal recurrence:
    /// `hₜ = αₜ hₜ₋₁ + βₜ prev_Bx + γₜ Bₜ xₜᵀ`
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<B, 4>,

    /// **Previous token's B⊗x outer product** = `B_{t-1} xₜ₋₁ᵀ`.
    ///
    /// Required for the `β` term of the trapezoidal recurrence.
    /// At the start of a sequence, this is zero.
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub prev_bx_bhpr: Tensor<B, 4>,

    /// **Cumulative data-dependent RoPE angle** up to the current position.
    ///
    /// Each step updates: `cum_angle_{t} = cum_angle_{t-1} - Δ_t · θ_t`
    ///
    /// Starts at zero for fresh sequences; continued across calls for streaming.
    ///
    /// Shape: `[batch, nheads, num_rope_angles]`
    pub cum_angle_bhr: Tensor<B, 3>,
}

impl<B: Backend> Mamba3Cache<B> {
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        san(&self.prev_bx_bhpr);
        san(&self.cum_angle_bhr);
    }
}

/// Configuration / factory for a single [`Mamba3Cache`].
#[derive(Config, Debug)]
pub struct Mamba3CacheConfig {
    /// Batch size.
    pub batch: usize,

    /// State rank N.
    #[config(default = 128)]
    pub state_rank: usize,

    /// Head dimension P.
    #[config(default = 64)]
    pub per_head_dim: usize,

    /// Number of SSM heads H.
    pub nheads: usize,

    /// Number of RoPE angle pairs = state_rank / 2.
    pub num_rope_angles: usize,
}

impl Mamba3CacheConfig {
    /// Derive cache shapes from a Mamba-3 block configuration plus a batch size.
    pub fn new_from_block_config(batch: usize, block_config: Mamba3Config) -> Self {
        Self {
            batch,
            state_rank: block_config.state_rank,
            per_head_dim: block_config.per_head_dim,
            nheads: block_config.nheads(),
            num_rope_angles: block_config.num_rope_angles(),
        }
    }

    /// Allocate zero-initialised cache tensors on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba3Cache<B> {
        let ssm_bhpr = Tensor::zeros(
            [self.batch, self.nheads, self.per_head_dim, self.state_rank],
            device,
        );
        let prev_bx_bhpr = Tensor::zeros(
            [self.batch, self.nheads, self.per_head_dim, self.state_rank],
            device,
        );
        let cum_angle_bhr = Tensor::zeros(
            [self.batch, self.nheads, self.num_rope_angles],
            device,
        );
        Mamba3Cache { ssm_bhpr, prev_bx_bhpr, cum_angle_bhr }
    }
}
