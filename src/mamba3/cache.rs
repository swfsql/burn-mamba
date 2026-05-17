//! # Mamba-3 Inference Caches
//!
//! During autoregressive (token-by-token) generation, three pieces of state
//! must be preserved between calls:
//!
//! 1. **SSM hidden state** — `hₜ ∈ ℝ^{P×N}` per head, compressed context.
//! 2. **Previous K state** — `B_{t-1}` per rank `[batch, mimo_rank, nheads, state_rank]`,
//!    needed for the β term of the trapezoidal recurrence.
//! 3. **Previous V state** — `x_{t-1}` per head `[batch, nheads, per_head_dim]`,
//!    paired with k_state to reconstruct β B_{t-1} ⊗ x_{t-1}.
//! 4. **Cumulative RoPE angle** — the accumulated rotation angle up to position
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
/// All tensors are updated at every call to [`crate::mamba3::mamba3::Mamba3::step`].
#[derive(Module, Debug)]
pub struct Mamba3Cache<B: Backend> {
    /// **SSM hidden state** `hₜ`.
    ///
    /// Updated via the trapezoidal recurrence:
    /// `hₜ = αₜ hₜ₋₁ + βₜ (sum_r K_{t-1}[r] ⊗ (V_{t-1} * mimo_x[r])) + γₜ (sum_r Bₜ[r] ⊗ (xₜ * mimo_x[r]))`
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<B, 4>,

    /// **Previous token's B per rank** = `B_{t-1}[r]`.
    ///
    /// Used to reconstruct the β term: `β * sum_r B_{t-1}[r] ⊗ (x_{t-1} * mimo_x[r])`.
    /// For SISO (mimo_rank=1) this is shape `[batch, 1, nheads, state_rank]`.
    ///
    /// Shape: `[batch, mimo_rank, nheads, state_rank]`
    pub k_state_brhn: Tensor<B, 4>,

    /// **Previous token's x** = `x_{t-1}`.
    ///
    /// Combined with `k_state_brhn` and `mimo_x` to produce the β term.
    ///
    /// Shape: `[batch, nheads, per_head_dim]`
    pub v_state_bhp: Tensor<B, 3>,

    /// **Cumulative data-dependent RoPE angle** up to the current position.
    ///
    /// Each step updates: `cum_angle_{t} = cum_angle_{t-1} + Δ_t · tanh(θ_t) · π`
    ///
    /// Starts at zero for fresh sequences; continued across calls for streaming.
    ///
    /// Shape: `[batch, nheads, num_rope_angles]`
    pub cum_angle_bhr: Tensor<B, 3>,
}

impl<B: Backend> Mamba3Cache<B> {
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        san(&self.k_state_brhn);
        san(&self.v_state_bhp);
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

    /// MIMO rank R.  1 = SISO.
    #[config(default = 1)]
    pub mimo_rank: usize,

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
            mimo_rank: block_config.mimo_rank,
            num_rope_angles: block_config.num_rope_angles(),
        }
    }

    /// Allocate zero-initialised cache tensors on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba3Cache<B> {
        let ssm_bhpr = Tensor::zeros(
            [self.batch, self.nheads, self.per_head_dim, self.state_rank],
            device,
        );
        let k_state_brhn = Tensor::zeros(
            [self.batch, self.mimo_rank, self.nheads, self.state_rank],
            device,
        );
        let v_state_bhp = Tensor::zeros(
            [self.batch, self.nheads, self.per_head_dim],
            device,
        );
        let cum_angle_bhr = Tensor::zeros(
            [self.batch, self.nheads, self.num_rope_angles],
            device,
        );
        Mamba3Cache { ssm_bhpr, k_state_brhn, v_state_bhp, cum_angle_bhr }
    }
}
