//! # Mamba-3 Merged-Form Inference Cache (`forward2` path)
//!
//! The cache used by [`crate::mamba3::mamba3::Mamba3::forward2`] (the single-pass,
//! "merged"/trapezoidal-fused SSD algorithm — see the Triton SISO and Tilelang MIMO
//! reference kernels). The four tensor fields mirror those of [`Mamba3Cache`] but
//! their **SSM accumulator carries different semantics**:
//!
//! - [`Mamba3Cache`]: `ssm_bhpr` holds the original trapezoidal hidden state
//!   `hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ`.
//! - [`Mamba3MergedCache`]: `ssm_bhpr` holds the **merged accumulator** `h'ₜ`
//!   defined by `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ ⊗ xₜ`, where
//!   `scaleₜ = γₜ + (1 − λₜ₊₁) · Δₜ₊₁`. The merged form gives the correct output
//!   `yₜ = Cₜᵀ h'ₜ` for all positions except the diagonal (s = t), which is
//!   patched by an explicit `γₜ · (Cₜᵀ Bₜ) · xₜ` correction term in the kernel.
//!
//! Because the two accumulators differ, the two caches are not interchangeable.
//! The distinct type prevents accidentally feeding a `forward` cache into
//! `forward2` (or vice versa) mid-sequence — that would silently corrupt state.

use crate::mamba3::prelude::*;
use crate::utils::sanity::sanity as san;
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba3MergedCaches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer merged-form caches for a complete Mamba-3 network.
#[derive(Module, Debug)]
pub struct Mamba3MergedCaches<B: Backend> {
    /// Per-layer caches. Length equals the number of virtual layers.
    pub caches: Vec<Mamba3MergedCache<B>>,
}

/// Configuration / factory for [`Mamba3MergedCaches`].
#[derive(Config, Debug)]
pub struct Mamba3MergedCachesConfig {
    /// Number of cache slots (= number of virtual layers).
    pub n_real_caches: usize,

    /// Shared configuration that determines the shape of each cache.
    pub cache: Mamba3MergedCacheConfig,
}

impl Mamba3MergedCachesConfig {
    /// Convenience constructor from a block config.
    pub fn new_from_block_config(
        n_real_caches: usize,
        batch: usize,
        block_config: Mamba3Config,
    ) -> Self {
        Self {
            n_real_caches,
            cache: Mamba3MergedCacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba3MergedCaches<B> {
        let caches = (0..self.n_real_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba3MergedCaches { caches }
    }
}

// ---------------------------------------------------------------------------
// Mamba3MergedCache  (state for a single layer)
// ---------------------------------------------------------------------------

/// Mutable state for a single Mamba-3 layer running the merged-form algorithm.
///
/// Tensor shapes match [`Mamba3Cache`]. The semantic difference lives entirely
/// in `ssm_bhpr` (see the module-level documentation).
#[derive(Module, Debug)]
pub struct Mamba3MergedCache<B: Backend> {
    /// **Merged-form SSM accumulator** `h'ₜ`.
    ///
    /// Update rule: `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ · sumₘ Bₜ[m] ⊗ (xₜ ⊙ mimo_xₘ)`.
    /// Different from `Mamba3Cache::ssm_bhpr`.
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<B, 4>,

    /// **Previous token's K per mimo rank** = post-RoPE, post-bias `Bₜ₋₁[m]`.
    ///
    /// Used at the start of the next forward2 call to seed the boundary β
    /// contribution `(1 − λ₀) · Δ₀ · Bₜ₋₁ ⊗ xₜ₋₁` (which the previous call could
    /// not yet add because it did not know `λ₀, Δ₀`).
    ///
    /// Shape: `[batch, mimo_rank, nheads, state_rank]`
    pub k_state_bmhr: Tensor<B, 4>,

    /// **Previous token's x** = `xₜ₋₁`.
    ///
    /// Paired with [`Self::k_state_bmhr`] to form the boundary β term.
    ///
    /// Shape: `[batch, nheads, per_head_dim]`
    pub v_state_bhp: Tensor<B, 3>,

    /// **Cumulative data-dependent RoPE angle** up to the current position.
    ///
    /// Same role as in [`Mamba3Cache`]: continued across calls for streaming.
    ///
    /// Shape: `[batch, nheads, num_rope_angles]`
    pub cum_angle_bha: Tensor<B, 3>,
}

impl<B: Backend> Mamba3MergedCache<B> {
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        san(&self.k_state_bmhr);
        san(&self.v_state_bhp);
        san(&self.cum_angle_bha);
    }
}

/// Configuration / factory for a single [`Mamba3MergedCache`].
#[derive(Config, Debug)]
pub struct Mamba3MergedCacheConfig {
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

    /// Number of RoPE angle pairs (see [`Mamba3CacheConfig::num_rope_angles`]).
    pub num_rope_angles: usize,
}

impl Mamba3MergedCacheConfig {
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba3MergedCache<B> {
        let ssm_bhpr = Tensor::zeros(
            [self.batch, self.nheads, self.per_head_dim, self.state_rank],
            device,
        );
        let k_state_bmhr = Tensor::zeros(
            [self.batch, self.mimo_rank, self.nheads, self.state_rank],
            device,
        );
        let v_state_bhp = Tensor::zeros([self.batch, self.nheads, self.per_head_dim], device);
        let cum_angle_bha = Tensor::zeros([self.batch, self.nheads, self.num_rope_angles], device);
        Mamba3MergedCache {
            ssm_bhpr,
            k_state_bmhr,
            v_state_bhp,
            cum_angle_bha,
        }
    }
}
