//! # Mamba-3 Single-pass SSD Inference Cache
//!
//! The cache used by [`crate::mamba3::mamba3::Mamba3::forward_single_ssd`]
//! (the single-pass SSD algorithm — see the Triton SISO and Tilelang MIMO
//! reference kernels).
//! The four tensor fields mirror those of [`Mamba3Cache`] but their
//! **SSM accumulator carries different semantics**:
//!
//! - [`Mamba3Cache`]: `ssm_bhpr` holds the double-ssd trapezoidal hidden state
//!   `hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ`.
//! - [`Mamba3SingleSsdCache`]: `ssm_bhpr` holds the **trapezoid accumulator** `h'ₜ`
//!   defined by `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ ⊗ xₜ`, where
//!   `scaleₜ = γₜ + (1 − λₜ₊₁) · Δₜ₊₁`. The single-ssd form gives the correct output
//!   `yₜ = Cₜᵀ h'ₜ` for all positions except the diagonal (s = t), which is
//!   patched by an explicit `γₜ · (Cₜᵀ Bₜ) · xₜ` correction term in the kernel.
//!
//! Because the two accumulators differ, the two caches are not interchangeable.
//! The distinct type prevents accidentally feeding a `forward_double_ssd` cache into
//! `forward_single_ssd` (or vice versa) mid-sequence — that would silently corrupt state.

use crate::mamba3::prelude::*;
use crate::modules::sanity as san;
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba3SingleSsdCaches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer single-ssd form caches for a complete Mamba-3 network.
#[derive(Module, Debug)]
pub struct Mamba3SingleSsdCaches {
    /// Per-layer caches. Length equals the number of virtual layers.
    pub caches: Vec<Mamba3SingleSsdCache>,
}

/// Configuration / factory for [`Mamba3SingleSsdCaches`].
#[derive(Config, Debug)]
pub struct Mamba3SingleSsdCachesConfig {
    /// Number of cache slots (= number of virtual layers).
    pub n_real_caches: usize,

    /// Shared configuration that determines the shape of each cache.
    pub cache: Mamba3SingleSsdCacheConfig,
}

impl Mamba3SingleSsdCachesConfig {
    /// Convenience constructor from a block config.
    pub fn new_from_block_config(
        n_real_caches: usize,
        batch: usize,
        block_config: Mamba3Config,
    ) -> Self {
        Self {
            n_real_caches,
            cache: Mamba3SingleSsdCacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init(&self, device: &Device) -> Mamba3SingleSsdCaches {
        let caches = (0..self.n_real_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba3SingleSsdCaches { caches }
    }
}

// ---------------------------------------------------------------------------
// Mamba3SingleSsdCache  (state for a single layer)
// ---------------------------------------------------------------------------

/// Mutable state for a single Mamba-3 layer running the single-ssd form algorithm.
///
/// Tensor shapes match [`Mamba3Cache`]. The semantic difference lives entirely
/// in `ssm_bhpr` (see the module-level documentation).
#[derive(Module, Debug)]
pub struct Mamba3SingleSsdCache {
    /// **SingleSsd-form SSM accumulator** `h'ₜ`.
    ///
    /// Update rule: `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ · sumₘ Bₜ[m] ⊗ (xₜ ⊙ mimo_xₘ)`.
    /// Different from `Mamba3Cache::ssm_bhpr`.
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<4>,

    /// **Previous token's K per mimo rank** = post-RoPE, post-bias `Bₜ₋₁[m]`.
    ///
    /// Used at the start of the next forward_single_ssd call to seed the boundary β
    /// contribution `(1 − λ₀) · Δ₀ · Bₜ₋₁ ⊗ xₜ₋₁` (which the previous call could
    /// not yet add because it did not know `λ₀, Δ₀`).
    ///
    /// Shape: `[batch, mimo_rank, nheads, state_rank]`
    pub k_state_bmhr: Tensor<4>,

    /// **Previous token's x** = `xₜ₋₁`.
    ///
    /// Paired with [`Self::k_state_bmhr`] to form the boundary β term.
    ///
    /// Shape: `[batch, nheads, per_head_dim]`
    pub v_state_bhp: Tensor<3>,

    /// **Cumulative data-dependent rotation** up to the current position
    /// ([`RotationState`]).
    ///
    /// Same role as in [`Mamba3Cache`]: continued across calls for streaming.
    /// Carries the same value as the double-ssd cache's field (the `From` impls
    /// move it across), so the two caches still inter-convert by field identity.
    pub rotation: RotationState,
}

impl Mamba3SingleSsdCache {
    /// Run the [`NaN`/`Inf` guards](crate::utils::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        san(&self.k_state_bmhr);
        san(&self.v_state_bhp);
        self.rotation.sanity();
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

    /// Which positional rotation the block uses (see
    /// [`crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCacheConfig::rotation`]).
    #[config(default = "crate::mamba3::rotation::RotationKind::Complex2D")]
    pub rotation: RotationKind,

    /// Number of quaternion blocks (`rope_dim / 4`); only used for
    /// [`RotationKind::Quaternion4D`].
    #[config(default = 1)]
    pub num_quat_blocks: usize,
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
        }
    }

    /// Allocate zero/identity-initialised cache tensors on `device`.
    pub fn init(&self, device: &Device) -> Mamba3SingleSsdCache {
        let ssm_bhpr = Tensor::zeros(
            [self.batch, self.nheads, self.per_head_dim, self.state_rank],
            device,
        );
        let k_state_bmhr = Tensor::zeros(
            [self.batch, self.mimo_rank, self.nheads, self.state_rank],
            device,
        );
        let v_state_bhp = Tensor::zeros([self.batch, self.nheads, self.per_head_dim], device);
        let rotation = match self.rotation {
            RotationKind::Quaternion4D => RotationState::identity_quaternion(
                self.batch,
                self.nheads,
                self.num_quat_blocks,
                device,
            ),
            RotationKind::Complex2D => {
                RotationState::zeros_angle(self.batch, self.nheads, self.num_rope_angles, device)
            }
        };
        Mamba3SingleSsdCache {
            ssm_bhpr,
            k_state_bmhr,
            v_state_bhp,
            rotation,
        }
    }
}
