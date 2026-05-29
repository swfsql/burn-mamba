//! # Mamba-3 Inference Caches
//!
//! During autoregressive (token-by-token) generation, three pieces of state
//! must be preserved between calls:
//!
//! 1. **SSM hidden state** — `hₜ ∈ ℝ^{per_head_dim×state_rank}` per head, compressed context.
//! 2. **Previous K state** — `Bₜ₋₁` per rank `[batch, mimo_rank, nheads, state_rank]`,
//!    needed for the β term of the (double-ssd) trapezoidal recurrence.
//! 3. **Previous V state** — `xₜ₋₁` per head `[batch, nheads, per_head_dim]`,
//!    paired with k_state to reconstruct β Bₜ₋₁ ⊗ xₜ₋₁.
//! 4. **Cumulative RoPE angle** — the accumulated rotation angle up to position
//!    `t`, needed to correctly continue data-dependent rotary embeddings.
//!
//! Note: Mamba-3 has **no conv cache** (the short 1-dimensional convolution present in
//! Mamba-3 is removed; its role is absorbed by the trapezoidal discretization
//! and the learnable B/C biases).

use crate::mamba3::prelude::*;
use crate::utils::sanity::sanity as san;
use burn::backend::Backend;
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

    /// **Previous token's B per mimo rank** = `Bₜ₋₁[m]`.
    ///
    /// Used to reconstruct the β term: `β * sum_r Bₜ₋₁[m] ⊗ (xₜ₋₁ * mimo_x[m])`.
    /// For SISO (mimo_rank=1) this is shape `[batch, 1, nheads, state_rank]`.
    ///
    /// Shape: `[batch, mimo_rank, nheads, state_rank]`
    pub k_state_bmhr: Tensor<4>,

    /// **Previous token's x** = `xₜ₋₁`.
    ///
    /// Combined with `k_state_bmhr` and `mimo_x` to produce the β term.
    ///
    /// Shape: `[batch, nheads, per_head_dim]`
    pub v_state_bhp: Tensor<3>,

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
    /// Run the [`NaN`/`Inf` guards](crate::utils::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        san(&self.k_state_bmhr);
        san(&self.v_state_bhp);
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

    /// Which positional rotation the block uses ([`RotationKind`]); selects the
    /// accumulator variant — [`RotationState::Quaternion`] for
    /// [`RotationKind::Quaternion4D`], else [`RotationState::Angle`].
    #[config(default = "crate::mamba3::rotation::RotationKind::Complex2D")]
    pub rotation: RotationKind,

    /// Number of quaternion blocks (`rope_dim / 4`); only used for
    /// [`RotationKind::Quaternion4D`].
    #[config(default = 1)]
    pub num_quat_blocks: usize,
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
        }
    }

    /// Allocate zero/identity-initialised cache tensors on `device`.
    pub fn init(&self, device: &Device) -> Mamba3DoubleSsdCache {
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
            RotationKind::Quaternion4D => {
                RotationState::identity_quaternion(self.batch, self.nheads, self.num_quat_blocks, device)
            }
            RotationKind::Complex2D => {
                RotationState::zeros_angle(self.batch, self.nheads, self.num_rope_angles, device)
            }
        };
        Mamba3DoubleSsdCache {
            ssm_bhpr,
            k_state_bmhr,
            v_state_bhp,
            rotation,
        }
    }
}
