//! # Mamba-2 Inference Caches
//!
//! The state that the block keeps between calls. In *training* or *prefill*,
//! the full sequence is available at once and [`Mamba2::forward`] uses the
//! chunked SSD algorithm. In *decoding*, [`Mamba2::step`] processes one token
//! per call with the pure recurrent form:
//!
//! ```text
//!   hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ        (state update)
//!   yₜ = Cₜᵀ hₜ + D xₜ            (output)
//! ```
//!
//! Each layer keeps two pieces of state:
//!
//! 1. **Convolution cache** — the last `conv_kernel` inputs to the depthwise
//!    Conv1d. With it, each decoding step applies the causal filter without
//!    processing earlier tokens again.
//!
//! 2. **SSM hidden state** — the matrix `hₜ ∈ ℝ^{per_head_dim×state_rank}` per
//!    head. It compresses the full past into a fixed size, for any number of
//!    tokens. This is the memory advantage of SSMs over attention: the
//!    KV-cache of a Transformer grows linearly with the sequence length, but
//!    the SSM state is always O(per_head_dim·state_rank).

use crate::mamba2::prelude::*;
use burn_stack::modules::sanity as san;
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba2Caches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer caches for a complete Mamba-2 network.
///
/// Every `step` of the family-generic [`burn_stack::modules::Layers`] stack
/// takes and returns one [`Mamba2Caches`]. Each element of `caches` is the
/// cache of one (virtual) layer of the network.
#[derive(Module, Debug)]
pub struct Mamba2Caches {
    /// Per-layer caches.
    ///
    /// Length: `n_caches`, the number of *virtual* layers. With shared weights
    /// (a layer schedule), it can be larger than the number of *real* weight
    /// layers.
    pub caches: Vec<Mamba2Cache>,
}

/// Configuration / factory for [`Mamba2Caches`].
#[derive(Config, Debug)]
pub struct Mamba2CachesConfig {
    /// Number of cache slots: the number of virtual layers in the network (one
    /// cache per layer, also when layers share weights).
    pub n_caches: usize,

    /// Shared configuration that determines the shape of each individual
    /// cache tensor.
    pub cache: Mamba2CacheConfig,
}

impl Mamba2CachesConfig {
    /// Convenience constructor that derives cache shapes directly from a
    /// [`Mamba2Config`] block configuration.
    pub fn new_from_block_config(
        n_caches: usize,
        batch: usize,
        block_config: Mamba2Config,
    ) -> Self {
        Self {
            n_caches,
            cache: Mamba2CacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init(&self, device: &Device) -> Mamba2Caches {
        let caches = (0..self.n_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba2Caches { caches }
    }
}

// ---------------------------------------------------------------------------
// Mamba2Cache  (state for a single layer)
// ---------------------------------------------------------------------------

/// The state carried between calls for a **single** Mamba-2 layer.
///
/// Each call to [`Mamba2::step`] or [`Mamba2::forward`] returns a new cache
/// with both tensors updated.
#[derive(Module, Debug)]
pub struct Mamba2Cache {
    /// **Convolution rolling window.**
    ///
    /// The last `conv_kernel` pre-activation feature vectors that went into
    /// the depthwise Conv1d. Each step removes the oldest column and appends
    /// the projection of the new token on the right.
    ///
    /// Shape: `[batch, conv_dim, conv_kernel]`
    ///   - `conv_dim  = d_inner + 2 · ngroups · state_rank`
    ///   - `conv_kernel` is typically 4
    pub conv_bvk: Tensor<3>,

    /// **SSM hidden state** `hₜ`.
    ///
    /// The O(per_head_dim·state_rank) compressed summary of all earlier
    /// tokens. Each decoding step applies `hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ`.
    ///
    /// The layout `[…, per_head_dim, state_rank]` is the transpose of the
    /// paper's `hₜ ∈ ℝ^{state_rank×per_head_dim}`. The content is the same.
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<4>,
}

impl Mamba2Cache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.conv_bvk);
        san(&self.ssm_bhpr);
    }
}

/// Configuration / factory for a single [`Mamba2Cache`].
#[derive(Config, Debug)]
pub struct Mamba2CacheConfig {
    /// Batch size.
    pub batch: usize,

    /// `state_rank` — the number of latent dimensions in the SSM hidden
    /// state.  Corresponds to `state_rank` in [`Mamba2Config`].
    #[config(default = 128)]
    pub state_rank: usize,

    /// Causal convolution window length.  Corresponds to `conv_kernel` in
    /// [`Mamba2Config`].
    #[config(default = 4)]
    pub conv_kernel: usize,

    /// Number of channels entering (and leaving) the depthwise convolution.
    /// Equal to `d_inner + 2 · ngroups · state_rank`.
    pub conv_dim: usize,

    /// Head dimension `per_head_dim`.  Corresponds to `per_head_dim` in [`Mamba2Config`].
    #[config(default = 64)]
    pub per_head_dim: usize,

    /// Number of SSM heads `nheads`.
    pub nheads: usize,
}

impl Mamba2CacheConfig {
    /// Derive cache shapes from a Mamba-2 block configuration plus a batch
    /// size.
    pub fn new_from_block_config(batch: usize, block_config: Mamba2Config) -> Self {
        Self {
            batch,
            state_rank: block_config.state_rank,
            conv_kernel: block_config.conv_kernel,
            conv_dim: block_config.conv_dim(),
            per_head_dim: block_config.per_head_dim,
            nheads: block_config.nheads(),
        }
    }

    /// Allocate zero-initialised cache tensors on `device`.
    ///
    /// Zeros are correct:
    /// - A zero convolution cache means "no previous tokens" (the causal zero
    ///   padding).
    /// - A zero SSM state is the standard initial condition `h₀ = 0`. If the
    ///   block has a learnable initial state, [`Mamba2::forward`] adds it to
    ///   this state.
    pub fn init(&self, device: &Device) -> Mamba2Cache {
        let conv_bvk = Tensor::zeros(
            Shape::new([self.batch, self.conv_dim, self.conv_kernel]),
            device,
        );
        let ssm_bhpr = Tensor::zeros(
            Shape::new([self.batch, self.nheads, self.per_head_dim, self.state_rank]),
            device,
        );
        Mamba2Cache { conv_bvk, ssm_bhpr }
    }
}

impl Mamba2Caches {
    /// Number of per-layer caches.
    pub fn caches_len(&self) -> usize {
        self.caches.len()
    }

    /// Wrap a vector of per-layer caches.
    pub fn from_vec(vec: Vec<Mamba2Cache>) -> Self {
        Self { caches: vec }
    }

    /// Wrap each per-layer cache in `Some`, so the layer loop can `take` it
    /// without a clone.
    pub fn into_options(self) -> Vec<Option<Mamba2Cache>> {
        self.caches.into_iter().map(Some).collect()
    }

    /// Inverse of [`Self::into_options`]: unwrap each slot and re-bundle.
    pub fn from_options(options: Vec<Option<Mamba2Cache>>) -> Self {
        let caches = options.into_iter().map(Option::unwrap).collect();
        Self::from_vec(caches)
    }
}
