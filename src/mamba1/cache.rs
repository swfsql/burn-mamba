//! # Mamba-1 Inference Caches
//!
//! The state that the block carries between calls. In *training* or
//! *prefill*, [`Mamba1::forward`] processes the whole sequence at once. In
//! *decoding*, [`Mamba1::step`] processes one token per call. Both modes carry
//! the same two pieces of state:
//!
//! 1. **Convolution window** — the last `conv_kernel` pre-activation inputs to
//!    the depthwise Conv1d. With it, each step applies the causal filter
//!    without processing earlier tokens again.
//!
//! 2. **SSM hidden state** — the per-channel state matrix. It compresses the
//!    full past into a fixed size, for any number of tokens.
//!
//! Mirrors [`crate::mamba2::cache`]. The Mamba-1 state has no head axis.

use crate::mamba1::prelude::*;
use burn_stack::modules::sanity as san;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba1Cache  (state for a single layer)
// ---------------------------------------------------------------------------

/// The mutable state carried between calls for a **single** Mamba-1 layer.
#[derive(Module, Debug)]
pub struct Mamba1Cache {
    /// **Convolution rolling window.**
    ///
    /// The last `conv_kernel` feature vectors that went into the depthwise
    /// Conv1d. Each step removes the oldest column and appends the projection
    /// of the new token on the right.
    ///
    /// Shape: `[batch, d_inner, conv_kernel]`
    pub conv_bik: Tensor<3>,

    /// **SSM hidden state.**
    ///
    /// The O(d_inner·state_rank) compressed summary of all earlier tokens. The
    /// selective-scan recurrence updates it at each step.
    ///
    /// Shape: `[batch, d_inner, state_rank]`
    pub ssm_bir: Tensor<3>,
}

impl Mamba1Cache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.conv_bik);
        san(&self.ssm_bir);
    }
}

/// Configuration / factory for a single [`Mamba1Cache`].
#[derive(Config, Debug)]
pub struct Mamba1CacheConfig {
    /// Batch size.
    pub batch: usize,

    /// State rank — the latent dimension of the SSM hidden state.
    /// Corresponds to `state_rank` in [`Mamba1Config`].
    #[config(default = 16)]
    pub state_rank: usize,

    /// Causal convolution window length.  Corresponds to `conv_kernel` in
    /// [`Mamba1Config`].
    #[config(default = 4)]
    pub conv_kernel: usize,

    /// Inner (expanded) channel width `d_inner`.
    pub d_inner: usize,
}

impl Mamba1CacheConfig {
    /// Derive cache shapes from a Mamba-1 block configuration plus a batch
    /// size.
    pub fn new_from_block_config(batch: usize, block_config: Mamba1Config) -> Self {
        Self {
            batch,
            state_rank: block_config.state_rank,
            conv_kernel: block_config.conv_kernel,
            d_inner: block_config.d_inner(),
        }
    }

    /// Allocate zero-initialised cache tensors on `device`.
    ///
    /// Zeros are correct: a zero convolution window means "no previous tokens"
    /// (the causal zero padding), and a zero SSM state is the standard initial
    /// condition `h₀ = 0`.
    pub fn init(&self, device: &Device) -> Mamba1Cache {
        let conv_bik = Tensor::zeros([self.batch, self.d_inner, self.conv_kernel], device);
        let ssm_bir = Tensor::zeros([self.batch, self.d_inner, self.state_rank], device);
        Mamba1Cache { conv_bik, ssm_bir }
    }
}

// ---------------------------------------------------------------------------
// Mamba1Caches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer caches for a complete Mamba-1 network.
///
/// Every `step` of the family-generic [`burn_stack::modules::Layers`] stack
/// takes and returns one [`Mamba1Caches`]. Each element is the cache of one
/// (virtual) layer of the network.
#[derive(Module, Debug)]
pub struct Mamba1Caches {
    /// Per-layer caches.
    ///
    /// Length: `n_caches`, the number of *virtual* layers. With shared
    /// weights (a layer schedule), it can be larger than the number of *real*
    /// weight layers.
    pub caches: Vec<Mamba1Cache>,
}

/// Configuration / factory for [`Mamba1Caches`].
#[derive(Config, Debug)]
pub struct Mamba1CachesConfig {
    /// Number of cache slots: the number of virtual layers in the network (one
    /// cache per layer, also when layers share weights).
    pub n_caches: usize,

    /// Shared configuration that determines the shape of each individual
    /// cache tensor.
    pub cache: Mamba1CacheConfig,
}

impl Mamba1CachesConfig {
    /// Convenience constructor that derives cache shapes directly from a
    /// [`Mamba1Config`] block configuration.
    pub fn new_from_block_config(
        n_caches: usize,
        batch: usize,
        block_config: Mamba1Config,
    ) -> Self {
        Self {
            n_caches,
            cache: Mamba1CacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init(&self, device: &Device) -> Mamba1Caches {
        let caches = (0..self.n_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba1Caches { caches }
    }
}

impl Mamba1Caches {
    /// Number of per-layer caches.
    pub fn caches_len(&self) -> usize {
        self.caches.len()
    }

    /// Wrap a vector of per-layer caches.
    pub fn from_vec(vec: Vec<Mamba1Cache>) -> Self {
        Self { caches: vec }
    }

    /// Wrap each per-layer cache in `Some`, so the layer loop can `take` it
    /// without a clone.
    pub fn into_options(self) -> Vec<Option<Mamba1Cache>> {
        self.caches.into_iter().map(Some).collect()
    }

    /// Inverse of [`Self::into_options`]: unwrap each slot and re-bundle.
    pub fn from_options(options: Vec<Option<Mamba1Cache>>) -> Self {
        let caches = options.into_iter().map(Option::unwrap).collect();
        Self::from_vec(caches)
    }
}
