//! # Mamba-1 Inference Caches
//!
//! State carried between calls during autoregressive (token-by-token)
//! generation.  During *training* or *prefill* the whole sequence is processed
//! at once by [`Mamba1::forward`]; during *decoding* one token is processed per
//! call by [`Mamba1::step`].  Both modes thread the same two pieces of state:
//!
//! 1. **Convolution window** — the last `conv_kernel` pre-activation inputs to
//!    the depthwise Conv1d, kept so each step can apply the causal filter
//!    without reprocessing earlier tokens.
//!
//! 2. **SSM hidden state** — the per-channel state matrix that compresses the
//!    entire past into a fixed-size representation (independent of how many
//!    tokens have been seen).
//!
//! Mirrors [`crate::mamba2::cache`]; Mamba-1 has a single SSD-free recurrence
//! so the cache holds plain tensors rather than the head-structured state of
//! Mamba-2.

use crate::mamba1::prelude::*;
use crate::utils::sanity::sanity as san;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba1Cache  (state for a single layer)
// ---------------------------------------------------------------------------

/// The mutable state carried between calls for a **single** Mamba-1 layer.
#[derive(Module, Debug)]
pub struct Mamba1Cache<B: Backend> {
    /// **Convolution rolling window.**
    ///
    /// The last `conv_kernel` feature vectors fed into the depthwise Conv1d.
    /// At each step the oldest column is dropped and the new token's projection
    /// is appended on the right, maintaining strict causality.
    ///
    /// Shape: `[batch, d_inner, conv_kernel]`
    pub conv_bik: Tensor<B, 3>,

    /// **SSM hidden state.**
    ///
    /// The O(d_inner·state_rank) compressed summary of all tokens seen so far,
    /// updated by the selective-scan recurrence at each step.
    ///
    /// Shape: `[batch, d_inner, state_rank]`
    pub ssm_bir: Tensor<B, 3>,
}

impl<B: Backend> Mamba1Cache<B> {
    /// Run the [`NaN`/`Inf` guards](crate::utils::sanity) on every cached tensor.
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
    /// Zero initialisation is correct because the convolution window represents
    /// "no previous tokens" (identity padding) and the SSM state represents the
    /// standard zero initial condition `h₀ = 0`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1Cache<B> {
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
/// During autoregressive decoding a [`Mamba1Caches`] instance is threaded
/// through every call to [`Mamba1Layers::step`].  Each element corresponds to
/// one (virtual) layer in the network.
#[derive(Module, Debug)]
pub struct Mamba1Caches<B: Backend> {
    /// Per-layer caches.
    ///
    /// Length: `n_real_caches` (the number of *virtual* layers, which may
    /// exceed the number of *real* weight layers when weight-sharing / layer
    /// scheduling is in use).
    pub caches: Vec<Mamba1Cache<B>>,
}

/// Configuration / factory for [`Mamba1Caches`].
#[derive(Config, Debug)]
pub struct Mamba1CachesConfig {
    /// Number of cache slots.  Equals the number of virtual layers in the
    /// network (one cache per layer, even when layers share weights).
    pub n_real_caches: usize,

    /// Shared configuration that determines the shape of each individual
    /// cache tensor.
    pub cache: Mamba1CacheConfig,
}

impl Mamba1CachesConfig {
    /// Convenience constructor that derives cache shapes directly from a
    /// [`Mamba1Config`] block configuration.
    pub fn new_from_block_config(
        n_real_caches: usize,
        batch: usize,
        block_config: Mamba1Config,
    ) -> Self {
        Self {
            n_real_caches,
            cache: Mamba1CacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1Caches<B> {
        let caches = (0..self.n_real_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba1Caches { caches }
    }
}

impl<B: Backend> Mamba1Caches<B> {
    /// Number of per-layer caches.
    pub fn caches_len(&self) -> usize {
        self.caches.len()
    }

    /// Wrap a vector of per-layer caches.
    pub fn from_vec(vec: Vec<Mamba1Cache<B>>) -> Self {
        Self { caches: vec }
    }

    /// Wrap each per-layer cache in `Some` so the layer loop can `take` it
    /// without cloning (Burn tensors are reference-counted).
    pub fn into_options(self) -> Vec<Option<Mamba1Cache<B>>> {
        self.caches.into_iter().map(Some).collect()
    }

    /// Inverse of [`Self::into_options`]: unwrap each slot and re-bundle.
    pub fn from_options(options: Vec<Option<Mamba1Cache<B>>>) -> Self {
        let caches = options.into_iter().map(Option::unwrap).collect();
        Self::from_vec(caches)
    }
}
