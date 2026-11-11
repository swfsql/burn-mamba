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
//! 2. **State** — the per-channel state matrix. It compresses the
//!    full past into a fixed size, for any number of tokens.
//!
//! Mirrors [`crate::mamba2::cache`]. The Mamba-1 state has no head axis.

use crate::mamba1::prelude::*;
use burn_stack::modules::sanity as san;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba1Cache  (cache for a single block)
// ---------------------------------------------------------------------------

/// The cache that a **single** Mamba-1 block carries between calls.
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

    /// **State.**
    ///
    /// The O(d_inner·state_rank) compressed summary of all earlier tokens. The
    /// selective-scan recurrence updates it at each token.
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

impl Mamba1 {
    /// The cache of a fresh sequence, for `batch` rows. A `forward` or a
    /// `step` without a cache starts from it. The shapes come from the
    /// weights of the block.
    ///
    /// Zeros are correct: a zero convolution window means "no previous tokens"
    /// (the causal zero padding), and a zero state is the standard initial
    /// condition `h₀ = 0`.
    pub fn zero_cache(&self, batch: usize, device: &Device) -> Mamba1Cache {
        let [d_inner, state_rank] = self.a_log.dims();
        let [_, _, conv_kernel] = self.conv1d.weight.dims();
        let conv_bik = Tensor::zeros([batch, d_inner, conv_kernel], device);
        let ssm_bir = Tensor::zeros([batch, d_inner, state_rank], device);
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

    /// Inverse of [`Self::into_options`]: unwrap each layer cache and re-bundle.
    pub fn from_options(options: Vec<Option<Mamba1Cache>>) -> Self {
        let caches = options.into_iter().map(Option::unwrap).collect();
        Self::from_vec(caches)
    }
}
