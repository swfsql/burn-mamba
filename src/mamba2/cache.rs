//! # Mamba-2 Inference Caches
//!
//! This module defines the state that must be preserved between calls during
//! autoregressive (token-by-token) generation.  During *training* or *prefill*
//! the full sequence is available at once and the chunked SSD algorithm is used
//! (see [`crate::mamba2::Mamba2::forward`]).  During *decoding* the model
//! processes one token per step and the SSM operates in its pure recurrent
//! form (see [`crate::mamba2::Mamba2::step`]):
//!
//! ```text
//!   hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ        (state update)
//!   yₜ = Cₜᵀ hₜ + D xₜ            (output)
//! ```
//!
//! Two pieces of state are required per layer:
//!
//! 1. **Convolution cache** — the last `conv_kernel` inputs to the depthwise
//!    Conv1d, kept so that every decoding step can apply the causal filter
//!    without re-processing previous tokens.
//!
//! 2. **SSM hidden state** — the matrix `hₜ ∈ ℝ^{P×N}` (per head), which
//!    compresses the entire past context into a fixed-size representation
//!    regardless of how many tokens have been generated.  This is the key
//!    memory-efficiency advantage of SSMs over attention: the KV-cache of a
//!    Transformer grows as O(T·N) with sequence length, whereas the SSM state
//!    is always O(P·N).

use crate::mamba2::*;
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba2Caches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer caches for a complete Mamba-2 network.
///
/// During autoregressive decoding, a [`Mamba2Caches`] instance is threaded
/// through every call to [`crate::layer::Mamba2Layers::step`].  Each element
/// of `caches` corresponds to one (virtual) layer in the network.
#[derive(Module, Debug)]
pub struct Mamba2Caches<B: Backend> {
    /// Per-layer caches.
    ///
    /// Length: `n_real_caches` (the number of *virtual* layers, which may
    /// exceed the number of *real* weight layers when weight-sharing / layer
    /// scheduling is in use).
    pub caches: Vec<Mamba2Cache<B>>,
}

/// Configuration / factory for [`Mamba2Caches`].
#[derive(Config, Debug)]
pub struct Mamba2CachesConfig {
    /// Number of cache slots.  Equals the number of virtual layers in the
    /// network (one cache per layer, even when layers share weights).
    pub n_real_caches: usize,

    /// Shared configuration that determines the shape of each individual
    /// cache tensor.
    pub cache: Mamba2CacheConfig,
}

impl Mamba2CachesConfig {
    /// Convenience constructor that derives cache shapes directly from a
    /// [`Mamba2Config`] block configuration.
    pub fn new_from_block_config(
        n_real_caches: usize,
        batch: usize,
        block_config: Mamba2Config,
    ) -> Self {
        Self {
            n_real_caches,
            cache: Mamba2CacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Allocate all cache tensors (zero-initialised) on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Caches<B> {
        let caches = (0..self.n_real_caches)
            .map(|_| self.cache.clone().init(device))
            .collect();
        Mamba2Caches { caches }
    }
}

// ---------------------------------------------------------------------------
// Mamba2Cache  (state for a single layer)
// ---------------------------------------------------------------------------

/// The mutable state carried between decoding steps for a **single** Mamba-2
/// layer.
///
/// Both tensors are updated in-place (via Burn's functional clone) at every
/// call to [`crate::mamba2::Mamba2::step`].
#[derive(Module, Debug)]
pub struct Mamba2Cache<B: Backend> {
    /// **Convolution rolling window.**
    ///
    /// Stores the last `conv_kernel` pre-activation feature vectors fed into
    /// the depthwise Conv1d.  At each step, the oldest column is discarded and
    /// the new token's projection is appended (a left-shift followed by an
    /// insert into the rightmost column), maintaining strict causality.
    ///
    /// Shape: `[batch, conv_dim, conv_kernel]`
    ///   - `conv_dim  = d_inner + 2 · ngroups · state_rank`
    ///   - `conv_kernel` is typically 4
    pub conv_bvk: Tensor<B, 3>,

    /// **SSM hidden state** `hₜ`.
    ///
    /// This is the O(P·N) compressed summary of all tokens seen so far.
    /// Updated via `hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ` at each decoding step.
    ///
    /// The tensor is indexed as `[batch, nheads, per_head_dim, state_rank]`
    /// (i.e. `[B, H, P, N]` in the paper's notation), which is the transpose
    /// of the mathematical `hₜ ∈ ℝ^{N×P}` but equivalent in content.
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<B, 4>,
}

/// Configuration / factory for a single [`Mamba2Cache`].
#[derive(Config, Debug)]
pub struct Mamba2CacheConfig {
    /// Batch size.
    pub batch: usize,

    /// State rank `N` — the number of latent dimensions in the SSM hidden
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

    /// Head dimension `P`.  Corresponds to `per_head_dim` in [`Mamba2Config`].
    #[config(default = 64)]
    pub per_head_dim: usize,

    /// Number of SSM heads `H`.
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
    /// Zero initialisation is correct because:
    /// - The convolution cache represents "no previous tokens" (identity padding).
    /// - The SSM state represents `h₀ = 0` (zero initial condition), which is
    ///   the standard default.  Learnable initial states (if configured) are
    ///   added on top of this inside [`crate::mamba2::Mamba2::forward`] /
    ///   [`crate::mamba2::Mamba2::step`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Cache<B> {
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
