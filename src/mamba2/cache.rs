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
//! Each block keeps two pieces of state:
//!
//! 1. **Convolution cache** — the last `conv_kernel` inputs to the depthwise
//!    Conv1d. With it, each decoding step applies the causal filter without
//!    processing earlier tokens again.
//!
//! 2. **State** — the matrix `hₜ ∈ ℝ^{per_head_dim×state_rank}` per
//!    head. It compresses the full past into a fixed size, for any number of
//!    tokens. This is the memory advantage of SSMs over attention: the
//!    KV-cache of a Transformer grows linearly with the sequence length, but
//!    the state is always O(per_head_dim·state_rank).
//!
//! [`Mamba2::zero_cache`] builds the cache of a fresh sequence.

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

// ---------------------------------------------------------------------------
// Mamba2Cache  (cache for a single block)
// ---------------------------------------------------------------------------

/// The cache that a **single** Mamba-2 block carries between calls.
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

    /// **State** `hₜ`.
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

impl Mamba2 {
    /// The cache of a fresh sequence, for `batch` rows. A `forward` or a
    /// `step` without a cache starts from it, and so does each segment of a
    /// packed call without an origin. The shapes come from the block.
    ///
    /// Zeros are correct:
    /// - A zero convolution cache means "no previous tokens" (the causal zero
    ///   padding).
    /// - A zero state is the standard initial condition `h₀ = 0`. If the
    ///   block has a learnable initial state, [`Mamba2::forward`] adds it to
    ///   this state.
    pub fn zero_cache(&self, batch: usize, device: &Device) -> Mamba2Cache {
        let [conv_dim, _, conv_kernel] = self.conv1d.weight.dims();
        let conv_bvk = Tensor::zeros(Shape::new([batch, conv_dim, conv_kernel]), device);
        let ssm_bhpr = Tensor::zeros(
            Shape::new([batch, self.nheads(), self.per_head_dim(), self.state_rank]),
            device,
        );
        Mamba2Cache { conv_bvk, ssm_bhpr }
    }
}
