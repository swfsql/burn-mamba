//! # Mamba-2 Language Model Network
//!
//! This module assembles a complete autoregressive language model from the
//! Mamba-2 components:
//!
//! ```text
//!   tokens [B, T]
//!       │
//!       ▼
//!   Embedding  (vocab_size → d_model)
//!       │
//!       ▼  (×n_layers)
//!   Mamba2Layer  [Pre-LN residual block]
//!       │
//!       ▼
//!   RMSNorm  (final normalisation)
//!       │
//!       ▼
//!   LM head  (d_model → vocab_size)
//!       │
//!       ▼
//!   logits [B, T, vocab_size]
//! ```
//!
//! ## Vocabulary padding
//!
//! The embedding and LM head dimensions are rounded up to the nearest
//! multiple of `pad_vocab_size_multiple`.  This improves memory alignment on
//! GPU without exposing the extra token slots to the model (they are never
//! sampled from in practice).
//!
//! ## Tied / untied LM head
//!
//! When `missing_lm_head = true`, the logit projection reuses the *transposed*
//! embedding weight matrix (`lm_head = None`, applied as a linear layer on
//! the fly).  This halves the parameter count for the output projection and is
//! standard in many LLM implementations.  When `missing_lm_head = false`, a
//! separate [`Linear`] layer is allocated.
//!
//! ## Two execution modes
//!
//! | Method | Input shape | Use case |
//! |--------|-------------|----------|
//! | [`Mamba2Network::forward`] | `[B, T]` | Training, prefill |
//! | [`Mamba2Network::step`]    | `[B]`    | Autoregressive decoding |

use crate::mamba2::*;
use crate::schedule::Schedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba2Network
// ---------------------------------------------------------------------------

/// A complete Mamba-2 language model.
///
/// See the [module-level documentation](self) for an overview of the
/// architecture and the two execution modes.
#[derive(Module, Debug)]
pub struct Mamba2Network<B: Backend> {
    /// Token embedding table.
    ///
    /// Shape of weight matrix: `[padded_vocab_size, d_model]`.
    /// Maps integer token IDs to `d_model`-dimensional vectors.
    pub embedding: Embedding<B>,

    /// The stack of Mamba-2 residual blocks.
    pub layers: Mamba2Layers<B>,

    /// Final layer normalisation applied after all Mamba-2 blocks and before
    /// the LM head.  This is the `norm_f` in the original implementation.
    pub norm_f: RmsNorm<B>,

    /// Optional separate LM head projection.
    ///
    /// - `Some(linear)` — dedicated weight matrix of shape
    ///   `[d_model, padded_vocab_size]`.
    /// - `None` — the embedding weights are reused (transposed).  This is the
    ///   "weight-tied" variant and is selected when `missing_lm_head = true`.
    pub lm_head: Option<Linear<B>>,
}

// ---------------------------------------------------------------------------
// Mamba2NetworkConfig
// ---------------------------------------------------------------------------

/// Configuration / factory for [`Mamba2Network`].
#[derive(Config, Debug)]
pub struct Mamba2NetworkConfig {
    /// Number of real (weight-bearing) Mamba-2 layers.
    pub n_real_layers: usize,

    /// Optional virtual-layer scheduling.  See [`Mamba2Layers`] for details.
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// The *unpadded* vocabulary size as specified by the tokenizer.
    ///
    /// At initialisation this value is rounded up to the nearest multiple of
    /// `pad_vocab_size_multiple` to obtain the actual embedding / logit
    /// dimension `padded_vocab_size`.
    pub vocab_size: usize,

    /// Vocabulary size will be rounded up to a multiple of this value.
    ///
    /// Set to `1` to disable rounding.  Common values: 8, 16, 64.
    pub pad_vocab_size_multiple: usize,

    /// Configuration shared by all Mamba-2 blocks.
    pub mamba_block: Mamba2Config,

    /// When `true`, the LM head weight is not allocated separately; instead
    /// the transposed embedding matrix is used directly (weight tying).
    pub missing_lm_head: bool,
}

impl Mamba2NetworkConfig {
    /// Allocate and initialise the full network on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Network<B> {
        let padded_vocab_size = Self::padded_vocab(self.vocab_size, self.pad_vocab_size_multiple);

        let layers = Mamba2LayersConfig {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            mamba_block: self.mamba_block.clone(),
            ignore_first_residual: false,
            ignore_last_residual: false,
        }
        .init(device);

        let lm_head = if self.missing_lm_head {
            None
        } else {
            Some(
                LinearConfig::new(self.mamba_block.d_model, padded_vocab_size)
                    .with_bias(false)
                    .init(device),
            )
        };

        Mamba2Network {
            embedding: EmbeddingConfig::new(padded_vocab_size, self.mamba_block.d_model)
                .init(device),
            layers,
            norm_f: RmsNormConfig::new(self.mamba_block.d_model).init(device),
            lm_head,
        }
    }

    /// Round `vocab_size` up to the next multiple of `multiple`.
    fn padded_vocab(vocab_size: usize, multiple: usize) -> usize {
        if vocab_size % multiple == 0 {
            vocab_size
        } else {
            ((vocab_size / multiple) + 1) * multiple
        }
    }
}

// ---------------------------------------------------------------------------
// Inference implementations
// ---------------------------------------------------------------------------

impl<B: Backend> Mamba2Network<B> {
    // -----------------------------------------------------------------------
    // forward  (full sequence — training / prefill)
    // -----------------------------------------------------------------------

    /// Process a full token sequence and return next-token logits.
    ///
    /// Internally this calls [`Mamba2Layers::forward`], which runs the
    /// chunkwise SSD algorithm over every layer.  This is the mode to use
    /// during training (backpropagation through the entire sequence) and
    /// during the prefill phase of inference.
    ///
    /// # Arguments
    /// - `x`          — integer token IDs, shape `[batch, sequence]`
    /// - `caches`     — optional pre-filled layer caches.  Pass `None` to
    ///                  start from a zero state (training) or to create fresh
    ///                  caches that can be returned and reused for a subsequent
    ///                  decoding step.
    /// - `chunk_size` — SSD chunk length Q (default 256).  Larger values
    ///                  increase the intra-chunk GEMM work and reduce the
    ///                  inter-chunk scan length.  Optimal value ≈ √(N·P).
    ///
    /// # Returns
    /// `(logits, caches)` where:
    /// - `logits` has shape `[batch, sequence, padded_vocab_size]`
    /// - `caches` contains the SSM and convolution state at the end of the
    ///   sequence, ready to be passed to the first [`Self::step`] call.
    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        caches: Option<Mamba2Caches<B>>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2Caches<B>) {
        let [batch, sequence] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        // Embed token IDs → dense vectors.
        let x_bsm = self.embedding.forward(x);
        assert_eq!([batch, sequence, d_model], x_bsm.dims());

        // Run the Mamba-2 layer stack (chunkwise SSD).
        let (mut x_bsm, caches) = self.layers.forward(x_bsm, caches, chunk_size);
        assert_eq!([batch, sequence, d_model], x_bsm.dims());

        // Final normalisation before projection.
        x_bsm = self.norm_f.forward(x_bsm);
        assert_eq!([batch, sequence, d_model], x_bsm.dims());

        // Project to vocabulary logits.
        x_bsm = self.apply_lm_head(x_bsm, d_model, padded_vocab);
        assert_eq!([batch, sequence, padded_vocab], x_bsm.dims());

        (x_bsm, caches)
    }

    // -----------------------------------------------------------------------
    // step  (single token — autoregressive decoding)
    // -----------------------------------------------------------------------

    /// Process a **single** token and return next-token logits.
    ///
    /// Internally this calls [`Mamba2Layers::step`], which advances each
    /// layer's recurrent state by one step:
    ///
    /// ```text
    ///   hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ
    ///   yₜ = Cₜᵀ hₜ + D xₜ
    /// ```
    ///
    /// This is O(H·P·N) per token — independent of sequence length — and is
    /// the correct mode for token-by-token generation after prefill.
    ///
    /// # Arguments
    /// - `x`      — current token IDs, shape `[batch]`
    /// - `caches` — layer caches from the previous step (or `None` for the
    ///              very first token, which starts from a zero hidden state)
    ///
    /// # Returns
    /// `(logits, caches)` where:
    /// - `logits` has shape `[batch, padded_vocab_size]`
    /// - `caches` contains the updated state for the **next** step.
    pub fn step(
        &self,
        x: Tensor<B, 1, Int>,
        caches: Option<Mamba2Caches<B>>,
    ) -> (Tensor<B, 2>, Mamba2Caches<B>) {
        let [batch] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        // Embed the single token.  We temporarily add a sequence dimension so
        // that the embedding module (which expects `[B, T]`) is satisfied, then
        // immediately squeeze it out.
        let x_b1 = x.unsqueeze_dim::<2>(1);
        assert_eq!([batch, 1], x_b1.dims());

        let x_b1m = self.embedding.forward(x_b1);
        assert_eq!([batch, 1, d_model], x_b1m.dims());

        let x_bm = x_b1m.squeeze_dim(1);
        assert_eq!([batch, d_model], x_bm.dims());

        // Advance each layer's recurrent state by one step.
        let (mut x_bm, caches) = self.layers.step(x_bm, caches);
        assert_eq!([batch, d_model], x_bm.dims());

        // Final normalisation.
        x_bm = self.norm_f.forward(x_bm);
        assert_eq!([batch, d_model], x_bm.dims());

        // Project to vocabulary logits.
        // Re-use the `apply_lm_head` helper by temporarily unsqueezing the
        // sequence dimension then squeezing it back out.
        let x_b1m = x_bm.unsqueeze_dim(1);
        let logits_b1v = self.apply_lm_head(x_b1m, d_model, padded_vocab);
        assert_eq!([batch, 1, padded_vocab], logits_b1v.dims());

        let logits_bv = logits_b1v.squeeze_dim(1);
        assert_eq!([batch, padded_vocab], logits_bv.dims());

        (logits_bv, caches)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Apply the LM head projection to a `[batch, sequence, d_model]` tensor,
    /// returning `[batch, sequence, padded_vocab]`.
    ///
    /// Uses the dedicated `lm_head` linear layer when available, or the
    /// transposed embedding weight matrix otherwise (weight tying).
    fn apply_lm_head(
        &self,
        x_bsm: Tensor<B, 3>,
        d_model: usize,
        padded_vocab: usize,
    ) -> Tensor<B, 3> {
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(x_bsm)
        } else {
            // Weight-tied variant: reuse embedding.weight^T as the projection.
            // embedding.weight has shape [padded_vocab, d_model], so we need
            // to transpose it to [d_model, padded_vocab].
            let weight_mv = self.embedding.weight.clone().map(|w| w.permute([1, 0]));
            assert_eq!([d_model, padded_vocab], weight_mv.dims());
            let tied_linear = Linear {
                weight: weight_mv,
                bias: None,
            };
            tied_linear.forward(x_bsm)
        }
    }
}
