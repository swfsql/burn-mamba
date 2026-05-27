//! # Mamba-1 Language Model Network
//!
//! Assembles a complete autoregressive language model from the Mamba-1
//! components:
//!
//! ```text
//!   tokens [batch, sequence]
//!       │  Embedding (vocab_size → d_model)
//!       ▼  (×n_virtual_layers)
//!   Mamba1Layer [Pre-LN residual block]
//!       │  RMSNorm (final)
//!       ▼  LM head (d_model → vocab_size)
//!   logits [batch, sequence, vocab_size]
//! ```
//!
//! Mirrors [`crate::mamba2::network`]: vocabulary is padded up to
//! `pad_vocab_size_multiple`, the LM head can be weight-tied to the embedding
//! (`missing_lm_head = true`), and both [`Mamba1Network::forward`] and
//! [`Mamba1Network::step`] thread a [`Mamba1Caches`] so prefill can be followed
//! by decoding.
//!
//! References:
//! - [huggingface/candle](https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/)
//! - [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/)

use crate::mamba1::prelude::*;
use crate::schedule::Schedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::backend::Backend;

/// A complete Mamba-1 language model.
#[derive(Module, Debug)]
pub struct Mamba1Network {
    /// Token embedding table (`padded_vocab_size → d_model`).
    pub embedding: Embedding,
    /// The stack of Mamba-1 residual layers.
    pub layers: Mamba1Layers,
    /// Final RMSNorm applied before the LM head.
    pub norm_f: RmsNorm,
    /// If missing, re-utilizes a transposed `embedding` weight.
    pub lm_head: Option<Linear>,
}

/// Configuration / factory for [`Mamba1Network`].
#[derive(Config, Debug)]
pub struct Mamba1NetworkConfig {
    /// Number of real (weight-bearing) Mamba-1 layers.
    pub n_real_layers: usize,

    /// Optional virtual-layer scheduling.  See [`Mamba1Layers`] for details.
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// If vocab_size is divisible by pad_vocab_size_multiple, this should be considered the unpadded vocab size.
    /// Otherwise, this is padded into `((vocab_size / self.pad_vocab_size_multiple) + 1) * pad_vocab_size_multiple`.
    pub vocab_size: usize,

    /// If no pad is required, vocab_size must be divisible by pad_vocab_size_multiple.
    /// If pad is required, vocab_size increases until it's divisible by pad_vocab_size_multiple.
    ///
    /// To disable vocab padding, you can set this to `1`.
    pub pad_vocab_size_multiple: usize,

    /// Configuration shared by all Mamba-1 blocks.
    pub mamba_block: Mamba1Config,

    /// If set to true, `lm_head` is set to `None` and it re-utilizes the transposed `embedding` weights.
    pub missing_lm_head: bool,
}

impl Mamba1NetworkConfig {
    /// Returns the initialized model.
    pub fn init(&self, device: &Device) -> Mamba1Network {
        let layers = Mamba1LayersConfig {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            mamba_block: self.mamba_block.clone(),
            ignore_first_residual: false,
            ignore_last_residual: false,
        }
        .init(device);

        let padded_vocab_size = {
            if self.vocab_size.is_multiple_of(self.pad_vocab_size_multiple) {
                self.vocab_size
            } else {
                ((self.vocab_size / self.pad_vocab_size_multiple) + 1)
                    * self.pad_vocab_size_multiple
            }
        };

        Mamba1Network {
            embedding: EmbeddingConfig::new(padded_vocab_size, self.mamba_block.d_model)
                .init(device),
            layers,
            norm_f: RmsNormConfig::new(self.mamba_block.d_model).init(device),
            lm_head: if self.missing_lm_head {
                None
            } else {
                Some(
                    LinearConfig::new(self.mamba_block.d_model, padded_vocab_size)
                        .with_bias(false)
                        .init(device),
                )
            },
        }
    }
}

impl Mamba1Network {
    /// See also [`Self::step`].
    ///
    /// Processes a full token sequence and returns next-token logits along with
    /// the updated caches (ready for a subsequent [`Self::step`] call).
    ///
    /// # Shapes
    ///   - Input `[batch, sequence]`
    ///   - Output `[batch, sequence, padded_vocab_size]`
    pub fn forward(
        &self,
        x: Tensor<2, Int>,
        caches: Option<Mamba1Caches>,
    ) -> (Tensor<3>, Mamba1Caches) {
        let [batch, sequence] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        let x = self.embedding.forward(x);
        assert_eq!([batch, sequence, d_model], x.dims());

        let (mut x, caches) = self.layers.forward(x, caches);
        assert_eq!([batch, sequence, d_model], x.dims());

        x = self.norm_f.forward(x);
        x = self.apply_lm_head(x, d_model, padded_vocab);
        assert_eq!([batch, sequence, padded_vocab], x.dims());

        (x, caches)
    }

    /// See also [`Self::forward`].
    ///
    /// Processes a **single** token and returns next-token logits along with
    /// the caches updated for the next step.
    ///
    /// # Shapes
    ///   - Input `[batch]`
    ///   - Output `[batch, padded_vocab_size]`
    pub fn step(
        &self,
        x: Tensor<1, Int>,
        caches: Option<Mamba1Caches>,
    ) -> (Tensor<2>, Mamba1Caches) {
        let [batch] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        let x = x.unsqueeze_dim::<2>(1);
        assert_eq!([batch, 1], x.dims());

        let x = self.embedding.forward(x);
        assert_eq!([batch, 1, d_model], x.dims());
        let x = x.squeeze_dim(1);
        assert_eq!([batch, d_model], x.dims());

        let (mut x, caches) = self.layers.step(x, caches);
        assert_eq!([batch, d_model], x.dims());

        x = self.norm_f.forward(x);

        // Re-use the `apply_lm_head` helper by temporarily unsqueezing the
        // sequence dimension then squeezing it back out.
        let x = x.unsqueeze_dim(1);
        let logits = self.apply_lm_head(x, d_model, padded_vocab);
        assert_eq!([batch, 1, padded_vocab], logits.dims());
        let logits = logits.squeeze_dim(1);
        assert_eq!([batch, padded_vocab], logits.dims());

        (logits, caches)
    }

    /// Apply the LM head projection to a `[batch, sequence, d_model]` tensor,
    /// returning `[batch, sequence, padded_vocab]`.
    ///
    /// Uses the dedicated `lm_head` linear layer when available, or the
    /// transposed embedding weight matrix otherwise (weight tying).
    fn apply_lm_head(&self, x: Tensor<3>, d_model: usize, padded_vocab: usize) -> Tensor<3> {
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(x)
        } else {
            let weight = self.embedding.weight.clone().map(|w| w.permute([1, 0]));
            assert_eq!([d_model, padded_vocab], weight.dims());
            let linear = Linear { weight, bias: None };
            linear.forward(x)
        }
    }
}
