//! Utilizes Mamba1 and other Modules to build a Mamba1 model capable of utilizing the state-spaces/mamba-130m text prediction models.
//!
//! References:
//! - https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/
//! - https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/

use crate::mamba1::*;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Mamba1Network<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<Mamba1Layer<B>>,
    pub norm_f: RmsNorm<B>,
    /// If missing, re-utilizes a transposed `embedding` weight.
    pub lm_head: Option<Linear<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba1NetworkConfig {
    pub n_layer: usize,

    /// If vocab_size is divisible by pad_vocab_size_multiple, this should be considered the unpadded vocab size.
    /// Otherwise, this is padded into `((vocab_size / self.pad_vocab_size_multiple) + 1) * pad_vocab_size_multiple`.
    pub vocab_size: usize,

    /// If no pad is required, vocab_size must be divisible by pad_vocab_size_multiple.
    /// If pad is required, vocab_size increases until it's divisible by pad_vocab_size_multiple.
    ///
    /// To disable vocab padding, you can set this to `1`.
    pub pad_vocab_size_multiple: usize,

    pub mamba_block: Mamba1Config,

    /// If set to true, `lm_head` is set to `None` and it re-utilizes the transposed `embedding` weights.
    pub missing_lm_head: bool,
}

impl Mamba1NetworkConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1Network<B> {
        let mut layers = Vec::with_capacity(self.n_layer);
        for _ in 0..self.n_layer {
            let layer = Mamba1LayerConfig::new(self.mamba_block.clone()).init(device);
            layers.push(layer);
        }

        let padded_vocab_size = {
            if self.vocab_size % self.pad_vocab_size_multiple == 0 {
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

impl<B: Backend> Mamba1Network<B> {
    /// See also [`Self::step`].
    ///
    /// # Shapes
    ///   - Input [batch, sequence]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, sequence] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        let mut x = self.embedding.forward(x);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        for layer in self.layers.iter() {
            x = layer.forward(x);
        }

        x = self.norm_f.forward(x);
        if let Some(lm_head) = &self.lm_head {
            x = lm_head.forward(x);
        } else {
            let weight = self.embedding.weight.clone().map(|w| w.swap_dims(0, 1));
            debug_assert_eq!([d_model, padded_vocab], weight.dims());

            let linear = Linear { weight, bias: None };
            x = linear.forward(x);
        };
        debug_assert_eq!([batch, sequence, padded_vocab], x.dims());

        x
    }

    /// See also [`Self::forward`].
    ///
    /// # Shapes
    ///   - Input [batch]
    ///   - Output [batch, d_model]
    pub fn step(
        &self,
        x: Tensor<B, 1, Int>,
        mut caches: Mamba1Caches<B>,
    ) -> (Tensor<B, 2>, Mamba1Caches<B>) {
        let [batch] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        let x = x.unsqueeze_dim(1);
        debug_assert_eq!([batch, 1], x.dims());

        let x = self.embedding.forward(x);
        debug_assert_eq!([batch, 1, d_model], x.dims());
        let mut x = x.squeeze_dim(1);
        debug_assert_eq!([batch, d_model], x.dims());

        for (i, layer) in self.layers.iter().enumerate() {
            let (x_, cache) = layer.step(x, caches.caches[i].clone());
            x = x_;
            caches.caches[i] = cache;
        }

        x = self.norm_f.forward(x);
        if let Some(lm_head) = &self.lm_head {
            x = lm_head.forward(x);
        } else {
            let weight = self.embedding.weight.clone().map(|w| w.swap_dims(0, 1));
            debug_assert_eq!([d_model, padded_vocab], weight.dims());

            let linear = Linear { weight, bias: None };
            x = linear.forward(x);
        };
        debug_assert_eq!([batch, padded_vocab], x.dims());

        (x, caches)
    }
}
