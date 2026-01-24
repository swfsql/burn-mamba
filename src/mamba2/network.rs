//! Utilizes Mamba2 and other Modules to build a Mamba2 model capable of utilizing the state-spaces/mamba2-130m text prediction models.

use crate::mamba2::*;
use crate::schedule::Schedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Mamba2Network<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Mamba2Layers<B>,
    pub norm_f: RmsNorm<B>,
    /// If missing, re-utilizes a transposed `embedding` weight.
    pub lm_head: Option<Linear<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2NetworkConfig {
    pub n_real_layers: usize,
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// If vocab_size is divisible by pad_vocab_size_multiple, this should be considered the unpadded vocab size.
    /// Otherwise, this is padded into `((self.vocab_size / self.pad_vocab_size_multiple) + 1) * self.pad_vocab_size_multiple`.
    pub vocab_size: usize,

    /// If no pad is required, vocab_size must be divisible by pad_vocab_size_multiple.
    /// If pad is required, vocab_size increases until it's divisible by pad_vocab_size_multiple.
    ///
    /// To disable vocab padding, you can set this to `1`.
    pub pad_vocab_size_multiple: usize,

    pub mamba_block: Mamba2Config,

    /// If set to true, `lm_head` is set to `None` and it re-utilizes the transposed `embedding` weights.
    pub missing_lm_head: bool,
}

impl Mamba2NetworkConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Network<B> {
        let layers =
            Mamba2LayersConfig::new(self.n_real_layers, self.mamba_block.clone()).init(device);
        let padded_vocab_size = {
            if self.vocab_size % self.pad_vocab_size_multiple == 0 {
                self.vocab_size
            } else {
                ((self.vocab_size / self.pad_vocab_size_multiple) + 1)
                    * self.pad_vocab_size_multiple
            }
        };

        Mamba2Network {
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

impl<B: Backend> Mamba2Network<B> {
    /// See also [`Self::step`].
    ///
    /// `chunk_size`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    ///   - Input [batch, sequence]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        caches: Option<Mamba2Caches<B>>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2Caches<B>) {
        let [batch, sequence] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        let x = self.embedding.forward(x);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        let (mut x, caches) = self.layers.forward(x, caches, chunk_size);

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

        (x, caches)
    }

    /// See also [`Self::forward`].
    ///
    /// # Shapes
    ///   - Input [batch]
    ///   - Output [batch, d_model]
    pub fn step(
        &self,
        x: Tensor<B, 1, Int>,
        caches: Option<Mamba2Caches<B>>,
    ) -> (Tensor<B, 2>, Mamba2Caches<B>) {
        let [batch] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        let x = x.unsqueeze_dim(1);
        debug_assert_eq!([batch, 1], x.dims());

        let x = self.embedding.forward(x);
        debug_assert_eq!([batch, 1, d_model], x.dims());
        let x = x.squeeze_dim(1);
        debug_assert_eq!([batch, d_model], x.dims());

        let (mut x, caches) = self.layers.step(x, caches);

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
