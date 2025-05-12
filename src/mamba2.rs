//! Utilizes Mamba2Block and other Modules to build a Mamba2 model capable of utilizing the state-spaces/mamba2-130m text prediction models.
// TODO: merge with Mamba1 by having an enum?

use crate::mamba2_block::{Mamba2Block, Mamba2BlockConfig};
use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Mamba2<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<Mamba2Layer<B>>,
    pub norm_f: RmsNorm<B>,
    /// If missing, re-utilizes a transposed `embedding` weight.
    pub lm_head: Option<Linear<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2Config {
    pub n_layer: usize,

    /// If vocab_size is divisible by pad_vocab_size_multiple, this should be considered the unpadded vocab size.
    /// Otherwise, this is padded into `((self.vocab_size / self.pad_vocab_size_multiple) + 1) * self.pad_vocab_size_multiple`.
    pub vocab_size: usize,

    /// If no pad is required, vocab_size must be divisible by pad_vocab_size_multiple.
    /// If pad is required, vocab_size increases until it's divisible by pad_vocab_size_multiple.
    ///
    /// To disable vocab padding, you can set this to `1`.
    pub pad_vocab_size_multiple: usize,

    pub mamba_block: Mamba2BlockConfig,

    /// If set to true, `lm_head` is set to `None` and it re-utilizes the transposed `embedding` weights.
    pub missing_lm_head: bool,
}

impl Mamba2Config {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2<B> {
        let mut layers = Vec::with_capacity(self.n_layer);
        for _ in 0..self.n_layer {
            let block_config = self.mamba_block.clone();
            // println!("mamba2 config layer headdim: {}", block_config.headdim);
            let layer = Mamba2LayerConfig::new(block_config).init(device);
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

        Mamba2 {
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

impl<B: Backend> Mamba2<B> {
    /// See also [`Self::step`].
    ///
    /// `chunk_size`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    ///   - Input [batch, sequence]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(&self, x: Tensor<B, 2, Int>, chunk_size: Option<usize>) -> Tensor<B, 3> {
        let [batch, sequence] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        let mut x = self.embedding.forward(x);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        let chunk_size = chunk_size.unwrap_or(256);
        for layer in self.layers.iter() {
            x = layer.forward(x, chunk_size);
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
}

#[derive(Module, Debug)]
pub struct Mamba2Layer<B: Backend> {
    pub norm: RmsNorm<B>,
    pub mamba_block: Mamba2Block<B>,
}

#[derive(Config, Debug)]
pub struct Mamba2LayerConfig {
    pub mamba_block: Mamba2BlockConfig,
}

impl Mamba2LayerConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Layer<B> {
        Mamba2Layer {
            norm: RmsNormConfig::new(self.mamba_block.d_model).init(device),
            mamba_block: self.mamba_block.init(device),
        }
    }
}

impl<B: Backend> Mamba2Layer<B> {
    /// See also [`Self::step`].
    ///
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(&self, x: Tensor<B, 3>, chunk_size: usize) -> Tensor<B, 3> {
        let [batch, sequence, d_model] = x.dims();

        let res = x.clone();
        let x = self.norm.forward(x);

        let x = self.mamba_block.forward(x, chunk_size);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        x + res
    }
}

mod step {
    use super::*;
    use crate::mamba2_block::step::Mamba2BlockCache;

    impl<B: Backend> Mamba2<B> {
        /// See also [`Self::forward`].
        ///
        /// # Shapes
        ///   - Input [batch]
        ///   - Output [batch, d_model]
        pub fn step(
            &self,
            x: Tensor<B, 1, Int>,
            mut caches: Vec<Mamba2BlockCache<B>>,
        ) -> (Tensor<B, 2>, Vec<Mamba2BlockCache<B>>) {
            let [batch] = x.dims();
            let [padded_vocab, d_model] = self.embedding.weight.dims();

            let x = x.unsqueeze_dim(1);
            debug_assert_eq!([batch, 1], x.dims());

            let x = self.embedding.forward(x);
            debug_assert_eq!([batch, 1, d_model], x.dims());
            let mut x = x.squeeze(1);
            debug_assert_eq!([batch, d_model], x.dims());

            for (i, layer) in self.layers.iter().enumerate() {
                let (x_, cache) = layer.step(x, caches[i].clone());
                x = x_;
                caches[i] = cache;
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

    impl<B: Backend> Mamba2Layer<B> {
        /// See also [`Self::forward`].
        ///
        /// # Shapes
        ///   - Input [batch, d_model]
        ///   - Output [batch, d_model]
        pub fn step(
            &self,
            x: Tensor<B, 2>,
            cache: Mamba2BlockCache<B>,
        ) -> (Tensor<B, 2>, Mamba2BlockCache<B>) {
            let [batch, d_model] = x.dims();

            let res = x.clone();
            let x = self.norm.forward(x);
            let (x, cache) = self.mamba_block.step(x, cache);
            debug_assert_eq!([batch, d_model], x.dims());

            (x + res, cache)
        }
    }
}
