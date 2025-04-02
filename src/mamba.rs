//! Utilizes MambaBlock and other Modules to build a Mamba model capable of utilizing the state-spaces/mamba-130m text prediction models.
//!
//! References:
//! - https://github.com/huggingface/candle/blob/fd7c8565646039e35925b8730d27ddad195d7e73/candle-examples/examples/mamba-minimal/
//! - https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/

use crate::{layer_rms_norm_1d::*, *};
use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Mamba<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Vec<MambaLayer<B>>,
    pub norm_f: LayerRmsNorm1D<B>,
    /// If missing, re-utilizes a transposed `embedding` weight.
    pub lm_head: Option<Linear<B>>,
}

#[derive(Config, Debug)]
pub struct MambaConfig {
    pub n_layer: usize,

    /// If no pad is required, this should be considered the unpadded vocab size.
    /// If pad is required, this should be the result of `((unpadded_vocab_size + pad - 1) / pad) * pad`.
    pub padded_vocab_size: usize,

    pub mamba_block: MambaBlockConfig,

    /// If set to true, `lm_head` is set to `None` and it re-utilizes the transposed `embedding` weights.
    pub missing_lm_head: bool,
}

impl MambaConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba<B> {
        let mut layers = Vec::with_capacity(self.n_layer);
        for _ in 0..self.n_layer {
            let layer = MambaLayerConfig::new(self.mamba_block.clone()).init(device);
            layers.push(layer);
        }
        Mamba {
            embedding: EmbeddingConfig::new(self.padded_vocab_size, self.mamba_block.d_model)
                .init(device),
            layers,
            norm_f: LayerRmsNorm1DConfig::new(self.mamba_block.d_model).init(device),
            lm_head: if self.missing_lm_head {
                None
            } else {
                Some(
                    LinearConfig::new(self.mamba_block.d_model, self.padded_vocab_size)
                        .with_bias(false)
                        .init(device),
                )
            },
        }
    }
}

impl<B: Backend> Mamba<B> {
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
            let weight = self.embedding.weight.clone().map(|w| w.movedim(0, 1));
            debug_assert_eq!([d_model, padded_vocab], weight.dims());

            let linear = Linear { weight, bias: None };
            x = linear.forward(x);
        };
        debug_assert_eq!([batch, sequence, padded_vocab], x.dims());

        x
    }
}

#[derive(Module, Debug)]
pub struct MambaLayer<B: Backend> {
    pub norm: LayerRmsNorm1D<B>,
    pub mamba_block: MambaBlock<B>,
}

#[derive(Config, Debug)]
pub struct MambaLayerConfig {
    pub mamba_block: MambaBlockConfig,
}

impl MambaLayerConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MambaLayer<B> {
        MambaLayer {
            norm: LayerRmsNorm1DConfig::new(self.mamba_block.d_model).init(device),
            mamba_block: MambaBlockConfig::new(self.mamba_block.d_model)
                .with_d_state(self.mamba_block.d_state)
                .with_dt_rank(self.mamba_block.dt_rank)
                .with_d_conv(self.mamba_block.d_conv)
                .with_d_inner(self.mamba_block.d_inner)
                .init(device),
        }
    }
}

impl<B: Backend> MambaLayer<B> {
    /// See also [`Self::step`].
    ///
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, sequence, d_model] = x.dims();

        let res = x.clone();
        let x = self.norm.forward(x);

        let x = self.mamba_block.forward(x);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        x + res
    }
}

mod step {
    use super::*;
    use crate::mamba_block::step::MambaBlockCache;

    impl<B: Backend> Mamba<B> {
        /// See also [`Self::forward`].
        ///
        /// # Shapes
        ///   - Input [batch]
        ///   - Output [batch, d_model]
        pub fn step(
            &self,
            x: Tensor<B, 1, Int>,
            mut caches: Vec<MambaBlockCache<B>>,
        ) -> (Tensor<B, 2>, Vec<MambaBlockCache<B>>) {
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
                let weight = self.embedding.weight.clone().map(|w| w.movedim(0, 1));
                debug_assert_eq!([d_model, padded_vocab], weight.dims());

                let linear = Linear { weight, bias: None };
                x = linear.forward(x);
            };
            debug_assert_eq!([batch, padded_vocab], x.dims());

            (x, caches)
        }
    }

    impl<B: Backend> MambaLayer<B> {
        /// See also [`Self::forward`].
        ///
        /// # Shapes
        ///   - Input [batch, d_model]
        ///   - Output [batch, d_model]
        pub fn step(
            &self,
            x: Tensor<B, 2>,
            cache: MambaBlockCache<B>,
        ) -> (Tensor<B, 2>, MambaBlockCache<B>) {
            let [batch, d_model] = x.dims();

            let res = x.clone();
            let x = self.norm.forward(x);
            let (x, cache) = self.mamba_block.step(x, cache);
            debug_assert_eq!([batch, d_model], x.dims());

            (x + res, cache)
        }
    }
}
