//! Utilizes Mamba2Block and other Modules to build a Mamba2 model capable of utilizing the state-spaces/mamba2-130m text prediction models.

use crate::mamba2_block::{
    Mamba2Block, Mamba2BlockCache, Mamba2BlockCacheConfig, Mamba2BlockConfig,
};
use crate::rms_norm::{RmsNorm, RmsNormConfig};
use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Mamba2BlockCaches<B: Backend> {
    /// # Shape
    /// [n_layers]
    pub caches: Vec<Mamba2BlockCache<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2BlockCachesConfig {
    pub n_layers: usize,
    pub cache: Mamba2BlockCacheConfig,
}

impl Mamba2BlockCachesConfig {
    pub fn new_from_block_config(
        n_layers: usize,
        batch: usize,
        block_config: Mamba2BlockConfig,
    ) -> Self {
        Self {
            n_layers,
            cache: Mamba2BlockCacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2BlockCaches<B> {
        let mut caches: Vec<Mamba2BlockCache<B>> = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            let cache: Mamba2BlockCache<B> = self.cache.clone().init(device);
            caches.push(cache);
        }
        Mamba2BlockCaches { caches }
    }
}

#[derive(Module, Debug)]
pub struct Mamba2<B: Backend> {
    pub embedding: Embedding<B>,
    pub layers: Mamba2Layers<B>,
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
        let layers = Mamba2LayersConfig::new(self.n_layer, self.mamba_block.clone()).init(device);
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
    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2BlockCaches<B>) {
        let device = &x.device();
        let [batch, _sequence] = x.dims();
        let layer0_block = &self.layers.layers[0].mamba_block;
        let [conv_dim, _, d_conv] = layer0_block.conv1d.weight.dims();

        let caches = Mamba2BlockCachesConfig::new(
            self.layers.layers.len(),
            Mamba2BlockCacheConfig {
                batch,
                d_state: layer0_block.d_state,
                d_conv,
                conv_dim,
                headdim: layer0_block.headdim(),
                nheads: layer0_block.nheads(),
            },
        )
        .init(device);

        self.forward_with_caches(x, caches, chunk_size)
    }

    /// See also [`Self::step`].
    ///
    /// `chunk_size`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    ///   - Input [batch, sequence]
    ///   - Output [batch, sequence, d_model]
    pub fn forward_with_caches(
        &self,
        x: Tensor<B, 2, Int>,
        caches: Mamba2BlockCaches<B>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2BlockCaches<B>) {
        let [batch, sequence] = x.dims();
        let [padded_vocab, d_model] = self.embedding.weight.dims();

        let x = self.embedding.forward(x);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        let (mut x, caches) = self.layers.forward_with_caches(x, caches, chunk_size);

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
}

#[derive(Module, Debug)]
pub struct Mamba2Layers<B: Backend> {
    pub layers: Vec<Mamba2Layer<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2LayersConfig {
    pub n_layer: usize,
    pub mamba_block: Mamba2BlockConfig,
}

impl Mamba2LayersConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Layers<B> {
        let mut layers = Vec::with_capacity(self.n_layer);
        for _ in 0..self.n_layer {
            let block_config = self.mamba_block.clone();
            let layer = Mamba2LayerConfig::new(block_config).init(device);
            layers.push(layer);
        }
        Mamba2Layers { layers }
    }
}

impl<B: Backend> Mamba2Layers<B> {
    /// See also [`Self::step`].
    ///
    /// `chunk_size`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2BlockCaches<B>) {
        let device = &x.device();
        let [batch, _sequence, _d_model] = x.dims();
        let layer0_block = &self.layers[0].mamba_block;
        let [conv_dim, _, d_conv] = layer0_block.conv1d.weight.dims();

        let caches = Mamba2BlockCachesConfig::new(
            self.layers.len(),
            Mamba2BlockCacheConfig {
                batch,
                d_state: layer0_block.d_state,
                d_conv,
                conv_dim,
                headdim: layer0_block.headdim(),
                nheads: layer0_block.nheads(),
            },
        )
        .init(device);

        self.forward_with_caches(x, caches, chunk_size)
    }

    /// See also [`Self::step`].
    ///
    /// `chunk_size`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output [batch, sequence, d_model]
    pub fn forward_with_caches(
        &self,
        mut x: Tensor<B, 3>,
        mut caches: Mamba2BlockCaches<B>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2BlockCaches<B>) {
        let chunk_size = chunk_size.unwrap_or(256);
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_, cache_) = layer.forward_with_cache(x, caches.caches[i].clone(), chunk_size);
            x = x_;
            caches.caches[i] = cache_;
        }

        (x, caches)
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
    ///   - Output.0 [batch, sequence, d_model]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        chunk_size: usize,
    ) -> (Tensor<B, 3>, Mamba2BlockCache<B>) {
        let device = &x.device();
        let [batch, _sequence, _d_model] = x.dims();
        let [conv_dim, _, d_conv] = self.mamba_block.conv1d.weight.dims();

        let cache = Mamba2BlockCacheConfig {
            batch,
            d_state: self.mamba_block.d_state,
            d_conv,
            conv_dim,
            headdim: self.mamba_block.headdim(),
            nheads: self.mamba_block.nheads(),
        }
        .init(device);

        self.forward_with_cache(x, cache, chunk_size)
    }

    /// See also [`Self::step`].
    ///
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output.0 [batch, sequence, d_model]
    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        cache: Mamba2BlockCache<B>,
        chunk_size: usize,
    ) -> (Tensor<B, 3>, Mamba2BlockCache<B>) {
        let [batch, sequence, d_model] = x.dims();

        let res = x.clone();
        let x = self.norm.forward(x);

        let (x, cache) = self.mamba_block.forward_with_cache(x, cache, chunk_size);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        (x + res, cache)
    }
}

mod step {
    use super::*;

    impl<B: Backend> Mamba2<B> {
        /// See also [`Self::forward`].
        ///
        /// # Shapes
        ///   - Input [batch]
        ///   - Output [batch, d_model]
        pub fn step(
            &self,
            x: Tensor<B, 1, Int>,
            caches: Mamba2BlockCaches<B>,
        ) -> (Tensor<B, 2>, Mamba2BlockCaches<B>) {
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

    impl<B: Backend> Mamba2Layers<B> {
        /// See also [`Self::forward`].
        ///
        /// # Shapes
        ///   - Input [batch, d_model]
        ///   - Output [batch, d_model]
        pub fn step(
            &self,
            mut x: Tensor<B, 2>,
            mut caches: Mamba2BlockCaches<B>,
        ) -> (Tensor<B, 2>, Mamba2BlockCaches<B>) {
            for (i, layer) in self.layers.iter().enumerate() {
                let (x_, cache) = layer.step(x, caches.caches[i].clone());
                x = x_;
                caches.caches[i] = cache;
            }

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
