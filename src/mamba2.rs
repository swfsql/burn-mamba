//! Utilizes Mamba2Block and other Modules to build a Mamba2 model capable of utilizing the state-spaces/mamba2-130m text prediction models.

use crate::mamba2_block::{
    Mamba2Block, Mamba2BlockCache, Mamba2BlockCacheConfig, Mamba2BlockConfig,
};
use crate::schedule::Schedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Mamba2BlockCaches<B: Backend> {
    pub n_real_caches: usize,
    pub n_virtual_caches: Option<(usize, Schedule)>,
    /// # Shape
    /// [n_real_caches]
    pub caches: Vec<Mamba2BlockCache<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2BlockCachesConfig {
    pub n_real_caches: usize,
    #[config(default = "None")]
    pub n_virtual_caches: Option<(usize, Schedule)>,
    pub cache: Mamba2BlockCacheConfig,
}

impl Mamba2BlockCachesConfig {
    pub fn new_from_block_config(
        n_real_caches: usize,
        n_virtual_caches: Option<(usize, Schedule)>,
        batch: usize,
        block_config: Mamba2BlockConfig,
    ) -> Self {
        Self {
            n_real_caches,
            n_virtual_caches,
            cache: Mamba2BlockCacheConfig::new_from_block_config(batch, block_config),
        }
    }

    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2BlockCaches<B> {
        let mut caches: Vec<Mamba2BlockCache<B>> = Vec::with_capacity(self.n_real_caches);
        for _ in 0..self.n_real_caches {
            let cache: Mamba2BlockCache<B> = self.cache.clone().init(device);
            caches.push(cache);
        }
        Mamba2BlockCaches {
            n_real_caches: self.n_real_caches,
            n_virtual_caches: self.n_virtual_caches.clone(),
            caches,
        }
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
    pub n_real_layers: usize,
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,
    #[config(default = "None")]
    pub n_real_caches: Option<usize>,
    #[config(default = "None")]
    pub n_virtual_caches: Option<(usize, Schedule)>,

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
        caches: Option<Mamba2BlockCaches<B>>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2BlockCaches<B>) {
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
}

#[derive(Module, Debug)]
pub struct Mamba2Layers<B: Backend> {
    pub n_real_layers: usize,
    pub n_virtual_layers: Option<(usize, Schedule)>,
    pub n_real_caches: Option<usize>,
    pub n_virtual_caches: Option<(usize, Schedule)>,
    /// # Shape
    /// [n_real_layers]
    pub real_layers: Vec<Mamba2Layer<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2LayersConfig {
    pub n_real_layers: usize,
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,
    #[config(default = "None")]
    pub n_real_caches: Option<usize>,
    #[config(default = "None")]
    pub n_virtual_caches: Option<(usize, Schedule)>,
    pub mamba_block: Mamba2BlockConfig,
}

impl Mamba2LayersConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Layers<B> {
        let mut real_layers = Vec::with_capacity(self.n_real_layers);
        for _ in 0..self.n_real_layers {
            let block_config = self.mamba_block.clone();
            let layer = Mamba2LayerConfig::new(block_config).init(device);
            real_layers.push(layer);
        }
        Mamba2Layers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            n_real_caches: self.n_real_caches.clone(),
            n_virtual_caches: self.n_virtual_caches.clone(),
            real_layers,
        }
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
        mut x: Tensor<B, 3>,
        caches: Option<Mamba2BlockCaches<B>>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2BlockCaches<B>) {
        let chunk_size = chunk_size.unwrap_or(256);

        let n_virtual_layers = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _schedule)| *l)
            .unwrap_or(
                // virtual layers fallback to the real layers
                self.n_real_layers,
            );

        let caches = caches.unwrap_or_else(|| {
            let device = &x.device();
            let [batch, _sequence, _d_model] = x.dims();
            let layer0_block = &self.real_layers[0].mamba_block;
            let [conv_dim, _, d_conv] = layer0_block.conv1d.weight.dims();

            Mamba2BlockCachesConfig::new(
                self.n_real_caches // there may be cache sharing
                    .unwrap_or(
                        // or each virtual layer has its own cache
                        n_virtual_layers,
                    ),
                Mamba2BlockCacheConfig {
                    batch,
                    d_state: layer0_block.d_state,
                    d_conv,
                    conv_dim,
                    headdim: layer0_block.headdim(),
                    nheads: layer0_block.nheads(),
                },
            )
            .with_n_virtual_caches(self.n_virtual_caches.clone())
            .init(device)
        });

        // assertions
        // resulting virtual caches must match to resulting virtual layers
        match &caches.n_virtual_caches {
            None => {
                assert_eq!(caches.n_real_caches, n_virtual_layers);
                assert_eq!(caches.n_real_caches, caches.caches.len());
            }
            Some((n_virtual_caches, _schedule)) => {
                assert_eq!(*n_virtual_caches, n_virtual_layers);
            }
        }

        let n_real_caches = caches.n_real_caches;
        let n_virtual_caches = caches.n_virtual_caches;
        let mut caches: Vec<Option<Mamba2BlockCache<B>>> =
            caches.caches.into_iter().map(|c| Some(c)).collect();

        for i in 0..n_virtual_layers {
            // use real layers by reference (clone)
            let layer_idx = if let Some((n_virtual_layers, schedule)) = &self.n_virtual_layers {
                schedule.real_idx(i, *n_virtual_layers, self.n_real_layers)
            } else {
                i
            };
            let layer = self.real_layers.get(layer_idx).unwrap();

            // re-use real caches by value (replacement)
            let cache_idx = if let Some((n_virtual_caches, schedule)) = &n_virtual_caches {
                schedule.real_idx(
                    i,
                    *n_virtual_caches,
                    // n_real_caches may constrain the amount of actual caches,
                    // otherwise it's uncontrained
                    self.n_real_caches.unwrap_or(*n_virtual_caches),
                )
            } else {
                i
            };
            let cache = core::mem::take(caches.get_mut(cache_idx).unwrap()).unwrap();

            let (x_, cache_) = layer.forward(x, Some(cache), chunk_size);
            x = x_;
            caches[cache_idx] = Some(cache_);
        }

        let caches = Mamba2BlockCaches {
            n_real_caches,
            n_virtual_caches,
            caches: caches.into_iter().map(|c| c.unwrap()).collect(),
        };

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
        cache: Option<Mamba2BlockCache<B>>,
        chunk_size: usize,
    ) -> (Tensor<B, 3>, Mamba2BlockCache<B>) {
        let [batch, sequence, d_model] = x.dims();

        let res = x.clone();
        let x = self.norm.forward(x);

        let (x, cache) = self.mamba_block.forward(x, cache, chunk_size);
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
            caches: Option<Mamba2BlockCaches<B>>,
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
            caches: Option<Mamba2BlockCaches<B>>,
        ) -> (Tensor<B, 2>, Mamba2BlockCaches<B>) {
            let n_virtual_layers = self
                .n_virtual_layers
                .as_ref()
                .map(|(l, _schedule)| *l)
                .unwrap_or(
                    // virtual layers fallback to the real layers
                    self.n_real_layers,
                );

            let caches = caches.unwrap_or_else(|| {
                let device = &x.device();
                let [batch, _d_model] = x.dims();
                let layer0_block = &self.real_layers[0].mamba_block;
                let [conv_dim, _, d_conv] = layer0_block.conv1d.weight.dims();

                Mamba2BlockCachesConfig::new(
                    self.n_real_caches // there may be cache sharing
                        .unwrap_or(
                            // or each virtual layer has its own cache
                            n_virtual_layers,
                        ),
                    Mamba2BlockCacheConfig {
                        batch,
                        d_state: layer0_block.d_state,
                        d_conv,
                        conv_dim,
                        headdim: layer0_block.headdim(),
                        nheads: layer0_block.nheads(),
                    },
                )
                .with_n_virtual_caches(self.n_virtual_caches.clone())
                .init(device)
            });

            // assertions
            // resulting virtual caches must match to resulting virtual layers
            match &caches.n_virtual_caches {
                None => {
                    assert_eq!(caches.n_real_caches, n_virtual_layers);
                    assert_eq!(caches.n_real_caches, caches.caches.len());
                }
                Some((n_virtual_caches, _schedule)) => {
                    assert_eq!(*n_virtual_caches, n_virtual_layers);
                }
            }

            let n_real_caches = caches.n_real_caches;
            let n_virtual_caches = caches.n_virtual_caches;
            let mut caches: Vec<Option<Mamba2BlockCache<B>>> =
                caches.caches.into_iter().map(|c| Some(c)).collect();

            for i in 0..n_virtual_layers {
                // use real layers by reference (clone)
                let layer_idx = if let Some((n_virtual_layers, schedule)) = &self.n_virtual_layers {
                    schedule.real_idx(i, *n_virtual_layers, self.n_real_layers)
                } else {
                    i
                };
                let layer = self.real_layers.get(layer_idx).unwrap();

                // re-use real caches by value (replacement)
                let cache_idx = if let Some((n_virtual_caches, schedule)) = &n_virtual_caches {
                    schedule.real_idx(
                        i,
                        *n_virtual_caches,
                        // n_real_caches may constrain the amount of actual caches,
                        // otherwise it's uncontrained
                        self.n_real_caches.unwrap_or(*n_virtual_caches),
                    )
                } else {
                    i
                };
                let cache = core::mem::take(caches.get_mut(cache_idx).unwrap()).unwrap();

                let (x_, cache_) = layer.step(x, Some(cache));
                x = x_;
                caches[cache_idx] = Some(cache_);
            }

            let caches = Mamba2BlockCaches {
                n_real_caches,
                n_virtual_caches,
                caches: caches.into_iter().map(|c| c.unwrap()).collect(),
            };

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
            cache: Option<Mamba2BlockCache<B>>,
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
