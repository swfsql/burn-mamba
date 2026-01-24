use crate::mamba2::*;
use crate::schedule::Schedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Mamba2Layers<B: Backend> {
    pub n_real_layers: usize,
    pub n_virtual_layers: Option<(usize, Schedule)>,
    /// # Shape
    /// [n_real_layers]
    pub real_layers: Vec<Mamba2Layer<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2LayersConfig {
    pub n_real_layers: usize,
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,
    pub mamba_block: Mamba2Config,
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
        caches: Option<Mamba2Caches<B>>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2Caches<B>) {
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

            Mamba2CachesConfig::new(
                n_virtual_layers,
                Mamba2CacheConfig {
                    batch,
                    d_state: layer0_block.d_state,
                    d_conv,
                    conv_dim,
                    headdim: layer0_block.headdim(),
                    nheads: layer0_block.nheads(),
                },
            )
            .init(device)
        });

        // assertions
        assert_eq!(
            caches.caches.len(),
            n_virtual_layers,
            "layers in forward() currently cannot share caches"
        );

        let mut caches: Vec<Option<Mamba2Cache<B>>> =
            caches.caches.into_iter().map(|c| Some(c)).collect();

        for i in 0..n_virtual_layers {
            // use real layers by reference (clone)
            let layer_idx = if let Some((n_virtual_layers, schedule)) = &self.n_virtual_layers {
                schedule.real_idx(i, *n_virtual_layers, self.n_real_layers)
            } else {
                i
            };
            let layer = self.real_layers.get(layer_idx).unwrap();

            let cache_idx = i;
            let cache = core::mem::take(caches.get_mut(cache_idx).unwrap()).unwrap();

            let (x_, cache_) = layer.forward(x, Some(cache), chunk_size);
            x = x_;
            caches[cache_idx] = Some(cache_);
        }

        let caches = Mamba2Caches {
            caches: caches.into_iter().map(|c| c.unwrap()).collect(),
        };

        (x, caches)
    }

    /// See also [`Self::forward`].
    ///
    /// # Shapes
    ///   - Input [batch, d_model]
    ///   - Output [batch, d_model]
    pub fn step(
        &self,
        mut x: Tensor<B, 2>,
        caches: Option<Mamba2Caches<B>>,
    ) -> (Tensor<B, 2>, Mamba2Caches<B>) {
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

            Mamba2CachesConfig::new(
                n_virtual_layers,
                Mamba2CacheConfig {
                    batch,
                    d_state: layer0_block.d_state,
                    d_conv,
                    conv_dim,
                    headdim: layer0_block.headdim(),
                    nheads: layer0_block.nheads(),
                },
            )
            .init(device)
        });

        // assertions
        assert_eq!(
            caches.caches.len(),
            n_virtual_layers,
            "layers in step() currently cannot share caches"
        );

        let mut caches: Vec<Option<Mamba2Cache<B>>> =
            caches.caches.into_iter().map(|c| Some(c)).collect();

        for i in 0..n_virtual_layers {
            // use real layers by reference (clone)
            let layer_idx = if let Some((n_virtual_layers, schedule)) = &self.n_virtual_layers {
                schedule.real_idx(i, *n_virtual_layers, self.n_real_layers)
            } else {
                i
            };
            let layer = self.real_layers.get(layer_idx).unwrap();

            let cache_idx = i;
            let cache = core::mem::take(caches.get_mut(cache_idx).unwrap()).unwrap();

            let (x_, cache_) = layer.step(x, Some(cache));
            x = x_;
            caches[cache_idx] = Some(cache_);
        }

        let caches = Mamba2Caches {
            caches: caches.into_iter().map(|c| c.unwrap()).collect(),
        };

        (x, caches)
    }
}

#[derive(Module, Debug)]
pub struct Mamba2Layer<B: Backend> {
    pub norm: RmsNorm<B>,
    pub mamba_block: Mamba2<B>,
}

#[derive(Config, Debug)]
pub struct Mamba2LayerConfig {
    pub mamba_block: Mamba2Config,
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
        cache: Option<Mamba2Cache<B>>,
        chunk_size: usize,
    ) -> (Tensor<B, 3>, Mamba2Cache<B>) {
        let [batch, sequence, d_model] = x.dims();

        let res = x.clone();
        let x = self.norm.forward(x);

        let (x, cache) = self.mamba_block.forward(x, cache, chunk_size);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        (x + res, cache)
    }

    /// See also [`Self::forward`].
    ///
    /// # Shapes
    ///   - Input [batch, d_model]
    ///   - Output [batch, d_model]
    pub fn step(
        &self,
        x: Tensor<B, 2>,
        cache: Option<Mamba2Cache<B>>,
    ) -> (Tensor<B, 2>, Mamba2Cache<B>) {
        let [batch, d_model] = x.dims();

        let res = x.clone();
        let x = self.norm.forward(x);
        let (x, cache) = self.mamba_block.step(x, cache);
        debug_assert_eq!([batch, d_model], x.dims());

        (x + res, cache)
    }
}
