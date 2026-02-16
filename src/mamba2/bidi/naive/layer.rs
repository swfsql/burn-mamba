use crate::mamba2::bidi::naive::{OutputMerge, OutputMergeConfig};
use crate::mamba2::*;
use crate::schedule::BidiSchedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::module::Ignored;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Mamba2BidiLayers<B: Backend> {
    pub n_real_layers: usize,
    pub n_virtual_layers: Ignored<Option<(usize, BidiSchedule)>>,
    /// # Shape
    /// - [n_real_layers]
    pub real_layers: Vec<Mamba2Layer<B>>,
    pub ignore_first_residual: bool,
    pub ignore_last_residual: bool,
    /// # Shape
    /// - [n_real_layers / 2]
    pub outputs_merge: Vec<OutputMerge<B>>,
}

#[derive(Config, Debug)]
pub struct Mamba2BidiLayersConfig {
    pub n_real_layers: usize,
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    pub mamba_block: Mamba2Config,
    #[config(default = false)]
    pub ignore_first_residual: bool,
    #[config(default = false)]
    pub ignore_last_residual: bool,

    /// # Shape
    /// - [n_real_layers / 2]
    pub outputs_merge: Vec<OutputMergeConfig>,
}

impl Mamba2BidiLayersConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2BidiLayers<B> {
        let d_model = self.mamba_block.d_model;
        let mut real_layers = Vec::with_capacity(self.n_real_layers);
        let mut outputs_merge = Vec::with_capacity(self.n_real_layers);
        for _ in 0..self.n_real_layers {
            let block_config = self.mamba_block.clone();
            let layer = Mamba2LayerConfig::new(block_config).init(device);
            real_layers.push(layer);
        }
        for i in 0..self.n_real_layers / 2 {
            let output_merge = self.outputs_merge.get(i).unwrap().init(d_model, device);
            outputs_merge.push(output_merge);
        }

        Mamba2BidiLayers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: Ignored(self.n_virtual_layers.clone()),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            outputs_merge,
        }
    }
}

impl<B: Backend> Mamba2BidiLayers<B> {
    /// `chunk_size`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        caches: Option<Mamba2Caches<B>>,
        // straight_caches: Option<Mamba2Caches<B>>,
        // reverse_caches: Option<Mamba2Caches<B>>,
        chunk_size: Option<usize>,
    ) -> (Tensor<B, 3>, Mamba2Caches<B>) {
        let chunk_size = chunk_size.unwrap_or(256);

        let n_virtual_layers = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _schedule)| {
                assert!(l % 2 == 0, "Bidi virtual layers are used in pairs");
                *l
            })
            .unwrap_or({
                assert!(self.n_real_layers % 2 == 0, "Bidi layers are used in pairs");
                // virtual layers fallback to the real layers
                self.n_real_layers
            });

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
            "straight and reverse layers in forward() currently cannot share caches"
        );

        let mut caches: Vec<Option<Mamba2Cache<B>>> =
            caches.caches.into_iter().map(|c| Some(c)).collect();

        for i in 0..n_virtual_layers / 2 {
            // use real layers by reference (clone)
            let (straight_i, reverse_i) = (i * 2, i * 2 + 1);
            let (straight_layer_idx, reverse_layer_idx) =
                if let Some((n_virtual_layers, bidi_schedule)) = &self.n_virtual_layers.0 {
                    (
                        bidi_schedule.real_idx(straight_i, *n_virtual_layers, self.n_real_layers),
                        bidi_schedule.real_idx(reverse_i, *n_virtual_layers, self.n_real_layers),
                    )
                } else {
                    (straight_i, reverse_i)
                };
            let straight_layer = self.real_layers.get(straight_layer_idx).unwrap();
            let reverse_layer = self.real_layers.get(reverse_layer_idx).unwrap();

            let straight_cache_idx = straight_i;
            let reverse_cache_idx = reverse_i;
            let straight_cache =
                core::mem::take(caches.get_mut(straight_cache_idx).unwrap()).unwrap();
            let reverse_cache =
                core::mem::take(caches.get_mut(reverse_cache_idx).unwrap()).unwrap();

            let residual_scale = if (self.ignore_first_residual && i == 0)
                || (self.ignore_last_residual && i + 1 == n_virtual_layers / 2)
            {
                0.0
            } else {
                1.0
            };

            let bidi_pair = Mamba2BidiLayerPair {
                straight_norm: straight_layer.norm.clone(),
                reverse_norm: reverse_layer.norm.clone(),
                straight_block: straight_layer.mamba_block.clone(),
                reverse_block: reverse_layer.mamba_block.clone(),
                output_merge: self.outputs_merge.get(i).unwrap().clone(),
                residual_scale,
            };

            let (x_, straight_cache_, reverse_cache_) =
                bidi_pair.forward(x, Some(straight_cache), Some(reverse_cache), chunk_size);
            x = x_;
            caches[straight_cache_idx] = Some(straight_cache_);
            caches[reverse_cache_idx] = Some(reverse_cache_);
        }

        let caches = Mamba2Caches {
            caches: caches.into_iter().map(|c| c.unwrap()).collect(),
        };

        (x, caches)
    }
}

#[derive(Module, Debug)]
pub struct Mamba2BidiLayerPair<B: Backend> {
    pub straight_norm: RmsNorm<B>,
    pub reverse_norm: RmsNorm<B>,
    pub straight_block: Mamba2<B>,
    pub reverse_block: Mamba2<B>,
    pub output_merge: OutputMerge<B>,
    pub residual_scale: f32,
}

#[derive(Config, Debug)]
pub struct Mamba2BidiLayerPairConfig {
    pub straight_block: Mamba2Config,
    pub reverse_block: Mamba2Config,
    #[config(default = 1.0)]
    pub residual_scale: f32,
    pub output_merge: OutputMergeConfig,
}

impl Mamba2BidiLayerPairConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2BidiLayerPair<B> {
        let d_model = self.straight_block.d_model;
        Mamba2BidiLayerPair {
            straight_norm: RmsNormConfig::new(self.straight_block.d_model).init(device),
            reverse_norm: RmsNormConfig::new(self.reverse_block.d_model).init(device),
            straight_block: self.straight_block.init(device),
            reverse_block: self.reverse_block.init(device),
            residual_scale: self.residual_scale,
            output_merge: self.output_merge.init(d_model, device),
        }
    }
}

impl<B: Backend> Mamba2BidiLayerPair<B> {
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output.0 [batch, sequence, d_model]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        straight_cache: Option<Mamba2Cache<B>>,
        reverse_cache: Option<Mamba2Cache<B>>,
        chunk_size: usize,
    ) -> (Tensor<B, 3>, Mamba2Cache<B>, Mamba2Cache<B>) {
        let [batch, sequence, d_model] = x.dims();

        let res = x.clone() * self.residual_scale;

        // x is read as >x₀>x₁>x₂>x₃
        // x_rev is read as >x₃>x₂>x₁>x₀, i.e. x₀<x₁<x₂<x₃<
        let x_rev = x.clone().flip([1]); // flip sequence-wise

        // each layer (as stored) carries their own norm,
        // but perhaps it's redundant to apply two of them after the flip.
        // i.e. maybe a single norm applied before the flip is better
        let x = self.straight_norm.forward(x);
        let x_rev = self.reverse_norm.forward(x_rev);

        // straight reads inputs as:
        // t₀ >x₀
        // t₁ >x₀>x₁
        // t₂ >x₀>x₁>x₂
        // t₃ >x₀>x₁>x₂>x₃
        let (x, straight_cache) = self.straight_block.forward(x, straight_cache, chunk_size);
        debug_assert_eq!([batch, sequence, d_model], x.dims());

        // reverse reads inputs as:
        // t₀        x₃<
        // t₁      x₂<x₃<
        // t₂   x₁<x₂<x₃<
        // t₃ x₀<x₁<x₂<x₃<
        let (x_rev, reverse_cache) = self.reverse_block.forward(x_rev, reverse_cache, chunk_size);
        debug_assert_eq!([batch, sequence, d_model], x_rev.dims());

        // re-align the reversed read:
        // t₀ x₀<x₁<x₂<x₃<
        // t₁   x₁<x₂<x₃<
        // t₂      x₂<x₃<
        // t₃        x₃<
        let x_rev = x_rev.flip([1]);

        // merge both reads:
        // t₀ merge(>x₀ , x₀<x₁<x₂<x₃<)
        // t₁ merge(>x₀>x₁ , x₁<x₂<x₃<)
        // t₂ merge(>x₀>x₁>x₂ , x₂<x₃<)
        // t₃ merge(>x₀>x₁>x₂>x₃ , x₃<)
        let merged = self.output_merge.forward(x, x_rev);

        (merged + res, straight_cache, reverse_cache)
    }
}
