//! # Naive bidirectional Mamba-3 layer stack
//!
//! For non-autoregressive tasks.  Layers are consumed in **pairs**: each pair
//! runs a straight (→) pass and a reversed (←) pass (via `flip` on the sequence
//! axis, then flip back) and merges the two with an [`OutputMerge`].  The block
//! itself is unmodified — only how its two passes are scheduled and combined is
//! bidirectional.  Pairing is driven by a [`BidiSchedule`].

use crate::mamba3::bidi::naive::{OutputMerge, OutputMergeConfig};
use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::prelude::*;
use crate::mamba3::single_ssd::prelude::*;
use crate::schedule::BidiSchedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::backend::Backend;
use burn::prelude::*;

/// A stack of bidirectional Mamba-3 layer pairs with optional virtual-layer
/// scheduling.
#[derive(Module, Debug)]
pub struct Mamba3BidiLayers {
    /// Number of real (weight-bearing) layers; must be even (used in pairs).
    pub n_real_layers: usize,
    /// Optional `(n_virtual_layers, schedule)` for weight-sharing.  `module(skip)`
    /// so Burn does not treat it as a trainable parameter.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// The weight-bearing layer instances, length `n_real_layers`.
    pub real_layers: Vec<Mamba3Layer>,
    /// When `true`, the first virtual pair's residual is scaled to zero.
    pub ignore_first_residual: bool,
    /// When `true`, the last virtual pair's residual is scaled to zero.
    pub ignore_last_residual: bool,
    /// One direction-merge per pair, length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMerge>,
}

/// Configuration / factory for [`Mamba3BidiLayers`].
#[derive(Config, Debug)]
pub struct Mamba3BidiLayersConfig {
    /// Number of distinct weight sets to allocate (must be even).
    pub n_real_layers: usize,
    /// Optional virtual-layer scheduling.  See [`Mamba3BidiLayers`].
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// Configuration shared by all Mamba-3 blocks in the stack.
    pub mamba_block: Mamba3Config,
    /// See [`Mamba3BidiLayers::ignore_first_residual`].
    #[config(default = false)]
    pub ignore_first_residual: bool,
    /// See [`Mamba3BidiLayers::ignore_last_residual`].
    #[config(default = false)]
    pub ignore_last_residual: bool,
    /// One [`OutputMergeConfig`] per pair, length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMergeConfig>,
}

impl Mamba3BidiLayersConfig {
    /// Returns the initialized model.
    pub fn init(&self, device: &Device) -> Mamba3BidiLayers {
        let d_model = self.mamba_block.d_model;
        let mut real_layers = Vec::with_capacity(self.n_real_layers);
        let mut outputs_merge = Vec::with_capacity(self.n_real_layers);
        for _ in 0..self.n_real_layers {
            let block_config = self.mamba_block.clone();
            let layer = Mamba3LayerConfig::new(block_config).init(device);
            real_layers.push(layer);
        }
        for i in 0..self.n_real_layers / 2 {
            let output_merge = self.outputs_merge.get(i).unwrap().init(d_model, device);
            outputs_merge.push(output_merge);
        }

        Mamba3BidiLayers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            outputs_merge,
        }
    }
}

impl Mamba3BidiLayers {
    /// # Shapes
    ///   - Input `[batch, sequence, d_model]`
    ///   - Output `[batch, sequence, d_model]`
    pub fn forward(
        &self,
        mut x: Tensor<3>,
        caches: Option<Mamba3Caches>,
        ssd_path: Mamba3SsdPath,
    ) -> (Tensor<3>, Mamba3Caches) {
        let n_virtual_layers = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _schedule)| {
                assert!(l.is_multiple_of(2), "Bidi virtual layers are used in pairs");
                *l
            })
            .unwrap_or({
                assert!(
                    self.n_real_layers.is_multiple_of(2),
                    "Bidi layers are used in pairs"
                );
                // virtual layers fallback to the real layers
                self.n_real_layers
            });

        // Lazily allocate zero caches the first time (e.g. during training or
        // the first prefill call). The single-ssd pathway supports both rotation
        // kinds (see `Mamba3::forward_single_ssd`); mirrors `Mamba3Layers::forward`.
        let caches = caches.unwrap_or_else(|| {
            self.make_zero_caches_single_ssd_3d(&x, n_virtual_layers)
                .into()
        });

        // assertions
        assert_eq!(
            caches.caches_len(),
            n_virtual_layers,
            "straight and reverse layers in forward() currently cannot share caches"
        );

        // Wrap each cache slot into an `Option` so we can `take` it in the
        // loop without cloning (Burn tensors are reference-counted).
        let mut caches: Vec<Option<Mamba3Cache>> = caches.into_options();

        for i in 0..n_virtual_layers / 2 {
            // use real layers by reference (clone)
            let (straight_i, reverse_i) = (i * 2, i * 2 + 1);
            let (straight_layer_idx, reverse_layer_idx) =
                if let Some((n_virtual_layers, bidi_schedule)) = &self.n_virtual_layers {
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

            let bidi_pair = Mamba3BidiLayerPair {
                straight_norm: straight_layer.norm.clone(),
                reverse_norm: reverse_layer.norm.clone(),
                straight_block: straight_layer.mamba_block.clone(),
                reverse_block: reverse_layer.mamba_block.clone(),
                output_merge: self.outputs_merge.get(i).unwrap().clone(),
                residual_scale,
            };

            let (x_, straight_cache_, reverse_cache_) = bidi_pair.forward(
                x,
                Some(straight_cache),
                Some(reverse_cache),
                ssd_path.clone(),
            );
            x = x_;
            caches[straight_cache_idx] = Some(straight_cache_);
            caches[reverse_cache_idx] = Some(reverse_cache_);
        }

        let caches = Mamba3Caches::from_options(caches);
        (x, caches)
    }

    /// Build zero-initialised caches from a 3-dimensional input tensor `[batch, sequence, d_model]`.
    pub fn make_zero_caches_double_ssd_3d(
        &self,
        x: &Tensor<3>,
        n_virtual: usize,
    ) -> Mamba3DoubleSsdCaches {
        use crate::mamba3::layer::*;
        make_zero_caches_double_ssd_3d(&self.real_layers[0].mamba_block, x, n_virtual)
    }

    /// Build zero-initialised caches from a 2-dimensional input tensor `[batch, d_model]`.
    pub fn make_zero_caches_double_ssd_2d(
        &self,
        x: &Tensor<2>,
        n_virtual: usize,
    ) -> Mamba3DoubleSsdCaches {
        use crate::mamba3::layer::*;
        make_zero_caches_double_ssd_2d(&self.real_layers[0].mamba_block, x, n_virtual)
    }

    /// Build zero-initialised caches from a 3-dimensional input tensor `[batch, sequence, d_model]`.
    pub fn make_zero_caches_single_ssd_3d(
        &self,
        x: &Tensor<3>,
        n_virtual: usize,
    ) -> Mamba3SingleSsdCaches {
        use crate::mamba3::layer::*;
        make_zero_caches_single_ssd_3d(&self.real_layers[0].mamba_block, x, n_virtual)
    }

    /// Build zero-initialised caches from a 2-dimensional input tensor `[batch, d_model]`.
    pub fn make_zero_caches_single_ssd_2d(
        &self,
        x: &Tensor<2>,
        n_virtual: usize,
    ) -> Mamba3SingleSsdCaches {
        use crate::mamba3::layer::*;
        make_zero_caches_single_ssd_2d(&self.real_layers[0].mamba_block, x, n_virtual)
    }
}

/// A single bidirectional pair: a straight (→) and a reversed (←) Pre-LN block
/// whose outputs are merged, then added to the (scaled) residual.
#[derive(Module, Debug)]
pub struct Mamba3BidiLayerPair {
    /// Pre-norm for the straight pass.
    pub straight_norm: RmsNorm,
    /// Pre-norm for the reversed pass.
    pub reverse_norm: RmsNorm,
    /// The Mamba-3 block run left-to-right.
    pub straight_block: Mamba3,
    /// The Mamba-3 block run right-to-left (over the flipped sequence).
    pub reverse_block: Mamba3,
    /// Merge strategy combining the two directions.
    pub output_merge: OutputMerge,
    /// Residual scale (0.0 suppresses the skip connection, else 1.0).
    pub residual_scale: f32,
}

/// Configuration / factory for [`Mamba3BidiLayerPair`].
#[derive(Config, Debug)]
pub struct Mamba3BidiLayerPairConfig {
    /// Configuration for the straight-direction block.
    pub straight_block: Mamba3Config,
    /// Configuration for the reverse-direction block.
    pub reverse_block: Mamba3Config,
    /// See [`Mamba3BidiLayerPair::residual_scale`].
    #[config(default = 1.0)]
    pub residual_scale: f32,
    /// How to merge the two directions.
    pub output_merge: OutputMergeConfig,
}

impl Mamba3BidiLayerPairConfig {
    /// Returns the initialized model.
    pub fn init(&self, device: &Device) -> Mamba3BidiLayerPair {
        let d_model = self.straight_block.d_model;
        Mamba3BidiLayerPair {
            straight_norm: RmsNormConfig::new(self.straight_block.d_model).init(device),
            reverse_norm: RmsNormConfig::new(self.reverse_block.d_model).init(device),
            straight_block: self.straight_block.init(device),
            reverse_block: self.reverse_block.init(device),
            residual_scale: self.residual_scale,
            output_merge: self.output_merge.init(d_model, device),
        }
    }
}

impl Mamba3BidiLayerPair {
    /// # Shapes
    ///   - Input `[batch, sequence, d_model]`
    ///   - Output.0 `[batch, sequence, d_model]`
    pub fn forward(
        &self,
        x: Tensor<3>,
        straight_cache: Option<Mamba3Cache>,
        reverse_cache: Option<Mamba3Cache>,
        ssd_path: Mamba3SsdPath,
    ) -> (Tensor<3>, Mamba3Cache, Mamba3Cache) {
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
        let (x, straight_cache) = self
            .straight_block
            .forward(x, straight_cache, ssd_path.clone());
        assert_eq!([batch, sequence, d_model], x.dims());

        // reverse reads inputs as:
        // t₀        x₃<
        // t₁      x₂<x₃<
        // t₂   x₁<x₂<x₃<
        // t₃ x₀<x₁<x₂<x₃<
        let (x_rev, reverse_cache) = self.reverse_block.forward(x_rev, reverse_cache, ssd_path);
        assert_eq!([batch, sequence, d_model], x_rev.dims());

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
