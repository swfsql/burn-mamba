//! # Mamba-1 Layer and Layer Stack
//!
//! A **Mamba-1 layer** is the standard Pre-LN residual block used throughout
//! the network.  It wraps a single [`Mamba1`] SSM block with an RMSNorm
//! (applied to the input, *before* the block) and adds the input back as a
//! residual connection:
//!
//! ```text
//!   y = x + Mamba1( RMSNorm(x) )
//! ```
//!
//! ## Virtual layers
//!
//! [`Mamba1Layers`] supports *virtual layers*: a larger logical depth achieved
//! by cycling through a smaller set of *real* (weight-bearing) layers according
//! to a [`Schedule`].  Each virtual layer keeps its **own cache** but shares
//! the underlying parameters.  See [`crate::mamba2::layer`] for the structure
//! this mirrors.

use crate::mamba1::prelude::*;
use crate::schedule::Schedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba1Layers  (the full layer stack)
// ---------------------------------------------------------------------------

/// A stack of Mamba-1 layers with optional virtual-layer scheduling.
///
/// The stack maintains `n_real_layers` distinct weight sets but can execute
/// `n_virtual_layers` logical forward passes, cycling through weights
/// according to the provided [`Schedule`].
#[derive(Module, Debug)]
pub struct Mamba1Layers<B: Backend> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,

    /// Optional `(n_virtual_layers, schedule)` for weight-sharing.
    ///
    /// When `None`, the virtual layer count falls back to `n_real_layers` (no
    /// sharing).  Marked `module(skip)` so Burn does not treat it as a
    /// trainable parameter.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// The actual weight-bearing layer instances.
    ///
    /// Length: `n_real_layers`.
    pub real_layers: Vec<Mamba1Layer<B>>,

    /// When `true`, the residual connection of the **first** virtual layer is
    /// scaled to zero.
    pub ignore_first_residual: bool,

    /// When `true`, the residual connection of the **last** virtual layer is
    /// scaled to zero.
    pub ignore_last_residual: bool,
}

/// Configuration / factory for [`Mamba1Layers`].
#[derive(Config, Debug)]
pub struct Mamba1LayersConfig {
    /// Number of distinct weight sets to allocate.
    pub n_real_layers: usize,

    /// Optional virtual-layer scheduling.  See [`Mamba1Layers`] for details.
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// Configuration shared by all Mamba-1 blocks in the stack.
    pub mamba_block: Mamba1Config,

    /// See [`Mamba1Layers::ignore_first_residual`].
    #[config(default = false)]
    pub ignore_first_residual: bool,

    /// See [`Mamba1Layers::ignore_last_residual`].
    #[config(default = false)]
    pub ignore_last_residual: bool,
}

impl Mamba1LayersConfig {
    /// Allocate and initialise all layers on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1Layers<B> {
        let real_layers = (0..self.n_real_layers)
            .map(|_| Mamba1LayerConfig::new(self.mamba_block.clone()).init(device))
            .collect();

        Mamba1Layers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
        }
    }
}

impl<B: Backend> Mamba1Layers<B> {
    // -----------------------------------------------------------------------
    // forward  (full sequence — used for training / prefill)
    // -----------------------------------------------------------------------

    /// Process a full sequence through every (virtual) layer.
    ///
    /// If `caches` is `None`, zero-initialised caches are created automatically.
    ///
    /// # Returns
    /// `(output, updated_caches)` where `output` has shape
    /// `[batch, sequence, d_model]`.
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        caches: Option<Mamba1Caches<B>>,
    ) -> (Tensor<B, 3>, Mamba1Caches<B>) {
        let n_virtual_layers = self.n_virtual_count();
        let caches = caches.unwrap_or_else(|| self.make_zero_caches_3d(&x, n_virtual_layers));

        assert_eq!(
            caches.caches_len(),
            n_virtual_layers,
            "cache count must match the number of virtual layers; \
             layers in forward() cannot share caches"
        );

        let mut caches: Vec<Option<Mamba1Cache<B>>> = caches.into_options();

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_virtual_layers {
            let layer_idx = self.real_idx(i);
            let layer = &self.real_layers[layer_idx];
            let residual_scale = self.residual_scale(i, n_virtual_layers);

            let cache = caches[i].take().unwrap();
            let (x_, cache_) = layer.forward(x, Some(cache), residual_scale);
            x = x_;
            caches[i] = Some(cache_);
        }

        let caches = Mamba1Caches::from_options(caches);
        (x, caches)
    }

    // -----------------------------------------------------------------------
    // step  (single token — used for autoregressive decoding)
    // -----------------------------------------------------------------------

    /// Process a **single token** through every (virtual) layer.
    ///
    /// # Returns
    /// `(output, updated_caches)` where `output` has shape `[batch, d_model]`.
    pub fn step(
        &self,
        mut x: Tensor<B, 2>,
        caches: Option<Mamba1Caches<B>>,
    ) -> (Tensor<B, 2>, Mamba1Caches<B>) {
        let n_virtual_layers = self.n_virtual_count();
        let caches = caches.unwrap_or_else(|| self.make_zero_caches_2d(&x, n_virtual_layers));

        assert_eq!(
            caches.caches_len(),
            n_virtual_layers,
            "cache count must match the number of virtual layers; \
             layers in step() cannot share caches"
        );

        let mut caches: Vec<Option<Mamba1Cache<B>>> = caches.into_options();

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_virtual_layers {
            let layer_idx = self.real_idx(i);
            let layer = &self.real_layers[layer_idx];
            let residual_scale = self.residual_scale(i, n_virtual_layers);

            let cache = caches[i].take().unwrap();
            let (x_, cache_) = layer.step(x, cache, residual_scale);
            x = x_;
            caches[i] = Some(cache_);
        }

        let caches = Mamba1Caches::from_options(caches);
        (x, caches)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Effective number of forward passes (virtual layers).
    fn n_virtual_count(&self) -> usize {
        self.n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers)
    }

    /// Map a virtual layer index to the corresponding real layer index using
    /// the configured schedule (or identity when no schedule is set).
    fn real_idx(&self, virtual_idx: usize) -> usize {
        if let Some((n_virtual_layers, schedule)) = &self.n_virtual_layers {
            schedule.real_idx(virtual_idx, *n_virtual_layers, self.n_real_layers)
        } else {
            virtual_idx
        }
    }

    /// Returns 0.0 if this layer's residual should be suppressed, else 1.0.
    fn residual_scale(&self, i: usize, n_virtual: usize) -> f32 {
        let is_first = self.ignore_first_residual && i == 0;
        let is_last = self.ignore_last_residual && i + 1 == n_virtual;
        if is_first || is_last { 0.0 } else { 1.0 }
    }

    /// Build zero-initialised caches from a 3-dimensional input tensor
    /// `[batch, sequence, d_model]`.
    fn make_zero_caches_3d(&self, x: &Tensor<B, 3>, n_virtual: usize) -> Mamba1Caches<B> {
        let [batch, _sequence, _d_model] = x.dims();
        self.make_zero_caches(batch, n_virtual, &x.device())
    }

    /// Build zero-initialised caches from a 2-dimensional input tensor
    /// `[batch, d_model]`.
    fn make_zero_caches_2d(&self, x: &Tensor<B, 2>, n_virtual: usize) -> Mamba1Caches<B> {
        let [batch, _d_model] = x.dims();
        self.make_zero_caches(batch, n_virtual, &x.device())
    }

    /// Derive the cache shapes from the first real block's parameters and
    /// allocate `n_virtual` zero caches.
    fn make_zero_caches(
        &self,
        batch: usize,
        n_virtual: usize,
        device: &B::Device,
    ) -> Mamba1Caches<B> {
        let block0 = &self.real_layers[0].mamba_block;
        let [d_inner, state_rank] = block0.a_log.dims();
        let [_, _, conv_kernel] = block0.conv1d.weight.dims();

        Mamba1CachesConfig::new(
            n_virtual,
            Mamba1CacheConfig {
                batch,
                state_rank,
                conv_kernel,
                d_inner,
            },
        )
        .init(device)
    }
}

// ---------------------------------------------------------------------------
// Mamba1Layer  (single Pre-LN residual block)
// ---------------------------------------------------------------------------

/// A single Mamba-1 residual block:
///
/// ```text
///   output = x·scale + Mamba1( RMSNorm(x) )
/// ```
///
/// where `scale` is 1.0 normally and 0.0 when the residual connection is
/// intentionally suppressed by the layer stack configuration.
#[derive(Module, Debug)]
pub struct Mamba1Layer<B: Backend> {
    /// Pre-norm applied to the input before the SSM block.
    pub norm: RmsNorm<B>,
    /// The Mamba-1 SSM block (see [`Mamba1`]).
    pub mamba_block: Mamba1<B>,
}

/// Configuration / factory for [`Mamba1Layer`].
#[derive(Config, Debug)]
pub struct Mamba1LayerConfig {
    /// Configuration for the inner Mamba-1 block.
    pub mamba_block: Mamba1Config,
}

impl Mamba1LayerConfig {
    /// Allocate and initialise the layer on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1Layer<B> {
        Mamba1Layer {
            norm: RmsNormConfig::new(self.mamba_block.d_model).init(device),
            mamba_block: self.mamba_block.init(device),
        }
    }
}

impl<B: Backend> Mamba1Layer<B> {
    /// Run the Pre-LN residual block over a full sequence.
    ///
    /// Computes `output = x · residual_scale + Mamba1( RMSNorm(x) )`.
    ///
    /// # Shapes
    ///   - `x`    : `[batch, sequence, d_model]`
    ///   - output : `[batch, sequence, d_model]`
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cache: Option<Mamba1Cache<B>>,
        residual_scale: f32,
    ) -> (Tensor<B, 3>, Mamba1Cache<B>) {
        let [batch, sequence, d_model] = x.dims();

        let res = x.clone() * residual_scale;
        let normed = self.norm.forward(x);

        let (x, cache) = self.mamba_block.forward(normed, cache);
        assert_eq!([batch, sequence, d_model], x.dims());

        (x + res, cache)
    }

    /// Run the Pre-LN residual block for a **single** decoding step.
    ///
    /// Computes `output = x · residual_scale + Mamba1.step( RMSNorm(x) )`.
    ///
    /// # Shapes
    ///   - `x`    : `[batch, d_model]`
    ///   - output : `[batch, d_model]`
    pub fn step(
        &self,
        x: Tensor<B, 2>,
        cache: Mamba1Cache<B>,
        residual_scale: f32,
    ) -> (Tensor<B, 2>, Mamba1Cache<B>) {
        let [batch, d_model] = x.dims();

        let res = x.clone() * residual_scale;
        let normed = self.norm.forward(x);

        let (x, cache) = self.mamba_block.step(normed, cache);
        assert_eq!([batch, d_model], x.dims());

        (x + res, cache)
    }
}
