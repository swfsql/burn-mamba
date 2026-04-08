//! # Mamba-2 Layer and Layer Stack
//!
//! A **Mamba-2 layer** is the standard Pre-LN residual block used throughout
//! the network.  It wraps a single [`Mamba2`] SSM block with an RMSNorm
//! (applied to the input, *before* the block) and adds the input back as a
//! residual connection:
//!
//! ```text
//!   y = x + Mamba2( RMSNorm(x) )
//! ```
//!
//! This matches the architecture described in §5 of the Mamba-2 paper and is
//! identical in structure to Pre-LN Transformer layers.
//!
//! ## Virtual layers
//!
//! [`Mamba2Layers`] supports *virtual layers*: a larger logical depth achieved
//! by cycling through a smaller set of *real* (weight-bearing) layers
//! according to a [`Schedule`].  For example, 48 virtual layers over 12 real
//! layers repeats each weight set 4 times.  Each virtual layer still has its
//! **own cache** (the hidden state evolves independently), but shares the
//! underlying parameters.
//!
//! ## Residual scale
//!
//! The first and/or last residual connection in the stack can optionally be
//! zeroed out (`ignore_first_residual` / `ignore_last_residual`), which is
//! useful when composing Mamba-2 blocks with other module types (e.g. in a
//! hybrid Mamba-2 + attention architecture where neighbouring blocks already
//! carry residuals).

use crate::mamba2::prelude::*;
use crate::schedule::Schedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba2Layers  (the full layer stack)
// ---------------------------------------------------------------------------

/// A stack of Mamba-2 layers with optional virtual-layer scheduling.
///
/// The stack maintains `n_real_layers` distinct weight sets but can execute
/// `n_virtual_layers` logical forward passes, cycling through weights
/// according to the provided [`Schedule`].
#[derive(Module, Debug)]
pub struct Mamba2Layers<B: Backend> {
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
    pub real_layers: Vec<Mamba2Layer<B>>,

    /// When `true`, the residual connection of the **first** virtual layer is
    /// scaled to zero (i.e. the first block acts as a pure projection, not a
    /// residual update).
    pub ignore_first_residual: bool,

    /// When `true`, the residual connection of the **last** virtual layer is
    /// scaled to zero.
    pub ignore_last_residual: bool,
}

/// Configuration / factory for [`Mamba2Layers`].
#[derive(Config, Debug)]
pub struct Mamba2LayersConfig {
    /// Number of distinct weight sets to allocate.
    pub n_real_layers: usize,

    /// Optional virtual-layer scheduling.  See [`Mamba2Layers`] for details.
    #[config(default = "None")]
    pub n_virtual_layers: Option<(usize, Schedule)>,

    /// Configuration shared by all Mamba-2 blocks in the stack.
    pub mamba_block: Mamba2Config,

    /// See [`Mamba2Layers::ignore_first_residual`].
    #[config(default = false)]
    pub ignore_first_residual: bool,

    /// See [`Mamba2Layers::ignore_last_residual`].
    #[config(default = false)]
    pub ignore_last_residual: bool,
}

impl Mamba2LayersConfig {
    /// Allocate and initialise all layers on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Layers<B> {
        let real_layers = (0..self.n_real_layers)
            .map(|_| Mamba2LayerConfig::new(self.mamba_block.clone()).init(device))
            .collect();

        Mamba2Layers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
        }
    }
}

impl<B: Backend> Mamba2Layers<B> {
    // -----------------------------------------------------------------------
    // forward  (chunked SSD — used for training / prefill)
    // -----------------------------------------------------------------------

    /// Process a full sequence through every (virtual) layer.
    ///
    /// Internally each layer calls [`Mamba2::forward`], which runs the
    /// chunkwise SSD algorithm.  This is efficient for training because the
    /// intra-chunk products can exploit GEMM / tensor cores.
    ///
    /// If `caches` is `None`, zero-initialised caches are created automatically.
    ///
    /// # Arguments
    /// - `x`          — input tensor, shape `[batch, sequence, d_model]`
    /// - `caches`     — optional pre-filled layer caches (useful for prefill
    ///                  followed by decode)
    /// - `ssd_path`   — SSD algorithm and chunk length selection.
    ///                  Defaults to the Core SSD algorithm with the chunk length
    ///                  value of `√(state_rank · per_head_dim)`.
    ///
    /// # Returns
    /// `(output, updated_caches)` where `output` has shape
    /// `[batch, sequence, d_model]`.
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        caches: Option<Mamba2Caches<B>>,
        ssd_path: Option<SsdPath>,
    ) -> (Tensor<B, 3>, Mamba2Caches<B>) {
        // The effective number of forward passes equals the number of *virtual*
        // layers.  When no scheduling is configured this equals n_real_layers.
        let n_virtual_layers = self.n_virtual_count();

        // Lazily allocate zero caches the first time (e.g. during training or
        // the first prefill call).
        let caches = caches.unwrap_or_else(|| self.make_zero_caches(&x, n_virtual_layers));

        assert_eq!(
            caches.caches.len(),
            n_virtual_layers,
            "cache count must match the number of virtual layers; \
             layers in forward() cannot share caches"
        );

        // Unwrap each cache slot into an `Option` so we can `take` it in the
        // loop without cloning (Burn tensors are reference-counted).
        let mut caches: Vec<Option<Mamba2Cache<B>>> = caches.caches.into_iter().map(Some).collect();

        for i in 0..n_virtual_layers {
            // Map virtual layer index → real (weight-bearing) layer index.
            let layer_idx = self.real_idx(i);
            let layer = &self.real_layers[layer_idx];

            // The residual scale is 0.0 for the first/last layer if the
            // corresponding `ignore_*_residual` flag is set, and 1.0 otherwise.
            let residual_scale = self.residual_scale(i, n_virtual_layers);

            let cache = caches[i].take().unwrap();
            let (x_, cache_) = layer.forward(x, Some(cache), ssd_path.clone(), residual_scale);
            x = x_;
            caches[i] = Some(cache_);
        }

        let caches = Mamba2Caches {
            caches: caches.into_iter().map(Option::unwrap).collect(),
        };
        (x, caches)
    }

    // -----------------------------------------------------------------------
    // step  (recurrent SSM — used for autoregressive decoding)
    // -----------------------------------------------------------------------

    /// Process a **single token** through every (virtual) layer.
    ///
    /// Each layer calls [`Mamba2::step`], which runs one tick of the recurrent
    /// SSM:  `hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ`,  `yₜ = Cₜᵀ hₜ + D xₜ`.
    /// This is O(H·P·N) per step — independent of sequence length — and
    /// requires no KV-cache.
    ///
    /// # Arguments
    /// - `x`      — current token embedding, shape `[batch, d_model]`
    /// - `caches` — layer caches from the previous step (or `None` for the
    ///              first token, in which case zero caches are created)
    ///
    /// # Returns
    /// `(output, updated_caches)` where `output` has shape `[batch, d_model]`.
    pub fn step(
        &self,
        mut x: Tensor<B, 2>,
        caches: Option<Mamba2Caches<B>>,
    ) -> (Tensor<B, 2>, Mamba2Caches<B>) {
        let n_virtual_layers = self.n_virtual_count();
        let caches = caches.unwrap_or_else(|| self.make_zero_caches_2d(&x, n_virtual_layers));

        assert_eq!(
            caches.caches.len(),
            n_virtual_layers,
            "cache count must match the number of virtual layers; \
             layers in step() cannot share caches"
        );

        let mut caches: Vec<Option<Mamba2Cache<B>>> = caches.caches.into_iter().map(Some).collect();

        for i in 0..n_virtual_layers {
            let layer_idx = self.real_idx(i);
            let layer = &self.real_layers[layer_idx];
            let residual_scale = self.residual_scale(i, n_virtual_layers);

            let cache = caches[i].take().unwrap();
            let (x_, cache_) = layer.step(x, Some(cache), residual_scale);
            x = x_;
            caches[i] = Some(cache_);
        }

        let caches = Mamba2Caches {
            caches: caches.into_iter().map(Option::unwrap).collect(),
        };
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

    /// Build zero-initialised caches from a 3-D input tensor `[B, S, D]`.
    fn make_zero_caches(&self, x: &Tensor<B, 3>, n_virtual: usize) -> Mamba2Caches<B> {
        let device = &x.device();
        let [batch, _sequence, _d_model] = x.dims();
        let layer0 = &self.real_layers[0].mamba_block;
        let [conv_dim, _, conv_kernel] = layer0.conv1d.weight.dims();

        Mamba2CachesConfig::new(
            n_virtual,
            Mamba2CacheConfig {
                batch,
                state_rank: layer0.state_rank,
                conv_kernel,
                conv_dim,
                per_head_dim: layer0.per_head_dim(),
                nheads: layer0.nheads(),
            },
        )
        .init(device)
    }

    /// Build zero-initialised caches from a 2-D input tensor `[B, D]`.
    fn make_zero_caches_2d(&self, x: &Tensor<B, 2>, n_virtual: usize) -> Mamba2Caches<B> {
        let device = &x.device();
        let [batch, _d_model] = x.dims();
        let layer0 = &self.real_layers[0].mamba_block;
        let [conv_dim, _, conv_kernel] = layer0.conv1d.weight.dims();

        Mamba2CachesConfig::new(
            n_virtual,
            Mamba2CacheConfig {
                batch,
                state_rank: layer0.state_rank,
                conv_kernel,
                conv_dim,
                per_head_dim: layer0.per_head_dim(),
                nheads: layer0.nheads(),
            },
        )
        .init(device)
    }
}

// ---------------------------------------------------------------------------
// Mamba2Layer  (single Pre-LN residual block)
// ---------------------------------------------------------------------------

/// A single Mamba-2 residual block:
///
/// ```text
///   output = x·scale + Mamba2( RMSNorm(x) )
/// ```
///
/// where `scale` is 1.0 normally and 0.0 when the residual connection is
/// intentionally suppressed by the layer stack configuration.
#[derive(Module, Debug)]
pub struct Mamba2Layer<B: Backend> {
    /// Pre-norm applied to the input before the SSM block.
    ///
    /// Using RMSNorm *before* the block (Pre-LN) is standard practice in
    /// modern LLMs and improves training stability.
    pub norm: RmsNorm<B>,

    /// The Mamba-2 SSM block (see [`Mamba2`]).
    pub mamba_block: Mamba2<B>,
}

/// Configuration / factory for [`Mamba2Layer`].
#[derive(Config, Debug)]
pub struct Mamba2LayerConfig {
    /// Configuration for the inner Mamba-2 block.
    pub mamba_block: Mamba2Config,
}

impl Mamba2LayerConfig {
    /// Allocate and initialise the layer on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Layer<B> {
        Mamba2Layer {
            norm: RmsNormConfig::new(self.mamba_block.d_model).init(device),
            mamba_block: self.mamba_block.init(device),
        }
    }
}

impl<B: Backend> Mamba2Layer<B> {
    // -----------------------------------------------------------------------
    // forward  (full sequence)
    // -----------------------------------------------------------------------

    /// Run the Pre-LN residual block over a full sequence.
    ///
    /// Computes:
    /// ```text
    ///   output = x · residual_scale + Mamba2( RMSNorm(x) )
    /// ```
    ///
    /// # Shapes
    /// - `x`      : `[batch, sequence, d_model]`
    /// - output   : `[batch, sequence, d_model]`
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cache: Option<Mamba2Cache<B>>,
        ssd_path: Option<SsdPath>,
        residual_scale: f32,
    ) -> (Tensor<B, 3>, Mamba2Cache<B>) {
        let [batch, sequence, d_model] = x.dims();

        // Save the (optionally scaled) residual *before* normalisation so that
        // the norm does not affect the skip path.
        let res_bsm = x.clone() * residual_scale;

        let normed_bsm = self.norm.forward(x);
        assert_eq!([batch, sequence, d_model], normed_bsm.dims());

        let (mamba_out_bsm, cache) = self
            .mamba_block
            .forward(normed_bsm, cache, ssd_path.clone());
        assert_eq!([batch, sequence, d_model], mamba_out_bsm.dims());

        // Residual addition:  y = x · scale + Mamba2(norm(x))
        let out_bsm = mamba_out_bsm + res_bsm;
        assert_eq!([batch, sequence, d_model], out_bsm.dims());

        (out_bsm, cache)
    }

    // -----------------------------------------------------------------------
    // step  (single token)
    // -----------------------------------------------------------------------

    /// Run the Pre-LN residual block for a **single** decoding step.
    ///
    /// Computes:
    /// ```text
    ///   output = x · residual_scale + Mamba2.step( RMSNorm(x) )
    /// ```
    ///
    /// # Shapes
    /// - `x`  : `[batch, d_model]`
    /// - output: `[batch, d_model]`
    pub fn step(
        &self,
        x: Tensor<B, 2>,
        cache: Option<Mamba2Cache<B>>,
        residual_scale: f32,
    ) -> (Tensor<B, 2>, Mamba2Cache<B>) {
        let [batch, d_model] = x.dims();

        let res_bm = x.clone() * residual_scale;

        let normed_bm = self.norm.forward(x);
        assert_eq!([batch, d_model], normed_bm.dims());

        let (mamba_out_bm, cache) = self.mamba_block.step(normed_bm, cache);
        assert_eq!([batch, d_model], mamba_out_bm.dims());

        let out_bm = mamba_out_bm + res_bm;
        assert_eq!([batch, d_model], out_bm.dims());

        (out_bm, cache)
    }
}
