//! # Family-generic Mamba abstraction
//!
//! A single set of generic structs — `Layer<M>`, `Layers<M>`,
//! `LatentNetwork<M>` — that work for **any** Mamba-x block (`M = Mamba1 |
//! Mamba2 | Mamba3`), replacing the three near-identical per-family copies. The
//! per-family differences (cache type, ssd-path type, the two execution modes)
//! are abstracted behind the [`MambaBlock`] trait; the cache *collection* behind
//! [`CacheStack`]; and config→module construction behind [`MambaBlockConfig`].
//!
//! Burn's `#[derive(Module)]` is generic-aware (verified), so these derive
//! cleanly. Its `#[derive(Config)]` is **not** generic-aware, so the
//! serializable, user-facing configs are concrete (see the `MambaLatentNet`
//! unifying enum, added separately); here the builders are plain factory structs.

use crate::schedule::Schedule;
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::config::Config;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

// ===========================================================================
// Traits
// ===========================================================================

/// The uniform interface a per-network cache collection exposes for the generic
/// [`Layers`] loop. The existing `Mamba{1,2,3}Caches` already provide it (under
/// `caches_len`/`into_options`/`from_options`).
pub trait CacheStack: Sized {
    /// The per-layer cache element.
    type Cache;
    /// Number of per-(virtual-)layer slots.
    fn slot_count(&self) -> usize;
    /// Move each slot into an `Option` so the loop can `take` without cloning.
    fn into_slots(self) -> Vec<Option<Self::Cache>>;
    /// Inverse of [`Self::into_slots`].
    fn from_slots(slots: Vec<Option<Self::Cache>>) -> Self;
}

/// Per-family block interface the generic [`Layer`]/[`Layers`] delegate to.
pub trait MambaBlock: Module {
    /// Per-block streaming cache (one layer's worth of state).
    type Cache;
    /// The per-network cache collection for this family.
    type Caches: CacheStack<Cache = Self::Cache>;
    /// SSD algorithm / chunk-length selector. `()` for families without one.
    type SsdPath;

    /// Full-sequence (chunked) pass — training / prefill.
    fn block_forward(
        &self,
        x: Tensor<3>,
        cache: Option<Self::Cache>,
        ssd_path: Self::SsdPath,
    ) -> (Tensor<3>, Self::Cache);

    /// Single-token recurrent step — decoding.
    fn block_step(&self, x: Tensor<2>, cache: Option<Self::Cache>) -> (Tensor<2>, Self::Cache);

    /// Build `n_virtual` zero caches sized for a `[batch, sequence, d_model]` input.
    fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Self::Caches;
    /// Build `n_virtual` zero caches sized for a `[batch, d_model]` input.
    fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Self::Caches;
}

/// A block *config* that knows its `d_model` and how to build its [`MambaBlock`].
/// Lets the generic builders construct `Layers<M>` without knowing the family.
pub trait MambaBlockConfig: Config {
    /// The block this config builds.
    type Block: MambaBlock;
    /// Model width, used to size each layer's pre-norm.
    fn d_model(&self) -> usize;
    /// Allocate and initialise the block on `device`.
    fn init_block(&self, device: &Device) -> Self::Block;
}

// ===========================================================================
// Layer<M>
// ===========================================================================

/// A single Pre-LN residual block: `output = x·residual_scale + M(RMSNorm(x))`.
#[derive(Module, Debug)]
pub struct Layer<M: Module> {
    /// Pre-norm applied before the inner block.
    pub norm: RmsNorm,
    /// The inner Mamba-x SSM block.
    pub mamba_block: M,
}

impl<M: MambaBlock> Layer<M> {
    /// Full-sequence Pre-LN residual pass.
    pub fn forward(
        &self,
        x: Tensor<3>,
        cache: Option<M::Cache>,
        ssd_path: M::SsdPath,
        residual_scale: f32,
    ) -> (Tensor<3>, M::Cache) {
        let res = x.clone() * residual_scale;
        let normed = self.norm.forward(x);
        let (out, cache) = self.mamba_block.block_forward(normed, cache, ssd_path);
        (out + res, cache)
    }

    /// Single-token Pre-LN residual step.
    pub fn step(
        &self,
        x: Tensor<2>,
        cache: Option<M::Cache>,
        residual_scale: f32,
    ) -> (Tensor<2>, M::Cache) {
        let res = x.clone() * residual_scale;
        let normed = self.norm.forward(x);
        let (out, cache) = self.mamba_block.block_step(normed, cache);
        (out + res, cache)
    }
}

// ===========================================================================
// Layers<M>
// ===========================================================================

/// A stack of [`Layer`]s with optional virtual-layer scheduling — one struct for
/// every Mamba-x family.
#[derive(Module, Debug)]
pub struct Layers<M: Module> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,
    /// Optional `(n_virtual_layers, schedule)` for weight-sharing.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, Schedule)>,
    /// The weight-bearing layers, length `n_real_layers`.
    pub real_layers: Vec<Layer<M>>,
    /// Zero the first virtual layer's residual when `true`.
    pub ignore_first_residual: bool,
    /// Zero the last virtual layer's residual when `true`.
    pub ignore_last_residual: bool,
}

impl<M: MambaBlock> Layers<M>
where
    M::SsdPath: Clone,
{
    fn n_virtual_count(&self) -> usize {
        self.n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers)
    }

    fn real_idx(&self, virtual_idx: usize) -> usize {
        if let Some((n, schedule)) = &self.n_virtual_layers {
            schedule.real_idx(virtual_idx, *n, self.n_real_layers)
        } else {
            virtual_idx
        }
    }

    fn residual_scale(&self, i: usize, n: usize) -> f32 {
        let first = self.ignore_first_residual && i == 0;
        let last = self.ignore_last_residual && i + 1 == n;
        if first || last { 0.0 } else { 1.0 }
    }

    /// Full-sequence pass through every (virtual) layer.
    pub fn forward(
        &self,
        mut x: Tensor<3>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Caches) {
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_3d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");

        let mut slots = caches.into_slots();
        for i in 0..n {
            let layer = &self.real_layers[self.real_idx(i)];
            let rs = self.residual_scale(i, n);
            let cache = slots[i].take().unwrap();
            let (x_, c_) = layer.forward(x, Some(cache), ssd_path.clone(), rs);
            x = x_;
            slots[i] = Some(c_);
        }
        (x, M::Caches::from_slots(slots))
    }

    /// Single-token step through every (virtual) layer.
    pub fn step(&self, mut x: Tensor<2>, caches: Option<M::Caches>) -> (Tensor<2>, M::Caches) {
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_2d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");

        let mut slots = caches.into_slots();
        for i in 0..n {
            let layer = &self.real_layers[self.real_idx(i)];
            let rs = self.residual_scale(i, n);
            let cache = slots[i].take().unwrap();
            let (x_, c_) = layer.step(x, Some(cache), rs);
            x = x_;
            slots[i] = Some(c_);
        }
        (x, M::Caches::from_slots(slots))
    }
}

/// Plain (non-serde) factory for [`Layers`]. The serializable surface is the
/// concrete `MambaLatentNetConfig` enum; this is just the generic builder it
/// delegates to.
pub struct LayersBuilder<C> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,
    /// Optional virtual-layer scheduling.
    pub n_virtual_layers: Option<(usize, Schedule)>,
    /// Shared block config.
    pub mamba_block: C,
    /// Zero the first virtual layer's residual.
    pub ignore_first_residual: bool,
    /// Zero the last virtual layer's residual.
    pub ignore_last_residual: bool,
}

impl<C: MambaBlockConfig> LayersBuilder<C> {
    /// Builder with no virtual scheduling and residuals enabled.
    pub fn new(n_real_layers: usize, mamba_block: C) -> Self {
        Self {
            n_real_layers,
            n_virtual_layers: None,
            mamba_block,
            ignore_first_residual: false,
            ignore_last_residual: false,
        }
    }

    /// Allocate and initialise the stack on `device`.
    pub fn init(&self, device: &Device) -> Layers<C::Block> {
        let d_model = self.mamba_block.d_model();
        let real_layers = (0..self.n_real_layers)
            .map(|_| Layer {
                norm: RmsNormConfig::new(d_model).init(device),
                mamba_block: self.mamba_block.init_block(device),
            })
            .collect();
        Layers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
        }
    }
}

// ===========================================================================
// LatentNetwork<M>
// ===========================================================================

/// A feature/regression network on latents:
/// `in_proj (input_size → d_model) → Layers<M> → out_proj (d_model → output_size)`.
#[derive(Module, Debug)]
pub struct LatentNetwork<M: Module> {
    /// Linear projection `input_size → d_model`.
    pub in_proj: Linear,
    /// The shared Mamba-x layer stack.
    pub layers: Layers<M>,
    /// Linear projection `d_model → output_size`.
    pub out_proj: Linear,
}

impl<M: MambaBlock> LatentNetwork<M>
where
    M::SsdPath: Clone,
{
    /// `in_proj → layers → out_proj` over a full sequence
    /// (`[batch, sequence, input_size]` → `[batch, sequence, output_size]`).
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Caches) {
        let x = self.in_proj.forward(x);
        let (x, caches) = self.layers.forward(x, caches, ssd_path);
        let x = self.out_proj.forward(x);
        (x, caches)
    }

    /// Single-token step (`[batch, input_size]` → `[batch, output_size]`).
    pub fn step(&self, x: Tensor<2>, caches: Option<M::Caches>) -> (Tensor<2>, M::Caches) {
        let x = self.in_proj.forward(x);
        let (x, caches) = self.layers.step(x, caches);
        let x = self.out_proj.forward(x);
        (x, caches)
    }
}

/// Plain factory for [`LatentNetwork`].
pub struct LatentNetworkBuilder<C> {
    /// Width of the input features fed to `in_proj`.
    pub input_size: usize,
    /// Builder for the layer stack.
    pub layers: LayersBuilder<C>,
    /// Width of the output features produced by `out_proj`.
    pub output_size: usize,
}

impl<C: MambaBlockConfig> LatentNetworkBuilder<C> {
    /// Allocate and initialise the network on `device`.
    pub fn init(&self, device: &Device) -> LatentNetwork<C::Block> {
        let d_model = self.layers.mamba_block.d_model();
        LatentNetwork {
            in_proj: LinearConfig::new(self.input_size, d_model)
                .with_bias(true)
                .init(device),
            layers: self.layers.init(device),
            out_proj: LinearConfig::new(d_model, self.output_size)
                .with_bias(true)
                .init(device),
        }
    }
}

// ===========================================================================
// Per-family impls
// ===========================================================================

#[cfg(feature = "mamba2")]
mod impl_mamba2 {
    use super::*;
    use crate::mamba2::prelude::{
        Mamba2, Mamba2Cache, Mamba2CacheConfig, Mamba2Caches, Mamba2CachesConfig, Mamba2Config,
        Mamba2SsdPath,
    };

    impl CacheStack for Mamba2Caches {
        type Cache = Mamba2Cache;
        fn slot_count(&self) -> usize {
            self.caches.len()
        }
        fn into_slots(self) -> Vec<Option<Mamba2Cache>> {
            self.caches.into_iter().map(Some).collect()
        }
        fn from_slots(slots: Vec<Option<Mamba2Cache>>) -> Self {
            Self {
                caches: slots.into_iter().map(Option::unwrap).collect(),
            }
        }
    }

    impl MambaBlock for Mamba2 {
        type Cache = Mamba2Cache;
        type Caches = Mamba2Caches;
        type SsdPath = Mamba2SsdPath;

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba2Cache>,
            ssd_path: Mamba2SsdPath,
        ) -> (Tensor<3>, Mamba2Cache) {
            self.forward(x, cache, ssd_path)
        }
        fn block_step(&self, x: Tensor<2>, cache: Option<Mamba2Cache>) -> (Tensor<2>, Mamba2Cache) {
            self.step(x, cache)
        }
        fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Mamba2Caches {
            let [batch, _seq, _d] = x.dims();
            self.make_zero(batch, n_virtual, &x.device())
        }
        fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Mamba2Caches {
            let [batch, _d] = x.dims();
            self.make_zero(batch, n_virtual, &x.device())
        }
    }

    impl Mamba2 {
        fn make_zero(&self, batch: usize, n_virtual: usize, device: &Device) -> Mamba2Caches {
            let [conv_dim, _, conv_kernel] = self.conv1d.weight.dims();
            Mamba2CachesConfig::new(
                n_virtual,
                Mamba2CacheConfig {
                    batch,
                    state_rank: self.state_rank,
                    conv_kernel,
                    conv_dim,
                    per_head_dim: self.per_head_dim(),
                    nheads: self.nheads(),
                },
            )
            .init(device)
        }
    }

    impl MambaBlockConfig for Mamba2Config {
        type Block = Mamba2;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba2 {
            self.init(device)
        }
    }
}

#[cfg(feature = "mamba3")]
mod impl_mamba3 {
    use super::*;
    use crate::mamba3::layer::{make_zero_caches_single_ssd_2d, make_zero_caches_single_ssd_3d};
    use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Caches, Mamba3Config, Mamba3SsdPath};

    impl CacheStack for Mamba3Caches {
        type Cache = Mamba3Cache;
        fn slot_count(&self) -> usize {
            self.caches_len()
        }
        fn into_slots(self) -> Vec<Option<Mamba3Cache>> {
            self.into_options()
        }
        fn from_slots(slots: Vec<Option<Mamba3Cache>>) -> Self {
            Self::from_options(slots)
        }
    }

    impl MambaBlock for Mamba3 {
        type Cache = Mamba3Cache;
        type Caches = Mamba3Caches;
        type SsdPath = Mamba3SsdPath;

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba3Cache>,
            ssd_path: Mamba3SsdPath,
        ) -> (Tensor<3>, Mamba3Cache) {
            self.forward(x, cache, ssd_path)
        }
        fn block_step(&self, x: Tensor<2>, cache: Option<Mamba3Cache>) -> (Tensor<2>, Mamba3Cache) {
            self.step(x, cache)
        }
        fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Mamba3Caches {
            make_zero_caches_single_ssd_3d(self, x, n_virtual).into()
        }
        fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Mamba3Caches {
            make_zero_caches_single_ssd_2d(self, x, n_virtual).into()
        }
    }

    impl MambaBlockConfig for Mamba3Config {
        type Block = Mamba3;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba3 {
            self.init(device)
        }
    }
}

#[cfg(feature = "mamba1")]
mod impl_mamba1 {
    use super::*;
    use crate::mamba1::prelude::{
        Mamba1, Mamba1Cache, Mamba1CacheConfig, Mamba1Caches, Mamba1CachesConfig, Mamba1Config,
    };

    impl CacheStack for Mamba1Caches {
        type Cache = Mamba1Cache;
        fn slot_count(&self) -> usize {
            self.caches.len()
        }
        fn into_slots(self) -> Vec<Option<Mamba1Cache>> {
            self.caches.into_iter().map(Some).collect()
        }
        fn from_slots(slots: Vec<Option<Mamba1Cache>>) -> Self {
            Self {
                caches: slots.into_iter().map(Option::unwrap).collect(),
            }
        }
    }

    impl MambaBlock for Mamba1 {
        type Cache = Mamba1Cache;
        type Caches = Mamba1Caches;
        /// Mamba-1 has no SSD chunking, so there is no path selector.
        type SsdPath = ();

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba1Cache>,
            _ssd_path: (),
        ) -> (Tensor<3>, Mamba1Cache) {
            self.forward(x, cache)
        }
        fn block_step(&self, x: Tensor<2>, cache: Option<Mamba1Cache>) -> (Tensor<2>, Mamba1Cache) {
            self.step(x, cache)
        }
        fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Mamba1Caches {
            let [batch, _seq, _d] = x.dims();
            self.make_zero(batch, n_virtual, &x.device())
        }
        fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Mamba1Caches {
            let [batch, _d] = x.dims();
            self.make_zero(batch, n_virtual, &x.device())
        }
    }

    impl Mamba1 {
        fn cache_config(&self, batch: usize) -> Mamba1CacheConfig {
            let [d_inner, state_rank] = self.a_log.dims();
            let [_, _, conv_kernel] = self.conv1d.weight.dims();
            Mamba1CacheConfig::new(batch, d_inner)
                .with_state_rank(state_rank)
                .with_conv_kernel(conv_kernel)
        }
        fn make_zero(&self, batch: usize, n_virtual: usize, device: &Device) -> Mamba1Caches {
            Mamba1CachesConfig::new(n_virtual, self.cache_config(batch)).init(device)
        }
    }

    impl MambaBlockConfig for Mamba1Config {
        type Block = Mamba1;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba1 {
            self.init(device)
        }
    }
}

// ===========================================================================
// Unifying enums: one runtime + one serializable Config across all families
// ===========================================================================
//
// The generic `LatentNetwork<M>` above is family-typed (`M` is fixed at the type
// level). To let an example (or a user) choose the family at *runtime* — and to
// serialize that choice for docs/config round-trips — we wrap the three
// monomorphisations in enums. `#[derive(Module)]` and `#[derive(Config)]` both
// support enums (verified), so this stays first-class Burn.

/// An explicit, family-tagged SSD-path selector for the unified API.
///
/// Each variant carries the concrete per-family path so callers can choose the
/// algorithm/chunk explicitly; the `*_default` constructors offer the common
/// "ride along the family default" path without making it the *only* option.
#[derive(Debug, Clone)]
pub enum MambaSsdPath {
    /// Mamba-1 has no SSD chunking (path is the unit type).
    #[cfg(feature = "mamba1")]
    Mamba1,
    /// Mamba-2 SSD path.
    #[cfg(feature = "mamba2")]
    Mamba2(crate::mamba2::prelude::Mamba2SsdPath),
    /// Mamba-3 SSD path.
    #[cfg(feature = "mamba3")]
    Mamba3(crate::mamba3::prelude::Mamba3SsdPath),
}

impl MambaSsdPath {
    /// The Mamba-2 default path (`SerialRecalculated`, optimal chunk).
    #[cfg(feature = "mamba2")]
    pub fn mamba2_default() -> Self {
        Self::Mamba2(Default::default())
    }
    /// The Mamba-3 default path (`SerialRecalculated`, optimal chunk).
    #[cfg(feature = "mamba3")]
    pub fn mamba3_default() -> Self {
        Self::Mamba3(Default::default())
    }
}

/// Runtime-tagged caches: one variant per family, matching [`MambaLatentNet`].
///
/// This is plain runtime state (not a `Module`): caches are threaded through
/// `forward`/`step`, never recorded or optimised. (`Mamba3Caches` is itself a
/// non-`Module` enum, so a `Module` derive here would not even apply.)
#[derive(Debug)]
pub enum MambaCaches {
    /// Mamba-1 caches.
    #[cfg(feature = "mamba1")]
    Mamba1(crate::mamba1::prelude::Mamba1Caches),
    /// Mamba-2 caches.
    #[cfg(feature = "mamba2")]
    Mamba2(crate::mamba2::prelude::Mamba2Caches),
    /// Mamba-3 caches.
    #[cfg(feature = "mamba3")]
    Mamba3(crate::mamba3::prelude::Mamba3Caches),
}

/// A runtime-selectable latent network: the same `in_proj → Layers → out_proj`
/// shape over any Mamba-x family, chosen at runtime.
#[derive(Module, Debug)]
pub enum MambaLatentNet {
    /// Mamba-1 latent network.
    #[cfg(feature = "mamba1")]
    Mamba1(LatentNetwork<crate::mamba1::prelude::Mamba1>),
    /// Mamba-2 latent network.
    #[cfg(feature = "mamba2")]
    Mamba2(LatentNetwork<crate::mamba2::prelude::Mamba2>),
    /// Mamba-3 latent network.
    #[cfg(feature = "mamba3")]
    Mamba3(LatentNetwork<crate::mamba3::prelude::Mamba3>),
}

impl MambaLatentNet {
    /// Full-sequence pass. The `ssd_path` must match the network's family; a
    /// mismatch is a caller error and panics with an explanatory message.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<MambaCaches>,
        ssd_path: MambaSsdPath,
    ) -> (Tensor<3>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 network"),
                });
                match ssd_path {
                    MambaSsdPath::Mamba1 => {}
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-1 network"),
                }
                let (y, c) = net.forward(x, caches, ());
                (y, MambaCaches::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 network"),
                });
                let path = match ssd_path {
                    MambaSsdPath::Mamba2(p) => p,
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-2 network"),
                };
                let (y, c) = net.forward(x, caches, path);
                (y, MambaCaches::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 network"),
                });
                let path = match ssd_path {
                    MambaSsdPath::Mamba3(p) => p,
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-3 network"),
                };
                let (y, c) = net.forward(x, caches, path);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }

    /// Single-token step. No path argument (decoding is recurrent for all
    /// families). Cache family must match the network.
    pub fn step(&self, x: Tensor<2>, caches: Option<MambaCaches>) -> (Tensor<2>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 network"),
                });
                let (y, c) = net.step(x, caches);
                (y, MambaCaches::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 network"),
                });
                let (y, c) = net.step(x, caches);
                (y, MambaCaches::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 network"),
                });
                let (y, c) = net.step(x, caches);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }
}

/// The serializable, documentation-friendly config for [`MambaLatentNet`]. Each
/// variant is concrete (per-family), so `#[derive(Config)]` applies; `init`
/// builds the matching network variant.
#[derive(Config, Debug)]
pub enum MambaLatentNetConfig {
    /// Build a Mamba-1 latent network.
    #[cfg(feature = "mamba1")]
    Mamba1 {
        /// Input feature width.
        input_size: usize,
        /// Number of real layers.
        n_real_layers: usize,
        /// Shared block config.
        mamba_block: crate::mamba1::prelude::Mamba1Config,
        /// Output feature width.
        output_size: usize,
    },
    /// Build a Mamba-2 latent network.
    #[cfg(feature = "mamba2")]
    Mamba2 {
        /// Input feature width.
        input_size: usize,
        /// Number of real layers.
        n_real_layers: usize,
        /// Shared block config.
        mamba_block: crate::mamba2::prelude::Mamba2Config,
        /// Output feature width.
        output_size: usize,
    },
    /// Build a Mamba-3 latent network.
    #[cfg(feature = "mamba3")]
    Mamba3 {
        /// Input feature width.
        input_size: usize,
        /// Number of real layers.
        n_real_layers: usize,
        /// Shared block config.
        mamba_block: crate::mamba3::prelude::Mamba3Config,
        /// Output feature width.
        output_size: usize,
    },
}

impl MambaLatentNetConfig {
    /// Allocate and initialise the selected network on `device`.
    pub fn init(&self, device: &Device) -> MambaLatentNet {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                input_size,
                n_real_layers,
                mamba_block,
                output_size,
            } => MambaLatentNet::Mamba1(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone()),
                    output_size: *output_size,
                }
                .init(device),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                input_size,
                n_real_layers,
                mamba_block,
                output_size,
            } => MambaLatentNet::Mamba2(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone()),
                    output_size: *output_size,
                }
                .init(device),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                input_size,
                n_real_layers,
                mamba_block,
                output_size,
            } => MambaLatentNet::Mamba3(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone()),
                    output_size: *output_size,
                }
                .init(device),
            ),
        }
    }
}

// ===========================================================================
// Smoke tests
// ===========================================================================

#[cfg(all(test, feature = "_dev-test"))]
mod tests {
    use super::*;

    #[cfg(feature = "mamba2")]
    #[test]
    fn latent_network_builder_mamba2() {
        use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
        let device = Device::default();
        let block = Mamba2Config::new(16)
            .with_expand(2)
            .with_per_head_dim(4)
            .with_state_rank(8)
            .with_ngroups(1)
            .with_conv_kernel(4);
        let net = LatentNetworkBuilder {
            input_size: 3,
            layers: LayersBuilder::new(2, block),
            output_size: 2,
        }
        .init(&device);

        let (y, _c) = net.forward(
            Tensor::<3>::zeros([2, 5, 3], &device),
            None,
            Mamba2SsdPath::default(),
        );
        assert_eq!([2, 5, 2], y.dims());
        let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None);
        assert_eq!([2, 2], yt.dims());
    }

    #[cfg(feature = "mamba2")]
    #[test]
    fn unified_net_config_mamba2() {
        use crate::mamba2::prelude::Mamba2Config;
        let device = Device::default();
        let block = Mamba2Config::new(16)
            .with_expand(2)
            .with_per_head_dim(4)
            .with_state_rank(8)
            .with_ngroups(1)
            .with_conv_kernel(4);
        let net = MambaLatentNetConfig::Mamba2 {
            input_size: 3,
            n_real_layers: 2,
            mamba_block: block,
            output_size: 2,
        }
        .init(&device);

        // Explicit, family-tagged path.
        let (y, caches) = net.forward(
            Tensor::<3>::zeros([2, 5, 3], &device),
            None,
            MambaSsdPath::mamba2_default(),
        );
        assert_eq!([2, 5, 2], y.dims());

        // Thread the returned caches back in (round-trips the enum cache).
        let (y2, _c) = net.forward(
            Tensor::<3>::zeros([2, 5, 3], &device),
            Some(caches),
            MambaSsdPath::mamba2_default(),
        );
        assert_eq!([2, 5, 2], y2.dims());

        let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None);
        assert_eq!([2, 2], yt.dims());
    }

    #[cfg(feature = "mamba3")]
    #[test]
    fn unified_net_config_mamba3() {
        use crate::mamba3::prelude::Mamba3Config;
        let device = Device::default();
        let block = Mamba3Config::new(16)
            .with_expand(2)
            .with_per_head_dim(4)
            .with_state_rank(8)
            .with_ngroups(1)
            .with_mimo_rank(1)
            .with_rope_fraction(0.5);
        let net = MambaLatentNetConfig::Mamba3 {
            input_size: 3,
            n_real_layers: 2,
            mamba_block: block,
            output_size: 2,
        }
        .init(&device);

        let (y, _c) = net.forward(
            Tensor::<3>::zeros([2, 5, 3], &device),
            None,
            MambaSsdPath::mamba3_default(),
        );
        assert_eq!([2, 5, 2], y.dims());
        let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None);
        assert_eq!([2, 2], yt.dims());
    }

    #[cfg(feature = "mamba1")]
    #[test]
    fn unified_net_config_mamba1() {
        use crate::mamba1::prelude::Mamba1Config;
        let device = Device::default();
        let block = Mamba1Config::new(16).with_state_rank(8);
        let net = MambaLatentNetConfig::Mamba1 {
            input_size: 3,
            n_real_layers: 2,
            mamba_block: block,
            output_size: 2,
        }
        .init(&device);

        let (y, _c) = net.forward(
            Tensor::<3>::zeros([2, 5, 3], &device),
            None,
            MambaSsdPath::Mamba1,
        );
        assert_eq!([2, 5, 2], y.dims());
        let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None);
        assert_eq!([2, 2], yt.dims());
    }
}
