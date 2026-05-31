use crate::modules::{RmsNorm, RmsNormConfig};
use crate::prelude::*;
use crate::utils::ClassLatent;
use crate::utils::class::{class_marker_output_indices, init_class_emb, insert_class_markers};
use crate::utils::{BidiSchedule, Schedule};
use burn::config::Config;
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
use burn::prelude::*;

// ===========================================================================
// Bidirectional support (family-generic; forward-only, non-autoregressive)
// ===========================================================================
//
// A `BidiLayerPair<M>` runs a straight (→) and a reversed (← via `flip`) Pre-LN
// pass and merges them with an [`OutputMerge`]; `BidiLayers<M>` stacks pairs with
// a [`BidiSchedule`]. The block itself is unchanged — only how its two passes are
// scheduled and combined is bidirectional. Written once for all families; the
// merge is family-agnostic (`RmsNorm`/`Linear` over `Tensor<3>`).

/// A zero-parameter placeholder for the parameterless `Mean` merge.
#[derive(Module, Debug)]
pub struct NoOp;

/// How the two directions of a bidirectional pair are combined.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum OutputMerge {
    /// Element-wise average of the two directions (no parameters).
    Mean(NoOp),
    /// Concatenate along the feature axis and project back down with a learnable
    /// `[2 · d_model, d_model]` linear layer.
    CatLinear(Linear),
}

impl OutputMerge {
    /// Merge the two directional outputs (each `[batch, sequence, d_model]`).
    pub fn forward(&self, straight: Tensor<3>, reverse: Tensor<3>) -> Tensor<3> {
        let [batch, sequence, d_model] = straight.dims();
        assert_eq!(straight.dims(), reverse.dims());
        match self {
            OutputMerge::Mean(_) => (straight + reverse) * 0.5,
            OutputMerge::CatLinear(proj) => {
                let cat = Tensor::cat([straight, reverse].to_vec(), 2);
                assert_eq!([batch, sequence, 2 * d_model], cat.dims());
                let merged = proj.forward(cat);
                assert_eq!([batch, sequence, d_model], merged.dims());
                merged
            }
        }
    }
}

/// Configuration / factory for [`OutputMerge`].
#[derive(Config, Debug)]
pub enum OutputMergeConfig {
    /// Build an [`OutputMerge::Mean`].
    Mean,
    /// Build an [`OutputMerge::CatLinear`].
    CatLinear,
}

impl OutputMergeConfig {
    /// A vector of `n_real_layers / 2` [`Self::Mean`] configs (one per pair).
    pub fn mean(n_real_layers: usize) -> Vec<Self> {
        vec![Self::Mean; n_real_layers / 2]
    }
    /// A vector of `n_real_layers / 2` [`Self::CatLinear`] configs (one per pair).
    pub fn cat_linear(n_real_layers: usize) -> Vec<Self> {
        vec![Self::CatLinear; n_real_layers / 2]
    }
    /// Allocate the merge module on `device` for the given `d_model`.
    pub fn init(&self, d_model: usize, device: &Device) -> OutputMerge {
        match self {
            OutputMergeConfig::Mean => OutputMerge::Mean(NoOp),
            OutputMergeConfig::CatLinear => {
                OutputMerge::CatLinear(LinearConfig::new(d_model * 2, d_model).init(device))
            }
        }
    }
}

/// A single bidirectional pair: a straight (→) and a reversed (←) Pre-LN block
/// whose outputs are merged, then added to the (scaled) residual.
#[derive(Module, Debug)]
pub struct BidiLayerPair<M: Module> {
    /// Pre-norm for the straight pass.
    pub straight_norm: RmsNorm,
    /// Pre-norm for the reversed pass.
    pub reverse_norm: RmsNorm,
    /// The block run left-to-right.
    pub straight_block: M,
    /// The block run right-to-left (over the flipped sequence).
    pub reverse_block: M,
    /// Merge strategy combining the two directions.
    pub output_merge: OutputMerge,
    /// Residual scale (0.0 suppresses the skip connection, else 1.0).
    pub residual_scale: f32,
    /// Positions of this pair's class latents, spliced in before either
    /// direction runs (both directions, and the residual, see the lengthened
    /// sequence). Empty ⇒ none.
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// This pair's class-latent embeddings, `[num_class_latents, d_model]`.
    pub class_latents_emb: Option<Param<Tensor<2>>>,
}

impl<M: MambaBlock> BidiLayerPair<M>
where
    M::SsdPath: Clone,
{
    /// Splice this bidi-layer-pair's class latents into `x` (no-op when there are none).
    fn insert_latents(&self, x: Tensor<3>) -> Tensor<3> {
        if self.class_latents_emb.is_none() {
            return x;
        }
        insert_class_markers(x, &self.class_latents, self.class_latents_emb.as_ref()).0
    }

    /// `[batch, sequence, d_model]` → `[batch, sequence, d_model]`, plus the two
    /// updated direction caches. (`sequence` grows by the class-latent count.)
    pub fn forward(
        &self,
        x: Tensor<3>,
        straight_cache: Option<M::Cache>,
        reverse_cache: Option<M::Cache>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Cache, M::Cache) {
        let x = self.insert_latents(x);
        let [batch, sequence, d_model] = x.dims();
        let res = x.clone() * self.residual_scale;

        // x reads >x₀>x₁>…; x_rev (flipped) reads the sequence backwards.
        let x_rev = x.clone().flip([1]);
        let x = self.straight_norm.forward(x);
        let x_rev = self.reverse_norm.forward(x_rev);

        let (x, straight_cache) =
            self.straight_block
                .block_forward(x, straight_cache, ssd_path.clone());
        assert_eq!([batch, sequence, d_model], x.dims());

        let (x_rev, reverse_cache) =
            self.reverse_block
                .block_forward(x_rev, reverse_cache, ssd_path);
        assert_eq!([batch, sequence, d_model], x_rev.dims());

        // Re-align the reversed read, then merge.
        let x_rev = x_rev.flip([1]);
        let merged = self.output_merge.forward(x, x_rev);
        (merged + res, straight_cache, reverse_cache)
    }
}

/// A stack of bidirectional [`Layer`] pairs with optional virtual-layer
/// scheduling — one struct for every Mamba-x family.
#[derive(Module, Debug)]
pub struct BidiLayers<M: Module> {
    /// Number of real (weight-bearing) layers; must be even (used in pairs).
    pub n_real_layers: usize,
    /// Optional `(n_virtual_layers, schedule)` for weight-sharing.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// The weight-bearing layers, length `n_real_layers`.
    pub real_layers: Vec<Layer<M>>,
    /// Zero the first virtual pair's residual when `true`.
    pub ignore_first_residual: bool,
    /// Zero the last virtual pair's residual when `true`.
    pub ignore_last_residual: bool,
    /// One direction-merge per pair, length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMerge>,
    /// Positions of the stack-level class latents, spliced into the sequence
    /// once before the first pair (independent of any per-pair class latents).
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The stack-level class-latent embeddings, `[num_class_latents, d_model]`.
    pub class_latents_emb: Option<Param<Tensor<2>>>,
}

impl<M: MambaBlock + Clone> BidiLayers<M>
where
    M::SsdPath: Clone,
{
    /// Output positions of the stack-level class latents for an `orig_len` input.
    pub fn class_latent_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_latents, orig_len)
    }

    /// Splice this bidi-layers' class latents into `x` (no-op when there are none).
    fn insert_latents(&self, x: Tensor<3>) -> Tensor<3> {
        if self.class_latents_emb.is_none() {
            return x;
        }
        insert_class_markers(x, &self.class_latents, self.class_latents_emb.as_ref()).0
    }

    /// `[batch, sequence, d_model]` → `[batch, sequence, d_model]`
    /// (`sequence` grows by the stack-level class-latent count).
    pub fn forward(
        &self,
        mut x: Tensor<3>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Caches) {
        x = self.insert_latents(x);
        let n = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _)| {
                assert!(l.is_multiple_of(2), "Bidi virtual layers are used in pairs");
                *l
            })
            .unwrap_or_else(|| {
                assert!(
                    self.n_real_layers.is_multiple_of(2),
                    "Bidi layers are used in pairs"
                );
                self.n_real_layers
            });

        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_3d(&x, n));
        assert_eq!(
            caches.slot_count(),
            n,
            "straight and reverse layers cannot share caches"
        );

        let mut slots = caches.into_slots();
        for i in 0..n / 2 {
            let (straight_i, reverse_i) = (i * 2, i * 2 + 1);
            let (straight_idx, reverse_idx) =
                if let Some((n_virtual, schedule)) = &self.n_virtual_layers {
                    (
                        schedule.real_idx(straight_i, *n_virtual, self.n_real_layers),
                        schedule.real_idx(reverse_i, *n_virtual, self.n_real_layers),
                    )
                } else {
                    (straight_i, reverse_i)
                };
            let straight_layer = &self.real_layers[straight_idx];
            let reverse_layer = &self.real_layers[reverse_idx];

            let straight_cache = slots[straight_i].take().unwrap();
            let reverse_cache = slots[reverse_i].take().unwrap();

            let residual_scale = if (self.ignore_first_residual && i == 0)
                || (self.ignore_last_residual && i + 1 == n / 2)
            {
                0.0
            } else {
                1.0
            };

            let pair = BidiLayerPair {
                straight_norm: straight_layer.norm.clone(),
                reverse_norm: reverse_layer.norm.clone(),
                straight_block: straight_layer.mamba_block.clone(),
                reverse_block: reverse_layer.mamba_block.clone(),
                output_merge: self.outputs_merge[i].clone(),
                residual_scale,
                // Stack-level class latents are spliced once above; the
                // transient per-pair slot stays empty.
                class_latents: Vec::new(),
                class_latents_emb: None,
            };

            let (x_, sc, rc) = pair.forward(
                x,
                Some(straight_cache),
                Some(reverse_cache),
                ssd_path.clone(),
            );
            x = x_;
            slots[straight_i] = Some(sc);
            slots[reverse_i] = Some(rc);
        }

        (x, M::Caches::from_slots(slots))
    }
}

/// Plain (non-serde) factory for [`BidiLayers`].
pub struct BidiLayersBuilder<C> {
    /// Number of real (weight-bearing) layers (must be even).
    pub n_real_layers: usize,
    /// Optional virtual-layer scheduling.
    pub n_virtual_layers: Option<(usize, BidiSchedule)>,
    /// Shared block config.
    pub mamba_block: C,
    /// Zero the first virtual pair's residual.
    pub ignore_first_residual: bool,
    /// Zero the last virtual pair's residual.
    pub ignore_last_residual: bool,
    /// One merge config per pair, length `n_real_layers / 2`.
    pub outputs_merge: Vec<OutputMergeConfig>,
    /// Stack-level class latents (spliced once before the first pair).
    pub class_latents: Vec<ClassLatent>,
}

impl<C: MambaBlockConfig> BidiLayersBuilder<C> {
    /// Allocate and initialise the bidirectional stack on `device`.
    pub fn init(&self, device: &Device) -> BidiLayers<C::Block> {
        let d_model = self.mamba_block.d_model();
        let real_layers = (0..self.n_real_layers)
            .map(|_| Layer {
                norm: RmsNormConfig::new(d_model).init(device),
                mamba_block: self.mamba_block.init_block(device),
                class_latents: Vec::new(),
                class_latents_emb: None,
            })
            .collect();
        let outputs_merge = (0..self.n_real_layers / 2)
            .map(|i| self.outputs_merge[i].init(d_model, device))
            .collect();
        BidiLayers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            outputs_merge,
            class_latents_emb: init_class_emb(self.class_latents.len(), d_model, device),
            class_latents: self.class_latents.clone(),
        }
    }
}

// ===========================================================================
// Unifying enums: one runtime + one serializable Config across all families
// ===========================================================================

/// A runtime-selectable bidirectional stack: the same paired straight/reverse
/// structure over any Mamba-x family, chosen at runtime. The forward-only
/// counterpart of [`MambaLatentNet`] for non-autoregressive tasks.
#[derive(Module, Debug)]
pub enum MambaBidiLayers {
    /// Mamba-1 bidirectional stack.
    #[cfg(feature = "mamba1")]
    Mamba1(BidiLayers<crate::mamba1::prelude::Mamba1>),
    /// Mamba-2 bidirectional stack.
    #[cfg(feature = "mamba2")]
    Mamba2(BidiLayers<crate::mamba2::prelude::Mamba2>),
    /// Mamba-3 bidirectional stack.
    #[cfg(feature = "mamba3")]
    Mamba3(BidiLayers<crate::mamba3::prelude::Mamba3>),
}

impl MambaBidiLayers {
    /// Output positions of the stack-level class latents for an `orig_len`
    /// input (so a caller can read a class latent back out of the lengthened
    /// `forward` output — e.g. as a pooled summary).
    pub fn class_latent_output_indices(&self, orig_len: usize) -> Vec<usize> {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(layers) => layers.class_latent_output_indices(orig_len),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(layers) => layers.class_latent_output_indices(orig_len),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(layers) => layers.class_latent_output_indices(orig_len),
        }
    }

    /// Full-sequence bidirectional pass. The `ssd_path` must match the stack's
    /// family; a mismatch is a caller error and panics.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<MambaCaches>,
        ssd_path: MambaSsdPath,
    ) -> (Tensor<3>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(layers) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 bidi stack"),
                });
                match ssd_path {
                    MambaSsdPath::Mamba1 => {}
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-1 bidi stack"),
                }
                let (y, c) = layers.forward(x, caches, ());
                (y, MambaCaches::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(layers) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 bidi stack"),
                });
                let path = match ssd_path {
                    MambaSsdPath::Mamba2(p) => p,
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-2 bidi stack"),
                };
                let (y, c) = layers.forward(x, caches, path);
                (y, MambaCaches::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(layers) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 bidi stack"),
                });
                let path = match ssd_path {
                    MambaSsdPath::Mamba3(p) => p,
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-3 bidi stack"),
                };
                let (y, c) = layers.forward(x, caches, path);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }
}

/// The serializable config for [`MambaBidiLayers`]. Each variant is concrete
/// (per-family), so `#[derive(Config)]` applies; `init` builds the matching
/// stack variant.
#[derive(Config, Debug)]
pub enum MambaBidiLayersConfig {
    /// Build a Mamba-1 bidirectional stack.
    #[cfg(feature = "mamba1")]
    Mamba1 {
        /// Number of real layers (must be even — used in pairs).
        n_real_layers: usize,
        /// Shared block config.
        mamba_block: crate::mamba1::prelude::Mamba1Config,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
    },
    /// Build a Mamba-2 bidirectional stack.
    #[cfg(feature = "mamba2")]
    Mamba2 {
        /// Number of real layers (must be even — used in pairs).
        n_real_layers: usize,
        /// Shared block config.
        mamba_block: crate::mamba2::prelude::Mamba2Config,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
    },
    /// Build a Mamba-3 bidirectional stack.
    #[cfg(feature = "mamba3")]
    Mamba3 {
        /// Number of real layers (must be even — used in pairs).
        n_real_layers: usize,
        /// Shared block config.
        mamba_block: crate::mamba3::prelude::Mamba3Config,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
    },
}

impl MambaBidiLayersConfig {
    /// Allocate and initialise the selected bidirectional stack on `device`.
    pub fn init(&self, device: &Device) -> MambaBidiLayers {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                n_real_layers,
                mamba_block,
                outputs_merge,
                class_latents,
            } => MambaBidiLayers::Mamba1(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: None,
                    mamba_block: mamba_block.clone(),
                    ignore_first_residual: false,
                    ignore_last_residual: false,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                n_real_layers,
                mamba_block,
                outputs_merge,
                class_latents,
            } => MambaBidiLayers::Mamba2(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: None,
                    mamba_block: mamba_block.clone(),
                    ignore_first_residual: false,
                    ignore_last_residual: false,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                n_real_layers,
                mamba_block,
                outputs_merge,
                class_latents,
            } => MambaBidiLayers::Mamba3(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: None,
                    mamba_block: mamba_block.clone(),
                    ignore_first_residual: false,
                    ignore_last_residual: false,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                }
                .init(device),
            ),
        }
    }
}
