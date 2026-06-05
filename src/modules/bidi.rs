use crate::modules::{Residuals, ResidualsConfig, RmsNorm, RmsNormConfig};
use crate::prelude::*;
use crate::utils::BidiSchedule;
use crate::utils::ClassLatent;
use crate::utils::class::{class_marker_output_indices, init_class_emb, insert_class_markers};
use burn::config::Config;
use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[cfg(test)]
mod tests;

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
/// whose outputs are merged. The residual is **not** applied here — the
/// enclosing [`BidiLayers`] adds it (or suppresses it on the first/last pair),
/// mirroring the [`Layer`](crate::modules::Layer) / [`Layers`](crate::modules::Layers) split.
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
    /// Returns the merged directions **without** the residual — the enclosing
    /// [`BidiLayers`] adds it.
    pub fn forward(
        &self,
        x: Tensor<3>,
        straight_cache: Option<M::Cache>,
        reverse_cache: Option<M::Cache>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Cache, M::Cache) {
        let x = self.insert_latents(x);
        bidi_pair_forward(
            &self.straight_norm,
            &self.reverse_norm,
            &self.straight_block,
            &self.reverse_block,
            &self.output_merge,
            x,
            straight_cache,
            reverse_cache,
            ssd_path,
        )
    }
}

/// The straight + reverse + merge computation of a bidirectional pair, over
/// **borrowed** sub-modules.
///
/// Taking references (rather than owning clones) is load-bearing: a Burn `Param`
/// that is still lazily-initialised re-runs its random initialiser **on every
/// clone**, so cloning a not-yet-materialised block per forward would resample
/// fresh random weights each call. [`BidiLayers`] therefore calls this directly
/// on its real layers instead of building a transient (cloned) [`BidiLayerPair`].
#[allow(clippy::too_many_arguments)]
fn bidi_pair_forward<M: MambaBlock>(
    straight_norm: &RmsNorm,
    reverse_norm: &RmsNorm,
    straight_block: &M,
    reverse_block: &M,
    output_merge: &OutputMerge,
    x: Tensor<3>,
    straight_cache: Option<M::Cache>,
    reverse_cache: Option<M::Cache>,
    ssd_path: M::SsdPath,
) -> (Tensor<3>, M::Cache, M::Cache)
where
    M::SsdPath: Clone,
{
    let [batch, sequence, d_model] = x.dims();

    // x reads >x₀>x₁>…; x_rev (flipped) reads the sequence backwards.
    let x_rev = x.clone().flip([1]);
    let x = straight_norm.forward(x);
    let x_rev = reverse_norm.forward(x_rev);

    let (x, straight_cache) = straight_block.block_forward(x, straight_cache, ssd_path.clone());
    assert_eq!([batch, sequence, d_model], x.dims());

    let (x_rev, reverse_cache) = reverse_block.block_forward(x_rev, reverse_cache, ssd_path);
    assert_eq!([batch, sequence, d_model], x_rev.dims());

    // Re-align the reversed read, then merge.
    let x_rev = x_rev.flip([1]);
    let merged = output_merge.forward(x, x_rev);
    (merged, straight_cache, reverse_cache)
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
    /// How residuals are threaded between **pairs** (plain additive vs
    /// Multi-Gate). The MGR unit is the pair: one module per real/virtual pair.
    pub residuals: Residuals,
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

    /// Seed the MultiGate streams from a full-sequence input — `n_stream` copies
    /// of `x` as `[batch, sequence, n_stream, d_model]` — or `None` for the
    /// Standard path. Panics if MultiGate is paired with stack-level class latents.
    fn multi_gate_streams_seed(&self, x: &Tensor<3>) -> Option<Tensor<4>> {
        let Residuals::MultiGate(mg) = &self.residuals else {
            return None;
        };
        assert!(
            self.class_latents_emb.is_none(),
            "MultiGate residuals do not support stack-level class latents"
        );
        let [batch, sequence, d_model] = x.dims();
        Some(
            x.clone()
                .unsqueeze_dim::<4>(2)
                .expand([batch, sequence, mg.n_stream, d_model]),
        )
    }

    /// `[batch, sequence, d_model]` → `[batch, sequence, d_model]`
    /// (`sequence` grows by the stack-level class-latent count).
    ///
    /// Each pair returns its merged transform `F_l` (no residual). With
    /// [`Residuals::Standard`] the input skip is added per pair (unless
    /// suppressed). With [`Residuals::MultiGate`] the skip is dropped and
    /// `n_stream` parallel streams — seeded from `x` — carry the residual between
    /// pairs: each pair reads their attention-pooled aggregate as input and its
    /// merged output is gated back into every stream (see [`MultiGate`]).
    ///
    /// [`MultiGate`]: crate::modules::MultiGate
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
        // MultiGate keeps `n_stream` parallel streams (seeded from the input);
        // Standard threads the single tensor `x` directly (streams stays `None`).
        let mut streams = self.multi_gate_streams_seed(&x);
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

            let first = self.ignore_first_residual && i == 0;
            let last = self.ignore_last_residual && i + 1 == n / 2;

            // For the Standard path the residual is the (pre-pair) input skip;
            // clone it before the pair consumes `x`, and only when it is used.
            // MultiGate carries the residual in its streams, so clones nothing.
            let residual = match &self.residuals {
                Residuals::Standard(_) if !(first || last) => Some(x.clone()),
                _ => None,
            };

            // Run the pair directly on the (borrowed) real layers — never clone a
            // block, since cloning a lazily-initialised `Param` resamples its
            // random weights (see [`bidi_pair_forward`]). Stack-level class
            // latents were already spliced above; pairs carry none of their own.
            //
            // The pair returns its merged transform `F_l` without the residual.
            // The merge is a per-real-pair weight set (`n_real_layers / 2` of
            // them), so it is indexed by the *real* pair `straight_idx / 2` — not
            // the virtual pair `i` — sharing weights under virtual scheduling just
            // like the blocks (and matching the MGR real-pair index below). In the
            // non-virtual case `straight_idx == i * 2`, so this is `i`.
            let (merged, sc, rc) = bidi_pair_forward(
                &straight_layer.norm,
                &reverse_layer.norm,
                &straight_layer.mamba_block,
                &reverse_layer.mamba_block,
                &self.outputs_merge[straight_idx / 2],
                x,
                Some(straight_cache),
                Some(reverse_cache),
                ssd_path.clone(),
            );
            slots[straight_i] = Some(sc);
            slots[reverse_i] = Some(rc);

            match &self.residuals {
                Residuals::Standard(_noop) => {
                    // Add the input skip here (the pair already consumed `x`), or
                    // output the bare transform when the residual is suppressed.
                    x = match residual {
                        Some(r) => merged + r,
                        None => merged,
                    };
                }
                Residuals::MultiGate(mg) => {
                    let s = streams.take().unwrap();
                    // A skipped residual is β ≡ 1 in the mixer (`new_streams =
                    // F_l`), the aggregator then collapsing to `F_l` — both
                    // branches shortcut that (mirrors `Layers::forward`). The MGR
                    // unit is the pair: virtual pair `i`, real pair `straight_idx
                    // / 2` (the straight index of a pair is even).
                    if last {
                        x = merged;
                        streams = Some(s);
                    } else if first {
                        let [b, seq, d] = merged.dims();
                        streams = Some(merged.clone().unsqueeze_dim::<4>(2).expand([
                            b,
                            seq,
                            mg.n_stream,
                            d,
                        ]));
                        x = merged;
                    } else {
                        let idx = mg.module_index(i, straight_idx / 2);
                        let (new_h, new_streams) = mg.layers[idx].forward(merged, s);
                        x = new_h;
                        streams = Some(new_streams);
                    }
                }
            }
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
    /// Inter-pair residual scheme (defaults to plain additive).
    pub residuals: ResidualsConfig,
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
        // The MGR unit is the pair, so size the modules by *pairs* (halved real
        // and virtual layer counts).
        let n_virtual = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers);
        let residuals =
            self.residuals
                .init(d_model, self.n_real_layers / 2, n_virtual / 2, device);
        BidiLayers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            outputs_merge,
            residuals,
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
        n_virtual_layers: Option<(usize, BidiSchedule)>,
        /// Shared block config.
        mamba_block: crate::mamba1::prelude::Mamba1Config,
        ignore_first_residual: bool,
        ignore_last_residual: bool,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
        /// Inter-pair residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
    },
    /// Build a Mamba-2 bidirectional stack.
    #[cfg(feature = "mamba2")]
    Mamba2 {
        /// Number of real layers (must be even — used in pairs).
        n_real_layers: usize,
        n_virtual_layers: Option<(usize, BidiSchedule)>,
        /// Shared block config.
        mamba_block: crate::mamba2::prelude::Mamba2Config,
        ignore_first_residual: bool,
        ignore_last_residual: bool,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
        /// Inter-pair residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
    },
    /// Build a Mamba-3 bidirectional stack.
    #[cfg(feature = "mamba3")]
    Mamba3 {
        /// Number of real layers (must be even — used in pairs).
        n_real_layers: usize,
        n_virtual_layers: Option<(usize, BidiSchedule)>,
        /// Shared block config.
        mamba_block: crate::mamba3::prelude::Mamba3Config,
        ignore_first_residual: bool,
        ignore_last_residual: bool,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
        /// Inter-pair residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
    },
}

impl MambaBidiLayersConfig {
    /// Allocate and initialise the selected bidirectional stack on `device`.
    pub fn init(&self, device: &Device) -> MambaBidiLayers {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                n_real_layers,
                n_virtual_layers,
                mamba_block,
                ignore_first_residual,
                ignore_last_residual,
                outputs_merge,
                class_latents,
                residuals,
            } => MambaBidiLayers::Mamba1(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: n_virtual_layers.clone(),
                    mamba_block: mamba_block.clone(),
                    ignore_first_residual: *ignore_first_residual,
                    ignore_last_residual: *ignore_last_residual,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                    residuals: residuals.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                n_real_layers,
                n_virtual_layers,
                mamba_block,
                ignore_first_residual,
                ignore_last_residual,
                outputs_merge,
                class_latents,
                residuals,
            } => MambaBidiLayers::Mamba2(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: n_virtual_layers.clone(),
                    mamba_block: mamba_block.clone(),
                    ignore_first_residual: *ignore_first_residual,
                    ignore_last_residual: *ignore_last_residual,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                    residuals: residuals.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                n_real_layers,
                n_virtual_layers,
                mamba_block,
                ignore_first_residual,
                ignore_last_residual,
                outputs_merge,
                class_latents,
                residuals,
            } => MambaBidiLayers::Mamba3(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: n_virtual_layers.clone(),
                    mamba_block: mamba_block.clone(),
                    ignore_first_residual: *ignore_first_residual,
                    ignore_last_residual: *ignore_last_residual,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                    residuals: residuals.clone(),
                }
                .init(device),
            ),
        }
    }
}
