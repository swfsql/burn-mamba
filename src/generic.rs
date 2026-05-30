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

use crate::schedule::{BidiSchedule, Schedule};
use crate::utils::rms_norm::{RmsNorm, RmsNormConfig};
use burn::config::Config;
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
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
// Class tokens / latents (learnable sequence-inserted tokens)
// ===========================================================================
//
// A *class token* / *class latent* is a learnable embedding spliced into the
// sequence — a transformer-`[CLS]`-style register the model can read/write
// through. They are inserted at the input boundary of a container (a network's
// input for [`ClassToken`], width = the input feature width; a layer's working
// sequence for [`ClassLatent`], width = `d_model`), permanently lengthening the
// sequence for everything downstream. A container can carry any number; the
// markers below say *where* each one lands, while a single `Param<Tensor<2>>`
// of shape `[num_markers, width]` holds the embeddings (row `i` ↔ marker `i`).
//
// Insertion order (all relative to the *original* length `L`): every `Start`
// first (index 0), then `Middle` (index `L/2`, splitting the original
// sequence), then `End` (index `L`), then `Custom(index)` (explicit index,
// inserted last). Markers sharing an index keep their `Vec` order. Because
// `Middle`/`End` materialise positions that a single-token `step()` cannot
// reproduce, their presence makes `step()` panic; `Start`/`Custom` are a
// forward-time concern and are simply not re-inserted during `step()`.

/// Position marker for a learnable class **token** inserted into a *network's*
/// input sequence (embedding width = the network input width / "d_input").
#[derive(Config, Debug)]
pub enum ClassToken {
    /// Prepend before the whole sequence (index 0).
    Start,
    /// Insert at the middle of the original sequence (index `L/2`).  
    /// Incompatible with `step()` calls.
    Middle,
    /// Append after the whole sequence (index `L`).  
    /// Incompatible with `step()` calls.
    End,
    /// Insert at an explicit index into the original sequence.
    Custom(usize),
}

/// Position marker for a learnable class **latent** inserted into a *layer's*
/// working sequence (embedding width = `d_model`).
#[derive(Config, Debug)]
pub enum ClassLatent {
    /// Prepend before the whole sequence (index 0).
    Start,
    /// Insert at the middle of the original sequence (index `L/2`).
    /// Incompatible with `step()` calls.  
    Middle,
    /// Append after the whole sequence (index `L`).
    /// Incompatible with `step()` calls.  
    End,
    /// Insert at an explicit index into the original sequence.
    Custom(usize),
}

/// Shared behaviour of the [`ClassToken`] / [`ClassLatent`] position markers,
/// letting one generic helper place either kind.
pub trait ClassMarker: Clone {
    /// Insertion index measured against the *original* sequence length `orig_len`.
    fn insert_pos(&self, orig_len: usize) -> usize;
    /// Tie-break rank among markers sharing an index (`Start`<`Middle`<`End`<`Custom`).
    fn group_rank(&self) -> usize;
    /// Whether this marker is incompatible with single-token `step()`
    /// (`Middle`/`End` create positions a per-token recurrence cannot reproduce).
    fn forbids_step(&self) -> bool;
}

macro_rules! impl_class_marker {
    ($ty:ty) => {
        impl ClassMarker for $ty {
            fn insert_pos(&self, orig_len: usize) -> usize {
                match self {
                    Self::Start => 0,
                    Self::Middle => orig_len / 2,
                    Self::End => orig_len,
                    Self::Custom(index) => *index,
                }
            }
            fn group_rank(&self) -> usize {
                match self {
                    Self::Start => 0,
                    Self::Middle => 1,
                    Self::End => 2,
                    Self::Custom(_) => 3,
                }
            }
            fn forbids_step(&self) -> bool {
                matches!(self, Self::Middle | Self::End)
            }
        }
    };
}
impl_class_marker!(ClassToken);
impl_class_marker!(ClassLatent);

/// Insert the learnable class tokens `emb` (`[k, width]`, row `i` ↔ `markers[i]`)
/// into `x` (`[batch, orig_len, width]`) per the `markers`, returning the
/// lengthened sequence (`[batch, orig_len + k, width]`) and, for each marker in
/// `Vec` order, its position in the output sequence.
///
/// `markers` empty ⇒ `x` is returned unchanged with an empty index vector.
fn insert_class_markers<M: ClassMarker>(
    x: Tensor<3>,
    markers: &[M],
    emb: Option<&Param<Tensor<2>>>,
) -> (Tensor<3>, Vec<usize>) {
    let [batch, orig_len, width] = x.dims();
    let k = markers.len();
    if k == 0 {
        return (x, Vec::new());
    }
    let emb = emb.expect("class-token markers present but no embedding param").val();
    assert_eq!(emb.dims(), [k, width], "one embedding row per class marker");

    // Emit in (insert_pos, group_rank, vec order) order.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by_key(|&i| (markers[i].insert_pos(orig_len), markers[i].group_rank(), i));

    let mut segments: Vec<Tensor<3>> = Vec::new();
    let mut cursor = 0usize; // consumed prefix of the original sequence
    let mut out_len = 0usize; // length emitted so far
    let mut out_index = vec![0usize; k];
    for &i in &order {
        let p = markers[i].insert_pos(orig_len);
        assert!(p <= orig_len, "class-token insert index {p} > sequence length {orig_len}");
        if p > cursor {
            segments.push(x.clone().narrow(1, cursor, p - cursor));
            out_len += p - cursor;
            cursor = p;
        }
        let row = emb
            .clone()
            .narrow(0, i, 1) // [1, width]
            .unsqueeze_dim::<3>(0) // [1, 1, width]
            .expand([batch, 1, width]);
        segments.push(row);
        out_index[i] = out_len;
        out_len += 1;
    }
    if cursor < orig_len {
        segments.push(x.narrow(1, cursor, orig_len - cursor));
    }
    let out = Tensor::cat(segments, 1);
    assert_eq!(out.dims(), [batch, orig_len + k, width]);
    (out, out_index)
}

/// The output-sequence position of each marker (in `Vec` order) for an input of
/// length `orig_len`, without materialising any tensor. Mirrors the placement in
/// [`insert_class_markers`] — useful for reading a class token back out.
fn class_marker_output_indices<M: ClassMarker>(markers: &[M], orig_len: usize) -> Vec<usize> {
    let k = markers.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by_key(|&i| (markers[i].insert_pos(orig_len), markers[i].group_rank(), i));
    let mut cursor = 0usize;
    let mut out_len = 0usize;
    let mut out_index = vec![0usize; k];
    for &i in &order {
        let p = markers[i].insert_pos(orig_len).min(orig_len);
        if p > cursor {
            out_len += p - cursor;
            cursor = p;
        }
        out_index[i] = out_len;
        out_len += 1;
    }
    out_index
}

/// Build the embedding param for `n` class markers of the given `width`
/// (`None` when there are none — Burn has no zero-width tensors).
fn init_class_emb(n: usize, width: usize, device: &Device) -> Option<Param<Tensor<2>>> {
    (n > 0).then(|| Initializer::Normal { mean: 0.0, std: 0.02 }.init([n, width], device))
}

/// Panic if any marker is incompatible with single-token `step()`.
fn assert_step_compatible<M: ClassMarker>(markers: &[M], who: &str) {
    assert!(
        !markers.iter().any(|m| m.forbids_step()),
        "{who}: Middle/End class tokens are not compatible with step()"
    );
}

/// The output-sequence position of each step-injectable marker (in `Vec` order),
/// for use by `step`'s cursor. Asserts no `Middle`/`End` (those need the full
/// length — `forward` only). `Start`/`Custom` positions are length-independent,
/// so an unbounded `orig_len` resolves them exactly.
fn class_step_injections<M: ClassMarker>(markers: &[M], who: &str) -> Vec<usize> {
    assert_step_compatible(markers, who);
    class_marker_output_indices(markers, usize::MAX)
}

// ===========================================================================
// Layer<M>
// ===========================================================================

/// A single Pre-LN residual block: `output = x·residual_scale + M(RMSNorm(x))`.
///
/// May carry its own [`ClassLatent`]s, spliced into the sequence at the start of
/// `forward` (so the residual and the inner block both see the lengthened
/// sequence). They are independent of any class latents on the enclosing
/// [`Layers`].
#[derive(Module, Debug)]
pub struct Layer<M: Module> {
    /// Pre-norm applied before the inner block.
    pub norm: RmsNorm,
    /// The inner Mamba-x SSM block.
    pub mamba_block: M,
    /// Positions of this layer's class latents (empty ⇒ none).
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The class-latent embeddings, `[num_class_latents, d_model]` (`None` ⇒ none).
    pub class_latents_emb: Option<Param<Tensor<2>>>,
}

impl<M: MambaBlock> Layer<M> {
    /// Splice this layer's class latents into `x` (no-op when there are none).
    fn insert_latents(&self, x: Tensor<3>) -> Tensor<3> {
        if self.class_latents_emb.is_none() {
            return x;
        }
        insert_class_markers(x, &self.class_latents, self.class_latents_emb.as_ref()).0
    }

    /// Full-sequence Pre-LN residual pass.
    pub fn forward(
        &self,
        x: Tensor<3>,
        cache: Option<M::Cache>,
        ssd_path: M::SsdPath,
        residual_scale: f32,
    ) -> (Tensor<3>, M::Cache) {
        let x = self.insert_latents(x);
        let res = x.clone() * residual_scale;
        let normed = self.norm.forward(x);
        let (out, cache) = self.mamba_block.block_forward(normed, cache, ssd_path);
        (out + res, cache)
    }

    /// Single-token Pre-LN residual step.
    ///
    /// `index` is the running cursor into this layer's *output* sequence. With
    /// `Some`, whenever it lands on one of this layer's class-latent positions
    /// those latents are stepped first (each advancing `index`, recursing with
    /// `None`), then the user token (also advancing `index`); only the user
    /// token's output and cache are returned. With `None` no class latents are
    /// injected — and `Middle`/`End` latents panic (their positions need the full
    /// sequence; use `forward`).
    pub fn step(
        &self,
        x: Tensor<2>,
        cache: Option<M::Cache>,
        residual_scale: f32,
        index: Option<&mut usize>,
    ) -> (Tensor<2>, M::Cache) {
        let Some(cursor) = index else {
            // The actual one-token work (no class injection).
            assert_step_compatible(&self.class_latents, "Layer");
            let res = x.clone() * residual_scale;
            let normed = self.norm.forward(x);
            let (out, cache) = self.mamba_block.block_step(normed, cache);
            return (out + res, cache);
        };
        let [batch, d_model] = x.dims();
        let inj = class_step_injections(&self.class_latents, "Layer");
        let emb = self.class_latents_emb.as_ref();
        let mut cache = cache;
        while let Some(i) = inj.iter().position(|&p| p == *cursor) {
            let row = emb.unwrap().val().narrow(0, i, 1).expand([batch, d_model]);
            let (_discard, c) = self.step(row, cache, residual_scale, None);
            cache = Some(c);
            *cursor += 1;
        }
        let (out, cache) = self.step(x, cache, residual_scale, None);
        *cursor += 1;
        (out, cache)
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
    /// Positions of the stack-level class latents, spliced into the sequence
    /// once before the first virtual layer (independent of any per-[`Layer`]
    /// class latents). Empty ⇒ none.
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The stack-level class-latent embeddings, `[num_class_latents, d_model]`.
    pub class_latents_emb: Option<Param<Tensor<2>>>,
}

impl<M: MambaBlock> Layers<M>
where
    M::SsdPath: Clone,
{
    /// Output positions of the stack-level class latents for an `orig_len` input.
    pub fn class_latent_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_latents, orig_len)
    }

    /// Splice this layers' class latents into `x` (no-op when there are none).
    fn insert_latents(&self, x: Tensor<3>) -> Tensor<3> {
        if self.class_latents_emb.is_none() {
            return x;
        }
        insert_class_markers(x, &self.class_latents, self.class_latents_emb.as_ref()).0
    }

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
        x: Tensor<3>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Caches) {
        let mut x = self.insert_latents(x);
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
    ///
    /// Two independent class-latent cursors:
    /// - `own_index` — the stack-level [`Self::class_latents`] (spliced once
    ///   before the first layer in `forward`). When it lands on one of those
    ///   positions the stack latent(s) are stepped first (each a full stack pass,
    ///   advancing `own_index`), then the user token.
    /// - `layer_indices` — one cursor **per virtual layer**
    ///   (`len == n_virtual_layers`), distributed as the per-[`Layer`] cursor so
    ///   each layer injects its own class latents independently.
    ///
    /// A `None` cursor skips that level's injection; `Middle`/`End` latents panic
    /// for the cursored level (their positions need the full sequence — use
    /// `forward`).
    pub fn step(
        &self,
        mut x: Tensor<2>,
        caches: Option<M::Caches>,
        own_index: Option<&mut usize>,
        mut layer_indices: Option<&mut Vec<usize>>,
    ) -> (Tensor<2>, M::Caches) {
        // Stack-level class-latent injection (around full stack passes).
        if let Some(cursor) = own_index {
            let [batch, d_model] = x.dims();
            let inj = class_step_injections(&self.class_latents, "Layers");
            let emb = self.class_latents_emb.as_ref();
            let mut caches = caches;
            while let Some(i) = inj.iter().position(|&p| p == *cursor) {
                let row = emb.unwrap().val().narrow(0, i, 1).expand([batch, d_model]);
                let (_discard, c) = self.step(row, caches, None, None);
                caches = Some(c);
                *cursor += 1;
            }
            let (out, caches) = self.step(x, caches, None, layer_indices);
            *cursor += 1;
            return (out, caches);
        }

        // The actual one-token work: thread the token through the stack, giving
        // each virtual layer its own class-latent cursor from `layer_indices`.
        assert_step_compatible(&self.class_latents, "Layers");
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_2d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");
        if let Some(v) = layer_indices.as_deref() {
            assert_eq!(v.len(), n, "one class-latent cursor per virtual layer");
        }

        let mut slots = caches.into_slots();
        for i in 0..n {
            let layer = &self.real_layers[self.real_idx(i)];
            let rs = self.residual_scale(i, n);
            let cache = slots[i].take().unwrap();
            let layer_cursor = layer_indices.as_deref_mut().map(|v| &mut v[i]);
            let (x_, c_) = layer.step(x, Some(cache), rs, layer_cursor);
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
    /// Stack-level class latents (spliced once before the first virtual layer).
    pub class_latents: Vec<ClassLatent>,
}

impl<C: MambaBlockConfig> LayersBuilder<C> {
    /// Builder with no virtual scheduling, no class latents, residuals enabled.
    pub fn new(n_real_layers: usize, mamba_block: C) -> Self {
        Self {
            n_real_layers,
            n_virtual_layers: None,
            mamba_block,
            ignore_first_residual: false,
            ignore_last_residual: false,
            class_latents: Vec::new(),
        }
    }

    /// Set the optional virtual-layer scheduling.
    pub fn with_n_virtual_layers(mut self, n: Option<(usize, Schedule)>) -> Self {
        self.n_virtual_layers = n;
        self
    }

    /// Set the stack-level class latents.
    #[cfg(test)]
    pub fn with_class_latents(mut self, class_latents: Vec<ClassLatent>) -> Self {
        self.class_latents = class_latents;
        self
    }

    /// Allocate and initialise the stack on `device`.
    pub fn init(&self, device: &Device) -> Layers<C::Block> {
        let d_model = self.mamba_block.d_model();
        let real_layers = (0..self.n_real_layers)
            .map(|_| Layer {
                norm: RmsNormConfig::new(d_model).init(device),
                mamba_block: self.mamba_block.init_block(device),
                class_latents: Vec::new(),
                class_latents_emb: None,
            })
            .collect();
        Layers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            class_latents_emb: init_class_emb(self.class_latents.len(), d_model, device),
            class_latents: self.class_latents.clone(),
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
    /// Positions of the network's class tokens, spliced into the input sequence
    /// (at `input_size` width) **before** `in_proj`. Empty ⇒ none.
    #[module(skip)]
    pub class_tokens: Vec<ClassToken>,
    /// The class-token embeddings, `[num_class_tokens, input_size]`.
    pub class_tokens_emb: Option<Param<Tensor<2>>>,
}

impl<M: MambaBlock> LatentNetwork<M>
where
    M::SsdPath: Clone,
{
    /// Output positions of the class tokens for an `orig_len` input.
    pub fn class_token_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_tokens, orig_len)
    }

    /// Splice this network's class latents into `x` (no-op when there are none).
    fn insert_tokens(&self, x: Tensor<3>) -> Tensor<3> {
        if self.class_tokens_emb.is_none() {
            return x;
        }
        insert_class_markers(x, &self.class_tokens, self.class_tokens_emb.as_ref()).0
    }

    /// `in_proj → layers → out_proj` over a full sequence
    /// (`[batch, sequence, input_size]` → `[batch, sequence (+ class tokens),
    /// output_size]`).
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Caches) {
        let mut x = self.insert_tokens(x);
        let x = self.in_proj.forward(x);
        let (x, caches) = self.layers.forward(x, caches, ssd_path);
        let x = self.out_proj.forward(x);
        (x, caches)
    }

    /// Single-token step (`[batch, input_size]` → `[batch, output_size]`).
    ///
    /// Three independent class cursors:
    /// - `own_index` — the network's own [`Self::class_tokens`] (spliced before
    ///   `in_proj`). When it lands on a class-token position those tokens are
    ///   stepped first (each a full network pass, advancing `own_index`), then
    ///   the user token; only the user token's output is returned.
    /// - `layers_own_index` / `layer_indices` — forwarded straight to the inner
    ///   [`Layers::step`] (stack-level latents, and the per-virtual-layer cursor
    ///   vector respectively).
    ///
    /// A `None` cursor skips that level; `Middle`/`End` markers panic for the
    /// cursored level (use `forward`).
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        own_index: Option<&mut usize>,
        layers_own_index: Option<&mut usize>,
        layer_indices: Option<&mut Vec<usize>>,
    ) -> (Tensor<2>, M::Caches) {
        // Network-level class-token injection (around full network passes). The
        // injected class tokens use their own recurrence only (inner cursors are
        // not advanced for them); the user token carries the inner cursors.
        if let Some(cursor) = own_index {
            let [batch, input_size] = x.dims();
            let inj = class_step_injections(&self.class_tokens, "LatentNetwork");
            let emb = self.class_tokens_emb.as_ref();
            let mut caches = caches;
            while let Some(i) = inj.iter().position(|&p| p == *cursor) {
                let row = emb.unwrap().val().narrow(0, i, 1).expand([batch, input_size]);
                let (_discard, c) = self.step(row, caches, None, None, None);
                caches = Some(c);
                *cursor += 1;
            }
            let (out, caches) = self.step(x, caches, None, layers_own_index, layer_indices);
            *cursor += 1;
            return (out, caches);
        }

        // The actual one-token work: forward the inner cursors to the stack.
        assert_step_compatible(&self.class_tokens, "LatentNetwork");
        let x = self.in_proj.forward(x);
        let (x, caches) = self.layers.step(x, caches, layers_own_index, layer_indices);
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
    /// Network-level class tokens (spliced into the input before `in_proj`).
    pub class_tokens: Vec<ClassToken>,
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
            class_tokens_emb: init_class_emb(self.class_tokens.len(), self.input_size, device),
            class_tokens: self.class_tokens.clone(),
        }
    }
}

// ===========================================================================
// VocabNetwork<M>
// ===========================================================================

/// A complete autoregressive language model over a token vocabulary:
/// `Embedding (vocab → d_model) → Layers<M> → norm_f → LM head (d_model →
/// vocab)`.
///
/// This is the token-LM counterpart of [`LatentNetwork`]; both are built on the
/// shared [`Layers`] core. The only differences are the I/O boundary (a token
/// `Embedding` and a vocab logit head, instead of two latent `Linear`s) and a
/// final pre-head [`RmsNorm`].
///
/// The LM head is **tied** (`lm_head = None`, the transposed embedding weight is
/// reused) or **untied** (a dedicated `Linear`); the vocabulary is rounded up to
/// a multiple for GPU alignment (see [`VocabNetworkBuilder`]).
#[derive(Module, Debug)]
pub struct VocabNetwork<M: Module> {
    /// Token embedding table, weight shape `[padded_vocab, d_model]`.
    pub embedding: Embedding,
    /// The shared Mamba-x layer stack.
    pub layers: Layers<M>,
    /// Final RMSNorm applied before the LM head (`norm_f`).
    pub norm_f: RmsNorm,
    /// Optional dedicated LM head. `None` ⇒ weight-tied (reuse embedding`ᵀ`).
    pub lm_head: Option<Linear>,
}

impl<M: MambaBlock> VocabNetwork<M>
where
    M::SsdPath: Clone,
{
    /// Full-sequence pass: token IDs `[batch, sequence]` → logits
    /// `[batch, sequence, padded_vocab]`.
    pub fn forward(
        &self,
        x: Tensor<2, Int>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Caches) {
        let x = self.embedding.forward(x);
        let (x, caches) = self.layers.forward(x, caches, ssd_path);
        let x = self.norm_f.forward(x);
        (self.apply_lm_head(x), caches)
    }

    /// Single-token step: token IDs `[batch]` → logits `[batch, padded_vocab]`.
    ///
    /// The vocab network has no class tokens of its own (those would duplicate
    /// the layers' class latents); it simply forwards the inner [`Layers`]
    /// cursors — `layers_own_index` (stack-level latents) and `layer_indices`
    /// (per-virtual-layer) — to [`Layers::step`].
    pub fn step(
        &self,
        x: Tensor<1, Int>,
        caches: Option<M::Caches>,
        layers_own_index: Option<&mut usize>,
        layer_indices: Option<&mut Vec<usize>>,
    ) -> (Tensor<2>, M::Caches) {
        // Embed the single token via a temporary unit sequence axis.
        let x = self.embedding.forward(x.unsqueeze_dim::<2>(1)).squeeze_dim(1);
        let (x, caches) = self.layers.step(x, caches, layers_own_index, layer_indices);
        let x = self.norm_f.forward(x);
        // Reuse the 3-D head by lifting/lowering the sequence axis.
        let logits = self.apply_lm_head(x.unsqueeze_dim(1)).squeeze_dim(1);
        (logits, caches)
    }

    /// Project `[batch, sequence, d_model]` → `[batch, sequence, padded_vocab]`
    /// using the dedicated head, or the tied (transposed embedding) weight.
    fn apply_lm_head(&self, x: Tensor<3>) -> Tensor<3> {
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(x)
        } else {
            // Weight tying: reuse embedding.weight^T ([d_model, padded_vocab]).
            let weight = self.embedding.weight.clone().map(|w| w.permute([1, 0]));
            Linear { weight, bias: None }.forward(x)
        }
    }
}

/// Plain factory for [`VocabNetwork`]. Mirrors [`LatentNetworkBuilder`] but adds
/// vocab padding and the tied/untied LM-head choice.
pub struct VocabNetworkBuilder<C> {
    /// Unpadded vocabulary size (rounded up at init).
    pub vocab_size: usize,
    /// Round `vocab_size` up to a multiple of this (1 disables rounding).
    pub pad_vocab_size_multiple: usize,
    /// Builder for the layer stack.
    pub layers: LayersBuilder<C>,
    /// When `true`, tie the LM head to the (transposed) embedding weights.
    pub missing_lm_head: bool,
}

impl<C: MambaBlockConfig> VocabNetworkBuilder<C> {
    /// Round `vocab_size` up to the next multiple of `multiple`.
    fn padded_vocab(vocab_size: usize, multiple: usize) -> usize {
        if vocab_size.is_multiple_of(multiple) {
            vocab_size
        } else {
            ((vocab_size / multiple) + 1) * multiple
        }
    }

    /// Allocate and initialise the network on `device`.
    pub fn init(&self, device: &Device) -> VocabNetwork<C::Block> {
        let d_model = self.layers.mamba_block.d_model();
        let padded_vocab = Self::padded_vocab(self.vocab_size, self.pad_vocab_size_multiple);
        let lm_head = if self.missing_lm_head {
            None
        } else {
            Some(
                LinearConfig::new(d_model, padded_vocab)
                    .with_bias(false)
                    .init(device),
            )
        };
        VocabNetwork {
            embedding: EmbeddingConfig::new(padded_vocab, d_model).init(device),
            layers: self.layers.init(device),
            norm_f: RmsNormConfig::new(d_model).init(device),
            lm_head,
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
    use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Caches, Mamba3Config, Mamba3SsdPath};
    use crate::mamba3::single_ssd::prelude::{
        Mamba3SingleSsdCacheConfig, Mamba3SingleSsdCaches, Mamba3SingleSsdCachesConfig,
    };

    /// Zero single-ssd caches sized from a `[batch, sequence, d_model]` input.
    /// (A missing cache defaults to the single-ssd pathway — ≈½ the SSD memory
    /// of double-ssd — for either rotation kind.)
    fn zero_single_ssd_caches(
        mamba_block: &Mamba3,
        batch: usize,
        n_virtual: usize,
        device: &Device,
    ) -> Mamba3SingleSsdCaches {
        Mamba3SingleSsdCachesConfig::new(
            n_virtual,
            Mamba3SingleSsdCacheConfig {
                batch,
                state_rank: mamba_block.state_rank,
                num_rope_angles: mamba_block.num_rope_angles,
                per_head_dim: mamba_block.per_head_dim(),
                nheads: mamba_block.nheads(),
                mimo_rank: mamba_block.mimo_rank,
                rotation: mamba_block.rotation,
                num_quat_blocks: mamba_block.num_quat_blocks,
            },
        )
        .init(device)
    }

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
            let [batch, _seq, _d] = x.dims();
            zero_single_ssd_caches(self, batch, n_virtual, &x.device()).into()
        }
        fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Mamba3Caches {
            let [batch, _d] = x.dims();
            zero_single_ssd_caches(self, batch, n_virtual, &x.device()).into()
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

        let (x_rev, reverse_cache) = self.reverse_block.block_forward(x_rev, reverse_cache, ssd_path);
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

            let (x_, sc, rc) =
                pair.forward(x, Some(straight_cache), Some(reverse_cache), ssd_path.clone());
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
    /// families). Cache family must match the network. The three class cursors
    /// (`own_index` for the network's class tokens, `layers_own_index` for the
    /// stack-level class latents, `layer_indices` for the per-virtual-layer
    /// vector) are threaded to the inner network — see [`LatentNetwork::step`].
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<MambaCaches>,
        own_index: Option<&mut usize>,
        layers_own_index: Option<&mut usize>,
        layer_indices: Option<&mut Vec<usize>>,
    ) -> (Tensor<2>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 network"),
                });
                let (y, c) = net.step(x, caches, own_index, layers_own_index, layer_indices);
                (y, MambaCaches::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 network"),
                });
                let (y, c) = net.step(x, caches, own_index, layers_own_index, layer_indices);
                (y, MambaCaches::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 network"),
                });
                let (y, c) = net.step(x, caches, own_index, layers_own_index, layer_indices);
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
        /// Optional virtual-layer scheduling.
        n_virtual_layers: Option<(usize, Schedule)>,
        /// Shared block config.
        mamba_block: crate::mamba1::prelude::Mamba1Config,
        /// Output feature width.
        output_size: usize,
        /// Network-level class tokens, spliced into the input before `in_proj`.
        class_tokens: Vec<ClassToken>,
    },
    /// Build a Mamba-2 latent network.
    #[cfg(feature = "mamba2")]
    Mamba2 {
        /// Input feature width.
        input_size: usize,
        /// Number of real layers.
        n_real_layers: usize,
        /// Optional virtual-layer scheduling.
        n_virtual_layers: Option<(usize, Schedule)>,
        /// Shared block config.
        mamba_block: crate::mamba2::prelude::Mamba2Config,
        /// Output feature width.
        output_size: usize,
        /// Network-level class tokens, spliced into the input before `in_proj`.
        class_tokens: Vec<ClassToken>,
    },
    /// Build a Mamba-3 latent network.
    #[cfg(feature = "mamba3")]
    Mamba3 {
        /// Input feature width.
        input_size: usize,
        /// Number of real layers.
        n_real_layers: usize,
        /// Optional virtual-layer scheduling.
        n_virtual_layers: Option<(usize, Schedule)>,
        /// Shared block config.
        mamba_block: crate::mamba3::prelude::Mamba3Config,
        /// Output feature width.
        output_size: usize,
        /// Network-level class tokens, spliced into the input before `in_proj`.
        class_tokens: Vec<ClassToken>,
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
                n_virtual_layers,
                mamba_block,
                output_size,
                class_tokens,
            } => MambaLatentNet::Mamba1(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone()),
                    output_size: *output_size,
                    class_tokens: class_tokens.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                input_size,
                n_real_layers,
                n_virtual_layers,
                mamba_block,
                output_size,
                class_tokens,
            } => MambaLatentNet::Mamba2(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone()),
                    output_size: *output_size,
                    class_tokens: class_tokens.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                input_size,
                n_real_layers,
                n_virtual_layers,
                mamba_block,
                output_size,
                class_tokens,
            } => MambaLatentNet::Mamba3(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone()),
                    output_size: *output_size,
                    class_tokens: class_tokens.clone(),
                }
                .init(device),
            ),
        }
    }
}

/// A runtime-selectable token language model: the same `Embedding → Layers →
/// norm_f → LM head` shape over any Mamba-x family, chosen at runtime. The
/// vocabulary counterpart of [`MambaLatentNet`].
#[derive(Module, Debug)]
pub enum MambaVocabNet {
    /// Mamba-1 language model.
    #[cfg(feature = "mamba1")]
    Mamba1(VocabNetwork<crate::mamba1::prelude::Mamba1>),
    /// Mamba-2 language model.
    #[cfg(feature = "mamba2")]
    Mamba2(VocabNetwork<crate::mamba2::prelude::Mamba2>),
    /// Mamba-3 language model.
    #[cfg(feature = "mamba3")]
    Mamba3(VocabNetwork<crate::mamba3::prelude::Mamba3>),
}

impl MambaVocabNet {
    /// Full-sequence pass: token IDs `[batch, sequence]` → logits
    /// `[batch, sequence, padded_vocab]`. The `ssd_path`/`caches` family must
    /// match the network; a mismatch is a caller error and panics.
    pub fn forward(
        &self,
        x: Tensor<2, Int>,
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

    /// Single-token step: token IDs `[batch]` → logits `[batch, padded_vocab]`.
    /// Cache family must match the network. The two inner [`Layers`] class
    /// cursors (`layers_own_index`, `layer_indices`) are forwarded — see
    /// [`VocabNetwork::step`].
    pub fn step(
        &self,
        x: Tensor<1, Int>,
        caches: Option<MambaCaches>,
        layers_own_index: Option<&mut usize>,
        layer_indices: Option<&mut Vec<usize>>,
    ) -> (Tensor<2>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 network"),
                });
                let (y, c) = net.step(x, caches, layers_own_index, layer_indices);
                (y, MambaCaches::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 network"),
                });
                let (y, c) = net.step(x, caches, layers_own_index, layer_indices);
                (y, MambaCaches::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 network"),
                });
                let (y, c) = net.step(x, caches, layers_own_index, layer_indices);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }
}

/// The serializable, documentation-friendly config for [`MambaVocabNet`]. Each
/// variant is concrete (per-family), so `#[derive(Config)]` applies; `init`
/// builds the matching network variant.
#[derive(Config, Debug)]
pub enum MambaVocabNetConfig {
    /// Build a Mamba-1 language model.
    #[cfg(feature = "mamba1")]
    Mamba1 {
        /// Number of real layers.
        n_real_layers: usize,
        /// Optional virtual-layer scheduling.
        n_virtual_layers: Option<(usize, Schedule)>,
        /// Unpadded vocabulary size.
        vocab_size: usize,
        /// Round `vocab_size` up to a multiple of this (1 disables rounding).
        pad_vocab_size_multiple: usize,
        /// Shared block config.
        mamba_block: crate::mamba1::prelude::Mamba1Config,
        /// Tie the LM head to the (transposed) embedding weights when `true`.
        missing_lm_head: bool,
    },
    /// Build a Mamba-2 language model.
    #[cfg(feature = "mamba2")]
    Mamba2 {
        /// Number of real layers.
        n_real_layers: usize,
        /// Optional virtual-layer scheduling.
        n_virtual_layers: Option<(usize, Schedule)>,
        /// Unpadded vocabulary size.
        vocab_size: usize,
        /// Round `vocab_size` up to a multiple of this (1 disables rounding).
        pad_vocab_size_multiple: usize,
        /// Shared block config.
        mamba_block: crate::mamba2::prelude::Mamba2Config,
        /// Tie the LM head to the (transposed) embedding weights when `true`.
        missing_lm_head: bool,
    },
    /// Build a Mamba-3 language model.
    #[cfg(feature = "mamba3")]
    Mamba3 {
        /// Number of real layers.
        n_real_layers: usize,
        /// Optional virtual-layer scheduling.
        n_virtual_layers: Option<(usize, Schedule)>,
        /// Unpadded vocabulary size.
        vocab_size: usize,
        /// Round `vocab_size` up to a multiple of this (1 disables rounding).
        pad_vocab_size_multiple: usize,
        /// Shared block config.
        mamba_block: crate::mamba3::prelude::Mamba3Config,
        /// Tie the LM head to the (transposed) embedding weights when `true`.
        missing_lm_head: bool,
    },
}

impl MambaVocabNetConfig {
    /// Allocate and initialise the selected language model on `device`.
    pub fn init(&self, device: &Device) -> MambaVocabNet {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                n_real_layers,
                n_virtual_layers,
                vocab_size,
                pad_vocab_size_multiple,
                mamba_block,
                missing_lm_head,
            } => MambaVocabNet::Mamba1(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone()),
                    missing_lm_head: *missing_lm_head,
                }
                .init(device),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                n_real_layers,
                n_virtual_layers,
                vocab_size,
                pad_vocab_size_multiple,
                mamba_block,
                missing_lm_head,
            } => MambaVocabNet::Mamba2(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone()),
                    missing_lm_head: *missing_lm_head,
                }
                .init(device),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                n_real_layers,
                n_virtual_layers,
                vocab_size,
                pad_vocab_size_multiple,
                mamba_block,
                missing_lm_head,
            } => MambaVocabNet::Mamba3(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone()),
                    missing_lm_head: *missing_lm_head,
                }
                .init(device),
            ),
        }
    }
}

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

// ===========================================================================
// Smoke tests
// ===========================================================================

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
