use crate::modules::LayersBuilder;
use crate::modules::{ResidualsConfig, RmsNorm, RmsNormConfig};
use crate::prelude::*;
use crate::utils::Schedule;
use crate::utils::class::{
    assert_full_len_known, class_chunk_plan, class_marker_output_indices, class_row,
    init_class_emb, insert_class_markers,
};
use crate::utils::{ClassCursor, ClassCursors};
use burn::config::Config;
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

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
    ///
    /// A marker that never lands (a `Custom` at or past the end) reports a
    /// position past the emitted sequence — compare against its length.
    pub fn class_token_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_tokens, orig_len)
    }

    /// Splice this network's class tokens into the chunk `x` (no-op when there
    /// are none), advancing the network-level cursor.
    fn insert_tokens(&self, x: Tensor<3>, class: &mut ClassCursors) -> Tensor<3> {
        let mut cursor = ClassCursor::at(class.network, class.full_len);
        let x = insert_class_markers(
            x,
            &self.class_tokens,
            self.class_tokens_emb.as_ref(),
            &mut cursor,
            "LatentNetwork",
        );
        class.network = cursor.offset;
        x
    }

    /// `in_proj → layers → out_proj` over a full sequence
    /// (`[batch, sequence, input_size]` → `[batch, sequence (+ class tokens),
    /// output_size]`).
    ///
    /// `class` places this network's class tokens *and* the inner stack's class
    /// latents; `None` takes `x` for the whole sequence. Handing the same
    /// [`ClassCursors`] to consecutive chunks places every marker exactly where
    /// a single call over the concatenated sequence would.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<3>, M::Caches) {
        // No cursors ⇒ this one call covers the whole sequence.
        let mut whole = ClassCursors::new(x.dims()[1]);
        let class = class.unwrap_or(&mut whole);
        let x = self.insert_tokens(x, class);
        let x = self.in_proj.forward(x);
        // The stack's sequence is this one, lengthened by the class tokens.
        let saved = class.enter(self.class_tokens.len());
        let (x, caches) = self.layers.forward(x, caches, ssd_path, Some(&mut *class));
        class.leave(saved);
        let x = self.out_proj.forward(x);
        (x, caches)
    }

    /// Single-token step (`[batch, input_size]` → `[batch, output_size]`).
    ///
    /// `class` drives all three class levels at once: this network's own
    /// [`Self::class_tokens`] (`class.network`) plus the inner [`Layers::step`]
    /// cursors (`class.stack`, `class.per_layer`).
    ///
    /// As in `forward`, the network's class tokens are part of the sequence that
    /// enters the layers, so each is run through a full network pass (carrying
    /// the inner cursors, so the layers splice their own latents around it
    /// exactly as in `forward`). What comes back is the output of the **last**
    /// token the step emitted: the user token, unless an `End` marker (at either
    /// level) follows it, that marker being then the sequence's true last token.
    /// `None` injects nothing anywhere; `Middle`/`End` markers then panic, as
    /// they do without a [`ClassCursors::full_len`] hint.
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        let Some(class) = class else {
            assert_full_len_known(&self.class_tokens, None, "LatentNetwork");
            return self.step_one(x, caches, None);
        };
        let mut cursor = ClassCursor::at(class.network, class.full_len);
        let plan = class_chunk_plan(&self.class_tokens, 1, &mut cursor, "LatentNetwork");
        class.network = cursor.offset;
        if plan.is_empty() {
            return self.step_one(x, caches, Some(&mut *class));
        }
        // `at == 0` ⇒ the class token precedes the user token, `at == 1` ⇒ it is
        // an `End` closing the sequence, and follows it.
        let [batch, input_size] = x.dims();
        let row = |i: usize| class_row(self.class_tokens_emb.as_ref(), i, batch, input_size);
        let (before, after): (Vec<_>, Vec<_>) = plan.into_iter().partition(|&(at, _)| at == 0);
        let mut caches = caches;
        for (_, i) in before {
            let (_discard, c) = self.step_one(row(i), caches, Some(&mut *class));
            caches = Some(c);
        }
        let (mut out, mut caches) = self.step_one(x, caches, Some(&mut *class));
        for (_, i) in after {
            // A closing `End` token *is* the sequence's last token — its output,
            // not the user token's, is what this step produced.
            let (o, c) = self.step_one(row(i), Some(caches), Some(&mut *class));
            out = o;
            caches = c;
        }
        (out, caches)
    }

    /// One token through `in_proj → layers → out_proj`; the network's own class
    /// tokens are placed by [`Self::step`], the inner cursors are forwarded.
    fn step_one(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        let x = self.in_proj.forward(x);
        let (x, caches) = match class {
            // The stack's sequence is this one, lengthened by the class tokens.
            Some(class) => {
                let saved = class.enter(self.class_tokens.len());
                let out = self.layers.step(x, caches, Some(&mut *class));
                class.leave(saved);
                out
            }
            None => self.layers.step(x, caches, None),
        };
        (self.out_proj.forward(x), caches)
    }

    /// Stationary fixed point of the network under a constant input token:
    /// `in_proj → `[`Layers::step_infinite`]` → out_proj`, no caches.
    /// Cursorless (class tokens are not injected).
    pub fn step_infinite(&self, x: Tensor<2>) -> Tensor<2> {
        assert_full_len_known(&self.class_tokens, None, "LatentNetwork");
        let x = self.in_proj.forward(x);
        let x = self.layers.step_infinite(x);
        self.out_proj.forward(x)
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
    /// `[batch, sequence, padded_vocab]`. `class` places the inner stack's class
    /// latents (`None` ⇒ `x` is the whole sequence) — see [`Layers::forward`].
    pub fn forward(
        &self,
        x: Tensor<2, Int>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<3>, M::Caches) {
        let x = self.embedding.forward(x);
        let (x, caches) = self.layers.forward(x, caches, ssd_path, class);
        let x = self.norm_f.forward(x);
        (self.apply_lm_head(x), caches)
    }

    /// Single-token step: token IDs `[batch]` → logits `[batch, padded_vocab]`.
    ///
    /// The vocab network has no class tokens of its own (those would duplicate
    /// the layers' class latents); it simply forwards `class` — the stack-level
    /// and per-virtual-layer cursors — to [`Layers::step`].
    pub fn step(
        &self,
        x: Tensor<1, Int>,
        caches: Option<M::Caches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, M::Caches) {
        // Embed the single token via a temporary unit sequence axis.
        let x = self
            .embedding
            .forward(x.unsqueeze_dim::<2>(1))
            .squeeze_dim(1);
        let (x, caches) = self.layers.step(x, caches, class);
        let x = self.norm_f.forward(x);
        // Reuse the 3-D head by lifting/lowering the sequence axis.
        let logits = self.apply_lm_head(x.unsqueeze_dim(1)).squeeze_dim(1);
        (logits, caches)
    }

    /// Stationary fixed point of the LM under a constant token: logits
    /// `[batch, padded_vocab]` after infinitely many repeats of `x`, no caches
    /// (see [`Layers::step_infinite`]).
    pub fn step_infinite(&self, x: Tensor<1, Int>) -> Tensor<2> {
        let x = self
            .embedding
            .forward(x.unsqueeze_dim::<2>(1))
            .squeeze_dim(1);
        let x = self.layers.step_infinite(x);
        let x = self.norm_f.forward(x);
        self.apply_lm_head(x.unsqueeze_dim(1)).squeeze_dim(1)
    }

    /// Project `[batch, sequence, d_model]` → `[batch, sequence, padded_vocab]`
    /// using the dedicated head, or the tied (transposed embedding) weight.
    fn apply_lm_head(&self, x: Tensor<3>) -> Tensor<3> {
        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(x)
        } else {
            // Weight tying: reuse embedding.weight^T ([d_model, padded_vocab]).
            let weight = self.embedding.weight.clone().map(|w| w.transpose());
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
// Unifying enums: one runtime + one serializable Config across all families
// ===========================================================================

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
        class: Option<&mut ClassCursors>,
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
                let (y, c) = net.forward(x, caches, (), class);
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
                let (y, c) = net.forward(x, caches, path, class);
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
                let (y, c) = net.forward(x, caches, path, class);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }

    /// Single-token step. No path argument (decoding is recurrent for all
    /// families). Cache family must match the network. `class` — the cursors of
    /// all three class levels — is threaded to the inner network, see
    /// [`LatentNetwork::step`].
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<MambaCaches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 network"),
                });
                let (y, c) = net.step(x, caches, class);
                (y, MambaCaches::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 network"),
                });
                let (y, c) = net.step(x, caches, class);
                (y, MambaCaches::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 network"),
                });
                let (y, c) = net.step(x, caches, class);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }

    /// Stationary fixed point under a constant token (no caches) — see
    /// [`LatentNetwork::step_infinite`]. Only the Mamba-3 family implements the
    /// closed form; the other variants panic.
    pub fn step_infinite(&self, x: Tensor<2>) -> Tensor<2> {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => net.step_infinite(x),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => net.step_infinite(x),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => net.step_infinite(x),
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
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`](crate::modules::Layers).
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`](crate::modules::Layers).
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`](crate::modules::Layer).
        mlp: Option<crate::modules::GatedMlpConfig>,
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
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`](crate::modules::Layers).
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`](crate::modules::Layers).
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`](crate::modules::Layer).
        mlp: Option<crate::modules::GatedMlpConfig>,
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
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`](crate::modules::Layers).
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`](crate::modules::Layers).
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`](crate::modules::Layer).
        mlp: Option<crate::modules::GatedMlpConfig>,
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
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaLatentNet::Mamba1(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_mlp(mlp.clone()),
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
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaLatentNet::Mamba2(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_mlp(mlp.clone()),
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
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaLatentNet::Mamba3(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_mlp(mlp.clone()),
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
        class: Option<&mut ClassCursors>,
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
                let (y, c) = net.forward(x, caches, (), class);
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
                let (y, c) = net.forward(x, caches, path, class);
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
                let (y, c) = net.forward(x, caches, path, class);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }

    /// Single-token step: token IDs `[batch]` → logits `[batch, padded_vocab]`.
    /// Cache family must match the network. `class` — the inner [`Layers`]
    /// cursors — is forwarded, see [`VocabNetwork::step`].
    pub fn step(
        &self,
        x: Tensor<1, Int>,
        caches: Option<MambaCaches>,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<2>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 network"),
                });
                let (y, c) = net.step(x, caches, class);
                (y, MambaCaches::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 network"),
                });
                let (y, c) = net.step(x, caches, class);
                (y, MambaCaches::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 network"),
                });
                let (y, c) = net.step(x, caches, class);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }

    /// Stationary fixed point under a constant token (no caches) — see
    /// [`VocabNetwork::step_infinite`]. Only the Mamba-3 family implements the
    /// closed form; the other variants panic.
    pub fn step_infinite(&self, x: Tensor<1, Int>) -> Tensor<2> {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => net.step_infinite(x),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => net.step_infinite(x),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => net.step_infinite(x),
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
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`](crate::modules::Layers).
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`](crate::modules::Layers).
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`](crate::modules::Layer).
        mlp: Option<crate::modules::GatedMlpConfig>,
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
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`](crate::modules::Layers).
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`](crate::modules::Layers).
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`](crate::modules::Layer).
        mlp: Option<crate::modules::GatedMlpConfig>,
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
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`](crate::modules::Layers).
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`](crate::modules::Layers).
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`](crate::modules::Layer).
        mlp: Option<crate::modules::GatedMlpConfig>,
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
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaVocabNet::Mamba1(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_mlp(mlp.clone()),
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
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaVocabNet::Mamba2(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_mlp(mlp.clone()),
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
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaVocabNet::Mamba3(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_mlp(mlp.clone()),
                    missing_lm_head: *missing_lm_head,
                }
                .init(device),
            ),
        }
    }
}
