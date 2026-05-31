use crate::modules::LayersBuilder;
use crate::modules::{ResidualsConfig, RmsNorm, RmsNormConfig};
use crate::prelude::*;
use crate::utils::Schedule;
use crate::utils::class::{
    assert_step_compatible, class_marker_output_indices, class_step_injections, init_class_emb,
    insert_class_markers,
};
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
        let x = self.insert_tokens(x);
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
    /// As in `forward`, the network's class tokens are part of the sequence that
    /// enters the layers, so each is threaded through the layers (carrying the
    /// inner cursors) just like the user token — only the user token's output is
    /// returned. A `None` cursor skips that level; `Middle`/`End` markers panic
    /// for the cursored level (use `forward`).
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        own_index: Option<&mut usize>,
        mut layers_own_index: Option<&mut usize>,
        mut layer_indices: Option<&mut Vec<usize>>,
    ) -> (Tensor<2>, M::Caches) {
        // Network-level class-token injection. Each class token is run through a
        // full network pass (carrying the inner cursors, so the layers splice
        // their own latents around it exactly as in `forward`), then the user
        // token; only the user token's output is returned.
        if let Some(cursor) = own_index {
            let [batch, input_size] = x.dims();
            let inj = class_step_injections(&self.class_tokens, "LatentNetwork");
            let emb = self.class_tokens_emb.as_ref();
            let mut caches = caches;
            while let Some(i) = inj.iter().position(|&p| p == *cursor) {
                let row = emb
                    .unwrap()
                    .val()
                    .narrow(0, i, 1)
                    .expand([batch, input_size]);
                let (_discard, c) = self.step(
                    row,
                    caches,
                    None,
                    layers_own_index.as_deref_mut(),
                    layer_indices.as_deref_mut(),
                );
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
        let x = self
            .embedding
            .forward(x.unsqueeze_dim::<2>(1))
            .squeeze_dim(1);
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
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
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
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
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
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
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
                residuals,
            } => MambaLatentNet::Mamba1(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone()),
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
                residuals,
            } => MambaLatentNet::Mamba2(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone()),
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
                residuals,
            } => MambaLatentNet::Mamba3(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone()),
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
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
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
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
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
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
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
                residuals,
            } => MambaVocabNet::Mamba1(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone()),
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
                residuals,
            } => MambaVocabNet::Mamba2(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone()),
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
                residuals,
            } => MambaVocabNet::Mamba3(
                VocabNetworkBuilder {
                    vocab_size: *vocab_size,
                    pad_vocab_size_multiple: *pad_vocab_size_multiple,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_residuals(residuals.clone()),
                    missing_lm_head: *missing_lm_head,
                }
                .init(device),
            ),
        }
    }
}
