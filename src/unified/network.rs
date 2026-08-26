//! Runtime-selectable networks: one enum (plus a serializable `Config`) over
//! the three families' [`LatentNetwork`](burn_stack::modules::LatentNetwork) /
//! [`VocabNetwork`](burn_stack::modules::VocabNetwork) monomorphisations.

use crate::prelude::*;
use burn::config::Config;
use burn::prelude::*;
use burn_stack::modules::{
    LatentNetwork, LatentNetworkBuilder, LayersBuilder, ResidualsConfig, VocabNetwork,
    VocabNetworkBuilder,
};
use burn_stack::utils::{ClassCursors, ClassToken, Schedule};

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

    /// Class-only step: emits the class markers waiting for the next user token
    /// without one, returning the last of them (`None` when none were waiting).
    /// Cache family must match the network — see [`LatentNetwork::prime`].
    pub fn prime(
        &self,
        batch: usize,
        caches: Option<MambaCaches>,
        class: Option<&mut ClassCursors>,
    ) -> (Option<Tensor<2>>, Option<MambaCaches>) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 network"),
                });
                let (y, c) = net.prime(batch, caches, class);
                (y, c.map(MambaCaches::Mamba1))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 network"),
                });
                let (y, c) = net.prime(batch, caches, class);
                (y, c.map(MambaCaches::Mamba2))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 network"),
                });
                let (y, c) = net.prime(batch, caches, class);
                (y, c.map(MambaCaches::Mamba3))
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
        /// Back-propagate only the last `K` virtual layers, running everything
        /// below on the inner backend (truncated BPTT for deep recursion).
        /// `None` ⇒ track the whole stack. See
        /// [`Layers::grad_horizon`](burn_stack::modules::Layers::grad_horizon).
        grad_horizon: Option<usize>,
        /// Shared block config.
        mamba_block: crate::mamba1::prelude::Mamba1Config,
        /// Output feature width.
        output_size: usize,
        /// Insert a final RMSNorm before `out_proj` (see
        /// [`LatentNetwork::norm_f`]).
        final_norm: bool,
        /// Network-level class tokens, spliced into the input before `in_proj`.
        class_tokens: Vec<ClassToken>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first layer (width `d_model`, unlike the class tokens above).
        class_latents: Vec<ClassLatent>,
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`].
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`].
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`].
        mlp: Option<burn_stack::modules::GatedMlpConfig>,
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
        /// Back-propagate only the last `K` virtual layers, running everything
        /// below on the inner backend (truncated BPTT for deep recursion).
        /// `None` ⇒ track the whole stack. See
        /// [`Layers::grad_horizon`](burn_stack::modules::Layers::grad_horizon).
        grad_horizon: Option<usize>,
        /// Shared block config.
        mamba_block: crate::mamba2::prelude::Mamba2Config,
        /// Output feature width.
        output_size: usize,
        /// Insert a final RMSNorm before `out_proj` (see
        /// [`LatentNetwork::norm_f`]).
        final_norm: bool,
        /// Network-level class tokens, spliced into the input before `in_proj`.
        class_tokens: Vec<ClassToken>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first layer (width `d_model`, unlike the class tokens above).
        class_latents: Vec<ClassLatent>,
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`].
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`].
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`].
        mlp: Option<burn_stack::modules::GatedMlpConfig>,
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
        /// Back-propagate only the last `K` virtual layers, running everything
        /// below on the inner backend (truncated BPTT for deep recursion).
        /// `None` ⇒ track the whole stack. See
        /// [`Layers::grad_horizon`](burn_stack::modules::Layers::grad_horizon).
        grad_horizon: Option<usize>,
        /// Shared block config.
        mamba_block: crate::mamba3::prelude::Mamba3Config,
        /// Output feature width.
        output_size: usize,
        /// Insert a final RMSNorm before `out_proj` (see
        /// [`LatentNetwork::norm_f`]).
        final_norm: bool,
        /// Network-level class tokens, spliced into the input before `in_proj`.
        class_tokens: Vec<ClassToken>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first layer (width `d_model`, unlike the class tokens above).
        class_latents: Vec<ClassLatent>,
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`].
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`].
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`].
        mlp: Option<burn_stack::modules::GatedMlpConfig>,
    },
}

impl MambaLatentNetConfig {
    /// The [`MuonPlan`] for this network: the block's
    /// (and the optional MLP's) fused projections.
    ///
    /// The network's own boundary weights — `in_proj`/`out_proj` (or the
    /// embedding and LM head) and any class-token table — are deliberately left
    /// out; see [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        let (specs, mlp) = match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                mamba_block, mlp, ..
            } => (mamba_block.muon_projections(), mlp.clone()),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                mamba_block, mlp, ..
            } => (mamba_block.muon_projections(), mlp.clone()),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                mamba_block, mlp, ..
            } => (mamba_block.muon_projections(), mlp.clone()),
        };
        burn_stack::optim::MuonPlan::new(specs).with_mlp(mlp.as_ref())
    }

    /// Allocate and initialise the selected network on `device`.
    pub fn init(&self, device: &Device) -> MambaLatentNet {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                input_size,
                n_real_layers,
                n_virtual_layers,
                grad_horizon,
                mamba_block,
                output_size,
                final_norm,
                class_tokens,
                class_latents,
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaLatentNet::Mamba1(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_grad_horizon(*grad_horizon)
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_class_latents(class_latents.clone())
                        .with_mlp(mlp.clone()),
                    output_size: *output_size,
                    final_norm: *final_norm,
                    class_tokens: class_tokens.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                input_size,
                n_real_layers,
                n_virtual_layers,
                grad_horizon,
                mamba_block,
                output_size,
                final_norm,
                class_tokens,
                class_latents,
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaLatentNet::Mamba2(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_grad_horizon(*grad_horizon)
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_class_latents(class_latents.clone())
                        .with_mlp(mlp.clone()),
                    output_size: *output_size,
                    final_norm: *final_norm,
                    class_tokens: class_tokens.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                input_size,
                n_real_layers,
                n_virtual_layers,
                grad_horizon,
                mamba_block,
                output_size,
                final_norm,
                class_tokens,
                class_latents,
                ignore_first_residual,
                ignore_last_residual,
                residuals,
                mlp,
            } => MambaLatentNet::Mamba3(
                LatentNetworkBuilder {
                    input_size: *input_size,
                    layers: LayersBuilder::new(*n_real_layers, mamba_block.clone())
                        .with_n_virtual_layers(n_virtual_layers.clone())
                        .with_grad_horizon(*grad_horizon)
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_class_latents(class_latents.clone())
                        .with_mlp(mlp.clone()),
                    output_size: *output_size,
                    final_norm: *final_norm,
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

    /// Class-only step: emits the class latents waiting for the next token
    /// without one, returning the logits of the last of them (`None` when none
    /// were waiting). Cache family must match — see [`VocabNetwork::prime`].
    pub fn prime(
        &self,
        batch: usize,
        caches: Option<MambaCaches>,
        class: Option<&mut ClassCursors>,
    ) -> (Option<Tensor<2>>, Option<MambaCaches>) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 network"),
                });
                let (y, c) = net.prime(batch, caches, class);
                (y, c.map(MambaCaches::Mamba1))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 network"),
                });
                let (y, c) = net.prime(batch, caches, class);
                (y, c.map(MambaCaches::Mamba2))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 network"),
                });
                let (y, c) = net.prime(batch, caches, class);
                (y, c.map(MambaCaches::Mamba3))
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
        /// Back-propagate only the last `K` virtual layers, running everything
        /// below on the inner backend (truncated BPTT for deep recursion).
        /// `None` ⇒ track the whole stack. See
        /// [`Layers::grad_horizon`](burn_stack::modules::Layers::grad_horizon).
        grad_horizon: Option<usize>,
        /// Unpadded vocabulary size.
        vocab_size: usize,
        /// Round `vocab_size` up to a multiple of this (1 disables rounding).
        pad_vocab_size_multiple: usize,
        /// Shared block config.
        mamba_block: crate::mamba1::prelude::Mamba1Config,
        /// Tie the LM head to the (transposed) embedding weights when `true`.
        missing_lm_head: bool,
        /// Stack-level class latents, spliced into the sequence before the
        /// first layer (width `d_model`).
        class_latents: Vec<ClassLatent>,
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`].
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`].
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`].
        mlp: Option<burn_stack::modules::GatedMlpConfig>,
    },
    /// Build a Mamba-2 language model.
    #[cfg(feature = "mamba2")]
    Mamba2 {
        /// Number of real layers.
        n_real_layers: usize,
        /// Optional virtual-layer scheduling.
        n_virtual_layers: Option<(usize, Schedule)>,
        /// Back-propagate only the last `K` virtual layers, running everything
        /// below on the inner backend (truncated BPTT for deep recursion).
        /// `None` ⇒ track the whole stack. See
        /// [`Layers::grad_horizon`](burn_stack::modules::Layers::grad_horizon).
        grad_horizon: Option<usize>,
        /// Unpadded vocabulary size.
        vocab_size: usize,
        /// Round `vocab_size` up to a multiple of this (1 disables rounding).
        pad_vocab_size_multiple: usize,
        /// Shared block config.
        mamba_block: crate::mamba2::prelude::Mamba2Config,
        /// Tie the LM head to the (transposed) embedding weights when `true`.
        missing_lm_head: bool,
        /// Stack-level class latents, spliced into the sequence before the
        /// first layer (width `d_model`).
        class_latents: Vec<ClassLatent>,
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`].
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`].
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`].
        mlp: Option<burn_stack::modules::GatedMlpConfig>,
    },
    /// Build a Mamba-3 language model.
    #[cfg(feature = "mamba3")]
    Mamba3 {
        /// Number of real layers.
        n_real_layers: usize,
        /// Optional virtual-layer scheduling.
        n_virtual_layers: Option<(usize, Schedule)>,
        /// Back-propagate only the last `K` virtual layers, running everything
        /// below on the inner backend (truncated BPTT for deep recursion).
        /// `None` ⇒ track the whole stack. See
        /// [`Layers::grad_horizon`](burn_stack::modules::Layers::grad_horizon).
        grad_horizon: Option<usize>,
        /// Unpadded vocabulary size.
        vocab_size: usize,
        /// Round `vocab_size` up to a multiple of this (1 disables rounding).
        pad_vocab_size_multiple: usize,
        /// Shared block config.
        mamba_block: crate::mamba3::prelude::Mamba3Config,
        /// Tie the LM head to the (transposed) embedding weights when `true`.
        missing_lm_head: bool,
        /// Stack-level class latents, spliced into the sequence before the
        /// first layer (width `d_model`).
        class_latents: Vec<ClassLatent>,
        /// Suppress the first virtual layer's residual (Pre-LN skip / MultiGate
        /// seed carry). See [`Layers`].
        ignore_first_residual: bool,
        /// Suppress the last virtual layer's residual (output is the last
        /// layer's transform alone). See [`Layers`].
        ignore_last_residual: bool,
        /// Inter-layer residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
        /// Optional per-layer SwiGLU feed-forward sub-block (`d_intermediate` in
        /// the reference configs), with its own pre-norm and inner residual.
        /// `None` ⇒ mixer-only layers. See [`Layer`].
        mlp: Option<burn_stack::modules::GatedMlpConfig>,
    },
}

impl MambaVocabNetConfig {
    /// The [`MuonPlan`] for this network: the block's
    /// (and the optional MLP's) fused projections.
    ///
    /// The network's own boundary weights — `in_proj`/`out_proj` (or the
    /// embedding and LM head) and any class-token table — are deliberately left
    /// out; see [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        let (specs, mlp) = match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                mamba_block, mlp, ..
            } => (mamba_block.muon_projections(), mlp.clone()),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                mamba_block, mlp, ..
            } => (mamba_block.muon_projections(), mlp.clone()),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                mamba_block, mlp, ..
            } => (mamba_block.muon_projections(), mlp.clone()),
        };
        burn_stack::optim::MuonPlan::new(specs).with_mlp(mlp.as_ref())
    }

    /// Allocate and initialise the selected language model on `device`.
    pub fn init(&self, device: &Device) -> MambaVocabNet {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                n_real_layers,
                n_virtual_layers,
                grad_horizon,
                vocab_size,
                pad_vocab_size_multiple,
                mamba_block,
                missing_lm_head,
                class_latents,
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
                        .with_grad_horizon(*grad_horizon)
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_class_latents(class_latents.clone())
                        .with_mlp(mlp.clone()),
                    missing_lm_head: *missing_lm_head,
                }
                .init(device),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                n_real_layers,
                n_virtual_layers,
                grad_horizon,
                vocab_size,
                pad_vocab_size_multiple,
                mamba_block,
                missing_lm_head,
                class_latents,
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
                        .with_grad_horizon(*grad_horizon)
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_class_latents(class_latents.clone())
                        .with_mlp(mlp.clone()),
                    missing_lm_head: *missing_lm_head,
                }
                .init(device),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                n_real_layers,
                n_virtual_layers,
                grad_horizon,
                vocab_size,
                pad_vocab_size_multiple,
                mamba_block,
                missing_lm_head,
                class_latents,
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
                        .with_grad_horizon(*grad_horizon)
                        .with_residuals(residuals.clone())
                        .with_ignore_first_residual(*ignore_first_residual)
                        .with_ignore_last_residual(*ignore_last_residual)
                        .with_class_latents(class_latents.clone())
                        .with_mlp(mlp.clone()),
                    missing_lm_head: *missing_lm_head,
                }
                .init(device),
            ),
        }
    }
}
