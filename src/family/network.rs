//! Runtime-selectable networks: one enum over the three families'
//! [`LatentNetwork`](burn_stack::modules::LatentNetwork) /
//! [`VocabNetwork`](burn_stack::modules::VocabNetwork) monomorphisations. Their
//! serializable configs are in [`crate::family::config`].

use crate::prelude::*;
use burn::prelude::*;
use burn_stack::modules::{LatentNetwork, VocabNetwork};
use burn_stack::utils::{ClassCursors, Packed};

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
    /// Full-sequence pass. The `ssd_path` must match the family of the
    /// network. A mismatch is a caller error and panics with a message. `pad`
    /// marks a right-padded batch ([`LatentNetwork::forward`]).
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<MambaCaches>,
        ssd_path: MambaSsdPath,
        class: Option<&mut ClassCursors>,
        pad: Option<Tensor<2, Bool>>,
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
                let (y, c) = net.forward(x, caches, (), class, pad);
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
                let (y, c) = net.forward(x, caches, path, class, pad);
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
                let (y, c) = net.forward(x, caches, path, class, pad);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }

    /// Single-token step. It has no path argument (decoding is recurrent for
    /// all families). The cache family must match the network. `class` (the
    /// cursors of both marker levels) goes to the wrapped network (see
    /// [`LatentNetwork::step`]).
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

    /// Class-only step: emits the class markers that wait for the next user
    /// token, without one. Returns the last of them (`None` when none were
    /// waiting). The cache family must match the network (see
    /// [`LatentNetwork::prime`]).
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
    /// Full-sequence pass: token pairs `[batch, sequence, 2]` (id, flag) →
    /// logits `[batch, sequence, padded_vocab + 1]` (the vocab logits, then the
    /// flag logit, see [`VocabNetwork`]). The `ssd_path`/`caches` family must
    /// match the network. A mismatch is a caller error and panics. `pad` marks
    /// a right-padded batch ([`VocabNetwork::forward`]).
    pub fn forward(
        &self,
        x: Tensor<3, Int>,
        caches: Option<MambaCaches>,
        ssd_path: MambaSsdPath,
        class: Option<&mut ClassCursors>,
        pad: Option<Tensor<2, Bool>>,
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
                let (y, c) = net.forward(x, caches, (), class, pad);
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
                let (y, c) = net.forward(x, caches, path, class, pad);
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
                let (y, c) = net.forward(x, caches, path, class, pad);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }

    /// [`Self::forward`] over packed rows ([`VocabNetwork::forward_packed`]).
    /// At each reset of `packed`, every block starts again from a zero cache. A
    /// reset must be at a multiple of [`Self::pack_align`], and each sequence
    /// opens with [`Self::n_class_latents`] slots. Mamba-1 takes no packed
    /// rows (it panics).
    pub fn forward_packed(
        &self,
        x: Tensor<3, Int>,
        caches: Option<MambaCaches>,
        ssd_path: MambaSsdPath,
        packed: &Packed,
    ) -> (Tensor<3>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(_) => panic!("Mamba-1 takes no packed rows"),
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
                let (y, c) = net.forward_packed(x, caches, path, packed);
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
                let (y, c) = net.forward_packed(x, caches, path, packed);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }

    /// The reset alignment of the first block on `ssd_path`, in tokens (see
    /// [`Block::reset_align`]). [`Self::forward_packed`] accepts a reset only
    /// at a multiple of it. `None` for Mamba-1, which takes no packed rows.
    pub fn pack_align(&self, ssd_path: &MambaSsdPath) -> Option<usize> {
        match (self, ssd_path) {
            #[cfg(feature = "mamba1")]
            (Self::Mamba1(net), _) => net.layers.real_layers[0].block.reset_align(&()),
            #[cfg(feature = "mamba2")]
            (Self::Mamba2(net), MambaSsdPath::Mamba2(path)) => net.layers.real_layers[0].block.reset_align(path),
            #[cfg(feature = "mamba3")]
            (Self::Mamba3(net), MambaSsdPath::Mamba3(path)) => net.layers.real_layers[0].block.reset_align(path),
            #[allow(unreachable_patterns)]
            _ => panic!("ssd_path family does not match the network"),
        }
    }

    /// The class latents of the stack: the opening slots of each sequence of
    /// a packed row.
    pub fn n_class_latents(&self) -> usize {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => net.layers.class_latents.len(),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => net.layers.class_latents.len(),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => net.layers.class_latents.len(),
        }
    }

    /// Single-token step: token pairs `[batch, 2]` → logits
    /// `[batch, padded_vocab + 1]`. The cache family must match the network.
    /// `class` (the cursors of the inner [`Layers`]) goes to the wrapped network
    /// (see [`VocabNetwork::step`]).
    pub fn step(
        &self,
        x: Tensor<2, Int>,
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

    /// Class-only step: emits the class latents that wait for the next token,
    /// without one. Returns the logits of the last of them (`None` when none
    /// were waiting). The cache family must match (see
    /// [`VocabNetwork::prime`]).
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

    /// Whether every class latent is a `Start`. Then, after the opening, every
    /// `step` runs the same launches (the condition for a captured step, see
    /// [`Layers::only_start_latents`](burn_stack::modules::Layers::only_start_latents)).
    pub fn only_start_latents(&self) -> bool {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(net) => net.layers.only_start_latents(),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(net) => net.layers.only_start_latents(),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(net) => net.layers.only_start_latents(),
        }
    }
}
