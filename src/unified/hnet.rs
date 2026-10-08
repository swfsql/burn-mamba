//! Runtime-selectable H-Net language models: [`MambaHNetVocabNet`] and its
//! serializable [`MambaHNetVocabNetConfig`].
//!
//! The H-Net itself is block-generic and lives in `burn_stack::modules::hnet`.
//! Its encoders and decoders take one block family `E`, and its main network
//! takes another family `M` (`HNetVocabNetwork<E, M>`). This module wraps each
//! enabled pair `(E, M)` of Mamba families in one enum variant. So a caller
//! can choose both at runtime and serialize the choice. A cache or an SSD path
//! of the wrong family panics, as in the other enums of [`crate::unified`].

use crate::prelude::*;
use burn::config::Config;
use burn::prelude::*;
use burn_stack::modules::hnet::{HNetCaches, HNetVocabNetwork, HNetVocabShape, Routing, StepMode};
use burn_stack::modules::{Block, BlockConfig, CacheTensors, ModelConfigExt, TensorZip};

#[cfg(feature = "mamba1")]
use crate::mamba1::prelude::{Mamba1, Mamba1Config};
#[cfg(feature = "mamba2")]
use crate::mamba2::prelude::{Mamba2, Mamba2Config};
#[cfg(feature = "mamba3")]
use crate::mamba3::prelude::{Mamba3, Mamba3Config};

/// A Mamba block config, tagged with its family.
#[derive(Config, Debug)]
pub enum MambaBlockConfig {
    /// A Mamba-1 block.
    #[cfg(feature = "mamba1")]
    Mamba1(Mamba1Config),
    /// A Mamba-2 block.
    #[cfg(feature = "mamba2")]
    Mamba2(Mamba2Config),
    /// A Mamba-3 block.
    #[cfg(feature = "mamba3")]
    Mamba3(Mamba3Config),
}

impl MambaBlockConfig {
    /// The family tag: `"mamba1"`, `"mamba2"` or `"mamba3"`.
    pub fn family(&self) -> &'static str {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(_) => "mamba1",
            #[cfg(feature = "mamba2")]
            Self::Mamba2(_) => "mamba2",
            #[cfg(feature = "mamba3")]
            Self::Mamba3(_) => "mamba3",
        }
    }
}

/// The serializable config of a [`MambaHNetVocabNet`]: the shape of
/// `burn-stack` plus one block config per stage and one for the main network.
#[derive(Config, Debug)]
pub struct MambaHNetVocabNetConfig {
    /// The stacks, the vocabulary and the knobs of the chunking.
    pub shape: HNetVocabShape,
    /// One block config per stage, outermost first, all of one family. The
    /// block config of a stage builds both its encoder and its decoder, and
    /// sets the width of the stage.
    pub stage_blocks: Vec<MambaBlockConfig>,
    /// The block config of the main network.
    pub main_block: MambaBlockConfig,
}

impl MambaHNetVocabNetConfig {
    /// `(E, M)`: the family of the stages and that of the main network. With
    /// no stage, `E` is the family of the main network.
    pub fn families(&self) -> (&'static str, &'static str) {
        let main = self.main_block.family();
        let Some(first) = self.stage_blocks.first() else {
            return (main, main);
        };
        let stages = first.family();
        assert!(
            self.stage_blocks.iter().all(|b| b.family() == stages),
            "every stage of an H-Net takes the same block family"
        );
        (stages, main)
    }
}

/// A Mamba block type, with its config and its SSD path read out of the
/// family-tagged enums.
trait Family: Block {
    type Config: BlockConfig<Block = Self> + Clone;
    fn config(config: &MambaBlockConfig) -> &Self::Config;
    fn options(path: MambaSsdPath) -> Self::Options;
}

#[cfg(feature = "mamba1")]
impl Family for Mamba1 {
    type Config = Mamba1Config;
    fn config(config: &MambaBlockConfig) -> &Self::Config {
        match config {
            MambaBlockConfig::Mamba1(c) => c,
            #[allow(unreachable_patterns)]
            _ => panic!("the block config is not of Mamba-1"),
        }
    }
    fn options(path: MambaSsdPath) -> Self::Options {
        match path {
            MambaSsdPath::Mamba1 => (),
            #[allow(unreachable_patterns)]
            _ => panic!("ssd_path family does not match Mamba-1"),
        }
    }
}

#[cfg(feature = "mamba2")]
impl Family for Mamba2 {
    type Config = Mamba2Config;
    fn config(config: &MambaBlockConfig) -> &Self::Config {
        match config {
            MambaBlockConfig::Mamba2(c) => c,
            #[allow(unreachable_patterns)]
            _ => panic!("the block config is not of Mamba-2"),
        }
    }
    fn options(path: MambaSsdPath) -> Self::Options {
        match path {
            MambaSsdPath::Mamba2(p) => p,
            #[allow(unreachable_patterns)]
            _ => panic!("ssd_path family does not match Mamba-2"),
        }
    }
}

#[cfg(feature = "mamba3")]
impl Family for Mamba3 {
    type Config = Mamba3Config;
    fn config(config: &MambaBlockConfig) -> &Self::Config {
        match config {
            MambaBlockConfig::Mamba3(c) => c,
            #[allow(unreachable_patterns)]
            _ => panic!("the block config is not of Mamba-3"),
        }
    }
    fn options(path: MambaSsdPath) -> Self::Options {
        match path {
            MambaSsdPath::Mamba3(p) => p,
            #[allow(unreachable_patterns)]
            _ => panic!("ssd_path family does not match Mamba-3"),
        }
    }
}

/// The block configs of the stages, as the config type of family `F`.
fn stage_configs<F: Family>(config: &MambaHNetVocabNetConfig) -> Vec<F::Config> {
    config.stage_blocks.iter().map(|b| F::config(b).clone()).collect()
}

/// Calls `$mac!` with every pair of families: `(variant, E, M, feature of E,
/// feature of M)`. The feature names are also the family tags.
macro_rules! with_family_pairs {
    ($mac:ident) => {
        $mac! {
            (Mamba1Mamba1, Mamba1, Mamba1, "mamba1", "mamba1"),
            (Mamba1Mamba2, Mamba1, Mamba2, "mamba1", "mamba2"),
            (Mamba1Mamba3, Mamba1, Mamba3, "mamba1", "mamba3"),
            (Mamba2Mamba1, Mamba2, Mamba1, "mamba2", "mamba1"),
            (Mamba2Mamba2, Mamba2, Mamba2, "mamba2", "mamba2"),
            (Mamba2Mamba3, Mamba2, Mamba3, "mamba2", "mamba3"),
            (Mamba3Mamba1, Mamba3, Mamba1, "mamba3", "mamba1"),
            (Mamba3Mamba2, Mamba3, Mamba2, "mamba3", "mamba2"),
            (Mamba3Mamba3, Mamba3, Mamba3, "mamba3", "mamba3"),
        }
    };
}

macro_rules! define_hnet {
    ($(($v:ident, $e:ident, $m:ident, $fe:literal, $fm:literal)),* $(,)?) => {
        /// A runtime-selectable H-Net language model: one variant per pair
        /// `(family of the stages, family of the main network)`.
        #[derive(Module, Debug)]
        pub enum MambaHNetVocabNet {
            $(
                #[doc = concat!("Stages of `", stringify!($e), "`, main network of `", stringify!($m), "`.")]
                #[cfg(all(feature = $fe, feature = $fm))]
                $v(HNetVocabNetwork<$e, $m>),
            )*
        }

        /// The caches of a [`MambaHNetVocabNet`], tagged like it.
        ///
        /// Plain runtime state (not a `Module`), as [`MambaCaches`].
        #[derive(Clone)]
        pub enum MambaHNetCaches {
            $(
                #[doc = concat!("Caches of the `", stringify!($v), "` network.")]
                #[cfg(all(feature = $fe, feature = $fm))]
                $v(HNetCaches<$e, $m>),
            )*
        }

        impl MambaHNetVocabNet {
            /// Full-sequence pass: token pairs `[batch, sequence, 2]` → logits
            /// `[batch, sequence, padded_vocab + 1]`, the caches, and one
            /// [`Routing`] per stage. `paths` is `(the SSD path of the stages,
            /// that of the main network)`. `pad` marks a right-padded batch.
            /// See `HNet::forward` of `burn-stack`.
            pub fn forward(
                &self,
                x: Tensor<3, Int>,
                caches: Option<MambaHNetCaches>,
                paths: (MambaSsdPath, MambaSsdPath),
                pad: Option<Tensor<2, Bool>>,
            ) -> (Tensor<3>, MambaHNetCaches, Vec<Routing>) {
                match self {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        Self::$v(net) => {
                            let caches = caches.map(|c| match c {
                                MambaHNetCaches::$v(c) => c,
                                #[allow(unreachable_patterns)]
                                _ => panic!(concat!("the cache does not match the ", stringify!($v), " network")),
                            });
                            let options = (<$e as Family>::options(paths.0), <$m as Family>::options(paths.1));
                            let (y, c, routing) = net.forward(x, caches, options, pad);
                            (y, MambaHNetCaches::$v(c), routing)
                        }
                    )*
                }
            }

            /// Single-token step: token pairs `[batch, 2]` → logits `[batch,
            /// padded_vocab + 1]`, the caches, and one [`Routing`] per stage.
            /// See `HNet::step` of `burn-stack`.
            pub fn step(
                &self,
                x: Tensor<2, Int>,
                caches: Option<MambaHNetCaches>,
                mode: StepMode,
            ) -> (Tensor<2>, MambaHNetCaches, Vec<Routing>) {
                match self {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        Self::$v(net) => {
                            let caches = caches.map(|c| match c {
                                MambaHNetCaches::$v(c) => c,
                                #[allow(unreachable_patterns)]
                                _ => panic!(concat!("the cache does not match the ", stringify!($v), " network")),
                            });
                            let (y, c, routing) = net.step(x, caches, mode);
                            (y, MambaHNetCaches::$v(c), routing)
                        }
                    )*
                }
            }

            /// [`Self::forward`] over a **new** sequence, with the start rows of
            /// the network in front of it. The logits are `[batch, n_start +
            /// sequence, padded_vocab + 1]`. See
            /// `HNetVocabNetwork::forward_opened` of `burn-stack`.
            pub fn forward_opened(
                &self,
                x: Tensor<3, Int>,
                paths: (MambaSsdPath, MambaSsdPath),
                pad: Option<Tensor<2, Bool>>,
            ) -> (Tensor<3>, MambaHNetCaches, Vec<Routing>) {
                match self {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        Self::$v(net) => {
                            let options = (<$e as Family>::options(paths.0), <$m as Family>::options(paths.1));
                            let (y, c, routing) = net.forward_opened(x, options, pad);
                            (y, MambaHNetCaches::$v(c), routing)
                        }
                    )*
                }
            }

            /// Step the start rows alone, from a new sequence. Returns the
            /// logits of the last start row and the caches, or `None` when the
            /// network has no start rows. See `HNetVocabNetwork::prime` of
            /// `burn-stack`.
            pub fn prime(&self, batch: usize, mode: StepMode) -> Option<(Tensor<2>, MambaHNetCaches)> {
                match self {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        Self::$v(net) => net.prime(batch, mode).map(|(y, c)| (y, MambaHNetCaches::$v(c))),
                    )*
                }
            }

            /// The number of start rows.
            pub fn n_start(&self) -> usize {
                match self {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        Self::$v(net) => net.n_start(),
                    )*
                }
            }

            /// The number of stages.
            pub fn n_stages(&self) -> usize {
                match self {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        Self::$v(net) => net.hnet.stages.len(),
                    )*
                }
            }
        }

        impl MambaHNetCaches {
            /// The same values, cut off the autodiff graph (see
            /// `HNetCaches::detach` of `burn-stack`).
            pub fn detach(self) -> Self {
                match self {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        Self::$v(c) => Self::$v(c.detach()),
                    )*
                }
            }
        }

        impl CacheTensors for MambaHNetCaches {
            fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
                match (self, other) {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        (Self::$v(a), Self::$v(b)) => Self::$v(a.zip_tensors(b, z)),
                    )*
                    #[allow(unreachable_patterns)]
                    _ => panic!("the two caches differ in family"),
                }
            }
        }

        impl MambaHNetVocabNetConfig {
            /// Allocate and initialise the network on `device`.
            pub fn init(&self, device: &Device) -> MambaHNetVocabNet {
                match self.families() {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        ($fe, $fm) => MambaHNetVocabNet::$v(self.shape.init(
                            &stage_configs::<$e>(self),
                            <$m as Family>::config(&self.main_block),
                            device,
                        )),
                    )*
                    (e, m) => panic!("no H-Net of {e} stages and a {m} main network in this build"),
                }
            }

            /// The Muon plan of the network: each stack at its own block
            /// config, scoped to that stack (see `HNetShape::muon_plan` of
            /// `burn-stack`).
            #[cfg(feature = "optim")]
            pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
                match self.families() {
                    $(
                        #[cfg(all(feature = $fe, feature = $fm))]
                        ($fe, $fm) => self.shape.muon_plan(
                            &stage_configs::<$e>(self),
                            <$m as Family>::config(&self.main_block),
                        ),
                    )*
                    (e, m) => panic!("no H-Net of {e} stages and a {m} main network in this build"),
                }
            }
        }
    };
}

with_family_pairs!(define_hnet);

impl ModelConfigExt for MambaHNetVocabNetConfig {
    type Model = MambaHNetVocabNet;
    fn init(&self, device: &Device) -> Self::Model {
        self.init(device)
    }
    #[cfg(feature = "optim")]
    fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.muon_plan()
    }
}
