//! Throwaway spike: the run-time family choice at the block.
//!
//! A blank project, with only the public interface of `burn-mamba` and
//! `burn-stack`. It holds what candidates 1, 2 and the Q4 decision would add
//! to `burn-mamba/src/unified/`.

use burn::module::Module;
use burn::prelude::*;
use burn_mamba::prelude::MambaSsdPath;
#[cfg(feature = "mamba1")]
use burn_mamba::mamba1::prelude::{Mamba1, Mamba1Cache, Mamba1Caches, Mamba1Config};
#[cfg(feature = "mamba2")]
use burn_mamba::mamba2::prelude::{Mamba2, Mamba2Cache, Mamba2Caches, Mamba2Config, Mamba2SsdPath};
#[cfg(feature = "mamba3")]
use burn_mamba::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Caches, Mamba3Config, Mamba3SsdPath};
use burn_stack::modules::bidi::BidiLayers;
use burn_stack::modules::hnet::{HNetVocabNetwork, HNetVocabShape};
use burn_stack::modules::{
    BidiShape, Block, BlockConfig, CacheStack, CacheTensors, LatentNetwork, LatentShape, ModelConfigExt, TensorZip,
    VocabNetwork, VocabShape,
};
use burn_stack::utils::UntiedParam;

// ---------------------------------------------------------------------------
// The block
// ---------------------------------------------------------------------------

/// A Mamba block, tagged with its family.
#[derive(Module, Debug)]
pub enum MambaBlock {
    /// A Mamba-1 block.
    #[cfg(feature = "mamba1")]
    Mamba1(Mamba1),
    /// A Mamba-2 block.
    #[cfg(feature = "mamba2")]
    Mamba2(Mamba2),
    /// A Mamba-3 block.
    #[cfg(feature = "mamba3")]
    Mamba3(Mamba3),
}

/// The cache of one block, tagged with its family.
#[derive(Clone, Debug)]
pub enum MambaCache {
    /// A Mamba-1 cache.
    #[cfg(feature = "mamba1")]
    Mamba1(Mamba1Cache),
    /// A Mamba-2 cache.
    #[cfg(feature = "mamba2")]
    Mamba2(Mamba2Cache),
    /// A Mamba-3 cache.
    #[cfg(feature = "mamba3")]
    Mamba3(Mamba3Cache),
}

/// The caches of a stack: a flat list of slots. Each slot has its own tag, so
/// `from_slots` guesses nothing.
#[derive(Clone, Debug)]
pub struct MambaCaches {
    /// One entry for each virtual layer.
    pub slots: Vec<Option<MambaCache>>,
}

const WRONG_CACHE: &str = "the cache family does not match the block";
const WRONG_PATH: &str = "ssd_path family does not match the block";

#[cfg(feature = "mamba1")]
fn cache1(cache: MambaCache) -> Mamba1Cache {
    match cache {
        MambaCache::Mamba1(c) => c,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_CACHE}"),
    }
}
#[cfg(feature = "mamba2")]
fn cache2(cache: MambaCache) -> Mamba2Cache {
    match cache {
        MambaCache::Mamba2(c) => c,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_CACHE}"),
    }
}
#[cfg(feature = "mamba3")]
fn cache3(cache: MambaCache) -> Mamba3Cache {
    match cache {
        MambaCache::Mamba3(c) => c,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_CACHE}"),
    }
}
#[cfg(feature = "mamba1")]
fn path1(path: MambaSsdPath) {
    match path {
        MambaSsdPath::Mamba1 => (),
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_PATH}"),
    }
}
#[cfg(feature = "mamba2")]
fn path2(path: MambaSsdPath) -> Mamba2SsdPath {
    match path {
        MambaSsdPath::Mamba2(p) => p,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_PATH}"),
    }
}
#[cfg(feature = "mamba3")]
fn path3(path: MambaSsdPath) -> Mamba3SsdPath {
    match path {
        MambaSsdPath::Mamba3(p) => p,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_PATH}"),
    }
}

fn tag_slots<C: CacheStack>(stack: C, tag: fn(C::Cache) -> MambaCache) -> MambaCaches {
    MambaCaches {
        slots: stack.into_slots().into_iter().map(|slot| slot.map(tag)).collect(),
    }
}

impl Block for MambaBlock {
    type Cache = MambaCache;
    type Caches = MambaCaches;
    type Options = MambaSsdPath;

    fn block_forward(
        &self,
        x: Tensor<3>,
        cache: Option<MambaCache>,
        options: MambaSsdPath,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, MambaCache) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                path1(options);
                let (y, c) = b.block_forward(x, cache.map(cache1), (), pad);
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) = b.block_forward(x, cache.map(cache2), path2(options), pad);
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) = b.block_forward(x, cache.map(cache3), path3(options), pad);
                (y, MambaCache::Mamba3(c))
            }
        }
    }

    fn block_forward_packed(
        &self,
        x: Tensor<3>,
        cache: Option<MambaCache>,
        options: MambaSsdPath,
        reset: Tensor<2, Bool>,
    ) -> (Tensor<3>, MambaCache) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                path1(options);
                let (y, c) = b.block_forward_packed(x, cache.map(cache1), (), reset);
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) = b.block_forward_packed(x, cache.map(cache2), path2(options), reset);
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) = b.block_forward_packed(x, cache.map(cache3), path3(options), reset);
                (y, MambaCache::Mamba3(c))
            }
        }
    }

    fn block_step(&self, x: Tensor<2>, cache: Option<MambaCache>) -> (Tensor<2>, MambaCache) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                let (y, c) = b.block_step(x, cache.map(cache1));
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) = b.block_step(x, cache.map(cache2));
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) = b.block_step(x, cache.map(cache3));
                (y, MambaCache::Mamba3(c))
            }
        }
    }

    fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> MambaCaches {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => tag_slots(b.zero_caches_3d(x, n_virtual), MambaCache::Mamba1),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => tag_slots(b.zero_caches_3d(x, n_virtual), MambaCache::Mamba2),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => tag_slots(b.zero_caches_3d(x, n_virtual), MambaCache::Mamba3),
        }
    }

    fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> MambaCaches {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => tag_slots(b.zero_caches_2d(x, n_virtual), MambaCache::Mamba1),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => tag_slots(b.zero_caches_2d(x, n_virtual), MambaCache::Mamba2),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => tag_slots(b.zero_caches_2d(x, n_virtual), MambaCache::Mamba3),
        }
    }

    fn untied_params(&self) -> Vec<UntiedParam> {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => Block::untied_params(b),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => Block::untied_params(b),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => Block::untied_params(b),
        }
    }
}

impl CacheStack for MambaCaches {
    type Cache = MambaCache;

    fn slot_count(&self) -> usize {
        self.slots.len()
    }
    fn into_slots(self) -> Vec<Option<MambaCache>> {
        self.slots
    }
    fn from_slots(slots: Vec<Option<MambaCache>>) -> Self {
        Self { slots }
    }
    fn cache_to_inner(cache: MambaCache) -> MambaCache {
        match cache {
            #[cfg(feature = "mamba1")]
            MambaCache::Mamba1(c) => MambaCache::Mamba1(Mamba1Caches::cache_to_inner(c)),
            #[cfg(feature = "mamba2")]
            MambaCache::Mamba2(c) => MambaCache::Mamba2(Mamba2Caches::cache_to_inner(c)),
            #[cfg(feature = "mamba3")]
            MambaCache::Mamba3(c) => MambaCache::Mamba3(Mamba3Caches::cache_to_inner(c)),
        }
    }
    fn cache_from_inner(cache: MambaCache, device: &Device) -> MambaCache {
        match cache {
            #[cfg(feature = "mamba1")]
            MambaCache::Mamba1(c) => MambaCache::Mamba1(Mamba1Caches::cache_from_inner(c, device)),
            #[cfg(feature = "mamba2")]
            MambaCache::Mamba2(c) => MambaCache::Mamba2(Mamba2Caches::cache_from_inner(c, device)),
            #[cfg(feature = "mamba3")]
            MambaCache::Mamba3(c) => MambaCache::Mamba3(Mamba3Caches::cache_from_inner(c, device)),
        }
    }
    fn cache_device(cache: &MambaCache) -> Device {
        match cache {
            #[cfg(feature = "mamba1")]
            MambaCache::Mamba1(c) => Mamba1Caches::cache_device(c),
            #[cfg(feature = "mamba2")]
            MambaCache::Mamba2(c) => Mamba2Caches::cache_device(c),
            #[cfg(feature = "mamba3")]
            MambaCache::Mamba3(c) => Mamba3Caches::cache_device(c),
        }
    }
}

impl CacheTensors for MambaCache {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        match (self, other) {
            #[cfg(feature = "mamba1")]
            (Self::Mamba1(a), Self::Mamba1(b)) => Self::Mamba1(a.zip_tensors(b, z)),
            #[cfg(feature = "mamba2")]
            (Self::Mamba2(a), Self::Mamba2(b)) => Self::Mamba2(a.zip_tensors(b, z)),
            #[cfg(feature = "mamba3")]
            (Self::Mamba3(a), Self::Mamba3(b)) => Self::Mamba3(a.zip_tensors(b, z)),
            #[allow(unreachable_patterns)]
            _ => panic!("the two caches differ in family"),
        }
    }
}

impl CacheTensors for MambaCaches {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        assert_eq!(self.slots.len(), other.slots.len(), "the two caches differ in slot count");
        let slots = self
            .slots
            .into_iter()
            .zip(other.slots)
            .map(|pair| match pair {
                (Some(a), Some(b)) => Some(a.zip_tensors(b, z)),
                (None, None) => None,
                _ => panic!("the two caches differ in which slots they hold"),
            })
            .collect();
        Self { slots }
    }
}

// ---------------------------------------------------------------------------
// The configs
// ---------------------------------------------------------------------------

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

impl BlockConfig for MambaBlockConfig {
    type Block = MambaBlock;

    fn d_model(&self) -> usize {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(c) => BlockConfig::d_model(c),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(c) => BlockConfig::d_model(c),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(c) => BlockConfig::d_model(c),
        }
    }
    fn init_block(&self, n_applications: usize, device: &Device) -> MambaBlock {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(c) => MambaBlock::Mamba1(c.init_block(n_applications, device)),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(c) => MambaBlock::Mamba2(c.init_block(n_applications, device)),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(c) => MambaBlock::Mamba3(c.init_block(n_applications, device)),
        }
    }
    fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(c) => BlockConfig::muon_projections(c),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(c) => BlockConfig::muon_projections(c),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(c) => BlockConfig::muon_projections(c),
        }
    }
}

/// A latent network over any family.
pub type MambaLatentNet = LatentNetwork<MambaBlock>;
/// A vocab network over any family.
pub type MambaVocabNet = VocabNetwork<MambaBlock>;
/// A bidirectional stack over any family.
pub type MambaBidiLayers = BidiLayers<MambaBlock>;
/// An H-Net vocab network, with a family for its stages and one for its main stack.
pub type MambaHNetVocabNet = HNetVocabNetwork<MambaBlock, MambaBlock>;

/// The config of a [`MambaLatentNet`].
#[derive(Config, Debug)]
pub struct MambaLatentNetConfig {
    /// Everything about the network except its block.
    pub shape: LatentShape,
    /// The block.
    pub block: MambaBlockConfig,
}

impl ModelConfigExt for MambaLatentNetConfig {
    type Model = MambaLatentNet;
    fn init(&self, device: &Device) -> MambaLatentNet {
        self.shape.init(self.block.clone(), device)
    }
    fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.shape.stack.muon_plan(&self.block)
    }
}

/// The config of a [`MambaVocabNet`].
#[derive(Config, Debug)]
pub struct MambaVocabNetConfig {
    /// Everything about the network except its block.
    pub shape: VocabShape,
    /// The block.
    pub block: MambaBlockConfig,
}

impl ModelConfigExt for MambaVocabNetConfig {
    type Model = MambaVocabNet;
    fn init(&self, device: &Device) -> MambaVocabNet {
        self.shape.init(self.block.clone(), device)
    }
    fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.shape.stack.muon_plan(&self.block)
    }
}

/// The config of a [`MambaBidiLayers`].
#[derive(Config, Debug)]
pub struct MambaBidiLayersConfig {
    /// Everything about the stack except its block.
    pub shape: BidiShape,
    /// The block.
    pub block: MambaBlockConfig,
}

impl ModelConfigExt for MambaBidiLayersConfig {
    type Model = MambaBidiLayers;
    fn init(&self, device: &Device) -> MambaBidiLayers {
        self.shape.init(self.block.clone(), device)
    }
    fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.shape.muon_plan(&self.block)
    }
}

/// The config of a [`MambaHNetVocabNet`].
#[derive(Config, Debug)]
pub struct MambaHNetVocabNetConfig {
    /// Everything about the network except its blocks.
    pub shape: HNetVocabShape,
    /// One block config per stage, outermost first.
    pub stage_blocks: Vec<MambaBlockConfig>,
    /// The block config of the main stack.
    pub main_block: MambaBlockConfig,
}

impl ModelConfigExt for MambaHNetVocabNetConfig {
    type Model = MambaHNetVocabNet;
    fn init(&self, device: &Device) -> MambaHNetVocabNet {
        self.shape.init(&self.stage_blocks, &self.main_block, device)
    }
    fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.shape.muon_plan(&self.stage_blocks, &self.main_block)
    }
}
