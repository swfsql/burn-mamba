//! The run-time family choice: one enum over the three blocks.
//!
//! [`MambaBlock`] fills [`Block`](burn_stack::modules::Block). So each generic
//! container of `burn-stack` takes it with no extra code, and
//! [`MambaLatentNet`], [`MambaVocabNet`], [`MambaBidiLayers`] and
//! [`MambaHNetVocabNet`] are only type aliases. A call panics when its cache or
//! its SSD path is of another family than the block.
//! `docs/adr/0009-family-choice-at-the-block.md` gives the consequences for the
//! parameter paths, the caches and a mixed stack.

use crate::family::MambaSsdPath;
use burn::prelude::*;
use burn_stack::modules::bidi::BidiLayers;
use burn_stack::modules::hnet::HNetVocabNetwork;
use burn_stack::modules::{
    Block, CacheStack, CacheTensors, LatentNetwork, TensorZip, VocabNetwork,
};
use burn_stack::utils::UntiedParam;

#[cfg(feature = "mamba1")]
use crate::mamba1::prelude::{Mamba1, Mamba1Cache, Mamba1Caches};
#[cfg(feature = "mamba2")]
use crate::mamba2::prelude::{Mamba2, Mamba2Cache, Mamba2Caches, Mamba2SsdPath};
#[cfg(feature = "mamba3")]
use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Caches, Mamba3SsdPath};

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

/// The caches of a stack: a flat list of slots. Each slot has its own family
/// tag.
///
/// This is run-time state, and not a `Module`. `forward` and `step` take and
/// give caches, and no record holds them.
#[derive(Clone, Debug)]
pub struct MambaCaches {
    /// One entry for each virtual layer.
    pub slots: Vec<Option<MambaCache>>,
}

/// A latent network over any family.
pub type MambaLatentNet = LatentNetwork<MambaBlock>;
/// A vocab network over any family.
pub type MambaVocabNet = VocabNetwork<MambaBlock>;
/// A bidirectional stack over any family.
pub type MambaBidiLayers = BidiLayers<MambaBlock>;
/// An H-Net vocab network. Its stages hold one family, and its main stack can
/// hold another.
pub type MambaHNetVocabNet = HNetVocabNetwork<MambaBlock, MambaBlock>;

const WRONG_CACHE: &str = "the cache family does not match the block";
const WRONG_PATH: &str = "the SSD path family does not match the block";

#[cfg(feature = "mamba1")]
fn mamba1_cache(cache: MambaCache) -> Mamba1Cache {
    match cache {
        MambaCache::Mamba1(c) => c,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_CACHE}"),
    }
}

#[cfg(feature = "mamba2")]
fn mamba2_cache(cache: MambaCache) -> Mamba2Cache {
    match cache {
        MambaCache::Mamba2(c) => c,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_CACHE}"),
    }
}

#[cfg(feature = "mamba3")]
fn mamba3_cache(cache: MambaCache) -> Mamba3Cache {
    match cache {
        MambaCache::Mamba3(c) => c,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_CACHE}"),
    }
}

/// Mamba-1 has no SSD path, so this only checks the family.
#[cfg(feature = "mamba1")]
fn check_mamba1_path(path: MambaSsdPath) {
    match path {
        MambaSsdPath::Mamba1 => (),
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_PATH}"),
    }
}

#[cfg(feature = "mamba2")]
fn mamba2_path(path: MambaSsdPath) -> Mamba2SsdPath {
    match path {
        MambaSsdPath::Mamba2(p) => p,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_PATH}"),
    }
}

#[cfg(feature = "mamba3")]
fn mamba3_path(path: MambaSsdPath) -> Mamba3SsdPath {
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
                check_mamba1_path(options);
                let (y, c) = b.block_forward(x, cache.map(mamba1_cache), (), pad);
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) = b.block_forward(x, cache.map(mamba2_cache), mamba2_path(options), pad);
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) = b.block_forward(x, cache.map(mamba3_cache), mamba3_path(options), pad);
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
                check_mamba1_path(options);
                let (y, c) = b.block_forward_packed(x, cache.map(mamba1_cache), (), reset);
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) =
                    b.block_forward_packed(x, cache.map(mamba2_cache), mamba2_path(options), reset);
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) =
                    b.block_forward_packed(x, cache.map(mamba3_cache), mamba3_path(options), reset);
                (y, MambaCache::Mamba3(c))
            }
        }
    }

    fn reset_align(&self, options: &MambaSsdPath) -> Option<usize> {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                check_mamba1_path(options.clone());
                b.reset_align(&())
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => b.reset_align(&mamba2_path(options.clone())),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => b.reset_align(&mamba3_path(options.clone())),
        }
    }

    fn block_step(&self, x: Tensor<2>, cache: Option<MambaCache>) -> (Tensor<2>, MambaCache) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                let (y, c) = b.block_step(x, cache.map(mamba1_cache));
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) = b.block_step(x, cache.map(mamba2_cache));
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) = b.block_step(x, cache.map(mamba3_cache));
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
