//! The run-time family choice: one enum over the three blocks.
//!
//! [`MambaBlock`] fills [`Block`](burn_stack::modules::Block). So each generic
//! container of `burn-stack` takes it with no extra code, and
//! [`MambaLatentNet`], [`MambaVocabNet`], [`MambaBidiLayers`] and
//! [`MambaHNetVocabNet`] are only type aliases. A call panics when its cache or
//! its scan options are of another family than the block.
//! `docs/adr/0009-family-choice-at-the-block.md` gives the consequences for the
//! parameter paths, the caches and a mixed stack.

use crate::family::MambaScanOptions;
use burn::prelude::*;
use burn_stack::modules::bidi::BidiLayers;
use burn_stack::modules::hnet::HNetVocabNetwork;
use burn_stack::modules::{Block, CacheTensors, LatentNetwork, TensorZip, VocabNetwork};
use burn_stack::utils::UntiedParam;

#[cfg(feature = "mamba1")]
use crate::mamba1::prelude::{Mamba1, Mamba1Cache};
#[cfg(feature = "mamba2")]
use crate::mamba2::prelude::{Mamba2, Mamba2Cache, Mamba2ScanOptions};
#[cfg(feature = "mamba3")]
use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3ScanOptions};

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

/// The cache of one block, tagged with its family. The caches of a layer stack
/// are a `Vec<MambaCache>`, and each cache has its own family tag.
///
/// This is run-time state, and not a `Module`. `forward` and `step` take and
/// give caches, and no record holds them.
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
const WRONG_SCAN_OPTIONS: &str = "the family of the scan options does not match the block";

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

/// Mamba-1 has no scan options, so this only checks the family.
#[cfg(feature = "mamba1")]
fn check_mamba1_scan_options(scan_options: MambaScanOptions) {
    match scan_options {
        MambaScanOptions::Mamba1 => (),
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_SCAN_OPTIONS}"),
    }
}

#[cfg(feature = "mamba2")]
fn mamba2_scan_options(scan_options: MambaScanOptions) -> Mamba2ScanOptions {
    match scan_options {
        MambaScanOptions::Mamba2(scan_options) => scan_options,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_SCAN_OPTIONS}"),
    }
}

#[cfg(feature = "mamba3")]
fn mamba3_scan_options(scan_options: MambaScanOptions) -> Mamba3ScanOptions {
    match scan_options {
        MambaScanOptions::Mamba3(scan_options) => scan_options,
        #[allow(unreachable_patterns)]
        _ => panic!("{WRONG_SCAN_OPTIONS}"),
    }
}

impl Block for MambaBlock {
    type Cache = MambaCache;
    type Options = MambaScanOptions;

    fn block_forward(
        &self,
        x: Tensor<3>,
        cache: MambaCache,
        options: MambaScanOptions,
        pad: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, MambaCache) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                check_mamba1_scan_options(options);
                let (y, c) = b.block_forward(x, mamba1_cache(cache), (), pad);
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) = b.block_forward(x, mamba2_cache(cache), mamba2_scan_options(options), pad);
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) = b.block_forward(x, mamba3_cache(cache), mamba3_scan_options(options), pad);
                (y, MambaCache::Mamba3(c))
            }
        }
    }

    fn block_forward_packed(
        &self,
        x: Tensor<3>,
        cache: MambaCache,
        origin: MambaCache,
        options: MambaScanOptions,
        reset: Tensor<2, Bool>,
    ) -> (Tensor<3>, MambaCache) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                check_mamba1_scan_options(options);
                let (y, c) =
                    b.block_forward_packed(x, mamba1_cache(cache), mamba1_cache(origin), (), reset);
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) = b.block_forward_packed(
                    x,
                    mamba2_cache(cache),
                    mamba2_cache(origin),
                    mamba2_scan_options(options),
                    reset,
                );
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) = b.block_forward_packed(
                    x,
                    mamba3_cache(cache),
                    mamba3_cache(origin),
                    mamba3_scan_options(options),
                    reset,
                );
                (y, MambaCache::Mamba3(c))
            }
        }
    }

    fn reset_align(&self, options: &MambaScanOptions) -> Option<usize> {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                check_mamba1_scan_options(options.clone());
                b.reset_align(&())
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => b.reset_align(&mamba2_scan_options(options.clone())),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => b.reset_align(&mamba3_scan_options(options.clone())),
        }
    }

    fn block_step(&self, x: Tensor<2>, cache: MambaCache) -> (Tensor<2>, MambaCache) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => {
                let (y, c) = b.block_step(x, mamba1_cache(cache));
                (y, MambaCache::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => {
                let (y, c) = b.block_step(x, mamba2_cache(cache));
                (y, MambaCache::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => {
                let (y, c) = b.block_step(x, mamba3_cache(cache));
                (y, MambaCache::Mamba3(c))
            }
        }
    }

    /// The zero cache of the family of the block, tagged with that family.
    fn zero_cache(&self, batch: usize, device: &Device) -> MambaCache {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(b) => MambaCache::Mamba1(Block::zero_cache(b, batch, device)),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(b) => MambaCache::Mamba2(Block::zero_cache(b, batch, device)),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(b) => MambaCache::Mamba3(Block::zero_cache(b, batch, device)),
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
