//! # The run-time family choice
//!
//! [`burn_stack`] composes any [`Block`](burn_stack::modules::Block) into
//! layers and networks, but `M` is fixed at the type level. This module lets a
//! caller choose the family at *run time*, and serialize that choice. The choice
//! sits at the block: [`MambaBlock`] is one enum over the three blocks, and it
//! fills `Block` (`docs/adr/0009-family-choice-at-the-block.md`). So the
//! containers need no enum of their own. [`MambaLatentNet`], [`MambaVocabNet`],
//! [`MambaBidiLayers`] and [`MambaHNetVocabNet`] are type aliases, and each has
//! a `{ shape, block }` config in [`config`]. [`MambaCaches`] is a flat list of
//! slots, and [`MambaSsdPath`] is the `Options` of the enum. A cache or an SSD
//! path of the wrong family panics.
//!
//! This is also where the three families meet the generic stack: the
//! `impl Block for Mamba{1,2,3}`, `impl BlockConfig for Mamba{1,2,3}Config` and
//! `impl CacheStack for Mamba{1,2,3}Caches` blocks are in [`adapters`]. The
//! cache traversal of each family is in [`cache_traversal`].
//!
//! ## Muon: why the 3-D tensors are not "stacked matrices"
//!
//! The model keeps its fused projections, and only the Muon plan splits them.
//! [`burn_stack::optim`] lets Muon own only the weights that a
//! [`BlockConfig::muon_projections`](burn_stack::modules::BlockConfig::muon_projections)
//! names. Each family lists there the weights that Muon can own, and splits
//! each fused projection as its forward does. The item doc of each list gives
//! what stays on the fallback optimizer, and why. The list of each family
//! deliberately stops at rank 2. For the MIMO tensors of Mamba-3 this needs an
//! argument, because their *shape* suggests the opposite conclusion.
//!
//! The *math* of MIMO does decompose along the rank: R write-channels summed
//! into one state and read out R ways, each part a **standalone** SISO run.
//! This holds because the transition is a function of no sample
//! (`info/mamba-3/mimo-as-batch.md` §5: it is isotropy, not a property of
//! MIMO, and no delta-rule shape has it). So a `[nheads, mimo_rank,
//! per_head_dim]` tensor looks like a stack of matrices that a stack-aware Muon
//! could take one slice at a time.
//!
//! It is not. The paper deliberately does not instantiate the R maps, because
//! that would multiply the parameter count of every per-head projection by R
//! (`DP → DPR` *per head*). It keeps the SISO projection and scales its output
//! element-wise to size R with a learnable vector (`DP + PR` per head). In the
//! contraction notation of the appendix, `X = contract(PR, P → PR)(W_X, X')`:
//! `P` is in both inputs *and* the output, so it is never contracted. So
//! `mimo_x`/`mimo_z`/`mimo_o` are **diagonals**, and their matrix shape is a
//! layout coincidence. To orthogonalise one would constrain a set of gains: R
//! fixed measurement gains on one value, never R maps
//! (`info/mamba-3/mimo-as-batch.md` §4).
//!
//! The part of MIMO that really is an R-fold matrix expansion is B/C
//! (`DN → DNR`), which is in `in_proj` and already belongs to Muon. The other
//! 3-D tensors are a bias (`b_bias_hmr`/`c_bias_hmr`), an initial condition
//! (`init_state_hpr`), or a depthwise filter (the Mamba-1/2 conv): again
//! diagonal or embedding-like.
//!

/// The `Block` / `BlockConfig` / `CacheStack` impls of each family.
pub mod adapters;
/// [`MambaBlock`], its caches, and the type aliases of the containers.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub mod block;
/// The cache traversal of each family: the one list of the tensors of each
/// cache type.
pub mod cache_traversal;
/// The serializable `{ shape, block }` configs, and [`MambaBlockConfig`].
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub mod config;

#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub use block::{
    MambaBidiLayers, MambaBlock, MambaCache, MambaCaches, MambaHNetVocabNet, MambaLatentNet,
    MambaVocabNet,
};
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub use config::{
    MambaBidiLayersConfig, MambaBlockConfig, MambaHNetVocabNetConfig, MambaLatentNetConfig,
    MambaVocabNetConfig,
};

#[cfg(all(test, feature = "_dev-test"))]
mod tests;

/// An explicit, family-tagged SSD-path selector.
///
/// Each variant carries the concrete per-family path, so callers can choose
/// the SSD path and the chunk length explicitly. The `*_default`
/// constructors give the default path of the family, without making it the
/// *only* option.
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
