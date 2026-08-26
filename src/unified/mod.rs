//! # The unified, runtime-selectable API
//!
//! [`burn_stack`] composes any [`Block`](burn_stack::modules::Block) into
//! layers and networks, but `M` is fixed at the type level. To let a caller
//! choose the family at *runtime* — and to serialize that choice — this module
//! wraps the three monomorphisations in enums: [`MambaLatentNet`] /
//! [`MambaVocabNet`] / [`MambaBidiLayers`] / [`MambaCaches`] /
//! [`MambaSsdPath`], each with a `#[derive(Config)]` companion where one
//! applies. They panic on a family-mismatched cache or SSD path.
//!
//! This is also where the three families meet the generic stack: the
//! `impl Block for Mamba{1,2,3}`, `impl BlockConfig for Mamba{1,2,3}Config` and
//! `impl CacheStack for Mamba{1,2,3}Caches` blocks live in [`cache`].
//!
//! ## Muon: why the 3-D tensors are not "stacked matrices"
//!
//! [`burn_stack::optim`] only lets Muon own weights a
//! [`BlockConfig::muon_projections`](burn_stack::modules::BlockConfig::muon_projections)
//! names, and each family's list deliberately stops at rank 2. For Mamba-3's
//! MIMO tensors that deserves an argument, since their *shape* invites the
//! opposite conclusion:
//!
//! MIMO's *math* is R independent SSMs summed together, so a `[nheads,
//! mimo_rank, per_head_dim]` tensor looks like a stack of matrices that a
//! stack-aware Muon could take a slice at a time. It is not: the paper
//! deliberately avoids instantiating the R maps, because that would multiply
//! every per-head projection's parameter count by R (`DP → DPR` *per head*).
//! Instead it keeps the SISO projection and element-wise scales its output to
//! size R with a learnable vector (`DP + PR` per head) — in the appendix's
//! contraction notation `X = contract(PR, P → PR)(W_X, X')`, where `P` appears
//! in both inputs *and* the output and so is never contracted. `mimo_x`/`mimo_z`
//! /`mimo_o` are therefore **diagonals**, and their matrix shape is a layout
//! coincidence; orthogonalising one would constrain a set of gains. The part of
//! MIMO that really is an R-fold matrix expansion is B/C (`DN → DNR`), which
//! lives in `in_proj` and is already Muon's. The remaining 3-D tensors are a
//! bias (`b_bias_hmr`/`c_bias_hmr`), an initial condition (`init_state_hpr`), or
//! a depthwise filter (the Mamba-1/2 conv) — diagonal or embedding-like again.
//!

/// A runtime-tagged cache collection + the per-family `Block` / `BlockConfig` /
/// `CacheStack` impls.
pub mod cache;
/// Runtime-selectable networks ([`MambaLatentNet`] / [`MambaVocabNet`]).
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub mod network;
/// Runtime-selectable bidirectional stacks ([`MambaBidiLayers`]).
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub mod bidi;

pub use cache::MambaCaches;
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub use bidi::{MambaBidiLayers, MambaBidiLayersConfig};
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
pub use network::{MambaLatentNet, MambaLatentNetConfig, MambaVocabNet, MambaVocabNetConfig};

#[cfg(all(test, feature = "_dev-test"))]
mod tests;

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
