//! Runtime-selectable bidirectional stacks: one enum (plus a serializable
//! `Config`) over the three families'
//! [`BidiLayers`](burn_stack::modules::BidiLayers) monomorphisations.

use crate::prelude::*;
use burn::config::Config;
use burn::prelude::*;
use burn_stack::modules::bidi::{BidiLayers, BidiLayersBuilder, OutputMergeConfig};
use burn_stack::modules::ResidualsConfig;
use burn_stack::utils::{BidiSchedule, ClassCursors, ClassLatent};

// ===========================================================================
// Unifying enums: one runtime + one serializable Config across all families
// ===========================================================================

/// A runtime-selectable bidirectional stack: the same paired straight/reverse
/// structure over any Mamba-x family, chosen at runtime. The forward-only
/// counterpart of [`MambaLatentNet`] for non-autoregressive tasks.
#[derive(Module, Debug)]
pub enum MambaBidiLayers {
    /// Mamba-1 bidirectional stack.
    #[cfg(feature = "mamba1")]
    Mamba1(BidiLayers<crate::mamba1::prelude::Mamba1>),
    /// Mamba-2 bidirectional stack.
    #[cfg(feature = "mamba2")]
    Mamba2(BidiLayers<crate::mamba2::prelude::Mamba2>),
    /// Mamba-3 bidirectional stack.
    #[cfg(feature = "mamba3")]
    Mamba3(BidiLayers<crate::mamba3::prelude::Mamba3>),
}

impl MambaBidiLayers {
    /// Output positions of the stack-level class latents for an `orig_len`
    /// input (so a caller can read a class latent back out of the lengthened
    /// `forward` output — e.g. as a pooled summary). A marker that never lands
    /// (a `Custom` at or past the end) reports a position past that output.
    pub fn class_latent_output_indices(&self, orig_len: usize) -> Vec<usize> {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(layers) => layers.class_latent_output_indices(orig_len),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(layers) => layers.class_latent_output_indices(orig_len),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(layers) => layers.class_latent_output_indices(orig_len),
        }
    }

    /// Full-sequence bidirectional pass. The `ssd_path` must match the stack's
    /// family; a mismatch is a caller error and panics.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<MambaCaches>,
        ssd_path: MambaSsdPath,
        class: Option<&mut ClassCursors>,
    ) -> (Tensor<3>, MambaCaches) {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(layers) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba1(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-1 bidi stack"),
                });
                match ssd_path {
                    MambaSsdPath::Mamba1 => {}
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-1 bidi stack"),
                }
                let (y, c) = layers.forward(x, caches, (), class);
                (y, MambaCaches::Mamba1(c))
            }
            #[cfg(feature = "mamba2")]
            Self::Mamba2(layers) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba2(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-2 bidi stack"),
                });
                let path = match ssd_path {
                    MambaSsdPath::Mamba2(p) => p,
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-2 bidi stack"),
                };
                let (y, c) = layers.forward(x, caches, path, class);
                (y, MambaCaches::Mamba2(c))
            }
            #[cfg(feature = "mamba3")]
            Self::Mamba3(layers) => {
                let caches = caches.map(|c| match c {
                    MambaCaches::Mamba3(c) => c,
                    #[allow(unreachable_patterns)]
                    _ => panic!("cache family does not match Mamba-3 bidi stack"),
                });
                let path = match ssd_path {
                    MambaSsdPath::Mamba3(p) => p,
                    #[allow(unreachable_patterns)]
                    _ => panic!("ssd_path family does not match Mamba-3 bidi stack"),
                };
                let (y, c) = layers.forward(x, caches, path, class);
                (y, MambaCaches::Mamba3(c))
            }
        }
    }
}

/// The serializable config for [`MambaBidiLayers`]. Each variant is concrete
/// (per-family), so `#[derive(Config)]` applies; `init` builds the matching
/// stack variant.
#[derive(Config, Debug)]
pub enum MambaBidiLayersConfig {
    /// Build a Mamba-1 bidirectional stack.
    #[cfg(feature = "mamba1")]
    Mamba1 {
        /// Number of real layers (must be even — used in pairs).
        n_real_layers: usize,
        /// Optional virtual-layer scheduling (pairs; must be even).
        n_virtual_layers: Option<(usize, BidiSchedule)>,
        /// Shared block config.
        mamba_block: crate::mamba1::prelude::Mamba1Config,
        /// Suppress the first virtual pair's residual.
        ignore_first_residual: bool,
        /// Suppress the last virtual pair's residual (the stack outputs that
        /// pair's merged transform alone).
        ignore_last_residual: bool,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
        /// Inter-pair residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
    },
    /// Build a Mamba-2 bidirectional stack.
    #[cfg(feature = "mamba2")]
    Mamba2 {
        /// Number of real layers (must be even — used in pairs).
        n_real_layers: usize,
        /// Optional virtual-layer scheduling (pairs; must be even).
        n_virtual_layers: Option<(usize, BidiSchedule)>,
        /// Shared block config.
        mamba_block: crate::mamba2::prelude::Mamba2Config,
        /// Suppress the first virtual pair's residual.
        ignore_first_residual: bool,
        /// Suppress the last virtual pair's residual (the stack outputs that
        /// pair's merged transform alone).
        ignore_last_residual: bool,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
        /// Inter-pair residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
    },
    /// Build a Mamba-3 bidirectional stack.
    #[cfg(feature = "mamba3")]
    Mamba3 {
        /// Number of real layers (must be even — used in pairs).
        n_real_layers: usize,
        /// Optional virtual-layer scheduling (pairs; must be even).
        n_virtual_layers: Option<(usize, BidiSchedule)>,
        /// Shared block config.
        mamba_block: crate::mamba3::prelude::Mamba3Config,
        /// Suppress the first virtual pair's residual.
        ignore_first_residual: bool,
        /// Suppress the last virtual pair's residual (the stack outputs that
        /// pair's merged transform alone).
        ignore_last_residual: bool,
        /// One merge config per pair, length `n_real_layers / 2`.
        outputs_merge: Vec<OutputMergeConfig>,
        /// Stack-level class latents, spliced into the sequence before the
        /// first pair (e.g. a `Middle` summary latent in place of mean-pooling).
        class_latents: Vec<ClassLatent>,
        /// Inter-pair residual scheme (plain additive vs Multi-Gate).
        residuals: ResidualsConfig,
    },
}

impl MambaBidiLayersConfig {
    /// The [`MuonPlan`] for this stack: the block's fused
    /// projections plus each pair's `CatLinear` merge (a plain hidden matrix;
    /// `Mean` merges have no parameters). See [`burn_stack::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        use burn_stack::modules::BlockConfig;
        let (mut specs, d_model) = match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 { mamba_block, .. } => (
                mamba_block.muon_projections(),
                BlockConfig::d_model(mamba_block),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 { mamba_block, .. } => (
                mamba_block.muon_projections(),
                BlockConfig::d_model(mamba_block),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 { mamba_block, .. } => (
                mamba_block.muon_projections(),
                BlockConfig::d_model(mamba_block),
            ),

        };
        specs.push(burn_stack::optim::ProjSpec::path_whole("CatLinear.weight", d_model));
        burn_stack::optim::MuonPlan::new(specs)
    }

    /// Allocate and initialise the selected bidirectional stack on `device`.
    pub fn init(&self, device: &Device) -> MambaBidiLayers {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1 {
                n_real_layers,
                n_virtual_layers,
                mamba_block,
                ignore_first_residual,
                ignore_last_residual,
                outputs_merge,
                class_latents,
                residuals,
            } => MambaBidiLayers::Mamba1(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: n_virtual_layers.clone(),
                    block: mamba_block.clone(),
                    ignore_first_residual: *ignore_first_residual,
                    ignore_last_residual: *ignore_last_residual,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                    residuals: residuals.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba2")]
            Self::Mamba2 {
                n_real_layers,
                n_virtual_layers,
                mamba_block,
                ignore_first_residual,
                ignore_last_residual,
                outputs_merge,
                class_latents,
                residuals,
            } => MambaBidiLayers::Mamba2(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: n_virtual_layers.clone(),
                    block: mamba_block.clone(),
                    ignore_first_residual: *ignore_first_residual,
                    ignore_last_residual: *ignore_last_residual,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                    residuals: residuals.clone(),
                }
                .init(device),
            ),
            #[cfg(feature = "mamba3")]
            Self::Mamba3 {
                n_real_layers,
                n_virtual_layers,
                mamba_block,
                ignore_first_residual,
                ignore_last_residual,
                outputs_merge,
                class_latents,
                residuals,
            } => MambaBidiLayers::Mamba3(
                BidiLayersBuilder {
                    n_real_layers: *n_real_layers,
                    n_virtual_layers: n_virtual_layers.clone(),
                    block: mamba_block.clone(),
                    ignore_first_residual: *ignore_first_residual,
                    ignore_last_residual: *ignore_last_residual,
                    outputs_merge: outputs_merge.clone(),
                    class_latents: class_latents.clone(),
                    residuals: residuals.clone(),
                }
                .init(device),
            ),
        }
    }
}
