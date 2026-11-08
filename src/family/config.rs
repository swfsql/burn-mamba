//! The serializable configs of the run-time family choice.
//!
//! Each config is the pair `{ shape, block }` (burn-stack ADR-0007). The shape
//! is the half that has no block (see [`burn_stack::modules::shape`]). The block
//! is a [`MambaBlockConfig`], and its variant selects the family. A new knob of
//! a stack goes in the shape of `burn-stack`, and no config here changes.
//!
//! `#[derive(Config)]` takes no generics. So each container has one concrete
//! struct here.

use crate::prelude::*;
use burn::config::Config;
use burn::prelude::*;
use burn_stack::modules::hnet::HNetVocabShape;
use burn_stack::modules::{BlockConfig, ModelConfigExt};

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
    /// The default SSD path of the family of this block config.
    pub fn default_ssd_path(&self) -> MambaSsdPath {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(_) => MambaSsdPath::Mamba1,
            #[cfg(feature = "mamba2")]
            Self::Mamba2(_) => MambaSsdPath::mamba2_default(),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(_) => MambaSsdPath::mamba3_default(),
        }
    }
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

    #[cfg(feature = "optim")]
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

/// The serializable config of a [`MambaLatentNet`].
#[derive(Config, Debug)]
pub struct MambaLatentNetConfig {
    /// Everything about the network except its block.
    pub shape: LatentShape,
    /// The block, tagged with its family.
    pub block: MambaBlockConfig,
}

impl MambaLatentNetConfig {
    /// Allocate and initialise the network on `device`, in the family of the
    /// block.
    pub fn init(&self, device: &Device) -> MambaLatentNet {
        self.shape.init(self.block.clone(), device)
    }

    /// The [`MuonPlan`] of this network: the fused projections of the block and
    /// of the optional MLP. See [`NetworkShape::muon_plan`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> MuonPlan {
        self.shape.stack.muon_plan(&self.block)
    }
}

/// The serializable config of a [`MambaVocabNet`].
#[derive(Config, Debug)]
pub struct MambaVocabNetConfig {
    /// Everything about the network except its block.
    pub shape: VocabShape,
    /// The block, tagged with its family.
    pub block: MambaBlockConfig,
}

impl MambaVocabNetConfig {
    /// Allocate and initialise the language model on `device`, in the family of
    /// the block.
    pub fn init(&self, device: &Device) -> MambaVocabNet {
        self.shape.init(self.block.clone(), device)
    }

    /// The [`MuonPlan`] of this network: the fused projections of the block and
    /// of the optional MLP. See [`NetworkShape::muon_plan`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> MuonPlan {
        self.shape.stack.muon_plan(&self.block)
    }
}

/// The serializable config of a [`MambaBidiLayers`].
#[derive(Config, Debug)]
pub struct MambaBidiLayersConfig {
    /// Everything about the stack except its block.
    pub shape: BidiShape,
    /// The block, tagged with its family.
    pub block: MambaBlockConfig,
}

impl MambaBidiLayersConfig {
    /// Allocate and initialise the bidirectional stack on `device`, in the
    /// family of the block.
    pub fn init(&self, device: &Device) -> MambaBidiLayers {
        self.shape.init(self.block.clone(), device)
    }

    /// The [`MuonPlan`] of this stack: the fused projections of the block, and
    /// the spec of the `CatLinear` merges. See [`BidiShape::muon_plan`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> MuonPlan {
        self.shape.muon_plan(&self.block)
    }
}

/// The serializable config of a [`MambaHNetVocabNet`].
#[derive(Config, Debug)]
pub struct MambaHNetVocabNetConfig {
    /// The stacks, the vocabulary and the knobs of the chunking.
    pub shape: HNetVocabShape,
    /// One block config per stage, outermost first, all of one family. The
    /// block config of a stage builds both its encoder and its decoder, and
    /// sets the width of the stage.
    pub stage_blocks: Vec<MambaBlockConfig>,
    /// The block config of the main stack.
    pub main_block: MambaBlockConfig,
}

impl MambaHNetVocabNetConfig {
    /// Allocate and initialise the network on `device`.
    pub fn init(&self, device: &Device) -> MambaHNetVocabNet {
        self.shape.init(&self.stage_blocks, &self.main_block, device)
    }

    /// The [`MuonPlan`] of the network: each stack at its own block config,
    /// scoped to that stack. See `HNetShape::muon_plan` of `burn-stack`.
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> MuonPlan {
        self.shape.muon_plan(&self.stage_blocks, &self.main_block)
    }
}

// In method-call syntax, an inherent method has priority over a trait method.
// So `self.init(..)` calls the inherent method above, and these impls do not
// recurse.

impl ModelConfigExt for MambaLatentNetConfig {
    type Model = MambaLatentNet;
    fn init(&self, device: &Device) -> Self::Model {
        self.init(device)
    }
    #[cfg(feature = "optim")]
    fn muon_plan(&self) -> MuonPlan {
        self.muon_plan()
    }
}

impl ModelConfigExt for MambaVocabNetConfig {
    type Model = MambaVocabNet;
    fn init(&self, device: &Device) -> Self::Model {
        self.init(device)
    }
    #[cfg(feature = "optim")]
    fn muon_plan(&self) -> MuonPlan {
        self.muon_plan()
    }
}

impl ModelConfigExt for MambaBidiLayersConfig {
    type Model = MambaBidiLayers;
    fn init(&self, device: &Device) -> Self::Model {
        self.init(device)
    }
    #[cfg(feature = "optim")]
    fn muon_plan(&self) -> MuonPlan {
        self.muon_plan()
    }
}

impl ModelConfigExt for MambaHNetVocabNetConfig {
    type Model = MambaHNetVocabNet;
    fn init(&self, device: &Device) -> Self::Model {
        self.init(device)
    }
    #[cfg(feature = "optim")]
    fn muon_plan(&self) -> MuonPlan {
        self.muon_plan()
    }
}
