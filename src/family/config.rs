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
use burn_stack::modules::ModelConfigExt;

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
        match &self.block {
            #[cfg(feature = "mamba1")]
            MambaBlockConfig::Mamba1(c) => MambaLatentNet::Mamba1(self.shape.init(c.clone(), device)),
            #[cfg(feature = "mamba2")]
            MambaBlockConfig::Mamba2(c) => MambaLatentNet::Mamba2(self.shape.init(c.clone(), device)),
            #[cfg(feature = "mamba3")]
            MambaBlockConfig::Mamba3(c) => MambaLatentNet::Mamba3(self.shape.init(c.clone(), device)),
        }
    }

    /// The [`MuonPlan`] of this network: the fused projections of the block and
    /// of the optional MLP. See [`NetworkShape::muon_plan`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> MuonPlan {
        match &self.block {
            #[cfg(feature = "mamba1")]
            MambaBlockConfig::Mamba1(c) => self.shape.stack.muon_plan(c),
            #[cfg(feature = "mamba2")]
            MambaBlockConfig::Mamba2(c) => self.shape.stack.muon_plan(c),
            #[cfg(feature = "mamba3")]
            MambaBlockConfig::Mamba3(c) => self.shape.stack.muon_plan(c),
        }
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
        match &self.block {
            #[cfg(feature = "mamba1")]
            MambaBlockConfig::Mamba1(c) => MambaVocabNet::Mamba1(self.shape.init(c.clone(), device)),
            #[cfg(feature = "mamba2")]
            MambaBlockConfig::Mamba2(c) => MambaVocabNet::Mamba2(self.shape.init(c.clone(), device)),
            #[cfg(feature = "mamba3")]
            MambaBlockConfig::Mamba3(c) => MambaVocabNet::Mamba3(self.shape.init(c.clone(), device)),
        }
    }

    /// The [`MuonPlan`] of this network: the fused projections of the block and
    /// of the optional MLP. See [`NetworkShape::muon_plan`].
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> MuonPlan {
        match &self.block {
            #[cfg(feature = "mamba1")]
            MambaBlockConfig::Mamba1(c) => self.shape.stack.muon_plan(c),
            #[cfg(feature = "mamba2")]
            MambaBlockConfig::Mamba2(c) => self.shape.stack.muon_plan(c),
            #[cfg(feature = "mamba3")]
            MambaBlockConfig::Mamba3(c) => self.shape.stack.muon_plan(c),
        }
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
        match &self.block {
            #[cfg(feature = "mamba1")]
            MambaBlockConfig::Mamba1(c) => MambaBidiLayers::Mamba1(self.shape.init(c.clone(), device)),
            #[cfg(feature = "mamba2")]
            MambaBlockConfig::Mamba2(c) => MambaBidiLayers::Mamba2(self.shape.init(c.clone(), device)),
            #[cfg(feature = "mamba3")]
            MambaBlockConfig::Mamba3(c) => MambaBidiLayers::Mamba3(self.shape.init(c.clone(), device)),
        }
    }

    /// The [`MuonPlan`] of this stack: the fused projections of the block, plus
    /// the `CatLinear` merge of each pair. A `CatLinear` merge is a plain hidden
    /// matrix, and a `Mean` merge has no parameters. [`BidiShape::muon_plan`]
    /// gives only the block specs, so this method adds the merge spec.
    #[cfg(feature = "optim")]
    pub fn muon_plan(&self) -> MuonPlan {
        let (plan, d_model) = match &self.block {
            #[cfg(feature = "mamba1")]
            MambaBlockConfig::Mamba1(c) => (self.shape.muon_plan(c), BlockConfig::d_model(c)),
            #[cfg(feature = "mamba2")]
            MambaBlockConfig::Mamba2(c) => (self.shape.muon_plan(c), BlockConfig::d_model(c)),
            #[cfg(feature = "mamba3")]
            MambaBlockConfig::Mamba3(c) => (self.shape.muon_plan(c), BlockConfig::d_model(c)),
        };
        plan.extend(MuonPlan::new(vec![ProjSpec::path_whole("CatLinear.weight", d_model)]))
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
