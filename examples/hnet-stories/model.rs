//! The model config of the `hnet-stories` example: a small character-level
//! H-Net of Mamba-3 stacks, and the recipe that trains it. See
//! [`model_config`].
//!
//! This example is a work in progress. The sizes and hyperparameters here are
//! placeholders until a parameter search.

use crate::cli::ModelFlags;
use crate::dataset::VOCAB_SIZE;
use burn::prelude::*;
use burn_mamba::prelude::{
    Mamba3Config, MambaBlockConfig, MambaHNetVocabNet, MambaHNetVocabNetConfig, RotationKind,
};
use burn_stack::modules::hnet::{HNetShape, HNetStageShape, HNetVocabShape};
use burn_stack::modules::{GatedMlpConfig, ModelConfigExt, NetworkShape};
use burn_stack::utils::InitPolicy;

/// The width of stage 0 (the character resolution) and of stage 1.
const STAGE_WIDTHS: [usize; 2] = [32, 48];

/// The width of the main network.
const MAIN_WIDTH: usize = 64;

/// The state rank of the blocks of every stage.
const STAGE_STATE_RANK: usize = 32;

/// The state rank of the blocks of the main network.
const MAIN_STATE_RANK: usize = 64;

/// The target mean chunk length of each stage, in rows of its own input. At
/// stage 0 that is characters: about a short word.
const RATIO_TARGETS: [f64; 2] = [4.0, 2.0];

/// The learnable rows that open every story (see `HNetVocabNetwork`). The
/// output of the last one is scored against the first character.
const N_START_TOKENS: usize = 4;

/// `α`, ten times the reference value. Early in training, the encoders of a
/// model this small make adjacent rows similar, and the router starts almost
/// no chunk. The reference weight is then too weak to bring the chunks back.
const RATIO_WEIGHT: f64 = 0.3;

/// The H-Net and its training recipe: what a run saves as its model config.
#[derive(Config, Debug)]
pub struct HNetStoriesConfig {
    /// The network.
    pub model: MambaHNetVocabNetConfig,
    /// `Nˢ`: the target mean chunk length of each stage (see `ratio_loss`).
    pub ratio_targets: Vec<f64>,
    /// `α`: the weight of the ratio loss of each stage, beside the language
    /// model loss (the reference value).
    #[config(default = 0.03)]
    pub ratio_weight: f64,
    /// `λˢ`: the learning-rate multiplier of each stage, then of the main
    /// network. The embedding and the heads take that of stage 0.
    pub lr_multipliers: Vec<f64>,
}

impl ModelConfigExt for HNetStoriesConfig {
    type Model = MambaHNetVocabNet;
    fn init(&self, device: &Device) -> Self::Model {
        self.model.init(device)
    }
    fn muon_plan(&self) -> burn_stack::optim::MuonPlan {
        self.model.muon_plan()
    }
}

/// The learning-rate multiplier of each stage and of the main network: the
/// heuristic of the paper, normalised so that the main network has `1`.
///
/// ```text
///   λˢ = √( Dᴹ/Dˢ · Πᵢ₌ₛ^{S−1} Nⁱ )
/// ```
///
/// A stage sees `Πᵢ Nⁱ` times more rows per step than the main network: a
/// larger effective batch. It is also narrower, which allows a larger rate
/// per unit of width.
pub fn lr_multipliers(widths: &[usize], main_width: usize, targets: &[f64]) -> Vec<f64> {
    let mut multipliers: Vec<f64> = (0..widths.len())
        .map(|s| (main_width as f64 / widths[s] as f64 * targets[s..].iter().product::<f64>()).sqrt())
        .collect();
    multipliers.push(1.0);
    multipliers
}

/// A Mamba-3 block config at width `d` with a state of rank `state_rank`.
fn mamba3(d: usize, state_rank: usize) -> MambaBlockConfig {
    MambaBlockConfig::Mamba3(
        Mamba3Config::new(d)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_state_rank(state_rank)
            .with_ngroups(1)
            .with_mimo_rank(1)
            .with_rope_fraction(1.0)
            .with_has_proj_bias(true)
            .with_has_outproj_norm(true)
            .with_rotation(RotationKind::Complex2D),
    )
}

/// The character-level H-Net, with `flags.stages` stages.
///
/// The layout follows the reference at a tiny size, with Mamba-3 in every
/// stack. The reference has Mamba-2 in the stages and attention in the main
/// network. The stacks at character resolution are short and mixer-only. The
/// main network is deeper and has a SwiGLU MLP. Every stack takes the
/// reference init (`N(0, 0.02)`, with the residual rescale over the stacks
/// down to it).
pub fn model_config(flags: &ModelFlags) -> HNetStoriesConfig {
    let init = || InitPolicy::new().with_residual_paths(InitPolicy::default_residual_paths());
    let outer = || NetworkShape::new(2).with_init(Some(init()));
    let stages: Vec<HNetStageShape> = (0..flags.stages).map(|_| HNetStageShape::new(outer(), outer())).collect();
    let main = NetworkShape::new(4)
        .with_mlp(Some(GatedMlpConfig::from_hidden_ratio(MAIN_WIDTH, 4)))
        .with_init(Some(init()));
    let shape = HNetVocabShape::new(HNetShape::new(stages, main), VOCAB_SIZE)
        // keep the softmax exactly `VOCAB_SIZE`-way (see `tiny-stories`)
        .with_pad_vocab_size_multiple(1)
        // one table for "which character is this" and "which comes next"
        .with_missing_lm_head(true)
        .with_head_init(Some(InitPolicy::new()))
        .with_n_start_tokens(N_START_TOKENS);
    let widths = &STAGE_WIDTHS[..flags.stages];
    let targets = RATIO_TARGETS[..flags.stages].to_vec();
    HNetStoriesConfig::new(
        MambaHNetVocabNetConfig::new(
            shape,
            widths.iter().map(|&d| mamba3(d, STAGE_STATE_RANK)).collect(),
            mamba3(MAIN_WIDTH, MAIN_STATE_RANK),
        ),
        targets.clone(),
        lr_multipliers(widths, MAIN_WIDTH, &targets),
    )
    .with_ratio_weight(RATIO_WEIGHT)
}
