//! The model configuration for the `tiny-stories` example: a small
//! character-level Mamba-3 language model (real layers cycled to a virtual
//! stack). See [`model_config`].
//!
//! This example is a work in progress. The sizes and hyperparameters here are
//! placeholders until a parameter search.

use crate::dataset::VOCAB_SIZE;
use burn_mamba::prelude::{Mamba3Config, MambaVocabNetConfig, ResidualsConfig, RotationKind};
use burn_stack::utils::{ClassLatent, GradHorizon, Schedule};

/// Depth of the (virtual) layer stack: the number of applications of the real
/// weight sets. Virtual depth costs no parameters.
const N_VIRTUAL_LAYERS: usize = 8;

/// Back-propagate only the top `Depth(K)` applications of each real layer. The
/// applications below run on the inner backend. `None` tracks the whole stack.
///
/// A language model is scored at *every* position. So if most applications of a
/// *shared* weight get no gradient, every one of those readouts is biased. A task
/// that reads out once, at the end of the sequence, does not have this problem.
const GRAD_HORIZON: Option<GradHorizon> = None;

/// Learnable `[CLS]`-style registers (width `d_model`) spliced in front of every
/// story. They are the "a story starts here" of the model, in place of a
/// separator character.
///
/// They make seedless generation honest. During training, the last of them is
/// scored against the **first** character of the story. So `prime` replays them
/// and returns the distribution of that character, with no input. A separator
/// character as a seed would be out of distribution: in a joined stream it
/// occurs only *between* two stories, so always on a state that carries the
/// previous story.
const N_CLASS_LATENTS: usize = 4;

/// The character-level LM. The tied embedding is only `VOCAB_SIZE · d_model`
/// parameters. Nearly all the others are in the Mamba-3 blocks.
pub fn model_config() -> MambaVocabNetConfig {
    // d_model = 32 (intra/inter-layer expressivity, high impact on disk size)
    let d_model = 32;
    let mamba_block = Mamba3Config::new(d_model)
        // state_rank = 64 (time-wise expressivity, average impact on disk size)
        .with_state_rank(64)
        .with_expand(4)
        // d_inner = expand·d_model = 4·32 = 128
        // per_head_dim = 32
        // nheads = d_inner/per_head_dim = 128/32 = 4
        .with_per_head_dim(32)
        .with_ngroups(1)
        .with_mimo_rank(1)
        // rope_fraction = 1.0 (rotate 100% of the B/C projections)
        .with_rope_fraction(1.0)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        // The abelian rotation: text needs no state-tracking group other than a
        // (data-dependent) phase. The reset-* examples use `quaternion`/`rotor`.
        .with_rotation(RotationKind::Complex2D);

    MambaVocabNetConfig::Mamba3 {
        // the real layers, cycled to `N_VIRTUAL_LAYERS` for depth at no
        // parameter cost
        n_real_layers: 2,
        n_virtual_layers: Some((N_VIRTUAL_LAYERS, Schedule::Cyclic)),
        grad_horizon: GRAD_HORIZON,
        // the 48 case-folded characters that the corpus contains
        vocab_size: VOCAB_SIZE,
        // keep the softmax exactly `VOCAB_SIZE`-way: no padded class can be
        // sampled, so every logit is a character that the decoder knows
        pad_vocab_size_multiple: 1,
        mamba_block,
        // true ⇒ the LM head is the (transposed) embedding: one table for
        // "which character is this" and "which character comes next"
        missing_lm_head: true,
        class_latents: vec![ClassLatent::Start; N_CLASS_LATENTS],
        ignore_first_residual: false,
        ignore_last_residual: false,
        // Multi-Gate Residuals: `n_stream` pooled streams between layers instead
        // of one additive skip. The per-stream gates are the only added
        // parameters. `per_virtual_layer: false` keeps one MGR per *real*
        // layer, used again by each virtual pass.
        residuals: ResidualsConfig::MultiGate {
            n_stream: 4,
            // Start every stream on an equal, unbiased gate, and let training
            // break the symmetry. The accumulation phase (the first
            // `n_stream − 1` layers append and do not mix) makes the streams
            // distinct, so they need no carry bias.
            init_bias: 0.0,
            init_bias_step: 0.0,
            per_virtual_layer: false,
        },
        // No feed-forward interleave: these examples are mixer-only.
        mlp: None,
        untied: Vec::new(),
    }
}
