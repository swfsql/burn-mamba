//! The model configuration for the `tiny-stories` example — a small
//! character-level Mamba-3 language model (2 real layers, cycled to a 4-deep
//! virtual stack); see [`model_config`].

use crate::dataset::VOCAB_SIZE;
use burn_mamba::prelude::{Mamba3Config, MambaVocabNetConfig, ResidualsConfig, RotationKind};
use burn_mamba::utils::Schedule;

/// Depth of the (virtual) layer stack: the 2 real weight sets applied four times.
///
/// Virtual depth is **free in parameters**, so it is the cheapest capacity this
/// budget can buy — but only once the optimizer is good enough to use it. At the
/// original `lr = 2e-3` an 8-deep stack was *worse* than a 4-deep one (1.758 vs
/// 1.749 bits/char); at the tuned optimizer it wins clearly (1.475 vs 1.502).
/// Depth has an interior optimum: 4 → 8 → 12 virtual layers score
/// 1.502 → **1.475** → 1.504, so 8 is a peak and not just the edge of a search.
///
/// Deeper recursion under a truncated [`GRAD_HORIZON`] is a different story and
/// still a bad deal — see the README's table: 3.21 bits/char at 16 virtual
/// layers tracking only the top 4. Truncated BPTT does not suit a language
/// model: it is scored at *every* position, so leaving most applications of a
/// *shared* weight undifferentiated biases every one of those readouts — unlike
/// a task that reads out once, at the end of the sequence.
const N_VIRTUAL_LAYERS: usize = 8;

/// Back-propagate only the top `K` of the [`N_VIRTUAL_LAYERS`], everything below
/// running on the inner backend; `None` tracks the whole stack — which is what a
/// 4-deep stack over a 256-character window can afford.
const GRAD_HORIZON: Option<usize> = None;

/// The character-level LM: 39,632 parameters (~161KB on disk in FP32), of which
/// the tied embedding is only `VOCAB_SIZE · d_model` = 1536 — nearly everything
/// is the two Mamba-3 blocks.
///
/// With `seq_len = 256` and `batch_size = 8` (the [`crate::launch`] defaults)
/// training needs ~1.2GB of vram, and 16 epochs over the default corpus reach
/// **1.386 bits/char (70.2% next-character accuracy)**.
///
/// Everything that made that number came either from the optimizer or from
/// structure that costs no parameters — in decreasing order: batch size, learning
/// rate, virtual depth, [`ResidualsConfig::MultiGate`], epochs, Muon. Not one
/// reallocation of the parameter budget ever paid: at a *tuned* optimizer, adding
/// a SwiGLU MLP, trading `expand` for real layers, or switching to a non-abelian
/// rotation all scored at or below this block's own settings.
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
        // rope_fraction = 1.0 (apply RoPE to 100% of the B/C projections)
        .with_rope_fraction(1.0)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        // The abelian rotation: text needs no state-tracking group beyond a
        // (data-dependent) phase — `quaternion`/`rotor` are the reset-* examples.
        .with_rotation(RotationKind::Complex2D);

    MambaVocabNetConfig::Mamba3 {
        // two real layers, virtually cycled once more (2×2) for depth at no
        // parameter cost
        n_real_layers: 2,
        n_virtual_layers: Some((N_VIRTUAL_LAYERS, Schedule::Cyclic)),
        grad_horizon: GRAD_HORIZON,
        // the 48 case-folded characters the corpus actually contains
        vocab_size: VOCAB_SIZE,
        // keep the softmax exactly `VOCAB_SIZE`-way: no padded class can ever be
        // sampled, so every logit is a character the decoder understands
        pad_vocab_size_multiple: 1,
        mamba_block,
        // true ⇒ the LM head is the (transposed) embedding: one table for
        // "which character is this" and "which character comes next", which at
        // `d_model = 32` is also a third of the parameters saved
        missing_lm_head: true,
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        // Multi-Gate Residuals: `n_stream` pooled streams between layers instead
        // of one additive skip. It costs +136 parameters (the per-stream gates)
        // and is the second of the two free wins here, stacking with the deeper
        // [`N_VIRTUAL_LAYERS`]: 1.516 baseline → 1.502 (MGR alone) → 1.498
        // (depth alone) → 1.475 (both).
        //
        // `n_stream = 4` is a peak, not a ceiling — 8 streams price at only
        // 39,640 parameters and still score worse (1.484). `per_virtual_layer:
        // false` keeps one MGR per *real* layer, reused across the virtual
        // passes, which is both cheaper and better than one per virtual layer.
        residuals: ResidualsConfig::MultiGate {
            n_stream: 4,
            // Start every stream on an equal, unbiased gate and let training
            // break the symmetry; the accumulation phase (the first
            // `n_stream − 1` layers append rather than mix) is what makes the
            // streams distinct, so no carry bias is needed to get there.
            init_bias: 0.0,
            init_bias_step: 0.0,
            per_virtual_layer: false,
        },
        // No feed-forward interleave: these examples are mixer-only.
        mlp: None,
    }
}
