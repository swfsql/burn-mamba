//! The model configuration for the grokking example — a small single-layer
//! Mamba-2 language model over the `p` residue tokens (see [`model_config`]).

use burn_mamba::prelude::{Mamba2Config, MambaVocabNetConfig, ResidualsConfig};

/// A deliberately constrained Mamba-2 LM (~29k params at the
/// `p = 97, d_model = 64, expand = 1, state_rank = 32, 1 layer` default):
///
/// - **Mamba-2, not Mamba-3**: a real input-gated exponential accumulator with
///   no oscillatory (RoPE) channel, so periodic structure must live in the
///   B/C/embedding geometry — the cleanest form of the state-rank hypothesis.
/// - **`conv_kernel = 1`**: no cross-token conv mixing; with 1 layer and the
///   two-token sequences, all `(a, b)` interaction is forced through the
///   recurrent state, whose rank `state_rank` caps pair-separability.
/// - **1 head** (`per_head_dim = d_inner`) and `state_rank ≤ d_model`, so the
///   participation-ratio ceiling is `state_rank` itself (not silently lowered
///   by the projection from `d_model`).
/// - **Sized to memory as much as to the task**: activation memory scales with
///   `d_inner·state_rank` (the full-batch state tensor is
///   `[p²/2, d_inner, state_rank]`). `state_rank = 32` is the smallest size
///   with PR headroom above the predicted generalizing rank (~10–12);
///   `d_model = 32` (expand 1) memorizes too slowly, 64 is the working floor.
/// - **No interleaved MLP** (no such module in the stack anyway): memorization
///   and generalization must share recurrence + gating + readout.
/// - **Untied LM head**: separate embedding/unembedding, so the embedding
///   Fourier-spectrum diagnostic is not coupled to readout dynamics.
pub fn model_config(
    p: usize,
    d_model: usize,
    expand: usize,
    state_rank: usize,
    n_layers: usize,
) -> MambaVocabNetConfig {
    let mamba_block = Mamba2Config::new(d_model)
        .with_state_rank(state_rank)
        .with_conv_kernel(1)
        .with_expand(expand)
        // one head:
        .with_per_head_dim(expand * d_model)
        .with_ngroups(1);

    MambaVocabNetConfig::Mamba2 {
        n_real_layers: n_layers,
        n_virtual_layers: None,
        grad_horizon: None,
        vocab_size: p,
        // keep logits exactly `p`-way (no padded classes in the softmax)
        pad_vocab_size_multiple: 1,
        mamba_block,
        // false ⇒ a dedicated (untied) LM head
        missing_lm_head: false,
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: ResidualsConfig::Standard,
        // No feed-forward interleave: these examples are mixer-only.
        mlp: None,
    }
}
