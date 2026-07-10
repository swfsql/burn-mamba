//! The model configuration for the grokking example — a small single-layer
//! Mamba-2 (default) or Mamba-3 (`--mamba3`) language model over the `p`
//! residue tokens (see [`model_config`]).

use burn_mamba::prelude::{
    Mamba2Config, Mamba3Config, MambaVocabNetConfig, ResidualsConfig, RotationKind,
};

/// The Mamba-3 arm's rotation knobs (`--mamba3 [--quat] [--rope-fraction f]`).
#[derive(Debug, Clone, Copy)]
pub struct Mamba3Arm {
    /// `Quaternion4D` instead of the default `Complex2D` rotation.
    pub quaternion: bool,
    /// Fraction of `state_rank` the data-dependent rotation acts on
    /// (0.0 | 0.5 | 1.0). `1.0` is the cleanest probe for "does the circuit
    /// move into the rotation angles" — every state coordinate is a complex
    /// plane; `0.5` is the reference default.
    pub rope_fraction: f64,
}

/// A deliberately constrained Mamba LM (~29k params for the Mamba-2
/// `p = 97, d_model = 64, expand = 1, state_rank = 32, 1 layer` default):
///
/// - **Mamba-2 by default**: a real input-gated exponential accumulator with
///   no oscillatory (RoPE) channel, so periodic structure must live in the
///   B/C/embedding geometry — the cleanest form of the state-rank hypothesis.
///   The **`--mamba3` arm** adds exactly that channel (data-dependent
///   rotation ⇒ a genuinely complex state): mod-p addition *is* rotation
///   composition, so the same task probes whether the circuit migrates into
///   the angles (measured by the Hermitian `PR_ℂ(M_phys)` — see the README's
///   Mamba-3 read-out section).
/// - **No cross-token conv mixing** (`conv_kernel = 1` for Mamba-2; Mamba-3
///   has no conv at all): with 1 layer, all summand interaction is forced
///   through the recurrent state, whose rank `state_rank` caps
///   pair-separability.
/// - **1 head** (`per_head_dim = d_inner`) and `state_rank ≤ d_model`, so the
///   participation-ratio ceiling is `state_rank` itself (not silently lowered
///   by the projection from `d_model`). SISO (`mimo_rank = 1`) on the Mamba-3
///   arm, so the rotation pairing is the interleaved/NeoX one.
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
    mamba3: Option<Mamba3Arm>,
) -> MambaVocabNetConfig {
    match mamba3 {
        None => {
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
                vocab_size: p,
                // keep logits exactly `p`-way (no padded classes in the softmax)
                pad_vocab_size_multiple: 1,
                mamba_block,
                // false ⇒ a dedicated (untied) LM head
                missing_lm_head: false,
                ignore_first_residual: false,
                ignore_last_residual: false,
                residuals: ResidualsConfig::Standard,
            }
        }
        Some(arm) => {
            let mamba_block = Mamba3Config::new(d_model)
                .with_state_rank(state_rank)
                .with_expand(expand)
                // one head, SISO:
                .with_per_head_dim(expand * d_model)
                .with_ngroups(1)
                .with_rope_fraction(arm.rope_fraction)
                .with_rotation(if arm.quaternion {
                    RotationKind::Quaternion4D
                } else {
                    RotationKind::Complex2D
                });

            MambaVocabNetConfig::Mamba3 {
                n_real_layers: n_layers,
                n_virtual_layers: None,
                vocab_size: p,
                pad_vocab_size_multiple: 1,
                mamba_block,
                missing_lm_head: false,
                ignore_first_residual: false,
                ignore_last_residual: false,
                residuals: ResidualsConfig::Standard,
            }
        }
    }
}
