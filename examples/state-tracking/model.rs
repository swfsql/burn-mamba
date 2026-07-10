//! The model configuration for the state-tracking example — the same
//! deliberately constrained single-head Mamba-3 LM as the grokking example's
//! `--mamba3` arm, over the shared symbol/class vocabulary; the varying knobs
//! are the rotation kind and the rotated fraction (see [`model_config`]).

use crate::dataset::VOCAB_SIZE;
use burn_mamba::prelude::{
    Mamba3Config, MambaVocabNetConfig, ResidualsConfig, RotationKind,
};

/// The rotation knobs (`--quat`, `--rope-fraction f`) — the contrast this
/// example exists for.
#[derive(Debug, Clone, Copy)]
pub struct RotationArm {
    /// `Quaternion4D` (non-abelian `SU(2)`) instead of the default
    /// `Complex2D` (abelian `SO(2)`) rotation.
    pub quaternion: bool,
    /// Fraction of `state_rank` the data-dependent rotation acts on
    /// (0.0 | 0.5 | 1.0). `1.0` gives the rotation the whole state — the
    /// cleanest arena for "can the rotation compose the group".
    pub rope_fraction: f64,
}

/// A deliberately constrained Mamba-3 LM, mirroring the grokking model's
/// design constraints so its findings carry over:
///
/// - **1 head** (`per_head_dim = d_inner`), SISO (`mimo_rank = 1`),
///   `ngroups = 1`, `state_rank ≤ d_model` — the state-PR ceiling is
///   `state_rank` itself, and the rotation pairing is interleaved/NeoX.
/// - **No conv** (Mamba-3 has none) and **1 layer by default**: with the
///   final-position-only supervision, all composition is forced through the
///   recurrent state — the rotation is the only mechanism that can *compose*
///   the group inside a single layer.
/// - **Untied LM head**, and a vocabulary laid out symbols-first
///   (`dataset::CLASS_BASE`) so the weight diagnostics read the input
///   alphabet rows directly.
/// - `state_rank` must be a multiple of 4 for `Quaternion4D` (asserted by
///   the library); the default 32 satisfies it.
pub fn model_config(
    d_model: usize,
    expand: usize,
    state_rank: usize,
    n_layers: usize,
    arm: RotationArm,
) -> MambaVocabNetConfig {
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
        vocab_size: VOCAB_SIZE,
        // keep logits exactly VOCAB_SIZE-way (no padded classes in the softmax)
        pad_vocab_size_multiple: 1,
        mamba_block,
        // false ⇒ a dedicated (untied) LM head
        missing_lm_head: false,
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: ResidualsConfig::Standard,
    }
}
