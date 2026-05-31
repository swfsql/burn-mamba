//! The model configuration for the state-tracking example — a deliberately tiny
//! single-layer Mamba-3 classifier whose only varying knob is the rotation kind
//! (`Complex2D` vs `Quaternion4D`); see [`model_config`].

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind};

/// Build the example model config for the chosen `rotation`.
///
/// The defaults are deliberately **tiny** (fibonacci-scale): `d_model = 32`,
/// `expand = 2` (`d_inner = 64`), `nheads = 8` (`per_head_dim = 8`),
/// `state_rank = 16` (a multiple of 4, required by `Quaternion4D`), and a single
/// real layer. RoPE is applied to 100% of the B/C projections. The whole point
/// of the example is to contrast `RotationKind::Complex2D` (abelian, the
/// Mamba-3 default) against `RotationKind::Quaternion4D` (non-abelian) on the
/// otherwise identical model.
///
/// For a wider, cleaner capability gap, scale the model / sequence length /
/// epochs up and run on GPU (`--features backend-cuda`) — this is a
/// demonstration of the capability, not a tuned benchmark.
pub fn model_config(rotation: RotationKind) -> MambaLatentNetConfig {
    // d_inner = expand * d_model = 64, so per_head_dim = d_inner / nheads = 8.
    let mamba_block = Mamba3Config::new(32)
        .with_state_rank(16) // multiple of 4, required by Quaternion4D
        .with_expand(2)
        .with_per_head_dim(8) // nheads = d_inner / per_head_dim = 8
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(1.0) // apply RoPE to 100% of the B/C projections
        .with_has_proj_bias(true)
        // the one knob this example varies: abelian vs non-abelian rotation
        .with_rotation(rotation);

    // input  [batch_size, sequence_len = SEQ_LENGTH + 1, input_size = NUM_SYMBOLS]
    // output [batch_size, sequence_len = SEQ_LENGTH + 1, output_size = NUM_CLASSES]
    MambaLatentNetConfig::Mamba3 {
        input_size: NUM_SYMBOLS,
        n_real_layers: 1,       // a single layer is sufficient
        n_virtual_layers: None, // don't virtually extend the amount of layers
        mamba_block,
        output_size: NUM_CLASSES,
        class_tokens: Vec::new(),
        residuals: ResidualsConfig::Standard,
    }
}
