//! The model configuration for the state-tracking example — a deliberately tiny
//! single-layer Mamba-3 classifier whose only varying knob is the rotation kind
//! (`Complex2D` vs `Quaternion4D`); see [`model_config`].

pub use crate::common::model::{MyMamba3NetworkConfig, mamba3_block_config, mamba3_layers_config};
use crate::dataset::{NUM_CLASSES, NUM_GENERATORS};
use burn_mamba::prelude::RotationKind;

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
pub fn model_config(rotation: RotationKind) -> MyMamba3NetworkConfig {
    MyMamba3NetworkConfig::new()
        // the input is a sequence of one-hot generators
        // the input shape is [batch_size, sequence_len = SEQ_LENGTH, input_size = NUM_GENERATORS]
        .with_input_size(NUM_GENERATORS)
        .with_layers(mamba3_layers_config(
            1,    // a single layer is sufficient
            None, // don't virtually extend the amount of layers
            mamba3_block_config(
                //
                32, // d_model
                16, // state_rank (multiple of 4, required by Quaternion4D)
                8,  // nheads (d_inner = expand * d_model = 64, so per_head_dim = 8)
                1,  // ngroups
                1,  // mimo_rank
                1., // apply RoPE to 100% of the B/C projections
                2,  // expand
            )
            // the one knob this example varies: abelian vs non-abelian rotation
            .with_rotation(rotation),
        ))
        // the output is a 60-way per-position classification (the running product)
        // the output shape is [batch_size, sequence_len = SEQ_LENGTH, output_size = NUM_CLASSES]
        .with_output_size(NUM_CLASSES)
}
