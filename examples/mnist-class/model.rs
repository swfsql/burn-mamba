//! The model configuration for the `mnist-class` example — a small Mamba-3
//! classifier (2 real layers stretched to 16 virtual layers); see
//! [`model_config`].

pub use crate::common::model::{MyMamba3NetworkConfig, mamba3_block_config, mamba3_layers_config};
use burn_mamba::schedule::Schedule;

/// This model configuration uses ~37K params (~153KB disk space in FP32).  
/// Reaches ~85% validation accuracy at the first epoch.  
/// With a batch_size=16 in FP32, this requires ~3.5GB vram during training.
pub fn model_config() -> MyMamba3NetworkConfig {
    MyMamba3NetworkConfig::new()
        // the input is a sequence of a single-dimensioned values
        // the input shape is [batch_size, sequence_len = HEIGHT * WIDTH, 1]
        .with_input_size(1)
        // to keep it simple, don't use any class token
        // .with_class_tokens(Vec::new()) // TODO: merge fork
        .with_layers(mamba3_layers_config(
            2, // two layers backed by unique weights is sufficient
            Some((
                16,                  // allow more expressivity by virtually extending to 16 layers,
                Schedule::Stretched, // by looping (8x) each layer in sequence
            )),
            mamba3_block_config(
                //
                32, // d_model (intra- and inter-layer expressivity, high impact on disk size)
                64, // state_rank (intra-layer and time-wise expressivity, average impact on disk size)
                4, // nheads (intra-layer expressivity, no impact on disk size, high impact on vram)
                1, // ngroups (intra-layer expressivity)
                1, // mimo_rank (intra-layer expressivity)
                // Apply RoPE to 100% of the B/C projections.
                //
                // Ablation:
                // With RoPE at 000% (disabled), acc reaches ~(10%, 20%, 25%) at baches (100, 200, 300).
                // With RoPE at 100% ( enabled), acc reaches ~(10%, 45%, 50%) at baches (100, 200, 300).
                //
                // Note: disabling RoPE saves vram training requirements by 25%.
                1., // apply RoPE to 100% of the B/C projections
                4,  // expand (intra-layer expressivity, small impact on disk size)
            ),
        ))
        // the output is a 10-bins classification
        // the output shape is [batch_size, sequence_len = HEIGHT * WIDTH, output_size = 10]
        // which is later narrowed to the last timestep [batch_size, sequence_len = 1, output_size = 10]
        .with_output_size(10)
}
// notes:
// - this small model requires quite a lot of vram because the whole 28*28 sequence for each image
//   is processed in parallel, and a high amount of virtual layers are used.
// - this should benefit from a bidi encoder since a single output is predicted
//   after the whole image is read.
