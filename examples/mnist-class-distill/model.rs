pub use crate::common::model::{MyMamba3NetworkConfig, mamba3_block_config, mamba3_layers_config};
use burn_mamba::schedule::Schedule;

/// The student model uses the same architecture as the teacher, but it's weaker (has fewer parameters
/// and overall requires less vram).
///
/// See also the teacher model's [model_config](crate::teacher_model::model_config).
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
                32,  // d_model (intra- and inter-layer expressivity, high impact on disk size)
                64, // state_rank (intra-layer and time-wise expressivity, average impact on disk size)
                4, // nheads (intra-layer expressivity, no impact on disk size, high impact on vram)
                1, // ngroups (intra-layer expressivity)
                1, // mimo_rank (intra-layer expressivity)
                1.0, // apply rope to 100% of the projections
                4, // expand (intra-layer expressivity, small impact on disk size)
            ),
        ))
        // the output is a 10-bins classification
        // the output shape is [batch_size, sequence_len = HEIGHT * WIDTH, output_size = 10]
        // which is later narrowed to the last timestep [batch_size, sequence_len = 1, output_size = 10]
        .with_output_size(10)
}
