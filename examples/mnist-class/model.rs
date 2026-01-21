pub use crate::common::model::{Mamba2NetworkConfig, mamba2_block_config};
use burn_mamba::schedule::Schedule;

/// In FP16, this model configuration uses ~75KB when stored to disk.  
/// Reaches ~65% valid accuracy after the first epoch, reaching ~90% after 5 epochs.  
/// With a batch_size=16, this requires ~3.6GB vram during training.
pub fn model_config() -> Mamba2NetworkConfig {
    Mamba2NetworkConfig::new()
        // the input is a sequence of a single-dimensioned values
        // the input shape is [batch_size, sequence_len = HEIGHT * WIDTH, 1]
        .with_input_size(1)
        // two layers backed by unique weights is sufficient
        .with_n_real_layers(2)
        // allow more expressivity by virtually extending to 16 layers,
        // by looping (8x) the first layer, followed by another loop (8x) of the second layer
        .with_n_virtual_layers(Some((16, Schedule::Stretched)))
        .with_mamba_block(mamba2_block_config(
            //
            32, // d_model
            64, // d_state
            4,  // d_conv
            4,  // n_heads
            4,  // expand
        ))
        // the output is a 10-bins classification
        // the output shape is [batch_size, sequence_len = HEIGHT * WIDTH, output_size = 10]
        // which is later narrowed to the last timestep [batch_size, sequence_len = 1, output_size = 10]
        .with_output_size(10)
}
// notes:
// - this small model requires quite a lot of vram because the whole 28*28 sequence for each image
// is processed in parallel, and a high amount of virtual layers are used.
// - this could use a bidi encoder since a single output is given after the whole image is read
