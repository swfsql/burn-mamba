pub use crate::common::model::{MyMamba2NetworkConfig, mamba2_block_config, mamba2_layers_config};
use burn_mamba::schedule::Schedule;

/// In FP16, this model configuration uses ~75KB when stored to disk.  
/// Reaches ~65% valid accuracy after the first epoch, reaching ~90% after 5 epochs.  
/// With a batch_size=16, this requires ~3.6GB vram during training.
pub fn model_config() -> MyMamba2NetworkConfig {
    MyMamba2NetworkConfig::new()
        // the input is a sequence of a single-dimensioned values
        // the input shape is [batch_size, sequence_len = HEIGHT * WIDTH, 1]
        .with_input_size(1)
        .with_layers(mamba2_layers_config(
            2, // two layers backed by unique weights is sufficient
            Some((
                16,                  // allow more expressivity by virtually extending to 16 layers,
                Schedule::Stretched, // by looping (8x) each layer in sequence
            )),
            mamba2_block_config(
                //
                32, // d_model (intra- and inter-layer expressivity, high impact on disk size)
                64, // d_state (intra-layer and time-wise expressivity, average impact on disk size)
                4,  // d_conv (input convolution, possibly not needed)
                4, // n_heads (intra-layer expressivity, no impact on disk size, high impact on vram)
                4, // expand (intra-layer expressivity, small impact on disk size)
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
