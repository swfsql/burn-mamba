use crate::common::model::{Mamba2NetworkConfig, mamba2_block_config};

/// This model configuration results in 363_530 params, using 717KB when stored to disk.
///
/// After the first epoch, this model reaches ~85% accuracy.  
/// With a batch_size=16 and in fp16, this requires ~3700 MB vram during training.  
pub fn model_config() -> Mamba2NetworkConfig {
    Mamba2NetworkConfig::new()
        // the input is a sequence of a single-dimensioned values
        .with_input_size(1)
        .with_n_layer(8)
        .with_mamba_block(mamba2_block_config(
            //
            64,  // d_model
            128, // d_state
            7,   // d_conv
            8,   // n_heads
            2,   // expand
        ))
        // the output is a 10-bins classification
        .with_output_size(10)
}
// notes:
// - this small model requires quite a lot of vram because the whole 28*28 sequence for each image
// is processed in parallel.
// - no parameter search/ablation were made
