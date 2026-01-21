pub use crate::common::model::{Mamba2NetworkConfig, mamba2_block_config};

/// The Fibonacci-like sequence xₜ = xₜ₋₁ + xₜ₋₂ can be modeled as:
///
/// ```ignore
/// ⎧ hₜ⁽¹⁾ ⎫ = ⎧ 1 1 ⎫ ⎧ hₜ₋₁⁽¹⁾ ⎫ + ⎧ b₁ ⎫ xₜ
/// ⎩ hₜ⁽²⁾ ⎭   ⎩ 1 0 ⎭ ⎩ hₜ₋₁⁽²⁾ ⎭ + ⎩ b₂ ⎭
/// ```
///
/// And this can be well represented by a Mamba-2 model, which is defined as:
///
/// ```ignore
/// hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ
/// yₜ = Cₜ hₜ + D xₜ
/// ```
///
/// By default, this is a small model (~1.5 KB when stored) that shows a good convergence.
///
/// Counting the io projections, conv and norm layers -- none of which
/// are needed for the simplest case -- this results in 35 params.
/// Without those, 17 params should be the required minimum.
pub fn model_config() -> Mamba2NetworkConfig {
    Mamba2NetworkConfig::new()
        // the input is a sequence of a single-dimensioned values
        // the input shape is [batch_size, sequence_len = SEQ_LENGTH, input_size = 1]
        .with_input_size(1)
        // a single layer is sufficient
        .with_n_real_layers(1)
        // a very small configuration
        .with_mamba_block(mamba2_block_config(
            //
            1, // d_model: one feature is sufficient
            2, // d_state: two states are sufficient
            1, // d_conv: input conv is not necessary
            1, // n_heads: a single head is sufficient
            1, // expand: expansion is not necessary
        ))
        // the output is a single-dimensioned value
        // the output shape is [batch_size, sequence_len = SEQ_LENGTH, output_size = 1]
        // which is later narrowed to the last timestep [batch_size, sequence_len = 1, output_size = 1]
        .with_output_size(1)
}
