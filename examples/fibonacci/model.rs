//! The model configuration for the fibonacci example — a deliberately tiny
//! Mamba-2 network sized to the synthetic task (see [`model_config`]).

use burn_mamba::prelude::{Mamba2Config, MambaLatentNetConfig, ResidualsConfig};

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
/// hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ
/// yₜ = Cₜ hₜ + D xₜ
/// ```
///
/// By default, this is a small model (~1.5 KB when stored) that shows a good convergence.
///
/// Counting the io projections, conv and norm layers -- none of which
/// are needed for the simplest case -- this results in 35 params.
/// Without those, 17 params should be the required minimum.
pub fn model_config() -> MambaLatentNetConfig {
    // A tiny Mamba-2 block sized to the task:
    let mamba_block = Mamba2Config::new(1) // d_model = 1 (one feature is sufficient)
        .with_state_rank(2)
        // state_rank = 2 (two states suffice for the order-2 recurrence)
        // conv_kernel = 1 (no input conv needed)
        .with_conv_kernel(1)
        .with_expand(1)
        // d_inner = expand·d_model = 1
        // nheads = 1
        // per_head_dim = d_inner/nheads = 1
        .with_per_head_dim(1)
        .with_ngroups(1)
        .with_has_proj_bias(true);

    // the input/output are single-dimensioned values
    MambaLatentNetConfig::Mamba2 {
        // input shape  [batch_size, sequence_len = SEQ_LENGTH, input_size = 1]
        input_size: 1,
        // output shape [batch_size, sequence_len = SEQ_LENGTH, output_size = 1]
        // (later narrowed to the last timestep)
        output_size: 1,
        // one real layer suffices
        n_real_layers: 1,
        n_virtual_layers: None,
        mamba_block,
        class_tokens: Vec::new(),
        residuals: ResidualsConfig::Standard,
    }
}
