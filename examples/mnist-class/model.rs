//! The model configuration for the `mnist-class` example — a small Mamba-3
//! classifier (2 real layers stretched to 16 virtual layers); see
//! [`model_config`].

use burn_mamba::prelude::{Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind};
use burn_mamba::utils::Schedule;

/// This model configuration uses ~37K params (~153KB disk space in FP32).
/// Reaches ~85% validation accuracy at the first epoch.
/// With a batch_size=16 in FP32, this requires ~3.5GB vram during training.
pub fn model_config() -> MambaLatentNetConfig {
    // A small Mamba-3 block:
    let mamba_block = Mamba3Config::new(
        // d_model = 32 (intra/inter-layer expressivity, high impact on disk size)
        32,
    )
    // state_rank = 64 (time-wise expressivity, average impact on disk size)
    .with_state_rank(64)
    .with_expand(4)
    // d_inner = expand·d_model = 128
    // nheads = 4
    // per_head_dim = d_inner/nheads = 32
    .with_per_head_dim(32)
    .with_ngroups(1)
    .with_mimo_rank(1)
    // rope_fraction = 1.0 (apply RoPE to 100% of the B/C projections)
    //
    // RoPE-kind ablation (batches 100/200/300):
    //   | RoPE Kind | RoPE | Memory |    Accuracy   |
    //   | Complex2D |   0% |  2.6GB | 10%, 20%, 25% |
    //   | Complex2D | 100% |  3.5GB | 10%, 45%, 50% |
    //   | Complex4D | 100% |  4.3GB | 35%, 55%, 60% |
    .with_rope_fraction(1.0)
    .with_has_proj_bias(true)
    .with_has_outproj_norm(true)
    .with_rotation(RotationKind::Complex2D); // 2D rotations on B/C

    MambaLatentNetConfig::Mamba3 {
        // input  [batch_size, sequence_len = HEIGHT * WIDTH, input_size = 1]
        input_size: 1,
        // output [batch_size, sequence_len = HEIGHT * WIDTH, output_size = 10]
        // (later narrowed to the last timestep for the 10-bin classification)
        output_size: 10,
        // two real layers, virtually stretched (8×) to 16 for more expressivity
        n_real_layers: 2,
        n_virtual_layers: Some((16, Schedule::Stretched)),
        mamba_block,
        class_tokens: Vec::new(),
        // the first input/last output could skip their residual here too
        ignore_first_residual: false,
        ignore_last_residual: false,
        // residuals: ResidualsConfig::Standard,
        // Multi-Gate Residuals (independent sigmoid gate) over 4 streams: the
        // 16 virtual layers update 4 parallel residual streams via gated mixing,
        // pooled per layer by depth-wise attention. A slightly negative init bias
        // biases the gates towards carry early in training (see the paper's
        // depth-aware formula); 4 streams over 16 virtual layers is modest.
        residuals: ResidualsConfig::MultiGate {
            n_stream: 4,
            init_bias: -1.0,
            // // one MGR per real layer (the 2 weight sets), reused across the 16
            // // virtual passes; set `true` to give each virtual layer its own.
            // per_virtual_layer: false,
            per_virtual_layer: true,
        },
    }
}
// notes:
// - this small model requires quite a lot of vram because the whole 28*28 sequence for each image
//   is processed in parallel, and a high amount of virtual layers are used.
// - this should benefit from a bidi encoder since a single output is predicted
//   after the whole image is read.
