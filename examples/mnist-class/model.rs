//! The model configuration for the `mnist-class` example — a small Mamba-3
//! classifier (2 real layers stretched to 16 virtual layers); see
//! [`model_config`].

use burn_mamba::prelude::MultiGateResidualConfig;
use burn_mamba::prelude::{Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind};
use burn_mamba::utils::Schedule;

/// Depth of the (virtual) layer stack.
const N_VIRTUAL_LAYERS: usize = 16;

/// This model configuration uses ~37K params (~154KB disk space in FP32).
/// Reaches ~85% validation accuracy at the first epoch.
/// With a batch_size=16 in FP32, this requires ~3.6GB vram during training.
pub fn model_config() -> MambaLatentNetConfig {
    // d_model = 32 (intra/inter-layer expressivity, high impact on disk size)
    let d_model = 32;
    let mamba_block = Mamba3Config::new(d_model)
        // state_rank = 64 (time-wise expressivity, average impact on disk size)
        .with_state_rank(64)
        .with_expand(4)
        // d_inner = expand·d_model = 4·32 = 128
        // per_head_dim = 32
        // nheads = d_inner/per_head_dim = 128/32 = 4
        .with_per_head_dim(32)
        .with_ngroups(1)
        .with_mimo_rank(1)
        // rope_fraction = 1.0 (apply RoPE to 100% of the B/C projections)
        //
        // RoPE-kind ablation (batches 100/200/300):
        //   |    RoPE Kind | RoPE | Memory |    Accuracy   |
        //   |    Complex2D |   0% |  2.6GB | 10%, 20%, 25% |
        //   |    Complex2D | 100% |  3.5GB | 10%, 45%, 50% |
        //   | Quaternion4D | 100% |  4.3GB | 35%, 55%, 60% |
        .with_rope_fraction(1.0)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        .with_rotation(RotationKind::Complex2D); // 2D rotations on B/C

    // for MultiGate residuals (commented-out)
    const N_STREAM: usize = 4;
    let _carry_bias =
        MultiGateResidualConfig::depth_init_bias(N_VIRTUAL_LAYERS - (N_STREAM - 1), N_STREAM);

    MambaLatentNetConfig::Mamba3 {
        // input  [batch_size, sequence_len = HEIGHT * WIDTH, input_size = 1]
        input_size: 1,
        // output [batch_size, sequence_len = HEIGHT * WIDTH, output_size = 10]
        // (later narrowed to the last timestep for the 10-bin classification)
        output_size: 10,
        // best true for MultiGate Residuals (small model, few batches)
        final_norm: false,
        // two real layers, virtually stretched (8×) to 16 for more expressivity
        n_real_layers: 2,
        n_virtual_layers: Some((N_VIRTUAL_LAYERS, Schedule::Stretched)),
        mamba_block,
        class_tokens: Vec::new(),
        // the first input/last output could skip their residual here too
        ignore_first_residual: false,
        ignore_last_residual: false,
        // alternative:
        //
        // residuals: ResidualsConfig::MultiGate {
        //     n_stream: N_STREAM,
        //     init_bias: _carry_bias,
        // // useful ramp for the few batches in this example
        //     init_bias_step: -_carry_bias / (N_STREAM - 1) as f64,
        //     per_virtual_layer: true,
        // }
        residuals: ResidualsConfig::Standard,
        // No feed-forward interleave: these examples are mixer-only.
        mlp: None,
    }
}
// notes:
// - this small model requires quite a lot of vram because the whole 28*28 sequence for each image
//   is processed in parallel, and a high amount of virtual layers are used.
// - this should benefit from a bidi encoder since a single output is predicted
//   after the whole image is read.
