//! The model configuration for the `mnist-class` example — a tiny Mamba-3
//! classifier (one real layer applied as 4 virtual layers); see
//! [`model_config`].

use burn_mamba::prelude::{
    ClassLatent, LayerUntied, Mamba3Config, Mamba3Untied, MambaLatentNetConfig, ResidualsConfig,
    RotationKind,
};
use burn_stack::utils::{GradHorizon, Schedule};

/// Depth of the stack: how many times the single real layer is applied.
/// Virtual layers reuse the real layer's weights (except its untied tensors, see
/// [`model_config`]), so depth costs compute rather than parameters.
const N_VIRTUAL_LAYERS: usize = 4;

/// Back-propagate only the last `K` applications of **each real layer** —
/// [`GradHorizon::Depth`], counted per weight set — with everything below
/// running on the inner backend; `None` tracks the whole stack. Truncation
/// trades gradient for vram, which a stack this shallow does not need.
const GRAD_HORIZON: Option<GradHorizon> = None;

/// Stack-level class latents: learnable `[CLS]`-style registers (width
/// `d_model`) prepended to every image's pixel sequence. None are used here.
pub const N_CLASS_LATENTS: usize = 0;

/// How much longer the model's output is than its pixel input, in timesteps.
/// The class latents all sit at the **front** (`Start`), so the classification
/// readout is still the sequence's last position — just not index `784 - 1`.
pub const OUTPUT_SEQUENCE_EXTRA: usize = N_CLASS_LATENTS;

/// A 954-parameter classifier. With the training config in `main.rs` (batch 16,
/// fp32, one cosine LR schedule over 4 epochs) it reaches ~90% validation
/// accuracy by the end of the 4th epoch.
pub fn model_config() -> MambaLatentNetConfig {
    let d_model = 6;
    let mamba_block = Mamba3Config::new(d_model)
        .with_state_rank(8)
        .with_expand(2)
        // d_inner = expand·d_model = 2·6 = 12
        // per_head_dim = 6
        // nheads = d_inner/per_head_dim = 12/6 = 2
        .with_per_head_dim(6)
        // B/C are projected once per group; with ngroups = nheads every head
        // writes and reads its state through its own B/C.
        .with_ngroups(2)
        .with_mimo_rank(1)
        // The transition's rotation: Real1D | Complex2D | Quaternion4D | Rotor4D.
        // `rope_fraction = 0.5` turns half of the state rank (one quaternion
        // block per head) and leaves the other half unrotated.
        .with_rope_fraction(0.5)
        .with_has_proj_bias(false)
        .with_has_outproj_norm(true)
        .with_rotation(RotationKind::Quaternion4D)
        // Held once per virtual layer instead of shared by all of them: the
        // in-projection's per-head tail (Δ, A, λ and the rotation) and the small
        // per-head and norm tensors.
        .with_untied(vec![
            Mamba3Untied::InProjTail,
            Mamba3Untied::DtBias,
            Mamba3Untied::D,
            Mamba3Untied::BNorm,
            Mamba3Untied::CNorm,
            Mamba3Untied::OutNorm,
        ]);

    MambaLatentNetConfig::Mamba3 {
        // input  [batch_size, sequence_len = HEIGHT * WIDTH, input_size = 1]
        input_size: 1,
        // output [batch_size, HEIGHT * WIDTH + OUTPUT_SEQUENCE_EXTRA, output_size = 10]
        // (later narrowed to the last timestep for the 10-bin classification)
        output_size: 10,
        final_norm: false,
        // one real layer, applied `N_VIRTUAL_LAYERS` times
        n_real_layers: 1,
        n_virtual_layers: Some((N_VIRTUAL_LAYERS, Schedule::Stretched)),
        grad_horizon: GRAD_HORIZON,
        mamba_block,
        // Network-level class tokens would sit at `input_size = 1` (a single
        // learnable scalar each); the stack-level latents are `d_model` wide.
        class_tokens: Vec::new(),
        class_latents: vec![ClassLatent::Start; N_CLASS_LATENTS],
        // the first input/last output could skip their residual here too
        ignore_first_residual: false,
        ignore_last_residual: false,
        // Multi-Gate residuals: `n_stream` gated residual streams, pooled into
        // each layer's input, instead of one additive skip
        // (`ResidualsConfig::Standard`).
        residuals: ResidualsConfig::MultiGate {
            n_stream: 4,
            init_bias: 0.0,
            init_bias_step: 0.0,
            // one gate module per real layer, reused by its virtual passes
            per_virtual_layer: false,
        },
        // No feed-forward interleave: these examples are mixer-only.
        mlp: None,
        // The layer's own untied tensors (the block's are `mamba_block`'s, above):
        // the mixer's pre-norm. `LayerUntied::Norm2` would need an `mlp`.
        untied: vec![LayerUntied::Norm],
    }
}
