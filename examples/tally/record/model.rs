//! The model configuration for tally-record: the plant of `reset-majority` (a
//! single real scalar per head), plus **one tropical register per head**. The
//! register holds the running maximum that the task asks for.
//!
//! ```ignore
//! cₜ = τ·log(exp((cₜ₋₁ + aₜ)/τ) + exp(bₜ/τ))   →   max(cₜ₋₁ + aₜ, bₜ)
//! yₜ = Cₜ·hₜ + D·xₜ + cₜ·eₕ
//! ```
//!
//! The construction (`tests.rs`) sets the register of one head to `a = 0`,
//! `b = S·v`: a running maximum in units of `S`. It reads `y = S·v + θ − c`,
//! which is positive exactly on a record. The state of the plant is unused
//! (`Δ ≈ 0`), and that is the rung: only the register moves.
//!
//! `d_model = 3` is the load-bearing width, against both routes of a stock
//! block. It is *not* the floor for a thirteen-symbol alphabet (that is 12). It
//! is the width at which the values still fit, on a circle, so that one
//! function of the symbol is an affine read of the embedding. But twelve
//! *independent* channels do not fit, and that closes the **latch** route (one
//! state scalar per value, selected by the gate). The other route, a sum in the
//! exponential domain, needs a single channel but a span of `e^{45.8}`. Every
//! channel reads the same 3-D embedding, so the block gets only the seven
//! digits of f32, however many heads it uses. See `tests.rs` and the README.

use crate::dataset::NUM_SYMBOLS;
use burn_mamba::prelude::{
    Gain, Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid, Tropical,
};

/// Number of output classes (see [`crate::dataset`]).
pub const NUM_CLASSES: usize = 2;

/// The config of the rung. `tropical = false` is the ablation arm
/// (`-- --stock`).
pub fn model_config(tropical: bool) -> MambaLatentNetConfig {
    let mamba_block = Mamba3Config::new(3)
        .with_state_rank(1) // a scalar state — nothing to rotate
        .with_expand(1)
        .with_per_head_dim(1) // ⇒ nheads = 3: the register, a reference, a spare
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rotation(RotationKind::Real1D)
        .with_trapezoid(Trapezoid::None)
        .with_gain(Gain::Projected)
        .with_tropical(if tropical {
            Tropical::MaxPlus
        } else {
            Tropical::None
        })
        .with_has_proj_bias(true);

    MambaLatentNetConfig::Mamba3 {
        input_size: NUM_SYMBOLS,
        output_size: NUM_CLASSES,
        final_norm: true,
        n_real_layers: 1,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block,
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: true,
        residuals: ResidualsConfig::Standard,
        mlp: None,
        untied: Vec::new(),
    }
}
