//! The model configuration for tally-depth: one Mamba-3 block. Its plant is the
//! plant of `reset-majority` (a single real scalar per head), plus **one
//! tropical register per head**. The register solves the task.
//!
//! The block unrolls to, per head,
//!
//! ```ignore
//! cₜ = τ·log(exp((cₜ₋₁ + aₜ)/τ) + exp(bₜ/τ))   →   max(cₜ₋₁ + aₜ, bₜ)
//! yₜ = Cₜ·hₜ + D·xₜ + cₜ·eₕ
//! ```
//!
//! with `a` and `b` projected per (head, token). The construction (see
//! `tests.rs`) uses one head for the register: `a = ±S` on `(` / `)`, and `b`
//! is the floor that the token guarantees (`S` on `(`, 0 elsewhere). So `c` is
//! the depth clamped at zero, times `S`. It uses the other
//! head for a constant reference, so the `final_norm` of the network reads the
//! direction of the pair. The state of the plant is **unused**: `Δ ≈ 0` in both
//! heads, and that is the point of the rung. Only the register moves.
//!
//! These config choices are load-bearing:
//!
//! - `Tropical::MaxPlus`: the register. It is structural: two in-projection
//!   channels per (head, micro-step), one cache slot, and the readout `eₕ`.
//! - `Gain::Projected`, `RotationKind::Real1D`, `Trapezoid::None`,
//!   `state_rank = 1`: the plant at the floor of the `reset` ladder, so the
//!   rung measures only the register.
//! - `d_model = 2` is the floor for a three-symbol alphabet (the argument of
//!   `reset-majority`). The pre-`RmsNorm` of the layer puts a token on a
//!   circle, and three points of `ℝ²` are affinely independent. So every
//!   channel can take any value on the three symbols.
//! - `ignore_last_residual`: the classification head sees only the block.

use crate::dataset::NUM_SYMBOLS;
use burn_mamba::prelude::{
    Gain, Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid, Tropical,
};

/// Number of output classes (see [`crate::dataset`]).
pub const NUM_CLASSES: usize = 2;

/// The config of the rung. `tropical = false` is the ablation arm
/// (`-- --stock`): the same block without the register, and nothing else
/// changed.
pub fn model_config(tropical: bool) -> MambaLatentNetConfig {
    let mamba_block = Mamba3Config::new(2)
        .with_state_rank(1) // a scalar state — nothing to rotate
        .with_expand(1)
        .with_per_head_dim(1) // ⇒ nheads = 2: the register's head, and a reference
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rotation(RotationKind::Real1D) // a real transition: decay only
        .with_trapezoid(Trapezoid::None) // no β tap: nothing here integrates
        .with_gain(Gain::Projected) // the decay is not the point of this rung
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
