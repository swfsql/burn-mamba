//! The model configuration for tally-drift — `reset-majority`'s plant with a
//! **Kalman gate** on its decay: the block's own precision `Λ` decides how much
//! of the past survives a gap.
//!
//! ```ignore
//! dₜ = αₜ / (1 + qₜ·αₜ·Λₜ₋₁)      qₜ = κₕ·Δₜ·exp(rₜ)     (rₜ projected)
//! Λₜ = dₜ·Λₜ₋₁ + Δₜ,   ηₜ = dₜ·ηₜ₋₁ + Δₜ·Bₜ·xₜ
//! yₜ = Cₜ·ηₜ·(Λₜ + ε)^(−ωₕ) + D·xₜ
//! ```
//!
//! The construction (`tests.rs`) gives the estimator head `Δ = 1` and `q ≈ 0`
//! on a value (evidence, no doubt) and `Δ ≈ 0`, `q = Q_GAP` on a gap (doubt, no
//! evidence) — which is what [`Gain::KalmanProjectedNoise`] is for, the tied arm
//! having no way to separate the two — and reads `ω = 1`, the estimate `η/Λ`
//! rather than the information `η`. The output is then `v − S`.
//!
//! Config choices that are load-bearing:
//!
//! - `Gain::KalmanProjectedNoise` — the computed decay, plus one in-projection
//!   channel per (head, micro-step) for `r`.
//! - `has_outproj_norm = false` — with the per-head norm on, `(Λ + ε)^(−ω)`
//!   would be removed by it, and the block would hold `η`, not `S`.
//! - `a_floor = 1e-8` — the estimator has to hold its sum *unweighted* across a
//!   whole sequence; the block's floor on `|A|` is the only thing that decays
//!   it, and at the default `1e-4` that leak is comparable to the margins here.
//! - `d_model = 3` with the nine values on a circle, as in `tally-record`.

use crate::dataset::NUM_SYMBOLS;
use burn_mamba::prelude::{
    Gain, Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid, Tropical,
};

/// Number of output classes (see [`crate::dataset`]).
pub const NUM_CLASSES: usize = 2;

/// The rung's config; `kalman = false` is the ablation arm (`-- --stock`), the
/// same block with a projected decay.
pub fn model_config(kalman: bool) -> MambaLatentNetConfig {
    let mamba_block = Mamba3Config::new(3)
        .with_state_rank(1)
        .with_expand(1)
        .with_per_head_dim(1) // ⇒ nheads = 3: the estimator, a reference, a spare
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rotation(RotationKind::Real1D)
        .with_trapezoid(Trapezoid::None)
        .with_gain(if kalman {
            Gain::KalmanProjectedNoise
        } else {
            Gain::Projected
        })
        .with_tropical(Tropical::None)
        .with_a_floor(1e-8)
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
