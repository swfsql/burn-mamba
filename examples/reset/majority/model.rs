//! The model configuration for the reset-majority example: one Mamba-3 block
//! whose whole state is a **single real scalar** per head. Its size makes sure
//! that only the block can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{
    Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid,
};

/// A single Mamba-3 block at `state_rank = 1` and [`RotationKind::Real1D`],
/// unrolled, is two data-dependent scalar recurrences:
///
/// ```ignore
/// Δₕ(u) = softplus(⟨aₕ, u⟩ + bₕ)     Aₕ(u) = −softplus(⟨cₕ, u⟩)     ᾱₕ = exp(Δₕ Aₕ)
/// γₕ(u) = Δₕ(u)                     (`Trapezoid::None`: β = 0, current token only)
/// hₜ⁽ʰ⁾ = ᾱₕ(uₜ)·hₜ₋₁⁽ʰ⁾ + γₕ(uₜ)·B(uₜ)·xₕ(uₜ)
/// yₜ⁽ʰ⁾ = C(uₜ)·hₜ⁽ʰ⁾ + Dₕ·xₕ(uₜ)
/// ```
///
/// with `x` `silu(affine(uₜ))`, `B`/`C` QK-normed affines of `uₜ`, and `Δ`, `A`,
/// `D` **per head**. The task uses exactly that shape:
///
/// - **Head 0 is the ballot box.** `A₀` reads the reset flag. On `±` it is at
///   the `a_floor` of the block, so `ᾱ₀ ≈ 1` and `h₀` is an unweighted running
///   sum. On `RESET` it is large, so `ᾱ₀ ≈ 0` erases the sum. `x₀(±) = ±v`,
///   `x₀(RESET) = 0` and `D₀ = 0`, so `y₀ = C·h₀` is the running vote.
/// - **Head 1 is a fixed reference.** `Δ₁ ≈ 0` always, so `h₁ ≈ 0` and
///   `y₁ = D₁·x₁ = c > 0`, a constant.
///
/// The `final_norm` of the network then keeps only the **direction** of
/// `(y₀, y₁)`. That is the right shape for the task. The sign of the direction
/// is the answer. When `y₀` is near zero, the reference axis keeps the direction
/// well defined, and the margin proportional to the vote. At this width, the
/// block *can* emit only a bounded, sign-like output. That is why the task is a
/// classification and not a regression.
///
/// Necessary config choices:
///
/// - `state_rank = 1` with [`RotationKind::Real1D`]: the bottom rung of the
///   rotation ladder, the trivial group. The transition is a plain real decay.
///   So the block projects no rotation channels and caches no rotation
///   accumulator. The state is a *scalar*, and the other rungs of the ladder do
///   exactly what this one cannot. (`Real1D` is also the one kind that accepts
///   an odd `state_rank`: there is no pair to rotate.)
/// - Mamba-3 has no short convolution, so the SSM state is the only memory of
///   the model. There is no local window to use as a shortcut.
/// - `Trapezoid::None`: the construction pins `λ ≈ 1`, so the `β` tap is dead
///   weight. Switching it off is structural: no `λ` segment in the
///   in-projection, no tap slot in the cache, one SSD call instead of two.
/// - `ignore_last_residual` zeroes the residual of the single layer, so the
///   head reads the output of the block *alone*. Without it, the head also sees
///   the embedding of the current token. That embedding cannot give the answer,
///   but it makes the claim less clear.
///
/// `d_model = 2` (not 1) keeps this constructible in closed form. With a 2-D
/// token, every projection is an independent affine functional, so `A` can read
/// the reset flag while `x` reads the vote. At `d_model = 1`, they are all
/// monotone functions of the same scalar.
pub fn model_config() -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 2, per_head_dim = 1 ⇒ nheads = 2 (one head for
    // the vote, one for the reference), each with its own Δ, A and D.
    // state_rank = 1 ⇒ each head's state is a single scalar.
    let mamba_block = Mamba3Config::new(2)
        .with_state_rank(1) // a scalar state — nothing to rotate
        .with_expand(1)
        .with_per_head_dim(1)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rotation(RotationKind::Real1D) // a real transition: decay only
        .with_trapezoid(Trapezoid::None) // no β tap: nothing here integrates
        .with_has_proj_bias(true);

    // input  [batch, seq, NUM_SYMBOLS]  (one-hot symbol)
    // output [batch, seq, NUM_CLASSES]  (Neg / Pos logits, every scored position)
    MambaLatentNetConfig::Mamba3 {
        input_size: NUM_SYMBOLS,
        output_size: NUM_CLASSES,
        // the head reads a direction, not a magnitude: this norm is what bounds
        // the block's output and keeps the reference axis load-bearing.
        final_norm: true,
        n_real_layers: 1,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block,
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        // the single layer's residual: dropped, so the head sees only the state
        ignore_last_residual: true,
        residuals: ResidualsConfig::Standard,
        // No feed-forward interleave: these examples are mixer-only.
        mlp: None,
        untied: Vec::new(),
    }
}
