//! The model configuration for the reset-majority example — one Mamba-3 block
//! whose whole state is a **single real scalar** per head, sized so that nothing
//! *but* the block can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind};

/// A single Mamba-3 block at `state_rank = 1` and [`RotationKind::Real1D`],
/// unrolled, is two data-dependent scalar recurrences:
///
/// ```ignore
/// Δₕ(u) = softplus(⟨aₕ, u⟩ + bₕ)     Aₕ(u) = −softplus(⟨cₕ, u⟩)     ᾱₕ = exp(Δₕ Aₕ)
/// γₕ(u) = λₕ(u)·Δₕ(u)                (λ ≈ 1 ⇒ β = 0: only the current token)
/// hₜ⁽ʰ⁾ = ᾱₕ(uₜ)·hₜ₋₁⁽ʰ⁾ + γₕ(uₜ)·B(uₜ)·xₕ(uₜ)
/// yₜ⁽ʰ⁾ = C(uₜ)·hₜ⁽ʰ⁾ + Dₕ·xₕ(uₜ)
/// ```
///
/// with `x` `silu(affine(uₜ))`, `B`/`C` QK-normed affines of `uₜ`, and `Δ`, `A`,
/// `λ`, `D` **per head**. The task is built around exactly that shape:
///
/// - **head 0 is the ballot box.** `A₀` reads the reset flag: at the block's
///   `a_floor` on `±` (so `ᾱ₀ ≈ 1` and `h₀` is an unweighted running sum) and
///   large on `RESET` (so `ᾱ₀ ≈ 0` wipes it). `x₀(±) = ±v`, `x₀(RESET) = 0`,
///   `D₀ = 0`, so `y₀ = C·h₀` is the running vote.
/// - **head 1 is a fixed reference.** `Δ₁ ≈ 0` always, so `h₁ ≈ 0` and
///   `y₁ = D₁·x₁ = c > 0`, a constant.
///
/// The network's `final_norm` then keeps only the **direction** of `(y₀, y₁)`.
/// That is exactly the right shape for the task: the direction's sign is the
/// answer, and the reference axis keeps it well defined (and the margin
/// proportional to the vote) when `y₀` is near zero. A bounded, sign-like output
/// is all this block *can* emit at this width, which is why the task is a
/// classification and not a regression.
///
/// Config choices that are load-bearing:
///
/// - `state_rank = 1` with [`RotationKind::Real1D`] — the bottom rung of the
///   rotation ladder, the trivial group. The transition is a plain real decay,
///   so the block projects no rotation channels and caches no rotation
///   accumulator: a *scalar* state, and the ladder's other three rungs are
///   exactly what this one cannot do. (`Real1D` is also the one kind that
///   admits an odd `state_rank`: there is no pair to rotate.)
/// - Mamba-3 has no short convolution, so the SSM state is automatically the
///   model's only memory — there is no local window to shortcut through.
/// - `ignore_last_residual` zeroes the single layer's residual, so the head
///   reads the block's output *alone*. Without it it also sees the embedding of
///   the current token, which cannot give the answer but does muddy the claim.
///
/// `d_model = 2` (rather than 1) is what keeps this constructible in closed
/// form: with a 2-D token every projection is an independent affine functional,
/// so `A` can read the reset flag while `x` reads the vote. At `d_model = 1`
/// they are all monotone functions of the same scalar.
pub fn model_config() -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 2, per_head_dim = 1 ⇒ nheads = 2 (one head for
    // the vote, one for the reference), each with its own Δ, A, λ and D.
    // state_rank = 1 ⇒ each head's state is a single scalar.
    let mamba_block = Mamba3Config::new(2)
        .with_state_rank(1) // a scalar state — nothing to rotate
        .with_expand(1)
        .with_per_head_dim(1)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rotation(RotationKind::Real1D) // a real transition: decay only
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
    }
}
