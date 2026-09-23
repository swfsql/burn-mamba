//! The model configuration for the reset-rotor example: one Mamba-3 block whose
//! whole state is a **single rotating pair**. Its size makes sure that only the
//! complex transition of the block can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{
    Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid,
};

/// A single Mamba-3 block at `state_rank = 2`, unrolled, is one data-dependent
/// **rotation** per head plus a data-dependent decay:
///
/// ```ignore
/// Δₕ(u) = softplus(⟨aₕ, u⟩ + bₕ)     Aₕ(u) = −softplus(⟨cₕ, u⟩)     ᾱₕ = exp(Δₕ Aₕ)
/// ϱ(u)  = Δₕ · 2π · tanh(ϑ(u))       θₜ = θₜ₋₁ + ϱ(uₜ)              (cumulative)
/// hₜ⁽ʰ⁾ = ᾱₕ hₜ₋₁⁽ʰ⁾ + γₕ R(θₜ)B(uₜ) xₕ(uₜ)
/// yₜ⁽ʰ⁾ = (R(θₜ)Cₕ)ᵀ hₜ⁽ʰ⁾ + Dₕ xₕ(uₜ)
/// ```
///
/// Because the rotation is orthogonal, the readout sees only the rotation
/// *accumulated between* the write and the read. If the state was last written
/// at step `τ`, then `yₜ = Cₕᵀ R(θ_τ − θₜ) B · (what was written)`: a function
/// of `θₜ − θ_τ` alone. The task uses exactly that:
///
/// - **`R` writes the state.** `x(R) ≠ 0` and `A(R) ≈ −20` (so `ᾱ ≈ 0` erases
///   the old state). That leaves `h = R(θ_R)B`: the zero detent of the rotor,
///   recorded at the current phase of the sequence.
/// - **`±` only turn it.** `x(±) = 0`, so nothing is written, and `ᾱ ≈ 1` holds
///   the state. Only `θ` changes, by `±2π/3` (one detent).
/// - **The two heads read the same phase on two axes.** They share `Δ` (so the
///   same angle). The per-head bias `c_bias_hmr` puts their `C` a quarter turn
///   apart. So `(y₀, y₁) ∝ (cos φ, −sin φ)`, with
///   `φ = θₜ − θ_R = (2π/3)·turns`. The three-class head is then a phase
///   decoder: logit `j` ∝ `cos(φ − 2πj/3)`.
///
/// The absolute phase drifts forever. It is never reset, and `wrap_angle` only
/// folds it mod `2π`. The readout reads only the *difference* since the last
/// write, and that makes the construction exact, not approximate.
///
/// Necessary config choices:
///
/// - `state_rank = 2` ⇒ exactly one rotation pair. With `rope_fraction = 1`, the
///   whole state rotates. The rotor *is* the state.
/// - `per_head_dim = 1`, `expand = 1` ⇒ `nheads = 2`: the cos axis and the sin
///   axis, and nothing else.
/// - `Trapezoid::None`: the construction pins `λ ≈ 1`, so the `β` tap is dead
///   weight. Switching it off is structural: no `λ` segment in the
///   in-projection, no tap slot in the cache, one SSD call instead of two.
/// - `ignore_last_residual` zeroes the residual of the single layer, so
///   `out_proj` reads the output of the block *alone*. Without it, the head also
///   sees the embedding of the current token. That embedding cannot give the
///   answer, but it makes the claim less clear. (Mamba-3 has no short
///   convolution, so there is no local window to close: `conv_kernel` has no
///   counterpart here.)
///
/// `d_model = 2` (not 1) keeps this constructible in closed form. With a 2-D
/// token, every projection is an independent affine functional, so `ϑ` can read
/// the turn direction while `x` and `A` read the reset flag.
pub fn model_config() -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 2, per_head_dim = 1 ⇒ nheads = 2 (the cos head
    // and the sin head), each with its own Δ, A and D.
    // state_rank = 2 ⇒ the state is a single plane, and rope_fraction = 1.0
    // rotates all of it.
    let mamba_block = Mamba3Config::new(2)
        .with_state_rank(2) // one rotation pair — the rotor itself
        .with_expand(1)
        .with_per_head_dim(1)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(1.0)
        .with_rotation(RotationKind::Complex2D)
        .with_trapezoid(Trapezoid::None) // no β tap: nothing here integrates
        .with_has_proj_bias(true);

    // input  [batch, seq, NUM_SYMBOLS]  (one-hot symbol)
    // output [batch, seq, NUM_CLASSES]  (one logit per detent, every position)
    MambaLatentNetConfig::Mamba3 {
        input_size: NUM_SYMBOLS,
        output_size: NUM_CLASSES,
        // no final norm: the block's output is already O(1) (the state is a
        // rotating vector of fixed length) and the head reads it directly.
        final_norm: false,
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
