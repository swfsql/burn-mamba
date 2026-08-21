//! The model configuration for the reset-rotor example — one Mamba-3 block with
//! a **single rotating pair** as its whole state, sized so that nothing but the
//! block's complex transition can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind};

/// A single Mamba-3 block at `state_rank = 2`, unrolled, is one data-dependent
/// **rotation** per head plus a data-dependent decay:
///
/// ```ignore
/// Δₕ(u) = softplus(⟨aₕ, u⟩ + bₕ)     Aₕ(u) = −softplus(⟨cₕ, u⟩)     ᾱₕ = exp(Δₕ Aₕ)
/// ϱ(u)  = Δₕ · π · tanh(ϑ(u))        θₜ = θₜ₋₁ + ϱ(uₜ)              (cumulative)
/// hₜ⁽ʰ⁾ = ᾱₕ hₜ₋₁⁽ʰ⁾ + γₕ R(θₜ)B(uₜ) xₕ(uₜ)
/// yₜ⁽ʰ⁾ = (R(θₜ)Cₕ)ᵀ hₜ⁽ʰ⁾ + Dₕ xₕ(uₜ)
/// ```
///
/// Orthogonality makes the readout see only the rotation *accumulated between*
/// the injection and the read: if the state was last written at step `τ`, then
/// `yₜ = Cₕᵀ R(θ_τ − θₜ) B · (what was written)` — a function of `θₜ − θ_τ`
/// alone. The task is built around exactly that:
///
/// - **`R` writes the state.** `x(R) ≠ 0` and `A(R) ≈ −20` (so `ᾱ ≈ 0` wipes
///   what was there), leaving `h = R(θ_R)B` — the rotor's zero detent, recorded
///   at whatever phase the sequence happens to be at.
/// - **`±` only turn it.** `x(±) = 0`, so nothing is written and `ᾱ ≈ 1` holds
///   the state; all that happens is `θ` advancing by `±2π/3` — one detent.
/// - **The two heads read the same phase on two axes.** They share `Δ` (hence
///   the same angle), and their `C` differ by a quarter turn through the
///   per-head bias `c_bias_hmr`, so `(y₀, y₁) ∝ (cos φ, −sin φ)` with
///   `φ = θₜ − θ_R = (2π/3)·turns`. The three-class head is then a phase
///   decoder: logit `j` ∝ `cos(φ − 2πj/3)`.
///
/// The absolute phase drifts forever (it is never reset, and `wrap_angle` only
/// folds it mod `2π`); only the *difference* since the last write is read, which
/// is what makes the construction exact rather than approximate.
///
/// Config choices that are load-bearing:
///
/// - `state_rank = 2` ⇒ exactly one rotation pair, and with `rope_fraction = 1`
///   the whole state rotates. The rotor *is* the state.
/// - `per_head_dim = 1`, `expand = 1` ⇒ `nheads = 2`: the cos axis and the sin
///   axis, and nothing else.
/// - `ignore_last_residual` zeroes the single layer's residual, so `out_proj`
///   reads the block's output *alone* — without it the head also sees the
///   embedding of the current token, which cannot give the answer but does
///   muddy the claim. (Mamba-3 has no short convolution, so there is no local
///   window to close off: `conv_kernel` has no counterpart here.)
///
/// `d_model = 2` (rather than 1) is what keeps this constructible in closed
/// form: with a 2-D token every projection is an independent affine functional,
/// so `ϑ` can read the turn direction while `x` and `A` read the reset flag.
pub fn model_config() -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 2, per_head_dim = 1 ⇒ nheads = 2 (the cos head
    // and the sin head), each with its own Δ, A, λ and D.
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
    }
}
