//! The model configuration for the reset-spinor example — one Mamba-3 block
//! whose state is a **single quaternion**, sized so that nothing but its
//! non-abelian rotation can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind};

/// A Mamba-3 block at `state_rank = 4` with [`RotationKind::Quaternion4D`]
/// carries, per head, a cumulative **unit quaternion** built by an ordered
/// product (an associative scan, not a `cumsum`):
///
/// ```ignore
/// qₜ = quat(Δₕ · π · tanh(ϑ(uₜ)))        Qₜ = qₜ ⊗ qₜ₋₁ ⊗ ⋯ ⊗ q₁
/// B̄ₜ = Qₜ* ⊗ B(uₜ)     C̄ₜ = Qₜ* ⊗ C(uₜ)   (the rotation, absorbed into B/C)
/// hₜ = ᾱₕ hₜ₋₁ + γₕ B̄ₜ xₕ(uₜ)            yₜ⁽ʰ⁾ = ⟨C̄ₜ⁽ʰ⁾, hₜ⁽ʰ⁾⟩ + Dₕ xₕ(uₜ)
/// ```
///
/// Left multiplication by a unit quaternion is orthogonal, so a state written
/// at step `τ` and read at step `t` gives
///
/// ```ignore
/// yₜ⁽ʰ⁾ = ⟨Qₜ* ⊗ C⁽ʰ⁾, Q_τ* ⊗ B⟩ = ⟨C⁽ʰ⁾, (Qₜ ⊗ Q_τ*) ⊗ B⟩
/// ```
///
/// and `Qₜ ⊗ Q_τ* = qₜ ⊗ ⋯ ⊗ q_{τ+1}` is the **ordered product of the steps
/// since the write** — the group word itself, newest factor on the left. That is
/// the whole task, so the construction is an embedding rather than an encoding:
///
/// - **`R` writes the identity.** `x(R) ≠ 0` and `A(R) ≈ −20` (so `ᾱ ≈ 0` wipes
///   what was there), leaving `h = B = 1`, the identity quaternion, recorded at
///   whatever cumulative rotation the sequence has reached.
/// - **`i` and `j` only turn it.** `x(i) = x(j) = 0`, so nothing is written and
///   `ᾱ ≈ 1` holds; each contributes its own half-turn quaternion to the
///   product. `i` and `j` are half-turns about orthogonal axes, and they do not
///   commute — `ij = k`, `ji = −k`.
/// - **The four heads read the four components.** They share `Δ` (hence the same
///   rotation) and their `C` are the four basis quaternions, set apart by the
///   per-head bias `c_bias_hmr` alone, so `(y₀, y₁, y₂, y₃) ∝ q_rel` and the
///   eight-class head is a nearest-element decoder: logit `g` = `⟨q_rel, g⟩`.
///
/// A half-turn needs `Δ · π · tanh(ϑ) = π`, i.e. `tanh(ϑ) = 1`. That is exactly
/// representable: `tanh` saturates to `1.0` in f32 well before `ϑ = 20`, so the
/// generator is `π` to the last bit and the per-step quaternion is `i` up to
/// `cos(π/2) ≈ 4e-8`.
///
/// Config choices that are load-bearing:
///
/// - `state_rank = 4` is the smallest quaternion block — one `SU(2)` factor, and
///   `rope_fraction = 1.0` turns all of it. The group *is* the state.
/// - `per_head_dim = 1`, `expand = 1`, `d_model = 4` ⇒ `nheads = 4`: one head
///   per quaternion component, and nothing else. `d_model = 4` also carries the
///   three symbol embeddings as **orthogonal** vectors, which is what makes the
///   in-projection a plain lookup table (see `tests.rs`).
/// - `ignore_last_residual` zeroes the single layer's residual, so `out_proj`
///   reads the block's output alone.
///
/// Passing [`RotationKind::Complex2D`] leaves everything else identical and is
/// the example's ablation: the cumulative rotation collapses to a `cumsum` of
/// angles, which is a function of the symbol *counts* — and the counts cannot
/// tell `ij` from `ji`.
pub fn model_config(rotation: RotationKind) -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 4, per_head_dim = 1 ⇒ nheads = 4 (one per
    // quaternion component), each with its own Δ, A, λ and D.
    let mamba_block = Mamba3Config::new(4)
        .with_state_rank(4) // one quaternion block — the group element itself
        .with_expand(1)
        .with_per_head_dim(1)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(1.0)
        .with_rotation(rotation)
        .with_has_proj_bias(true);

    // input  [batch, seq, NUM_SYMBOLS]  (one-hot symbol)
    // output [batch, seq, NUM_CLASSES]  (one logit per group element, every position)
    MambaLatentNetConfig::Mamba3 {
        input_size: NUM_SYMBOLS,
        output_size: NUM_CLASSES,
        // no final norm: the state is a unit quaternion, so the block's output
        // is already O(1) and the head reads it directly.
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
