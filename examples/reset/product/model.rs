//! The model configuration for the `spinor-product` example — `reset-spinor`'s
//! quaternion block, reading **two symbols per token**, with `micro_steps` the
//! only knob the example varies (see [`model_config`]).

use crate::dataset::{INPUT_SIZE, NUM_CLASSES};
use burn_mamba::prelude::{Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind};

/// `d_model = d_inner = 8`: four dimensions per slot, which is the smallest that
/// carries five symbols affinely (a 4-simplex). The state is one quaternion
/// (`state_rank = 4`) and `nheads = 4`, one head per component.
pub const D_MODEL: usize = 8;

/// A Mamba-3 block at `state_rank = 4` with [`RotationKind::Quaternion4D`]
/// carries, per head, a cumulative **unit quaternion** built by an ordered
/// product. Over the folded micro-step axis (`u` = `micro_steps` steps per
/// token, see `burn_mamba::mamba3::product`) one token runs
///
/// ```ignore
/// qₜ,ⱼ = quat(2π · Δₕ · tanh(‖ϑ(uₜ,ⱼ)‖) · ϑ̂(uₜ,ⱼ))    j = 0 … u−1
/// Pₜ  = qₜ,ᵤ₋₁ ⊗ ⋯ ⊗ qₜ,₀ ⊗ Pₜ₋₁       (the cumulative rotation)
/// hₜ,ⱼ = ᾱₕ hₜ,ⱼ₋₁ + γₕ (Pₜ,ⱼ* ⊗ B(uₜ,ⱼ)) xₕ(uₜ,ⱼ)
/// yₜ   = ⟨Pₜ* ⊗ C(uₜ), hₜ,ᵤ₋₁⟩            (the read is per token)
/// ```
///
/// so a **token's** transition is the product `Pₜ,ᵤ₋₁ ⋯ Pₜ,₀` — one group
/// element per symbol it carries. Left multiplication is orthogonal, so a state
/// written at token `τ` and read at `t` gives `⟨C, (Pₜ ⊗ P_τ*) ⊗ B⟩`, and
/// `Pₜ ⊗ P_τ*` is the group word since the write. Writing the identity at every
/// reset makes the readout the four components of the word itself, which is the
/// task; `tests.rs` writes every weight down.
///
/// **What `micro_steps` changes, and nothing else does.** At `u = 1` a token
/// gets one rotation, whose generator `ϑ` is an affine functional of the token
/// — and a token is two one-hot slots, so `ϑ(a, b) = v_a + w_b`. Generators
/// **add** where the group **multiplies**: with the hold symbol in the alphabet
/// `v_i` is pinned along `±x̂` and `w_j` along `±ŷ` (the axes of `i` and `j`),
/// so `ϑ(i, j)` lies in the `xy`-plane and `exp` of it can never be the `±k`
/// that token needs. At `u = 2` the token is two steps and its transition is
/// `exp(w_b) ⊗ exp(v_a)` — the product, exactly.
///
/// Config choices that are load-bearing:
///
/// - `state_rank = 4` is the smallest quaternion block, and `rope_fraction =
///   1.0` turns all of it: the group *is* the state.
/// - `d_model = 8` gives each slot four dimensions, which is the smallest that
///   carries five symbols affinely — `tests.rs` puts them on a regular
///   4-simplex of norm 2, so every token has RMS 1 (the layer's pre-`RmsNorm`
///   passes it through unchanged) and every in-projection channel has a
///   closed-form weight. `per_head_dim = 2` then keeps `nheads = 4` — one head
///   per quaternion component, as in `reset-spinor` — and the construction uses
///   each head's first value channel.
/// - `ignore_last_residual` zeroes the single layer's residual, so `out_proj`
///   reads the block's output alone.
pub fn model_config(micro_steps: usize) -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 8, per_head_dim = 2 ⇒ nheads = 4 (one per
    // quaternion component), each with its own Δ, A, λ and D — per micro-step.
    let mamba_block = Mamba3Config::new(D_MODEL)
        .with_state_rank(4) // one quaternion block — the group element itself
        .with_expand(1)
        .with_per_head_dim(2)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(1.0)
        .with_rotation(RotationKind::Quaternion4D)
        .with_micro_steps(micro_steps)
        .with_has_proj_bias(true);

    // input  [batch, tokens, INPUT_SIZE]  (two one-hot symbol slots)
    // output [batch, tokens, NUM_CLASSES] (one logit per group element, every token)
    MambaLatentNetConfig::Mamba3 {
        input_size: INPUT_SIZE,
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
