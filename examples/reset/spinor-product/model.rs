//! The model configuration for the `spinor-product` example. It is the
//! quaternion block of `reset-spinor`, and it reads **two symbols per token**.
//! The example varies two knobs: `micro_steps` and `layers` (see
//! [`model_config`]).

use crate::dataset::{INPUT_SIZE, NUM_CLASSES};
use burn_mamba::prelude::{Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind};

/// `d_model = d_inner = 8`: four dimensions per slot, which is the smallest that
/// carries five symbols affinely (a 4-simplex). The state is one quaternion
/// (`state_rank = 4`) and `nheads = 4`, one head per component.
pub const D_MODEL: usize = 8;

/// A Mamba-3 block at `state_rank = 4` with [`RotationKind::Quaternion4D`]
/// carries a cumulative **unit quaternion** per head, and an ordered product
/// builds it. The micro-steps fold into the sequence axis (`u` = `micro_steps`
/// steps per token, see `burn_mamba::mamba3::product`). One token runs:
///
/// ```ignore
/// qₜ,ⱼ = quat(2π · Δₕ · tanh(‖ϑ(uₜ,ⱼ)‖) · ϑ̂(uₜ,ⱼ))    j = 0 … u−1
/// Pₜ  = qₜ,ᵤ₋₁ ⊗ ⋯ ⊗ qₜ,₀ ⊗ Pₜ₋₁       (the cumulative rotation)
/// hₜ,ⱼ = ᾱₕ hₜ,ⱼ₋₁ + γₕ (Pₜ,ⱼ* ⊗ B(uₜ,ⱼ)) xₕ(uₜ,ⱼ)
/// yₜ   = ⟨Pₜ* ⊗ C(uₜ), hₜ,ᵤ₋₁⟩            (the read is per token)
/// ```
///
/// So the transition of a **token** is the product `qₜ,ᵤ₋₁ ⊗ ⋯ ⊗ qₜ,₀`, one
/// group element per symbol in the token. Left multiplication is orthogonal. So
/// a state written at token `τ` and read at `t` gives `⟨C, (Pₜ ⊗ P_τ*) ⊗ B⟩`,
/// and `Pₜ ⊗ P_τ*` is the group word since the write. If the block writes the
/// identity at every reset, the readout is the four components of the word
/// itself, which is the task. `tests.rs` writes every weight down.
///
/// **What `micro_steps` changes.** At `u = 1`, a token gets one rotation. Its
/// generator `ϑ` is an affine map of the token, and a token is two one-hot
/// slots, so `ϑ(a, b) = v_a + w_b`. Generators **add** where the group
/// **multiplies**. The alphabet has the hold symbol, so `v_i` must lie along
/// `±x̂` and `w_j` along `±ŷ` (the axes of `i` and `j`). Then `ϑ(i, j)` lies in
/// the `xy`-plane, and its `exp` can never be the `±k` that the token needs. At
/// `u = 2`, the token is two steps, and its transition is exactly the product
/// `exp(w_b) ⊗ exp(v_a)`.
///
/// These config choices are load-bearing:
///
/// - `state_rank = 4` is the smallest quaternion block, and `rope_fraction =
///   1.0` turns all of it: the group *is* the state.
/// - `d_model = 8` gives each slot four dimensions, the smallest number that
///   carries five symbols affinely. `tests.rs` puts the symbols on a regular
///   4-simplex of norm 2. So every token has RMS 1 (the pre-`RmsNorm` of the
///   layer does not change it), and every in-projection channel has a
///   closed-form weight. `per_head_dim = 2` then keeps `nheads = 4`, one head
///   per quaternion component. The construction uses the first value channel
///   of each head. Two heads are enough for the *readout*, and the five
///   `reset-*` rungs run at two heads. But here, the same schedule finds
///   nothing with two heads: 61 / 52 / 53% with the trapezoid, 46 / 30 / 33%
///   without it.
/// - The **trapezoid stays on** (the default
///   [`Trapezoid::HorizontalCarryOver`](burn_mamba::prelude::Trapezoid)). This
///   is the only rung that keeps it. The hand-built solution pins `λ ≈ 1` and
///   never uses the `β` tap, so `Trapezoid::None` looks free. It does reach
///   100% at the trained length, but it gets only 83% on the long column (96
///   tokens), and this rung reports that column. The five `reset-*` rungs are
///   tapless.
/// - `ignore_last_residual` zeroes the residual of the last layer, so
///   `out_proj` reads only the output of the block.
///
/// **The other dial of the example: `layers`.** `micro_steps = 2` is not the
/// only way to put two rotations in a token. A second *layer* also applies one,
/// and its generator reads the layer below, not the token. So the axis-pinning
/// argument of `tests.rs` does not apply to it. But a second layer cannot turn
/// the *same* state, because each layer carries its own state. So the word must
/// end in the state of the last layer, and the per-token rotation of that layer
/// is the product of the whole pair. The layer below must compute that product
/// as a **feature**: a function of the pair with no additive form, so a
/// bilinear one (its `C·B` term is bilinear). That is expressible, but a
/// learned product is approximate. `u = 2` gives the two rotations to the
/// recurrence instead, and stays exact. `examples/reset/README.md` has the
/// measured rows.
pub fn model_config(micro_steps: usize, layers: usize) -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 8, per_head_dim = 2 ⇒ nheads = 4 (one per
    // quaternion component). Each head has its own Δ, A and λ per micro-step,
    // and its own D.
    let mamba_block = Mamba3Config::new(D_MODEL)
        .with_state_rank(4) // one quaternion block: the group element itself
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
        // No final norm: the state is a unit quaternion, so the output of the
        // block is already O(1), and the head reads it directly.
        final_norm: false,
        n_real_layers: layers,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block,
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        // the last layer's residual: dropped, so the head sees only the state
        ignore_last_residual: true,
        residuals: ResidualsConfig::Standard,
        // No feed-forward interleave: these examples are mixer-only.
        mlp: None,
        untied: Vec::new(),
    }
}
