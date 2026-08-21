//! The model configuration for the reset-majority example — one Mamba-2 block
//! with two scalar states, sized so that nothing *but* the block can solve the
//! task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{Mamba2Config, MambaLatentNetConfig, ResidualsConfig};

/// A single Mamba-2 block, unrolled, is two data-dependent scalar recurrences:
///
/// ```ignore
/// Δₕ(u) = softplus(⟨aₕ, u⟩ + bₕ)        ᾱₕ(u) = exp(Δₕ(u)·Aₕ) ∈ (0, 1)
/// hₜ⁽ʰ⁾ = ᾱₕ(uₜ)·hₜ₋₁⁽ʰ⁾ + Δₕ(uₜ)·B(uₜ)·xₕ(uₜ)
/// yₜ⁽ʰ⁾ = C(uₜ)·hₜ⁽ʰ⁾ + Dₕ·xₕ(uₜ)
/// ```
///
/// with `x`, `B`, `C` each `silu(affine(uₜ))` and `Δ`, `A`, `D` **per head**.
/// The task is built around exactly that shape:
///
/// - **head 0 is the ballot box.** `Δ₀` reads the reset flag: near `0` on `±`
///   (so `ᾱ₀ ≈ 1` and `h₀` is an unweighted running sum) and large on `RESET`
///   (so `ᾱ₀ ≈ 0` wipes it). `x₀(±) = ±v`, `x₀(RESET) = 0`, `D₀ = 0`, so
///   `y₀ = C·h₀` is the running vote.
/// - **head 1 is a fixed reference.** `Δ₁ ≈ 0` always, so `h₁ ≈ 0` and
///   `y₁ = D₁·x₁ = c > 0`, a constant.
///
/// The gated RMSNorm then keeps only the **direction** of `(y₀, y₁)`. That is
/// exactly the right shape for the task: the direction's sign is the answer, and
/// the reference axis keeps it well defined (and the margin proportional to the
/// vote) when `y₀` is near zero. A bounded, sign-like output is all a Mamba-2
/// block *can* emit at this width, which is why the task is a classification and
/// not a regression.
///
/// Two config choices are load-bearing:
///
/// - `conv_kernel = 1` removes the short convolution, so the SSM state is the
///   model's only memory — there is no local window to shortcut through.
/// - `ignore_last_residual` zeroes the single layer's residual, so `out_proj`
///   reads the block's output *alone*. Without it the head also sees the
///   embedding of the current token, which cannot give the answer but does
///   muddy the claim.
///
/// `d_model = 2` (rather than 1) is what keeps this constructible in closed
/// form: with a 2-D token every projection is an independent affine functional,
/// so `Δ` can read the reset flag while `x` reads the vote. At `d_model = 1`
/// they are all monotone functions of the same scalar and the same solution
/// needs `Δ` ratios around `10¹²`.
pub fn model_config() -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 2, per_head_dim = 1 ⇒ nheads = 2 (one head for
    // the vote, one for the reference), each with its own Δ, A and D.
    // state_rank = 1 ⇒ each head's state is a single scalar.
    let mamba_block = Mamba2Config::new(2)
        .with_state_rank(1)
        .with_conv_kernel(1) // no conv: the SSM state is the only memory
        .with_expand(1)
        .with_per_head_dim(1)
        .with_ngroups(1)
        .with_has_proj_bias(true);

    // input  [batch, seq, NUM_SYMBOLS]  (one-hot symbol)
    // output [batch, seq, NUM_CLASSES]  (Neg / Pos logits, every scored position)
    MambaLatentNetConfig::Mamba2 {
        input_size: NUM_SYMBOLS,
        output_size: NUM_CLASSES,
        // no final norm: the block's output is already O(1) (the gated norm
        // bounds it) and the head reads it directly.
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
