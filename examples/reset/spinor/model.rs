//! The model configuration for the reset-spinor example: one Mamba-3 block
//! whose state is a **single quaternion**. Its size makes sure that only its
//! non-abelian rotation can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{
    Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid,
};

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
/// since the write**: the group word itself, with the newest factor on the left.
/// That is the whole task, so the construction is an embedding, not an encoding:
///
/// - **`R` writes the identity.** `x(R) ≠ 0` and `A(R) ≈ −20` (so `ᾱ ≈ 0` erases
///   the old state). That leaves `h = B = 1`, the identity quaternion, recorded
///   at the current cumulative rotation of the sequence.
/// - **`i` and `j` only turn it.** `x(i) = x(j) = 0`, so nothing is written, and
///   `ᾱ ≈ 1` holds the state. Each adds its own half-turn quaternion to the
///   product. `i` and `j` are half-turns about orthogonal axes, and they do not
///   commute: `ij = k`, `ji = −k`.
/// - **Two heads read that element on a plane.** Both hold a copy of the same
///   quaternion (same `B`, same `ᾱ`, same rotation). They differ only in the `C`
///   from the per-head bias `c_bias_hmr`, so `nheads` here counts *readouts*.
///   Two are enough. Under `e_k ↦ (cos kπ/4, sin kπ/4)`, the eight elements of
///   `Q₈` become eight **directions** `45°` apart: distinct, equidistant and in
///   convex position. Head `h` reads the `h`-th coordinate of that map.
///   `(y₀, y₁) ∝` the direction of `q_rel`, and the eight-class head is a sector
///   decoder over them: still `logit g = ⟨·, g⟩`, but read on the plane, not on
///   the quaternion.
///
/// A half-turn is an ordinary interior point of the parameterisation, not a
/// limit. The block bounds one step to `rotation_range · π · Δ`, with a default
/// of `range = 2`. That is a full traverse of the rotation group per unit `Δ`.
/// For `SU(2)` that is every element, because its period is `4π` (`q` and `−q`
/// turn the state differently). So `i` is at `tanh(‖ϑ‖) = 1/2`, with a live
/// gradient. The bound is on the **magnitude** of the generator, so the axis is
/// exactly the direction of the projection. The axis is projected **per head**,
/// so the two heads could turn about two different axes. This construction sets
/// them equal.
///
/// Necessary config choices:
///
/// - `state_rank = 4` is the smallest quaternion block: one `SU(2)` factor.
///   `rope_fraction = 1.0` turns all of it. The group *is* the state.
/// - `per_head_dim = 1`, `expand = 1`, `d_model = 2` ⇒ `d_inner = 2`,
///   `nheads = 2`. That gives two readouts of the one state, which the eight-way
///   label needs. `d_inner = d_model`, so `out_proj` is the identity. Four heads
///   (two more copies of the state, which the head cannot use) cost 212
///   parameters against 134, and give nothing here. `reset-swap` gets the same
///   answer. `spinor-product` is the one rung that needs four heads.
/// - `d_model = 2`, the floor for a three-symbol alphabet. The pre-`RmsNorm` of
///   the layer sends a token to the unit sphere, so `d_model = 1` would leave
///   two distinguishable symbols. Three points of `ℝ²` are affinely
///   independent, so every in-projection channel can take any value on the three
///   symbols (a 3×3 solve, see `tests.rs`).
/// - `Trapezoid::None`: the construction pins `λ ≈ 1`, so the `β` tap is dead
///   weight. Switching it off is structural: no `λ` segment in the
///   in-projection, no tap slot in the cache, one SSD call instead of two.
/// - `ignore_last_residual` zeroes the residual of the single layer, so
///   `out_proj` reads the output of the block alone.
///
/// [`RotationKind::Complex2D`] keeps everything else the same, and is the
/// ablation of the example. The cumulative rotation becomes a `cumsum` of
/// angles, which is a function of the symbol *counts*. The counts cannot tell
/// `ij` from `ji`.
pub fn model_config(rotation: RotationKind) -> MambaLatentNetConfig {
    // d_inner = expand·d_model = 2, per_head_dim = 1 ⇒ nheads = 2 (one per plane
    // coordinate the head reads), each with its own Δ, A and D.
    let mamba_block = Mamba3Config::new(2)
        .with_state_rank(4) // one quaternion block — the group element itself
        .with_expand(1)
        .with_per_head_dim(1)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(1.0)
        .with_rotation(rotation)
        .with_trapezoid(Trapezoid::None) // no β tap: nothing here integrates
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
        untied: Vec::new(),
    }
}
