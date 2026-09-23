//! The model configuration for the reset-swap example: one Mamba-3 block whose
//! state is a **single quaternion block**. Its size makes sure that only a
//! two-sided (`SO(4)`) rotation can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{
    Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid,
};

/// A Mamba-3 block at `state_rank = 4` carries, per head, a cumulative rotation
/// of one 4-block, built by an ordered product. The available rotations are the
/// whole subject of this example:
///
/// ```ignore
/// Quaternion4D:  vₜ ↦ qₜ ⊗ v           (left-isoclinic, SU(2))
/// Rotor4D:       vₜ ↦ qₜ ⊗ v ⊗ p̄ₜ      (the full SO(4))
/// ```
///
/// With `p = q`, the second one is **conjugation**, `v ↦ q v q̄`. It fixes the
/// real axis and acts on the imaginary 3-space as `SO(3)`, and `S₃` lives in
/// `SO(3)`:
///
/// - A transposition has order 2, so it must be a **half-turn**.
/// - Two half-turns about axes `60°` apart compose to a `120°` rotation, so
///   `s∘t` has order 3. Two axes at `60°` generate the group correctly.
/// - `±q` conjugate identically, so the double cover collapses. The state *is*
///   the permutation, and a linear head can read it.
///
/// Under [`RotationKind::Quaternion4D`], none of that is available:
///
/// - A half-turn lifts to the pure quaternion `(0, û)`, whose **square is `−1`,
///   not `1`**. The accumulated state runs in the binary dihedral group `2D₃` of
///   order 12. The two lifts `±W` of one permutation are antipodal vectors with
///   the same label, and no linear readout merges them.
/// - There is no alternative. Every finite subgroup of `SU(2)` has a single
///   element of order two. So the three transpositions can go only to `±1`,
///   which is the sign character and no more.
///
/// [`RotationKind::Complex2D`] loses even that: a `cumsum` of angles is a
/// function of the symbol counts, and `st ≠ ts`.
///
/// Necessary config choices:
///
/// - `state_rank = 4` is one quaternion block: the smallest state that can hold
///   a rotation of this kind. `rope_fraction = 1.0` turns all of it.
/// - `per_head_dim = 1`, `expand = 1`, `d_model = 2` ⇒ `d_inner = 2`,
///   `nheads = 2`. That gives two readouts of the one state, which the six-way
///   label needs. `d_inner = d_model`, so `out_proj` is the identity. Both heads
///   hold the same rotated vector, and differ only in the `C` that the per-head
///   bias aims at it. So `nheads` counts projections, not state.
/// - `d_model = 2`, the floor for a three-symbol alphabet. The pre-`RmsNorm` of
///   the layer sends a token to the unit sphere, so `d_model = 1` would leave two
///   distinguishable symbols. Three points of `ℝ²` are affinely independent (a
///   3×3 solve, see `tests.rs`). The two heads read the `x` and `y` of the
///   rotation. There the six orbit points are on one circle, in six distinct
///   directions, so the six-way head is a nearest-point decoder. (Conjugation
///   fixes the real component, and a half-turn about an `xy` axis only flips
///   `z`. So the other two components carry nothing that the counts do not
///   already give.)
/// - `Trapezoid::None`: the construction pins `λ ≈ 1`, so the `β` tap is dead
///   weight. Switching it off is structural: no `λ` segment in the
///   in-projection, no tap slot in the cache, one SSD call instead of two.
/// - `ignore_last_residual` zeroes the residual of the single layer, so
///   `out_proj` reads the output of the block alone.
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
    // output [batch, seq, NUM_CLASSES]  (one logit per permutation, every position)
    MambaLatentNetConfig::Mamba3 {
        input_size: NUM_SYMBOLS,
        output_size: NUM_CLASSES,
        // no final norm, and this is load-bearing: a norm beside a constant is
        // even in the state, which merges the two lifts `±W` and lets
        // `Quaternion4D` solve the task too (`left_isoclinic_with_final_norm`).
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
