//! The model configuration for the reset-swap example — one Mamba-3 block whose
//! state is a **single quaternion block**, sized so that nothing but a two-sided
//! (`SO(4)`) rotation can solve the task (see [`model_config`]).

use crate::dataset::{NUM_CLASSES, NUM_SYMBOLS};
use burn_mamba::prelude::{
    Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid,
};

/// A Mamba-3 block at `state_rank = 4` carries, per head, a cumulative rotation
/// of one 4-block, built by an ordered product. Which rotations are available is
/// the whole subject of this example:
///
/// ```ignore
/// Quaternion4D:  vₜ ↦ qₜ ⊗ v           (left-isoclinic, SU(2))
/// Rotor4D:       vₜ ↦ qₜ ⊗ v ⊗ p̄ₜ      (the full SO(4))
/// ```
///
/// Setting `p = q` makes the second one **conjugation**, `v ↦ q v q̄`, which
/// fixes the real axis and acts on the imaginary 3-space as `SO(3)` — and
/// `SO(3)` is where `S₃` lives:
///
/// - a transposition has order 2, so it must be a **half-turn**;
/// - two half-turns about axes `60°` apart compose to a `120°` rotation, so
///   `s∘t` has order 3 — the group is generated correctly by two axes at `60°`;
/// - `±q` conjugate identically, so the double cover collapses and the state
///   *is* the permutation, readable by a linear head.
///
/// Under [`RotationKind::Quaternion4D`] none of that is available. A half-turn
/// lifts to the pure quaternion `(0, û)`, whose **square is `−1`, not `1`**: the
/// accumulated state runs in the binary dihedral group `2D₃` of order 12, and
/// the two lifts `±W` of one permutation are antipodal vectors carrying the same
/// label. No linear readout merges them, and there is no other choice available
/// — every finite subgroup of `SU(2)` has a single element of order two, so the
/// three transpositions have nowhere to go but `±1`, which is the sign character
/// and no more. [`RotationKind::Complex2D`] loses even that: a `cumsum` of
/// angles is a function of the symbol counts, and `st ≠ ts`.
///
/// Config choices that are load-bearing:
///
/// - `state_rank = 4` is one quaternion block — the smallest state that can hold
///   a rotation of this kind, and `rope_fraction = 1.0` turns all of it.
/// - `per_head_dim = 1`, `expand = 1`, `d_model = 2` ⇒ `d_inner = 2`,
///   `nheads = 2`: two readouts of the one state, which is what the six-way
///   label needs, and `d_inner = d_model` so `out_proj` is the identity. Both
///   heads hold the same rotated vector and differ only in the `C` the per-head
///   bias aims at it, so `nheads` counts projections rather than state.
/// - `d_model = 2`, the floor for a three-symbol alphabet: the layer's
///   pre-`RmsNorm` sends a token to the unit sphere, so `d_model = 1` would
///   leave two distinguishable symbols, and three points of `ℝ²` are already
///   affinely independent (a 3×3 solve; see `tests.rs`). The two heads read the
///   rotation's `x` and `y`, where the six orbit points sit on one circle in six
///   distinct directions, so the six-way head is a nearest-point decoder.
///   (Conjugation fixes the real component and a half-turn about an `xy` axis
///   only flips `z`, so neither of the other two components carries anything the
///   counts do not already give.)
/// - `Trapezoid::None`: the construction pins `λ ≈ 1`, so the `β` tap is dead
///   weight, and switching it off is structural — no `λ` segment in the
///   in-projection, no tap slot in the cache, one SSD call instead of two.
/// - `ignore_last_residual` zeroes the single layer's residual, so `out_proj`
///   reads the block's output alone.
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
        // no final norm: the state is a rotated fixed vector, so the block's
        // output is already O(1) and the head reads it directly.
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
