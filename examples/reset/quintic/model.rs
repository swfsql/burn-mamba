//! The model configuration for the reset-quintic example. The state of each
//! Mamba-3 head is a **single quaternion block**. `A₅` uses one layer, and `S₅`
//! uses two (see [`model_config`]).

use crate::dataset::{Group, NUM_SYMBOLS};
use burn_mamba::prelude::{
    Mamba3Config, MambaLatentNetConfig, ResidualsConfig, RotationKind, Trapezoid,
};

/// The default depth for a group: the fewest layers that can hold it.
///
/// - `A₅` is a group of rotations of `ℝ³`. Conjugation `v ↦ q v q̄` puts `SO(3)`
///   inside one `Rotor4D` 4-block, so **one** layer is enough.
/// - `S₅` is not a quotient of any group of block rotations. Its odd elements
///   act on `A₅` by an automorphism that changes rotation angles, and that
///   needs a reflection. So the transition of a layer follows `S₅` at most as
///   far as its sign. **Two** layers: the first holds the sign. The second
///   reads the sign and holds the even part `a` of `σ = sᵉ ∘ a`. The per-token
///   step `sᵉ'∘g∘sᵉ` of `a` depends on the token *and* on the sign below.
pub fn default_layers(group: Group) -> usize {
    match group {
        Group::Alternating => 1,
        Group::Symmetric => 2,
    }
}

/// The width at which a model is **trained**: `(d_model, nheads, mimo_rank)`.
/// So two heads of two channels each, and every channel reads the state of its
/// head at two ranks.
///
/// This is wider than the [`floor_width`] of the hand-built solutions, because
/// of where training puts the group. Every head carries its own rotation. A
/// trained block finds the icosahedral rotation (`180°` on one generator,
/// `120°` on the other, the axes `20.9°` or `69.1°` apart) on **one head at a
/// time**. The other heads find something smaller, a `Z₃` or a recency trace.
/// A one-channel head then gives the class head a single projection of the
/// group state. In the floor construction, every head reads the *same* state.
/// MIMO instead gives the head that finds the group its own readouts:
/// `per_head_dim` channels, each a different mix of the `mimo_rank`
/// projections of one state under one transition. The second head leaves room
/// for a head that misses the group.
pub const DEFAULT_WIDTH: (usize, usize, usize) = (4, 2, 2);

/// The narrowest width at which an exact solution exists: `(d_model, nheads)`
/// with `mimo_rank = 1`. `tests.rs` builds these by hand.
///
/// - `A₅`: `(2, 2)`, the same as `reset-swap`. `d_model = 2` is the floor for a
///   three-symbol alphabet (the pre-`RmsNorm` puts a 1-D token on `±1`). Two
///   readouts are enough: the shadow of the orbit on the `xy`-plane points in
///   sixty distinct directions. A bias-free head decodes them at any scale of
///   the state.
/// - `S₅`: `(3, 3)`. The input of the second layer must tell four situations
///   apart: `s`, `c` at either sign, and `R`. Each in-projection channel is one
///   affine function of it. The write must be zero except on `R`. The turn must
///   be zero on `s` and take two non-parallel values on the two `c`s. Four
///   points of `ℝ²` are affinely dependent, and those three constraints leave
///   the dependency nowhere to live. The third head then has a use on the
///   output side: it tracks the sign in the second layer, so the first layer
///   must send the sign only on `c`.
pub fn floor_width(group: Group) -> (usize, usize) {
    match group {
        Group::Alternating => (2, 2),
        Group::Symmetric => (3, 3),
    }
}

/// The trained configuration: [`default_layers`] Mamba-3 layers at
/// [`DEFAULT_WIDTH`].
///
/// ```ignore
/// Quaternion4D:  vₜ ↦ qₜ ⊗ v           (left-isoclinic, SU(2))
/// Rotor4D:       vₜ ↦ qₜ ⊗ v ⊗ p̄ₜ      (the full SO(4))
/// ```
///
/// With `p = q`, the second is **conjugation**. It fixes the real axis and
/// turns the imaginary 3-space by `SO(3)`, where the rotations of the
/// icosahedron (`A₅`) live. Training does not always find that solution. It
/// also finds the *four*-dimensional representation of `A₅`: `q` and `p` run
/// in the two Galois-twin copies of the binary icosahedral group (axes `69.1°`
/// apart on one side, `20.9°` on the other). Both are `SO(4)`, and neither is
/// `SU(2)`.
///
/// These config choices are load-bearing:
///
/// - `state_rank = 4`, `rope_fraction = 1.0`: one quaternion block, turned
///   whole. It is the smallest non-abelian state. `A₅` is perfect, so an
///   abelian state carries none of it.
/// - `layers`: the one knob that separates the two groups.
/// - `Trapezoid::None`: every construction pins `λ ≈ 1`.
/// - `final_norm: false`, `ignore_last_residual`: the head sees only the output
///   of the last block, through nothing but an affine map.
pub fn model_config(group: Group, rotation: RotationKind, layers: usize) -> MambaLatentNetConfig {
    let (d_model, nheads, mimo_rank) = DEFAULT_WIDTH;
    model_config_with(group, rotation, layers, d_model, 1, nheads, mimo_rank)
}

/// [`model_config`] at an explicit width: `nheads` heads of
/// `expand·d_model/nheads` channels each, each read at `mimo_rank` ranks.
pub fn model_config_with(
    group: Group,
    rotation: RotationKind,
    layers: usize,
    d_model: usize,
    expand: usize,
    nheads: usize,
    mimo_rank: usize,
) -> MambaLatentNetConfig {
    let d_inner = expand * d_model;
    assert_eq!(d_inner % nheads, 0, "d_inner must split evenly into heads");
    let mamba_block = Mamba3Config::new(d_model)
        .with_state_rank(4) // one quaternion block
        .with_expand(expand)
        .with_per_head_dim(d_inner / nheads)
        .with_ngroups(1)
        .with_mimo_rank(mimo_rank)
        .with_rope_fraction(1.0)
        .with_rotation(rotation)
        .with_trapezoid(Trapezoid::None)
        .with_has_proj_bias(true);

    // input  [batch, seq, NUM_SYMBOLS]  (one-hot symbol)
    // output [batch, seq, num_classes]  (one logit per arrangement, every position)
    MambaLatentNetConfig::Mamba3 {
        input_size: NUM_SYMBOLS,
        output_size: group.num_classes(),
        final_norm: false,
        n_real_layers: layers,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block,
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        // the last layer's residual: dropped, so the head sees only its state
        ignore_last_residual: true,
        residuals: ResidualsConfig::Standard,
        mlp: None,
        untied: Vec::new(),
    }
}
