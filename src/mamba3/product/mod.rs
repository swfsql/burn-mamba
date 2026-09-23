//! # MambaProduct — `u` micro-steps per token
//!
//! `u` full Mamba-3 recurrence steps per token. Each step has its own projected
//! `x`, `B`, `Δ`, `A`, `λ` and rotation, and the block reads the token at the
//! last step.
//! [`Mamba3Config::micro_steps`](crate::mamba3::mamba3::Mamba3Config::micro_steps)
//! (`u`) selects it. `u = 1` is stock Mamba-3, byte for byte.
//!
//! ## The dial, and whose it is
//!
//! The *dial* comes from DeltaProduct (*Improving State-Tracking in Linear
//! RNNs via Householder Products*, Siems, Carstensen, Zela, Hutter, Pontil,
//! Grazzi, 2025): `u` first-order steps per token. It buys transition
//! expressiveness for `u`× the recurrence work and **no** extra state. The
//! *mechanism* is different, and the difference decides everything below.
//!
//! Write both as `u` steps of an online learner, `Mₜ = ∏ⱼ (I − ηⱼ ∇²Lⱼ)`. The
//! two families turn different dials in it:
//!
//! - **DeltaProduct turns the curvature.** Its `∇²Lⱼ = kⱼkⱼᵀ` is rank-one and
//!   depends on the direction. So factors with different `kⱼ` do not commute,
//!   and the product leaves the real axis (Cartan–Dieudonné). The condition in
//!   the paper is that *both* `βⱼ > 1`: both micro-steps overshoot.
//! - **Mamba-3 turns the step size.** Its curvature is isotropic (`∇²L = ρI`,
//!   the scalar transition), so every factor is a *scalar*, and scalars
//!   commute. No number of micro-steps can rotate a Mamba transition. The
//!   rotation must come from `ηⱼ` leaving `ℝ`, which is exactly what
//!   [`RotationKind`](crate::mamba3::rotation::RotationKind) is.
//!
//! So the mechanism of DeltaProduct has no instance here. With isotropic
//! curvature, `u` micro-writes under a shared transition provably collapse into
//! a rank-`u` write, which is
//! [`mimo_rank`](crate::mamba3::mamba3::Mamba3Config::mimo_rank). This module
//! reaches the same construction by the other dial.
//! `info/mamba-3/rotation-as-optimization.md` derives the above and the table
//! below.
//!
//! The transition of Mamba-3 is `αₜ Rₜ` (§*Complex-Valued SSMs*): a scalar
//! decay times a rotation. With `u` micro-steps per token, the per-token
//! transition is
//!
//! ```text
//!   Mₜ = (∏ⱼ αₜ,ⱼ) · Rₜ,ᵤ ⋯ Rₜ,₁
//! ```
//!
//! and the write is a sum of `u` outer products, each at its own point of that
//! rotation product. So what `u` buys depends only on the
//! [`RotationKind`](crate::mamba3::rotation::RotationKind):
//!
//! | kind | the product `Rₜ,ᵤ⋯Rₜ,₁` | what `u > 1` buys |
//! |---|---|---|
//! | [`Real1D`] | `I` | the write only: `u` staggered rank-1 writes per token |
//! | [`Complex2D`] | `R(Σⱼ θⱼ)`, abelian | `u`× the per-token **angle reach**, plus the staggered write |
//! | [`Quaternion4D`] / [`Rotor4D`] | a non-commuting product | a per-token transition that **no single step can express** |
//!
//! [`Real1D`]: crate::mamba3::rotation::RotationKind::Real1D
//! [`Complex2D`]: crate::mamba3::rotation::RotationKind::Complex2D
//! [`Quaternion4D`]: crate::mamba3::rotation::RotationKind::Quaternion4D
//! [`Rotor4D`]: crate::mamba3::rotation::RotationKind::Rotor4D
//!
//! Three consequences decide when the dial is worth a turn:
//!
//! - **On a real transition, it gives only a wider write, not a new
//!   transition.** With [`Real1D`] the step size is real too. So every factor
//!   is a *real* scalar, the scalars commute, and the `u` micro-writes collapse
//!   into one rank-`u` update with decay-staggered weights. That is the
//!   *sequential* reading of the cell that
//!   [`mimo_rank`](crate::mamba3::mamba3::Mamba3Config::mimo_rank) fills
//!   *jointly*: an epoch of `u` samples against a minibatch of `M`. It is also
//!   why this dial is on Mamba-3 and not on Mamba-2, whose transition is
//!   scalar by construction: there `u` would buy nothing. The containment is
//!   exact and one-way. `MambaProduct(u = M)` reproduces a whole `MIMO(M)`
//!   trajectory. The converse fails by a closed-form dimension count, because
//!   `M` ties its values, step sizes and rotation where `u` leaves them free
//!   (`info/mamba-3/mimo-as-batch.md` §9). What `M` gives back is cost: it is
//!   parallel, and its state bytes stay flat.
//! - **On the abelian rotation, it lifts a bound that the block cannot
//!   otherwise cross.** One step turns by at most
//!   [`rotation_range`](crate::mamba3::mamba3::Mamba3Config::rotation_range)`·π·Δ`.
//!   A rotation *at* that bound is on the asymptote of `tanh`, where the f32
//!   gradient is exactly zero. So at `rotation_range = 1`, descent cannot reach
//!   the half-turn that state-tracking wants. `u` micro-rotations, each well
//!   inside the bound, compose to `u`× the reach with live gradients at every
//!   factor. Note the price: the token gets `u` *full-size* steps, so its
//!   effective interval is `u`× longer, not subdivided. The consistent
//!   alternative (`Δⱼ = Δ/u`, the same per-token transition, only the
//!   staggered writes and the ordering below) stays inside the reach of the
//!   model, because `dt_limit` has no lower floor by default. So this is a
//!   superset of it, not a substitute.
//! - **On the non-abelian rotations, it is the argument of DeltaProduct
//!   reached by the other dial.** The factors do not commute, so the product
//!   is not a single bounded step. `u` is the number of generators that the
//!   token can compose. The group is given directly, not factored into
//!   reflections.
//!
//! ## How it is evaluated
//!
//! Not with a new kernel: the micro-steps fold into the **sequence**. The
//! in-projection widens exactly the per-micro-step segments,
//!
//! ```text
//!   [ z | x·u | B·u | C | Δ·u | A·u | λ·u | μ·u | rotation·u | r·u | a·u | b·u ]
//! ```
//!
//! and [`unfold_micro_bs`] reads each `u`-wide segment as `u` consecutive
//! sequence positions. So the SSD core, the trapezoid, the rotation scan, the
//! chunking and the caches all run unchanged on a sequence of length
//! `sequence · u`. Two placements make that exactly the recurrence above:
//!
//! - **The read `C` is on the last micro-step, and only there.** Every
//!   micro-step *writes* to the state, so `B`, `x`, `Δ`, `A`, `λ` and the
//!   rotation are on the folded axis. The token is *read* once, so `C` stays at
//!   token resolution, and so does the `y` that the SSD returns. That is the
//!   **read axis** of the chunk (`helpers::read_rows`). `u` widens only the
//!   write axis of a chunk. So the intra-chunk score, the state-to-output
//!   product and the QK-norm/rotation of `C` are `u`-invariant. The block does
//!   not compute `u`× the output to discard all rows but one.
//! - **`z`, the `D` skip and the output gate are per token.** The skip takes
//!   the `x` of the last micro-step, which is contemporaneous with the readout
//!   ([`last_micro4`]).
//!
//! `step` folds the same way, over a block of `u` positions (not a sequence),
//! and evaluates that block **at once**. With the rotation factored out, the
//! transition inside a token is the scalar `α`. So the block has the closed
//! form `h = (∏ⱼ αⱼ)·h₋₁ + Σⱼ wⱼ·writeⱼ`, and every write in it is an outer
//! product into one shared state. The transport by `wⱼ = ∏_{r>j} αᵣ`, and the
//! fusion of `(u, mimo_rank)` into one contracted axis, make each side of the
//! recurrence one `mimo_outer_sum`. So a decode step costs the same number of
//! kernel launches at every `u`.
//!
//! Everything else is per micro-step, including the decay. Mamba has no forget
//! gate separate from its step size: `α = exp(ΔA)`, and `Δ` also weights the
//! write and paces the rotation. So the "forget gate on micro-step 0" of
//! DeltaProduct has no faithful analogue here. `α ≡ 1` on the interior
//! micro-steps would also stop the rotation. A scalar decay composes either way
//! (`∏ⱼ αₜ,ⱼ` is one decay per token), so the uniform placement costs nothing
//! and keeps every micro-step a plain Mamba-3 step.
//!
//! ## Caches, and what does *not* change
//!
//! The state is one `[batch, nheads, per_head_dim, state_rank]` matrix at
//! every `u`. Like DeltaProduct, this buys transition expressiveness, not
//! memory. The rotation accumulator holds the **last micro-step** of the last
//! token, which is the position just before the first micro-step of the next
//! call. So a chunked prefill splits at token boundaries, as at `u = 1`.
//!
//! What `u > 1` does change is the taps of the trapezoid. They are now on the
//! folded chain, so they become a *choice*
//! ([`Trapezoid`](crate::mamba3::trapezoid::Trapezoid)). The default
//! `HorizontalCarryOver` reads the previous **micro-step**: the 2-tap FIR
//! filter runs at the finer rate, and only `1/u` of the taps still cross a
//! token. `Vertical` reads the same micro-step of the previous **token**. The
//! tap cache (`k_state`, `v_state`) is the FIFO that the choice needs: one
//! slot, or `u`. Either way, `forward` runs the same folded chain as `step`,
//! which keeps them in exact agreement.
//!
//! ## Notation
//!
//! `u` is `micro_steps`. In the folded region of `forward`, the shape letter
//! `s` counts **micro-steps**, not tokens. A name at token resolution says so
//! (`tokens`). See the [`mamba3`](crate::mamba3::mamba3) module header for the
//! other dimension keys.

use burn::prelude::*;

/// Fold a `u`-wide in-projection segment into the sequence axis.
///
/// The projection lays the micro-steps of a token out contiguously
/// (`channel = j·width + c`), which is already the memory order of
/// `[batch, sequence, u, width]`. So the fold is one reshape and never moves
/// data.
///
/// # Shapes
/// - `t_bsW` : `[batch, sequence, u · width]`
/// - out     : `[batch, sequence · u, width]`
#[allow(non_snake_case)]
pub fn unfold_micro_bs(t_bsW: Tensor<3>, micro_steps: usize) -> Tensor<3> {
    let [batch, sequence, fused] = t_bsW.dims();
    assert_eq!(
        fused % micro_steps,
        0,
        "a per-micro-step segment is a multiple of micro_steps wide"
    );
    t_bsW.reshape([batch, sequence * micro_steps, fused / micro_steps])
}

/// [`unfold_micro_bs`] for a single token: move the micro-steps onto their own
/// axis.
///
/// # Shapes
/// - `t_bW` : `[batch, u · width]`
/// - out    : `[batch, u, width]`
#[allow(non_snake_case)]
pub fn unfold_micro_b(t_bW: Tensor<2>, micro_steps: usize) -> Tensor<3> {
    let [batch, fused] = t_bW.dims();
    assert_eq!(
        fused % micro_steps,
        0,
        "a per-micro-step segment is a multiple of micro_steps wide"
    );
    t_bW.reshape([batch, micro_steps, fused / micro_steps])
}

/// Keep the **last** micro-step of each token of the value stream (the
/// position that is contemporaneous with the readout). This collapses the
/// folded sequence back to token resolution for the `D` skip.
///
/// The output `y` needs no such collapse. The SSD computes only the readout
/// row (the read axis of the chunk, see `helpers::read_rows`), so it already
/// returns token resolution.
///
/// # Shapes
/// - `t_bShp` : `[batch, sequence · u, nheads, per_head_dim]`
/// - out      : `[batch, sequence, nheads, per_head_dim]`
#[allow(non_snake_case)]
pub fn last_micro4(t_bShp: Tensor<4>, micro_steps: usize) -> Tensor<4> {
    let [batch, folded, nheads, per_head_dim] = t_bShp.dims();
    if micro_steps == 1 {
        return t_bShp;
    }
    let sequence = folded / micro_steps;
    t_bShp
        .reshape([batch, sequence, micro_steps, nheads, per_head_dim])
        .narrow(2, micro_steps - 1, 1)
        .squeeze_dim(2)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
