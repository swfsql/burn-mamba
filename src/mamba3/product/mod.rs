//! # MambaProduct — `u` micro-steps per token
//!
//! *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
//! Products* (Siems, Carstensen, Zela, Hutter, Pontil, Grazzi; 2025), carried
//! over to the Mamba-3 recurrence. Selected by
//! [`Mamba3Config::micro_steps`](crate::mamba3::mamba3::Mamba3Config::micro_steps)
//! (`u`); `u = 1` is stock Mamba-3, byte for byte.
//!
//! ## What DeltaProduct does, and what carries over
//!
//! DeltaNet's transition is one generalised Householder `I − βkkᵀ`, so one step
//! moves the state along one direction. DeltaProduct takes `u` delta-rule
//! micro-steps per token, making the transition a **product**
//! `∏ⱼ (I − βⱼkⱼkⱼᵀ)`, which by Cartan–Dieudonné reaches every element of the
//! orthogonal group of a `u`-dimensional subspace — a direct dial on how much
//! group structure one transition can track, at `u`× the recurrence work and no
//! extra state.
//!
//! Mamba-3's transition is `αₜ Rₜ` (§*Complex-Valued SSMs*): a scalar decay
//! times a rotation. Running `u` micro-steps per token makes the per-token
//! transition
//!
//! ```text
//!   Mₜ = (∏ⱼ αₜ,ⱼ) · Rₜ,ᵤ ⋯ Rₜ,₁
//! ```
//!
//! and the write a sum of `u` outer products, each landing at its own point of
//! that rotation product. What `u` buys therefore depends entirely on the
//! [`RotationKind`](crate::mamba3::rotation::RotationKind), and the answer is
//! sharp:
//!
//! | kind | the product `Rₜ,ᵤ⋯Rₜ,₁` | what `u > 1` buys |
//! |---|---|---|
//! | [`Real1D`] | `I` | the write only: `u` staggered rank-1 writes per token |
//! | [`Complex2D`] | `R(Σⱼ θⱼ)`, abelian | `u`× the per-token **angle reach**, plus the staggered write |
//! | [`Quaternion4D`] / [`Rotor4D`] | a genuinely non-commuting product | a per-token transition **no single step can express** |
//!
//! [`Real1D`]: crate::mamba3::rotation::RotationKind::Real1D
//! [`Complex2D`]: crate::mamba3::rotation::RotationKind::Complex2D
//! [`Quaternion4D`]: crate::mamba3::rotation::RotationKind::Quaternion4D
//! [`Rotor4D`]: crate::mamba3::rotation::RotationKind::Rotor4D
//!
//! Three consequences worth stating outright, because they decide when the dial
//! is worth turning:
//!
//! - **On a real transition it is not a new transition, only a wider write.**
//!   With [`Real1D`] every factor is a scalar, scalars commute, and the `u`
//!   micro-writes collapse into one rank-`u` update with decay-staggered
//!   weights. That is the *sequential* reading of the cell
//!   [`mimo_rank`](crate::mamba3::mamba3::Mamba3Config::mimo_rank) occupies
//!   *jointly* — an epoch of `u` samples versus a minibatch of `M` — and it is
//!   why this dial lives on Mamba-3 and not on Mamba-2, whose transition is
//!   scalar by construction and for which `u` would buy nothing at all.
//! - **On the abelian rotation it lifts a bound the block otherwise cannot
//!   cross.** One step turns by at most
//!   [`rotation_range`](crate::mamba3::mamba3::Mamba3Config::rotation_range)`·π·Δ`,
//!   and a rotation *at* that bound sits on `tanh`'s asymptote where the f32
//!   gradient is exactly zero — so the half-turn state-tracking wants is
//!   unreachable by descent at `rotation_range = 1`. `u` micro-rotations, each
//!   comfortably inside the bound, compose to `u`× the reach with live
//!   gradients at every factor.
//! - **On the non-abelian rotations it is DeltaProduct's own argument,
//!   verbatim.** The factors do not commute, so the product is not any single
//!   bounded step; `u` is the number of generators the token may compose.
//!
//! ## How it is evaluated
//!
//! Not with a new kernel — the micro-steps are folded into the **sequence**.
//! The in-projection widens exactly the per-micro-step segments,
//!
//! ```text
//!   [ z | x·u | B·u | C | Δ·u | A·u | λ·u | rotation·u ]
//! ```
//!
//! and [`unfold_micro_bs`] reinterprets each `u`-wide segment as `u` consecutive
//! sequence positions, so the SSD core, the trapezoid, the rotation scan, the
//! chunking and the caches all run unchanged on a sequence of length
//! `sequence · u`. Two placements make that exactly the recurrence above:
//!
//! - **The read `C` sits on the last micro-step.** It is
//!   [`repeat_micro_bs`]-broadcast across the group so the last position sees
//!   the correct cumulative rotation; the other positions' outputs are sliced
//!   away by [`last_micro5`], so what they hold never matters.
//! - **`z`, the `D` skip and the output gate are per token**, the skip taking
//!   the last micro-step's `x` — the value the readout is contemporaneous with.
//!
//! Everything else is per micro-step, including the decay. Mamba has no forget
//! gate separate from its step size — `α = exp(ΔA)` and `Δ` also weights the
//! write and paces the rotation — so DeltaProduct's "forget gate on micro-step
//! 0" has no faithful analogue here, and pinning `α ≡ 1` on the interior
//! micro-steps would silence the rotation with it. A scalar decay composes
//! either way (`∏ⱼ αₜ,ⱼ` is one decay per token), so the uniform placement costs
//! nothing and keeps every micro-step a plain Mamba-3 step.
//!
//! ## Caches, and what does *not* change
//!
//! Nothing. The state is one `[batch, nheads, per_head_dim, state_rank]` matrix
//! at every `u`: like DeltaProduct, this buys transition expressiveness, not
//! memory. The trapezoid's previous-token taps (`k_state`, `v_state`) and the
//! rotation accumulator hold the **last micro-step** of the last token, which is
//! precisely the position the next call's first micro-step follows — so a
//! chunked prefill splits at token boundaries exactly as it did before.
//!
//! One thing genuinely does move: the trapezoid's two taps now straddle
//! *micro-steps* rather than tokens, i.e. the 2-tap FIR filter on the state
//! input runs at the finer rate. That is what "the recurrence is the folded
//! sequence" means, and it is what keeps `forward` and `step` in exact
//! agreement.
//!
//! ## The one thing `u > 1` takes away
//!
//! [`Mamba3::step_infinite`](crate::mamba3::mamba3::Mamba3::step_infinite) —
//! the stationary output under a constant token — **exists only for the abelian
//! kinds** once
//! `u > 1`. The readout at token `t` sees a write from token `t − n` through the
//! relative rotation `P⁻ᵗ Qⱼ Pᵗ⁻ⁿ⁻¹`, where `P = Rᵤ⋯R₁` is the per-token product
//! and `Qⱼ = Rⱼ⋯R₁` the partial one. That collapses to a function of `n` alone
//! iff `Qⱼ` commutes with `P`. When it does not — [`Quaternion4D`],
//! [`Rotor4D`] — the conjugation `P⁻ᵗQⱼPᵗ` keeps turning with `t` and the output
//! is almost-periodic rather than convergent: there is no limit to return, which
//! is a fact about the recurrence and not a gap in the implementation.
//!
//! ## Notation
//!
//! `u` is `micro_steps`. In the folded region of `forward` the shape letter `s`
//! counts **micro-steps**, not tokens; a name still at token resolution says so
//! (`tokens`). See the [`mamba3`](crate::mamba3::mamba3) module header for the
//! rest of the dimension keys.

use burn::prelude::*;

/// Fold a `u`-wide in-projection segment into the sequence axis.
///
/// The projection lays a token's micro-steps out contiguously
/// (`channel = j·width + c`), which is already the memory order of
/// `[batch, sequence, u, width]` — so the fold is a single reshape and never
/// moves data.
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

/// [`unfold_micro_bs`] for a single token: split the micro-steps onto an axis of
/// their own.
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

/// Broadcast a **per-token** projection across the token's micro-steps, so it
/// can ride the folded sequence alongside the per-micro-step ones.
///
/// Used for the read `C`: only its last-micro-step copy is ever read (the
/// others' outputs are dropped by [`last_micro5`]), but that copy has to sit at
/// the last position to pick up the right cumulative rotation.
///
/// # Shapes
/// - `t_bsw` : `[batch, sequence, width]`
/// - out     : `[batch, sequence · u, width]`
pub fn repeat_micro_bs(t_bsw: Tensor<3>, micro_steps: usize) -> Tensor<3> {
    let [batch, sequence, width] = t_bsw.dims();
    if micro_steps == 1 {
        return t_bsw;
    }
    t_bsw
        .unsqueeze_dim::<4>(2)
        .repeat_dim(2, micro_steps)
        .reshape([batch, sequence * micro_steps, width])
}

/// Keep each token's **last** micro-step — the position the readout happens at —
/// collapsing the folded sequence back to token resolution.
///
/// # Shapes
/// - `t_bSmhp` : `[batch, sequence · u, mimo_rank, nheads, per_head_dim]`
/// - out       : `[batch, sequence, mimo_rank, nheads, per_head_dim]`
#[allow(non_snake_case)]
pub fn last_micro5(t_bSmhp: Tensor<5>, micro_steps: usize) -> Tensor<5> {
    let [batch, folded, mimo_rank, nheads, per_head_dim] = t_bSmhp.dims();
    if micro_steps == 1 {
        return t_bSmhp;
    }
    let sequence = folded / micro_steps;
    t_bSmhp
        .reshape([
            batch,
            sequence,
            micro_steps,
            mimo_rank,
            nheads,
            per_head_dim,
        ])
        .narrow(2, micro_steps - 1, 1)
        .squeeze_dim(2)
}

/// [`last_micro5`] one rank down — the value stream feeding the `D` skip.
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
