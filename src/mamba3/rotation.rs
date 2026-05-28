//! # Quaternion (k=4) rotational state — the non-abelian generalisation of RoPE
//!
//! Mamba-3's data-dependent RoPE realises a **complex-valued** SSM: the state
//! transition factors as a per-head scalar decay times a block-diagonal of
//! `2×2` rotations (paper Prop. *Complex-to-Real SSM Equivalence*), and because
//! `SO(2) ≅ U(1)` is **abelian** the cumulative rotation collapses to a
//! `cumsum` of angles and is absorbed into `B`/`C` (the "RoPE trick", Prop.
//! *Complex SSM, Data-Dependent RoPE Equivalence*).  See
//! [`crate::mamba3::double_ssd::double_ssd::apply_rope`].
//!
//! This module implements the next rung of the ladder: a **quaternion**
//! (`k = 4`) rotational state, i.e. the transition's rotation lives in the
//! left-isoclinic subgroup `SU(2) ⊂ SO(4)` instead of `SO(2)`.  Unit
//! quaternions under multiplication are `SU(2)`, which is **non-abelian** and
//! contains non-solvable finite subgroups (the binary icosahedral group
//! `2I = SL(2,5)`, a double cover of `A₅`).  By Barrington's theorem this lifts
//! the layer's reachable state-tracking from the solvable/`TC⁰` regime (parity,
//! mod-k) toward `NC¹`, which abelian rotations provably cannot reach.
//!
//! ## What survives, what changes
//!
//! The key fact (derivable purely from telescoping + orthogonality, **without**
//! commutativity — see the crate discussion) is that the RoPE *factoring*
//! survives intact: with the **ordered** cumulative rotation
//! `Pₜ = Rₜ Rₜ₋₁ ⋯ R₁`,
//!
//! ```text
//!   Cₜᵀ (Rₜ⋯Rᵢ₊₁) Bᵢ  =  (Pₜᵀ Cₜ)ᵀ (Pᵢᵀ Bᵢ)  =  C̄ₜᵀ B̄ᵢ ,
//! ```
//!
//! so the scalar-decay SSD core (`L ⊙ C̄B̄ᵀ`) is **unchanged** — only the
//! projections `B̄ᵢ = Pᵢᵀ Bᵢ`, `C̄ₜ = Pₜᵀ Cₜ` are rotated.  What is lost is the
//! closed-form `cumsum`: the cumulative rotation must be built by an
//! **associative scan over the per-step quaternions** ([`quat_cumprod`]) rather
//! than a sum of angles.  Because a product of unit quaternions is again a unit
//! quaternion, the scan stays exactly orthogonal (no drift, no `wrap_angle`
//! needed), and the cross-chunk carry is a single quaternion per block/head —
//! the exact analogue of `cum_angle` in the existing caches.
//!
//! `SO(2)` (today's `apply_rope`) is the abelian collapse: restricting each
//! quaternion to a single fixed axis makes them commute and reduces
//! [`quat_cumprod`] to a `cumsum` of half-angles (asserted in the tests).
//!
//! ## Pipeline (the `k = 4` instantiation of the rotation block)
//!
//! ```text
//!   per-step unit quaternion qₜ      (materialise from the in-projection; caller)
//!        │  quat_cumprod (assoc. scan, + cross-chunk carry)
//!        ▼
//!   cumulative rotation Qₜ
//!        │  rotate_state_rank_blocks(B, conj(Qₜ)) , rotate_state_rank_blocks(C, conj(Qₜ))
//!        ▼
//!   B̄, C̄  ──►  standard scalar-decay SSD  (unchanged)
//! ```
//!
//! Quaternion layout: the last axis has size 4 and holds `(w, x, y, z)` with
//! `w` the real part.  A `state_rank` of `r = 4·J` is treated as `J` independent
//! quaternion blocks; the rotation acts within each block, exactly as RoPE acts
//! within each `2`-pair.  This module is a self-contained, tested reference for
//! the math; wiring it into the [`Mamba3`](crate::mamba3::mamba3::Mamba3) block
//! is a separate, larger change (the SSD kernels themselves need no edits).

use burn::prelude::*;

// ---------------------------------------------------------------------------
// Quaternion algebra on the trailing `(w, x, y, z)` axis
// ---------------------------------------------------------------------------

/// Hamilton product `a ⊗ b` of two quaternion tensors.
///
/// Both inputs have shape `[..., 4]` with the last axis ordered `(w, x, y, z)`;
/// the product is computed component-wise and broadcasts over the leading dims.
/// Quaternion multiplication is **non-commutative** (`a ⊗ b ≠ b ⊗ a` in
/// general) but associative.
///
/// Identifying `ℝ⁴` with the quaternions, left-multiplication `v ↦ a ⊗ v` is
/// exactly the action of the `4×4` rotation matrix [`quat_to_rot4`]`(a)`, so
/// this is also how a rotation is *applied* to a state/`B`/`C` block (see
/// [`rotate_state_rank_blocks`]).
pub fn quat_mul<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
    let n = D - 1;
    let aw = a.clone().narrow(n, 0, 1);
    let ax = a.clone().narrow(n, 1, 1);
    let ay = a.clone().narrow(n, 2, 1);
    let az = a.narrow(n, 3, 1);
    let bw = b.clone().narrow(n, 0, 1);
    let bx = b.clone().narrow(n, 1, 1);
    let by = b.clone().narrow(n, 2, 1);
    let bz = b.narrow(n, 3, 1);

    // Hamilton product (each term is shape [..., 1]).
    let w = aw.clone() * bw.clone() - ax.clone() * bx.clone() - ay.clone() * by.clone()
        - az.clone() * bz.clone();
    let x = aw.clone() * bx.clone() + ax.clone() * bw.clone() + ay.clone() * bz.clone()
        - az.clone() * by.clone();
    let y = aw.clone() * by.clone() - ax.clone() * bz.clone() + ay.clone() * bw.clone()
        + az.clone() * bx.clone();
    let z = aw * bz + ax * by - ay * bx + az * bw;

    Tensor::cat(vec![w, x, y, z], n)
}

/// Quaternion conjugate `q* = (w, −x, −y, −z)` (shape `[..., 4]`).
///
/// For a **unit** quaternion `q* = q⁻¹`, and the corresponding rotation matrix
/// satisfies `Lₚ⋆ = Lₚᵀ = Lₚ⁻¹`.  Hence rotating by the *inverse* cumulative
/// rotation (`B̄ = Pᵀ B`) is `rotate_state_rank_blocks(B, conj(Q))`.
pub fn quat_conj<const D: usize>(q: Tensor<D>) -> Tensor<D> {
    let n = D - 1;
    let w = q.clone().narrow(n, 0, 1);
    let xyz = q.narrow(n, 1, 3);
    Tensor::cat(vec![w, -xyz], n)
}

/// Normalise quaternions to unit norm along the last axis (shape `[..., 4]`).
///
/// The per-step rotation is materialised from a raw, unconstrained projection
/// and normalised here so it is a genuine unit quaternion (an element of
/// `SU(2)`), the analogue of `tanh(θ)·π` bounding the RoPE angle.  A tiny floor
/// guards the zero-quaternion.
pub fn quat_normalize<const D: usize>(q: Tensor<D>) -> Tensor<D> {
    let n = D - 1;
    let norm = (q.clone() * q.clone()).sum_dim(n).sqrt().clamp_min(1e-12);
    q / norm
}

/// Materialise the `4×4` orthogonal matrix of left-multiplication by `q`.
///
/// Maps `q` of shape `[..., 4]` to `[..., 4, 4]` such that, for `v` of shape
/// `[..., 4]`, `Lq · v == quat_mul(q, v)`.  Concretely (rows = output coords,
/// cols = input coords, all in `(w, x, y, z)` order):
///
/// ```text
///   ⎡ w  -x  -y  -z ⎤
///   ⎢ x   w  -z   y ⎥
///   ⎢ y   z   w  -x ⎥
///   ⎣ z  -y   x   w ⎦
/// ```
///
/// For a unit `q` this is orthogonal with `det = 1` (a left-isoclinic rotation).
/// Provided mainly for the generic / verification path; the cheap way to apply a
/// rotation is [`rotate_state_rank_blocks`] (a quaternion product, no `4×4`
/// materialisation).  `DR` must equal `D + 1`.
pub fn quat_to_rot4<const D: usize, const DR: usize>(q: Tensor<D>) -> Tensor<DR> {
    assert_eq!(D + 1, DR, "quat_to_rot4 maps rank D to rank D+1");
    let n = D - 1;
    let w = q.clone().narrow(n, 0, 1);
    let x = q.clone().narrow(n, 1, 1);
    let y = q.clone().narrow(n, 2, 1);
    let z = q.narrow(n, 3, 1);

    // Each row is a [..., 4] tensor (the four column entries).
    let row0 = Tensor::cat(vec![w.clone(), -x.clone(), -y.clone(), -z.clone()], n);
    let row1 = Tensor::cat(vec![x.clone(), w.clone(), -z.clone(), y.clone()], n);
    let row2 = Tensor::cat(vec![y.clone(), z.clone(), w.clone(), -x.clone()], n);
    let row3 = Tensor::cat(vec![z, -y, x, w], n);

    // Stack the rows along a freshly inserted row axis → [..., 4, 4].
    Tensor::cat(
        vec![
            row0.unsqueeze_dim::<DR>(n),
            row1.unsqueeze_dim::<DR>(n),
            row2.unsqueeze_dim::<DR>(n),
            row3.unsqueeze_dim::<DR>(n),
        ],
        n,
    )
}

// ---------------------------------------------------------------------------
// Rotation application on the state_rank axis
// ---------------------------------------------------------------------------

/// Apply a per-block quaternion rotation to the `state_rank` axis of `v`.
///
/// `v` has shape `[..., state_rank]` with `state_rank = 4·J`, viewed as `J`
/// independent quaternion blocks; `q` has shape `[..., J, 4]` (one unit
/// quaternion per block, same leading dims as `v`).  Returns `q ⊗ v` per block,
/// i.e. the rotation `L_q` applied within each `4`-block, reshaped back to
/// `[..., state_rank]`.
///
/// This is the generalisation of RoPE's per-pair `2×2` rotation to per-block
/// `4×4`.  To rotate by the *inverse* cumulative rotation when absorbing into
/// `B`/`C` (`B̄ = Pᵀ B`), pass `q = conj(Qcum)`:
/// `rotate_state_rank_blocks(b, conj(qcum))`.
///
/// `DB` must equal `D + 1` (the block-split inserts the `J` axis).
pub fn rotate_state_rank_blocks<const D: usize, const DB: usize>(
    v: Tensor<D>,
    q: Tensor<DB>,
) -> Tensor<D> {
    assert_eq!(D + 1, DB, "rotate_state_rank_blocks splits one axis into (J, 4)");
    let dims = v.dims();
    let state_rank = dims[D - 1];
    assert_eq!(state_rank % 4, 0, "state_rank must be a multiple of 4 (quaternion blocks)");
    let blocks = state_rank / 4;

    // Build the block-split shape [..., J, 4] (rank DB) and the flat shape
    // [..., state_rank] (rank D) for the round trip.
    let mut split_shape = [0usize; DB];
    split_shape[..D - 1].copy_from_slice(&dims[..D - 1]);
    split_shape[DB - 2] = blocks;
    split_shape[DB - 1] = 4;

    let v_blocks = v.reshape(split_shape); // [..., J, 4]
    let rotated = quat_mul(q, v_blocks); // L_q applied per block
    rotated.reshape(dims) // [..., state_rank]
}

// ---------------------------------------------------------------------------
// Cumulative rotation scan (the associative, non-abelian replacement for cumsum)
// ---------------------------------------------------------------------------

/// Cumulative (ordered, left-accumulating) quaternion product along the
/// sequence axis, with a cross-chunk carry.
///
/// This is the non-abelian analogue of the cumulative *sum of angles* used by
/// RoPE: where complex rotations compose by adding angles (a `cumsum`),
/// quaternions compose by multiplication, which is order-dependent, so a real
/// scan is required.
///
/// # Shapes
/// - `q_bshj4` : `[batch, sequence, nheads, J, 4]` per-step **unit** quaternions
///   (block count `J = state_rank / 4`).
/// - `init`    : optional carry `[batch, nheads, J, 4]` — the cumulative
///   rotation at the end of the previous chunk (identity `(1,0,0,0)` for a fresh
///   start).
/// - returns `(cum, final_carry)` where `cum` is `[batch, sequence, nheads, J, 4]`
///   with `cum[:, t] = qₜ ⊗ qₜ₋₁ ⊗ ⋯ ⊗ q₀ ⊗ init` (newest on the left, matching
///   `Pₜ = Rₜ ⋯ R₁`), and `final_carry` `[batch, nheads, J, 4]` is `cum[:, −1]`
///   to thread into the next chunk.
///
/// Running this over a split sequence while threading `final_carry` is exactly
/// equal to running it over the whole sequence (asserted in the tests) — the
/// chunked-prefill / streaming guarantee, here for the rotation accumulator.
///
/// Implemented as a readable sequential scan (the reference); an `O(log T)`
/// associative parallel scan computes the same values.
pub fn quat_cumprod(
    q_bshj4: Tensor<5>,
    init: Option<Tensor<4>>,
) -> (Tensor<5>, Tensor<4>) {
    let [batch, sequence, nheads, blocks, _four] = q_bshj4.dims();
    let device = q_bshj4.device();

    // Carry as [batch, nheads, J, 4]; default = identity quaternion (1,0,0,0).
    let carry0 = init.unwrap_or_else(|| {
        let w = Tensor::ones([batch, nheads, blocks, 1], &device);
        let xyz = Tensor::zeros([batch, nheads, blocks, 3], &device);
        Tensor::cat(vec![w, xyz], 3)
    });
    assert_eq!([batch, nheads, blocks, 4], carry0.dims());

    // Promote the carry to a sequence slice [batch, 1, nheads, J, 4].
    let mut carry = carry0.unsqueeze_dim::<5>(1);
    let mut cums: Vec<Tensor<5>> = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let q_t = q_bshj4.clone().narrow(1, t, 1); // [batch, 1, nheads, J, 4]
        let cur = quat_mul(q_t, carry); // qₜ ⊗ (running product)
        carry = cur.clone();
        cums.push(cur);
    }

    let cum = Tensor::cat(cums, 1); // [batch, sequence, nheads, J, 4]
    let final_carry = carry.squeeze_dim::<4>(1); // [batch, nheads, J, 4]
    (cum, final_carry)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
