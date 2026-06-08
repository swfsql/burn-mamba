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

use crate::modules::{apply_rope_partial, wrap_angle};
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Rotation kind (config switch) and cache accumulator variant
// ---------------------------------------------------------------------------

/// Which rotational-state algebra the block uses for the data-dependent
/// positional rotation of `B`/`C`.
///
/// - [`Complex2D`](RotationKind::Complex2D) — the abelian `SO(2)`/complex RoPE
///   that Mamba-3 ships: cumulative *angles* via `cumsum`, applied by
///   [`apply_rope`]. The default; behaviourally unchanged.
/// - [`Quaternion4D`](RotationKind::Quaternion4D) — the non-abelian
///   `SU(2) ⊂ SO(4)` quaternion rotation of this module: cumulative *product*
///   via [`quat_cumprod`], applied by [`rotate_state_rank_blocks`]. Richer
///   state-tracking; selects the [`RotationState::Quaternion`] cache accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RotationKind {
    /// Abelian complex (`SO(2)`) RoPE — the current default behaviour.
    Complex2D,
    /// Non-abelian quaternion (`SU(2)`) rotation.
    Quaternion4D,
}

impl Default for RotationKind {
    fn default() -> Self {
        RotationKind::Complex2D
    }
}

/// The cumulative-rotation accumulator carried between calls in a Mamba-3 cache
/// — the variant matching the block's [`RotationKind`].
///
/// - [`Angle`](RotationState::Angle) — abelian per-pair cumulative RoPE angle,
///   shape `[batch, nheads, num_rope_angles]` (today's `cum_angle`).
/// - [`Quaternion`](RotationState::Quaternion) — per-block cumulative unit
///   quaternion, shape `[batch, nheads, blocks, 4]`, produced by
///   [`quat_cumprod`].
///
/// This is the cache-level counterpart of [`RotationKind`]. It is defined here
/// (the rotation module owns the accumulator type); substituting it for the
/// pathway caches' `cum_angle_bha` field happens together with the forward/step
/// wiring that consumes it.
#[derive(Module, Debug)]
pub enum RotationState {
    /// Abelian RoPE cumulative angle, shape `[batch, nheads, num_rope_angles]`.
    Angle(Tensor<3>),
    /// Quaternion cumulative rotation, shape `[batch, nheads, blocks, 4]`.
    Quaternion(Tensor<4>),
}

impl RotationState {
    /// Zero-initialised abelian angle accumulator `[batch, nheads, num_rope_angles]`.
    pub fn zeros_angle(
        batch: usize,
        nheads: usize,
        num_rope_angles: usize,
        device: &Device,
    ) -> Self {
        RotationState::Angle(Tensor::zeros([batch, nheads, num_rope_angles], device))
    }

    /// Identity-initialised quaternion accumulator `[batch, nheads, blocks, 4]`
    /// (every block is the identity quaternion `(1, 0, 0, 0)`).
    pub fn identity_quaternion(
        batch: usize,
        nheads: usize,
        blocks: usize,
        device: &Device,
    ) -> Self {
        let w = Tensor::ones([batch, nheads, blocks, 1], device);
        let xyz = Tensor::zeros([batch, nheads, blocks, 3], device);
        RotationState::Quaternion(Tensor::cat(vec![w, xyz], 3))
    }

    /// Unwrap the abelian angle accumulator; panics if this is a quaternion.
    pub fn angle(self) -> Tensor<3> {
        match self {
            RotationState::Angle(a) => a,
            RotationState::Quaternion(_) => {
                panic!("RotationState is Quaternion, expected Angle")
            }
        }
    }

    /// Unwrap the quaternion accumulator; panics if this is an angle.
    pub fn quaternion(self) -> Tensor<4> {
        match self {
            RotationState::Quaternion(q) => q,
            RotationState::Angle(_) => panic!("RotationState is Angle, expected Quaternion"),
        }
    }

    /// Run the [`NaN`/`Inf` guards](crate::utils::sanity) on the held tensor.
    pub fn sanity(&self) {
        match self {
            RotationState::Angle(a) => crate::modules::sanity(a),
            RotationState::Quaternion(q) => crate::modules::sanity(q),
        }
    }
}

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
    let w = aw.clone() * bw.clone()
        - ax.clone() * bx.clone()
        - ay.clone() * by.clone()
        - az.clone() * bz.clone();
    let x = aw.clone() * bx.clone() + ax.clone() * bw.clone() + ay.clone() * bz.clone()
        - az.clone() * by.clone();
    let y = aw.clone() * by.clone() - ax.clone() * bz.clone()
        + ay.clone() * bw.clone()
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
    // Clamp the sum-of-squares *before* `sqrt`: at a zero quaternion the forward
    // `sqrt(0)=0` is fine, but `sqrt`'s backward is `1/(2·0)=∞`, and `∞·(2·0)=NaN`.
    // Clamping pre-`sqrt` puts the degenerate point in `clamp_min`'s flat region,
    // so its gradient is a finite 0 (and a genuine unit quaternion, sumsq=1, is
    // untouched). The floor also keeps `norm` away from 0 for the division.
    //
    // The floor is the dtype-aware `div_eps` applied to the *sum-of-squares*
    // (giving a norm floor of `√div_eps`). It must engage as a representable
    // normal in the working dtype: in f16 a `div_eps²`-sized floor (~5e-7) would
    // underflow below the min-normal (~6.1e-5) and silently no-op, so we floor
    // the squared quantity at `div_eps` itself, which sits above each format's
    // denormal floor by construction.
    let eps = crate::utils::div_eps(q.dtype());
    let norm = (q.clone() * q.clone()).sum_dim(n).clamp_min(eps).sqrt();
    q / norm
}

/// Materialise a unit quaternion from a **scaled rotation vector** `g ∈ ℝ³`
/// (axis · angle) via the exponential map — the data-dependent "materialise
/// `Rₜ`" step, analogous to RoPE's `Δₜ · π · tanh(θₜ)` angle.
///
/// With `‖g‖ = angle` and `ĝ = g / angle` the axis, returns the unit quaternion
/// `q = (cos(angle/2), sin(angle/2)·ĝ)`.  A vanishing `g` maps to the identity
/// `(1, 0, 0, 0)`, so scaling `g` by a small `Δₜ` (the discretisation step)
/// yields a near-identity rotation — exactly the regime where a small step
/// barely rotates the state.  The `sin(angle/2)/angle` factor is the numerically
/// stable form of the (otherwise `0/0`) per-component scale near `g = 0`.
///
/// # Shapes
/// - `g` : `[..., 3]`
/// - out : `[..., 4]` (ordered `(w, x, y, z)`), unit norm.
pub fn quat_from_scaled_axis<const D: usize>(g: Tensor<D>) -> Tensor<D> {
    let n = D - 1;
    // Clamp the sum-of-squares *before* `sqrt`: at `g = 0` the forward `sqrt(0)=0`
    // is finite, but `sqrt`'s backward is `1/(2·0)=∞` and `∞·(2·0)=NaN`. Clamping
    // pre-`sqrt` puts `g = 0` in `clamp_min`'s flat (zero-gradient) region, so the
    // near-identity rotation gets a finite 0 gradient instead of a NaN. (This is
    // the FiLM-triggered decoder-backward NaN: a per-position rotation generator
    // hitting exactly zero.) The floor is the dtype-aware `div_eps` on the squared
    // quantity — see [`quat_normalize`] for why it floors `sumsq`, not the norm.
    let eps = crate::utils::div_eps(g.dtype());
    let angle = (g.clone() * g.clone()).sum_dim(n).clamp_min(eps).sqrt(); // [..., 1]
    let half = angle.clone() * 0.5;
    let w = half.clone().cos(); // [..., 1]
    // sin(angle/2) / angle  → 1/2 as angle → 0 (no rotation); `angle ≥ √div_eps`
    // after the pre-`sqrt` clamp above, so the division is already guarded.
    let scale = half.sin() / angle; // [..., 1]
    let v = g * scale; // [..., 3]
    quat_normalize(Tensor::cat(vec![w, v], n))
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
    assert_eq!(
        D + 1,
        DB,
        "rotate_state_rank_blocks splits one axis into (J, 4)"
    );
    let dims = v.dims();
    let state_rank = dims[D - 1];
    assert_eq!(
        state_rank % 4,
        0,
        "state_rank must be a multiple of 4 (quaternion blocks)"
    );
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
/// Implemented as a **Hillis–Steele** inclusive associative scan: the quaternion
/// product is associative (just not commutative), so a log-depth scan applies as
/// long as operand order is preserved (newest-on-left). Each doubling step is a
/// single full-tensor [`quat_mul`] plus a sequence shift, so the *sequential
/// dependency depth* is `O(log sequence)` rather than the `O(sequence)` of a
/// token-by-token loop — the same values, but a handful of large batched kernels
/// instead of thousands of serialized tiny ones (and a correspondingly shallow
/// autodiff graph). The sequential reference it replaces is kept as a test oracle
/// (`quat_cumprod_sequential` in the tests module) and asserted equal on values
/// **and** gradients.
pub fn quat_cumprod(q_bshj4: Tensor<5>, init: Option<Tensor<4>>) -> (Tensor<5>, Tensor<4>) {
    let [batch, sequence, nheads, blocks, _four] = q_bshj4.dims();
    let device = q_bshj4.device();

    // Pure prefix product Pₜ = qₜ ⊗ qₜ₋₁ ⊗ ⋯ ⊗ q₀ by Hillis–Steele doubling.
    // Invariant after each step with offset `d`: a[t] holds the product of the
    // window [t .. max(t-2d+1, 0)] (newest on the left). After ⌈log₂ sequence⌉
    // doublings the window covers [t .. 0], i.e. a[t] = Pₜ.
    let mut a = q_bshj4;
    let mut offset = 1usize;
    while offset < sequence {
        // shifted[t] = a[t-offset] for t ≥ offset, else the identity quaternion
        // (1,0,0,0) — so the first `offset` prefixes pass through unchanged
        // (a ⊗ identity = a).
        let ident = {
            let w = Tensor::ones([batch, offset, nheads, blocks, 1], &device);
            let xyz = Tensor::zeros([batch, offset, nheads, blocks, 3], &device);
            Tensor::cat(vec![w, xyz], 4)
        };
        let shifted = Tensor::cat(vec![ident, a.clone()], 1).narrow(1, 0, sequence);
        // Recent block (a) on the left, older block (shifted) on the right.
        a = quat_mul(a, shifted);
        offset *= 2;
    }

    // Fold the cross-chunk carry once: cumₜ = Pₜ ⊗ init. `init` (the previous
    // chunk's final cumulative rotation) is the oldest factor, hence on the
    // right; a missing carry is the identity and needs no multiply.
    let cum = match init {
        Some(init_bhj4) => {
            assert_eq!([batch, nheads, blocks, 4], init_bhj4.dims());
            quat_mul(a, init_bhj4.unsqueeze_dim::<5>(1)) // [batch, 1, nheads, J, 4] broadcasts over seq
        }
        None => a,
    };

    let final_carry = cum.clone().narrow(1, sequence - 1, 1).squeeze_dim::<4>(1); // [batch, nheads, J, 4]
    (cum, final_carry)
}

// ---------------------------------------------------------------------------
// Partial block rotation (rope_fraction support)
// ---------------------------------------------------------------------------

/// Apply a per-block quaternion rotation to the first `rope_width` entries of
/// the `state_rank` axis (a multiple of 4); the remainder passes through. The
/// quaternion analogue of [`apply_rope_partial`].
///
/// `q` has one quaternion per rotated block (`rope_width / 4` of them). `DB`
/// must equal `D + 1`.
pub fn rotate_blocks_partial<const D: usize, const DB: usize>(
    v: Tensor<D>,
    q: Tensor<DB>,
    rope_width: usize,
) -> Tensor<D> {
    let r = v.dims()[D - 1];
    if rope_width == r {
        rotate_state_rank_blocks::<D, DB>(v, q)
    } else {
        let head = v.clone().narrow(D - 1, 0, rope_width);
        let tail = v.narrow(D - 1, rope_width, r - rope_width);
        let head_rot = rotate_state_rank_blocks::<D, DB>(head, q);
        Tensor::cat(vec![head_rot, tail], D - 1)
    }
}

// ---------------------------------------------------------------------------
// Forward / step rotation of B and C (shared by both SSD pathways)
// ---------------------------------------------------------------------------

/// Rotate `B`/`C` for a **full sequence** by the data-dependent positional
/// rotation, returning the rotated projections and the new cumulative
/// [`RotationState`] to store in the cache.
///
/// Branches on [`RotationKind`]:
/// - [`Complex2D`](RotationKind::Complex2D): the abelian RoPE — cumulative
///   angle `cumsum` continued from `prev`, then [`apply_rope_partial`]. Exactly
///   the original Mamba-3 behaviour.
/// - [`Quaternion4D`](RotationKind::Quaternion4D): per-step unit quaternion
///   [`quat_from_scaled_axis`] (the in-projection generators scaled per-head by
///   `Δ`), composed by [`quat_cumprod`] continuing the cached quaternion, then
///   applied to `B`/`C` as `rotate(·, conj(Qₜ))` over the first `4·blocks`
///   state-rank entries.
///
/// # Shapes
/// - `rot_bsa` : `[batch, sequence, num_rotation_channels]` — the in-projection
///   rotation channels (angles for Complex2D, `3·blocks` quaternion generators
///   for Quaternion4D).
/// - `dt_bsh`  : `[batch, sequence, nheads]` (`Δ`).
/// - `b_bsmhr` / `c_bsmhr` : `[batch, sequence, mimo_rank, nheads, state_rank]`.
pub fn rotate_bc_forward(
    rot_bsa: Tensor<3>,
    dt_bsh: Tensor<3>,
    prev: RotationState,
    b_bsmhr: Tensor<5>,
    c_bsmhr: Tensor<5>,
    kind: RotationKind,
    rope_dim: usize,
) -> (Tensor<5>, Tensor<5>, RotationState) {
    let [batch, sequence, mimo_rank, nheads, _state_rank] = b_bsmhr.dims();
    match kind {
        RotationKind::Complex2D => {
            let prev_angle_bha = prev.angle();
            let num_rope_angles = prev_angle_bha.dims()[2];
            let theta_scaled_bsa = rot_bsa.tanh() * std::f32::consts::PI;
            let raw_angles_bsha =
                dt_bsh.unsqueeze_dim::<4>(3) * theta_scaled_bsa.unsqueeze_dim::<4>(2);
            let cum_angles_bsha = prev_angle_bha.unsqueeze_dim::<4>(1) + raw_angles_bsha.cumsum(1);
            let cum_angles_bsmha = cum_angles_bsha.clone().unsqueeze_dim::<5>(2).expand([
                batch,
                sequence,
                mimo_rank,
                nheads,
                num_rope_angles,
            ]);
            let rotate_pairwise = mimo_rank == 1;
            let b = apply_rope_partial::<5>(
                b_bsmhr,
                cum_angles_bsmha.clone(),
                rope_dim,
                rotate_pairwise,
            );
            let c = apply_rope_partial::<5>(c_bsmhr, cum_angles_bsmha, rope_dim, rotate_pairwise);
            let last = wrap_angle(
                cum_angles_bsha
                    .narrow(1, sequence - 1, 1)
                    .squeeze_dim::<3>(1),
            );
            (b, c, RotationState::Angle(last))
        }
        RotationKind::Quaternion4D => {
            let prev_q_bhj4 = prev.quaternion();
            let blocks = prev_q_bhj4.dims()[2];
            let rope_width = blocks * 4;
            // Generators [b,s,blocks,3] (shared across heads), scaled per-head by Δ.
            //
            // Bound the raw generator with `tanh·π` before scaling by Δ — the
            // direct analogue of the Complex2D path (`rot.tanh()·π`). Without it
            // the generator is unbounded, so a large in-projection activation makes
            // `g = rot·Δ` overflow f32 to `inf`, and `quat_from_scaled_axis`'s
            // `cos(∞)` then yields a forward NaN. The bound caps each per-step
            // rotation to `±π·Δ` (cos/sin still give the periodicity within range);
            // healthy `O(1)` generators stay in tanh's near-linear region.
            let g_bshj3 = (rot_bsa.tanh() * core::f32::consts::PI)
                .reshape([batch, sequence, blocks, 3])
                .unsqueeze_dim::<5>(2)
                * dt_bsh.unsqueeze_dim::<4>(3).unsqueeze_dim::<5>(4);
            let q_step_bshj4 = quat_from_scaled_axis::<5>(g_bshj3);
            // Memory-efficient scan: a custom recompute backward (saves only the
            // leaf inputs) instead of retaining the scan's intermediates. Equal
            // to [`quat_cumprod`] on values and gradients (asserted in tests).
            let (cum_bshj4, final_bhj4) = crate::mamba3::quat_scan::quat_cumprod_recalculated(
                q_step_bshj4,
                Some(prev_q_bhj4),
            );
            // B̄ = rotate by the inverse cumulative rotation (conjugate), per block,
            // broadcast over the mimo_rank axis.
            let conj_bsmhj4 = quat_conj(cum_bshj4)
                .unsqueeze_dim::<6>(2)
                .expand([batch, sequence, mimo_rank, nheads, blocks, 4]);
            let b = rotate_blocks_partial::<5, 6>(b_bsmhr, conj_bsmhj4.clone(), rope_width);
            let c = rotate_blocks_partial::<5, 6>(c_bsmhr, conj_bsmhj4, rope_width);
            (b, c, RotationState::Quaternion(quat_normalize(final_bhj4)))
        }
    }
}

/// Single-token counterpart of [`rotate_bc_forward`] for the recurrent `step`.
///
/// # Shapes
/// - `rot_ba`  : `[batch, num_rotation_channels]`.
/// - `dt_bh`   : `[batch, nheads]`.
/// - `b_bmhr` / `c_bmhr` : `[batch, mimo_rank, nheads, state_rank]`.
pub fn rotate_bc_step(
    rot_ba: Tensor<2>,
    dt_bh: Tensor<2>,
    prev: RotationState,
    b_bmhr: Tensor<4>,
    c_bmhr: Tensor<4>,
    kind: RotationKind,
    rope_dim: usize,
) -> (Tensor<4>, Tensor<4>, RotationState) {
    let [batch, mimo_rank, nheads, _state_rank] = b_bmhr.dims();
    match kind {
        RotationKind::Complex2D => {
            let prev_angle_bha = prev.angle();
            let num_rope_angles = prev_angle_bha.dims()[2];
            let theta_scaled_ba = rot_ba.tanh() * std::f32::consts::PI;
            let raw_angle_bha = dt_bh.unsqueeze_dim::<3>(2) * theta_scaled_ba.unsqueeze_dim::<3>(1);
            let new_cum_angle_bha = wrap_angle(prev_angle_bha + raw_angle_bha);
            let new_cum_angle_bmha = new_cum_angle_bha.clone().unsqueeze_dim::<4>(1).expand([
                batch,
                mimo_rank,
                nheads,
                num_rope_angles,
            ]);
            let rotate_pairwise = mimo_rank == 1;
            let b = apply_rope_partial::<4>(
                b_bmhr,
                new_cum_angle_bmha.clone(),
                rope_dim,
                rotate_pairwise,
            );
            let c = apply_rope_partial::<4>(c_bmhr, new_cum_angle_bmha, rope_dim, rotate_pairwise);
            (b, c, RotationState::Angle(new_cum_angle_bha))
        }
        RotationKind::Quaternion4D => {
            let prev_q_bhj4 = prev.quaternion();
            let blocks = prev_q_bhj4.dims()[2];
            let rope_width = blocks * 4;
            // `tanh·π` bound, matching `rotate_bc_forward` (see the note there).
            let g_bhj3 = (rot_ba.tanh() * core::f32::consts::PI)
                .reshape([batch, blocks, 3])
                .unsqueeze_dim::<4>(1)
                * dt_bh.unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3);
            let q_step_bhj4 = quat_from_scaled_axis::<4>(g_bhj3);
            // Single step: Qₜ = qₜ ⊗ Qₜ₋₁.
            let new_q_bhj4 = quat_normalize(quat_mul(q_step_bhj4, prev_q_bhj4));
            let conj_bmhj4 = quat_conj(new_q_bhj4.clone())
                .unsqueeze_dim::<5>(1)
                .expand([batch, mimo_rank, nheads, blocks, 4]);
            let b = rotate_blocks_partial::<4, 5>(b_bmhr, conj_bmhj4.clone(), rope_width);
            let c = rotate_blocks_partial::<4, 5>(c_bmhr, conj_bmhj4, rope_width);
            (b, c, RotationState::Quaternion(new_q_bhj4))
        }
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
