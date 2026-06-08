//! Tests for the quaternion (`k = 4`) rotational-state reference.
//!
//! The headline result is [`factored_matches_explicit`]: the factored form
//! (rotate `B`/`C` by the cumulative quaternion, then run a plain scalar-decay
//! SSM) reproduces the explicit recurrence that carries the `4×4` rotation
//! inside the state — for **non-commuting** quaternions. The two paths use
//! deliberately different primitives (quaternion product vs materialised `4×4`
//! matmul), so agreement cross-validates both the homomorphism and the RoPE
//! factoring at once.
//!
//! The other tests pin the supporting facts: the `q ↦ L_q` homomorphism and
//! orthogonality, the cross-chunk carry of [`quat_cumprod`], the abelian
//! collapse to a `cumsum` of angles (RoPE), and gradient parity.

use super::*;
use crate::modules::apply_rope;
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

const VAL_TOL: f32 = 1e-4;
const GRAD_TOL: f32 = 1e-3;

// ---------------------------------------------------------------------------
// Reference SSMs (test-only): vector state h ∈ ℝ^state_rank per (batch, head).
//
// per_head_dim is a pure spectator on the rotation, so we drop it here and use
// a scalar input xₜ per head, keeping the recurrence crisp:
//
//   explicit :  hₜ = αₜ · (Rₜ hₜ₋₁) + xₜ Bₜ ,   yₜ = Cₜᵀ hₜ
//   factored :  h̃ₜ = αₜ h̃ₜ₋₁ + xₜ B̄ₜ ,         yₜ = C̄ₜᵀ h̃ₜ
//
// with Rₜ = block-diag(L_{qₜ}), B̄ₜ = Pₜᵀ Bₜ, C̄ₜ = Pₜᵀ Cₜ, Pₜ = Rₜ⋯R₁.
// ---------------------------------------------------------------------------

/// Slice a `[batch, sequence, ...]` tensor at time `t`, dropping the seq axis.
fn at_t_bh(a: Tensor<3>, t: usize) -> Tensor<2> {
    a.narrow(1, t, 1).squeeze_dim::<2>(1)
}
fn at_t_bhr(a: Tensor<4>, t: usize) -> Tensor<3> {
    a.narrow(1, t, 1).squeeze_dim::<3>(1)
}
fn at_t_bhj4(a: Tensor<5>, t: usize) -> Tensor<4> {
    a.narrow(1, t, 1).squeeze_dim::<4>(1)
}

/// Explicit recurrence: carries the `4×4` rotation inside the state.
fn explicit_recurrence(
    q_bshj4: Tensor<5>,
    alpha_bsh: Tensor<3>,
    x_bsh: Tensor<3>,
    b_bshr: Tensor<4>,
    c_bshr: Tensor<4>,
    init_state_bhr: Tensor<3>,
) -> Tensor<3> {
    let [batch, sequence, nheads, blocks, _4] = q_bshj4.dims();
    let state_rank = blocks * 4;
    let mut h_bhr = init_state_bhr;
    let mut ys: Vec<Tensor<3>> = Vec::with_capacity(sequence);

    for t in 0..sequence {
        let q_t_bhj4 = at_t_bhj4(q_bshj4.clone(), t);
        let alpha_bh1 = at_t_bh(alpha_bsh.clone(), t).unsqueeze_dim::<3>(2);
        let x_bh1 = at_t_bh(x_bsh.clone(), t).unsqueeze_dim::<3>(2);
        let b_t_bhr = at_t_bhr(b_bshr.clone(), t);
        let c_t_bhr = at_t_bhr(c_bshr.clone(), t);

        // Rotate the state on its state_rank axis via the materialised 4×4.
        let rot_bhj44 = quat_to_rot4::<4, 5>(q_t_bhj4); // [b, h, J, 4, 4]
        let h_blocks_bhj41 = h_bhr
            .clone()
            .reshape([batch, nheads, blocks, 4])
            .unsqueeze_dim::<5>(4); // [b, h, J, 4, 1]
        let rotated_bhr = rot_bhj44
            .matmul(h_blocks_bhj41) // [b, h, J, 4, 1]
            .squeeze_dim::<4>(4) // [b, h, J, 4]
            .reshape([batch, nheads, state_rank]);

        h_bhr = rotated_bhr * alpha_bh1 + b_t_bhr * x_bh1;
        let y_bh = (c_t_bhr * h_bhr.clone()).sum_dim(2).squeeze_dim::<2>(2);
        ys.push(y_bh.unsqueeze_dim::<3>(1));
    }
    Tensor::cat(ys, 1) // [batch, sequence, nheads]
}

/// Factored recurrence: cumulative-rotate `B`/`C`, then a plain scalar SSM.
fn factored_recurrence(
    q_bshj4: Tensor<5>,
    alpha_bsh: Tensor<3>,
    x_bsh: Tensor<3>,
    b_bshr: Tensor<4>,
    c_bshr: Tensor<4>,
    init_state_bhr: Tensor<3>,
) -> Tensor<3> {
    let [_batch, sequence, _nheads, _blocks, _4] = q_bshj4.dims();

    // Cumulative rotation Pₜ (fresh identity carry), then absorb its inverse
    // (conjugate) into B and C.
    let (cum_bshj4, _final) = quat_cumprod(q_bshj4, None);
    let conj_bshj4 = quat_conj(cum_bshj4);
    let b_bar_bshr = rotate_state_rank_blocks::<4, 5>(b_bshr, conj_bshj4.clone());
    let c_bar_bshr = rotate_state_rank_blocks::<4, 5>(c_bshr, conj_bshj4);

    // Plain scalar-decay SSM on the rotated projections (drop-in for the SSD).
    let mut h_bhr = init_state_bhr;
    let mut ys: Vec<Tensor<3>> = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let alpha_bh1 = at_t_bh(alpha_bsh.clone(), t).unsqueeze_dim::<3>(2);
        let x_bh1 = at_t_bh(x_bsh.clone(), t).unsqueeze_dim::<3>(2);
        let b_t_bhr = at_t_bhr(b_bar_bshr.clone(), t);
        let c_t_bhr = at_t_bhr(c_bar_bshr.clone(), t);

        h_bhr = h_bhr * alpha_bh1 + b_t_bhr * x_bh1;
        let y_bh = (c_t_bhr * h_bhr.clone()).sum_dim(2).squeeze_dim::<2>(2);
        ys.push(y_bh.unsqueeze_dim::<3>(1));
    }
    Tensor::cat(ys, 1)
}

// ---------------------------------------------------------------------------
// Random-input builders
// ---------------------------------------------------------------------------

struct Inputs {
    q: Tensor<5>,     // unit quaternions [b, s, h, J, 4]
    alpha: Tensor<3>, // decay in (0, 1] [b, s, h]
    x: Tensor<3>,     // scalar input [b, s, h]
    b: Tensor<4>,     // [b, s, h, r]
    c: Tensor<4>,     // [b, s, h, r]
    init: Tensor<3>,  // [b, h, r]
}

fn random_inputs(
    batch: usize,
    sequence: usize,
    nheads: usize,
    blocks: usize,
    random_init: bool,
    device: &Device,
) -> Inputs {
    let state_rank = blocks * 4;
    let normal = Distribution::Normal(0.0, 1.0);
    // Unit quaternions from a random raw projection (generic non-commuting case).
    let q_raw = Tensor::<5>::random([batch, sequence, nheads, blocks, 4], normal, device);
    let q = quat_normalize(q_raw);
    // Decay in (0, 1]: exp of a negative log-decay (matches Δ·A with A < 0).
    let alpha = Tensor::<3>::random(
        [batch, sequence, nheads],
        Distribution::Normal(-0.5, 0.1),
        device,
    )
    .exp()
    .clamp(0.0, 1.0);
    let x = Tensor::<3>::random([batch, sequence, nheads], normal, device);
    let b = Tensor::<4>::random([batch, sequence, nheads, state_rank], normal, device);
    let c = Tensor::<4>::random([batch, sequence, nheads, state_rank], normal, device);
    let init = if random_init {
        Tensor::<3>::random(
            [batch, nheads, state_rank],
            Distribution::Normal(0.0, 0.1),
            device,
        )
    } else {
        Tensor::<3>::zeros([batch, nheads, state_rank], device)
    };
    Inputs {
        q,
        alpha,
        x,
        b,
        c,
        init,
    }
}

// ---------------------------------------------------------------------------
// 1. The headline: factored == explicit, for non-commuting quaternions
// ---------------------------------------------------------------------------

fn check_factored_matches_explicit(
    batch: usize,
    sequence: usize,
    nheads: usize,
    blocks: usize,
    random_init: bool,
) {
    let device: Device = Default::default();
    let inp = random_inputs(batch, sequence, nheads, blocks, random_init, &device);

    let y_exp = explicit_recurrence(
        inp.q.clone(),
        inp.alpha.clone(),
        inp.x.clone(),
        inp.b.clone(),
        inp.c.clone(),
        inp.init.clone(),
    );
    let y_fac = factored_recurrence(inp.q, inp.alpha, inp.x, inp.b, inp.c, inp.init);

    let d = max_abs_diff(y_exp, y_fac);
    assert!(
        d < VAL_TOL,
        "factored vs explicit: y max abs diff = {d:.6} (tol {VAL_TOL})"
    );
}

#[test]
fn factored_matches_explicit() {
    // Single quaternion block, fresh and continued (random) initial state.
    check_factored_matches_explicit(2, 7, 3, 1, false);
    check_factored_matches_explicit(2, 7, 3, 1, true);
}

#[test]
fn factored_matches_explicit_multiblock() {
    // state_rank = 4·J with J > 1 — independent quaternion blocks.
    check_factored_matches_explicit(2, 6, 2, 3, true);
}

// ---------------------------------------------------------------------------
// 2. q ↦ L_q is a homomorphism, and L_q is orthogonal
// ---------------------------------------------------------------------------

#[test]
fn rot4_is_orthogonal() {
    let device: Device = Default::default();
    let q = quat_normalize(Tensor::<2>::random(
        [16, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    let r = quat_to_rot4::<2, 3>(q); // [16, 4, 4]
    let rt = r.clone().transpose();
    let prod = r.matmul(rt); // should be identity per batch
    let eye = Tensor::<2>::eye(4, &device)
        .unsqueeze_dim::<3>(0)
        .expand([16, 4, 4]);
    let d = max_abs_diff(prod, eye);
    assert!(d < VAL_TOL, "L_q Lqᵀ ≠ I: max abs diff = {d:.6}");
}

#[test]
fn rot4_homomorphism() {
    // L_{a⊗b} == L_a · L_b, the property that makes the cumulative-product
    // scan equivalent to materialising and multiplying the 4×4 matrices.
    let device: Device = Default::default();
    let a = quat_normalize(Tensor::<2>::random(
        [16, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    let b = quat_normalize(Tensor::<2>::random(
        [16, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));

    let lhs = quat_to_rot4::<2, 3>(quat_mul(a.clone(), b.clone()));
    let rhs = quat_to_rot4::<2, 3>(a).matmul(quat_to_rot4::<2, 3>(b));
    let d = max_abs_diff(lhs, rhs);
    assert!(d < VAL_TOL, "L_(a⊗b) ≠ L_a·L_b: max abs diff = {d:.6}");
}

// ---------------------------------------------------------------------------
// 3. Cross-chunk carry: split scan threading the carry == full scan
// ---------------------------------------------------------------------------

#[test]
fn cumprod_split_equals_full() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, blocks) = (2, 9, 2, 2);
    let q = quat_normalize(Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));

    let (cum_full, final_full) = quat_cumprod(q.clone(), None);

    // Split at an arbitrary boundary, threading the carry across the seam.
    let split = 4;
    let q1 = q.clone().narrow(1, 0, split);
    let q2 = q.narrow(1, split, sequence - split);
    let (cum1, carry) = quat_cumprod(q1, None);
    let (cum2, final_split) = quat_cumprod(q2, Some(carry));
    let cum_split = Tensor::cat(vec![cum1, cum2], 1);

    let dc = max_abs_diff(cum_full, cum_split);
    let df = max_abs_diff(final_full, final_split);
    assert!(dc < VAL_TOL, "split cum ≠ full cum: {dc:.6}");
    assert!(df < VAL_TOL, "split final carry ≠ full: {df:.6}");
}

// ---------------------------------------------------------------------------
// 3b. Parallel scan == sequential oracle (values + grads)
//
// `quat_cumprod` is the Hillis–Steele log-depth scan; `quat_cumprod_sequential`
// below is the readable token-by-token reference it replaced. They must agree
// exactly (it is the same associative product, just a different evaluation
// order), on both values and gradients — this pins the fast path to the oracle.
// ---------------------------------------------------------------------------

/// Sequential (O(sequence)) reference for [`quat_cumprod`]: a token-by-token
/// running product `cumₜ = qₜ ⊗ cumₜ₋₁` seeded with the carry. Kept only as the
/// test oracle for the parallel scan.
fn quat_cumprod_sequential(q_bshj4: Tensor<5>, init: Option<Tensor<4>>) -> (Tensor<5>, Tensor<4>) {
    let [batch, sequence, nheads, blocks, _four] = q_bshj4.dims();
    let device = q_bshj4.device();

    let carry0 = init.unwrap_or_else(|| {
        let w = Tensor::<4>::ones([batch, nheads, blocks, 1], &device);
        let xyz = Tensor::<4>::zeros([batch, nheads, blocks, 3], &device);
        Tensor::cat(vec![w, xyz], 3)
    });
    assert_eq!([batch, nheads, blocks, 4], carry0.dims());

    let mut carry = carry0.unsqueeze_dim::<5>(1); // [batch, 1, nheads, J, 4]
    let mut cums: Vec<Tensor<5>> = Vec::with_capacity(sequence);
    for t in 0..sequence {
        let q_t = q_bshj4.clone().narrow(1, t, 1);
        let cur = quat_mul(q_t, carry); // qₜ ⊗ (running product)
        carry = cur.clone();
        cums.push(cur);
    }
    let cum = Tensor::cat(cums, 1);
    let final_carry = carry.squeeze_dim::<4>(1);
    (cum, final_carry)
}

#[test]
fn cumprod_parallel_matches_sequential_values() {
    let device: Device = Default::default();
    // A non-power-of-two sequence exercises the identity-padded shift edges.
    let (batch, sequence, nheads, blocks) = (2, 13, 3, 2);
    let q = quat_normalize(Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));

    // Fresh start and a continued (random unit) carry.
    for init in [
        None,
        Some(quat_normalize(Tensor::<4>::random(
            [batch, nheads, blocks, 4],
            Distribution::Normal(0.0, 1.0),
            &device,
        ))),
    ] {
        let (cum_par, fin_par) = quat_cumprod(q.clone(), init.clone());
        let (cum_seq, fin_seq) = quat_cumprod_sequential(q.clone(), init);
        let dc = max_abs_diff(cum_par, cum_seq);
        let df = max_abs_diff(fin_par, fin_seq);
        assert!(dc < VAL_TOL, "parallel vs sequential cum: {dc:.6}");
        assert!(df < VAL_TOL, "parallel vs sequential final carry: {df:.6}");
    }
}

#[test]
fn cumprod_parallel_matches_sequential_grads() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, blocks) = (2, 11, 2, 2);
    let q_raw = Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let head = Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Backprop a scalar loss on `cum` through each scan; the input gradient on
    // the (normalised) quaternion must match.
    let grad_for = |parallel: bool| -> Tensor<5> {
        let p = Param::from_tensor(Tensor::from_inner(q_raw.clone()));
        let q = quat_normalize(p.val());
        let (cum, _final) = if parallel {
            quat_cumprod(q, None)
        } else {
            quat_cumprod_sequential(q, None)
        };
        let loss = (cum * Tensor::from_inner(head.clone())).sum();
        let grads = loss.backward();
        p.val().grad(&grads).expect("grad q_raw")
    };

    let d = max_abs_diff(grad_for(true), grad_for(false));
    assert!(
        d < GRAD_TOL,
        "parallel vs sequential cum grad: {d:.6} (tol {GRAD_TOL})"
    );
}

// ---------------------------------------------------------------------------
// 4. Abelian collapse: single-axis quaternions ⇒ cumsum of half-angles (RoPE)
// ---------------------------------------------------------------------------

#[test]
fn single_axis_collapses_to_cumsum() {
    // Quaternions about a fixed axis (here î) commute; their cumulative product
    // is the half-angle-cumsum quaternion — exactly RoPE's closed form.
    let device: Device = Default::default();
    let (batch, sequence, nheads, blocks) = (2, 8, 2, 2);

    // Random per-step angles θₜ; per-step quaternion = (cos(θ/2), sin(θ/2), 0, 0).
    let theta_bsh1 = Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 1],
        Distribution::Normal(0.0, 0.7),
        &device,
    );
    let half = theta_bsh1.clone() * 0.5;
    let zeros = Tensor::<5>::zeros([batch, sequence, nheads, blocks, 1], &device);
    let q = Tensor::cat(
        vec![half.clone().cos(), half.sin(), zeros.clone(), zeros],
        4,
    );

    let (cum, _final) = quat_cumprod(q, None);

    // Closed form: cumulative half-angle Φₜ/2 = cumsum(θ/2), rebuilt as a quaternion.
    let cum_half = (theta_bsh1 * 0.5).cumsum(1);
    let zeros2 = Tensor::<5>::zeros([batch, sequence, nheads, blocks, 1], &device);
    let expected = Tensor::cat(
        vec![
            cum_half.clone().cos(),
            cum_half.sin(),
            zeros2.clone(),
            zeros2,
        ],
        4,
    );

    let d = max_abs_diff(cum, expected);
    assert!(
        d < VAL_TOL,
        "single-axis cumprod ≠ half-angle cumsum: {d:.6}"
    );
}

// ---------------------------------------------------------------------------
// 5. k=2 cross-check: the abelian restriction of the quaternion pathway
//    reproduces the *production* RoPE (`apply_rope`, the current pathway).
// ---------------------------------------------------------------------------

/// A single quaternion's left-multiplication `L_q` is **isoclinic**: with
/// `q = (cos φ, sin φ, 0, 0)` it rotates both the `(0,1)` and `(2,3)` planes of
/// its 4-block by the same angle `φ`. The production `apply_rope` (interleaved)
/// rotates each 2-pair `(0,1),(2,3),…` by its own angle.
///
/// So restricting the quaternion machinery to a single fixed axis (the abelian
/// case) and locking each block's two RoPE pairs to a shared angle, the two
/// pathways must agree **exactly** — and the non-abelian `quat_cumprod`
/// collapses to RoPE's cumulative-angle `cumsum`.  This pins the new code to the
/// battle-tested current implementation at `k = 2`.
#[test]
fn k2_quaternion_matches_production_rope() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, blocks) = (2, 6, 2, 2); // state_rank = 4·blocks = 8
    let state_rank = blocks * 4;

    // Per-step, per-block single-axis angle increments aₜⱼ about the î axis.
    let a_bshj = Tensor::<4>::random(
        [batch, sequence, nheads, blocks],
        Distribution::Normal(0.0, 0.6),
        &device,
    );

    // ── Quaternion pathway (single-axis î): qₜ = (cos a, sin a, 0, 0) ────────
    let a_bshj1 = a_bshj.clone().unsqueeze_dim::<5>(4); // [b,s,h,J,1]
    let zeros = Tensor::<5>::zeros([batch, sequence, nheads, blocks, 1], &device);
    let q = Tensor::cat(
        vec![a_bshj1.clone().cos(), a_bshj1.sin(), zeros.clone(), zeros],
        4,
    ); // [b,s,h,J,4]
    let (qcum, _final) = quat_cumprod(q, None);

    let bvec = Tensor::<4>::random(
        [batch, sequence, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let b_quat = rotate_state_rank_blocks::<4, 5>(bvec.clone(), qcum);

    // ── Production RoPE pathway ──────────────────────────────────────────────
    // Cumulative per-block angle Φⱼ = cumsum(aⱼ) (single-axis quaternions
    // commute, so the product's angle is the sum), duplicated across each
    // block's two interleaved pairs: angles = [Φ⁰, Φ⁰, Φ¹, Φ¹, …].
    let phi_bshj = a_bshj.cumsum(1);
    let angles_bshr2 = phi_bshj
        .unsqueeze_dim::<5>(4) // [b,s,h,J,1]
        .expand([batch, sequence, nheads, blocks, 2]) // [b,s,h,J,2]
        .reshape([batch, sequence, nheads, state_rank / 2]); // [b,s,h,r/2]
    let b_rope = apply_rope::<4>(bvec, angles_bshr2, /* rotate_pairwise = */ true);

    let d = max_abs_diff(b_quat, b_rope);
    assert!(
        d < VAL_TOL,
        "k=2 quaternion vs production apply_rope: max abs diff = {d:.6} (tol {VAL_TOL})"
    );
}

// ---------------------------------------------------------------------------
// 6. Non-commutativity is real and load-bearing (the expressivity beyond RoPE)
// ---------------------------------------------------------------------------

/// Two generic (multi-axis) quaternions do **not** commute, so swapping the
/// order of two rotation steps changes the cumulative rotation — something no
/// `cumsum`-of-angles (abelian RoPE) can represent. This is the structural
/// source of the extra state-tracking power.
#[test]
fn rotation_order_matters_for_generic_quaternions() {
    let device: Device = Default::default();
    let a = quat_normalize(Tensor::<2>::random(
        [8, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    let b = quat_normalize(Tensor::<2>::random(
        [8, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    let ab = quat_mul(a.clone(), b.clone());
    let ba = quat_mul(b, a);
    // a⊗b and b⊗a should differ materially for random (non-coaxial) quaternions.
    let d = max_abs_diff(ab, ba);
    assert!(
        d > 1e-2,
        "expected non-commuting quaternions to differ, got {d:.6}"
    );
}

// ---------------------------------------------------------------------------
// 7. Data-dependent materialisation: quat_from_scaled_axis
// ---------------------------------------------------------------------------

#[test]
fn scaled_axis_is_unit_and_identity_at_zero() {
    let device: Device = Default::default();
    // Random generators → unit quaternions.
    let g = Tensor::<2>::random([32, 3], Distribution::Normal(0.0, 1.0), &device);
    let q = quat_from_scaled_axis(g);
    let norm = (q.clone() * q).sum_dim(1).sqrt();
    let d_unit = max_abs_diff(norm, Tensor::<2>::ones([32, 1], &device));
    assert!(
        d_unit < VAL_TOL,
        "quat_from_scaled_axis not unit norm: {d_unit:.6}"
    );

    // Zero generator (e.g. Δ → 0) → identity quaternion (1, 0, 0, 0).
    let zero = Tensor::<2>::zeros([4, 3], &device);
    let q0 = quat_from_scaled_axis(zero);
    let ident = Tensor::cat(
        vec![
            Tensor::<2>::ones([4, 1], &device),
            Tensor::<2>::zeros([4, 3], &device),
        ],
        1,
    );
    let d_id = max_abs_diff(q0, ident);
    assert!(
        d_id < VAL_TOL,
        "zero generator ≠ identity quaternion: {d_id:.6}"
    );
}

/// True iff every element of `t` is finite (no NaN, no Inf).
fn all_finite<const D: usize>(t: Tensor<D>) -> bool {
    !t.clone().is_nan().any().into_scalar::<bool>().to_bool()
        && !t.is_inf().any().into_scalar::<bool>().to_bool()
}

/// Regression: a generator that is **exactly zero** (Δ → 0, or a zero rotation
/// channel) must yield a *finite* gradient. The norm `angle = √(Σg²)` is finite
/// in the forward (`√0 = 0`, the identity rotation), but an unguarded `sqrt`
/// has backward `1/(2√x)·2g = ∞·0 = NaN` at the origin. The pre-`sqrt`
/// `clamp_min(div_eps(dtype))` puts the origin in `clamp_min`'s flat region, so
/// the gradient is a finite 0 there. (This is the FiLM-decoder NaN we hunted:
/// the per-position rotation generator can land on exactly zero.)
///
/// The clamp floors the **sum-of-squares** at `div_eps`, which is a representable
/// normal in every supported dtype (f64/f32/f16/bf16); a `div_eps²`-sized floor
/// would underflow to a no-op in f16 and re-open the NaN.
#[test]
fn scaled_axis_zero_generator_grad_finite() {
    let device: Device = Default::default();
    // Row 0 is the exact zero generator; the rest are generic. Mixing the two
    // checks both that the degenerate row is guarded and that the clamp does not
    // disturb the ordinary rows (which still backprop normally).
    let random = Tensor::<2>::random([7, 3], Distribution::Normal(0.0, 1.0), &device);
    let raw = Tensor::cat(vec![Tensor::<2>::zeros([1, 3], &device), random], 0); // [8, 3]
    let p = Param::from_tensor(Tensor::from_inner(raw));

    let q = quat_from_scaled_axis(p.val());
    let grads = q.sum().backward();
    let dg = p.val().grad(&grads).expect("grad g");
    assert!(
        all_finite(dg),
        "quat_from_scaled_axis gradient non-finite at a zero generator"
    );
}

/// Regression companion to [`scaled_axis_zero_generator_grad_finite`] for
/// [`quat_normalize`]: a zero quaternion (all four components 0) must backprop
/// to a finite gradient. Same `sqrt`-at-origin mechanism, same pre-`sqrt`
/// `clamp_min` fix; a unit quaternion (sum-of-squares 1) is left untouched.
#[test]
fn normalize_zero_quaternion_grad_finite() {
    let device: Device = Default::default();
    let random = Tensor::<2>::random([7, 4], Distribution::Normal(0.0, 1.0), &device);
    let raw = Tensor::cat(vec![Tensor::<2>::zeros([1, 4], &device), random], 0); // [8, 4]
    let p = Param::from_tensor(Tensor::from_inner(raw));

    let q = quat_normalize(p.val());
    let grads = q.sum().backward();
    let dq = p.val().grad(&grads).expect("grad q");
    assert!(
        all_finite(dq),
        "quat_normalize gradient non-finite at a zero quaternion"
    );
}

#[test]
fn scaled_axis_single_axis_matches_half_angle() {
    // g = (2φ, 0, 0) ⇒ angle = 2φ, q = (cos φ, sin φ, 0, 0): the single-axis
    // parameterisation used by the RoPE collapse / cross-check above.
    let device: Device = Default::default();
    let phi = Tensor::<2>::random([16, 1], Distribution::Normal(0.0, 0.7), &device);
    let zeros = Tensor::<2>::zeros([16, 1], &device);
    let g = Tensor::cat(vec![phi.clone() * 2.0, zeros.clone(), zeros.clone()], 1); // (2φ,0,0)
    let q = quat_from_scaled_axis(g);
    let expected = Tensor::cat(vec![phi.clone().cos(), phi.sin(), zeros.clone(), zeros], 1);
    let d = max_abs_diff(q, expected);
    assert!(d < VAL_TOL, "scaled-axis single-axis mismatch: {d:.6}");
}

// ---------------------------------------------------------------------------
// 8. Config switch (step 1): RotationKind + in-projection sizing
// ---------------------------------------------------------------------------

#[test]
fn config_rotation_channels_and_d_in_proj() {
    use crate::mamba3::mamba3::Mamba3Config;
    // state_rank=16, rope_fraction=0.5 ⇒ rope_dim=8, num_rope_angles=4,
    // num_quat_blocks = (16·0.5 → 8, /4) = 2.
    let base = Mamba3Config::new(64)
        .with_state_rank(16)
        .with_expand(2)
        .with_per_head_dim(8);

    // Default is the abelian pathway, and d_in_proj is unchanged from the
    // legacy formula (num_rotation_channels == num_rope_angles).
    assert_eq!(base.rotation, RotationKind::Complex2D);
    assert_eq!(base.num_rotation_channels(), base.num_rope_angles());
    let legacy = 2 * base.d_inner()
        + 2 * base.ngroups * base.state_rank * base.mimo_rank
        + 3 * base.nheads()
        + base.num_rope_angles();
    assert_eq!(base.d_in_proj(), legacy);

    // Quaternion4D: rotation channels = 3 · num_quat_blocks, reflected in d_in_proj.
    let q = base.clone().with_rotation(RotationKind::Quaternion4D);
    assert_eq!(q.num_quat_blocks(), 2);
    assert_eq!(q.num_rotation_channels(), 3 * 2);
    let q_expected = 2 * q.d_inner()
        + 2 * q.ngroups * q.state_rank * q.mimo_rank
        + 3 * q.nheads()
        + 3 * q.num_quat_blocks();
    assert_eq!(q.d_in_proj(), q_expected);
    // The two pathways differ only in the rotation-channel count.
    assert_eq!(
        q.d_in_proj() as i64 - base.d_in_proj() as i64,
        3 * q.num_quat_blocks() as i64 - base.num_rope_angles() as i64
    );
}

/// The block stores its [`RotationKind`] as a `#[module(skip)]` field (a
/// non-parameter constant). It must therefore be excluded from the record yet
/// survive a `into_record` → `load_record` round-trip (carried from the module
/// being loaded into), and the loaded block must still run the quaternion path.
#[test]
fn quaternion_rotation_field_survives_record_roundtrip() {
    use crate::mamba3::mamba3::Mamba3Config;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    use burn::module::Module;
    let device: Device = Default::default();
    let cfg = Mamba3Config::new(32)
        .with_state_rank(16)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_rotation(RotationKind::Quaternion4D);

    let block = cfg.init(&device);
    assert_eq!(block.rotation_kind(), RotationKind::Quaternion4D);

    // Round-trip the record into a freshly-initialised (same-config) block.
    let record = block.clone().into_record();
    let block2 = cfg.init(&device).load_record(record);
    assert_eq!(block2.rotation_kind(), RotationKind::Quaternion4D);

    // The loaded block still runs the quaternion forward.
    let x = Tensor::<3>::random([2, 4, 32], Distribution::Normal(0.0, 1.0), &device);
    let (out, _) = block2.forward(x, None, Mamba3SsdPath::Minimal(None));
    assert_eq!([2, 4, 32], out.dims());
}

/// A Quaternion4D block's chunked `forward` must equal its recurrent `step`
/// unrolling — the same parity guarantee the abelian path satisfies, now for the
/// non-abelian quaternion rotation (chunked `quat_cumprod` vs single-step
/// `quat_mul`, feeding the unchanged double-ssd SSD core).
fn quaternion_forward_step_parity(rope_fraction: f64, mimo_rank: usize) {
    use crate::mamba3::mamba3::Mamba3Config;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    let device: Device = Default::default();
    let model = Mamba3Config::new(32)
        .with_state_rank(16)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_mimo_rank(mimo_rank)
        .with_rope_fraction(rope_fraction)
        .with_rotation(RotationKind::Quaternion4D)
        .init(&device);

    let (batch, seq) = (2, 5);
    let input = Tensor::<3>::random([batch, seq, 32], Distribution::Normal(0.0, 1.0), &device);

    // Chunked forward (fresh cache ⇒ double-ssd quaternion pathway).
    let (out_fwd, _cache) = model.forward(input.clone(), None, Mamba3SsdPath::Minimal(None));

    // Recurrent step unrolling from a fresh cache.
    let mut cache = None;
    let mut outs: Vec<Tensor<3>> = Vec::with_capacity(seq);
    for t in 0..seq {
        let x_t = input.clone().narrow(1, t, 1).squeeze_dim::<2>(1); // [batch, d_model]
        let (o, c) = model.step(x_t, cache);
        outs.push(o.unsqueeze_dim::<3>(1));
        cache = Some(c);
    }
    let out_step = Tensor::cat(outs, 1);

    let d = max_abs_diff(out_fwd, out_step);
    assert!(
        d < 1e-3,
        "quaternion forward vs step parity (rope_fraction={rope_fraction}, mimo_rank={mimo_rank}): {d:.6}"
    );
}

#[test]
fn quaternion_block_forward_step_parity_full_rope() {
    quaternion_forward_step_parity(1.0, 1);
}

#[test]
fn quaternion_block_forward_step_parity_partial_rope() {
    quaternion_forward_step_parity(0.5, 1);
}

#[test]
fn quaternion_block_forward_step_parity_mimo() {
    quaternion_forward_step_parity(1.0, 2);
}

/// Chunked-prefill continuity for Quaternion4D: a single `forward` over the full
/// sequence must equal a split `forward(prefix)` → `forward(suffix)` that threads
/// the cache across the seam (exercises the cross-chunk quaternion carry at the
/// block level).
#[test]
fn quaternion_split_prefill_matches_full() {
    use crate::mamba3::mamba3::Mamba3Config;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    let device: Device = Default::default();
    let model = Mamba3Config::new(32)
        .with_state_rank(16)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_rope_fraction(0.5)
        .with_rotation(RotationKind::Quaternion4D)
        .init(&device);

    let (batch, seq, split) = (2, 6, 4);
    let input = Tensor::<3>::random([batch, seq, 32], Distribution::Normal(0.0, 1.0), &device);

    let (out_full, cache_full) = model.forward(input.clone(), None, Mamba3SsdPath::Minimal(None));

    let prefix = input.clone().narrow(1, 0, split);
    let suffix = input.narrow(1, split, seq - split);
    let (out_pre, mid) = model.forward(prefix, None, Mamba3SsdPath::Minimal(None));
    let (out_suf, cache_split) = model.forward(suffix, Some(mid), Mamba3SsdPath::Minimal(None));
    let out_cat = Tensor::cat(vec![out_pre, out_suf], 1);

    let d_out = max_abs_diff(out_full, out_cat);
    assert!(
        d_out < 1e-3,
        "quaternion split-prefill output mismatch: {d_out:.6}"
    );

    // Final SSM state must also match. A fresh cache now defaults to the
    // single-ssd pathway for Quaternion4D too (forward_single_ssd supports it).
    let s_full = cache_full.single_ssd().unwrap().ssm_bhpr;
    let s_split = cache_split.single_ssd().unwrap().ssm_bhpr;
    let d_state = max_abs_diff(s_full, s_split);
    assert!(
        d_state < 1e-3,
        "quaternion split-prefill final-state mismatch: {d_state:.6}"
    );
}

/// Gradient parity for Quaternion4D: backprop through the chunked `forward` and
/// through the recurrent `step` unrolling must agree (same function ⇒ same
/// gradients), confirming the backward path through the quaternion scan matches
/// the recurrent form.
#[test]
fn quaternion_forward_step_grad_parity() {
    use crate::mamba3::mamba3::Mamba3Config;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    let device: Device = Default::default();
    let model = Mamba3Config::new(32)
        .with_state_rank(16)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_rope_fraction(1.0)
        .with_rotation(RotationKind::Quaternion4D)
        .init(&device.clone().autodiff());

    let (batch, seq) = (2, 4);
    let input = Tensor::<3>::random([batch, seq, 32], Distribution::Normal(0.0, 1.0), &device);
    let head = Tensor::<3>::random([batch, seq, 32], Distribution::Normal(0.0, 1.0), &device);

    // Fresh autodiff leaves per path.
    let p_fwd = Param::from_tensor(Tensor::from_inner(input.clone()));
    let p_step = Param::from_tensor(Tensor::from_inner(input));

    // Chunked forward.
    let (out_fwd, _) = model.forward(p_fwd.val(), None, Mamba3SsdPath::Minimal(None));
    let loss_fwd = (out_fwd * Tensor::from_inner(head.clone())).sum();
    let g_fwd = loss_fwd.backward();
    let d_in_fwd = p_fwd.val().grad(&g_fwd).expect("grad input (forward)");
    let d_w_fwd = model
        .in_proj
        .weight
        .val()
        .grad(&g_fwd)
        .expect("grad in_proj (forward)");

    // Recurrent step unrolling.
    let mut cache = None;
    let mut outs: Vec<Tensor<3>> = Vec::with_capacity(seq);
    for t in 0..seq {
        let x_t = p_step.val().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (o, c) = model.step(x_t, cache);
        outs.push(o.unsqueeze_dim::<3>(1));
        cache = Some(c);
    }
    let out_step = Tensor::cat(outs, 1);
    let loss_step = (out_step * Tensor::from_inner(head)).sum();
    let g_step = loss_step.backward();
    let d_in_step = p_step.val().grad(&g_step).expect("grad input (step)");
    let d_w_step = model
        .in_proj
        .weight
        .val()
        .grad(&g_step)
        .expect("grad in_proj (step)");

    let d_in = max_abs_diff(d_in_fwd, d_in_step);
    let d_w = max_abs_diff(d_w_fwd, d_w_step);
    assert!(
        d_in < 1e-2,
        "quaternion forward/step input-grad mismatch: {d_in:.6}"
    );
    assert!(
        d_w < 1e-2,
        "quaternion forward/step in_proj-grad mismatch: {d_w:.6}"
    );
}

// ---------------------------------------------------------------------------
// 9. Cache accumulator variant (step 2): RotationState
// ---------------------------------------------------------------------------

#[test]
fn rotation_state_constructors_and_accessors() {
    let device: Device = Default::default();

    let a = RotationState::zeros_angle(2, 3, 4, &device);
    a.sanity();
    assert_eq!(a.clone().angle().dims(), [2, 3, 4]);

    let q = RotationState::identity_quaternion(2, 3, 5, &device);
    q.sanity();
    let qt = q.quaternion();
    assert_eq!(qt.dims(), [2, 3, 5, 4]);
    // Identity quaternion: real part 1, vector part 0.
    let w = qt.clone().narrow(3, 0, 1);
    let xyz = qt.narrow(3, 1, 3);
    assert!(max_abs_diff(w, Tensor::<4>::ones([2, 3, 5, 1], &device)) < VAL_TOL);
    assert!(max_abs_diff(xyz, Tensor::<4>::zeros([2, 3, 5, 3], &device)) < VAL_TOL);
}

#[test]
#[should_panic(expected = "expected Quaternion")]
fn rotation_state_wrong_unwrap_panics() {
    let device: Device = Default::default();
    let a = RotationState::zeros_angle(1, 1, 1, &device);
    let _ = a.quaternion();
}

/// The bidirectional wrapper must also run with the quaternion rotation: it
/// defaults its caches to the double-ssd pathway when the block is Quaternion4D
/// (a straight + reversed double-ssd pass), rather than panicking on a single-ssd
/// cache.
#[test]
fn quaternion_bidi_forward_runs() {
    use crate::mamba3::mamba3::Mamba3Config;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    use crate::modules::bidi::OutputMergeConfig;
    use crate::modules::{MambaBidiLayersConfig, MambaSsdPath};
    let device: Device = Default::default();
    let block = Mamba3Config::new(32)
        .with_state_rank(16)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_rope_fraction(1.0)
        .with_rotation(RotationKind::Quaternion4D);
    let n_real = 2; // one bidirectional pair
    let layers = MambaBidiLayersConfig::Mamba3 {
        n_real_layers: n_real,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(n_real),
        class_latents: Vec::new(),
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);

    let (batch, seq) = (2, 5);
    let x = Tensor::<3>::random([batch, seq, 32], Distribution::Normal(0.0, 1.0), &device);
    let (out, _caches) =
        layers.forward(x, None, MambaSsdPath::Mamba3(Mamba3SsdPath::Minimal(None)));
    assert_eq!([batch, seq, 32], out.dims());
}

// ---------------------------------------------------------------------------
// 10. Gradient parity: factored vs explicit agree on input gradients
// ---------------------------------------------------------------------------

struct Params {
    q_raw: Param<Tensor<5>>,
    alpha: Param<Tensor<3>>,
    x: Param<Tensor<3>>,
    b: Param<Tensor<4>>,
    c: Param<Tensor<4>>,
    init: Param<Tensor<3>>,
}

impl Params {
    fn from_inner(inp: &Inputs, q_raw: Tensor<5>) -> Self {
        Params {
            q_raw: Param::from_tensor(Tensor::from_inner(q_raw)),
            alpha: Param::from_tensor(Tensor::from_inner(inp.alpha.clone())),
            x: Param::from_tensor(Tensor::from_inner(inp.x.clone())),
            b: Param::from_tensor(Tensor::from_inner(inp.b.clone())),
            c: Param::from_tensor(Tensor::from_inner(inp.c.clone())),
            init: Param::from_tensor(Tensor::from_inner(inp.init.clone())),
        }
    }
}

struct Grads {
    d_q: Tensor<5>,
    d_alpha: Tensor<3>,
    d_x: Tensor<3>,
    d_b: Tensor<4>,
    d_c: Tensor<4>,
    d_init: Tensor<3>,
}

fn run_grads(p: &Params, head: Tensor<3>, factored: bool) -> Grads {
    // Normalise inside the graph so q_raw gets a gradient through the unit map.
    let q = quat_normalize(p.q_raw.val());
    let y = if factored {
        factored_recurrence(
            q,
            p.alpha.val(),
            p.x.val(),
            p.b.val(),
            p.c.val(),
            p.init.val(),
        )
    } else {
        explicit_recurrence(
            q,
            p.alpha.val(),
            p.x.val(),
            p.b.val(),
            p.c.val(),
            p.init.val(),
        )
    };
    let loss = (y * Tensor::from_inner(head)).sum();
    let grads = loss.backward();
    Grads {
        d_q: p.q_raw.val().grad(&grads).expect("grad q_raw"),
        d_alpha: p.alpha.val().grad(&grads).expect("grad alpha"),
        d_x: p.x.val().grad(&grads).expect("grad x"),
        d_b: p.b.val().grad(&grads).expect("grad b"),
        d_c: p.c.val().grad(&grads).expect("grad c"),
        d_init: p.init.val().grad(&grads).expect("grad init"),
    }
}

#[test]
fn factored_matches_explicit_grads() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, blocks) = (2, 5, 2, 2);
    let inp = random_inputs(batch, sequence, nheads, blocks, true, &device);
    // Re-draw a raw (un-normalised) quaternion so the unit map carries gradient.
    let q_raw = Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let head = Tensor::<3>::random(
        [batch, sequence, nheads],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let p_exp = Params::from_inner(&inp, q_raw.clone());
    let p_fac = Params::from_inner(&inp, q_raw);
    let g_exp = run_grads(&p_exp, head.clone(), false);
    let g_fac = run_grads(&p_fac, head, true);

    let pairs = [
        ("q", max_abs_diff(g_exp.d_q, g_fac.d_q)),
        ("alpha", max_abs_diff(g_exp.d_alpha, g_fac.d_alpha)),
        ("x", max_abs_diff(g_exp.d_x, g_fac.d_x)),
        ("b", max_abs_diff(g_exp.d_b, g_fac.d_b)),
        ("c", max_abs_diff(g_exp.d_c, g_fac.d_c)),
        ("init", max_abs_diff(g_exp.d_init, g_fac.d_init)),
    ];
    for (name, d) in pairs {
        assert!(
            d < GRAD_TOL,
            "grad {name}: factored vs explicit max abs diff = {d:.6} (tol {GRAD_TOL})"
        );
    }
}

// ---------------------------------------------------------------------------
// TEMP DIAGNOSTIC: hunting the Quaternion4D forward NaN.
//
// In-flight training (midi-gen) hits a forward NaN where `g = (tanh(rot)·π)·dt`
// reports max≈3.8e34 despite `rot≈2.2`, `dt≈0.29` — which the bound makes
// impossible (|g| ≤ π·0.29 ≈ 0.9). These tests reproduce the exact arithmetic
// in isolation to decide between (a) a source bug, (b) a stale binary, or
// (c) UB in burn's tensor ops. Remove once resolved.
// ---------------------------------------------------------------------------

/// Tightest possible check: the literal `g` expression from
/// [`rotate_bc_forward`]'s Quaternion4D branch must stay within `π·max(dt)`,
/// even for absurd raw `rot` activations. If this fails, the bug is in the
/// source/primitive; if it passes, the failing run used a different binary.
#[test]
fn g_generator_is_bounded() {
    let device: Device = Default::default();
    let (batch, sequence, blocks, nheads) = (1, 8, 64, 12);

    for &rot_amp in &[1.0f32, 10.0, 50.0, 1e4, 1e30] {
        let rot_bsa = Tensor::<3>::random(
            [batch, sequence, blocks * 3],
            Distribution::Uniform(-rot_amp as f64, rot_amp as f64),
            &device,
        );
        let dt_bsh = Tensor::<3>::random(
            [batch, sequence, nheads],
            Distribution::Uniform(0.0, 0.3),
            &device,
        );

        // Exact copy of the forward expression.
        let g_bshj3 = (rot_bsa.tanh() * core::f32::consts::PI)
            .reshape([batch, sequence, blocks, 3])
            .unsqueeze_dim::<5>(2)
            * dt_bsh.unsqueeze_dim::<4>(3).unsqueeze_dim::<5>(4);

        let gmax = g_bshj3.clone().abs().max().into_scalar::<f32>();
        let nan = g_bshj3.is_nan().any().into_scalar::<bool>();
        let bound = core::f32::consts::PI * 0.3 + 1e-3;
        eprintln!("[g-bound] rot_amp={rot_amp:.0e} gmax={gmax:.3e} nan={nan} (bound {bound:.3e})");
        assert!(!nan, "g is NaN at rot_amp={rot_amp}");
        assert!(gmax <= bound, "g={gmax:.3e} exceeds π·dt bound at rot_amp={rot_amp}");
    }
}

/// Confirms the failure *mechanism*: a finite-but-huge `g` overflows
/// `quat_from_scaled_axis` (`‖g‖² → inf`, `cos(inf) → NaN`). This is what we'd
/// see if the bound were somehow absent — used to validate the diagnosis.
#[test]
fn huge_g_overflows_quat_from_scaled_axis() {
    let device: Device = Default::default();
    // ‖g‖ ≈ 3.8e34 → ‖g‖² ≈ 1.4e69 > f32::MAX → inf → cos(inf)=NaN.
    let g = Tensor::<2>::full([4, 3], 3.8e34, &device);
    let q = quat_from_scaled_axis::<2>(g);
    let nan = q.is_nan().any().into_scalar::<bool>();
    eprintln!("[huge-g] q_nan={nan}");
    assert!(nan, "expected NaN from cos(inf); if false, quat_from_scaled_axis is robust to huge g");
}

/// End-to-end: drive the real [`rotate_bc_forward`] with absurd `rot` and verify
/// the rotated B/C come out finite (the bound must protect the full path).
#[test]
fn rotate_bc_forward_survives_huge_rot() {
    let device: Device = Default::default();
    let (batch, sequence, mimo_rank, nheads, blocks) = (1, 8, 1, 12, 64);
    let state_rank = blocks * 4;

    let rot_bsa = Tensor::<3>::random(
        [batch, sequence, blocks * 3],
        Distribution::Uniform(-1e30, 1e30),
        &device,
    );
    let dt_bsh =
        Tensor::<3>::random([batch, sequence, nheads], Distribution::Uniform(0.0, 0.3), &device);
    let b = Tensor::<5>::random(
        [batch, sequence, mimo_rank, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let c = b.clone();

    // Fresh identity carry.
    let init_q = {
        let w = Tensor::<4>::ones([batch, nheads, blocks, 1], &device);
        let xyz = Tensor::<4>::zeros([batch, nheads, blocks, 3], &device);
        Tensor::cat(vec![w, xyz], 3)
    };
    let (b_out, c_out, _) = rotate_bc_forward(
        rot_bsa,
        dt_bsh,
        RotationState::Quaternion(init_q),
        b,
        c,
        RotationKind::Quaternion4D,
        state_rank,
    );
    let b_nan = b_out.is_nan().any().into_scalar::<bool>();
    let c_nan = c_out.is_nan().any().into_scalar::<bool>();
    eprintln!("[rotate-bc] b_nan={b_nan} c_nan={c_nan}");
    assert!(!b_nan && !c_nan, "rotate_bc_forward produced NaN under huge rot");
}

/// Faithful repro of the real call site: `rot_bsa` is the **last slice of a
/// `split_into`** over a wide projection (a non-contiguous view), not a fresh
/// contiguous tensor. Mirrors midi-gen's AE config
/// (d_inner=384, bc=1024, nheads=12, num_rotation_channels=192 → proj width 3044).
/// If `split_with_sizes` mis-views the buffer, `tanh` reads garbage and `g`
/// blows past the bound here.
#[test]
fn g_bounded_from_split_view() {
    use crate::modules::split_into;
    let device: Device = Default::default();
    let (batch, sequence, blocks, nheads) = (1, 8, 64, 12);
    let (d_inner, bc) = (384usize, 1024usize);
    let num_rot = blocks * 3; // 192
    let widths = [d_inner, d_inner, bc, bc, nheads, nheads, nheads, num_rot];
    let total: usize = widths.iter().sum(); // 3044

    // Wide projection with realistic-ish spread (some large activations).
    let proj = Tensor::<3>::random(
        [batch, sequence, total],
        Distribution::Normal(0.0, 5.0),
        &device,
    );
    let [_z, _x, _b, _c, dd_dt, _a, _lam, rot_bsa] = split_into(proj, widths, 2);

    // dt path roughly as in single_ssd: softplus then clamp to [0, 1].
    let dt_bsh = burn::tensor::activation::softplus(dd_dt, 1.0).clamp(0.0, 0.3);

    let g_bshj3 = (rot_bsa.tanh() * core::f32::consts::PI)
        .reshape([batch, sequence, blocks, 3])
        .unsqueeze_dim::<5>(2)
        * dt_bsh.unsqueeze_dim::<4>(3).unsqueeze_dim::<5>(4);

    let gmax = g_bshj3.clone().abs().max().into_scalar::<f32>();
    let nan = g_bshj3.is_nan().any().into_scalar::<bool>();
    eprintln!("[g-split] gmax={gmax:.3e} nan={nan} (bound 9.435e-1)");
    assert!(!nan, "g NaN from split-view path");
    assert!(gmax <= core::f32::consts::PI * 0.3 + 1e-3, "g={gmax:.3e} exceeds bound via split view");
}

/// Stress repro for the **non-deterministic** Quaternion4D `g` overflow (took 4
/// training trials to surface). Runs the real split-view `g` computation many
/// times; on a CPU backend this is deterministically bounded, so any breach here
/// (esp. on CUDA) indicates a kernel-level race/UB rather than a math bug. Prints
/// the worst `gmax` observed. Ignored by default (slow); run explicitly.
#[test]
// #[ignore]
fn g_split_view_stress() {
    use crate::modules::split_into;
    let device: Device = Default::default();
    let (batch, sequence, blocks, nheads) = (1, 16, 64, 12);
    let (d_inner, bc) = (384usize, 1024usize);
    let num_rot = blocks * 3;
    let widths = [d_inner, d_inner, bc, bc, nheads, nheads, nheads, num_rot];
    let total: usize = widths.iter().sum();
    let bound = core::f32::consts::PI * 0.3 + 1e-2;

    let iters = std::env::var("STRESS_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(2000usize);
    let mut worst = 0.0f32;
    for i in 0..iters {
        let proj = Tensor::<3>::random(
            [batch, sequence, total],
            Distribution::Normal(0.0, 5.0),
            &device,
        );
        let [_z, _x, _b, _c, dd_dt, _a, _lam, rot_bsa] = split_into(proj, widths, 2);
        let dt_bsh = burn::tensor::activation::softplus(dd_dt, 1.0).clamp(0.0, 0.3);
        let g = (rot_bsa.tanh() * core::f32::consts::PI)
            .reshape([batch, sequence, blocks, 3])
            .unsqueeze_dim::<5>(2)
            * dt_bsh.unsqueeze_dim::<4>(3).unsqueeze_dim::<5>(4);
        let gmax = g.clone().abs().max().into_scalar::<f32>();
        let nan = g.is_nan().any().into_scalar::<bool>();
        if gmax > worst { worst = gmax; }
        if nan || gmax > bound {
            panic!("[g-stress] BREACH at iter {i}: gmax={gmax:.3e} nan={nan} (bound {bound:.3e})");
        }
    }
    eprintln!("[g-stress] {iters} iters OK, worst gmax={worst:.3e} (bound {bound:.3e})");
}

/// Full-block stress repro for the **non-deterministic, CUDA-only** Quaternion4D
/// NaN. The isolated `g` repro ([`g_split_view_stress`]) is clean even on CUDA,
/// so the corruption needs the full forward context: the AE-config Mamba3 block
/// (matching midi-gen: d_model=384, state_rank=256, mimo_rank=4, ngroups=1,
/// per_head_dim=32, Quaternion4D, dt_limit=(0,1)), the `SerialRecalculated` SSD
/// path, and — critically — a **backward through the custom recompute scan**
/// (`quat_cumprod_recalculated`), the one kernel path the forward-only repro
/// never exercises.
///
/// Runs many forward+backward iterations and panics on the first non-finite
/// output. Deterministically clean on CPU; a breach (esp. on CUDA) is the repro.
/// Run explicitly (slow):
///   `cargo test --features backend-cuda --lib quaternion_ae_forward_stress -- --nocapture --test-threads=1`
/// `STRESS_ITERS` env overrides the iteration count (default 1000).
#[test]
#[ignore]
fn quaternion_ae_forward_stress() {
    use crate::mamba3::mamba3::Mamba3Config;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    let device: Device = Default::default();
    let model = Mamba3Config::new(384)
        .with_state_rank(256)
        .with_expand(1)
        .with_per_head_dim(32)
        .with_ngroups(1)
        .with_mimo_rank(4)
        .with_rope_fraction(1.0)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        .with_rotation(RotationKind::Quaternion4D)
        .with_dt_limit((0.0, 1.0))
        .init(&device.clone().autodiff());

    let (batch, seq) = (16, 4); // midi-gen: 16 channels, short patches.
    let iters = std::env::var("STRESS_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000usize);

    for i in 0..iters {
        // Fresh autodiff leaf each step (params on the autodiff device are
        // grad-tracked, so the graph + recompute backward run regardless).
        let input = Tensor::from_inner(Tensor::<3>::random(
            [batch, seq, 384],
            Distribution::Normal(0.0, 1.0),
            &device,
        ));
        let (out, _cache) =
            model.forward(input, None, Mamba3SsdPath::SerialRecalculated(None));

        let nan = out.clone().is_nan().any().into_scalar::<bool>();
        if nan {
            let mx = out.clone().abs().max().into_scalar::<f32>();
            panic!("[ae-stress] non-finite forward output at iter {i} (max={mx:.3e})");
        }

        // Exercise the recompute backward of the quaternion scan.
        let loss = (out.clone() * out).sum();
        let _grads = loss.backward();

        eprintln!("[ae-stress] iter {i} ok");
    }
    eprintln!("[ae-stress] {iters} iters OK (no NaN)");
}

/// **Most faithful** synthetic repro of midi-gen's Quaternion4D NaN. Adds the two
/// ingredients the single-block stress lacked and that match the failure's
/// signature (NaN at iter 2 = first patch consuming a non-trivial carried cache):
///
///   1. **Multi-layer `Layers<Mamba3>`** (default 14, like the AE's 6 enc + 8 dec)
///      — depth + the VRAM/graph footprint that pushes toward the allocator cap.
///   2. **Cross-patch detached cache carry** — the returned `Mamba3Caches` are
///      `set_require_grad(false).detach()`'d and threaded into the next iteration,
///      exactly as `midi-gen::utils::cache::unset_grad_and_detach` does.
///
/// Plus forward→backward each step (the recompute scan). Deterministically clean
/// on CPU; a breach (esp. on CUDA, ideally with VRAM pushed near the cap via the
/// env knobs below) is the repro we want for an upstream cubecl issue.
///
/// Run (push VRAM toward the 6 GB cap with the env knobs, then watch for a breach):
///   `STRESS_ITERS=5000 STRESS_BATCH=16 STRESS_SEQ=64 STRESS_LAYERS=14 \
///      cargo test --features backend-cuda --lib quaternion_ae_layers_cache_stress \
///      -- --ignored --nocapture --test-threads=1`
#[test]
// #[ignore]
fn quaternion_ae_layers_cache_stress() {
    use crate::mamba3::cache::Mamba3Caches;
    use crate::mamba3::mamba3::Mamba3Config;
    use crate::mamba3::single_ssd::cache::{Mamba3SingleSsdCache, Mamba3SingleSsdCaches};
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    use crate::modules::LayersBuilder;

    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let iters = env("STRESS_ITERS", 1000);
    let batch = env("STRESS_BATCH", 16); // midi-gen: 16 channels × batch_size
    let seq = env("STRESS_SEQ", 16);
    let n_layer = env("STRESS_LAYERS", 14); // 6 encoder + 8 decoder

    let device: Device = Default::default();
    let d_model = 384;
    let block = Mamba3Config::new(d_model)
        .with_state_rank(256)
        .with_expand(1)
        .with_per_head_dim(32)
        .with_ngroups(1)
        .with_mimo_rank(4)
        .with_rope_fraction(1.0)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        .with_rotation(RotationKind::Quaternion4D)
        .with_dt_limit((0.0, 1.0));
    let layers = LayersBuilder::<Mamba3Config>::new(n_layer, block).init(&device.clone().autodiff());

    // Inline copy of midi-gen's `unset_grad_and_detach` (single-ssd path).
    fn detach_caches(caches: Mamba3Caches) -> Mamba3Caches {
        match caches {
            Mamba3Caches::DoubleSsd(_) => unimplemented!("single-ssd path only"),
            Mamba3Caches::SingleSsd(cs) => {
                let detached = cs
                    .caches
                    .into_iter()
                    .map(|c| Mamba3SingleSsdCache {
                        ssm_bhpr: c.ssm_bhpr.set_require_grad(false).detach(),
                        k_state_bmhr: c.k_state_bmhr.set_require_grad(false).detach(),
                        v_state_bhp: c.v_state_bhp.set_require_grad(false).detach(),
                        rotation: match c.rotation {
                            RotationState::Angle(t) => {
                                RotationState::Angle(t.set_require_grad(false).detach())
                            }
                            RotationState::Quaternion(t) => {
                                RotationState::Quaternion(t.set_require_grad(false).detach())
                            }
                        },
                    })
                    .collect();
                Mamba3Caches::SingleSsd(Mamba3SingleSsdCaches { caches: detached })
            }
        }
    }

    let mut caches: Option<Mamba3Caches> = None;
    for i in 0..iters {
        let x = Tensor::from_inner(Tensor::<3>::random(
            [batch, seq, d_model],
            Distribution::Normal(0.0, 1.0),
            &device,
        ));
        let (out, new_caches) =
            layers.forward(x, caches.take(), Mamba3SsdPath::SerialRecalculated(None));

        let nan = out.clone().is_nan().any().into_scalar::<bool>();
        if nan {
            let mx = out.clone().abs().max().into_scalar::<f32>();
            panic!("[ae-layers-stress] non-finite output at iter {i} (max={mx:.3e})");
        }

        let loss = (out.clone() * out).sum();
        let _grads = loss.backward();

        // Cross-patch carry: detach and thread forward (like midi-gen patches).
        caches = Some(detach_caches(new_caches));

        eprintln!("[ae-layers-stress] iter {i} ok");
    }
    eprintln!("[ae-layers-stress] {iters} iters OK (no NaN)");
}
