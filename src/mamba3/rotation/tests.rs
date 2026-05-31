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
use crate::mamba3::double_ssd::double_ssd::apply_rope;
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
    use crate::modules::bidi::OutputMergeConfig;
    use crate::modules::{MambaBidiLayersConfig, MambaSsdPath};
    use crate::mamba3::mamba3::Mamba3Config;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
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
