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
use burn::module::Param;
use burn::tensor::Distribution;
use crate::utils::test_helpers::max_abs_diff;

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
    let alpha = Tensor::<3>::random([batch, sequence, nheads], Distribution::Normal(-0.5, 0.1), device)
        .exp()
        .clamp(0.0, 1.0);
    let x = Tensor::<3>::random([batch, sequence, nheads], normal, device);
    let b = Tensor::<4>::random([batch, sequence, nheads, state_rank], normal, device);
    let c = Tensor::<4>::random([batch, sequence, nheads, state_rank], normal, device);
    let init = if random_init {
        Tensor::<3>::random([batch, nheads, state_rank], Distribution::Normal(0.0, 0.1), device)
    } else {
        Tensor::<3>::zeros([batch, nheads, state_rank], device)
    };
    Inputs { q, alpha, x, b, c, init }
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
    assert!(d < VAL_TOL, "factored vs explicit: y max abs diff = {d:.6} (tol {VAL_TOL})");
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
    let q = quat_normalize(Tensor::<2>::random([16, 4], Distribution::Normal(0.0, 1.0), &device));
    let r = quat_to_rot4::<2, 3>(q); // [16, 4, 4]
    let rt = r.clone().transpose();
    let prod = r.matmul(rt); // should be identity per batch
    let eye = Tensor::<2>::eye(4, &device).unsqueeze_dim::<3>(0).expand([16, 4, 4]);
    let d = max_abs_diff(prod, eye);
    assert!(d < VAL_TOL, "L_q Lqᵀ ≠ I: max abs diff = {d:.6}");
}

#[test]
fn rot4_homomorphism() {
    // L_{a⊗b} == L_a · L_b, the property that makes the cumulative-product
    // scan equivalent to materialising and multiplying the 4×4 matrices.
    let device: Device = Default::default();
    let a = quat_normalize(Tensor::<2>::random([16, 4], Distribution::Normal(0.0, 1.0), &device));
    let b = quat_normalize(Tensor::<2>::random([16, 4], Distribution::Normal(0.0, 1.0), &device));

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
    let q = Tensor::cat(vec![half.clone().cos(), half.sin(), zeros.clone(), zeros], 4);

    let (cum, _final) = quat_cumprod(q, None);

    // Closed form: cumulative half-angle Φₜ/2 = cumsum(θ/2), rebuilt as a quaternion.
    let cum_half = (theta_bsh1 * 0.5).cumsum(1);
    let zeros2 = Tensor::<5>::zeros([batch, sequence, nheads, blocks, 1], &device);
    let expected = Tensor::cat(
        vec![cum_half.clone().cos(), cum_half.sin(), zeros2.clone(), zeros2],
        4,
    );

    let d = max_abs_diff(cum, expected);
    assert!(d < VAL_TOL, "single-axis cumprod ≠ half-angle cumsum: {d:.6}");
}

// ---------------------------------------------------------------------------
// 5. Gradient parity: factored vs explicit agree on input gradients
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
        factored_recurrence(q, p.alpha.val(), p.x.val(), p.b.val(), p.c.val(), p.init.val())
    } else {
        explicit_recurrence(q, p.alpha.val(), p.x.val(), p.b.val(), p.c.val(), p.init.val())
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
    let head = Tensor::<3>::random([batch, sequence, nheads], Distribution::Normal(0.0, 1.0), &device);

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
        assert!(d < GRAD_TOL, "grad {name}: factored vs explicit max abs diff = {d:.6} (tol {GRAD_TOL})");
    }
}
