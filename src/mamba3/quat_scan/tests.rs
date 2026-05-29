//! Tests for the memory-efficient quaternion scan: the recompute-backward
//! [`quat_cumprod_recalculated`] must equal the plain-autodiff reference
//! [`quat_cumprod`](crate::mamba3::rotation::quat_cumprod) on **values** and
//! **gradients** (the recompute math is exact for the unit quaternions the
//! rotation produces). Only the backward's memory profile differs.

use super::quat_cumprod_recalculated;
use crate::mamba3::rotation::{quat_cumprod, quat_normalize};
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

const VAL_TOL: f32 = 1e-4;
const GRAD_TOL: f32 = 1e-3;

#[test]
fn recalculated_matches_quat_cumprod_values() {
    let device: Device = Default::default();
    // Non-power-of-two sequence exercises the identity-padded shift edges.
    let (batch, sequence, nheads, blocks) = (2, 13, 3, 2);
    let q = quat_normalize(Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));

    // Fresh start and a continued (random unit) carry.
    let carry = quat_normalize(Tensor::<4>::random(
        [batch, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    for init in [None, Some(carry)] {
        let (cum_r, fin_r) = quat_cumprod_recalculated(q.clone(), init.clone());
        let (cum_s, fin_s) = quat_cumprod(q.clone(), init);
        let dc = max_abs_diff(cum_r, cum_s);
        let df = max_abs_diff(fin_r, fin_s);
        assert!(dc < VAL_TOL, "recalculated vs reference cum: {dc:.6}");
        assert!(
            df < VAL_TOL,
            "recalculated vs reference final carry: {df:.6}"
        );
    }
}

#[test]
fn recalculated_matches_quat_cumprod_grads() {
    let device: Device = Default::default();
    let (batch, sequence, nheads, blocks) = (2, 11, 2, 2);
    let q_raw = Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let init_raw = Tensor::<4>::random(
        [batch, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let head = Tensor::<5>::random(
        [batch, sequence, nheads, blocks, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Backprop a scalar loss touching BOTH `cum` and `final_carry` (so the
    // final-carry slice's gradient path is exercised) through the (normalised)
    // quaternion inputs `q` and the carry `init`. Both inputs must receive the
    // same gradient from each scan.
    let grad_for = |recalc: bool| -> (Tensor<5>, Tensor<4>) {
        let pq = Param::from_tensor(Tensor::from_inner(q_raw.clone()));
        let pi = Param::from_tensor(Tensor::from_inner(init_raw.clone()));
        let q = quat_normalize(pq.val());
        let init = quat_normalize(pi.val());
        let (cum, final_carry) = if recalc {
            quat_cumprod_recalculated(q, Some(init))
        } else {
            quat_cumprod(q, Some(init))
        };
        let loss = (cum * Tensor::from_inner(head.clone())).sum() + final_carry.sum();
        let grads = loss.backward();
        let d_q = pq.val().grad(&grads).expect("grad q");
        let d_init = pi.val().grad(&grads).expect("grad init");
        (d_q, d_init)
    };

    let (dq_r, di_r) = grad_for(true);
    let (dq_s, di_s) = grad_for(false);
    let dq = max_abs_diff(dq_r, dq_s);
    let di = max_abs_diff(di_r, di_s);
    assert!(
        dq < GRAD_TOL,
        "recalculated vs reference q-grad: {dq:.6} (tol {GRAD_TOL})"
    );
    assert!(
        di < GRAD_TOL,
        "recalculated vs reference init-grad: {di:.6} (tol {GRAD_TOL})"
    );
}
