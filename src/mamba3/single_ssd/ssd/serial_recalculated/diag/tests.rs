//! Parity for the primitive γ-correction: the SISO fast path must reproduce the
//! MIMO-general path (forward *and* analytic backward), and the analytic
//! backward must reproduce autodiff through the high-level forward.

use super::*;
use crate::mamba3::single_ssd::ssd::diag;
use burn_stack::utils::test_helpers::max_abs_diff;
use burn::backend::Dispatch;
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Distribution;

type Dev = burn::prelude::Device;
type P<const D: usize> = F<Dispatch, D>;

fn to_prim<const D: usize>(t: Tensor<D>) -> P<D> {
    F::new(t.into_dispatch())
}

fn to_tensor<const D: usize>(f: P<D>) -> Tensor<D> {
    Tensor::from_dispatch(f.inner())
}

/// `(v, b, c, gamma)` at the given shape; `gamma` non-negative like `Δ·σ(λ)`.
#[allow(clippy::too_many_arguments)]
fn random_input(
    batch: usize,
    nchunks: usize,
    chunk_len: usize,
    mimo_rank: usize,
    nheads: usize,
    per_head_dim: usize,
    state_rank: usize,
    device: &Dev,
) -> (Tensor<6>, Tensor<6>, Tensor<6>, Tensor<4>) {
    let v = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let b = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let c = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let gamma = Tensor::<4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Uniform(0.05, 0.5),
        device,
    );
    (v, b, c, gamma)
}

const SHAPE: (usize, usize, usize, usize, usize, usize) = (2, 3, 5, 4, 6, 8);

/// At `mimo_rank == 1` the two forward branches agree.
#[test]
fn siso_matches_mimo_forward() {
    let device = Dev::default();
    let (batch, nchunks, chunk_len, nheads, per_head_dim, state_rank) = SHAPE;
    let (v, b, c, gamma) = random_input(
        batch,
        nchunks,
        chunk_len,
        1,
        nheads,
        per_head_dim,
        state_rank,
        &device,
    );

    let y_mimo = to_tensor(y_diag_correction_mimo::<Dispatch>(
        to_prim(v.clone()),
        to_prim(b.clone()),
        to_prim(c.clone()),
        to_prim(gamma.clone()),
    ));
    let y_siso = to_tensor(y_diag_correction_siso::<Dispatch>(
        to_prim(v),
        to_prim(b),
        to_prim(c),
        to_prim(gamma),
    ));

    let d = max_abs_diff(y_mimo, y_siso);
    assert!(d < 1e-5, "forward: mimo↔siso max abs diff {d}");
}

/// At `mimo_rank == 1` the two analytic backward branches agree on all four
/// gradients.
#[test]
fn siso_matches_mimo_backward() {
    let device = Dev::default();
    let (batch, nchunks, chunk_len, nheads, per_head_dim, state_rank) = SHAPE;
    let (v, b, c, gamma) = random_input(
        batch,
        nchunks,
        chunk_len,
        1,
        nheads,
        per_head_dim,
        state_rank,
        &device,
    );
    let d_y = Tensor::<6>::random(
        [batch, nchunks, chunk_len, 1, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let g_mimo = y_diag_correction_backward_mimo::<Dispatch>(
        to_prim(d_y.clone()),
        to_prim(v.clone()),
        to_prim(b.clone()),
        to_prim(c.clone()),
        to_prim(gamma.clone()),
    );
    let g_siso = y_diag_correction_backward_siso::<Dispatch>(
        to_prim(d_y),
        to_prim(v),
        to_prim(b),
        to_prim(c),
        to_prim(gamma),
    );

    let tol = 1e-5;
    for (name, d) in [
        (
            "d_v",
            max_abs_diff(to_tensor(g_mimo.d_v_bnlmhp), to_tensor(g_siso.d_v_bnlmhp)),
        ),
        (
            "d_c",
            max_abs_diff(to_tensor(g_mimo.d_c_bnlmhr), to_tensor(g_siso.d_c_bnlmhr)),
        ),
        (
            "d_b",
            max_abs_diff(to_tensor(g_mimo.d_b_bnlmhr), to_tensor(g_siso.d_b_bnlmhr)),
        ),
        (
            "d_gamma",
            max_abs_diff(
                to_tensor(g_mimo.d_gamma_bnlh),
                to_tensor(g_siso.d_gamma_bnlh),
            ),
        ),
    ] {
        assert!(d < tol, "{name}: mimo↔siso max abs diff {d} >= {tol}");
    }
}

/// The analytic backward reproduces autodiff through the high-level forward —
/// at `mimo_rank == 1` (the fast path) and above it (the general path).
#[test]
fn analytic_backward_matches_autodiff() {
    let (batch, nchunks, chunk_len, nheads, per_head_dim, state_rank) = SHAPE;

    for mimo_rank in [1, 3] {
        // Inputs live on the plain device; `Tensor::from_inner` lifts them onto
        // the autodiff graph for the reference run, so `grad()` comes back
        // plain and is directly comparable with the analytic result.
        let device = Dev::default();
        let (v, b, c, gamma) = random_input(
            batch,
            nchunks,
            chunk_len,
            mimo_rank,
            nheads,
            per_head_dim,
            state_rank,
            &device,
        );
        // With `loss = Σ (y ⊙ head)`, the upstream gradient `d_y` *is* `head`.
        let head = Tensor::<6>::random(
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // At `mimo_rank == 1` both branches are reachable; above it only the
        // general one is. Forward and analytic backward must use the same one.
        let specializations: &[bool] = if mimo_rank == 1 {
            &[true, false]
        } else {
            &[true]
        };
        for &siso in specializations {
            let pv = Param::from_tensor(Tensor::from_inner(v.clone()));
            let pb = Param::from_tensor(Tensor::from_inner(b.clone()));
            let pc = Param::from_tensor(Tensor::from_inner(c.clone()));
            let pgamma = Param::from_tensor(Tensor::from_inner(gamma.clone()));
            let y = diag::y_diag_correction(pv.val(), pb.val(), pc.val(), pgamma.val(), siso);
            let grads = (y * Tensor::from_inner(head.clone())).sum().backward();

            let analytic = y_diag_correction_backward::<Dispatch>(
                to_prim(head.clone()),
                to_prim(v.clone()),
                to_prim(b.clone()),
                to_prim(c.clone()),
                to_prim(gamma.clone()),
                siso,
            );

            let tol = 1e-4;
            for (name, d) in [
                (
                    "d_v",
                    max_abs_diff(
                        pv.val().grad(&grads).unwrap(),
                        to_tensor(analytic.d_v_bnlmhp),
                    ),
                ),
                (
                    "d_b",
                    max_abs_diff(
                        pb.val().grad(&grads).unwrap(),
                        to_tensor(analytic.d_b_bnlmhr),
                    ),
                ),
                (
                    "d_c",
                    max_abs_diff(
                        pc.val().grad(&grads).unwrap(),
                        to_tensor(analytic.d_c_bnlmhr),
                    ),
                ),
                (
                    "d_gamma",
                    max_abs_diff(
                        pgamma.val().grad(&grads).unwrap(),
                        to_tensor(analytic.d_gamma_bnlh),
                    ),
                ),
            ] {
                assert!(
                    d < tol,
                    "mimo_rank={mimo_rank} siso_specialization={siso} {name}: \
                 analytic↔autodiff max abs diff {d} >= {tol}"
                );
            }
        }
    }
}
