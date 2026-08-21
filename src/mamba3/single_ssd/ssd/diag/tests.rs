//! The SISO fast path must be an exact reformulation of the MIMO-general path.
//!
//! [`y_diag_correction`] dispatches to [`y_diag_correction_siso`] whenever
//! `mimo_rank == 1`, so the general branch is unreachable through the public
//! entry point at that rank. These tests call the two branches directly and
//! compare them (values *and* gradients).

use super::*;
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// `(v, b, c, gamma)` at the given shape. `gamma` is non-negative, matching the
/// `Δ·σ(λ)` output of `helpers::trapezoidal_coefficients`.
#[allow(clippy::too_many_arguments)]
fn random_input(
    batch: usize,
    nchunks: usize,
    chunk_len: usize,
    mimo_rank: usize,
    nheads: usize,
    per_head_dim: usize,
    state_rank: usize,
    device: &Device,
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

/// Forward output plus the gradient of every input, for one branch.
struct Run {
    y: Tensor<6>,
    d_v: Tensor<6>,
    d_b: Tensor<6>,
    d_c: Tensor<6>,
    d_gamma: Tensor<4>,
}

/// Run `f` on autodiff-tracked copies of the (plain) inputs and collect the
/// output plus every input gradient, all as plain tensors.
fn run(
    f: impl Fn(Tensor<6>, Tensor<6>, Tensor<6>, Tensor<4>) -> Tensor<6>,
    v: Tensor<6>,
    b: Tensor<6>,
    c: Tensor<6>,
    gamma: Tensor<4>,
    head: Tensor<6>,
) -> Run {
    let v = Param::from_tensor(Tensor::from_inner(v));
    let b = Param::from_tensor(Tensor::from_inner(b));
    let c = Param::from_tensor(Tensor::from_inner(c));
    let gamma = Param::from_tensor(Tensor::from_inner(gamma));

    let y = f(v.val(), b.val(), c.val(), gamma.val());
    let y_inner = y.clone().inner();
    // A random head makes the loss sensitive to every output element (a plain
    // `sum()` would hide sign-symmetric errors).
    let loss = (y * Tensor::from_inner(head)).sum();
    let grads = loss.backward();

    Run {
        y: y_inner,
        d_v: v.val().grad(&grads).unwrap(),
        d_b: b.val().grad(&grads).unwrap(),
        d_c: c.val().grad(&grads).unwrap(),
        d_gamma: gamma.val().grad(&grads).unwrap(),
    }
}

/// At `mimo_rank == 1` the SISO branch reproduces the MIMO branch exactly, on
/// the forward output and on every input gradient.
#[test]
fn siso_matches_mimo_forward_and_grads() {
    let device = Device::default();
    let (batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim, state_rank) =
        (2, 3, 5, 1, 4, 6, 8);

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
    let head = Tensor::<6>::random(
        [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let r_mimo = run(
        y_diag_correction_mimo,
        v.clone(),
        b.clone(),
        c.clone(),
        gamma.clone(),
        head.clone(),
    );
    let r_siso = run(y_diag_correction_siso, v, b, c, gamma, head);

    let tol = 1e-5;
    for (name, d) in [
        ("y", max_abs_diff(r_mimo.y, r_siso.y)),
        ("d_v", max_abs_diff(r_mimo.d_v, r_siso.d_v)),
        ("d_b", max_abs_diff(r_mimo.d_b, r_siso.d_b)),
        ("d_c", max_abs_diff(r_mimo.d_c, r_siso.d_c)),
        ("d_gamma", max_abs_diff(r_mimo.d_gamma, r_siso.d_gamma)),
    ] {
        assert!(d < tol, "{name}: mimo↔siso max abs diff {d} >= {tol}");
    }
}

/// The public entry point must agree with the general branch called directly,
/// whichever branch it picks: SISO at `mimo_rank == 1` with `siso_specialization`
/// on, the MIMO branch otherwise.
#[test]
fn dispatch_agrees_with_both_branches() {
    let device = Device::default();
    let (batch, nchunks, chunk_len, nheads, per_head_dim, state_rank) = (2, 2, 4, 3, 6, 8);

    for mimo_rank in [1, 3] {
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
        for siso_specialization in [true, false] {
            let dispatched = y_diag_correction(
                v.clone(),
                b.clone(),
                c.clone(),
                gamma.clone(),
                siso_specialization,
            );
            let direct = y_diag_correction_mimo(v.clone(), b.clone(), c.clone(), gamma.clone());
            let d = max_abs_diff(dispatched, direct);
            assert!(
                d < 1e-5,
                "mimo_rank={mimo_rank} siso_specialization={siso_specialization}: \
                 dispatch mismatch {d}"
            );
        }
    }
}
