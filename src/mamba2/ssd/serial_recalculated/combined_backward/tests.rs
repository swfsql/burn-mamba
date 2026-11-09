use crate::mamba2::ssd::serial;
use crate::standard_ssd;
use burn_stack::utils::fprim::F;
use burn::backend::Dispatch;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::{dtype_tol, test_device};

/// The primitive backend the high-level `Tensor` is pinned to; `combined_backward`
/// runs generically over it here (dispatching to the feature-selected backend).
type B = Dispatch;

/// Oracle for the local (BLUE+ORANGE) part of `d_da_cumsum`.
///
/// `_chunk_scan_bwd_ddAcs_unstable` (ssd_chunk_scan.py:1618) gives this
/// identity with the ORANGE `d_dt`. The standard SSD scan takes `v = dt·x`, so
/// the identity here uses the ORANGE `d_v`:
/// ```text
/// d_da_cumsum_local[b,h,n,l]
///   = Σ_p out_x[b,n,l,h,p] · d_y[b,n,l,h,p]
///     − Σ_p d_v_orange[b,n,l,h,p] · v[b,n,l,h,p]
/// ```
/// where `out_x = y − D·x` (output before D-skip). The two forms agree because
/// `d_dt_orange·dt = Σ_p d_v_orange·x·dt`.
///
/// This is a stronger correctness check than the autodiff comparison:
/// both sides are computed via independent analytical paths and should
/// match to fp32 precision on small inputs. The identity is numerically
/// unstable for large inputs (catastrophic cancellation between the two
/// sums).
#[test]
fn oracle_da_local_matches_einsum_minus_ddt_dt() {
    let device = test_device();
    let batch = 2;
    let nchunks = 3;
    let chunk_len = 4;
    let nheads = 4;
    let per_head_dim = 4;
    let state_rank = 6;

    // ─── Random inputs (small, gentle distributions) ─────────────────
    let x_bnlhp = Tensor::<5>::random(
        [batch, nchunks, chunk_len, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let dt_bnlh = Tensor::<4>::random(
        [batch, nchunks, chunk_len, nheads],
        Distribution::Uniform(0.05, 0.3),
        &device,
    );
    let a_decay_h = Tensor::<1>::random([nheads], Distribution::Uniform(-1.0, -0.5), &device);
    let b_bnlhr = Tensor::<5>::random(
        [batch, nchunks, chunk_len, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let c_bnlhr = Tensor::<5>::random(
        [batch, nchunks, chunk_len, nheads, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let d_h = Tensor::<1>::random([nheads], Distribution::Normal(0.0, 0.1), &device);
    let initial_state_bhpr = Tensor::<4>::random(
        [batch, nheads, per_head_dim, state_rank],
        Distribution::Normal(0.0, 0.1),
        &device,
    );
    let dt_discretized_bhnl = dt_bnlh.clone().permute([0, 3, 1, 2]);

    // ─── Forward (Serial path) ───────────────────────────────────────
    let (da_cumsum_bhnl, da_chunk_end_bhn) =
        serial::k1_ssd_chunk_cumsum(dt_discretized_bhnl.clone(), a_decay_h.clone());
    let cb_bnhll = serial::k2_ssd_bmm(c_bnlhr.clone(), b_bnlhr.clone());
    let intra_chunk_state_bnhpr = serial::k3_ssd_chunk_state(
        x_bnlhp.clone(),
        b_bnlhr.clone(),
        da_cumsum_bhnl.clone(),
        dt_discretized_bhnl.clone(),
    );
    let (chunk_input_state_bnhpr, _final_state_bhpr) = serial::k4_ssd_state_passing(
        intra_chunk_state_bnhpr,
        da_chunk_end_bhn,
        initial_state_bhpr.clone(),
        None,
    );
    let y_bnlhp = serial::k5_ssd_chunk_scan(
        da_cumsum_bhnl,
        dt_discretized_bhnl,
        x_bnlhp.clone(),
        c_bnlhr.clone(),
        cb_bnhll,
        chunk_input_state_bnhpr,
        d_h.clone(),
    );

    // ─── out_x = y − D·x  (output before D-skip) ─────────────────────
    let skip_bnlhp = d_h
        .clone()
        .unsqueeze_dims::<5>(&[0, 1, 2, 4]) // d_111h1
        .expand([batch, nchunks, chunk_len, nheads, per_head_dim]) // d_bnlhp
        * x_bnlhp.clone();
    let out_x_bnlhp = y_bnlhp - skip_bnlhp;

    // ─── Random upstream gradients ────────────────────────────────────
    let d_y_bnlhp = Tensor::<5>::random(
        [batch, nchunks, chunk_len, nheads, per_head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let d_final_bhpr = Tensor::<4>::random(
        [batch, nheads, per_head_dim, state_rank],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // ─── The inputs of the standard SSD scan ──────────────────────────
    let v_bnlhp = x_bnlhp * dt_bnlh.clone().unsqueeze_dim::<5>(4);
    let da_bnlh = dt_bnlh * a_decay_h.unsqueeze_dims::<4>(&[0, 1, 2]);

    // ─── Run the scan backward ────────────────────────────────────────
    // It returns d_da_local and d_v_orange. It runs on `F<B>` primitives. So
    // wrap the `Tensor` inputs here, and unwrap the `F` outputs below.
    let grads = standard_ssd::combined_backward::<B>(
        F::new(d_y_bnlhp.clone().unsqueeze_dim::<6>(3).into_dispatch()),
        F::new(d_final_bhpr.into_dispatch()),
        F::new(v_bnlhp.clone().unsqueeze_dim::<6>(3).into_dispatch()),
        F::new(da_bnlh.into_dispatch()),
        F::new(b_bnlhr.unsqueeze_dim::<6>(3).into_dispatch()),
        F::new(c_bnlhr.unsqueeze_dim::<6>(3).into_dispatch()),
        F::new(initial_state_bhpr.into_dispatch()),
        1,
        None,
    );
    let d_da_local_bhnl = Tensor::<4>::from_dispatch(grads.d_da_local_bhnl.inner());
    let d_v_orange_bnlhp =
        Tensor::<6>::from_dispatch(grads.d_v_orange_bnlmhp.inner()).squeeze_dim::<5>(3);

    // ─── Oracle: Σ_p out_x·d_y − Σ_p d_v_orange·v ────────────────────
    let sum_p_bhnl = |lhs_bnlhp: Tensor<5>, rhs_bnlhp: Tensor<5>| -> Tensor<4> {
        (lhs_bnlhp * rhs_bnlhp)
            .sum_dim(4) // _bnlh1
            .squeeze_dim::<4>(4) // _bnlh
            .permute([0, 3, 1, 2]) // _bhnl
    };
    let oracle_bhnl = sum_p_bhnl(out_x_bnlhp, d_y_bnlhp) - sum_p_bhnl(d_v_orange_bnlhp, v_bnlhp);

    // ─── Compare ─────────────────────────────────────────────────────
    let diff: f32 = (d_da_local_bhnl - oracle_bhnl)
        .abs()
        .max()
        .into_scalar::<f32>();
    assert!(
        diff < dtype_tol(1e-3),
        "d_da_local oracle identity violated; max abs diff = {diff}",
    );
}
