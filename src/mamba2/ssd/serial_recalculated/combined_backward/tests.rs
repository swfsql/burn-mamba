use super::*;
use crate::mamba2::ssd::serial;
use crate::utils::fprim::F;
use burn::backend::Dispatch;
use burn::prelude::*;
use burn::tensor::Distribution;

/// The primitive backend the high-level `Tensor` is pinned to; `combined_backward`
/// runs generically over it here (dispatching to the feature-selected backend).
type B = Dispatch;

/// Oracle for the local (BLUE+ORANGE) part of `d_da_cumsum`.
///
/// Per Tri Dao's `_chunk_scan_bwd_ddAcs_unstable` (ssd_chunk_scan.py:1618):
/// ```text
/// d_da_cumsum_local[b,h,n,l]
///   = Σ_p out_x[b,n,l,h,p] · d_y[b,n,l,h,p]
///     − d_dt_orange[b,h,n,l] · dt[b,h,n,l]
/// ```
/// where `out_x = y − D·x` (output before D-skip) and `d_dt_orange` is
/// the same-chunk d_dt contribution from ORANGE only.
///
/// This is a stronger correctness check than the autodiff comparison:
/// both sides are computed via independent analytical paths and should
/// match to fp32 precision on small inputs. The identity is numerically
/// unstable for large inputs (catastrophic cancellation between einsum
/// and ddt·dt).
#[test]
fn oracle_da_local_matches_einsum_minus_ddt_dt() {
    let device = Default::default();
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
    let dt_discretized_bhnl = dt_bnlh.permute([0, 3, 1, 2]);

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
    );
    let y_bnlhp = serial::k5_ssd_chunk_scan(
        da_cumsum_bhnl,
        dt_discretized_bhnl.clone(),
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

    // ─── Run combined_backward (gets d_da_local + d_dt_orange) ───────
    // `combined_backward` operates on `F<B>` primitives, so wrap the high-level
    // `Tensor` inputs (and unwrap the `F` outputs back to `Tensor` below).
    let grads = combined_backward::<B>(
        F::new(d_y_bnlhp.clone().into_primitive()),
        F::new(d_final_bhpr.into_primitive()),
        F::new(x_bnlhp.into_primitive()),
        F::new(dt_discretized_bhnl.clone().into_primitive()),
        F::new(b_bnlhr.into_primitive()),
        F::new(c_bnlhr.into_primitive()),
        F::new(d_h.into_primitive()),
        F::new(initial_state_bhpr.into_primitive()),
        F::new(a_decay_h.into_primitive()),
    );
    let d_da_local_bhnl = Tensor::<4>::from_primitive(grads.d_da_local_bhnl.inner());
    let d_dt_orange_bhnl = Tensor::<4>::from_primitive(grads.d_dt_orange_bhnl.inner());

    // ─── Oracle: einsum(out_x, d_y, "bnlhp,bnlhp->bhnl") − d_dt_orange·dt
    let einsum_bhnl: Tensor<4> = (out_x_bnlhp * d_y_bnlhp)
        .sum_dim(4) // einsum_bnlh1
        .squeeze_dim::<4>(4) // einsum_bnlh
        .permute([0, 3, 1, 2]); // einsum_bhnl
    let oracle_bhnl = einsum_bhnl - d_dt_orange_bhnl * dt_discretized_bhnl;

    // ─── Compare ─────────────────────────────────────────────────────
    let diff: f32 = (d_da_local_bhnl - oracle_bhnl)
        .abs()
        .max()
        .into_scalar::<f32>();
    assert!(
        diff < 1e-3,
        "d_da_local oracle identity violated; max abs diff = {diff}",
    );
}
