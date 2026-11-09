//! # The kernels of the standard SSD scan
//!
//! The forward K1–K5 on backend primitives, and the analytic backward of each
//! kernel beside its forward. [`super::combined_backward()`] and the backward
//! of the single-SSD pathway assemble these functions. Only the terms of a
//! pathway stay outside this file.
//!
//! The K5 backward serves the two K5 forwards. The intra-chunk term of each
//! forward differs only in its mask diagonal and in a scale per source
//! position. The other term, BLUE, is the same in both.
//!
//! A backward takes each large tensor by value. A caller clones a tensor only
//! if the caller reads that tensor again.
//!
//! The kernels run under a generic backend `B`, where the high-level `Tensor`
//! (pinned to `Dispatch`) is not available. So the math uses the rank-tagged
//! [`F`] primitive wrapper.

#![allow(non_snake_case)]

use super::read_axis::{read_causal_mask, read_positions, scatter_read_positions};
use super::{backward_chunk_group, chunk_tokens};
use burn_stack::utils::fprim::{F, san};
use burn::backend::Backend;
use burn::tensor::s;

/// The forward of one SSD call: K1–K5.
///
/// # Arguments
/// - `v_bnlmhp`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
/// - `da_bnlh`:            `[batch, nchunks, chunk_len, nheads]`, the log-decay `Δ·A`
/// - `b_bnlmhr`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
/// - `c_bntmhr`:           `[batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]`
/// - `initial_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
/// - `read_stride`:        write positions per read position (`micro_steps`)
/// - `restarts`:           `(keep_bn, origin_bhpr)`, or `None`. `keep_bn` is
///   `0` at a chunk that starts a new segment of a packed row. `origin_bhpr`
///   is the state that such a chunk starts from.
///
/// # Returns
/// - `y_bntmhp`:         `[batch, nchunks, chunk_tokens, mimo_rank, nheads, per_head_dim]`
/// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
pub fn forward<B: Backend>(
    v_bnlmhp: F<B, 6>,
    da_bnlh: F<B, 4>,
    b_bnlmhr: F<B, 6>,
    c_bntmhr: F<B, 6>,
    initial_state_bhpr: F<B, 4>,
    read_stride: usize,
    restarts: Option<(F<B, 2>, F<B, 4>)>,
) -> (F<B, 6>, F<B, 4>) {
    let nchunks = v_bnlmhp.dims()[1];
    assert!(nchunks > 0, "sequence length must be at least 1");

    let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum::<B>(da_bnlh);
    san(&da_cumsum_bhnl);

    let cb_bnhTMLM = k2_ssd_bmm::<B>(c_bntmhr.clone(), b_bnlmhr.clone());
    san(&cb_bnhTMLM);

    let intra_chunk_state_bnhpr =
        k3_ssd_chunk_state::<B>(v_bnlmhp.clone(), b_bnlmhr, da_cumsum_bhnl.clone());
    san(&intra_chunk_state_bnhpr);

    let (chunk_input_state_bnhpr, final_state_bhpr) = k4_ssd_state_passing::<B>(
        intra_chunk_state_bnhpr,
        da_chunk_end_bhn,
        initial_state_bhpr,
        restarts,
        None,
    );
    san(&chunk_input_state_bnhpr);
    san(&final_state_bhpr);

    let y_bntmhp = k5_ssd_chunk_scan::<B>(
        da_cumsum_bhnl,
        v_bnlmhp,
        c_bntmhr,
        cb_bnhTMLM,
        chunk_input_state_bnhpr,
        read_stride,
    );
    san(&y_bntmhp);

    (y_bntmhp, final_state_bhpr)
}

/// K1: the cumulative sum of `da` inside each chunk.
///
/// Returns the intra-chunk cumsum `da_cumsum_bhnl` and the per-chunk last value
/// `da_chunk_end_bhn`.
pub(crate) fn k1_ssd_chunk_cumsum<B: Backend>(da_bnlh: F<B, 4>) -> (F<B, 4>, F<B, 3>) {
    let da_bhnl = da_bnlh.permute([0, 3, 1, 2]);
    let da_cumsum_bhnl = da_bhnl.cumsum(3);
    let da_chunk_end_bhn = da_cumsum_bhnl
        .clone()
        .slice(s![.., .., .., -1])
        .squeeze_dim::<3>(3);
    (da_cumsum_bhnl, da_chunk_end_bhn)
}

/// Backward of [`k1_ssd_chunk_cumsum`].
///
/// Forward: `da_cumsum[l] = Σ_{k ≤ l} da[k]`. So the gradient is the suffix sum
/// `d_da[l] = Σ_{k ≥ l} d_da_cumsum[k]`. The code forms it as the total minus
/// the shifted prefix sum: `d_da[l] = total − cumsum(d_da_cumsum)[l − 1]`, with
/// `cumsum[−1] = 0`.
///
/// Returns `d_da_bnlh`.
pub(crate) fn k1_ssd_chunk_cumsum_backward<B: Backend>(d_da_cumsum_bhnl: F<B, 4>) -> F<B, 4> {
    let [batch, nheads, nchunks, chunk_len] = d_da_cumsum_bhnl.dims();
    let device = d_da_cumsum_bhnl.device();
    let dtype = d_da_cumsum_bhnl.dtype();

    let d_total_bhnl = d_da_cumsum_bhnl
        .clone()
        .sum_dim(3) // d_da_cumsum_bhn1
        .expand([batch, nheads, nchunks, chunk_len]); // d_total_bhnl
    let prefix_bhnl = d_da_cumsum_bhnl.cumsum(3);
    let zeros_bhn1 = F::<B, 4>::zeros([batch, nheads, nchunks, 1], &device, dtype);
    let prefix_shifted_bhnl = F::cat(vec![zeros_bhn1, prefix_bhnl.narrow(3, 0, chunk_len - 1)], 3);
    let d_da_bhnl = d_total_bhnl - prefix_shifted_bhnl;
    san(&d_da_bhnl);

    d_da_bhnl.permute([0, 2, 3, 1])
}

/// Put a per-chunk gradient at the last write position of each chunk, with
/// zeros at the other positions.
///
/// K3 and K4 read the cumsum at the chunk end. This is the gradient of that
/// read.
pub(crate) fn scatter_chunk_end<B: Backend>(d_end_bhn: F<B, 3>, chunk_len: usize) -> F<B, 4> {
    let [batch, nheads, nchunks] = d_end_bhn.dims();
    let zeros_bhnL = F::<B, 4>::zeros(
        [batch, nheads, nchunks, chunk_len - 1],
        &d_end_bhn.device(),
        d_end_bhn.dtype(),
    );
    F::cat(vec![zeros_bhnL, d_end_bhn.unsqueeze_dim::<4>(3)], 3)
}

/// K2: the intra-chunk matrix `C·Bᵀ`, on the fused axes.
///
/// Returns the block matrix `cb_bnhTMLM`.
pub(crate) fn k2_ssd_bmm<B: Backend>(c_bntmhr: F<B, 6>, b_bnlmhr: F<B, 6>) -> F<B, 5> {
    // `C` has the read axis of the chunk, and `B` its write axis. They are
    // equal unless `micro_steps > 1`.
    let [batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank] = c_bntmhr.dims();
    let [.., chunk_len, _, _, _] = b_bnlmhr.dims();
    let c_bnTMhr = c_bntmhr.reshape([batch, nchunks, chunk_tokens * mimo_rank, nheads, state_rank]);
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
    let c_bnhTMr = c_bnTMhr.swap_dims(2, 3);
    let b_bnhrLM = b_bnLMhr.permute([0, 1, 3, 4, 2]);
    c_bnhTMr.matmul(b_bnhrLM)
}

/// Backward of [`k2_ssd_bmm`].
///
/// Forward: `cb = c @ bᵀ`, which contracts `state_rank`. So `d_c = d_cb @ b` and
/// `d_bᵀ = cᵀ @ d_cb`.
///
/// `b_bnLMhr` has the fused write axis. A caller that also gives the same
/// tensor to K3 reshapes it one time.
///
/// Returns `d_c_bntmhr` and `d_b_bnlmhr`.
pub(crate) fn k2_ssd_bmm_backward<B: Backend>(
    d_cb_bnhTMLM: F<B, 5>,
    c_bntmhr: F<B, 6>,
    b_bnLMhr: F<B, 5>,
) -> (F<B, 6>, F<B, 6>) {
    let [batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank] = c_bntmhr.dims();
    let [.., fused, _, _] = b_bnLMhr.dims();

    let c_bnhTMr = c_bntmhr
        .reshape([batch, nchunks, chunk_tokens * mimo_rank, nheads, state_rank]) // c_bnTMhr
        .swap_dims(2, 3); // c_bnhTMr
    let b_bnhLMr = b_bnLMhr.swap_dims(2, 3);

    let d_c_bnhTMr = d_cb_bnhTMLM.clone().matmul(b_bnhLMr);
    let d_b_bnhrLM = c_bnhTMr
        .transpose() // c_bnhrTM
        .matmul(d_cb_bnhTMLM); // d_b_bnhrLM

    let d_c_bntmhr = split_fused_axis(d_c_bnhTMr, chunk_tokens);
    let d_b_bnlmhr = d_b_bnhrLM
        .permute([0, 1, 4, 2, 3]) // d_b_bnLMhr
        .reshape([batch, nchunks, fused / mimo_rank, mimo_rank, nheads, state_rank]); // d_b_bnlmhr
    (d_c_bntmhr, d_b_bnlmhr)
}

/// Split a fused chunk axis: `[b, n, h, x·m, q]` to `[b, n, x, m, h, q]`.
/// `len` is `x`.
pub(crate) fn split_fused_axis<B: Backend>(t: F<B, 5>, len: usize) -> F<B, 6> {
    let [batch, nchunks, nheads, fused, last] = t.dims();
    t.swap_dims(2, 3) // [b, n, x·m, h, q]
        .reshape([batch, nchunks, len, fused / len, nheads, last])
}

/// The K3 decay of each write position to the end of its chunk, on the fused
/// write axis: `exp(cumA_last − cumA)`.
fn k3_decay<B: Backend>(da_cumsum_bhnl: F<B, 4>, mimo_rank: usize) -> F<B, 4> {
    let [batch, nheads, nchunks, chunk_len] = da_cumsum_bhnl.dims();
    let da_cumsum_last_bhn1 = da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4) // da_cumsum_bhnl1
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // da_cumsum_bhnlm
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // da_cumsum_bhnLM
    (da_cumsum_last_bhn1 - da_cumsum_bhnLM).exp()
}

/// K3: the end state of each chunk, from a zero state at the chunk start.
///
/// Returns `intra_chunk_state_bnhpr`: the contribution of each chunk to its
/// end state. `v_bnlmhp` is pre-scaled. [`k3_ssd_chunk_state_extended`] also
/// returns the intermediates of the backward, so it holds a clone of each
/// during its matmul. A forward does not need them.
pub(crate) fn k3_ssd_chunk_state<B: Backend>(
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    da_cumsum_bhnl: F<B, 4>,
) -> F<B, 5> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();

    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

    let decay_bhnLM = k3_decay(da_cumsum_bhnl, mimo_rank);

    let decay_bnLMh1 = decay_bhnLM.permute([0, 2, 3, 1]).unsqueeze_dim::<5>(4); // decay_bnLMh1
    let decayed_v_bnLMhp = decay_bnLMh1 * v_bnLMhp;

    let decayed_v_bnhpLM = decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);
    let b_bnhLMr = b_bnLMhr.swap_dims(2, 3);
    let intra_chunk_state_bnhpr = decayed_v_bnhpLM.matmul(b_bnhLMr);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        intra_chunk_state_bnhpr.dims()
    );
    intra_chunk_state_bnhpr
}

/// The K3 kernel `k3_ssd_chunk_state`, with the intermediates of the backward:
/// - `intra_chunk_state_bnhpr`: the chunk-end state from a zero initial state
/// - `decay_bhnLM`: the fused-length K3 decay factor `exp(cumA_last − cumA_fused)`
/// - `decayed_v_bnLMhp`: `v` already scaled by `decay_bnLMh1`
pub fn k3_ssd_chunk_state_extended<B: Backend>(
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    da_cumsum_bhnl: F<B, 4>,
) -> (F<B, 5>, F<B, 4>, F<B, 5>) {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();

    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

    let decay_bhnLM = k3_decay(da_cumsum_bhnl, mimo_rank);
    san(&decay_bhnLM);

    let decay_bnLMh1 = decay_bhnLM
        .clone()
        .permute([0, 2, 3, 1])
        .unsqueeze_dim::<5>(4);
    let decayed_v_bnLMhp = decay_bnLMh1 * v_bnLMhp;
    san(&decayed_v_bnLMhp);

    let decayed_v_bnhpLM = decayed_v_bnLMhp.clone().permute([0, 1, 3, 4, 2]);
    let b_bnhLMr = b_bnLMhr.swap_dims(2, 3);
    let intra_chunk_state_bnhpr = decayed_v_bnhpLM.matmul(b_bnhLMr);
    san(&intra_chunk_state_bnhpr);

    (intra_chunk_state_bnhpr, decay_bhnLM, decayed_v_bnLMhp)
}

/// Backward of [`k3_ssd_chunk_state_extended`].
///
/// Forward, on the fused length, with `k` for the input projection of K3:
/// `decay = exp(cumA_last − cumA)`, `decayed_v = decay · v` and
/// `intra_state = decayed_v @ k`.
///
/// `k_bnLMhr` has the fused write axis. A pathway that scales `B` gives
/// the scaled tensor and splits the returned gradient itself.
///
/// # Returns
/// - `d_v_bnlmhp`: the K3 part of `d_v`
/// - `d_k_bnlmhr`: the gradient of the projection `k`
/// - `d_da_cumsum_bhnl`: the K3 part of `d_da_cumsum`
pub(crate) fn k3_ssd_chunk_state_backward<B: Backend>(
    d_intra_chunk_state_bnhpr: F<B, 5>,
    decay_bhnLM: F<B, 4>,
    decayed_v_bnLMhp: F<B, 5>,
    v_bnlmhp: F<B, 6>,
    k_bnLMhr: F<B, 5>,
) -> (F<B, 6>, F<B, 6>, F<B, 4>) {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = k_bnLMhr.dims();

    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let k_bnhLMr = k_bnLMhr.swap_dims(2, 3);
    let decayed_v_bnhpLM = decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);

    let d_decayed_v_bnhpLM = d_intra_chunk_state_bnhpr.clone().matmul(
        k_bnhLMr.transpose(), // k_bnhrLM
    ); // d_decayed_v_bnhpLM
    let d_k_bnhLMr = decayed_v_bnhpLM
        .transpose() // decayed_v_bnhLMp
        .matmul(d_intra_chunk_state_bnhpr); // d_k_bnhLMr

    let d_decayed_v_bnLMhp = d_decayed_v_bnhpLM.permute([0, 1, 4, 2, 3]);
    let d_decay_bhnLM = (d_decayed_v_bnLMhp.clone() * v_bnLMhp)
        .sum_dim(4) // d_decay_bnLMh1
        .squeeze_dim::<4>(4) // d_decay_bnLMh
        .permute([0, 3, 1, 2]); // d_decay_bhnLM

    // d_v = d_decayed_v · decay (broadcast)
    let decay_bnLMh1 = decay_bhnLM
        .clone()
        .permute([0, 2, 3, 1]) // decay_bnLMh
        .unsqueeze_dim::<5>(4); // decay_bnLMh1
    let d_v_bnLMhp = d_decayed_v_bnLMhp * decay_bnLMh1;
    let d_v_bnlmhp =
        d_v_bnLMhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

    // d(cumA_last − cumA) = d_decay · decay
    let d_decay_times_decay_bhnLM = d_decay_bhnLM * decay_bhnLM;
    // cumA_last gets the sum over LM (the broadcast dim).
    let d_da_cumsum_last_bhn = d_decay_times_decay_bhnLM
        .clone()
        .sum_dim(3) // d_decay_times_decay_bhn1
        .squeeze_dim::<3>(3); // d_da_cumsum_last_bhn
    // cumA gets the negated gradient (subtraction).
    let d_da_cumsum_bhnLM = -d_decay_times_decay_bhnLM;

    // The fused cumA is an expand, so the gradient sums the `mimo_rank` copies.
    let d_da_cumsum_from_fused_bhnl = d_da_cumsum_bhnLM
        .reshape([batch, nheads, nchunks, chunk_len, mimo_rank]) // d_da_cumsum_bhnlm
        .sum_dim(4) // d_da_cumsum_bhnl1
        .squeeze_dim::<4>(4); // d_da_cumsum_from_fused_bhnl
    let d_da_cumsum_from_last_bhnl = scatter_chunk_end(d_da_cumsum_last_bhn, chunk_len);
    let d_da_cumsum_bhnl = d_da_cumsum_from_fused_bhnl + d_da_cumsum_from_last_bhnl;

    let d_k_bnlmhr = d_k_bnhLMr
        .swap_dims(2, 3) // d_k_bnLMhr
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]); // d_k_bnlmhr

    (d_v_bnlmhp, d_k_bnlmhr, d_da_cumsum_bhnl)
}

/// K4: the state passing across the chunks.
///
/// Returns the per-chunk input-state stream `chunk_input_state_bnhpr` and the
/// `final_state_bhpr`.
///
/// `restarts` is `(keep_bn, origin_bhpr)`: the reset as a factor, and the
/// origin state. The state that goes into a chunk with `keep = 0` is its
/// restart state, `origin + seedᶜ` (`origin` without seeds). The call starts
/// from `initial`.
pub(crate) fn k4_ssd_state_passing<B: Backend>(
    intra_chunk_state_bnhpr: F<B, 5>,
    da_chunk_end_bhn: F<B, 3>,
    initial_state_bhpr: F<B, 4>,
    restarts: Option<(F<B, 2>, F<B, 4>)>,
    seed_bnhpr: Option<F<B, 5>>,
) -> (F<B, 5>, F<B, 4>) {
    let [batch, nchunks, nheads, per_head_dim, state_rank] = intra_chunk_state_bnhpr.dims();
    let state = [nheads, per_head_dim, state_rank];
    let restarts = restarts.map(|(keep_bn, origin_bhpr)| {
        let restart_bn = restart_factor(&keep_bn);
        (keep_bn, restart_bn, origin_bhpr)
    });
    let restart = |origin_bhpr: &F<B, 4>, i_chunk: usize| match &seed_bnhpr {
        Some(seed_bnhpr) => {
            origin_bhpr.clone()
                + seed_bnhpr
                    .clone()
                    .slice(s![.., i_chunk, .., .., ..])
                    .squeeze_dim::<4>(1)
        }
        None => origin_bhpr.clone(),
    };

    let mut running_state_bhpr = initial_state_bhpr;
    let mut chunk_input_state_vec_bhpr = Vec::with_capacity(nchunks + 1);
    chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());

    for i_chunk in 0..nchunks {
        // A chunk that starts a segment starts from its restart state. (A
        // learnable `init_state_hpr` would join it, but only `Minimal` takes
        // one.)
        if let Some((keep_bn, restart_bn, origin_bhpr)) = &restarts {
            running_state_bhpr = running_state_bhpr * keep_at(keep_bn, i_chunk, state)
                + restart(origin_bhpr, i_chunk) * keep_at(restart_bn, i_chunk, state);
            *chunk_input_state_vec_bhpr.last_mut().unwrap() = running_state_bhpr.clone();
        }
        let intra_state_bhpr = intra_chunk_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);
        let decay_bhpr = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk])
            .unsqueeze_dim::<4>(3)
            .exp()
            .expand([batch, nheads, per_head_dim, state_rank]);
        running_state_bhpr = decay_bhpr * running_state_bhpr + intra_state_bhpr;
        chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());
    }

    let final_state_bhpr = chunk_input_state_vec_bhpr.pop().unwrap();
    let chunk_input_state_bnhpr = F::stack(chunk_input_state_vec_bhpr, 1);
    (chunk_input_state_bnhpr, final_state_bhpr)
}

/// `1 − keep`: `1` at a chunk that restarts.
fn restart_factor<B: Backend>(keep_bn: &F<B, 2>) -> F<B, 2> {
    F::<B, 2>::full(keep_bn.dims(), 1.0, &keep_bn.device(), keep_bn.dtype()) - keep_bn.clone()
}

/// The factor of chunk `i_chunk`, broadcast over one state
/// (`[batch, nheads, per_head_dim, state_rank]`).
fn keep_at<B: Backend>(keep_bn: &F<B, 2>, i_chunk: usize, [nheads, per_head_dim, state_rank]: [usize; 3]) -> F<B, 4> {
    let batch = keep_bn.dims()[0];
    keep_bn
        .clone()
        .slice(s![.., i_chunk])
        .reshape([batch, 1, 1, 1])
        .expand([batch, nheads, per_head_dim, state_rank])
}

/// Backward of [`k4_ssd_state_passing`]: the reverse of its scalar-decay scan.
/// It is the only part of the recompute backward that walks the chunks, as the
/// forward K4 does.
///
/// Forward, with `sᵢ` for `chunk_input_stateᵢ`: `sᵢ₊₁ = decayᵢ·sᵢ + intraᵢ`.
/// Reverse, seeded by `d_final`: `d_intraᵢ = d_sᵢ₊₁` and
/// `d_sᵢ = decayᵢ·d_sᵢ₊₁ + d_chunk_input_stateᵢ`. The walk ends on `d_initial`.
/// `d_decayᵢ = d_sᵢ₊₁·sᵢ` needs no walk. `decay` is one scalar per `(b, h, n)`,
/// so it comes out of the `(p, r)` sum, and the whole stream is one batched
/// product.
///
/// Under a reset (`keep_bn`), `sᵢ = keepᵢ · (the state before chunk i) +
/// (1 − keepᵢ) · restartᵢ`. So the gradient that goes on to chunk `i − 1` is
/// `keepᵢ · d_sᵢ`, and `restartᵢ` gets `(1 − keepᵢ) · d_sᵢ`. The call starts
/// from `initial`, so it gets what the walk leaves at the end. `sᵢ` in
/// `d_decay` is already the restarted one (the caller recomputed it).
///
/// # Returns
/// - `d_intra_chunk_state_bnhpr` — gradient of K3's per-chunk contribution
/// - `d_da_chunk_end_bhn` — gradient of the per-chunk log-decay
/// - `d_initial_state_bhpr` — what the walk leaves at the end
/// - `d_restart_bnhpr` — the gradient of each chunk's restart state
///   (`origin + seedᶜ`) with resets, else `None`. The origin gets its sum
///   over the chunks, and each seed gets its own chunk.
pub(crate) fn k4_ssd_state_passing_backward<B: Backend>(
    d_chunk_input_state_bnhpr: F<B, 5>,
    chunk_input_state_bnhpr: F<B, 5>,
    da_chunk_end_bhn: F<B, 3>,
    d_final_state_bhpr: F<B, 4>,
    keep_bn: Option<F<B, 2>>,
) -> (F<B, 5>, F<B, 3>, F<B, 4>, Option<F<B, 5>>) {
    let [batch, nchunks, nheads, per_head_dim, state_rank] = chunk_input_state_bnhpr.dims();
    let state = [nheads, per_head_dim, state_rank];
    let restart_bn = keep_bn.as_ref().map(restart_factor);

    let mut d_running_state_bhpr = d_final_state_bhpr;
    let mut d_intra_vec_bhpr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    // `d_restartᵢ`, in reverse chunk order.
    let mut d_restart_vec_bhpr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    for i_chunk in (0..nchunks).rev() {
        d_intra_vec_bhpr.push(d_running_state_bhpr.clone());

        let decay_bhpr: F<B, 4> = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk]) // da_chunk_end_bh1
            .exp()
            .unsqueeze_dim::<4>(3) // decay_bh11
            .expand([batch, nheads, per_head_dim, state_rank]);
        let d_chunk_input_state_bhpr: F<B, 4> = d_chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);

        d_running_state_bhpr = decay_bhpr * d_running_state_bhpr + d_chunk_input_state_bhpr;
        if let (Some(keep_bn), Some(restart_bn)) = (&keep_bn, &restart_bn) {
            d_restart_vec_bhpr.push(d_running_state_bhpr.clone() * keep_at(restart_bn, i_chunk, state));
            d_running_state_bhpr = d_running_state_bhpr * keep_at(keep_bn, i_chunk, state);
        }
        san(&d_running_state_bhpr);
    }
    d_intra_vec_bhpr.reverse();
    let d_intra_chunk_state_bnhpr: F<B, 5> = F::stack(d_intra_vec_bhpr, 1);

    // The walk ends on the state before chunk 0: `initial`.
    let d_initial_state_bhpr = d_running_state_bhpr;
    let d_restart_bnhpr = (!d_restart_vec_bhpr.is_empty()).then(|| {
        d_restart_vec_bhpr.reverse();
        F::stack(d_restart_vec_bhpr, 1)
    });

    let d_decay_bhn: F<B, 3> = (d_intra_chunk_state_bnhpr.clone() * chunk_input_state_bnhpr)
        .reshape([batch, nchunks, nheads, per_head_dim * state_rank])
        .sum_dim(3) // d_decay_bnh1
        .squeeze_dim::<3>(3) // d_decay_bnh
        .permute([0, 2, 1]); // d_decay_bhn
    // decay = exp(da_chunk_end), so the chain rule multiplies it back in.
    let d_da_chunk_end_bhn = d_decay_bhn * da_chunk_end_bhn.exp();
    san(&d_da_chunk_end_bhn);

    (
        d_intra_chunk_state_bnhpr,
        d_da_chunk_end_bhn,
        d_initial_state_bhpr,
        d_restart_bnhpr,
    )
}

/// Concatenate the per-group outputs of a chunk-group pass back along the
/// chunk axis. Nothing to do when the pass ran in one group, which is the case
/// for a short enough chunk axis (see
/// [`backward_chunk_group`]).
fn cat_chunk_groups<B: Backend, const D: usize>(
    mut groups: Vec<F<B, D>>,
    dim: usize,
) -> F<B, D> {
    match groups.len() {
        1 => groups.pop().expect("one group"),
        _ => F::cat(groups, dim),
    }
}

/// The K1 cumsum on the two chunk axes of K5, with `mimo_rank` fused in.
///
/// Returns `da_cumsum_bhnLM` on the write axis, and `da_cumsum_bhnTM` at the
/// read positions.
pub(crate) fn k5_da_cumsum_on_chunk_axes<B: Backend>(
    da_cumsum_bhnl: F<B, 4>,
    mimo_rank: usize,
    read_stride: usize,
) -> (F<B, 4>, F<B, 4>) {
    let [batch, nheads, nchunks, chunk_len] = da_cumsum_bhnl.dims();
    let chunk_tokens = chunk_tokens(chunk_len, read_stride);

    // The expand-then-reshape repeats each position `mimo_rank` times along
    // the fused axis.
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .clone()
        .unsqueeze_dim::<5>(4) // da_cumsum_bhnl1
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // da_cumsum_bhnlm
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // da_cumsum_bhnLM
    let da_cumsum_bhnTM = read_positions::<B, 4, 5>(da_cumsum_bhnl, 3, read_stride)
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_tokens, mimo_rank])
        .reshape([batch, nheads, nchunks, chunk_tokens * mimo_rank]);

    (da_cumsum_bhnLM, da_cumsum_bhnTM)
}

/// The causal mask of K5 on its fused axes.
///
/// The expansion repeats each row of the base mask `mimo_rank` times, and then
/// each column. So all the MIMO channels of one position have the same entry.
///
/// # Shapes
/// - `base_tl` : `[chunk_tokens, chunk_len]`, from [`read_causal_mask`]
/// - out       : `[batch, nchunks, nheads, chunk_tokens·mimo_rank, chunk_len·mimo_rank]`
pub(crate) fn expand_causal_mask<B: Backend>(
    base_tl: F<B, 2>,
    batch: usize,
    nchunks: usize,
    nheads: usize,
    mimo_rank: usize,
) -> F<B, 5> {
    let [chunk_tokens, chunk_len] = base_tl.dims();
    let read = chunk_tokens * mimo_rank;
    base_tl
        .unsqueeze_dims::<5>(&[0, 1, 2]) // base_111tl
        .expand([batch, nchunks, nheads, chunk_tokens, chunk_len]) // base_bnhtl
        .unsqueeze_dim::<6>(4) // base_bnht1l
        .expand([batch, nchunks, nheads, chunk_tokens, mimo_rank, chunk_len]) // base_bnhtml
        .reshape([batch, nchunks, nheads, read, chunk_len]) // base_bnhTMl
        .unsqueeze_dim::<6>(5) // base_bnhTMl1
        .expand([batch, nchunks, nheads, read, chunk_len, mimo_rank]) // base_bnhTMlm
        .reshape([batch, nchunks, nheads, read, chunk_len * mimo_rank]) // base_bnhTMLM
}

/// K5: the output of each chunk.
///
/// Adds the intra-chunk (ORANGE, MIMO causal) and inter-chunk (BLUE,
/// state-carried) contributions into the output `y_bntmhp`. It applies no `D`
/// skip: the caller adds it. [`k5_ssd_chunk_scan_backward`] is its backward.
fn k5_ssd_chunk_scan<B: Backend>(
    da_cumsum_bhnl: F<B, 4>,
    v_bnlmhp: F<B, 6>,
    c_bntmhr: F<B, 6>,
    cb_bnhTMLM: F<B, 5>,
    chunk_input_state_bnhpr: F<B, 5>,
    read_stride: usize,
) -> F<B, 6> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., chunk_tokens, _, _, state_rank] = c_bntmhr.dims();
    let device = v_bnlmhp.device();
    let dtype = v_bnlmhp.dtype();
    let fused = chunk_len * mimo_rank; // write (source) axis
    let read = chunk_tokens * mimo_rank; // read (target) axis

    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, fused, nheads, per_head_dim]);
    let c_bnTMhr = c_bntmhr.reshape([batch, nchunks, read, nheads, state_rank]);

    let (da_cumsum_bhnLM, da_cumsum_bhnTM) =
        k5_da_cumsum_on_chunk_axes::<B>(da_cumsum_bhnl, mimo_rank, read_stride);

    // ── BLUE (Y_off): exp(cumA[i]) · C[i] · h[n-1] ─────────────────────
    let exp_da_bnhTMp = da_cumsum_bhnTM
        .clone()
        .exp()
        .swap_dims(1, 2) // exp_da_bnhTM
        .unsqueeze_dim::<5>(4) // exp_da_bnhTM1
        .expand([batch, nchunks, nheads, read, per_head_dim]);
    let c_bnhTMr = c_bnTMhr.swap_dims(2, 3);
    let chunk_input_state_bnhrp = chunk_input_state_bnhpr.transpose();
    let ch_bnhTMp = c_bnhTMr.matmul(chunk_input_state_bnhrp);
    let blue_bnhTMp = ch_bnhTMp * exp_da_bnhTMp;

    // ── ORANGE (Y_diag): MIMO causal decay matrix · CB @ V ────────────────────
    let target_da_cumsum_bnhTMLM = da_cumsum_bhnTM
        .swap_dims(1, 2) // da_cumsum_bnhTM
        .unsqueeze_dim::<5>(4) // da_cumsum_bnhTM1
        .expand([batch, nchunks, nheads, read, fused]);
    let source_da_cumsum_bnhTMLM = da_cumsum_bhnLM
        .swap_dims(1, 2) // da_cumsum_bnhLM
        .unsqueeze_dim::<5>(3) // da_cumsum_bnh1LM
        .expand([batch, nchunks, nheads, read, fused]);
    let diff_da_cumsum_bnhTMLM = target_da_cumsum_bnhTMLM - source_da_cumsum_bnhTMLM;

    // MIMO causal neg-inf mask: −∞ where j//m > i//m (source strictly ahead of
    // target in time). `triu(1)` keeps the diagonal, because the sum of this
    // scan includes the same position.
    let neg_inf_bnhTMLM = expand_causal_mask(
        read_causal_mask::<B>(chunk_len, read_stride, 1, &device, dtype),
        batch,
        nchunks,
        nheads,
        mimo_rank,
    );

    let decay_bnhTMLM = (diff_da_cumsum_bnhTMLM + neg_inf_bnhTMLM).exp();

    let v_bnhLMp = v_bnLMhp.swap_dims(2, 3);
    let orange_bnhTMp = (cb_bnhTMLM * decay_bnhTMLM).matmul(v_bnhLMp);

    // ── Combine and reshape ────────────────────────────────────────────────────
    let y_bntmhp = (blue_bnhTMp + orange_bnhTMp)
        .swap_dims(2, 3) // y_bnTMhp
        .reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, per_head_dim]); // y_bntmhp
    y_bntmhp
}

/// Bring a `d_da` on the fused read axis to the write axis.
///
/// The function sums the `mimo_rank` copies of K5, and then scatters the
/// result. The `da` of a read position is one write position.
fn scatter_fused_read_d_da<B: Backend>(
    d_da_bnhTM: F<B, 4>,
    mimo_rank: usize,
    read_stride: usize,
) -> F<B, 4> {
    let [batch, n, nheads, read] = d_da_bnhTM.dims();
    scatter_read_positions::<B, 4, 5>(
        d_da_bnhTM
            .reshape([batch, n, nheads, read / mimo_rank, mimo_rank]) // d_da_bnhtm
            .sum_dim(4) // d_da_bnht1
            .squeeze_dim::<4>(4), // d_da_bnht
        3,
        read_stride,
    )
}

/// Backward of the BLUE term of K5, for one chunk group.
///
/// Forward: `blue[TM, p] = exp(cumA[TM]) · Σᵣ C[TM, r] · state[p, r]`.
///
/// # Returns
/// - `d_chunk_input_state_bnhpr`: the gradient of the state that goes into each
///   chunk
/// - `d_c_bnhTMr`: the BLUE part of `d_c`
/// - `d_da_bnhl`: the BLUE part of `d_da_cumsum`, on the write axis
fn k5_blue_backward<B: Backend>(
    d_y_bnhTMp: F<B, 5>,
    da_cumsum_bnhTM: F<B, 4>,
    c_bnhTMr: F<B, 5>,
    chunk_input_state_bnhpr: F<B, 5>,
    mimo_rank: usize,
    read_stride: usize,
) -> (F<B, 5>, F<B, 5>, F<B, 4>) {
    let [batch, n, nheads, read, per_head_dim] = d_y_bnhTMp.dims();

    // exp_da depends only on the read position TM. It broadcasts over
    // per_head_dim.
    let exp_da_cumsum_bnhTMp = da_cumsum_bnhTM
        .exp()
        .unsqueeze_dim::<5>(4) // exp_da_cumsum_bnhTM1
        .expand([batch, n, nheads, read, per_head_dim]); // exp_da_cumsum_bnhTMp
    let d_ch_bnhTMp = d_y_bnhTMp.clone() * exp_da_cumsum_bnhTMp.clone();
    san(&d_ch_bnhTMp);

    // d_chunk_input_state[p, r] = Σ_TM C[TM, r] · d_ch[TM, p]
    let d_chunk_input_state_bnhpr = c_bnhTMr
        .clone()
        .transpose() // c_bnhrTM
        .matmul(d_ch_bnhTMp.clone()) // d_chunk_input_state_bnhrp
        .transpose(); // d_chunk_input_state_bnhpr
    san(&d_chunk_input_state_bnhpr);

    // d_C[TM, r] = Σₚ d_ch[TM, p] · state[p, r]
    let d_c_bnhTMr = d_ch_bnhTMp.matmul(chunk_input_state_bnhpr.clone());
    san(&d_c_bnhTMr);

    // ch[TM, p] = Σᵣ C[TM, r] · state[p, r]
    // d_da[TM] = (Σₚ d_y[TM, p] · ch[TM, p]) · exp_da[TM]
    let ch_bnhTMp = c_bnhTMr.matmul(
        chunk_input_state_bnhpr.transpose(), // chunk_input_state_bnhrp
    );
    let d_da_bnhTM = (d_y_bnhTMp * ch_bnhTMp * exp_da_cumsum_bnhTMp)
        .sum_dim(4) // d_da_bnhTM1
        .squeeze_dim::<4>(4); // d_da_bnhTM
    san(&d_da_bnhTM);

    let d_da_bnhl = scatter_fused_read_d_da(d_da_bnhTM, mimo_rank, read_stride);

    (d_chunk_input_state_bnhpr, d_c_bnhTMr, d_da_bnhl)
}

/// The gradients of the K5 terms, over all the chunks.
pub(crate) struct K5Grads<B: Backend> {
    /// The BLUE gradient of the state that goes into each chunk.
    pub d_chunk_input_state_bnhpr: F<B, 5>,
    /// The BLUE part of `d_c`.
    pub d_c_blue_bnhTMr: F<B, 5>,
    /// The BLUE part of `d_da_cumsum`.
    pub d_da_blue_bhnl: F<B, 4>,
    /// The intra-chunk part of `d_v`.
    pub d_v_intra_bnhLMp: F<B, 5>,
    /// The gradient of the K2 matrix.
    pub d_cb_bnhTMLM: F<B, 5>,
    /// The intra-chunk part of `d_da_cumsum`.
    pub d_da_intra_bhnl: F<B, 4>,
}

/// Backward of K5: BLUE and the intra-chunk term, one chunk group at a time.
///
/// Neither term is recurrent. Each needs only the slices of its chunk and
/// the input state of that chunk, which the recomputed K4 gave batched. The
/// group size sets the memory of the score `[b, group, h, TM, LM]` against the
/// peak of the forward (see `backward_chunk_group`). On single-SSD, the state
/// of this function is the accumulator of that pathway.
///
/// The intra-chunk term serves both K5 forwards:
///
/// ```text
/// w[TM, LM]    = cb[TM, LM] · decay[TM, LM] · scale[LM]
/// intra[TM, p] = Σ_LM w[TM, LM] · v[LM, p]
/// ```
///
/// - ORANGE: `scale_bnlh` is `None` and `mask_diagonal` is `1`. The sum includes
///   the same position.
/// - LOWER: `scale_bnlh` is `Some` and `mask_diagonal` is `0`. The sum excludes
///   the same position.
///
/// Without a scale, the statements below form no tensor for it. `prod` is `w`,
/// and `d_prod` is `d_w`.
///
/// # Arguments
/// - `d_y_bnhTMp`: the gradient of the terms, on the read axis
/// - `v_bnlmhp`, `c_bntmhr`, `cb_bnhTMLM`, `chunk_input_state_bnhpr`: the K5 inputs
/// - `da_cumsum_bhnLM`, `da_cumsum_bhnTM`: see [`k5_da_cumsum_on_chunk_axes`]
/// - `scale_bnlh`: the scale per source position, or `None`
/// - `mask_diagonal`: the diagonal of [`read_causal_mask`]
/// - `read_stride`: write positions per read position
///
/// # Returns
/// The gradients of the two terms, and `d_scale_bhnl`: the intra-chunk part of
/// `d_scale`, or `None` without a scale.
#[allow(clippy::too_many_arguments)]
pub(crate) fn k5_ssd_chunk_scan_backward<B: Backend>(
    d_y_bnhTMp: F<B, 5>,
    v_bnlmhp: F<B, 6>,
    c_bntmhr: F<B, 6>,
    cb_bnhTMLM: F<B, 5>,
    da_cumsum_bhnLM: F<B, 4>,
    da_cumsum_bhnTM: F<B, 4>,
    chunk_input_state_bnhpr: F<B, 5>,
    scale_bnlh: Option<F<B, 4>>,
    mask_diagonal: i64,
    read_stride: usize,
) -> (K5Grads<B>, Option<F<B, 4>>) {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = c_bntmhr.dims();
    let [.., read, _] = d_y_bnhTMp.dims();
    let fused = chunk_len * mimo_rank;

    let neg_inf_base_tl: F<B, 2> =
        read_causal_mask::<B>(chunk_len, read_stride, mask_diagonal, &v_bnlmhp.device(), v_bnlmhp.dtype());

    let group = backward_chunk_group(nchunks);
    let chunk_groups = nchunks.div_ceil(group);
    let mut vec_intra_d_v_bnhLMp: Vec<F<B, 5>> = Vec::with_capacity(chunk_groups);
    let mut vec_blue_d_c_bnhTMr: Vec<F<B, 5>> = Vec::with_capacity(chunk_groups);
    let mut vec_d_cb_bnhTMLM: Vec<F<B, 5>> = Vec::with_capacity(chunk_groups);
    let mut vec_blue_d_da_bnhl: Vec<F<B, 4>> = Vec::with_capacity(chunk_groups);
    let mut vec_intra_d_da_bnhl: Vec<F<B, 4>> = Vec::with_capacity(chunk_groups);
    let mut vec_d_scale_bnhl: Vec<F<B, 4>> = Vec::with_capacity(chunk_groups);
    let mut vec_d_chunk_input_state_bnhpr: Vec<F<B, 5>> = Vec::with_capacity(chunk_groups);

    for start in (0..nchunks).step_by(group) {
        // The chunk count of this group: `group`, or fewer at the tail. It is
        // the same `n` axis, only narrowed, so the names do not change.
        let n = group.min(nchunks - start);

        // ── Group slices (fused chunk_len · mimo_rank) ─────────────────
        let v_bnhLMp: F<B, 5> = v_bnlmhp
            .clone()
            .narrow(1, start, n) // v_bnlmhp
            .reshape([batch, n, fused, nheads, per_head_dim]) // v_bnLMhp
            .swap_dims(2, 3); // v_bnhLMp

        let c_bnhTMr: F<B, 5> = c_bntmhr
            .clone()
            .narrow(1, start, n) // c_bntmhr
            .reshape([batch, n, read, nheads, state_rank]) // c_bnTMhr
            .swap_dims(2, 3); // c_bnhTMr

        let cb_bnhTMLM: F<B, 5> = cb_bnhTMLM.clone().narrow(1, start, n);

        let da_cumsum_bnhLM: F<B, 4> = da_cumsum_bhnLM
            .clone()
            .narrow(2, start, n) // da_cumsum_bhnLM
            .swap_dims(1, 2); // da_cumsum_bnhLM
        let da_cumsum_bnhTM: F<B, 4> = da_cumsum_bhnTM
            .clone()
            .narrow(2, start, n) // da_cumsum_bhnTM
            .swap_dims(1, 2); // da_cumsum_bnhTM

        // The scale of a position, broadcast over the source `m`.
        let scale_bnhLM: Option<F<B, 4>> = scale_bnlh.as_ref().map(|scale_bnlh| {
            scale_bnlh
                .clone()
                .narrow(1, start, n) // scale_bnlh
                .swap_dims(2, 3) // scale_bnhl
                .unsqueeze_dim::<5>(4) // scale_bnhl1
                .expand([batch, n, nheads, chunk_len, mimo_rank]) // scale_bnhlm
                .reshape([batch, n, nheads, fused]) // scale_bnhLM
        });

        let chunk_input_state_bnhpr: F<B, 5> =
            chunk_input_state_bnhpr.clone().narrow(1, start, n);
        san(&chunk_input_state_bnhpr);

        let d_y_bnhTMp: F<B, 5> = d_y_bnhTMp.clone().narrow(1, start, n);

        // ── BLUE backward ──────────────────────────────────────────────
        let (d_chunk_input_state_bnhpr, d_c_blue_bnhTMr, d_da_blue_bnhl) = k5_blue_backward(
            d_y_bnhTMp.clone(),
            da_cumsum_bnhTM.clone(),
            c_bnhTMr,
            chunk_input_state_bnhpr,
            mimo_rank,
            read_stride,
        );
        vec_d_chunk_input_state_bnhpr.push(d_chunk_input_state_bnhpr);
        vec_blue_d_c_bnhTMr.push(d_c_blue_bnhTMr);
        vec_blue_d_da_bnhl.push(d_da_blue_bnhl);

        // ── Intra-chunk backward ───────────────────────────────────────
        let da_target_bnhTMLM: F<B, 5> = da_cumsum_bnhTM
            .unsqueeze_dim::<5>(4) // da_cumsum_bnhTMₜ1
            .expand([batch, n, nheads, read, fused]); // da_target_bnhTMₜLM
        let da_source_bnhTMLM: F<B, 5> = da_cumsum_bnhLM
            .unsqueeze_dim::<5>(3) // da_cumsum_bnh1LMₛ
            .expand([batch, n, nheads, read, fused]); // da_source_bnhTMLMₛ
        let diff_bnhTMLM = da_target_bnhTMLM - da_source_bnhTMLM;
        san(&diff_bnhTMLM);

        let neg_inf_bnhTMLM = expand_causal_mask(neg_inf_base_tl.clone(), batch, n, nheads, mimo_rank);
        let decay_bnhTMLM = (diff_bnhTMLM + neg_inf_bnhTMLM).exp();
        san(&decay_bnhTMLM);

        let scale_col_bnhTMLM: Option<F<B, 5>> = scale_bnhLM.map(|scale_bnhLM| {
            scale_bnhLM
                .unsqueeze_dim::<5>(3) // scale_bnh1LMₛ
                .expand([batch, n, nheads, read, fused]) // scale_col_bnhTMLMₛ
        });

        let prod_bnhTMLM = cb_bnhTMLM.clone() * decay_bnhTMLM.clone();
        let (w_bnhTMLM, w_factors) = match scale_col_bnhTMLM {
            Some(scale_col_bnhTMLM) => (
                prod_bnhTMLM.clone() * scale_col_bnhTMLM.clone(),
                Some((prod_bnhTMLM, scale_col_bnhTMLM)),
            ),
            None => (prod_bnhTMLM, None),
        };

        // d_w = d_y @ vᵀ, and d_v = wᵀ @ d_y.
        let d_w_bnhTMLM: F<B, 5> = d_y_bnhTMp.clone().matmul(
            v_bnhLMp.transpose(), // v_bnhpLM
        ); // d_w_bnhTMₜLMₛ
        san(&d_w_bnhTMLM);
        let d_v_intra_bnhLMp: F<B, 5> = w_bnhTMLM
            .transpose() // w_bnhLMₛTMₜ
            .matmul(d_y_bnhTMp); // d_v_intra_bnhLMₛp
        san(&d_v_intra_bnhLMp);
        vec_intra_d_v_bnhLMp.push(d_v_intra_bnhLMp);

        let (d_prod_bnhTMLM, d_scale_at_bnhTMLM) = match w_factors {
            Some((prod_bnhTMLM, scale_col_bnhTMLM)) => (
                d_w_bnhTMLM.clone() * scale_col_bnhTMLM,
                Some(d_w_bnhTMLM * prod_bnhTMLM),
            ),
            None => (d_w_bnhTMLM, None),
        };

        // d_cb = d_prod · decay, d_decay = d_prod · cb and d_diff = d_decay · decay.
        // A masked position has decay = 0, so it adds 0 to d_diff.
        let d_cb_bnhTMLM = d_prod_bnhTMLM.clone() * decay_bnhTMLM.clone();
        vec_d_cb_bnhTMLM.push(d_cb_bnhTMLM);

        let d_decay_bnhTMLM = d_prod_bnhTMLM * cb_bnhTMLM;
        let d_diff_bnhTMLM = d_decay_bnhTMLM * decay_bnhTMLM;

        // `diff` is target − source. The target term sums `d_diff` over the
        // sources LMₛ, and it is scattered onto the write axis. The source term
        // sums `d_diff` over the targets TMₜ, and it is on that axis.
        let d_da_target_bnhTM: F<B, 4> = d_diff_bnhTMLM
            .clone()
            .sum_dim(4) // d_diff_bnhTMₜ1
            .squeeze_dim::<4>(4); // d_da_target_bnhTMₜ
        let d_da_source_bnhLM: F<B, 4> = d_diff_bnhTMLM
            .sum_dim(3) // d_diff_bnh1LMₛ
            .squeeze_dim::<4>(3); // d_da_source_bnhLMₛ

        let d_da_target_bnhl = scatter_fused_read_d_da(d_da_target_bnhTM, mimo_rank, read_stride);
        let d_da_source_bnhl: F<B, 4> = d_da_source_bnhLM
            .reshape([batch, n, nheads, chunk_len, mimo_rank]) // d_da_source_bnhlm
            .sum_dim(4) // d_da_source_bnhl1
            .squeeze_dim::<4>(4); // d_da_source_bnhl
        let d_da_intra_bnhl = d_da_target_bnhl - d_da_source_bnhl;
        san(&d_da_intra_bnhl);
        vec_intra_d_da_bnhl.push(d_da_intra_bnhl);

        // d_scale[l] = Σ_{TMₜ, m} d_scale_at[TMₜ, LMₛ]
        if let Some(d_scale_at_bnhTMLM) = d_scale_at_bnhTMLM {
            let d_scale_bnhl: F<B, 4> = d_scale_at_bnhTMLM
                .sum_dim(3) // d_scale_at_bnh1LMₛ
                .squeeze_dim::<4>(3) // d_scale_at_bnhLMₛ
                .reshape([batch, n, nheads, chunk_len, mimo_rank]) // d_scale_at_bnhlm
                .sum_dim(4) // d_scale_at_bnhl1
                .squeeze_dim::<4>(4); // d_scale_bnhl
            vec_d_scale_bnhl.push(d_scale_bnhl);
        }
    }

    // ── Rejoin the groups along the chunk axis ────────────────────────────
    let d_v_intra_bnhLMp = cat_chunk_groups(vec_intra_d_v_bnhLMp, 1);
    let d_c_blue_bnhTMr = cat_chunk_groups(vec_blue_d_c_bnhTMr, 1);
    let d_cb_bnhTMLM = cat_chunk_groups(vec_d_cb_bnhTMLM, 1);
    let d_da_blue_bhnl = cat_chunk_groups(vec_blue_d_da_bnhl, 1).swap_dims(1, 2);
    let d_da_intra_bhnl = cat_chunk_groups(vec_intra_d_da_bnhl, 1).swap_dims(1, 2);
    let d_scale_bhnl = scale_bnlh.map(|_| cat_chunk_groups(vec_d_scale_bnhl, 1).swap_dims(1, 2));
    let d_chunk_input_state_bnhpr = cat_chunk_groups(vec_d_chunk_input_state_bnhpr, 1);

    let grads = K5Grads {
        d_chunk_input_state_bnhpr,
        d_c_blue_bnhTMr,
        d_da_blue_bhnl,
        d_v_intra_bnhLMp,
        d_cb_bnhTMLM,
        d_da_intra_bhnl,
    };
    (grads, d_scale_bhnl)
}
