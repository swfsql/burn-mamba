#![allow(non_snake_case)]

use crate::mamba3::prelude::*;
use burn::prelude::*;

impl<B: Backend> Mamba3SsdInput<B> {
    /// MIMO-first (Hybrid) Serial SSD.
    ///
    /// Implements K1-K5 with a sequential loop (K4) for the inter-chunk scan instead
    /// of the quadratic segsum approach in [`Self::ssd_minimal`].
    /// This is more memory-efficient for long sequences with many chunks.
    ///
    /// SISO (mimo_rank=1) is the special case where the fused length equals the chunk length.
    ///
    /// # Returns
    /// - `y_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn ssd_serial(self) -> (Tensor<B, 6>, Tensor<B, 4>) {
        let input = self;
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = input.v_bnlmhp.dims();
        let [.., state_rank] = input.b_bnlmhr.dims();

        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr is not yet supported in ssd_serial; use ssd_minimal instead"
        );
        assert!(nchunks > 0, "sequence length must be at least 1");

        // ── K1: da_cumsum from da_bnlh ────────────────────────────────────────
        let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum(input.da_bnlh.clone());
        assert_eq!([batch, nheads, nchunks, chunk_len], da_cumsum_bhnl.dims());
        assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());

        // ── K2: CB matrix on fused tensors ────────────────────────────────────
        let cb_bnhLMLM: Tensor<B, 5> = k2_ssd_bmm(input.c_bnlmhr.clone(), input.b_bnlmhr.clone());
        assert_eq!(
            [
                batch,
                nchunks,
                nheads,
                chunk_len * mimo_rank,
                chunk_len * mimo_rank
            ],
            cb_bnhLMLM.dims()
        );

        // ── K3: intra-chunk state ─────────────────────────────────────────────
        let intra_chunk_state_bnhpr: Tensor<B, 5> = k3_ssd_chunk_state(
            input.v_bnlmhp.clone(),
            input.b_bnlmhr.clone(),
            da_cumsum_bhnl.clone(),
        );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            intra_chunk_state_bnhpr.dims()
        );

        // ── K4: state passing (sequential loop) ───────────────────────────────
        let (chunk_input_state_bnhpr, final_state_bhpr): (Tensor<B, 5>, Tensor<B, 4>) =
            k4_ssd_state_passing(
                intra_chunk_state_bnhpr,
                da_chunk_end_bhn,
                input.initial_state_bhpr,
            );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            chunk_input_state_bnhpr.dims()
        );
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            final_state_bhpr.dims()
        );

        // ── K5: MIMO chunk scan ───────────────────────────────────────────────
        let y_bnlmhp: Tensor<B, 6> = k5_ssd_chunk_scan(
            da_cumsum_bhnl,
            input.v_bnlmhp,
            input.c_bnlmhr,
            cb_bnhLMLM,
            chunk_input_state_bnhpr,
        );

        (y_bnlmhp, final_state_bhpr)
    }
}

// ---------------------------------------------------------------------------
// K1 — chunk cumulative log-decay
// ---------------------------------------------------------------------------

/// Compute the intra-chunk cumulative log-decay and per-chunk decay totals.
///
/// # Arguments
/// - `da_bnlh`: pre-combined `Δ·A`, shape `[batch, nchunks, chunk_len, nheads]`
///
/// # Returns
/// - `da_cumsum_bhnl`: `[batch, nheads, nchunks, chunk_len]` — intra-chunk prefix sums
/// - `da_chunk_end_bhn`: `[batch, nheads, nchunks]` — last prefix sum per chunk (total decay)
pub fn k1_ssd_chunk_cumsum<B: Backend>(da_bnlh: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let [batch, nchunks, chunk_len, nheads] = da_bnlh.dims();
    // Permute to [batch, nheads, nchunks, chunk_len] for the cumsum along the last dim
    let da_bhnl = da_bnlh.permute([0, 3, 1, 2]);
    let da_cumsum_bhnl = da_bhnl.cumsum(3);
    assert_eq!([batch, nheads, nchunks, chunk_len], da_cumsum_bhnl.dims());

    let da_chunk_end_bhn: Tensor<B, 3> = da_cumsum_bhnl
        .clone()
        .slice(s![.., .., .., -1]) // da_cumsum_end_bhn1
        .squeeze_dim(3); // da_cumsum_end_bhn
    assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());

    (da_cumsum_bhnl, da_chunk_end_bhn)
}

// ---------------------------------------------------------------------------
// K2 — CB block matrix (C @ B^T on fused MIMO tensors)
// ---------------------------------------------------------------------------

/// Compute the intra-chunk CB matrix on fused (mimo_rank-into-chunk_len) tensors.
///
/// # Arguments
/// - `c_bnlmhr`: `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
/// - `b_bnlmhr`: `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
///
/// # Returns
/// - `cb_bnhLMLM`: `[batch, nchunks, nheads, chunk_len*mimo_rank, chunk_len*mimo_rank]`
pub fn k2_ssd_bmm<B: Backend>(c_bnlmhr: Tensor<B, 6>, b_bnlmhr: Tensor<B, 6>) -> Tensor<B, 5> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank] = c_bnlmhr.dims();

    // Fuse R into chunk_len
    let c_bnLMhr = c_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

    let c_bnhLMr = c_bnLMhr.permute([0, 1, 3, 2, 4]);
    let b_bnhrLM = b_bnLMhr.permute([0, 1, 3, 4, 2]);
    let cb_bnhLMLM: Tensor<B, 5> = c_bnhLMr.matmul(b_bnhrLM);
    assert_eq!(
        [
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank
        ],
        cb_bnhLMLM.dims()
    );
    cb_bnhLMLM
}

// ---------------------------------------------------------------------------
// K3 — intra-chunk state (chunk-end state assuming zero initial state)
// ---------------------------------------------------------------------------

/// Compute the SSM state at the end of each chunk, assuming zero initial hidden state.
///
/// Uses the pre-scaled V tensor — no `dt·B` scaling is performed here.
///
/// # Arguments
/// - `v_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]` — pre-scaled V
/// - `b_bnlmhr`: `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
/// - `da_cumsum_bhnl`: `[batch, nheads, nchunks, chunk_len]`
///
/// # Returns
/// - `intra_chunk_state_bnhpr`: `[batch, nchunks, nheads, per_head_dim, state_rank]`
pub fn k3_ssd_chunk_state<B: Backend>(
    v_bnlmhp: Tensor<B, 6>,
    b_bnlmhr: Tensor<B, 6>,
    da_cumsum_bhnl: Tensor<B, 4>,
) -> Tensor<B, 5> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();

    // Fuse mimo_rank into chunk_len
    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

    // Decay from each fused position to end of chunk
    let a_cumsum_last_bhn1 = da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
    // Expand base cumsum to fused
    let a_cumsum_bhnLM = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4) // da_cumsum_bhnl1
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // da_cumsum_bhnlm
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // da_cumsum_bhnLM
    let decay_bhnLM = (a_cumsum_last_bhn1 - a_cumsum_bhnLM).exp();

    // decay * V
    let decay_bnLMh1 = decay_bhnLM
        .permute([0, 2, 3, 1]) // decay_bnLMh
        .unsqueeze_dim(4); // decay_bnLMh1
    let decayed_v_bnLMhp = decay_bnLMh1 * v_bnLMhp;

    // state = decayed_V^T @ B
    let decayed_v_bnhpLM = decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);
    let b_bnhLMr = b_bnLMhr.permute([0, 1, 3, 2, 4]);
    let intra_chunk_state_bnhpr: Tensor<B, 5> = decayed_v_bnhpLM.matmul(b_bnhLMr);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        intra_chunk_state_bnhpr.dims()
    );
    intra_chunk_state_bnhpr
}

// ---------------------------------------------------------------------------
// K4 — inter-chunk state scan (sequential loop)
// ---------------------------------------------------------------------------

/// Propagate hidden state across chunk boundaries using a sequential scan.
///
/// This kernel is independent of MIMO rank — it operates on the `[nheads, per_head_dim, state_rank]` state
/// which is already aggregated over ranks.
///
/// # Arguments
/// - `intra_chunk_state_bnhpr`: `[batch, nchunks, nheads, per_head_dim, state_rank]`
/// - `da_chunk_end_bhn`: `[batch, nheads, nchunks]` — total log-decay per chunk
/// - `initial_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
///
/// # Returns
/// - `chunk_input_state_bnhpr`: `[batch, nchunks, nheads, per_head_dim, state_rank]`
/// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
pub fn k4_ssd_state_passing<B: Backend>(
    intra_chunk_state_bnhpr: Tensor<B, 5>,
    da_chunk_end_bhn: Tensor<B, 3>,
    initial_state_bhpr: Tensor<B, 4>,
) -> (Tensor<B, 5>, Tensor<B, 4>) {
    let [batch, nchunks, nheads, per_head_dim, state_rank] = intra_chunk_state_bnhpr.dims();

    let mut running_state_bhpr = initial_state_bhpr;
    assert_eq!(
        [batch, nheads, per_head_dim, state_rank],
        running_state_bhpr.dims()
    );

    let mut chunk_input_state_vec_bhpr = Vec::with_capacity(nchunks + 1);
    chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());

    for i_chunk in 0..nchunks {
        let intra_state_bhpr: Tensor<B, 4> = intra_chunk_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // intra_chunk_state_b1hpr
            .squeeze_dim(1); // intra_state_bhpr

        let decay_bhpr = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk]) // da_chunk_end_bh1
            .unsqueeze_dim::<4>(3) // da_chunk_end_bh
            .exp()
            .expand([batch, nheads, per_head_dim, state_rank]); // decay_bhpr

        // SSM recurrence: h[n] = decay * h[n-1] + s[n]
        running_state_bhpr = decay_bhpr * running_state_bhpr + intra_state_bhpr;
        chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());
    }

    let final_state_bhpr = chunk_input_state_vec_bhpr.pop().unwrap();
    assert_eq!(
        [batch, nheads, per_head_dim, state_rank],
        final_state_bhpr.dims()
    );

    let chunk_input_state_bnhpr = Tensor::stack(chunk_input_state_vec_bhpr, 1);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        chunk_input_state_bnhpr.dims()
    );

    (chunk_input_state_bnhpr, final_state_bhpr)
}

// ---------------------------------------------------------------------------
// K5 — MIMO chunk scan (Y_diag + Y_off)
// ---------------------------------------------------------------------------

/// Compute the chunk output by combining the intra-chunk (diagonal) and
/// inter-chunk (off-diagonal) contributions.
///
/// The MIMO causal mask uses interleaved time-step ordering:
/// `L_mimo[i,j] = exp(cumA[i//m] - cumA[j//m])` if `i//m >= j//m`, else 0.
///
/// No D skip is applied — the caller handles it.
///
/// # Arguments
/// - `da_cumsum_bhnl`: `[batch, nheads, nchunks, chunk_len]` — base (not fused)
/// - `v_bnlmhp`: `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
/// - `c_bnlmhr`: `[batch, nchunks, chunk_len, R, nheads, state_rank]`
/// - `cb_bnhLMLM`: `[batch, nchunks, nheads, L, L]` from K2
/// - `chunk_input_state_bnhpr`: `[batch, nchunks, nheads, per_head_dim, state_rank]`
///
/// # Returns
/// - `y_bnlmhp`: `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
pub fn k5_ssd_chunk_scan<B: Backend>(
    da_cumsum_bhnl: Tensor<B, 4>,
    v_bnlmhp: Tensor<B, 6>,
    c_bnlmhr: Tensor<B, 6>,
    cb_bnhLMLM: Tensor<B, 5>,
    chunk_input_state_bnhpr: Tensor<B, 5>,
) -> Tensor<B, 6> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = c_bnlmhr.dims();
    let device = v_bnlmhp.device();

    // Fuse mimo_rank into chunk_len
    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let c_bnLMhr = c_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

    // Expand base da_cumsum to fused length: [b, nheads, n, l] → [b, nheads, n, L]
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4) // da_cumsum_bhnl1
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // da_cumsum_bhnlm
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // da_cumsum_bhnLM

    // ── BLUE (Y_off): exp(cumA[i]) · C[i] · h[n-1] ─────────────────────
    let exp_da_bnhLMp = da_cumsum_bhnLM
        .clone()
        .exp()
        .permute([0, 2, 1, 3]) // exp_da_bnhLM
        .unsqueeze_dim::<5>(4) // // exp_da_bnhLM1
        .expand([batch, nchunks, nheads, chunk_len * mimo_rank, per_head_dim]); // exp_da_bnhLMp

    let c_bnhLMr = c_bnLMhr.permute([0, 1, 3, 2, 4]);
    let chunk_input_state_bnhrp = chunk_input_state_bnhpr.permute([0, 1, 2, 4, 3]);
    let ch_bnhLMp = c_bnhLMr.matmul(chunk_input_state_bnhrp);
    let blue_bnhLMp = ch_bnhLMp * exp_da_bnhLMp;

    // ── ORANGE (Y_diag): MIMO causal decay matrix · CB @ V ────────────────────
    //
    // MIMO pairwise decay: diff[i,j] = cumA[i] - cumA[j]
    //                                = cumA_base[i//m] - cumA_base[j//m]
    let da_cumsum_bnhLM = da_cumsum_bhnLM.permute([0, 2, 1, 3]);
    let target_da_cumsum_bnhLMLM = da_cumsum_bnhLM
        .clone()
        .unsqueeze_dim::<5>(4) // da_cumsum_bnhLM1
        .expand([
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank,
        ]);
    let source_da_cumsum_bnhLMLM = da_cumsum_bnhLM
        .unsqueeze_dim::<5>(3) // da_cumsum_bnh1LM
        .expand([
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank,
        ]);
    let diff_da_cumsum_bnhLMLM = target_da_cumsum_bnhLMLM - source_da_cumsum_bnhLMLM;

    // MIMO causal neg-inf mask: −∞ where j//m > i//m (source strictly ahead of target in time).
    // Build as interleaved expansion of the standard 2-dimensional upper-triangle mask.
    let neg_inf_base_bnhll: Tensor<B, 5> =
        Tensor::<B, 2>::full([chunk_len, chunk_len], f32::NEG_INFINITY, &device)
            .triu(1) // [chunk_len, chunk_len]: -inf above diagonal
            .unsqueeze_dims::<5>(&[0, 1, 2]) // neg_inf_base_111ll
            .expand([batch, nchunks, nheads, chunk_len, chunk_len]); // neg_inf_base_bnhll
    // Interleave-expand
    let neg_inf_bnhLMLM: Tensor<B, 5> = neg_inf_base_bnhll
        .unsqueeze_dim::<6>(4) // neg_inf_base_bnhl1l
        .expand([batch, nchunks, nheads, chunk_len, mimo_rank, chunk_len]) // neg_inf_base_bnhlml
        .reshape([batch, nchunks, nheads, chunk_len * mimo_rank, chunk_len]) // neg_inf_base_bnhLMl
        .unsqueeze_dim::<6>(5) // neg_inf_base_bnhLMl1
        .expand([
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len,
            mimo_rank,
        ]) // neg_inf_base_bnhLMlm
        .reshape([
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank,
        ]); // neg_inf_bnhLMLM

    let decay_bnhLMLM = (diff_da_cumsum_bnhLMLM + neg_inf_bnhLMLM).exp();

    let v_bnhLMp = v_bnLMhp.permute([0, 1, 3, 2, 4]);
    let orange_bnhLMp = (cb_bnhLMLM * decay_bnhLMLM).matmul(v_bnhLMp);

    // ── Combine and reshape ────────────────────────────────────────────────────
    let y_bnlmhp = (blue_bnhLMp + orange_bnhLMp)
        .permute([0, 1, 3, 2, 4]) // y_bnLMhp
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]); // y_bnlmhp

    y_bnlmhp
}
