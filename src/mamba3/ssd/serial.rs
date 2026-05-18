use crate::mamba3::prelude::*;
use burn::prelude::*;

impl<B: Backend> Mamba3<B> {
    /// MIMO-first (Hybrid) Serial SSD.
    ///
    /// Implements K1-K5 with a sequential loop (K4) for the inter-chunk scan instead
    /// of the quadratic segsum approach in [`ssd_minimal`](Self::ssd_minimal).
    /// This is more memory-efficient for long sequences with many chunks.
    ///
    /// SISO (R=1) is the special case where the fused length equals the chunk length.
    ///
    /// # Returns
    /// - `y_bnlrhp`: `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    #[allow(non_snake_case)]
    pub fn ssd_serial(input: super::SsdInput<B>) -> (Tensor<B, 6>, Tensor<B, 4>) {
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = input.v_bnlrhp.dims();
        let [.., state_rank] = input.b_bnlrhn.dims();

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
        let cb_bnhLL: Tensor<B, 5> = k2_ssd_bmm(input.c_bnlrhn.clone(), input.b_bnlrhn.clone());
        // [b, n, H, L, L] where L = chunk_len * mimo_rank

        // ── K3: intra-chunk state ─────────────────────────────────────────────
        let intra_chunk_state_bnhpr: Tensor<B, 5> = k3_ssd_chunk_state(
            input.v_bnlrhp.clone(),
            input.b_bnlrhn.clone(),
            da_cumsum_bhnl.clone(),
        );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            intra_chunk_state_bnhpr.dims()
        );

        // ── K4: state passing (sequential loop) ───────────────────────────────
        let (chunk_input_state_bnhpr, final_state_bhpr): (Tensor<B, 5>, Tensor<B, 4>) =
            k4_ssd_state_passing(intra_chunk_state_bnhpr, da_chunk_end_bhn, input.initial_state_bhpr);
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            chunk_input_state_bnhpr.dims()
        );
        assert_eq!([batch, nheads, per_head_dim, state_rank], final_state_bhpr.dims());

        // ── K5: MIMO chunk scan ───────────────────────────────────────────────
        let y_bnlrhp: Tensor<B, 6> = k5_ssd_chunk_scan(
            da_cumsum_bhnl,
            input.v_bnlrhp,
            input.c_bnlrhn,
            cb_bnhLL,
            chunk_input_state_bnhpr,
        );

        (y_bnlrhp, final_state_bhpr)
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
pub fn k1_ssd_chunk_cumsum<B: Backend>(
    da_bnlh: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let [batch, nchunks, chunk_len, nheads] = da_bnlh.dims();
    // Permute to [b, H, n, l] for the cumsum along the last dim
    let da_bhnl = da_bnlh.permute([0, 3, 1, 2]);
    let da_cumsum_bhnl = da_bhnl.cumsum(3);
    assert_eq!([batch, nheads, nchunks, chunk_len], da_cumsum_bhnl.dims());

    let da_chunk_end_bhn: Tensor<B, 3> = da_cumsum_bhnl
        .clone()
        .slice(s![.., .., .., -1]) // [b, H, n, 1]
        .squeeze_dim(3); // [b, H, n]
    assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());

    (da_cumsum_bhnl, da_chunk_end_bhn)
}

// ---------------------------------------------------------------------------
// K2 — CB block matrix (C @ B^T on fused MIMO tensors)
// ---------------------------------------------------------------------------

/// Compute the intra-chunk CB matrix on fused (R-into-L) tensors.
///
/// # Arguments
/// - `c_bnlrhn`: `[batch, nchunks, chunk_len, R, nheads, state_rank]`
/// - `b_bnlrhn`: `[batch, nchunks, chunk_len, R, nheads, state_rank]`
///
/// # Returns
/// - `cb_bnhLL`: `[batch, nchunks, nheads, L, L]` where `L = chunk_len * mimo_rank`
pub fn k2_ssd_bmm<B: Backend>(c_bnlrhn: Tensor<B, 6>, b_bnlrhn: Tensor<B, 6>) -> Tensor<B, 5> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank] = c_bnlrhn.dims();
    let fused_len = chunk_len * mimo_rank;

    // Fuse R into chunk_len: [b, n, l, R, H, N] → [b, n, L, H, N]
    let c_bnLhn = c_bnlrhn.reshape([batch, nchunks, fused_len, nheads, state_rank]);
    let b_bnLhn = b_bnlrhn.reshape([batch, nchunks, fused_len, nheads, state_rank]);

    // [b, n, H, L, N] @ [b, n, H, N, L] → [b, n, H, L, L]
    let c_bnhLr = c_bnLhn.permute([0, 1, 3, 2, 4]); // [b, n, H, L, N]
    let b_bnhrL = b_bnLhn.permute([0, 1, 3, 4, 2]); // [b, n, H, N, L]
    let cb_bnhLL: Tensor<B, 5> = c_bnhLr.matmul(b_bnhrL);
    assert_eq!(
        [batch, nchunks, nheads, fused_len, fused_len],
        cb_bnhLL.dims()
    );
    cb_bnhLL
}

// ---------------------------------------------------------------------------
// K3 — intra-chunk state (chunk-end state assuming zero initial state)
// ---------------------------------------------------------------------------

/// Compute the SSM state at the end of each chunk, assuming zero initial hidden state.
///
/// Uses the pre-scaled V tensor — no `dt·B` scaling is performed here.
///
/// # Arguments
/// - `v_bnlrhp`: `[batch, nchunks, chunk_len, R, nheads, per_head_dim]` — pre-scaled V
/// - `b_bnlrhn`: `[batch, nchunks, chunk_len, R, nheads, state_rank]`
/// - `da_cumsum_bhnl`: `[batch, nheads, nchunks, chunk_len]` — base (not fused)
///
/// # Returns
/// - `intra_chunk_state_bnhpr`: `[batch, nchunks, nheads, per_head_dim, state_rank]`
pub fn k3_ssd_chunk_state<B: Backend>(
    v_bnlrhp: Tensor<B, 6>,
    b_bnlrhn: Tensor<B, 6>,
    da_cumsum_bhnl: Tensor<B, 4>,
) -> Tensor<B, 5> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlrhp.dims();
    let [.., state_rank] = b_bnlrhn.dims();
    let fused_len = chunk_len * mimo_rank;

    // Fuse R into chunk_len
    let v_bnLhp = v_bnlrhp.reshape([batch, nchunks, fused_len, nheads, per_head_dim]);
    let b_bnLhn = b_bnlrhn.reshape([batch, nchunks, fused_len, nheads, state_rank]);

    // Decay from each fused position to end of chunk:
    //   decay_fused[t*R+r] = exp(cumA_last - cumA_base[t])
    let a_cumsum_last_bhn1 = da_cumsum_bhnl.clone().slice(s![.., .., .., -1]); // [b,H,n,1]
    // Expand base cumsum to fused: [b, H, n, l] → [b, H, n, l, R] → [b, H, n, L]
    let a_cumsum_fused_bhnL = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
        .reshape([batch, nheads, nchunks, fused_len]);
    // Broadcast [b,H,n,1] - [b,H,n,L] → [b,H,n,L]
    let decay_bhnL = (a_cumsum_last_bhn1 - a_cumsum_fused_bhnL).exp();

    // decay * V: [b, n, L, H, 1] * [b, n, L, H, P]
    let decay_bnLh1 = decay_bhnL.permute([0, 2, 3, 1]).unsqueeze_dim(4);
    let decayed_v_bnLhp = decay_bnLh1 * v_bnLhp;

    // state = decayed_V^T @ B: [b, n, H, P, L] × [b, n, H, L, N] → [b, n, H, P, N]
    let decayed_v_bnhpL = decayed_v_bnLhp.permute([0, 1, 3, 4, 2]);
    let b_bnhLN = b_bnLhn.permute([0, 1, 3, 2, 4]);
    let intra_chunk_state_bnhpr: Tensor<B, 5> = decayed_v_bnhpL.matmul(b_bnhLN);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        intra_chunk_state_bnhpr.dims()
    );
    intra_chunk_state_bnhpr
}

// ---------------------------------------------------------------------------
// K4 — inter-chunk state scan (sequential loop, unchanged from Mamba-2)
// ---------------------------------------------------------------------------

/// Propagate hidden state across chunk boundaries using a sequential scan.
///
/// This kernel is independent of MIMO rank — it operates on the `[H, P, N]` state
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
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim(1);

        let decay_bhpr = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk])
            .exp()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, per_head_dim, state_rank]);

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
/// `L_mimo[i,j] = exp(cumA[i//R] - cumA[j//R])` if `i//R >= j//R`, else 0.
///
/// No D skip is applied — the caller handles it.
///
/// # Arguments
/// - `da_cumsum_bhnl`: `[batch, nheads, nchunks, chunk_len]` — base (not fused)
/// - `v_bnlrhp`: `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
/// - `c_bnlrhn`: `[batch, nchunks, chunk_len, R, nheads, state_rank]`
/// - `cb_bnhLL`: `[batch, nchunks, nheads, L, L]` from K2
/// - `chunk_input_state_bnhpr`: `[batch, nchunks, nheads, per_head_dim, state_rank]`
///
/// # Returns
/// - `y_bnlrhp`: `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
pub fn k5_ssd_chunk_scan<B: Backend>(
    da_cumsum_bhnl: Tensor<B, 4>,
    v_bnlrhp: Tensor<B, 6>,
    c_bnlrhn: Tensor<B, 6>,
    cb_bnhLL: Tensor<B, 5>,
    chunk_input_state_bnhpr: Tensor<B, 5>,
) -> Tensor<B, 6> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlrhp.dims();
    let [.., state_rank] = c_bnlrhn.dims();
    let fused_len = chunk_len * mimo_rank;
    let device = v_bnlrhp.device();

    // Fuse R into chunk_len
    let v_bnLhp = v_bnlrhp.reshape([batch, nchunks, fused_len, nheads, per_head_dim]);
    let c_bnLhn = c_bnlrhn.reshape([batch, nchunks, fused_len, nheads, state_rank]);

    // Expand base da_cumsum to fused length: [b, H, n, l] → [b, H, n, L]
    let da_cumsum_fused_bhnL = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
        .reshape([batch, nheads, nchunks, fused_len]);

    // ── BLUE (Y_off): exp(cumA_fused[i]) · C[i] · h[n-1] ─────────────────────
    // [b, H, n, L] → [b, n, H, L, 1] → [b, n, H, L, P]
    let exp_da_fused_bnhLp = da_cumsum_fused_bhnL
        .clone()
        .exp()
        .permute([0, 2, 1, 3])
        .unsqueeze_dim::<5>(4)
        .expand([batch, nchunks, nheads, fused_len, per_head_dim]);

    let c_bnhLr = c_bnLhn.permute([0, 1, 3, 2, 4]); // [b, n, H, L, N]
    let state_bnhrp = chunk_input_state_bnhpr.permute([0, 1, 2, 4, 3]); // [b, n, H, N, P]
    let ch_bnhLp = c_bnhLr.matmul(state_bnhrp); // [b, n, H, L, P]
    let blue_bnhLp = ch_bnhLp * exp_da_fused_bnhLp; // [b, n, H, L, P]

    // ── ORANGE (Y_diag): MIMO causal decay matrix · CB @ V ────────────────────
    //
    // MIMO pairwise decay: diff[i,j] = cumA_fused[i] - cumA_fused[j]
    //                                = cumA_base[i//R] - cumA_base[j//R]
    let da_fused_bnhL = da_cumsum_fused_bhnL.permute([0, 2, 1, 3]); // [b, n, H, L]
    let da_target_bnhLL = da_fused_bnhL
        .clone()
        .unsqueeze_dim::<5>(4)
        .expand([batch, nchunks, nheads, fused_len, fused_len]); // [b, n, H, L, L]
    let da_source_bnhLL = da_fused_bnhL
        .unsqueeze_dim::<5>(3)
        .expand([batch, nchunks, nheads, fused_len, fused_len]); // [b, n, H, L, L]
    let diff_bnhLL = da_target_bnhLL - da_source_bnhLL;

    // MIMO causal neg-inf mask: −∞ where j//R > i//R (source strictly ahead of target in time).
    // Build as interleaved expansion of the standard 2D upper-triangle mask.
    let neg_inf_base_bnhll: Tensor<B, 5> = {
        let zero_ll: Tensor<B, 2> = Tensor::zeros([chunk_len, chunk_len], &device);
        Tensor::full_like(&zero_ll, f32::NEG_INFINITY)
            .triu(1) // [l, l]: -inf above diagonal
            .unsqueeze_dims::<5>(&[0, 1, 2])
            .expand([batch, nchunks, nheads, chunk_len, chunk_len])
    };
    // Interleave-expand: [b, n, H, l, l] → [b, n, H, L, L]
    let neg_inf_mimo_bnhLL: Tensor<B, 5> = neg_inf_base_bnhll
        .unsqueeze_dim::<6>(4)
        .expand([batch, nchunks, nheads, chunk_len, mimo_rank, chunk_len])
        .reshape([batch, nchunks, nheads, fused_len, chunk_len])
        .unsqueeze_dim::<6>(5)
        .expand([batch, nchunks, nheads, fused_len, chunk_len, mimo_rank])
        .reshape([batch, nchunks, nheads, fused_len, fused_len]);

    let decay_bnhLL = (diff_bnhLL + neg_inf_mimo_bnhLL).exp(); // [b, n, H, L, L]

    let v_bnhLp = v_bnLhp.permute([0, 1, 3, 2, 4]); // [b, n, H, L, P]
    let orange_bnhLp = (cb_bnhLL * decay_bnhLL).matmul(v_bnhLp); // [b, n, H, L, P]

    // ── Combine and reshape ────────────────────────────────────────────────────
    // [b, n, H, L, P] → [b, n, L, H, P] → [b, n, l, R, H, P]
    let y_bnlrhp = (blue_bnhLp + orange_bnhLp)
        .permute([0, 1, 3, 2, 4])
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

    y_bnlrhp
}
