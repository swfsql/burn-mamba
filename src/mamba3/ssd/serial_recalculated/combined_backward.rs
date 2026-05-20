#![allow(non_snake_case)]

use crate::mamba3::ssd::serial;
use crate::utils::sanity::sanity as san;
use burn::prelude::*;

/// Per-input gradients produced by [`combined_backward`].
#[non_exhaustive]
pub struct CombinedGrads<B: Backend> {
    pub d_v_bnlrhp: Tensor<B, 6>,
    pub d_da_bnlh: Tensor<B, 4>,
    pub d_b_bnlrhn: Tensor<B, 6>,
    pub d_c_bnlrhn: Tensor<B, 6>,
    pub d_initial_state_bhpr: Tensor<B, 4>,
}

/// Same as [`serial::k3_ssd_chunk_state`] but also returns intermediates needed
/// by the custom backward:
/// - `intra_chunk_state_bnhpr` — the chunk-end state assuming zero initial state
/// - `decay_bhnL` — the fused-length K3 decay factor `exp(cumA_last − cumA_fused)`
/// - `decayed_v_bnLhp` — V already scaled by `decay_bnLh1`
pub fn k3_ssd_chunk_state_extended<B: Backend>(
    v_bnlrhp: Tensor<B, 6>,
    b_bnlrhn: Tensor<B, 6>,
    da_cumsum_bhnl: Tensor<B, 4>,
) -> (Tensor<B, 5>, Tensor<B, 4>, Tensor<B, 5>) {
    use burn::tensor::s;

    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlrhp.dims();
    let [.., state_rank] = b_bnlrhn.dims();
    let fused_len = chunk_len * mimo_rank;

    let v_bnLhp = v_bnlrhp.reshape([batch, nchunks, fused_len, nheads, per_head_dim]);
    let b_bnLhn = b_bnlrhn.reshape([batch, nchunks, fused_len, nheads, state_rank]);

    let a_cumsum_last_bhn1 = da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
    let a_cumsum_fused_bhnL = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
        .reshape([batch, nheads, nchunks, fused_len]);
    let decay_bhnL = (a_cumsum_last_bhn1 - a_cumsum_fused_bhnL).exp();
    san(&decay_bhnL);

    let decay_bnLh1 = decay_bhnL.clone().permute([0, 2, 3, 1]).unsqueeze_dim(4);
    let decayed_v_bnLhp = decay_bnLh1 * v_bnLhp;
    san(&decayed_v_bnLhp);

    let decayed_v_bnhpL = decayed_v_bnLhp.clone().permute([0, 1, 3, 4, 2]);
    let b_bnhLN = b_bnLhn.permute([0, 1, 3, 2, 4]);
    let intra_chunk_state_bnhpr = decayed_v_bnhpL.matmul(b_bnhLN);
    san(&intra_chunk_state_bnhpr);

    (intra_chunk_state_bnhpr, decay_bhnL, decayed_v_bnLhp)
}

/// Memory-efficient backward for the Mamba-3 MIMO-first chunkwise SSD.
///
/// Recomputes the forward intermediates (K1-K4) from the saved inputs, then
/// runs a reverse per-chunk loop that fuses the K5 (BLUE + ORANGE) backward
/// with the K4 state-passing backward.  K3/K2/K1 backwards run as single
/// batched ops once the loop has collected all per-chunk slices.
///
/// # Arguments
/// - `d_y_bnlrhp` — upstream gradient of the SSD output
/// - `d_final_bhpr` — upstream gradient of the final SSM state
/// - `v_bnlrhp`, `da_bnlh`, `b_bnlrhn`, `c_bnlrhn`, `initial_state_bhpr` —
///   the five saved forward inputs
///
/// # Returns
/// One [`CombinedGrads`] struct containing gradients for all 5 inputs.
pub fn combined_backward<B: Backend>(
    d_y_bnlrhp: Tensor<B, 6>,
    d_final_bhpr: Tensor<B, 4>,
    //
    v_bnlrhp: Tensor<B, 6>,
    da_bnlh: Tensor<B, 4>,
    b_bnlrhn: Tensor<B, 6>,
    c_bnlrhn: Tensor<B, 6>,
    initial_state_bhpr: Tensor<B, 4>,
) -> CombinedGrads<B> {
    use burn::tensor::s;

    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlrhp.dims();
    let [.., state_rank] = b_bnlrhn.dims();
    let fused_len = chunk_len * mimo_rank;
    let device = v_bnlrhp.device();

    san(&d_y_bnlrhp);
    san(&d_final_bhpr);
    san(&v_bnlrhp);
    san(&da_bnlh);
    san(&b_bnlrhn);
    san(&c_bnlrhn);
    san(&initial_state_bhpr);

    // ═══════════════════════════════════════════════════════════════════════
    // RECOMPUTE FORWARD INTERMEDIATES
    // ═══════════════════════════════════════════════════════════════════════

    // K1 — pre-combined Δ·A → intra-chunk cumsum
    let (da_cumsum_bhnl, da_chunk_end_bhn) = serial::k1_ssd_chunk_cumsum(da_bnlh.clone());
    san(&da_cumsum_bhnl);

    // K2 — CB matrix used in K5 ORANGE
    let cb_bnhLL = serial::k2_ssd_bmm(c_bnlrhn.clone(), b_bnlrhn.clone());
    san(&cb_bnhLL);

    // K3 — intra-chunk state + decay/decayed-V intermediates
    let (intra_chunk_state_bnhpr, k3_decay_bhnL, k3_decayed_v_bnLhp) =
        k3_ssd_chunk_state_extended(v_bnlrhp.clone(), b_bnlrhn.clone(), da_cumsum_bhnl.clone());

    // K4 — chunk-input state stream consumed by K5 BLUE
    let (chunk_input_state_bnhpr, _final_state_bhpr) = serial::k4_ssd_state_passing(
        intra_chunk_state_bnhpr,
        da_chunk_end_bhn.clone(),
        initial_state_bhpr,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // FUSED-L INTERMEDIATES USED BY THE REVERSE LOOP
    // ═══════════════════════════════════════════════════════════════════════
    //
    // da_cumsum_fused_bhnL: cumA per fused position. The expand-then-reshape
    // repeats each base position R times along the fused dim, matching K5.
    let da_cumsum_fused_bhnL = da_cumsum_bhnl
        .clone()
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
        .reshape([batch, nheads, nchunks, fused_len]);

    // d_y in (b, n, H, L_fused, P) ordering — matches the per-chunk slicing.
    let d_y_bnhLp = d_y_bnlrhp
        .reshape([batch, nchunks, fused_len, nheads, per_head_dim])
        .permute([0, 1, 3, 2, 4]);
    san(&d_y_bnhLp);

    // Reusable [l, l] -inf upper-triangular base mask for ORANGE.
    let neg_inf_base_ll: Tensor<B, 2> = {
        let zero_ll: Tensor<B, 2> = Tensor::zeros([chunk_len, chunk_len], &device);
        Tensor::full_like(&zero_ll, f32::NEG_INFINITY).triu(1)
    };

    // ═══════════════════════════════════════════════════════════════════════
    // REVERSE PER-CHUNK LOOP — K5 (BLUE + ORANGE) + K4 fused
    //
    // Per-iteration working tensors are [B,H,L_fused,...] rather than the
    // [B,N,H,L_fused,...] tensors a fully batched K5 backward would allocate.
    // ═══════════════════════════════════════════════════════════════════════
    let mut d_v_orange_vec: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks); // [B,H,L,P]
    let mut d_c_blue_vec: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks); // [B,H,L,N]
    let mut d_cb_vec: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks); // [B,H,L,L]
    let mut d_da_blue_vec: Vec<Tensor<B, 3>> = Vec::with_capacity(nchunks); // [B,H,l]
    let mut d_da_orange_vec: Vec<Tensor<B, 3>> = Vec::with_capacity(nchunks); // [B,H,l]
    let mut d_intra_slices: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks); // [B,H,P,N]
    let mut d_da_end_bh_slices: Vec<Tensor<B, 2>> = Vec::with_capacity(nchunks); // [B,H]

    let mut d_running_state_bhpr: Tensor<B, 4> = d_final_bhpr; // [B,H,P,N]

    for i_chunk in (0..nchunks).rev() {
        // ── Per-chunk slices (fused L = chunk_len · mimo_rank) ─────────────
        let v_bhLp: Tensor<B, 4> = v_bnlrhp
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..])
            .squeeze_dim::<5>(1) // [B, l, R, H, P]
            .reshape([batch, fused_len, nheads, per_head_dim]) // [B, L, H, P]
            .permute([0, 2, 1, 3]); // [B, H, L, P]

        let c_bhLr: Tensor<B, 4> = c_bnlrhn
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..])
            .squeeze_dim::<5>(1)
            .reshape([batch, fused_len, nheads, state_rank])
            .permute([0, 2, 1, 3]); // [B, H, L, N]

        let cb_bhLL: Tensor<B, 4> = cb_bnhLL
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1); // [B,H,L,L]

        let da_cumsum_bhL: Tensor<B, 3> = da_cumsum_fused_bhnL
            .clone()
            .slice(s![.., .., i_chunk, ..])
            .squeeze_dim::<3>(2); // [B,H,L]

        let chunk_input_state_bhpr: Tensor<B, 4> = chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1); // [B,H,P,N]
        san(&chunk_input_state_bhpr);

        let d_y_bhLp: Tensor<B, 4> = d_y_bnhLp
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1); // [B,H,L,P]

        // ── BLUE backward ──────────────────────────────────────────────
        //
        //   blue[L,p] = exp(cumA_fused[L]) · Σ_r C[L,r] · state[p,r]
        //
        // exp_da_fused depends on the fused position L only — broadcast over P.
        let exp_da_cumsum_bhL: Tensor<B, 3> = da_cumsum_bhL.clone().exp();
        let exp_da_cumsum_bhLp: Tensor<B, 4> = exp_da_cumsum_bhL
            .clone()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, fused_len, per_head_dim]);
        let d_ch_bhLp: Tensor<B, 4> = d_y_bhLp.clone() * exp_da_cumsum_bhLp.clone();
        san(&d_ch_bhLp);

        // d_chunk_input_state[p,r] = Σ_L C[L,r] · d_ch[L,p]
        //   C^T (bhrL) @ d_ch (bhLp)  → bhrp  → permute → bhpr
        let d_chunk_input_state_bhpr: Tensor<B, 4> = c_bhLr
            .clone()
            .permute([0, 1, 3, 2]) // [B,H,r,L]
            .matmul(d_ch_bhLp.clone()) // [B,H,r,p]
            .permute([0, 1, 3, 2]); // [B,H,p,r]
        san(&d_chunk_input_state_bhpr);

        // d_C_blue[L,r] = Σ_p d_ch[L,p] · state[p,r]
        //   d_ch (bhLp) @ state (bhpr)  → bhLr
        let d_c_blue_bhLr: Tensor<B, 4> = d_ch_bhLp.matmul(chunk_input_state_bhpr.clone());
        san(&d_c_blue_bhLr);
        d_c_blue_vec.push(d_c_blue_bhLr);

        // d_da from BLUE:
        //   ch[L,p] = Σ_r C[L,r] · state[p,r]      (= C @ state_rp after permute)
        //   d_da[L] = (Σ_p d_y[L,p] · ch[L,p]) · exp_da[L]
        let ch_bhLp: Tensor<B, 4> = c_bhLr
            .clone()
            .matmul(chunk_input_state_bhpr.clone().permute([0, 1, 3, 2])); // [B,H,L,P]
        let d_da_blue_bhL: Tensor<B, 3> = (d_y_bhLp.clone() * ch_bhLp * exp_da_cumsum_bhLp)
            .sum_dim(3)
            .squeeze_dim::<3>(3); // [B,H,L_fused]
        san(&d_da_blue_bhL);

        // Reduce fused L → l (sum the R copies that K5 broadcast).
        let d_da_blue_bhl: Tensor<B, 3> = d_da_blue_bhL
            .reshape([batch, nheads, chunk_len, mimo_rank])
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        d_da_blue_vec.push(d_da_blue_bhl);

        // ── ORANGE backward ────────────────────────────────────────────
        //
        //   w[L_t,L_s] = CB[L_t,L_s] · decay[L_t,L_s]   (MIMO causal mask in decay)
        //   orange[L_t,p] = Σ_{L_s} w[L_t,L_s] · v[L_s,p]
        let da_target_bhLL: Tensor<B, 4> = da_cumsum_bhL
            .clone()
            .unsqueeze_dim::<4>(3) // [B,H,L_t,1]
            .expand([batch, nheads, fused_len, fused_len]);
        let da_source_bhLL: Tensor<B, 4> = da_cumsum_bhL
            .unsqueeze_dim::<4>(2) // [B,H,1,L_s]
            .expand([batch, nheads, fused_len, fused_len]);
        let diff_bhLL = da_target_bhLL - da_source_bhLL;
        san(&diff_bhLL);

        // MIMO causal mask: -inf where L_s//R > L_t//R — interleaved expansion
        // of the [l, l] upper-triangular base mask (matches K5).
        let neg_inf_mimo_bhLL: Tensor<B, 4> = neg_inf_base_ll
            .clone()
            .unsqueeze_dims::<4>(&[0, 1]) // [1,1,l,l]
            .expand([batch, nheads, chunk_len, chunk_len])
            .unsqueeze_dim::<5>(3) // [B,H,l,1,l]
            .expand([batch, nheads, chunk_len, mimo_rank, chunk_len])
            .reshape([batch, nheads, fused_len, chunk_len])
            .unsqueeze_dim::<5>(4) // [B,H,L,l,1]
            .expand([batch, nheads, fused_len, chunk_len, mimo_rank])
            .reshape([batch, nheads, fused_len, fused_len]);
        let decay_bhLL = (diff_bhLL + neg_inf_mimo_bhLL).exp(); // [B,H,L,L]
        san(&decay_bhLL);

        // d_v_orange = w^T @ d_orange ; d_w = d_orange @ v^T
        let d_orange_bhLp = d_y_bhLp;
        let w_bhLL = cb_bhLL.clone() * decay_bhLL.clone();
        let d_w_bhLL: Tensor<B, 4> = d_orange_bhLp
            .clone()
            .matmul(v_bhLp.clone().permute([0, 1, 3, 2])); // [B,H,L_t,L_s]
        san(&d_w_bhLL);
        let d_v_orange_bhLp: Tensor<B, 4> = w_bhLL.permute([0, 1, 3, 2]).matmul(d_orange_bhLp); // [B,H,L_s,P]
        san(&d_v_orange_bhLp);
        d_v_orange_vec.push(d_v_orange_bhLp);

        // d_cb = d_w · decay ; d_decay = d_w · cb ; d_diff = d_decay · decay
        // (masked positions where decay=0 contribute 0 to d_diff automatically)
        let d_cb_bhLL = d_w_bhLL.clone() * decay_bhLL.clone();
        d_cb_vec.push(d_cb_bhLL);

        let d_decay_bhLL = d_w_bhLL * cb_bhLL;
        let d_diff_bhLL = d_decay_bhLL * decay_bhLL;

        // d_da_target[L_t] = Σ_{L_s} d_diff[L_t, L_s] ;
        // d_da_source[L_s] = Σ_{L_t} d_diff[L_t, L_s] ;
        // d_da_orange = d_da_target − d_da_source  (diff = target − source).
        let d_da_target_bhL: Tensor<B, 3> = d_diff_bhLL.clone().sum_dim(3).squeeze_dim::<3>(3);
        let d_da_source_bhL: Tensor<B, 3> = d_diff_bhLL.sum_dim(2).squeeze_dim::<3>(2);
        let d_da_orange_bhL = d_da_target_bhL - d_da_source_bhL;
        san(&d_da_orange_bhL);

        // Reduce fused L → l (sum over the R-broadcast copies).
        let d_da_orange_bhl: Tensor<B, 3> = d_da_orange_bhL
            .reshape([batch, nheads, chunk_len, mimo_rank])
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        d_da_orange_vec.push(d_da_orange_bhl);

        // ── K4 backward step for chunk i_chunk ─────────────────────────
        //
        // Forward (recap):  s_{i+1} = decay_i · s_i + intra_state_i
        //   - d_intra_state_i      = d_s_{i+1}      (current d_running_state)
        //   - d_decay_i            = d_s_{i+1} · s_i
        //   - d_s_i (propagated)   = decay_i · d_s_{i+1} + d_chunk_input_state_blue
        d_intra_slices.push(d_running_state_bhpr.clone());

        let decay_chunk_bhpr: Tensor<B, 4> = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk]) // [B,H,1]
            .exp()
            .unsqueeze_dim::<4>(3) // [B,H,1,1]
            .expand([batch, nheads, per_head_dim, state_rank]); // [B,H,P,N]
        san(&decay_chunk_bhpr);

        let d_decay_chunk_bhpr = d_running_state_bhpr.clone() * chunk_input_state_bhpr;
        // d_da_chunk_end[b,h] = Σ_{p,r} d_decay · decay (since decay = exp(da_chunk_end))
        let d_da_chunk_end_bh: Tensor<B, 2> = (d_decay_chunk_bhpr * decay_chunk_bhpr.clone())
            .reshape([batch, nheads, per_head_dim * state_rank])
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        san(&d_da_chunk_end_bh);
        d_da_end_bh_slices.push(d_da_chunk_end_bh);

        d_running_state_bhpr = decay_chunk_bhpr * d_running_state_bhpr + d_chunk_input_state_bhpr;
        san(&d_running_state_bhpr);
    }
    let d_initial_state_bhpr = d_running_state_bhpr;

    // ── Restore natural (forward) chunk order ─────────────────────────────
    d_v_orange_vec.reverse();
    d_c_blue_vec.reverse();
    d_cb_vec.reverse();
    d_da_blue_vec.reverse();
    d_da_orange_vec.reverse();
    d_intra_slices.reverse();
    d_da_end_bh_slices.reverse();

    // ── Stack per-chunk slices back into batched tensors ──────────────────
    // d_v_orange: [B,H,L,P] → stack@1 → [B,n,H,L,P]
    let d_v_orange_bnhLp: Tensor<B, 5> = Tensor::stack(d_v_orange_vec, 1);
    // d_c_blue:   [B,H,L,N] → stack@1 → [B,n,H,L,N]
    let d_c_blue_bnhLn: Tensor<B, 5> = Tensor::stack(d_c_blue_vec, 1);
    // d_cb:       [B,H,L,L] → stack@1 → [B,n,H,L,L]
    let d_cb_bnhLL: Tensor<B, 5> = Tensor::stack(d_cb_vec, 1);
    // d_da_blue:  [B,H,l]   → stack@2 → [B,H,n,l]
    let d_da_blue_bhnl: Tensor<B, 4> = Tensor::stack(d_da_blue_vec, 2);
    // d_da_orange:[B,H,l]   → stack@2 → [B,H,n,l]
    let d_da_orange_bhnl: Tensor<B, 4> = Tensor::stack(d_da_orange_vec, 2);
    // d_intra:    [B,H,P,N] → stack@1 → [B,n,H,P,N]
    let d_intra_chunk_state_bnhpr: Tensor<B, 5> = Tensor::stack(d_intra_slices, 1);
    // d_da_end:   [B,H]     → stack@2 → [B,H,n]; scatter into last-l of d_da_cumsum_k4
    let d_da_end_bhn: Tensor<B, 3> = Tensor::stack(d_da_end_bh_slices, 2);
    let d_da_cumsum_k4_bhnl: Tensor<B, 4> = {
        let zeros = Tensor::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device);
        let d_da_end_bhn1 = d_da_end_bhn.unsqueeze_dim::<4>(3);
        Tensor::cat(vec![zeros, d_da_end_bhn1], 3)
    };

    // ═══════════════════════════════════════════════════════════════════════
    // K3 BACKWARD (batched)
    //
    // Forward (recap):
    //   v_bnLhp        = v.reshape
    //   b_bnLhn        = b.reshape
    //   decay_bhnL     = exp(cumA_last − cumA_fused)
    //   decay_bnLh1    = decay_bhnL.permute([0,2,3,1]).unsqueeze(4)
    //   decayed_v_bnLhp = decay_bnLh1 · v_bnLhp                      (elementwise)
    //   decayed_v_bnhpL = decayed_v_bnLhp.permute([0,1,3,4,2])
    //   b_bnhLN        = b_bnLhn.permute([0,1,3,2,4])
    //   intra_state    = decayed_v_bnhpL @ b_bnhLN
    // ═══════════════════════════════════════════════════════════════════════
    let v_bnLhp = v_bnlrhp
        .clone()
        .reshape([batch, nchunks, fused_len, nheads, per_head_dim]);
    let b_bnLhn = b_bnlrhn
        .clone()
        .reshape([batch, nchunks, fused_len, nheads, state_rank]);
    let b_bnhLN = b_bnLhn.clone().permute([0, 1, 3, 2, 4]); // [B,n,H,L,N]
    let decayed_v_bnhpL = k3_decayed_v_bnLhp.permute([0, 1, 3, 4, 2]); // [B,n,H,P,L]

    // d_decayed_v_bnhpL = d_intra @ b_bnhLN^T
    let d_decayed_v_bnhpL: Tensor<B, 5> = d_intra_chunk_state_bnhpr
        .clone()
        .matmul(b_bnhLN.permute([0, 1, 2, 4, 3])); // [B,n,H,P,L]
    // d_b_k3_bnhLN = decayed_v_bnhpL^T @ d_intra
    let d_b_k3_bnhLN: Tensor<B, 5> = decayed_v_bnhpL
        .permute([0, 1, 2, 4, 3]) // [B,n,H,L,P]
        .matmul(d_intra_chunk_state_bnhpr); // [B,n,H,L,N]

    // Inverse permute of [0,1,3,4,2] is [0,1,4,2,3]
    let d_decayed_v_bnLhp = d_decayed_v_bnhpL.permute([0, 1, 4, 2, 3]); // [B,n,L,H,P]

    // d_decay_bnLh1 = Σ_p d_decayed_v · v
    let d_decay_bnLh1: Tensor<B, 5> = (d_decayed_v_bnLhp.clone() * v_bnLhp).sum_dim(4); // [B,n,L,H,1]
    // Inverse permute of [0,2,3,1] is [0,3,1,2]; then squeeze the trailing unit dim.
    let d_decay_bhnL = d_decay_bnLh1.squeeze_dim::<4>(4).permute([0, 3, 1, 2]); // [B,H,n,L]

    // d_v_k3_bnLhp = d_decayed_v · decay (broadcast)
    let decay_bnLh1 = k3_decay_bhnL
        .clone()
        .permute([0, 2, 3, 1])
        .unsqueeze_dim::<5>(4);
    let d_v_k3_bnLhp: Tensor<B, 5> = d_decayed_v_bnLhp * decay_bnLh1;
    let d_v_k3_bnlrhp: Tensor<B, 6> =
        d_v_k3_bnLhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

    // d(cumA_last − cumA_fused) = d_decay · decay
    let d_decay_times_decay_bhnL = d_decay_bhnL * k3_decay_bhnL; // [B,H,n,L]
    // d_a_cumsum_last: Σ over L_fused (broadcast dim).
    let d_a_cumsum_last_bhn: Tensor<B, 3> = d_decay_times_decay_bhnL
        .clone()
        .sum_dim(3)
        .squeeze_dim::<3>(3);
    // d_a_cumsum_fused: negated (subtraction).
    let d_a_cumsum_fused_bhnL = -d_decay_times_decay_bhnL;

    // Contribution to d_da_cumsum from the fused-cumA expand (sum R copies).
    let d_da_cumsum_k3_from_fused_bhnl: Tensor<B, 4> = d_a_cumsum_fused_bhnL
        .reshape([batch, nheads, nchunks, chunk_len, mimo_rank])
        .sum_dim(4)
        .squeeze_dim::<4>(4); // [B,H,n,l]
    // Contribution from cumA_last: only the last-l position.
    let d_da_cumsum_k3_from_last_bhnl: Tensor<B, 4> = {
        let zeros = Tensor::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device);
        let d_last = d_a_cumsum_last_bhn.unsqueeze_dim::<4>(3);
        Tensor::cat(vec![zeros, d_last], 3)
    };
    let d_da_cumsum_k3_bhnl = d_da_cumsum_k3_from_fused_bhnl + d_da_cumsum_k3_from_last_bhnl;

    // d_b_k3: undo permute [0,1,3,2,4] (involution swap of pos 2,3), then reshape.
    let d_b_k3_bnLhn = d_b_k3_bnhLN.permute([0, 1, 3, 2, 4]); // [B,n,L,H,N]
    let d_b_k3_bnlrhn: Tensor<B, 6> =
        d_b_k3_bnLhn.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

    // ═══════════════════════════════════════════════════════════════════════
    // K2 BACKWARD (batched)
    //
    //   cb_bnhLL = c_bnhLr @ b_bnhrL   (contracts state_rank)
    //   d_c_bnhLr = d_cb @ b_bnhLr      (= d_cb @ b_bnhrL^T)
    //   d_b_bnhrL = c_bnhrL @ d_cb      (= c_bnhLr^T @ d_cb)
    // ═══════════════════════════════════════════════════════════════════════
    let c_bnLhn = c_bnlrhn
        .clone()
        .reshape([batch, nchunks, fused_len, nheads, state_rank]);
    let c_bnhLr = c_bnLhn.permute([0, 1, 3, 2, 4]); // [B,n,H,L,N]
    let b_bnhLr_for_k2 = b_bnLhn.permute([0, 1, 3, 2, 4]); // [B,n,H,L,N]

    let d_c_k2_bnhLr: Tensor<B, 5> = d_cb_bnhLL.clone().matmul(b_bnhLr_for_k2); // [B,n,H,L,N]
    let d_b_k2_bnhrL: Tensor<B, 5> = c_bnhLr
        .permute([0, 1, 2, 4, 3]) // [B,n,H,N,L]
        .matmul(d_cb_bnhLL); // [B,n,H,N,L]

    // Undo permute [0,1,3,2,4] → involution swap of pos 2,3.
    let d_c_k2_bnLhn = d_c_k2_bnhLr.permute([0, 1, 3, 2, 4]); // [B,n,L,H,N]
    let d_c_k2_bnlrhn: Tensor<B, 6> =
        d_c_k2_bnLhn.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
    // Undo permute [0,1,3,4,2] → inverse perm is [0,1,4,2,3].
    let d_b_k2_bnLhn = d_b_k2_bnhrL.permute([0, 1, 4, 2, 3]); // [B,n,L,H,N]
    let d_b_k2_bnlrhn: Tensor<B, 6> =
        d_b_k2_bnLhn.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

    // ── Unstack d_c_blue / d_v_orange and reshape back ────────────────────
    let d_c_blue_bnLhn = d_c_blue_bnhLn.permute([0, 1, 3, 2, 4]); // [B,n,L,H,N]
    let d_c_blue_bnlrhn: Tensor<B, 6> =
        d_c_blue_bnLhn.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
    let d_v_orange_bnLhp = d_v_orange_bnhLp.permute([0, 1, 3, 2, 4]); // [B,n,L,H,P]
    let d_v_orange_bnlrhp: Tensor<B, 6> =
        d_v_orange_bnLhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

    // ═══════════════════════════════════════════════════════════════════════
    // K1 BACKWARD + SUM CONTRIBUTIONS
    // ═══════════════════════════════════════════════════════════════════════
    let d_da_cumsum_bhnl =
        d_da_blue_bhnl + d_da_orange_bhnl + d_da_cumsum_k3_bhnl + d_da_cumsum_k4_bhnl;
    san(&d_da_cumsum_bhnl);

    // K1 inverse: da_cumsum[l] = cumsum(da)[l]  →  d_da[l] = Σ_{k ≥ l} d_da_cumsum[k]
    //
    // Suffix sum:  d_da[l] = total_sum − cumsum(d_da_cumsum)[l-1] (cumsum[-1] = 0).
    let d_da_bhnl = {
        let d_total = d_da_cumsum_bhnl
            .clone()
            .sum_dim(3)
            .expand([batch, nheads, nchunks, chunk_len]);
        let prefix = d_da_cumsum_bhnl.cumsum(3);
        let zeros_bhn1 = Tensor::<B, 4>::zeros([batch, nheads, nchunks, 1], &device);
        let prefix_shifted = Tensor::cat(vec![zeros_bhn1, prefix.narrow(3, 0, chunk_len - 1)], 3);
        d_total - prefix_shifted
    };
    san(&d_da_bhnl);
    // Undo permute [0,3,1,2] applied to da_bnlh: inverse is [0,2,3,1].
    let d_da_bnlh = d_da_bhnl.permute([0, 2, 3, 1]);

    // ── Combine per-input gradient contributions ──────────────────────────
    let d_v_bnlrhp = d_v_k3_bnlrhp + d_v_orange_bnlrhp;
    let d_b_bnlrhn = d_b_k2_bnlrhn + d_b_k3_bnlrhn;
    let d_c_bnlrhn = d_c_k2_bnlrhn + d_c_blue_bnlrhn;

    san(&d_v_bnlrhp);
    san(&d_da_bnlh);
    san(&d_b_bnlrhn);
    san(&d_c_bnlrhn);
    san(&d_initial_state_bhpr);

    CombinedGrads {
        d_v_bnlrhp,
        d_da_bnlh,
        d_b_bnlrhn,
        d_c_bnlrhn,
        d_initial_state_bhpr,
    }
}
