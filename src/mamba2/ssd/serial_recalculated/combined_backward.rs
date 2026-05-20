use crate::mamba2::ssd::serial;
use crate::utils::sanity::sanity as san;
use burn::prelude::*;

pub struct CombinedGrads<B: Backend> {
    pub d_x_bnlhp: Tensor<B, 5>,
    pub d_dt_discretized_bhnl: Tensor<B, 4>,
    pub d_b_bnlgr: Tensor<B, 5>,
    pub d_c_bnlgr: Tensor<B, 5>,
    pub d_d_h: Tensor<B, 1>,
    pub d_initial_state_bhpr: Tensor<B, 4>,
    pub d_a_decay_h: Tensor<B, 1>,
}

/// Same as [k3_ssd_chunk_state](serial::k3_ssd_chunk_state) but return some intermediaries
/// that are useful to the custom backward operation.
///
/// Returns:
/// - intra_chunk_state_bnhpr
/// - b_bar_scale_bhnl
/// - forward_decay_to_chunk_end_bhnl
/// - b_scaled_bnhlr
pub fn k3_ssd_chunk_state_extended<B: Backend>(
    x_bnlhp: Tensor<B, 5>,
    b_bnlgr: Tensor<B, 5>,
    da_cumsum_bhnl: Tensor<B, 4>,
    dt_discretized_bhnl: Tensor<B, 4>,
) -> (Tensor<B, 5>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 5>) {
    use burn::tensor::s;

    let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
    let [.., ngroups, state_rank] = b_bnlgr.dims();

    // permute b and x to prepare them for the mamtul
    // - 1/15: permute: (x_bnlhp [in][*]) -> (x_bnhpl)
    let x_bnhpl = x_bnlhp.clone().permute([0, 1, 3, 4, 2]);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, chunk_len],
        x_bnhpl.dims()
    );
    // - 2: permute: (b_bnlgr [in][*]) -> (b_bnglr)
    let b_bnglr = b_bnlgr.permute([0, 1, 3, 2, 4]); // note: still in groups instead of heads
    assert_eq!(
        [batch, nchunks, ngroups, chunk_len, state_rank],
        b_bnglr.dims()
    );

    // Expand B from ngroups to nheads by repeating each group's
    // projection across all heads_per_group heads in that group.
    let heads_per_group = nheads / ngroups;
    let b_bnhlr = b_bnglr
        // - 3: unsqueeze: (b_bnglr) -> (b_bng1lr)
        .unsqueeze_dim::<6>(3) // b_bng1lr
        // - 4: expand: (b_bng1lr) -> (b_bngHlr)
        .expand([
            batch,
            nchunks,
            ngroups,
            heads_per_group,
            chunk_len,
            state_rank,
        ]) // b_bngHlr
        // - 5: reshape: (b_bngHlr) -> (b_bnhlr)
        .reshape([batch, nchunks, nheads, chunk_len, state_rank]);

    // scale b
    let da_cumsum_last_in_chunk_bhn1 =
        // - 6: slice: (da_cumsum_bhnl [in][*]) -> (da_cumsum_last_in_chunk_bhn1)
        da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
    assert_eq!(
        [batch, nheads, nchunks, 1],
        da_cumsum_last_in_chunk_bhn1.dims()
    );

    // - 7: expand: (da_cumsum_last_in_chunk_bhn1) -> (da_cumsum_last_bhnl)
    let da_cumsum_last_bhnl =
        da_cumsum_last_in_chunk_bhn1.expand([batch, nheads, nchunks, chunk_len]);
    // - 8: sub: (da_cumsum_last_bhnl, da_cumsum_bhnl [from K1][*]) -> (da_delta_bhnl)
    let da_delta_bhnl = da_cumsum_last_bhnl - da_cumsum_bhnl.clone();
    san(&da_delta_bhnl);
    // - 9: exp: (da_delta_bhnl) -> (forward_decay_to_chunk_end_bhnl [+])
    let forward_decay_to_chunk_end_bhnl = da_delta_bhnl.exp();
    assert_eq!(
        [batch, nheads, nchunks, chunk_len],
        forward_decay_to_chunk_end_bhnl.dims()
    );
    san(&forward_decay_to_chunk_end_bhnl);

    // - 10: mul: (forward_decay_to_chunk_end_bhnl [+], dt_discretized_bhnl [in][*]) -> (b_bar_scale_bhnl [+])
    let b_bar_scale_bhnl = forward_decay_to_chunk_end_bhnl.clone() * dt_discretized_bhnl.clone();
    assert_eq!([batch, nheads, nchunks, chunk_len], b_bar_scale_bhnl.dims());
    san(&b_bar_scale_bhnl);

    // - 11: permute: (b_bar_scale_bhnl [+]) -> (b_bar_scale_bnhl)
    let b_bar_scale_bnhl = b_bar_scale_bhnl.clone().permute([0, 2, 1, 3]);
    assert_eq!([batch, nchunks, nheads, chunk_len], b_bar_scale_bnhl.dims());
    let b_bar_scale_bnhlr = b_bar_scale_bnhl
        // - 12: unsqueeze: (b_bar_scale_bnhl) -> (b_bar_scale_bnhl1)
        .unsqueeze_dim::<5>(4) // b_bar_scale_bnhl1
        // - 13: expand: (b_bar_scale_bnhl1) -> (b_bar_scale_bnhlr)
        .expand([batch, nchunks, nheads, chunk_len, state_rank]);
    // - 14: mul: (b_bnhlr, b_bar_scale_bnhlr) -> (b_scaled_bnhlr [+])
    let b_scaled_bnhlr = b_bnhlr * b_bar_scale_bnhlr;
    assert_eq!(
        [batch, nchunks, nheads, chunk_len, state_rank],
        b_scaled_bnhlr.dims()
    );
    san(&b_scaled_bnhlr);

    // - 15/15: matmul: (x_bnhpl, b_scaled_bnhlr [+]) -> (intra_chunk_state_bnhpr [out][!])
    let intra_chunk_state_bnhpr: Tensor<B, 5> = x_bnhpl.matmul(b_scaled_bnhlr.clone());
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        intra_chunk_state_bnhpr.dims()
    );
    san(&intra_chunk_state_bnhpr);
    (
        intra_chunk_state_bnhpr,
        b_bar_scale_bhnl,
        forward_decay_to_chunk_end_bhnl,
        b_scaled_bnhlr,
    )
}

/// Core gradient computation.  All arguments use the shapes from the forward.
///
/// `d_y_bnlhp`         : upstream gradient of the scan output  [B,N,L,H,P]
/// `d_final_bhpr`      : upstream gradient of the final state  [B,H,P,R]
///
/// Returns one `CombinedGrads` struct containing gradients for all 7 inputs.
#[allow(clippy::too_many_arguments)]
pub fn combined_backward<B: Backend>(
    d_y_bnlhp: Tensor<B, 5>,
    d_final_bhpr: Tensor<B, 4>,
    // Saved forward inputs
    x_bnlhp: Tensor<B, 5>,
    dt_discretized_bhnl: Tensor<B, 4>,
    b_bnlgr: Tensor<B, 5>,
    c_bnlgr: Tensor<B, 5>,
    d_h: Tensor<B, 1>,
    initial_state_bhpr: Tensor<B, 4>,
    a_decay_h: Tensor<B, 1>,
) -> CombinedGrads<B> {
    let [batch, nheads, nchunks, chunk_len] = dt_discretized_bhnl.dims();
    let [.., per_head_dim] = x_bnlhp.dims();
    let [.., ngroups, state_rank] = b_bnlgr.dims();
    let heads_per_group = nheads / ngroups;
    let device = dt_discretized_bhnl.device();

    san(&d_y_bnlhp);
    san(&d_final_bhpr);
    san(&x_bnlhp);
    san(&dt_discretized_bhnl);
    san(&b_bnlgr);
    san(&c_bnlgr);
    san(&d_h);
    san(&initial_state_bhpr);
    san(&a_decay_h);

    // ═══════════════════════════════════════════════════════════════════════
    // RECOMPUTE FORWARD INTERMEDIATES (the memory-saving heart of this op)
    // ═══════════════════════════════════════════════════════════════════════

    // K1 recomputation ─────────────────────────────────────────────────────
    // da_cumsum is not saved across the boundary; recompute from dt and a_decay.
    let (da_cumsum_bhnl, da_chunk_end_bhn) =
        serial::k1_ssd_chunk_cumsum(dt_discretized_bhnl.clone(), a_decay_h.clone());
    san(&da_cumsum_bhnl);
    san(&da_chunk_end_bhn);

    // K2 ───────────────────────────────────────────────────────────────────
    let cb_bngll = serial::k2_ssd_bmm(c_bnlgr.clone(), b_bnlgr.clone());
    // let cb_bngll = k2_forward(&c_bnlgr, &b_bnlgr);          // [B,N,G,L,L]
    san(&cb_bngll);

    // K3 (with intermediates) ──────────────────────────────────────────────
    let (
        intra_chunk_state_bnhpr,
        b_bar_scale_bhnl,
        forward_decay_to_chunk_end_bhnl,
        b_scaled_bnhlr,
    ) = k3_ssd_chunk_state_extended(
        x_bnlhp.clone(),
        b_bnlgr.clone(),
        da_cumsum_bhnl.clone(),
        dt_discretized_bhnl.clone(),
    );
    san(&intra_chunk_state_bnhpr);
    san(&b_bar_scale_bhnl);
    san(&forward_decay_to_chunk_end_bhnl);
    san(&b_scaled_bnhlr);

    // K4 ───────────────────────────────────────────────────────────────────
    let (chunk_input_state_bnhpr, _final_state_bhpr): (Tensor<B, 5>, Tensor<B, 4>) =
        serial::k4_ssd_state_passing(
            intra_chunk_state_bnhpr.clone(),
            da_chunk_end_bhn.clone(),
            initial_state_bhpr,
        );
    san(&chunk_input_state_bnhpr);
    san(&_final_state_bhpr);

    // ═══════════════════════════════════════════════════════════════════════
    // K5 BACKWARD
    // ═══════════════════════════════════════════════════════════════════════
    // Expand CB for all heads
    let cb_bnhll = cb_bngll
        .clone()
        .unsqueeze_dim::<6>(3) // cb_bng1ll
        .expand([
            batch,
            nchunks,
            ngroups,
            heads_per_group,
            chunk_len,
            chunk_len,
        ]) // cb_bngHll
        .reshape([batch, nchunks, nheads, chunk_len, chunk_len]);

    // Reshape inputs to [B,N,H,L,...] convention used inside K5
    let da_cumsum_bnhl: Tensor<B, 4> = da_cumsum_bhnl.permute([0, 2, 1, 3]);
    let dt_bnhl: Tensor<B, 4> = dt_discretized_bhnl.clone().permute([0, 2, 1, 3]);
    let x_bnhlp: Tensor<B, 5> = x_bnlhp.clone().permute([0, 1, 3, 2, 4]);
    let d_y_bnhlp: Tensor<B, 5> = d_y_bnlhp.clone().permute([0, 1, 3, 2, 4]);

    // GQA-expand C: [B,N,L,G,R] → [B,N,H,L,R]
    let c_bnhlr = c_bnlgr
        .clone()
        .unsqueeze_dim::<6>(4) // c_bnlg1r
        .expand([
            batch,
            nchunks,
            chunk_len,
            ngroups,
            heads_per_group,
            state_rank,
        ]) // c_bnlgHr
        .reshape([batch, nchunks, chunk_len, nheads, state_rank]) // c_bnlhr
        .permute([0, 1, 3, 2, 4]);

    // ── SKIP backward ──────────────────────────────────────────────────────
    // - | 36/36: add: (y_partial_bnlhp, skip_bnlhp) -> (y_bnlhp [out])
    // - | (d_skip_bnlhp = d_y_bnlhp)
    let d_skip_bnlhp = d_y_bnlhp.clone();
    //
    //
    // For d_h:
    // - - | 33: mul: (d_bnlhp, x_bnlhp[*]) -> (skip_bnlhp)
    // - - | (d_d_bnlhp = d_skip_bnlhp * x_bnlhp)
    // - - | 32: expand: (d_111h1) -> (d_bnlhp)
    // - - | 31: unsqueeze-dims: (d_h [*]) -> (d_111h1)
    //
    // - - | d_d[h] = Σ_{b,n,l,p} dy * x   — use permute+reshape to avoid chained sum_dim
    let d_d_h = {
        // [B,N,L,H,P] → permute to [H,B,N,L,P] → reshape [H, rest] → sum → [H]
        d_skip_bnlhp.clone()
            .permute([3, 0, 1, 2, 4]) // d_y_hbnlp
            .reshape([nheads, batch * nchunks * chunk_len * per_head_dim]) // d_y_hBNLP
            * x_bnlhp
                .clone()
                .permute([3, 0, 1, 2, 4]) // x_hbnlp
                .reshape([nheads, batch * nchunks * chunk_len * per_head_dim]) // x_hBNLP
    }
    .sum_dim(1) // d_d_h1
    .reshape([nheads]);
    san(&d_d_h);
    //
    // For d_x:
    // - - | 33: mul: (d_bnlhp, x_bnlhp[*]) -> (skip_bnlhp)
    // - - | (d_x_skip_bnlhp = d_skip_bnlhp * d_bnlhp)
    let d_x_skip_bnlhp = d_skip_bnlhp
        * d_h
            .clone()
            .unsqueeze_dims::<5>(&[0, 1, 2, 4]) // d_111h1
            // d_bnlhp
            .expand([batch, nchunks, chunk_len, nheads, per_head_dim]);
    san(&d_x_skip_bnlhp);

    // ── BLUE backward ──────────────────────────────────────────────────────
    // - | 36/36: add: (y_partial_bnlhp, skip_bnlhp) -> (y_bnlhp [out])
    // - | (d_y_partial_bnlhp = d_y_bnlhp)
    let d_y_partial_bnhlp = d_y_bnhlp.clone();
    //
    // - | 35: permute: (y_partial_bnhlp) -> (y_partial_bnlhp)
    // - | 34: add: (blue_scaled_bnhlp, orange_bnhlp) -> (y_partial_bnhlp)
    // - | (d_blue_scaled_bnhlp = d_y_partial_bnhlp)
    let d_blue_scaled_bnhlp = d_y_partial_bnhlp;
    // - | 16: mul: (blue_bnhlp, exp_da_cumsum_bnhlp) -> (blue_scaled_bnhlp)
    // - | (d_blue_bnhlp = d_blue_scaled_bnhlp * exp_da_cumsum_bnhlp)
    //
    // - | blue[b,n,h,l,p] = exp(da[b,n,h,l]) * Σ_r C[b,n,h,l,r] * state[b,n,h,p,r]
    let exp_da_cumsum_bnhl: Tensor<B, 4> = da_cumsum_bnhl.clone().exp();
    san(&exp_da_cumsum_bnhl);
    let exp_da_cumsum_bnhlp = exp_da_cumsum_bnhl.clone().unsqueeze_dim::<5>(4).expand([
        batch,
        nchunks,
        nheads,
        chunk_len,
        per_head_dim,
    ]);
    let d_blue_bnhlp: Tensor<B, 5> = d_blue_scaled_bnhlp.clone() * exp_da_cumsum_bnhlp.clone();
    san(&d_blue_bnhlp);
    //
    // For d_chunk_input_state_bnhpr:
    // - | 15: matmul: (c_bnhlr, chunk_input_state_bnhrp) -> (blue_bnhlp)
    // - - | (d_chunk_input_state_bnhrp = c_bnhlr^T @ d_blue_bnhlp)
    // - - | 14: permute: (chunk_input_state_bnhpr [!]) -> (chunk_input_state_bnhrp)
    //
    // - - | d_state[b,n,h,p,r] = Σ_l (scaled_dy[b,n,h,l,p] * C[b,n,h,l,r])
    // - - |  = C^T[R,L] @ scaled_dy[L,P]  for fixed (b,n,h)
    // - - |  [B,N,H,R,L] @ [B,N,H,L,P] → [B,N,H,R,P] → permute → [B,N,H,P,R]
    let d_chunk_input_state_bnhpr = c_bnhlr
        .clone()
        .permute([0, 1, 2, 4, 3]) // c_bnhrl
        .matmul(d_blue_bnhlp.clone()) // d_chunk_input_state_bnhrp
        .permute([0, 1, 2, 4, 3]);
    san(&d_chunk_input_state_bnhpr);
    //
    // For d_c from BLUE:
    // - | 15: matmul: (c_bnhlr, chunk_input_state_bnhrp) -> (blue_bnhlp)
    // - - | (d_c_bnhlr = d_blue_bnhlp @ chunk_input_state_bnhrp^T)
    // - - | 7: permute: (c_bnlhr) -> (c_bnhlr)
    // - - | 6: reshape: (c_bnlgHr) -> (c_bnlhr)
    // - - | 5: expand: (c_bnlg1r) -> (c_bnlgHr)
    // - - | 4: unsqueeze: (c_bnlgr [*]) -> (c_bnlg1r)
    //
    // - - | d_C[l,r] = Σ_p scaled_dy[l,p] * state[p,r]
    // - - |  [B,N,H,L,P] @ [B,N,H,P,R] → [B,N,H,L,R]
    let d_c_blue_bnhlr = d_blue_bnhlp.clone().matmul(chunk_input_state_bnhpr.clone());
    san(&d_c_blue_bnhlr);
    // - - | GQA reduce: [B,N,H,L,R] → [B,N,L,G,R]
    let d_c_blue_bnlgr = d_c_blue_bnhlr
        .reshape([
            batch,
            nchunks,
            ngroups,
            heads_per_group,
            chunk_len,
            state_rank,
        ]) // d_c_blue_bngHlr
        .sum_dim(3) // d_c_blue_bng1lr
        .squeeze_dim::<5>(3) // d_c_blue_bnglr
        .permute([0, 1, 3, 2, 4]);
    san(&d_c_blue_bnlgr);
    //
    // For d_da_cumsum from BLUE:
    // - | 16: mul: (blue_bnhlp, exp_da_cumsum_bnhlp) -> (blue_scaled_bnhlp)
    // - | (d_exp_da_cumsum_bnhlp = d_blue_scaled_bnhlp * blue_bnhlp)
    let blue_bnhlp = c_bnhlr
        .clone()
        .matmul(chunk_input_state_bnhpr.clone().permute([0, 1, 2, 4, 3])); // replay forward step 15
    san(&blue_bnhlp);
    let d_exp_da_cumsum_bnhlp = d_blue_scaled_bnhlp.clone() * blue_bnhlp;
    san(&d_exp_da_cumsum_bnhlp);
    //
    // - | blue_no_scale = C @ state^T  [L,P]
    // - - | 13: expand: (exp_da_cumsum_bnhl1) -> (exp_da_cumsum_bnhlp)
    // - - | 12: unsqueeze: (exp_da_cumsum_bnhl) -> (exp_da_cumsum_bnhl1)
    // - - | 11: exp: (da_cumsum_bnhl) -> (exp_da_cumsum_bnhl)
    // - - | (d_da_cumsum_bnhl = d_exp_da_cumsum_bnhlp * exp(da_cumsum_bnhl))
    // - - | 1/36: permute: (da_cumsum_bhnl [*]) -> (da_cumsum_bnhl)
    //
    // - - | d_da[l] = Σ_p dy[l,p] * exp_da[l] * blue_no_scale[l,p]
    let d_da_blue_bnhl = (d_exp_da_cumsum_bnhlp * exp_da_cumsum_bnhlp)
        .sum_dim(4) // d_da_blue_bnhl1
        .squeeze_dim::<4>(4);
    san(&d_da_blue_bnhl);
    let d_da_blue_bhnl = d_da_blue_bnhl.permute([0, 2, 1, 3]);

    // ── ORANGE backward ─────────────────────────────────────────────────────
    //  y_orange[l,p] = Σ_{s≤l} CB[l,s] * exp(da[l]-da[s]) * dt[s] * x[s,p]
    // Precompute weight matrix CB_w [B,N,H,L_tgt,L_src]
    // replay forward steps 17-29
    let da_cumsum_target_bnhll = da_cumsum_bnhl
        .clone()
        .unsqueeze_dim::<5>(4) // da_cumsum_bnhl1 // forward step 17
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]); // forward step 18
    let da_cumsum_source_bnhll = da_cumsum_bnhl
        .clone()
        .unsqueeze_dim::<5>(3) // da_cumsum_bnh1l // forward step 19
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]); // forward step 20
    let da_cumsum_diff_bnhll = da_cumsum_target_bnhll - da_cumsum_source_bnhll; // forward step 21
    san(&da_cumsum_diff_bnhll);
    // forward step 21.1: built at [L,L] and broadcast — mask values do not depend on (b,n,h).
    let causal_mask_bnhll: Tensor<B, 5, burn::prelude::Bool> =
        Tensor::<B, 2, burn::prelude::Bool>::tril_mask([chunk_len, chunk_len], 0, &device)
            .reshape([1, 1, 1, chunk_len, chunk_len])
            .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    // forward step 21.2
    // Causal mask and exp stabilizer (-inf above the main diagonal, 0 elsewhere).
    let da_cumsum_diff_masked_bnhll =
        da_cumsum_diff_bnhll.mask_fill(causal_mask_bnhll.clone(), f32::NEG_INFINITY);
    let da_cumsum_diff_exp_bnhll = (da_cumsum_diff_masked_bnhll).exp(); // forward steps 22
    san(&da_cumsum_diff_exp_bnhll);
    let dt_source_bnhll = dt_bnhl
        .clone()
        .unsqueeze_dim::<5>(3) // dt_bnh1l // forward step 23
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]); // forward step 24
    // // Causal mask (0 above the main diagonal, 1 elsewhere).
    // let causal_mask_bnhll =
    //     Tensor::ones([batch, nchunks, nheads, chunk_len, chunk_len], &device).tril(0); // forward steps 25-26
    // CB_w[l,s] = CB[l,s] * decay[l,s] * dt[s] * mask[l,s]
    let orange_lhs_partial1_bnhll: Tensor<B, 5> = // forward step 27
        cb_bnhll.clone() * da_cumsum_diff_exp_bnhll.clone();
    san(&orange_lhs_partial1_bnhll);
    let orange_lhs_partial2_bnhll: Tensor<B, 5> = // forward step 28
        orange_lhs_partial1_bnhll.clone() * dt_source_bnhll.clone();
    san(&orange_lhs_partial2_bnhll);
    // let orange_lhs_partial3_bnhll: Tensor<B, 5> = // forward step 29
    //     orange_lhs_partial2_bnhll.clone() * causal_mask_bnhll.clone();
    //
    // Backwads:
    // - | 36/36: add: (y_partial_bnlhp, skip_bnlhp) -> (y_bnlhp [out])
    // - | (d_y_partial_bnlhp = d_y_bnlhp)
    let d_y_partial_bnhlp = d_y_bnhlp.clone();
    // - | 35: permute: (y_partial_bnhlp) -> (y_partial_bnlhp)
    // - | 34: add: (blue_scaled_bnhlp, orange_bnhlp) -> (y_partial_bnhlp)
    // - | (d_orange_bnhlp = d_y_partial_bnhlp)
    let d_orange_bnhlp = d_y_partial_bnhlp;
    // - | 30: matmul: (orange_lhs_partial2_bnhll, x_bnhlp) -> (orange_bnhlp)
    // - | (d_orange_lhs_partial2_bnhll = d_orange_bnhlp @ x_bnhlp^T)
    // d_CB_w: dy @ x^T   [B,N,H,L_tgt,L_src]
    let d_orange_lhs_partial2_bnhll = d_orange_bnhlp
        .clone()
        .matmul(x_bnhlp.clone().permute([0, 1, 2, 4, 3])); // [B,N,H,L_tgt,L_src]
    san(&d_orange_lhs_partial2_bnhll);
    //
    // - | For d_x:
    // - - | (d_x_bnhlp = orange_lhs_partial2_bnhll^T @ d_orange_bnhlp)
    // - - | d_x from ORANGE: CB_w^T @ dy  (transpose source/target dims)
    // - - |  [B,N,H,L_src,L_tgt] @ [B,N,H,L_tgt,P] → [B,N,H,L_src,P]
    let d_x_orange_bnhlp = orange_lhs_partial2_bnhll
        .clone()
        .permute([0, 1, 2, 4, 3]) // [B,N,H,L_src,L_tgt]
        .matmul(d_orange_bnhlp.clone()); // [B,N,H,L_src,P]
    san(&d_x_orange_bnhlp);
    //
    // - | 21.2: mask-fill: (.., ..) -> (..)
    // Bring the (step 21.2) causal mask ahead: above upper diagonal set to 0.
    let d_orange_lhs_partial2_bnhll = d_orange_lhs_partial2_bnhll.mask_fill(causal_mask_bnhll, 0.);
    san(&d_orange_lhs_partial2_bnhll);
    // - | 28: mul: (orange_lhs_partial1_bnhll, dt_source_bnhll) -> (orange_lhs_partial2_bnhll)
    let d_orange_lhs_partial1_bnhll = d_orange_lhs_partial2_bnhll.clone() * dt_source_bnhll.clone();
    san(&d_orange_lhs_partial1_bnhll);
    // - | For d_dt from ORANGE:
    // - - | 24: expand: (dt_bnh1l) -> (dt_source_bnhll)
    // - - | 23: unsqueeze: (dt_bnhl) -> (dt_bnh1l)
    // - - | 2: permute: (dt_discretized_bhnl [*]) -> (dt_bnhl)
    // - - | d_dt[s] = Σ_{l≥s} d_CB_w[l,s] * CB[l,s] * decay[l,s] * mask[l,s]
    // - - |  = (d_cb_w * cb * decay * mask).sum(L_tgt dim=3)
    let d_dt_orange_bnhl = (d_orange_lhs_partial2_bnhll.clone()
        * orange_lhs_partial1_bnhll.clone())
    .sum_dim(3) // d_dt_orange_bnh1l
    .squeeze_dim::<4>(3);
    san(&d_dt_orange_bnhl);
    let d_dt_orange_bhnl = d_dt_orange_bnhl.permute([0, 2, 1, 3]);
    //
    // - | For d_da from ORANGE:
    // - - | decay = exp(da_tgt - da_src)
    // - - | d_decay = d_CB_w * CB * dt_src * mask
    // - - | d_da_tgt[l] += Σ_s (d_decay * decay)[l,s]
    // - - | d_da_src[s] -= Σ_l (d_decay * decay)[l,s]
    // - - | 27: mul: (cb_bnhll, da_cumsum_diff_exp_bnhll) -> (orange_lhs_partial1_bnhll)
    let d_da_cumsum_diff_exp_bnhll = d_orange_lhs_partial1_bnhll.clone() * cb_bnhll.clone();
    san(&d_da_cumsum_diff_exp_bnhll);
    // - - | 22: exp: (da_cumsum_diff_bnhll) -> (da_cumsum_diff_exp_bnhll)
    // - - | (d_da_cumsum_diff_bnhll = d_da_cumsum_diff_exp_bnhll * exp(da_cumsum_diff_bnhll))
    let d_da_cumsum_diff_bnhll = d_da_cumsum_diff_exp_bnhll * da_cumsum_diff_exp_bnhll.clone();
    san(&d_da_cumsum_diff_bnhll);
    // - - | 21: sub: (da_cumsum_target_bnhll, da_cumsum_source_bnhll) -> (da_cumsum_diff_bnhll)
    // - - | 20: expand: (da_cumsum_bnh1l) -> (da_cumsum_source_bnhll)
    // - - | 19: unsqueeze: (da_cumsum_bnhl) -> (da_cumsum_bnh1l)
    // - - | 18: expand: (da_cumsum_bnhl1) -> (da_cumsum_target_bnhll)
    // - - | 17: unsqueeze: (da_cumsum_bnhl) -> (da_cumsum_bnhl1)
    // - - | 1/36: permute: (da_cumsum_bhnl [*]) -> (da_cumsum_bnhl)
    let d_da_tgt_bnhl = d_da_cumsum_diff_bnhll
        .clone()
        .sum_dim(4) // d_da_cumsum_diff_bnhl1
        .squeeze_dim::<4>(4);
    san(&d_da_tgt_bnhl);
    let d_da_src_bnhl = d_da_cumsum_diff_bnhll
        .sum_dim(3) // d_da_cumsum_diff_bnh1l
        .squeeze_dim::<4>(3);
    san(&d_da_src_bnhl);
    let d_da_orange_bhnl = (d_da_tgt_bnhl - d_da_src_bnhl).permute([0, 2, 1, 3]); // [B,H,N,L]
    san(&d_da_orange_bhnl);
    //
    // - | For d_cb:
    // - - | 27: mul: (cb_bnhll, da_cumsum_diff_exp_bnhll) -> (orange_lhs_partial1_bnhll)
    let d_cb_bnhll = d_orange_lhs_partial1_bnhll * da_cumsum_diff_exp_bnhll.clone();
    san(&d_cb_bnhll);
    // - - | d_CB (per head, before GQA reduction):
    // - - |  CB_w = CB * decay * dt * mask  →  d_CB[l,s] = d_CB_w[l,s] * decay[l,s] * dt[s] * mask
    // - - | GQA reduce: [B,N,H,L,L] → [B,N,G,L,L]
    let d_cb_bngll = d_cb_bnhll
        .reshape([
            batch,
            nchunks,
            ngroups,
            heads_per_group,
            chunk_len,
            chunk_len,
        ]) // d_cb_bngHll
        .sum_dim(3) // d_cb_bng1ll
        .squeeze_dim::<5>(3);
    san(&d_cb_bngll);

    // ═══════════════════════════════════════════════════════════════════════
    // K4 BACKWARD (reverse serial recurrence)
    // ═══════════════════════════════════════════════════════════════════════
    //
    // - 5/5: stack: (chunk_input_state_vec_bhpr [!]) -> (chunk_input_state_bnhpr [out][!])
    // - 4: vec-pop: (chunk_input_state_vec_bhpr [vec][!]) -> (final_state_bhpr [elem][out][!])
    // - 3: serial-loop: (0..nchunks)
    //
    // last d_running_state_bhpr:
    let mut d_running_state_bhpr: Tensor<B, 4> = d_final_bhpr; // [B,H,P,R]
    //
    // d_intra[c] and d_da_end[c] collected during reverse traversal.
    let mut d_intra_slices: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks);
    let mut d_da_end_bh_slices: Vec<Tensor<B, 2>> = Vec::with_capacity(nchunks);
    //
    for i_chunk in (0..nchunks).rev() {
        // access re-calculated running state
        let running_state_bhpr = chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim(1);
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            running_state_bhpr.dims()
        );
        //
        // - 3.9/3.9: vec-push: (running_state_bhpr [elem]) -> (chunk_input_state_vec_bhpr [vec][!])
        d_intra_slices.push(d_running_state_bhpr.clone());
        //
        // - 3.8: add: (running_state_bhpr, intra_state_bhpr) -> (running_state_bhpr)
        let _d_intra_state_bhpr = d_running_state_bhpr.clone();
        //
        // - 3.7: mul: (decay_bhpr, running_state_bhpr) -> (running_state_bhpr)
        let d_decay_bhpr = d_running_state_bhpr.clone() * running_state_bhpr.clone();
        san(&d_decay_bhpr);
        // recalculate decay_bhpr
        let decay_bhpr = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk]) // da_chunk_end_bh1 // replay forward step 3.3
            .exp() // exp_da_chunk_end_bh1 // replay forward step 3.4
            .unsqueeze_dim::<4>(3) // exp_da_chunk_end_bh11 // replay forward step 3.5
            .expand([batch, nheads, per_head_dim, state_rank]); // replay forward step 3.6
        san(&decay_bhpr);
        // - 3.6: expand: (exp_da_chunk_end_bh11) -> (decay_bhpr)
        // - 3.5: unsqueeze: (exp_da_chunk_end_bh1) -> (exp_da_chunk_end_bh11)
        // - 3.4: exp: (da_chunk_end_bh1) -> (exp_da_chunk_end_bh1)
        // (d_da_chunk_end_bh1 = d_exp_da_chunk_end_bh1 * exp(da_chunk_end_bh1))
        // - 3.3: slice: (da_chunk_end_bhn [in][*]) -> (da_chunk_end_bh1)
        let d_da_chunk_end_bhpr = d_decay_bhpr * decay_bhpr.clone(); // note: decay is expanded exp(da_chunk_end)
        san(&d_da_chunk_end_bhpr);
        let d_da_chunk_end_bh = d_da_chunk_end_bhpr
            .reshape([batch, nheads, per_head_dim * state_rank]) // d_da_chunk_end_bhPR
            .sum_dim(2) // d_da_chunk_end_bh1
            .squeeze_dim::<2>(2);
        san(&d_da_chunk_end_bh);
        d_da_end_bh_slices.push(d_da_chunk_end_bh);
        //
        // - 3.2: squeeze: (intra_chunk_state_b1hpr) -> (intra_state_bhpr)
        // - 3.1/3.9: slice: (intra_chunk_state_bnhpr [in][!]) -> (intra_chunk_state_b1hpr)
        //
        // Propagate: d_running_state_bhpr_prev = scale * d_running_state_bhpr + d_chunk_input_state_bhpr
        //   (d_cis[c] = gradient of chunk_input_state[:, c] flowing in from K5 BLUE)
        let d_chunk_input_state_bhpr = d_chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // d_chunk_input_state_b1hpr // d_chunk_input_state_b1hpr
            .squeeze_dim::<4>(1);
        // TODO: understand this.
        d_running_state_bhpr = decay_bhpr * d_running_state_bhpr + d_chunk_input_state_bhpr;
        san(&d_running_state_bhpr);
    }
    // - 2: vec-push: (running_state_bhpr [elem]) -> (chunk_input_state_vec_bhpr [vec][!])
    // - 1/5: init-mut: (initial_state_bhpr [in][*]) -> (running_state_bhpr)
    //
    // After the loop, d_initial_state = the (reverse loop) tailing d_running_state_bhpr
    let d_initial_state_bhpr = d_running_state_bhpr;
    //
    // Restore natural order
    d_intra_slices.reverse();
    d_da_end_bh_slices.reverse();
    //
    let d_intra_chunk_state_bnhpr = Tensor::stack(d_intra_slices, 1);
    //
    // d_da_end_bhn [B,H,N]: scatter to last position of d_da_cumsum
    let d_da_end_bhn: Tensor<B, 3> = Tensor::stack(d_da_end_bh_slices, 2);
    //
    // TODO: understand this.
    // Pad to [B,H,N,L] — only last L-position is non-zero
    let d_da_cumsum_k4_bhnl = {
        let zeros = Tensor::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device);
        let d_da_end_bhn1 = d_da_end_bhn.unsqueeze_dim::<4>(3);
        Tensor::cat(vec![zeros, d_da_end_bhn1], 3)
    };

    // ═══════════════════════════════════════════════════════════════════════
    // K3 BACKWARD
    // ═══════════════════════════════════════════════════════════════════════
    let x_bnhpl = x_bnlhp.clone().permute([0, 1, 3, 4, 2]);
    // For d_x_bnlhp:
    // - 15/15: matmul: (x_bnhpl, b_scaled_bnhlr [+]) -> (intra_chunk_state_bnhpr [out][!])
    // - (d_x_bnhpl = d_intra_chunk_state_bnhpr @ b_scaled_bnhlr^T)
    let d_x_k3_bnhpl = d_intra_chunk_state_bnhpr
        .clone()
        .matmul(b_scaled_bnhlr.clone().permute([0, 1, 2, 4, 3]));
    san(&d_x_k3_bnhpl);
    // - 1/15: permute: (x_bnlhp [in][*]) -> (x_bnhpl)
    let d_x_k3_bnlhp = d_x_k3_bnhpl.permute([0, 1, 4, 2, 3]);
    //
    // - 15/15: matmul: (x_bnhpl, b_scaled_bnhlr [+]) -> (intra_chunk_state_bnhpr [out][!])
    // (d_b_scaled_bnhlr = x_bnhpl^T @ d_intra_chunk_state_bnhpr)
    let d_b_scaled_bnhlr = x_bnhpl
        .permute([0, 1, 2, 4, 3]) // x_bnhlp
        .matmul(d_intra_chunk_state_bnhpr);
    san(&d_b_scaled_bnhlr);
    //
    // For d_b:
    // - 14: mul: (b_bnhlr, b_bar_scale_bnhlr) -> (b_scaled_bnhlr [+])
    // - (d_b_bnhlr = d_b_scaled_bnhlr * b_bar_scale_bnhlr)
    let b_bar_scale_bnhlr = b_bar_scale_bhnl
        .clone()
        .permute([0, 2, 1, 3]) // b_bar_scale_bnhl // replay forward step 11
        .unsqueeze_dim::<5>(4) // b_bar_scale_bnhl1 // replay forward step 12
        .expand([batch, nchunks, nheads, chunk_len, state_rank]); // replay forward step 13
    let d_b_k3_bnhlr = d_b_scaled_bnhlr.clone() * b_bar_scale_bnhlr;
    san(&d_b_k3_bnhlr);
    // - 5: reshape: (b_bngHlr) -> (b_bnhlr)
    // - 4: expand: (b_bng1lr) -> (b_bngHlr)
    // - 3: unsqueeze: (b_bnglr) -> (b_bng1lr)
    // - 2: permute: (b_bnlgr [in][*]) -> (b_bnglr)
    // GQA reduce: [B,N,H,L,R] → [B,N,G,L,R] → [B,N,L,G,R]
    let d_b_k3_bnlgr = d_b_k3_bnhlr
        .reshape([
            batch,
            nchunks,
            ngroups,
            heads_per_group,
            chunk_len,
            state_rank,
        ]) // d_b_k3_bngHlr
        .sum_dim(3) // d_b_k3_bng1lr
        .squeeze_dim::<5>(3) // d_b_k3_bnglr
        .permute([0, 1, 3, 2, 4]);
    san(&d_b_k3_bnlgr);

    // - 14: mul: (b_bnhlr, b_bar_scale_bnhlr) -> (b_scaled_bnhlr [+])
    // - (d_b_bar_scale_bnhlr = d_b_scaled_bnhlr * b_bnhlr)
    // GQA-expand B back to per-head for the product: [B,N,G,L,R] → [B,N,H,L,R]
    let b_bnhlr = b_bnlgr
        .clone()
        .permute([0, 1, 3, 2, 4]) // b_bnglr // replay forward step 2
        .unsqueeze_dim::<6>(3) // b_bng1lr // replay forward step 3
        // b_bngHlr
        .expand([
            batch,
            nchunks,
            ngroups,
            heads_per_group,
            chunk_len,
            state_rank,
        ]) // replay forward step 4
        .reshape([batch, nchunks, nheads, chunk_len, state_rank]); // replay forward step 5
    let d_b_bar_scale_bnhlr = d_b_scaled_bnhlr.clone() * b_bnhlr;
    san(&d_b_bar_scale_bnhlr);
    // - 13: expand: (b_bar_scale_bnhl1) -> (b_bar_scale_bnhlr)
    // - 12: unsqueeze: (b_bar_scale_bnhl) -> (b_bar_scale_bnhl1)
    // - 11: permute: (b_bar_scale_bhnl [+]) -> (b_bar_scale_bnhl)
    let d_b_bar_scale_bhnl = d_b_bar_scale_bnhlr
        .sum_dim(4) // d_b_bar_scale_bnhl1
        .squeeze_dim::<4>(4) // d_b_bar_scale_bnhl
        .permute([0, 2, 1, 3]);
    san(&d_b_bar_scale_bhnl);
    //
    // For d_da_cumsum_bhnl:
    // - 10: mul: (forward_decay_to_chunk_end_bhnl [+], dt_discretized_bhnl [in][*]) -> (b_bar_scale_bhnl [+])
    // - (d_forward_decay_to_chunk_end_bhnl = d_b_bar_scale_bhnl * dt_discretized_bhnl)
    let d_forward_decay_to_chunk_end_bhnl =
        d_b_bar_scale_bhnl.clone() * dt_discretized_bhnl.clone();
    san(&d_forward_decay_to_chunk_end_bhnl);
    // - 9: exp: (da_delta_bhnl) -> (forward_decay_to_chunk_end_bhnl [+])
    // - (d_da_delta_bhnl = d_forward_decay_to_chunk_end_bhnl * exp(da_delta_bhnl))
    // note: forward_decay_to_chunk_end_bhnl = exp(da_delta_bhnl)
    let d_da_delta_bhnl =
        d_forward_decay_to_chunk_end_bhnl * forward_decay_to_chunk_end_bhnl.clone();
    san(&d_da_delta_bhnl);
    // - 8: sub: (da_cumsum_last_bhnl, da_cumsum_bhnl [from K1][*]) -> (da_delta_bhnl)
    let d_da_cumsum_last_bhnl = d_da_delta_bhnl.clone();
    let d_da_cumsum_sub_bhnl = -d_da_delta_bhnl.clone();
    // - 7: expand: (da_cumsum_last_in_chunk_bhn1) -> (da_cumsum_last_bhnl)
    // - 6: slice: (da_cumsum_bhnl [in][*]) -> (da_cumsum_last_in_chunk_bhn1)
    let d_da_cumsum_last_bhn = d_da_cumsum_last_bhnl
        .sum_dim(3) // d_da_cumsum_last_bhn1
        .squeeze_dim::<3>(3);
    san(&d_da_cumsum_last_bhn);
    //
    // For d_dt_discretized_bhnl:
    // - 10: mul: (forward_decay_to_chunk_end_bhnl [+], dt_discretized_bhnl [in][*]) -> (b_bar_scale_bhnl [+])
    // - (d_dt_discretized_bhnl = d_b_bar_scale_bhnl * forward_decay_to_chunk_end_bhnl)
    let d_dt_discretized_k3_bhnl = d_b_bar_scale_bhnl * forward_decay_to_chunk_end_bhnl;
    san(&d_dt_discretized_k3_bhnl);
    //

    // TODO: understand this.
    let d_da_cumsum_k3_bhnl = {
        let zeros = Tensor::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device);
        let d_last = d_da_cumsum_last_bhn.unsqueeze_dim::<4>(3);
        d_da_cumsum_sub_bhnl + Tensor::cat(vec![zeros, d_last], 3)
    };
    san(&d_da_cumsum_k3_bhnl);

    // ═══════════════════════════════════════════════════════════════════════
    // K2 BACKWARD (from d_cb_bngll)
    // ═══════════════════════════════════════════════════════════════════════
    let c_bnglr = c_bnlgr.clone().permute([0, 1, 3, 2, 4]);
    let b_bnglr = b_bnlgr.clone().permute([0, 1, 3, 2, 4]);
    // - 3/3: matmul: (c_bnglr, b_bngrl) -> (cb_bngll [out][!])
    // - cb[b,n,g,l,s] = Σ_r c[l,r]*b[s,r]  →  CB = C @ B^T
    // -  d_C_bngls = d_CB @ B   [B,N,G,L,L_src] @ [B,N,G,L_src,R] → [B,N,G,L,R]
    // -  d_B_bngls = d_CB^T @ C [B,N,G,L_src,L] @ [B,N,G,L,R]   → [B,N,G,L_src,R]
    let d_c_k2_bnglr = d_cb_bngll.clone().matmul(b_bnglr.clone());
    san(&d_c_k2_bnglr);
    let d_c_k2_bnlgr = d_c_k2_bnglr.permute([0, 1, 3, 2, 4]);

    let d_b_k2_bnglr = d_cb_bngll
        .permute([0, 1, 2, 4, 3]) // [B,N,G,L_src,L_tgt]
        .matmul(c_bnglr.clone()); // [B,N,G,L_src,R]
    san(&d_b_k2_bnglr);
    let d_b_k2_bnlgr = d_b_k2_bnglr.permute([0, 1, 3, 2, 4]); // [B,N,L,G,R]

    // ═══════════════════════════════════════════════════════════════════════
    // SUM GRADIENT CONTRIBUTIONS
    // ═══════════════════════════════════════════════════════════════════════

    // Accumulated gradient of the cumulative sum produced by K1.
    let d_da_cumsum_bhnl =
        d_da_blue_bhnl + d_da_orange_bhnl + d_da_cumsum_k3_bhnl + d_da_cumsum_k4_bhnl;
    san(&d_da_cumsum_bhnl);

    // ── K1 BACKWARD ────────────────────────────────────────────────────────
    // K1 forward: da_cumsum[l] = cumsum_l(dt[l] * a_decay)
    //
    // Reverse cumsum (suffix sum) converts d_da_cumsum → d_da:
    //   d_da[l] = sum_{k >= l} d_da_cumsum[k]
    //           = total_sum - cumsum(d_da_cumsum)[l-1]   (cumsum[-1] == 0)
    let d_da_cumsum_total_bhnl = d_da_cumsum_bhnl
        .clone()
        .sum_dim(3) // [B,H,N,1]
        .expand([batch, nheads, nchunks, chunk_len]);
    let prefix_sum_bhnl = d_da_cumsum_bhnl.clone().cumsum(3); // [B,H,N,L]
    let zeros_bhn1 = Tensor::<B, 4>::zeros([batch, nheads, nchunks, 1], &device);
    // prefix_sum shifted right by 1 (i.e., cumsum[l-1], with cumsum[-1] = 0)
    let prefix_sum_shifted_bhnl = Tensor::cat(
        vec![zeros_bhn1, prefix_sum_bhnl.narrow(3, 0, chunk_len - 1)],
        3,
    );
    let d_da_bhnl = d_da_cumsum_total_bhnl - prefix_sum_shifted_bhnl; // suffix sum [B,H,N,L]
    san(&d_da_bhnl);
    // d_dt from K1: d_dt = d_da * a_decay
    let a_decay_expand = a_decay_h
        .clone()
        .unsqueeze_dims::<4>(&[0, 2, 3])
        .expand([batch, nheads, nchunks, chunk_len]);
    let d_dt_k1_bhnl = d_da_bhnl.clone() * a_decay_expand;
    san(&d_dt_k1_bhnl);
    // d_a_decay_h from K1: d_a[h] = sum_{b,n,l} d_da[b,h,n,l] * dt[b,h,n,l]
    let d_a_decay_h = (d_da_bhnl * dt_discretized_bhnl.clone())
        .permute([1, 0, 2, 3]) // [H,B,N,L]
        .reshape([nheads, batch * nchunks * chunk_len])
        .sum_dim(1) // [H,1]
        .reshape([nheads]);
    san(&d_a_decay_h);

    let d_dt_discretized_bhnl = d_dt_orange_bhnl + d_dt_discretized_k3_bhnl + d_dt_k1_bhnl;
    san(&d_dt_discretized_bhnl);

    let d_x_orange_bnlhp = d_x_orange_bnhlp.permute([0, 1, 3, 2, 4]);
    let d_x_bnlhp = d_x_skip_bnlhp + d_x_k3_bnlhp + d_x_orange_bnlhp;
    san(&d_x_bnlhp);

    let d_b_bnlgr = d_b_k2_bnlgr + d_b_k3_bnlgr;
    san(&d_b_bnlgr);
    let d_c_bnlgr = d_c_k2_bnlgr + d_c_blue_bnlgr;
    san(&d_c_bnlgr);

    CombinedGrads {
        d_a_decay_h,
        d_dt_discretized_bhnl,
        d_x_bnlhp,
        d_b_bnlgr,
        d_c_bnlgr,
        d_d_h,
        d_initial_state_bhpr,
    }
}
