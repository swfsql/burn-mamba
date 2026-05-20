use crate::mamba2::ssd::serial;
use crate::utils::sanity::sanity as san;
use burn::prelude::*;

#[non_exhaustive]
pub struct CombinedGrads<B: Backend> {
    pub d_x_bnlhp: Tensor<B, 5>,
    pub d_dt_discretized_bhnl: Tensor<B, 4>,
    pub d_b_bnlgr: Tensor<B, 5>,
    pub d_c_bnlgr: Tensor<B, 5>,
    pub d_d_h: Tensor<B, 1>,
    pub d_initial_state_bhpr: Tensor<B, 4>,
    pub d_a_decay_h: Tensor<B, 1>,
    /// Local same-chunk contribution to `d_da_cumsum` from BLUE+ORANGE only
    /// (excludes K3 and K4 cross-chunk contributions). Exposed for the
    /// `out_x · dout − ddt · dt` oracle test from Tri Dao's reference
    /// (`_chunk_scan_bwd_ddAcs_unstable`). Test-only; absent in release builds.
    #[cfg(test)]
    pub d_da_local_bhnl: Tensor<B, 4>,
    /// Same-chunk d_dt contribution from ORANGE only (= what Tri Dao calls
    /// `ddt` in `_chunk_scan_bwd_ddAcs_unstable`). Test-only; absent in release
    /// builds.
    #[cfg(test)]
    pub d_dt_orange_bhnl: Tensor<B, 4>,
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
    // K5 + K4 FUSED BACKWARD (reverse per-chunk loop)
    // ═══════════════════════════════════════════════════════════════════════
    //
    // Peak memory: each iteration allocates [B,H,L,L] working tensors (vs the
    // [B,N,H,L,L] tensors a fully batched K5 backward would materialise). Saves
    // a factor of N (nchunks) in the dominant L×L terms.

    // ── SKIP backward (batched; allocates only [B,N,L,H,P] working sets) ───
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

    // ── Fused K5 (BLUE + ORANGE) + K4 reverse loop ─────────────────────────
    //
    // Per-iteration working set is [B,H,L,L] (not [B,N,H,L,L]). Per-chunk
    // gradients are collected in vecs and stacked once the loop is done.

    // [L,L] causal mask, broadcast as a view inside each iteration —
    // value does not depend on (b,n,h).
    let causal_mask_ll: Tensor<B, 2, burn::prelude::Bool> =
        Tensor::<B, 2, burn::prelude::Bool>::tril_mask([chunk_len, chunk_len], 0, &device);

    let mut d_x_orange_vec: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks); // [B,H,L,P]
    let mut d_dt_orange_vec: Vec<Tensor<B, 3>> = Vec::with_capacity(nchunks); // [B,H,L]
    let mut d_da_orange_vec: Vec<Tensor<B, 3>> = Vec::with_capacity(nchunks); // [B,H,L]
    let mut d_cb_vec: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks); // [B,G,L,L]
    let mut d_c_blue_vec: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks); // [B,H,L,R]
    let mut d_da_blue_vec: Vec<Tensor<B, 3>> = Vec::with_capacity(nchunks); // [B,H,L]
    let mut d_intra_slices: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks); // [B,H,P,R]
    let mut d_da_end_bh_slices: Vec<Tensor<B, 2>> = Vec::with_capacity(nchunks); // [B,H]

    let mut d_running_state_bhpr: Tensor<B, 4> = d_final_bhpr; // [B,H,P,R]

    for i_chunk in (0..nchunks).rev() {
        // ── Per-chunk inputs (slice the batched tensors to [B,H,...]) ──────
        let da_cumsum_bhl: Tensor<B, 3> = da_cumsum_bhnl
            .clone()
            .slice(s![.., .., i_chunk, ..])
            .squeeze_dim::<3>(2); // [B,H,L]
        let dt_bhl: Tensor<B, 3> = dt_discretized_bhnl
            .clone()
            .slice(s![.., .., i_chunk, ..])
            .squeeze_dim::<3>(2); // [B,H,L]
        let x_bhlp: Tensor<B, 4> = x_bnlhp
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1) // [B,L,H,P]
            .permute([0, 2, 1, 3]); // [B,H,L,P]
        let d_y_bhlp: Tensor<B, 4> = d_y_bnlhp
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1) // [B,L,H,P]
            .permute([0, 2, 1, 3]); // [B,H,L,P]

        // GQA-expand C for this chunk: [B,L,G,R] → [B,H,L,R]
        let c_bhlr: Tensor<B, 4> = c_bnlgr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1) // [B,L,G,R]
            .unsqueeze_dim::<5>(3) // [B,L,G,1,R]
            .expand([batch, chunk_len, ngroups, heads_per_group, state_rank])
            .reshape([batch, chunk_len, nheads, state_rank]) // [B,L,H,R]
            .permute([0, 2, 1, 3]); // [B,H,L,R]

        // GQA-expand CB for this chunk: [B,G,L,L] → [B,H,L,L]
        let cb_bhll: Tensor<B, 4> = cb_bngll
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1) // [B,G,L,L]
            .unsqueeze_dim::<5>(2) // [B,G,1,L,L]
            .expand([batch, ngroups, heads_per_group, chunk_len, chunk_len])
            .reshape([batch, nheads, chunk_len, chunk_len]); // [B,H,L,L]

        let causal_mask_bhll: Tensor<B, 4, burn::prelude::Bool> = causal_mask_ll
            .clone()
            .reshape([1, 1, chunk_len, chunk_len])
            .expand([batch, nheads, chunk_len, chunk_len]);

        // Running state entering chunk i.
        let chunk_input_state_bhpr: Tensor<B, 4> = chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1); // [B,H,P,R]
        san(&chunk_input_state_bhpr);

        // ── BLUE backward for chunk i ──────────────────────────────────────
        // y_blue[l,p] = exp(da[l]) · Σ_r C[l,r] · state[p,r]
        let exp_da_cumsum_bhl: Tensor<B, 3> = da_cumsum_bhl.clone().exp(); // [B,H,L]
        let exp_da_cumsum_bhlp: Tensor<B, 4> = exp_da_cumsum_bhl
            .clone()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, chunk_len, per_head_dim]); // [B,H,L,P]
        let d_blue_scaled_bhlp: Tensor<B, 4> = d_y_bhlp.clone(); // = d_y_partial
        let d_blue_bhlp: Tensor<B, 4> = d_blue_scaled_bhlp.clone() * exp_da_cumsum_bhlp.clone();
        san(&d_blue_bhlp);

        // d_chunk_input_state = c^T @ d_blue
        //   c_bhlr [B,H,L,R].permute([0,1,3,2]) → c_bhrl [B,H,R,L]
        //   c_bhrl @ d_blue_bhlp [B,H,L,P] → [B,H,R,P] → permute → [B,H,P,R]
        let d_chunk_input_state_bhpr: Tensor<B, 4> = c_bhlr
            .clone()
            .permute([0, 1, 3, 2]) // [B,H,R,L]
            .matmul(d_blue_bhlp.clone()) // [B,H,R,P]
            .permute([0, 1, 3, 2]); // [B,H,P,R]
        san(&d_chunk_input_state_bhpr);

        // d_c_blue = d_blue @ chunk_input_state  [B,H,L,P] @ [B,H,P,R] → [B,H,L,R]
        let d_c_blue_bhlr: Tensor<B, 4> =
            d_blue_bhlp.clone().matmul(chunk_input_state_bhpr.clone());
        san(&d_c_blue_bhlr);
        d_c_blue_vec.push(d_c_blue_bhlr);

        // d_da from BLUE:
        //   blue_no_scale = c @ state^T  [B,H,L,R] @ [B,H,R,P] → [B,H,L,P]
        //   d_da[l] = Σ_p (d_blue_scaled[l,p] * blue_no_scale[l,p]) * exp_da[l]
        let blue_bhlp: Tensor<B, 4> = c_bhlr
            .clone()
            .matmul(chunk_input_state_bhpr.clone().permute([0, 1, 3, 2])); // [B,H,L,P]
        let d_exp_da_cumsum_bhlp: Tensor<B, 4> = d_blue_scaled_bhlp.clone() * blue_bhlp;
        let d_da_blue_bhl: Tensor<B, 3> = (d_exp_da_cumsum_bhlp * exp_da_cumsum_bhlp)
            .sum_dim(3)
            .squeeze_dim::<3>(3); // [B,H,L]
        san(&d_da_blue_bhl);
        d_da_blue_vec.push(d_da_blue_bhl);

        // ── ORANGE backward for chunk i ────────────────────────────────────
        // y_orange[l,p] = Σ_{s≤l} CB[l,s] · exp(da[l]-da[s]) · dt[s] · x[s,p]
        //   CB_w[l,s] = CB[l,s] · exp_diff[l,s] · dt[s] · mask
        let da_cumsum_target_bhll: Tensor<B, 4> = da_cumsum_bhl
            .clone()
            .unsqueeze_dim::<4>(3) // [B,H,L_tgt,1]
            .expand([batch, nheads, chunk_len, chunk_len]); // [B,H,L_tgt,L_src]
        let da_cumsum_source_bhll: Tensor<B, 4> = da_cumsum_bhl
            .unsqueeze_dim::<4>(2) // [B,H,1,L_src]
            .expand([batch, nheads, chunk_len, chunk_len]);
        let da_cumsum_diff_bhll = da_cumsum_target_bhll - da_cumsum_source_bhll;
        san(&da_cumsum_diff_bhll);

        // Causal mask + exp stabiliser (-inf above the main diagonal).
        let da_cumsum_diff_masked_bhll =
            da_cumsum_diff_bhll.mask_fill(causal_mask_bhll.clone(), f32::NEG_INFINITY);
        let da_cumsum_diff_exp_bhll = da_cumsum_diff_masked_bhll.exp(); // [B,H,L,L]
        san(&da_cumsum_diff_exp_bhll);

        let dt_source_bhll: Tensor<B, 4> = dt_bhl
            .unsqueeze_dim::<4>(2) // [B,H,1,L_src]
            .expand([batch, nheads, chunk_len, chunk_len]);

        let orange_lhs_partial1_bhll: Tensor<B, 4> =
            cb_bhll.clone() * da_cumsum_diff_exp_bhll.clone();
        san(&orange_lhs_partial1_bhll);
        let orange_lhs_partial2_bhll: Tensor<B, 4> =
            orange_lhs_partial1_bhll.clone() * dt_source_bhll.clone();
        san(&orange_lhs_partial2_bhll);

        let d_orange_bhlp: Tensor<B, 4> = d_y_bhlp; // = d_y_partial
        // d_CB_w = d_orange @ x^T  [B,H,L_tgt,P] @ [B,H,P,L_src] → [B,H,L_tgt,L_src]
        let d_orange_lhs_partial2_bhll: Tensor<B, 4> = d_orange_bhlp
            .clone()
            .matmul(x_bhlp.permute([0, 1, 3, 2]));
        san(&d_orange_lhs_partial2_bhll);

        // d_x = CB_w^T @ d_orange  [B,H,L_src,L_tgt] @ [B,H,L_tgt,P] → [B,H,L_src,P]
        let d_x_orange_bhlp: Tensor<B, 4> = orange_lhs_partial2_bhll
            .permute([0, 1, 3, 2])
            .matmul(d_orange_bhlp);
        san(&d_x_orange_bhlp);
        d_x_orange_vec.push(d_x_orange_bhlp);

        // Mask off above-diagonal (set to 0 above the main diagonal).
        let d_orange_lhs_partial2_bhll =
            d_orange_lhs_partial2_bhll.mask_fill(causal_mask_bhll, 0.);
        san(&d_orange_lhs_partial2_bhll);
        let d_orange_lhs_partial1_bhll =
            d_orange_lhs_partial2_bhll.clone() * dt_source_bhll;
        san(&d_orange_lhs_partial1_bhll);

        // d_dt[s] = Σ_{l≥s} d_CB_w[l,s] · CB[l,s] · decay[l,s]
        let d_dt_orange_bhl: Tensor<B, 3> = (d_orange_lhs_partial2_bhll
            * orange_lhs_partial1_bhll.clone())
        .sum_dim(2)
        .squeeze_dim::<3>(2); // [B,H,L]
        san(&d_dt_orange_bhl);
        d_dt_orange_vec.push(d_dt_orange_bhl);

        // d_da from ORANGE: decay = exp(da_tgt - da_src)
        //   d_da_tgt[l] += Σ_s (d_decay · decay)[l,s]
        //   d_da_src[s] -= Σ_l (d_decay · decay)[l,s]
        let d_da_cumsum_diff_exp_bhll = d_orange_lhs_partial1_bhll.clone() * cb_bhll;
        let d_da_cumsum_diff_bhll =
            d_da_cumsum_diff_exp_bhll * da_cumsum_diff_exp_bhll.clone();
        let d_da_tgt_bhl: Tensor<B, 3> = d_da_cumsum_diff_bhll
            .clone()
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        let d_da_src_bhl: Tensor<B, 3> =
            d_da_cumsum_diff_bhll.sum_dim(2).squeeze_dim::<3>(2);
        let d_da_orange_bhl: Tensor<B, 3> = d_da_tgt_bhl - d_da_src_bhl;
        san(&d_da_orange_bhl);
        d_da_orange_vec.push(d_da_orange_bhl);

        // d_cb (GQA reduce H → G): [B,H,L,L] → [B,G,L,L]
        let d_cb_bhll = d_orange_lhs_partial1_bhll * da_cumsum_diff_exp_bhll;
        let d_cb_bgll: Tensor<B, 4> = d_cb_bhll
            .reshape([batch, ngroups, heads_per_group, chunk_len, chunk_len])
            .sum_dim(2)
            .squeeze_dim::<4>(2); // [B,G,L,L]
        san(&d_cb_bgll);
        d_cb_vec.push(d_cb_bgll);

        // ── K4 backward step for chunk i ───────────────────────────────────
        // d_intra[i] = current d_running_state (before the propagation step).
        d_intra_slices.push(d_running_state_bhpr.clone());

        // Recompute decay for this chunk.
        let decay_bhpr: Tensor<B, 4> = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk]) // [B,H,1]
            .exp()
            .unsqueeze_dim::<4>(3) // [B,H,1,1]
            .expand([batch, nheads, per_head_dim, state_rank]); // [B,H,P,R]
        san(&decay_bhpr);

        // d_decay = d_running_state · running_state (running_state = chunk_input_state[i]).
        let d_decay_bhpr = d_running_state_bhpr.clone() * chunk_input_state_bhpr;
        san(&d_decay_bhpr);

        // d_da_chunk_end[b,h] = Σ_{p,r} d_decay · decay   (decay = exp(da_chunk_end)).
        let d_da_chunk_end_bh: Tensor<B, 2> = (d_decay_bhpr * decay_bhpr.clone())
            .reshape([batch, nheads, per_head_dim * state_rank])
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        san(&d_da_chunk_end_bh);
        d_da_end_bh_slices.push(d_da_chunk_end_bh);

        // Propagate to previous chunk:
        //   d_running_state_prev = decay · d_running_state + d_chunk_input_state.
        d_running_state_bhpr =
            decay_bhpr * d_running_state_bhpr + d_chunk_input_state_bhpr;
        san(&d_running_state_bhpr);
    }
    // d_initial_state = the trailing d_running_state after the reverse loop.
    let d_initial_state_bhpr = d_running_state_bhpr;

    // ── Stack per-chunk gradients into the batched tensors K3/K2/K1 expect ──
    // Restore natural (forward) chunk order — vecs were filled in reverse.
    d_x_orange_vec.reverse();
    d_dt_orange_vec.reverse();
    d_da_orange_vec.reverse();
    d_cb_vec.reverse();
    d_c_blue_vec.reverse();
    d_da_blue_vec.reverse();
    d_intra_slices.reverse();
    d_da_end_bh_slices.reverse();

    // d_x_orange: [B,H,L,P] → stack@1 → [B,N,H,L,P] → permute → [B,N,L,H,P]
    let d_x_orange_bnlhp: Tensor<B, 5> =
        Tensor::stack::<5>(d_x_orange_vec, 1).permute([0, 1, 3, 2, 4]);
    san(&d_x_orange_bnlhp);

    // d_dt_orange: [B,H,L] → stack@2 → [B,H,N,L]
    let d_dt_orange_bhnl: Tensor<B, 4> = Tensor::stack(d_dt_orange_vec, 2);
    san(&d_dt_orange_bhnl);

    // d_da_orange: [B,H,L] → stack@2 → [B,H,N,L]
    let d_da_orange_bhnl: Tensor<B, 4> = Tensor::stack(d_da_orange_vec, 2);
    san(&d_da_orange_bhnl);

    // d_cb: [B,G,L,L] → stack@1 → [B,N,G,L,L]
    let d_cb_bngll: Tensor<B, 5> = Tensor::stack(d_cb_vec, 1);
    san(&d_cb_bngll);

    // d_c_blue: [B,H,L,R] → stack@1 → [B,N,H,L,R] → GQA reduce → [B,N,L,G,R]
    let d_c_blue_bnhlr: Tensor<B, 5> = Tensor::stack(d_c_blue_vec, 1);
    let d_c_blue_bnlgr: Tensor<B, 5> = d_c_blue_bnhlr
        .reshape([batch, nchunks, ngroups, heads_per_group, chunk_len, state_rank])
        .sum_dim(3)
        .squeeze_dim::<5>(3)
        .permute([0, 1, 3, 2, 4]);
    san(&d_c_blue_bnlgr);

    // d_da_blue: [B,H,L] → stack@2 → [B,H,N,L]
    let d_da_blue_bhnl: Tensor<B, 4> = Tensor::stack(d_da_blue_vec, 2);
    san(&d_da_blue_bhnl);

    // d_intra_chunk_state: [B,H,P,R] → stack@1 → [B,N,H,P,R]
    let d_intra_chunk_state_bnhpr: Tensor<B, 5> = Tensor::stack(d_intra_slices, 1);
    san(&d_intra_chunk_state_bnhpr);

    // d_da_end: [B,H] → stack@2 → [B,H,N]; scatter to last L position of d_da_cumsum_k4.
    let d_da_end_bhn: Tensor<B, 3> = Tensor::stack(d_da_end_bh_slices, 2);
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

    // Test-only: local same-chunk d_da contribution (BLUE + ORANGE) snapshot
    // for the `out_x · dout − ddt · dt` oracle. Production builds skip the
    // extra add and the retained [B,H,N,L] tensor.
    #[cfg(test)]
    let d_da_local_bhnl = d_da_blue_bhnl.clone() + d_da_orange_bhnl.clone();
    #[cfg(test)]
    san(&d_da_local_bhnl);
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

    // Test-only: keep d_dt_orange_bhnl alive for the oracle return.
    #[cfg(test)]
    let d_dt_orange_bhnl_save = d_dt_orange_bhnl.clone();
    let d_dt_discretized_bhnl =
        d_dt_orange_bhnl + d_dt_discretized_k3_bhnl + d_dt_k1_bhnl;
    san(&d_dt_discretized_bhnl);

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
        #[cfg(test)]
        d_da_local_bhnl,
        #[cfg(test)]
        d_dt_orange_bhnl: d_dt_orange_bhnl_save,
    }
}

#[cfg(test)]
#[cfg(feature = "backend-flex")]
mod tests {
    use super::*;
    use crate::mamba2::ssd::serial;
    use burn::backend::Flex;
    use burn::tensor::Distribution;

    type B = Flex;

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
        let ngroups = 2;
        let state_rank = 6;

        // ─── Random inputs (small, gentle distributions) ─────────────────
        let x_bnlhp = Tensor::<B, 5>::random(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let dt_bnlh = Tensor::<B, 4>::random(
            [batch, nchunks, chunk_len, nheads],
            Distribution::Uniform(0.05, 0.3),
            &device,
        );
        let a_decay_h =
            Tensor::<B, 1>::random([nheads], Distribution::Uniform(-1.0, -0.5), &device);
        let b_bnlgr = Tensor::<B, 5>::random(
            [batch, nchunks, chunk_len, ngroups, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let c_bnlgr = Tensor::<B, 5>::random(
            [batch, nchunks, chunk_len, ngroups, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let d_h = Tensor::<B, 1>::random([nheads], Distribution::Normal(0.0, 0.1), &device);
        let initial_state_bhpr = Tensor::<B, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 0.1),
            &device,
        );
        let dt_discretized_bhnl = dt_bnlh.permute([0, 3, 1, 2]); // [B,H,N,L]

        // ─── Forward (Serial path) ───────────────────────────────────────
        let (da_cumsum_bhnl, da_chunk_end_bhn) =
            serial::k1_ssd_chunk_cumsum(dt_discretized_bhnl.clone(), a_decay_h.clone());
        let cb_bngll = serial::k2_ssd_bmm(c_bnlgr.clone(), b_bnlgr.clone());
        let intra_chunk_state_bnhpr = serial::k3_ssd_chunk_state(
            x_bnlhp.clone(),
            b_bnlgr.clone(),
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
            c_bnlgr.clone(),
            cb_bngll,
            chunk_input_state_bnhpr,
            d_h.clone(),
        );

        // ─── out_x = y − D·x  (output before D-skip) ─────────────────────
        let skip_bnlhp = d_h
            .clone()
            .unsqueeze_dims::<5>(&[0, 1, 2, 4]) // [1,1,1,H,1]
            .expand([batch, nchunks, chunk_len, nheads, per_head_dim])
            * x_bnlhp.clone();
        let out_x_bnlhp = y_bnlhp - skip_bnlhp;

        // ─── Random upstream gradients ────────────────────────────────────
        let d_y_bnlhp = Tensor::<B, 5>::random(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let d_final_bhpr = Tensor::<B, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // ─── Run combined_backward (gets d_da_local + d_dt_orange) ───────
        let grads = combined_backward(
            d_y_bnlhp.clone(),
            d_final_bhpr,
            x_bnlhp,
            dt_discretized_bhnl.clone(),
            b_bnlgr,
            c_bnlgr,
            d_h,
            initial_state_bhpr,
            a_decay_h,
        );

        // ─── Oracle: einsum(out_x, d_y, "bnlhp,bnlhp->bhnl") − d_dt_orange·dt
        let einsum_bhnl: Tensor<B, 4> = (out_x_bnlhp * d_y_bnlhp)
            .sum_dim(4)                 // [B,N,L,H,1]
            .squeeze_dim::<4>(4)        // [B,N,L,H]
            .permute([0, 3, 1, 2]);     // [B,H,N,L]
        let oracle_bhnl = einsum_bhnl - grads.d_dt_orange_bhnl * dt_discretized_bhnl;

        // ─── Compare ─────────────────────────────────────────────────────
        let diff: f32 = (grads.d_da_local_bhnl - oracle_bhnl)
            .abs()
            .max()
            .into_scalar()
            .elem();
        assert!(
            diff < 1e-3,
            "d_da_local oracle identity violated; max abs diff = {diff}",
        );
    }
}
