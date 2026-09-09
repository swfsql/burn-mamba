//! # Recompute-based gradient math for the Mamba-3 double-SSD
//!
//! The analytic backward of the MIMO-first serial scan used by each pass of the
//! double-SSD decomposition.  The forward intermediates (K1–K4) are recomputed
//! from the saved leaf inputs rather than stashed, then the chunk-**local** K5
//! backward runs batched, a chunk group at a time
//! ([`Mamba3SsdPath::backward_chunk_group`](crate::mamba3::ssd_path::Mamba3SsdPath::backward_chunk_group)); only the K4 state-passing
//! backward, the one recurrence, is a walk.  K1/K2/K3 backwards run batched
//! after it.  The fused `L·M` length carries the `mimo_rank` axis through the
//! intra-chunk products.
//!
//! Everything operates on backend **primitives** through the rank-tagged `F`
//! wrapper: the custom [`Backward`](burn::backend::autodiff::ops::Backward) node
//! runs with a generic backend `B`, so the high-level `Tensor` is unavailable
//! and the math uses `B`'s `float_*` ops.  The recomputed K1/K2/K4 kernels are
//! local primitive ports of the high-level [`super::super::serial`](crate::mamba3::double_ssd::ssd::serial) kernels.

#![allow(non_snake_case)]

use super::serial_recalculated::{
    cat_chunk_groups, k1_ssd_chunk_cumsum, k2_ssd_bmm, k4_ssd_state_passing,
    k4_ssd_state_passing_backward,
};
use crate::mamba3::helpers::prim::{read_causal_mask, read_rows, scatter_read_rows};
use crate::mamba3::ssd_path::Mamba3SsdPath;
use burn_stack::utils::fprim::{F, san};
use burn::backend::Backend;
use burn::tensor::s;

/// Per-input gradients produced by [`combined_backward`] (one field per
/// differentiable forward input of the double-SSD scan).
#[non_exhaustive]
pub struct CombinedGrads<B: Backend> {
    /// Gradient of the (pre-scaled) input `v`.
    pub d_v_bnlmhp: F<B, 6>,
    /// Gradient of `Δ·A` (`da`).
    pub d_da_bnlh: F<B, 4>,
    /// Gradient of the input projection `B`.
    pub d_b_bnlmhr: F<B, 6>,
    /// Gradient of the output projection `C`, on the chunk's read axis.
    pub d_c_bntmhr: F<B, 6>,
    /// Gradient of the initial SSM state.
    pub d_initial_state_bhpr: F<B, 4>,
}

// ─── Recomputed forward kernels ──────────────────────────────────────────────
// The recompute backward replays the forward's K1/K2/K4 (imported above from
// [`super::serial_recalculated`]) plus the extended K3 below, which returns the
// extra intermediates the gradient math needs.

/// Same as `k3_ssd_chunk_state` but
/// also returns intermediates needed by the custom backward:
/// - `intra_chunk_state_bnhpr` — the chunk-end state assuming zero initial state
/// - `decay_bhnLM` — the fused-length K3 decay factor `exp(cumA_last − cumA_fused)`
/// - `decayed_v_bnLMhp` — V already scaled by `decay_bnLMh1`
pub fn k3_ssd_chunk_state_extended<B: Backend>(
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    da_cumsum_bhnl: F<B, 4>,
) -> (F<B, 5>, F<B, 4>, F<B, 5>) {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();

    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

    let da_cumsum_last_bhn1 = da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4) // da_cumsum_bhnl1
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // da_cumsum_bhnlm
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // da_cumsum_bhnLM
    let decay_bhnLM = (da_cumsum_last_bhn1 - da_cumsum_bhnLM).exp();
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

/// Memory-efficient backward for the Mamba-3 MIMO-first chunkwise SSD.
///
/// Recomputes the forward intermediates (K1-K4) from the saved inputs, then
/// runs a reverse per-chunk loop that fuses the K5 (BLUE + ORANGE) backward
/// with the K4 state-passing backward.  K3/K2/K1 backwards run as single
/// batched ops once the loop has collected all per-chunk slices.
///
/// # Arguments
/// - `d_y_bnlmhp` — upstream gradient of the SSD output
/// - `d_final_bhpr` — upstream gradient of the final SSM state
/// - `v_bnlmhp`, `da_bnlh`, `b_bnlmhr`, `c_bnlmhr`, `initial_state_bhpr` —
///   the five saved forward inputs
///
/// # Returns
/// One [`CombinedGrads`] struct containing gradients for all 5 inputs.
pub fn combined_backward<B: Backend>(
    d_y_bntmhp: F<B, 6>,
    d_final_bhpr: F<B, 4>,
    //
    v_bnlmhp: F<B, 6>,
    da_bnlh: F<B, 4>,
    b_bnlmhr: F<B, 6>,
    c_bntmhr: F<B, 6>,
    initial_state_bhpr: F<B, 4>,
    read_stride: usize,
) -> CombinedGrads<B> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();
    let device = v_bnlmhp.device();
    let dtype = v_bnlmhp.dtype();
    // The chunk's two axes: `fused` is what it writes, `read` what it is read at.
    let chunk_tokens = Mamba3SsdPath::chunk_tokens(chunk_len, read_stride);
    let fused = chunk_len * mimo_rank;
    let read = chunk_tokens * mimo_rank;

    san(&d_y_bntmhp);
    san(&d_final_bhpr);
    san(&v_bnlmhp);
    san(&da_bnlh);
    san(&b_bnlmhr);
    san(&c_bntmhr);
    san(&initial_state_bhpr);

    // ═══════════════════════════════════════════════════════════════════════
    // RECOMPUTE FORWARD INTERMEDIATES
    // ═══════════════════════════════════════════════════════════════════════

    // K1 — pre-combined Δ·A → intra-chunk cumsum
    let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum(da_bnlh.clone());
    san(&da_cumsum_bhnl);

    // K2 — CB matrix used in K5 ORANGE
    let cb_bnhTMLM = k2_ssd_bmm(c_bntmhr.clone(), b_bnlmhr.clone());
    san(&cb_bnhTMLM);

    // K3 — intra-chunk state + decay/decayed-V intermediates
    let (intra_chunk_state_bnhpr, k3_decay_bhnLM, k3_decayed_v_bnLMhp) =
        k3_ssd_chunk_state_extended(v_bnlmhp.clone(), b_bnlmhr.clone(), da_cumsum_bhnl.clone());

    // K4 — chunk-input state stream consumed by K5 BLUE
    let (chunk_input_state_bnhpr, _final_state_bhpr) = k4_ssd_state_passing(
        intra_chunk_state_bnhpr,
        da_chunk_end_bhn.clone(),
        initial_state_bhpr,
    );

    // ═══════════════════════════════════════════════════════════════════════
    // FUSED-L INTERMEDIATES USED BY THE REVERSE LOOP
    // ═══════════════════════════════════════════════════════════════════════
    //
    // da_cumsum_bhnLM: cumA per fused position. The expand-then-reshape
    // repeats each base position mimo_rank times along the fused dim, matching K5.
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .clone()
        .unsqueeze_dim::<5>(4) // da_cumsum_bhnl1
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // da_cumsum_bhnlm
        .reshape([batch, nheads, nchunks, fused]); // da_cumsum_bhnLM
    let da_cumsum_bhnTM = read_rows::<B, 4, 5>(da_cumsum_bhnl.clone(), 3, read_stride)
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_tokens, mimo_rank])
        .reshape([batch, nheads, nchunks, read]);

    // d_y in (batch, nchunks, nheads, chunk_tokens * mimo_rank, per_head_dim)
    // ordering — matches the per-chunk slicing.
    let d_y_bnhTMp = d_y_bntmhp
        .reshape([batch, nchunks, read, nheads, per_head_dim]) // d_y_bnTMhp
        .swap_dims(2, 3); // d_y_bnhTMp
    san(&d_y_bnhTMp);

    // Reusable [chunk_tokens, chunk_len] -inf upper-triangular base mask for
    // ORANGE (`triu(1)`: the double-SSD sum includes the same step).
    let neg_inf_base_tl: F<B, 2> = read_causal_mask::<B>(chunk_len, read_stride, 1, &device, dtype);

    // ═══════════════════════════════════════════════════════════════════════
    // CHUNK-LOCAL BACKWARD — K5 (BLUE + ORANGE), batched over a chunk group
    //
    // Neither term is recurrent: both need chunk `i`'s own slices plus
    // `chunk_input_state[i]`, which the recomputed K4 already produced batched.
    // So this side mirrors the forward's own batched K5, `group` chunks at a
    // time — the group being what prices the score `[b, group, h, TM, LM]`
    // against that forward's peak (see [`Mamba3SsdPath::backward_chunk_group`]).
    // The state gradient, the one part that *is* a scan, follows below.
    // ═══════════════════════════════════════════════════════════════════════
    let group = Mamba3SsdPath::backward_chunk_group(nchunks);
    let ngroups = nchunks.div_ceil(group);
    let mut vec_orange_d_v_bnhLMp: Vec<F<B, 5>> = Vec::with_capacity(ngroups);
    let mut vec_blue_d_c_bnhTMr: Vec<F<B, 5>> = Vec::with_capacity(ngroups);
    let mut vec_d_cb_bnhTMLM: Vec<F<B, 5>> = Vec::with_capacity(ngroups);
    let mut vec_blue_d_da_bnhl: Vec<F<B, 4>> = Vec::with_capacity(ngroups);
    let mut vec_orange_d_da_bnhl: Vec<F<B, 4>> = Vec::with_capacity(ngroups);
    let mut vec_d_chunk_input_state_bnhpr: Vec<F<B, 5>> = Vec::with_capacity(ngroups);

    for start in (0..nchunks).step_by(group) {
        // The group's own chunk count — `group`, short at the tail. It is the
        // same `n` axis throughout, just narrowed, so the names do not change.
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

        let chunk_input_state_bnhpr: F<B, 5> =
            chunk_input_state_bnhpr.clone().narrow(1, start, n);
        san(&chunk_input_state_bnhpr);

        let d_y_bnhTMp: F<B, 5> = d_y_bnhTMp.clone().narrow(1, start, n);

        // ── BLUE backward ──────────────────────────────────────────────
        //
        //   blue[TM,p] = exp(cumA[TM]) · Σᵣ C[TM,r] · state[p,r]
        //
        // exp_da depends on the read position TM only — broadcast over per_head_dim.
        let exp_da_cumsum_bnhTMp: F<B, 5> = da_cumsum_bnhTM
            .clone()
            .exp()
            .unsqueeze_dim::<5>(4) // exp_da_cumsum_bnhTM1
            .expand([batch, n, nheads, read, per_head_dim]); // exp_da_cumsum_bnhTMp
        let d_ch_bnhTMp: F<B, 5> = d_y_bnhTMp.clone() * exp_da_cumsum_bnhTMp.clone();
        san(&d_ch_bnhTMp);

        // d_chunk_input_state[p,r] = Σ_TM C[TM,r] · d_ch[TM,p]
        //   C^T (bnhrTM) @ d_ch (bnhTMp)  → bnhrp  → transpose → bnhpr
        let d_chunk_input_state_bnhpr: F<B, 5> = c_bnhTMr
            .clone()
            .transpose() // c_bnhrTM
            .matmul(d_ch_bnhTMp.clone()) // d_chunk_input_state_bnhrp
            .transpose(); // d_chunk_input_state_bnhpr
        san(&d_chunk_input_state_bnhpr);
        vec_d_chunk_input_state_bnhpr.push(d_chunk_input_state_bnhpr);

        // d_C_blue[TM,r] = Σₚ d_ch[TM,p] · state[p,r]
        //   d_ch (bnhTMp) @ state (bnhpr)  → bnhTMr
        let d_c_blue_bnhTMr: F<B, 5> = d_ch_bnhTMp.matmul(chunk_input_state_bnhpr.clone());
        san(&d_c_blue_bnhTMr);
        vec_blue_d_c_bnhTMr.push(d_c_blue_bnhTMr);

        // d_da from BLUE:
        //   ch[TM,p] = Σᵣ C[TM,r] · state[p,r]      (= C @ state_rp after transpose)
        //   d_da[TM] = (Σₚ d_y[TM,p] · ch[TM,p]) · exp_da[TM]
        let ch_bnhTMp: F<B, 5> = c_bnhTMr.clone().matmul(
            chunk_input_state_bnhpr.transpose(), // chunk_input_state_bnhrp
        ); // ch_bnhTMp
        let d_da_blue_bnhTM: F<B, 4> = (d_y_bnhTMp.clone() * ch_bnhTMp * exp_da_cumsum_bnhTMp)
            .sum_dim(4) // d_da_blue_bnhTM1
            .squeeze_dim::<4>(4); // d_da_blue_bnhTM
        san(&d_da_blue_bnhTM);

        // Reduce fused TM → t (sum the mimo_rank copies K5 broadcast), then
        // scatter back onto the folded axis: a read row's `da` is one position.
        let d_da_blue_bnhl: F<B, 4> = scatter_read_rows::<B, 4, 5>(
            d_da_blue_bnhTM
                .reshape([batch, n, nheads, chunk_tokens, mimo_rank]) // d_da_blue_bnhtm
                .sum_dim(4) // d_da_blue_bnht1
                .squeeze_dim::<4>(4), // d_da_blue_bnht
            3,
            read_stride,
        );
        vec_blue_d_da_bnhl.push(d_da_blue_bnhl);

        // ── ORANGE backward ────────────────────────────────────────────
        //
        //   w[LMₜ,LMₛ] = CB[LMₜ,LMₛ] · decay[LMₜ,LMₛ]   (MIMO causal mask in decay)
        //   orange[LMₜ,p] = Σ_{LMₛ} w[LMₜ,LMₛ] · v[LMₛ,p]
        let da_target_bnhTMLM: F<B, 5> = da_cumsum_bnhTM
            .unsqueeze_dim::<5>(4) // da_cumsum_bnhTMₜ1
            .expand([batch, n, nheads, read, fused]); // da_target_bnhTMₜLM
        let da_source_bnhTMLM: F<B, 5> = da_cumsum_bnhLM
            .unsqueeze_dim::<5>(3) // da_cumsum_bnh1LMₛ
            .expand([batch, n, nheads, read, fused]); // da_source_bnhTMLMₛ
        let diff_bnhTMLM = da_target_bnhTMLM - da_source_bnhTMLM;
        san(&diff_bnhTMLM);

        // MIMO causal mask: -inf where LMₛ//mimo_rank > row(TMₜ)//mimo_rank —
        // interleaved expansion of the [t, l] upper-triangular base mask
        // (matches K5).
        let neg_inf_mimo_bnhTMLM: F<B, 5> = neg_inf_base_tl
            .clone()
            .unsqueeze_dims::<5>(&[0, 1, 2]) // neg_inf_base_111tl
            .expand([batch, n, nheads, chunk_tokens, chunk_len]) // neg_inf_base_bnhtl
            .unsqueeze_dim::<6>(4) // neg_inf_base_bnht1l
            .expand([batch, n, nheads, chunk_tokens, mimo_rank, chunk_len]) // neg_inf_base_bnhtml
            .reshape([batch, n, nheads, read, chunk_len]) // neg_inf_base_bnhTMl
            .unsqueeze_dim::<6>(5) // neg_inf_base_bnhTMl1
            .expand([batch, n, nheads, read, chunk_len, mimo_rank]) // neg_inf_base_bnhTMlm
            .reshape([batch, n, nheads, read, fused]); // neg_inf_mimo_bnhTMLM
        let decay_bnhTMLM = (diff_bnhTMLM + neg_inf_mimo_bnhTMLM).exp();
        san(&decay_bnhTMLM);

        // d_v_orange = w^T @ d_orange ; d_w = d_orange @ v^T
        let d_orange_bnhTMp = d_y_bnhTMp;
        let w_bnhTMLM = cb_bnhTMLM.clone() * decay_bnhTMLM.clone();
        let d_w_bnhTMLM: F<B, 5> = d_orange_bnhTMp.clone().matmul(
            v_bnhLMp.transpose(), // v_bnhpLM
        ); // d_w_bnhTMₜLMₛ
        san(&d_w_bnhTMLM);
        let d_v_orange_bnhLMp: F<B, 5> = w_bnhTMLM
            .transpose() // w_bnhLMₛTMₜ
            .matmul(d_orange_bnhTMp); // d_v_orange_bnhLMₛp
        san(&d_v_orange_bnhLMp);
        vec_orange_d_v_bnhLMp.push(d_v_orange_bnhLMp);

        // d_cb = d_w · decay ; d_decay = d_w · cb ; d_diff = d_decay · decay
        // (masked positions where decay=0 contribute 0 to d_diff automatically)
        let d_cb_bnhTMLM = d_w_bnhTMLM.clone() * decay_bnhTMLM.clone();
        vec_d_cb_bnhTMLM.push(d_cb_bnhTMLM);

        let d_decay_bnhTMLM = d_w_bnhTMLM * cb_bnhTMLM;
        let d_diff_bnhTMLM = d_decay_bnhTMLM * decay_bnhTMLM;

        // d_da_target[TMₜ] = Σ_{LMₛ} d_diff[TMₜ, LMₛ] (scattered onto the folded
        // axis); d_da_source[LMₛ] = Σ_{TMₜ} d_diff[TMₜ, LMₛ] (already folded);
        // d_da_orange = d_da_target − d_da_source  (diff = target − source).
        let d_da_target_bnhTM: F<B, 4> = d_diff_bnhTMLM
            .clone()
            .sum_dim(4) // d_diff_bnhTMₜ1
            .squeeze_dim::<4>(4); // d_da_target_bnhTMₜ
        let d_da_source_bnhLM: F<B, 4> = d_diff_bnhTMLM
            .sum_dim(3) // d_diff_bnh1LMₛ
            .squeeze_dim::<4>(3); // d_da_source_bnhLMₛ

        let d_da_target_bnhl: F<B, 4> = scatter_read_rows::<B, 4, 5>(
            d_da_target_bnhTM
                .reshape([batch, n, nheads, chunk_tokens, mimo_rank])
                .sum_dim(4)
                .squeeze_dim::<4>(4),
            3,
            read_stride,
        );
        let d_da_source_bnhl: F<B, 4> = d_da_source_bnhLM
            .reshape([batch, n, nheads, chunk_len, mimo_rank]) // d_da_source_bnhlm
            .sum_dim(4) // d_da_source_bnhl1
            .squeeze_dim::<4>(4); // d_da_source_bnhl
        let d_da_orange_bnhl = d_da_target_bnhl - d_da_source_bnhl;
        san(&d_da_orange_bnhl);
        vec_orange_d_da_bnhl.push(d_da_orange_bnhl);
    }

    // ── Rejoin the groups along the chunk axis ────────────────────────────
    let d_v_orange_bnhLMp: F<B, 5> = cat_chunk_groups(vec_orange_d_v_bnhLMp, 1);
    let d_c_blue_bnhTMr: F<B, 5> = cat_chunk_groups(vec_blue_d_c_bnhTMr, 1);
    let d_cb_bnhTMLM: F<B, 5> = cat_chunk_groups(vec_d_cb_bnhTMLM, 1);
    let d_da_blue_bhnl: F<B, 4> = cat_chunk_groups(vec_blue_d_da_bnhl, 1).swap_dims(1, 2);
    let d_da_orange_bhnl: F<B, 4> = cat_chunk_groups(vec_orange_d_da_bnhl, 1).swap_dims(1, 2);
    let d_chunk_input_state_bnhpr: F<B, 5> = cat_chunk_groups(vec_d_chunk_input_state_bnhpr, 1);

    // ── K4 backward — the reverse of the state-passing scan ───────────────
    let (d_intra_chunk_state_bnhpr, d_da_end_bhn, d_initial_state_bhpr) =
        k4_ssd_state_passing_backward(
            d_chunk_input_state_bnhpr,
            chunk_input_state_bnhpr,
            da_chunk_end_bhn,
            d_final_bhpr,
        );
    // d_da_end: [batch,nheads,nchunks] scattered into the last `l` of d_da_cumsum_k4.
    let d_da_cumsum_k4_bhnl: F<B, 4> = {
        let zeros = F::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device, dtype);
        let d_da_end_bhn1 = d_da_end_bhn.unsqueeze_dim::<4>(3);
        F::cat(vec![zeros, d_da_end_bhn1], 3)
    };

    // ═══════════════════════════════════════════════════════════════════════
    // K3 BACKWARD (batched)
    //
    // Forward (recap):
    //   v_bnLMhp        = v.reshape
    //   b_bnLMhr        = b.reshape
    //   decay_bhnLM     = exp(cumA_last − cumA)
    //   decay_bnLMh1    = decay_bhnLM.permute([0,2,3,1]).unsqueeze(4)
    //   decayed_v_bnLMhp = decay_bnLMh1 · v_bnLMhp                      (elementwise)
    //   decayed_v_bnhpLM = decayed_v_bnLMhp.permute([0,1,3,4,2])
    //   b_bnhLMr        = b_bnLMhr.swap_dims(2, 3)
    //   intra_state    = decayed_v_bnhpLM @ b_bnhLMr
    // ═══════════════════════════════════════════════════════════════════════
    let v_bnLMhp =
        v_bnlmhp
            .clone()
            .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let b_bnLMhr =
        b_bnlmhr
            .clone()
            .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
    let b_bnhLMr = b_bnLMhr.clone().swap_dims(2, 3);
    let decayed_v_bnhpLM = k3_decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);

    let d_decayed_v_bnhpLM: F<B, 5> = d_intra_chunk_state_bnhpr.clone().matmul(
        b_bnhLMr.transpose(), // b_bnhrLM
    ); // d_decayed_v_bnhpLM
    let d_b_k3_bnhLMr: F<B, 5> = decayed_v_bnhpLM
        .transpose() // decayed_v_bnhLMp
        .matmul(d_intra_chunk_state_bnhpr); // d_b_k3_bnhLMr

    let d_decayed_v_bnLMhp = d_decayed_v_bnhpLM.permute([0, 1, 4, 2, 3]);
    let d_decay_bhnLM: F<B, 4> = (d_decayed_v_bnLMhp.clone() * v_bnLMhp)
        .sum_dim(4) // d_decay_bnLMh1
        .squeeze_dim::<4>(4) // d_decay_bnLMh
        .permute([0, 3, 1, 2]); // d_decay_bhnLM

    // d_v_k3_bnLMhp = d_decayed_v · decay (broadcast)
    let k3_decay_bnLMh1 = k3_decay_bhnLM
        .clone()
        .permute([0, 2, 3, 1]) // k3_decay_bnLMh
        .unsqueeze_dim::<5>(4); // k3_decay_bnLMh1
    let d_v_k3_bnLMhp: F<B, 5> = d_decayed_v_bnLMhp * k3_decay_bnLMh1;
    let d_v_k3_bnlrhp: F<B, 6> =
        d_v_k3_bnLMhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

    // d(cumA_last − cumA) = d_decay · decay
    let d_decay_times_decay_bhnLM = d_decay_bhnLM * k3_decay_bhnLM;
    // d_a_cumsum_last: Σ over LM (broadcast dim).
    let d_a_cumsum_last_bhn: F<B, 3> = d_decay_times_decay_bhnLM
        .clone()
        .sum_dim(3) // d_decay_times_decay_bhn1
        .squeeze_dim::<3>(3); // d_a_cumsum_last_bhn
    // d_a_cumsum: negated (subtraction).
    let d_da_cumsum_bhnLM = -d_decay_times_decay_bhnLM;

    // Contribution to d_da_cumsum from the fused-cumA expand (sum mimo_rank copies).
    let d_da_cumsum_k3_from_fused_bhnl: F<B, 4> = d_da_cumsum_bhnLM
        .reshape([batch, nheads, nchunks, chunk_len, mimo_rank]) // d_da_cumsum_bhnlm
        .sum_dim(4) // d_da_cumsum_bhnl1
        .squeeze_dim::<4>(4); // d_da_cumsum_k3_from_fused_bhnl
    // Contribution from cumA_last: only the last-l position.
    let d_da_cumsum_k3_from_last_bhnl: F<B, 4> = {
        let zeros = F::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device, dtype);
        let d_last = d_a_cumsum_last_bhn.unsqueeze_dim::<4>(3);
        F::cat(vec![zeros, d_last], 3)
    };
    let d_da_cumsum_k3_bhnl = d_da_cumsum_k3_from_fused_bhnl + d_da_cumsum_k3_from_last_bhnl;

    // d_b_k3
    let d_b_k3_bnLMhr = d_b_k3_bnhLMr.swap_dims(2, 3);
    let d_b_k3_bnlmhr: F<B, 6> =
        d_b_k3_bnLMhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

    // ═══════════════════════════════════════════════════════════════════════
    // K2 BACKWARD (batched)
    //
    //   cb_bnhLMLM = c_bnhLMr @ b_bnhrLM   (contracts state_rank)
    //   d_c_bnhLMr = d_cb @ b_bnhLMr      (= d_cb @ b_bnhrLM^T)
    //   d_b_bnhrLM = c_bnhrLM @ d_cb      (= c_bnhLMr^T @ d_cb)
    // ═══════════════════════════════════════════════════════════════════════
    let c_bnhTMr = c_bntmhr
        .clone()
        .reshape([batch, nchunks, read, nheads, state_rank]) // c_bnTMhr
        .swap_dims(2, 3); // c_bnhTMr
    let b_for_k2_bnhLMr = b_bnLMhr.swap_dims(2, 3);

    let d_c_k2_bnhTMr: F<B, 5> = d_cb_bnhTMLM.clone().matmul(b_for_k2_bnhLMr);
    let d_b_k2_bnhrLM: F<B, 5> = c_bnhTMr
        .transpose() // c_bnhrTM
        .matmul(d_cb_bnhTMLM); // d_b_k2_bnhrLM

    // Undo permutes and reshape back
    let d_c_k2_bntmhr: F<B, 6> = d_c_k2_bnhTMr
        .swap_dims(2, 3) // d_c_k2_bnTMhr
        .reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]); // d_c_k2_bntmhr
    let d_b_k2_bnlmhr: F<B, 6> = d_b_k2_bnhrLM
        .permute([0, 1, 4, 2, 3]) // d_b_k2_bnLMhr
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]); // d_b_k2_bnlmhr

    // ── Unstack d_c_blue / d_v_orange and reshape back ────────────────────
    let d_c_blue_bntmhr: F<B, 6> = d_c_blue_bnhTMr
        .swap_dims(2, 3) // d_c_blue_bnTMhr
        .reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]); // d_c_blue_bntmhr
    let d_v_orange_bnlrhp: F<B, 6> = d_v_orange_bnhLMp
        .swap_dims(2, 3) // d_v_orange_bnLMhp
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]); // d_v_orange_bnlrhp

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
        let d_total_bhnl = d_da_cumsum_bhnl
            .clone()
            .sum_dim(3) // d_da_cumsum_bhn1
            .expand([batch, nheads, nchunks, chunk_len]); // d_total_bhnl
        let prefix_bhnl = d_da_cumsum_bhnl.cumsum(3);
        let zeros_bhn1 = F::<B, 4>::zeros([batch, nheads, nchunks, 1], &device, dtype);
        let prefix_shifted_bhnl =
            F::cat(vec![zeros_bhn1, prefix_bhnl.narrow(3, 0, chunk_len - 1)], 3);
        d_total_bhnl - prefix_shifted_bhnl
    };
    san(&d_da_bhnl);
    // Undo permute
    let d_da_bnlh = d_da_bhnl.permute([0, 2, 3, 1]);

    // ── Combine per-input gradient contributions ──────────────────────────
    let d_v_bnlmhp = d_v_k3_bnlrhp + d_v_orange_bnlrhp;
    let d_b_bnlmhr = d_b_k2_bnlmhr + d_b_k3_bnlmhr;
    let d_c_bntmhr = d_c_k2_bntmhr + d_c_blue_bntmhr;

    san(&d_v_bnlmhp);
    san(&d_da_bnlh);
    san(&d_b_bnlmhr);
    san(&d_c_bntmhr);
    san(&d_initial_state_bhpr);

    CombinedGrads {
        d_v_bnlmhp,
        d_da_bnlh,
        d_b_bnlmhr,
        d_c_bntmhr,
        d_initial_state_bhpr,
    }
}
