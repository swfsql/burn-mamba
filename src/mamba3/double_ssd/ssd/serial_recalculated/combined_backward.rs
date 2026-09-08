//! # Recompute-based gradient math for the Mamba-3 double-SSD
//!
//! The analytic backward of the MIMO-first serial scan used by each pass of the
//! double-SSD decomposition.  The forward intermediates (K1–K4) are recomputed
//! from the saved leaf inputs rather than stashed, then a reverse per-chunk loop
//! fuses the K5 and K4 backwards; K1/K2/K3 backwards run batched once the loop
//! has gathered the per-chunk slices.  The fused `L·M` length carries the
//! `mimo_rank` axis through the intra-chunk products.
//!
//! Everything operates on backend **primitives** through the rank-tagged `F`
//! wrapper: the custom [`Backward`](burn::backend::autodiff::ops::Backward) node
//! runs with a generic backend `B`, so the high-level `Tensor` is unavailable
//! and the math uses `B`'s `float_*` ops.  The recomputed K1/K2/K4 kernels are
//! local primitive ports of the high-level [`super::super::serial`](crate::mamba3::double_ssd::ssd::serial) kernels.

#![allow(non_snake_case)]

use super::serial_recalculated::{k1_ssd_chunk_cumsum, k2_ssd_bmm, k4_ssd_state_passing};
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
    // REVERSE PER-CHUNK LOOP — K5 (BLUE + ORANGE) + K4 fused
    //
    // Per-iteration working tensors are [batch,nheads,chunk_len*mimo_rank,...] rather than the
    // [batch,state_rank,nheads,chunk_len*mimo_rank,...] tensors a fully batched K5 backward would allocate.
    // ═══════════════════════════════════════════════════════════════════════
    let mut vec_orange_d_v_bhLMp: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_c_bhTMr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_d_cb_bhTMLM: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_da_bhl: Vec<F<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_orange_d_da_bhl: Vec<F<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_d_intra_bhpr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_d_da_end_bh: Vec<F<B, 2>> = Vec::with_capacity(nchunks);

    let mut d_running_state_bhpr: F<B, 4> = d_final_bhpr;

    for i_chunk in (0..nchunks).rev() {
        // ── Per-chunk slices (fused chunk_len · mimo_rank) ─────────────
        let v_bhLMp: F<B, 4> = v_bnlmhp
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..]) // v_b1lmhp
            .squeeze_dim::<5>(1) // v_blmhp
            .reshape([batch, fused, nheads, per_head_dim]) // v_bLMhp
            .swap_dims(1, 2); // v_bhLMp

        let c_bhTMr: F<B, 4> = c_bntmhr
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..]) // c_b1tmhr
            .squeeze_dim::<5>(1) // c_btmhr
            .reshape([batch, read, nheads, state_rank]) // c_bTMhr
            .swap_dims(1, 2); // c_bhTMr

        let cb_bhTMLM: F<B, 4> = cb_bnhTMLM
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // cb_b1hTMLM
            .squeeze_dim::<4>(1); // cb_bhTMLM

        let da_cumsum_bhLM: F<B, 3> = da_cumsum_bhnLM
            .clone()
            .slice(s![.., .., i_chunk, ..]) // da_cumsum_bh1LM
            .squeeze_dim::<3>(2); // da_cumsum_bhLM
        let da_cumsum_bhTM: F<B, 3> = da_cumsum_bhnTM
            .clone()
            .slice(s![.., .., i_chunk, ..]) // da_cumsum_bh1TM
            .squeeze_dim::<3>(2); // da_cumsum_bhTM

        let chunk_input_state_bhpr: F<B, 4> = chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // chunk_input_state_b1hpr
            .squeeze_dim::<4>(1); // chunk_input_state_bhpr
        san(&chunk_input_state_bhpr);

        let d_y_bhTMp: F<B, 4> = d_y_bnhTMp
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // d_y_b1hTMp
            .squeeze_dim::<4>(1); // d_y_bhTMp

        // ── BLUE backward ──────────────────────────────────────────────
        //
        //   blue[TM,p] = exp(cumA[TM]) · Σᵣ C[TM,r] · state[p,r]
        //
        // exp_da depends on the read position TM only — broadcast over per_head_dim.
        let exp_da_cumsum_bhTM: F<B, 3> = da_cumsum_bhTM.clone().exp();
        let exp_da_cumsum_bhTMp: F<B, 4> = exp_da_cumsum_bhTM
            .clone()
            .unsqueeze_dim::<4>(3) // exp_da_cumsum_bhTM1
            .expand([batch, nheads, read, per_head_dim]); // exp_da_cumsum_bhTMp
        let d_ch_bhTMp: F<B, 4> = d_y_bhTMp.clone() * exp_da_cumsum_bhTMp.clone();
        san(&d_ch_bhTMp);

        // d_chunk_input_state[p,r] = Σ_TM C[TM,r] · d_ch[TM,p]
        //   C^T (bhrTM) @ d_ch (bhTMp)  → bhrp  → transpose → bhpr
        let d_chunk_input_state_bhpr: F<B, 4> = c_bhTMr
            .clone()
            .transpose() // c_bhrTM
            .matmul(d_ch_bhTMp.clone()) // d_chunk_input_state_bhrp
            .transpose(); // d_chunk_input_state_bhpr
        san(&d_chunk_input_state_bhpr);

        // d_C_blue[TM,r] = Σₚ d_ch[TM,p] · state[p,r]
        //   d_ch (bhTMp) @ state (bhpr)  → bhTMr
        let d_c_blue_bhTMr: F<B, 4> = d_ch_bhTMp.matmul(chunk_input_state_bhpr.clone());
        san(&d_c_blue_bhTMr);
        vec_blue_d_c_bhTMr.push(d_c_blue_bhTMr);

        // d_da from BLUE:
        //   ch[TM,p] = Σᵣ C[TM,r] · state[p,r]      (= C @ state_rp after transpose)
        //   d_da[TM] = (Σₚ d_y[TM,p] · ch[TM,p]) · exp_da[TM]
        let ch_bhTMp: F<B, 4> = c_bhTMr.clone().matmul(
            chunk_input_state_bhpr.clone().transpose(), // chunk_input_state_bhrp
        ); // ch_bhTMp
        let d_da_blue_bhTM: F<B, 3> = (d_y_bhTMp.clone() * ch_bhTMp * exp_da_cumsum_bhTMp)
            .sum_dim(3) // d_da_blue_bhTM1
            .squeeze_dim::<3>(3); // d_da_blue_bhTM
        san(&d_da_blue_bhTM);

        // Reduce fused TM → t (sum the mimo_rank copies K5 broadcast), then
        // scatter back onto the folded axis: a read row's `da` is one position.
        let d_da_blue_bhl: F<B, 3> = scatter_read_rows::<B, 3, 4>(
            d_da_blue_bhTM
                .reshape([batch, nheads, chunk_tokens, mimo_rank]) // d_da_blue_bhtm
                .sum_dim(3) // d_da_blue_bht1
                .squeeze_dim::<3>(3), // d_da_blue_bht
            2,
            read_stride,
        );
        vec_blue_d_da_bhl.push(d_da_blue_bhl);

        // ── ORANGE backward ────────────────────────────────────────────
        //
        //   w[LMₜ,LMₛ] = CB[LMₜ,LMₛ] · decay[LMₜ,LMₛ]   (MIMO causal mask in decay)
        //   orange[LMₜ,p] = Σ_{LMₛ} w[LMₜ,LMₛ] · v[LMₛ,p]
        let da_target_bhTMLM: F<B, 4> = da_cumsum_bhTM
            .unsqueeze_dim::<4>(3) // da_cumsum_bhTMₜ1
            .expand([batch, nheads, read, fused]); // da_target_bhTMₜLM
        let da_source_bhTMLM: F<B, 4> = da_cumsum_bhLM
            .unsqueeze_dim::<4>(2) // da_cumsum_bh1LMₛ
            .expand([batch, nheads, read, fused]); // da_source_bhTMLMₛ
        let diff_bhTMLM = da_target_bhTMLM - da_source_bhTMLM;
        san(&diff_bhTMLM);

        // MIMO causal mask: -inf where LMₛ//mimo_rank > row(TMₜ)//mimo_rank —
        // interleaved expansion of the [t, l] upper-triangular base mask
        // (matches K5).
        let neg_inf_mimo_bhTMLM: F<B, 4> = neg_inf_base_tl
            .clone()
            .unsqueeze_dims::<4>(&[0, 1]) // neg_inf_base_11tl
            .expand([batch, nheads, chunk_tokens, chunk_len]) // neg_inf_base_bhtl
            .unsqueeze_dim::<5>(3) // neg_inf_base_bht1l
            .expand([batch, nheads, chunk_tokens, mimo_rank, chunk_len]) // neg_inf_base_bhtml
            .reshape([batch, nheads, read, chunk_len]) // neg_inf_base_bhTMl
            .unsqueeze_dim::<5>(4) // neg_inf_base_bhTMl1
            .expand([batch, nheads, read, chunk_len, mimo_rank]) // neg_inf_base_bhTMlm
            .reshape([batch, nheads, read, fused]); // neg_inf_mimo_bhTMLM
        let decay_bhTMLM = (diff_bhTMLM + neg_inf_mimo_bhTMLM).exp();
        san(&decay_bhTMLM);

        // d_v_orange = w^T @ d_orange ; d_w = d_orange @ v^T
        let d_orange_bhTMp = d_y_bhTMp;
        let w_bhTMLM = cb_bhTMLM.clone() * decay_bhTMLM.clone();
        let d_w_bhTMLM: F<B, 4> = d_orange_bhTMp.clone().matmul(
            v_bhLMp.clone().transpose(), // v_bhpLM
        ); // d_w_bhTMₜLMₛ
        san(&d_w_bhTMLM);
        let d_v_orange_bhLMp: F<B, 4> = w_bhTMLM
            .transpose() // w_bhLMₛTMₜ
            .matmul(d_orange_bhTMp); // d_v_orange_bhLMₛp
        san(&d_v_orange_bhLMp);
        vec_orange_d_v_bhLMp.push(d_v_orange_bhLMp);

        // d_cb = d_w · decay ; d_decay = d_w · cb ; d_diff = d_decay · decay
        // (masked positions where decay=0 contribute 0 to d_diff automatically)
        let d_cb_bhTMLM = d_w_bhTMLM.clone() * decay_bhTMLM.clone();
        vec_d_cb_bhTMLM.push(d_cb_bhTMLM);

        let d_decay_bhTMLM = d_w_bhTMLM * cb_bhTMLM;
        let d_diff_bhTMLM = d_decay_bhTMLM * decay_bhTMLM;

        // d_da_target[TMₜ] = Σ_{LMₛ} d_diff[TMₜ, LMₛ] (scattered onto the folded
        // axis); d_da_source[LMₛ] = Σ_{TMₜ} d_diff[TMₜ, LMₛ] (already folded);
        // d_da_orange = d_da_target − d_da_source  (diff = target − source).
        let d_da_target_bhTM: F<B, 3> = d_diff_bhTMLM
            .clone()
            .sum_dim(3) // d_diff_bhTMₜ1
            .squeeze_dim::<3>(3); // d_da_target_bhTMₜ
        let d_da_source_bhLM: F<B, 3> = d_diff_bhTMLM
            .sum_dim(2) // d_diff_bh1LMₛ
            .squeeze_dim::<3>(2); // d_da_source_bhLMₛ

        let d_da_target_bhl: F<B, 3> = scatter_read_rows::<B, 3, 4>(
            d_da_target_bhTM
                .reshape([batch, nheads, chunk_tokens, mimo_rank])
                .sum_dim(3)
                .squeeze_dim::<3>(3),
            2,
            read_stride,
        );
        let d_da_source_bhl: F<B, 3> = d_da_source_bhLM
            .reshape([batch, nheads, chunk_len, mimo_rank]) // d_da_source_bhlm
            .sum_dim(3) // d_da_source_bhl1
            .squeeze_dim::<3>(3); // d_da_source_bhl
        let d_da_orange_bhl = d_da_target_bhl - d_da_source_bhl;
        san(&d_da_orange_bhl);
        vec_orange_d_da_bhl.push(d_da_orange_bhl);

        // ── K4 backward step for chunk i_chunk ─────────────────────────
        //
        // Forward (recap):  sᵢ₊₁ = decayᵢ · sᵢ + intra_stateᵢ
        //   - d_intra_stateᵢ      = d_sᵢ₊₁      (current d_running_state)
        //   - d_decayᵢ            = d_sᵢ₊₁ · sᵢ
        //   - d_sᵢ (propagated)   = decayᵢ · d_sᵢ₊₁ + d_chunk_input_state_blue
        vec_d_intra_bhpr.push(d_running_state_bhpr.clone());

        let decay_chunk_bhpr: F<B, 4> = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk]) // da_chunk_end_bh1
            .exp() // decay_chunk_bh
            .unsqueeze_dim::<4>(3) // decay_chunk_bh11
            .expand([batch, nheads, per_head_dim, state_rank]); // decay_chunk_bhpr
        san(&decay_chunk_bhpr);

        let d_decay_chunk_bhpr = d_running_state_bhpr.clone() * chunk_input_state_bhpr;
        // d_da_chunk_end[b,h] = Σ_{p,r} d_decay · decay (since decay = exp(da_chunk_end))
        let d_da_chunk_end_bh: F<B, 2> = (d_decay_chunk_bhpr * decay_chunk_bhpr.clone())
            .reshape([batch, nheads, per_head_dim * state_rank]) // d_da_chunk_end_bhPR
            .sum_dim(2) // d_da_chunk_end_bh1
            .squeeze_dim::<2>(2); // d_da_chunk_end_bh
        san(&d_da_chunk_end_bh);
        vec_d_da_end_bh.push(d_da_chunk_end_bh);

        d_running_state_bhpr = decay_chunk_bhpr * d_running_state_bhpr + d_chunk_input_state_bhpr;
        san(&d_running_state_bhpr);
    }
    let d_initial_state_bhpr = d_running_state_bhpr;

    // ── Restore natural (forward) chunk order ─────────────────────────────
    vec_orange_d_v_bhLMp.reverse();
    vec_blue_d_c_bhTMr.reverse();
    vec_d_cb_bhTMLM.reverse();
    vec_blue_d_da_bhl.reverse();
    vec_orange_d_da_bhl.reverse();
    vec_d_intra_bhpr.reverse();
    vec_d_da_end_bh.reverse();

    // ── Stack per-chunk slices back into batched tensors ──────────────────
    let d_v_orange_bnhLMp: F<B, 5> = F::stack(vec_orange_d_v_bhLMp, 1);
    let d_c_blue_bnhTMr: F<B, 5> = F::stack(vec_blue_d_c_bhTMr, 1);
    let d_cb_bnhTMLM: F<B, 5> = F::stack(vec_d_cb_bhTMLM, 1);
    let d_da_blue_bhnl: F<B, 4> = F::stack(vec_blue_d_da_bhl, 2);
    let d_da_orange_bhnl: F<B, 4> = F::stack(vec_orange_d_da_bhl, 2);
    let d_intra_chunk_state_bnhpr: F<B, 5> = F::stack(vec_d_intra_bhpr, 1);
    // d_da_end:
    // [batch,nheads]     → stack@2 → [batch,nheads,nchunks]; scatter into last-l of d_da_cumsum_k4
    let d_da_end_bhn: F<B, 3> = F::stack(vec_d_da_end_bh, 2);
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
