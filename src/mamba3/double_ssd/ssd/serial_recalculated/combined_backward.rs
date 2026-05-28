//! # Recompute-based gradient math for the Mamba-3 double-SSD
//!
//! The analytic backward of the MIMO-first serial scan used by each pass of the
//! double-SSD decomposition.  The forward intermediates (K1–K4) are recomputed
//! from the saved leaf inputs rather than stashed, then a reverse per-chunk loop
//! fuses the K5 and K4 backwards; K1/K2/K3 backwards run batched once the loop
//! has gathered the per-chunk slices.  The fused `L·M` length carries the
//! `mimo_rank` axis through the intra-chunk products.
//!
//! Everything operates on backend **primitives** through the rank-tagged [`F`]
//! wrapper: the custom [`Backward`](burn::backend::autodiff::ops::Backward) node
//! runs with a generic backend `B`, so the high-level `Tensor` is unavailable
//! and the math uses `B`'s `float_*` ops.  The recomputed K1/K2/K4 kernels are
//! local primitive ports of the high-level [`super::super::serial`] kernels.

#![allow(non_snake_case)]

use super::serial_recalculated::{k1_ssd_chunk_cumsum, k2_ssd_bmm, k4_ssd_state_passing};
use crate::utils::fprim::{F, san};
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
    /// Gradient of the output projection `C`.
    pub d_c_bnlmhr: F<B, 6>,
    /// Gradient of the initial SSM state.
    pub d_initial_state_bhpr: F<B, 4>,
}

// ─── Recomputed forward kernels ──────────────────────────────────────────────
// The recompute backward replays the forward's K1/K2/K4 (imported above from
// [`super::serial_recalculated`]) plus the extended K3 below, which returns the
// extra intermediates the gradient math needs.

/// Same as [`k3_ssd_chunk_state`](super::serial_recalculated::k3_ssd_chunk_state) but
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

    let decay_bnLMh1 = decay_bhnLM.clone().permute([0, 2, 3, 1]).unsqueeze_dim::<5>(4);
    let decayed_v_bnLMhp = decay_bnLMh1 * v_bnLMhp;
    san(&decayed_v_bnLMhp);

    let decayed_v_bnhpLM = decayed_v_bnLMhp.clone().permute([0, 1, 3, 4, 2]);
    let b_bnhLMr = b_bnLMhr.permute([0, 1, 3, 2, 4]);
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
    d_y_bnlmhp: F<B, 6>,
    d_final_bhpr: F<B, 4>,
    //
    v_bnlmhp: F<B, 6>,
    da_bnlh: F<B, 4>,
    b_bnlmhr: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    initial_state_bhpr: F<B, 4>,
) -> CombinedGrads<B> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();
    let device = v_bnlmhp.device();
    let dtype = v_bnlmhp.dtype();

    san(&d_y_bnlmhp);
    san(&d_final_bhpr);
    san(&v_bnlmhp);
    san(&da_bnlh);
    san(&b_bnlmhr);
    san(&c_bnlmhr);
    san(&initial_state_bhpr);

    // ═══════════════════════════════════════════════════════════════════════
    // RECOMPUTE FORWARD INTERMEDIATES
    // ═══════════════════════════════════════════════════════════════════════

    // K1 — pre-combined Δ·A → intra-chunk cumsum
    let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum(da_bnlh.clone());
    san(&da_cumsum_bhnl);

    // K2 — CB matrix used in K5 ORANGE
    let cb_bnhLMLM = k2_ssd_bmm(c_bnlmhr.clone(), b_bnlmhr.clone());
    san(&cb_bnhLMLM);

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
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // da_cumsum_bhnLM

    // d_y in (batch, nchunks, nheads, chunk_len * mimo_rank, per_head_dim) ordering
    // — matches the per-chunk slicing.
    let d_y_bnhLMp = d_y_bnlmhp
        .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]) // d_y_bnLMhp
        .permute([0, 1, 3, 2, 4]); // d_y_bnhLMp
    san(&d_y_bnhLMp);

    // Reusable [chunk_len, chunk_len] -inf upper-triangular base mask for ORANGE.
    let neg_inf_base_ll: F<B, 2> =
        { F::<B, 2>::full([chunk_len, chunk_len], f32::NEG_INFINITY, &device, dtype).triu(1) };

    // ═══════════════════════════════════════════════════════════════════════
    // REVERSE PER-CHUNK LOOP — K5 (BLUE + ORANGE) + K4 fused
    //
    // Per-iteration working tensors are [batch,nheads,chunk_len*mimo_rank,...] rather than the
    // [batch,state_rank,nheads,chunk_len*mimo_rank,...] tensors a fully batched K5 backward would allocate.
    // ═══════════════════════════════════════════════════════════════════════
    let mut vec_orange_d_v_bhLMp: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_c_bhLMr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_d_cb_bhLMLM: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
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
            .reshape([batch, chunk_len * mimo_rank, nheads, per_head_dim]) // v_bLMhp
            .permute([0, 2, 1, 3]); // v_bhLMp

        let c_bhLMr: F<B, 4> = c_bnlmhr
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..]) // c_b1lmhr
            .squeeze_dim::<5>(1) // c_blmhr
            .reshape([batch, chunk_len * mimo_rank, nheads, state_rank]) // c_bLMhr
            .permute([0, 2, 1, 3]); // c_bhLMr

        let cb_bhLMLM: F<B, 4> = cb_bnhLMLM
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // cb_b1hLMLM
            .squeeze_dim::<4>(1); // cb_bhLMLM

        let da_cumsum_bhLM: F<B, 3> = da_cumsum_bhnLM
            .clone()
            .slice(s![.., .., i_chunk, ..]) // da_cumsum_bh1LM
            .squeeze_dim::<3>(2); // da_cumsum_bhLM

        let chunk_input_state_bhpr: F<B, 4> = chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // chunk_input_state_b1hpr
            .squeeze_dim::<4>(1); // chunk_input_state_bhpr
        san(&chunk_input_state_bhpr);

        let d_y_bhLMp: F<B, 4> = d_y_bnhLMp
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // d_y_b1hLMp
            .squeeze_dim::<4>(1); // d_y_bhLMp

        // ── BLUE backward ──────────────────────────────────────────────
        //
        //   blue[LM,p] = exp(cumA[LM]) · Σᵣ C[LM,r] · state[p,r]
        //
        // exp_da depends on the fused position LM only — broadcast over per_head_dim.
        let exp_da_cumsum_bhLM: F<B, 3> = da_cumsum_bhLM.clone().exp();
        let exp_da_cumsum_bhLMp: F<B, 4> = exp_da_cumsum_bhLM
            .clone()
            .unsqueeze_dim::<4>(3) // exp_da_cumsum_bhLM1
            .expand([batch, nheads, chunk_len * mimo_rank, per_head_dim]); // exp_da_cumsum_bhLMp
        let d_ch_bhLMp: F<B, 4> = d_y_bhLMp.clone() * exp_da_cumsum_bhLMp.clone();
        san(&d_ch_bhLMp);

        // d_chunk_input_state[p,r] = Σ_LM C[LM,r] · d_ch[LM,p]
        //   C^T (bhrLM) @ d_ch (bhLMp)  → bhrp  → permute → bhpr
        let d_chunk_input_state_bhpr: F<B, 4> = c_bhLMr
            .clone()
            .permute([0, 1, 3, 2]) // c_bhrLM
            .matmul(d_ch_bhLMp.clone()) // d_chunk_input_state_bhrp
            .permute([0, 1, 3, 2]); // d_chunk_input_state_bhpr
        san(&d_chunk_input_state_bhpr);

        // d_C_blue[LM,r] = Σₚ d_ch[LM,p] · state[p,r]
        //   d_ch (bhLMp) @ state (bhpr)  → bhLMr
        let d_c_blue_bhLMr: F<B, 4> = d_ch_bhLMp.matmul(chunk_input_state_bhpr.clone());
        san(&d_c_blue_bhLMr);
        vec_blue_d_c_bhLMr.push(d_c_blue_bhLMr);

        // d_da from BLUE:
        //   ch[LM,p] = Σᵣ C[LM,r] · state[p,r]      (= C @ state_rp after permute)
        //   d_da[LM] = (Σₚ d_y[LM,p] · ch[LM,p]) · exp_da[LM]
        let ch_bhLMp: F<B, 4> = c_bhLMr.clone().matmul(
            chunk_input_state_bhpr.clone().permute([0, 1, 3, 2]), // chunk_input_state_bhrp
        ); // ch_bhLMp
        let d_da_blue_bhLM: F<B, 3> = (d_y_bhLMp.clone() * ch_bhLMp * exp_da_cumsum_bhLMp)
            .sum_dim(3) // d_da_blue_bhLM1
            .squeeze_dim::<3>(3); // d_da_blue_bhLM
        san(&d_da_blue_bhLM);

        // Reduce fused LM → l (sum the mimo_rank copies that K5 broadcast).
        let d_da_blue_bhl: F<B, 3> = d_da_blue_bhLM
            .reshape([batch, nheads, chunk_len, mimo_rank]) // d_da_blue_bhlm
            .sum_dim(3) // d_da_blue_bhl1
            .squeeze_dim::<3>(3); // d_da_blue_bhl
        vec_blue_d_da_bhl.push(d_da_blue_bhl);

        // ── ORANGE backward ────────────────────────────────────────────
        //
        //   w[LMₜ,LMₛ] = CB[LMₜ,LMₛ] · decay[LMₜ,LMₛ]   (MIMO causal mask in decay)
        //   orange[LMₜ,p] = Σ_{LMₛ} w[LMₜ,LMₛ] · v[LMₛ,p]
        let da_target_bhLMLM: F<B, 4> = da_cumsum_bhLM
            .clone()
            .unsqueeze_dim::<4>(3) // da_cumsum_bhLMₜ1
            .expand([batch, nheads, chunk_len * mimo_rank, chunk_len * mimo_rank]); // da_target_bhLMₜLM
        let da_source_bhLMLM: F<B, 4> = da_cumsum_bhLM
            .unsqueeze_dim::<4>(2) // da_cumsum_bh1LMₛ
            .expand([batch, nheads, chunk_len * mimo_rank, chunk_len * mimo_rank]); // da_source_bhLMLMₛ
        let diff_bhLMLM = da_target_bhLMLM - da_source_bhLMLM;
        san(&diff_bhLMLM);

        // MIMO causal mask: -inf where LMₛ//mimo_rank > LMₜ//mimo_rank — interleaved expansion
        // of the [l, l] upper-triangular base mask (matches K5).
        let neg_inf_mimo_bhLMLM: F<B, 4> = neg_inf_base_ll
            .clone()
            .unsqueeze_dims::<4>(&[0, 1]) // neg_inf_base_11ll
            .expand([batch, nheads, chunk_len, chunk_len]) // neg_inf_base_bhll
            .unsqueeze_dim::<5>(3) // neg_inf_base_bhl1l
            .expand([batch, nheads, chunk_len, mimo_rank, chunk_len]) // neg_inf_base_bhlml
            .reshape([batch, nheads, chunk_len * mimo_rank, chunk_len]) // neg_inf_base_bhLMl
            .unsqueeze_dim::<5>(4) // neg_inf_base_bhLMl1
            .expand([batch, nheads, chunk_len * mimo_rank, chunk_len, mimo_rank]) // neg_inf_base_bhLMlm
            .reshape([batch, nheads, chunk_len * mimo_rank, chunk_len * mimo_rank]); // neg_inf_mimo_bhLMLM
        let decay_bhLMLM = (diff_bhLMLM + neg_inf_mimo_bhLMLM).exp();
        san(&decay_bhLMLM);

        // d_v_orange = w^T @ d_orange ; d_w = d_orange @ v^T
        let d_orange_bhLMp = d_y_bhLMp;
        let w_bhLMLM = cb_bhLMLM.clone() * decay_bhLMLM.clone();
        let d_w_bhLMLM: F<B, 4> = d_orange_bhLMp.clone().matmul(
            v_bhLMp.clone().permute([0, 1, 3, 2]), // v_bhpLM
        ); // d_w_bhLMₜLMₛ
        san(&d_w_bhLMLM);
        let d_v_orange_bhLMp: F<B, 4> = w_bhLMLM
            .permute([0, 1, 3, 2]) // w_bhLMₛLMₜ
            .matmul(d_orange_bhLMp); // d_v_orange_bhLMₛp
        san(&d_v_orange_bhLMp);
        vec_orange_d_v_bhLMp.push(d_v_orange_bhLMp);

        // d_cb = d_w · decay ; d_decay = d_w · cb ; d_diff = d_decay · decay
        // (masked positions where decay=0 contribute 0 to d_diff automatically)
        let d_cb_bhLMLM = d_w_bhLMLM.clone() * decay_bhLMLM.clone();
        vec_d_cb_bhLMLM.push(d_cb_bhLMLM);

        let d_decay_bhLMLM = d_w_bhLMLM * cb_bhLMLM;
        let d_diff_bhLMLM = d_decay_bhLMLM * decay_bhLMLM;

        // d_da_target[LMₜ] = Σ_{LMₛ} d_diff[LMₜ, LMₛ] ;
        // d_da_source[LMₛ] = Σ_{LMₜ} d_diff[LMₜ, LMₛ] ;
        // d_da_orange = d_da_target − d_da_source  (diff = target − source).
        let d_da_target_bhLM: F<B, 3> = d_diff_bhLMLM
            .clone()
            .sum_dim(3) // d_diff_bhLMₜ1
            .squeeze_dim::<3>(3); // d_da_target_bhLMₜ
        let d_da_source_bhLM: F<B, 3> = d_diff_bhLMLM
            .sum_dim(2) // d_diff_bh1LMₛ
            .squeeze_dim::<3>(2); // d_da_source_bhLMₛ
        let d_da_orange_bhLM = d_da_target_bhLM - d_da_source_bhLM;
        san(&d_da_orange_bhLM);

        // Reduce fused LM → l (sum over the mimo_rank-broadcast copies).
        let d_da_orange_bhl: F<B, 3> = d_da_orange_bhLM
            .reshape([batch, nheads, chunk_len, mimo_rank]) // d_da_orange_bhlm
            .sum_dim(3) // d_da_orange_bhl1
            .squeeze_dim::<3>(3); // d_da_orange_bhl
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
    vec_blue_d_c_bhLMr.reverse();
    vec_d_cb_bhLMLM.reverse();
    vec_blue_d_da_bhl.reverse();
    vec_orange_d_da_bhl.reverse();
    vec_d_intra_bhpr.reverse();
    vec_d_da_end_bh.reverse();

    // ── Stack per-chunk slices back into batched tensors ──────────────────
    let d_v_orange_bnhLMp: F<B, 5> = F::stack(vec_orange_d_v_bhLMp, 1);
    let d_c_blue_bnhLMr: F<B, 5> = F::stack(vec_blue_d_c_bhLMr, 1);
    let d_cb_bnhLMLM: F<B, 5> = F::stack(vec_d_cb_bhLMLM, 1);
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
    //   b_bnhLMr        = b_bnLMhr.permute([0,1,3,2,4])
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
    let b_bnhLMr = b_bnLMhr.clone().permute([0, 1, 3, 2, 4]);
    let decayed_v_bnhpLM = k3_decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);

    let d_decayed_v_bnhpLM: F<B, 5> = d_intra_chunk_state_bnhpr.clone().matmul(
        b_bnhLMr.permute([0, 1, 2, 4, 3]), // b_bnhrLM
    ); // d_decayed_v_bnhpLM
    let d_b_k3_bnhLMr: F<B, 5> = decayed_v_bnhpLM
        .permute([0, 1, 2, 4, 3]) // decayed_v_bnhLMp
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
    let d_b_k3_bnLMhr = d_b_k3_bnhLMr.permute([0, 1, 3, 2, 4]);
    let d_b_k3_bnlmhr: F<B, 6> =
        d_b_k3_bnLMhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

    // ═══════════════════════════════════════════════════════════════════════
    // K2 BACKWARD (batched)
    //
    //   cb_bnhLMLM = c_bnhLMr @ b_bnhrLM   (contracts state_rank)
    //   d_c_bnhLMr = d_cb @ b_bnhLMr      (= d_cb @ b_bnhrLM^T)
    //   d_b_bnhrLM = c_bnhrLM @ d_cb      (= c_bnhLMr^T @ d_cb)
    // ═══════════════════════════════════════════════════════════════════════
    let c_bnhLMr = c_bnlmhr
        .clone()
        .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]) // c_bnLMhr
        .permute([0, 1, 3, 2, 4]); // c_bnhLMr
    let b_for_k2_bnhLMr = b_bnLMhr.permute([0, 1, 3, 2, 4]);

    let d_c_k2_bnhLMr: F<B, 5> = d_cb_bnhLMLM.clone().matmul(b_for_k2_bnhLMr);
    let d_b_k2_bnhrLM: F<B, 5> = c_bnhLMr
        .permute([0, 1, 2, 4, 3]) // c_bnhrLM
        .matmul(d_cb_bnhLMLM); // d_b_k2_bnhrLM

    // Undo permutes and reshape back
    let d_c_k2_bnlmhr: F<B, 6> = d_c_k2_bnhLMr
        .permute([0, 1, 3, 2, 4]) // d_c_k2_bnLMhr
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]); // d_c_k2_bnlmhr
    let d_b_k2_bnlmhr: F<B, 6> = d_b_k2_bnhrLM
        .permute([0, 1, 4, 2, 3]) // d_b_k2_bnLMhr
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]); // d_b_k2_bnlmhr

    // ── Unstack d_c_blue / d_v_orange and reshape back ────────────────────
    let d_c_blue_bnlmhr: F<B, 6> = d_c_blue_bnhLMr
        .permute([0, 1, 3, 2, 4]) // d_c_blue_bnLMhr
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]); // d_c_blue_bnlmhr
    let d_v_orange_bnlrhp: F<B, 6> = d_v_orange_bnhLMp
        .permute([0, 1, 3, 2, 4]) // d_v_orange_bnLMhp
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
    let d_c_bnlmhr = d_c_k2_bnlmhr + d_c_blue_bnlmhr;

    san(&d_v_bnlmhp);
    san(&d_da_bnlh);
    san(&d_b_bnlmhr);
    san(&d_c_bnlmhr);
    san(&d_initial_state_bhpr);

    CombinedGrads {
        d_v_bnlmhp,
        d_da_bnlh,
        d_b_bnlmhr,
        d_c_bnlmhr,
        d_initial_state_bhpr,
    }
}
