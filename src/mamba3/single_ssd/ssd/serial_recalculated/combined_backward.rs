#![allow(non_snake_case)]

use crate::mamba3::double_ssd::ssd::serial_recalculated::combined_backward::k3_ssd_chunk_state_extended;
use crate::mamba3::single_ssd::ssd;
use crate::utils::sanity::sanity as san;
use burn::prelude::*;
use ssd::serial;

/// Per-input gradients produced by [`combined_backward`] for the Single-SSD.
/// Adds `d_gamma_bnlh` and `d_scale_bnlh` over the double-ssd form
/// [`crate::mamba3::double_ssd::ssd::serial_recalculated::combined_backward::CombinedGrads`].
#[non_exhaustive]
pub struct CombinedSingleSsdGrads<B: Backend> {
    pub d_v_bnlmhp: Tensor<B, 6>,
    pub d_da_bnlh: Tensor<B, 4>,
    pub d_b_bnlmhr: Tensor<B, 6>,
    pub d_c_bnlmhr: Tensor<B, 6>,
    pub d_gamma_bnlh: Tensor<B, 4>,
    pub d_scale_bnlh: Tensor<B, 4>,
    pub d_initial_state_bhpr: Tensor<B, 4>,
}

/// Memory-efficient backward for the Mamba-3 MIMO-first chunkwise Single-SSD.
///
/// Recomputes the forward intermediates (K1–K4) from the saved inputs, then:
/// - runs a reverse per-chunk loop that fuses the K5 BLUE (state-to-output) and
///   the strict lower-triangular LOWER (intra-chunk) backward with the K4
///   state-passing backward, and
/// - computes the γ-weighted same-step DIAG backward batched (it has no
///   recurrence, and the `m × m` working tensors are tiny).
///
/// K3/K2/K1 backwards run as single batched ops once the loop has collected all
/// per-chunk slices.
///
/// # Arguments
/// - `d_y_bnlmhp` — upstream gradient of the SSD output
/// - `d_final_bhpr` — upstream gradient of the final SSM state
/// - `v_bnlmhp`, `da_bnlh`, `b_bnlmhr`, `c_bnlmhr`, `gamma_bnlh`, `scale_bnlh`,
///   `initial_state_bhpr` — the seven saved forward inputs
///
/// # Returns
/// One [`CombinedSingleSsdGrads`] with gradients for all 7 inputs.
#[allow(clippy::too_many_arguments)]
pub fn combined_backward<B: Backend>(
    d_y_bnlmhp: Tensor<B, 6>,
    d_final_bhpr: Tensor<B, 4>,
    //
    v_bnlmhp: Tensor<B, 6>,
    da_bnlh: Tensor<B, 4>,
    b_bnlmhr: Tensor<B, 6>,
    c_bnlmhr: Tensor<B, 6>,
    gamma_bnlh: Tensor<B, 4>,
    scale_bnlh: Tensor<B, 4>,
    initial_state_bhpr: Tensor<B, 4>,
) -> CombinedSingleSsdGrads<B> {
    use burn::tensor::s;

    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();
    let device = v_bnlmhp.device();

    san(&d_y_bnlmhp);
    san(&d_final_bhpr);
    san(&v_bnlmhp);
    san(&da_bnlh);
    san(&b_bnlmhr);
    san(&c_bnlmhr);
    san(&gamma_bnlh);
    san(&scale_bnlh);
    san(&initial_state_bhpr);

    // ═══════════════════════════════════════════════════════════════════════
    // RECOMPUTE FORWARD INTERMEDIATES (K1–K4, single-ssd form)
    // ═══════════════════════════════════════════════════════════════════════

    // K1
    let (da_cumsum_bhnl, da_chunk_end_bhn) = serial::k1_ssd_chunk_cumsum(da_bnlh.clone());
    san(&da_cumsum_bhnl);

    // K2 — CB matrix (unscaled), used by LOWER.
    let cb_bnhLMLM = serial::k2_ssd_bmm(c_bnlmhr.clone(), b_bnlmhr.clone());
    san(&cb_bnhLMLM);

    // K3 — chunk state from K_scaled = scaleₜ·B.
    let scale_bnlh11 = scale_bnlh.clone().unsqueeze_dims::<6>(&[3, 5]);
    let k_scaled_bnlmhr = b_bnlmhr.clone() * scale_bnlh11.clone();
    let (intra_chunk_state_bnhpr, k3_decay_bhnLM, k3_decayed_v_bnLMhp) =
        k3_ssd_chunk_state_extended(
            v_bnlmhp.clone(),
            k_scaled_bnlmhr.clone(),
            da_cumsum_bhnl.clone(),
        );

    // K4 — chunk-input state stream consumed by BLUE.
    let (chunk_input_state_bnhpr, _final_state_bhpr) = serial::k4_ssd_state_passing(
        intra_chunk_state_bnhpr,
        da_chunk_end_bhn.clone(),
        initial_state_bhpr,
    );

    // Fused-position cumulative decay.
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .clone()
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]);

    // d_y in (batch, nchunks, nheads, chunk_len·mimo_rank, per_head_dim) ordering.
    let d_y_bnhLMp = d_y_bnlmhp
        .clone()
        .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim])
        .permute([0, 1, 3, 2, 4]);
    san(&d_y_bnhLMp);

    // ═══════════════════════════════════════════════════════════════════════
    // DIAG BACKWARD (batched — no recurrence; m × m working set is tiny)
    //
    // Forward (per (b,n,l,h)):
    //   qk_dot[m_out, m_in] = Σ_r C[m_out, r] · B[m_in, r]
    //   y_diag[m_out, p]    = γ · Σ_{m_in} qk_dot[m_out, m_in] · V[m_in, p]
    // ═══════════════════════════════════════════════════════════════════════
    let (d_v_diag_bnlmhp, d_c_diag_bnlmhr, d_b_diag_bnlmhr, d_gamma_bnlh) = {
        let c_bnlhmr = c_bnlmhr.clone().permute([0, 1, 2, 4, 3, 5]); // [b,n,l,h,m_out,r]
        let b_bnlhmr = b_bnlmhr.clone().permute([0, 1, 2, 4, 3, 5]); // [b,n,l,h,m_in,r]
        let v_bnlhmp = v_bnlmhp.clone().permute([0, 1, 2, 4, 3, 5]); // [b,n,l,h,m_in,p]
        let d_y_bnlhmp = d_y_bnlmhp.clone().permute([0, 1, 2, 4, 3, 5]); // [b,n,l,h,m_out,p]

        // qk_dot[m_out, m_in] = Σ_r C[m_out,r] · B[m_in,r]
        let qk_dot_bnlhmM = c_bnlhmr
            .clone()
            .matmul(b_bnlhmr.clone().permute([0, 1, 2, 3, 5, 4]));
        // y_d_unweighted[m_out, p] = Σ_{m_in} qk_dot · V[m_in, p]
        let y_d_unw_bnlhmp = qk_dot_bnlhmM.clone().matmul(v_bnlhmp.clone());

        // d_gamma[b,n,l,h] = Σ_{m_out,p} d_y · y_d_unweighted
        let d_gamma_bnlh: Tensor<B, 4> = (d_y_bnlhmp.clone() * y_d_unw_bnlhmp)
            .sum_dim(5) // bnlhm1
            .squeeze_dim::<5>(5) // bnlhm
            .sum_dim(4) // bnlh1
            .squeeze_dim::<4>(4); // bnlh
        san(&d_gamma_bnlh);

        // d_y_d_unweighted = γ · d_y  (γ broadcast over m_out, p)
        let gamma_bnlh11 = gamma_bnlh.clone().unsqueeze_dims::<6>(&[4, 5]);
        let d_y_d_unw_bnlhmp = d_y_bnlhmp * gamma_bnlh11;

        // d_qk_dot[m_out, m_in] = Σ_p d_y_d_unweighted[m_out, p] · V[m_in, p]
        let d_qk_dot_bnlhmM = d_y_d_unw_bnlhmp
            .clone()
            .matmul(v_bnlhmp.clone().permute([0, 1, 2, 3, 5, 4])); // [b,n,l,h,m_out,m_in]

        // d_v_diag[m_in, p] = Σ_{m_out} qk_dot[m_out, m_in] · d_y_d_unweighted[m_out, p]
        let d_v_diag_bnlhmp = qk_dot_bnlhmM
            .permute([0, 1, 2, 3, 5, 4]) // qk_dot^T: [b,n,l,h,m_in,m_out]
            .matmul(d_y_d_unw_bnlhmp.clone()); // [b,n,l,h,m_in,p]

        // d_C_diag[m_out, r] = Σ_{m_in} d_qk_dot[m_out, m_in] · B[m_in, r]
        let d_c_diag_bnlhmr = d_qk_dot_bnlhmM.clone().matmul(b_bnlhmr); // [b,n,l,h,m_out,r]
        // d_B_diag[m_in, r] = Σ_{m_out} d_qk_dot[m_out, m_in] · C[m_out, r]
        let d_b_diag_bnlhmr = d_qk_dot_bnlhmM
            .permute([0, 1, 2, 3, 5, 4]) // d_qk_dot^T: [b,n,l,h,m_in,m_out]
            .matmul(c_bnlhmr); // [b,n,l,h,m_in,r]

        // Back to [b,n,l,m,h,*].
        let d_v_diag_bnlmhp = d_v_diag_bnlhmp.permute([0, 1, 2, 4, 3, 5]);
        let d_c_diag_bnlmhr = d_c_diag_bnlhmr.permute([0, 1, 2, 4, 3, 5]);
        let d_b_diag_bnlmhr = d_b_diag_bnlhmr.permute([0, 1, 2, 4, 3, 5]);
        (
            d_v_diag_bnlmhp,
            d_c_diag_bnlmhr,
            d_b_diag_bnlmhr,
            d_gamma_bnlh,
        )
    };

    // Reusable [chunk_len, chunk_len] -inf strict-upper mask (triu(0): on+above
    // diagonal → -inf) for the LOWER (strict lower-triangular) path.
    let neg_inf_strict_ll: Tensor<B, 2> =
        Tensor::full([chunk_len, chunk_len], f32::NEG_INFINITY, &device).triu(0);

    // ═══════════════════════════════════════════════════════════════════════
    // REVERSE PER-CHUNK LOOP — K5 (BLUE + LOWER) + K4 fused
    // ═══════════════════════════════════════════════════════════════════════
    let mut vec_lower_d_v_bhLMp: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_c_bhLMr: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_d_cb_bhLMLM: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_da_bhl: Vec<Tensor<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_lower_d_da_bhl: Vec<Tensor<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_lower_d_scale_bhl: Vec<Tensor<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_d_intra_bhpr: Vec<Tensor<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_d_da_end_bh: Vec<Tensor<B, 2>> = Vec::with_capacity(nchunks);

    let mut d_running_state_bhpr: Tensor<B, 4> = d_final_bhpr;

    for i_chunk in (0..nchunks).rev() {
        // ── Per-chunk slices (fused chunk_len · mimo_rank) ─────────────
        let v_bhLMp: Tensor<B, 4> = v_bnlmhp
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..])
            .squeeze_dim::<5>(1)
            .reshape([batch, chunk_len * mimo_rank, nheads, per_head_dim])
            .permute([0, 2, 1, 3]);

        let c_bhLMr: Tensor<B, 4> = c_bnlmhr
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..])
            .squeeze_dim::<5>(1)
            .reshape([batch, chunk_len * mimo_rank, nheads, state_rank])
            .permute([0, 2, 1, 3]);

        let cb_bhLMLM: Tensor<B, 4> = cb_bnhLMLM
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);

        let da_cumsum_bhLM: Tensor<B, 3> = da_cumsum_bhnLM
            .clone()
            .slice(s![.., .., i_chunk, ..])
            .squeeze_dim::<3>(2);

        // scaleₜ per fused source position: scale[s_time] broadcast over s_m.
        let scale_bhLM: Tensor<B, 3> = scale_bnlh
            .clone()
            .slice(s![.., i_chunk, .., ..]) // [b, l, h]
            .squeeze_dim::<3>(1)
            .swap_dims(1, 2) // [b, h, l]
            .unsqueeze_dim::<4>(3) // [b, h, l, 1]
            .expand([batch, nheads, chunk_len, mimo_rank])
            .reshape([batch, nheads, chunk_len * mimo_rank]);

        let chunk_input_state_bhpr: Tensor<B, 4> = chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);
        san(&chunk_input_state_bhpr);

        let d_y_bhLMp: Tensor<B, 4> = d_y_bnhLMp
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);

        // ── BLUE backward (identical to double-ssd form) ─────────────────
        let exp_da_cumsum_bhLM: Tensor<B, 3> = da_cumsum_bhLM.clone().exp();
        let exp_da_cumsum_bhLMp: Tensor<B, 4> = exp_da_cumsum_bhLM
            .clone()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, chunk_len * mimo_rank, per_head_dim]);
        let d_ch_bhLMp: Tensor<B, 4> = d_y_bhLMp.clone() * exp_da_cumsum_bhLMp.clone();
        san(&d_ch_bhLMp);

        let d_chunk_input_state_bhpr: Tensor<B, 4> = c_bhLMr
            .clone()
            .permute([0, 1, 3, 2]) // c_bhrLM
            .matmul(d_ch_bhLMp.clone()) // bhrp
            .permute([0, 1, 3, 2]); // bhpr
        san(&d_chunk_input_state_bhpr);

        let d_c_blue_bhLMr: Tensor<B, 4> =
            d_ch_bhLMp.clone().matmul(chunk_input_state_bhpr.clone());
        vec_blue_d_c_bhLMr.push(d_c_blue_bhLMr);

        let ch_bhLMp: Tensor<B, 4> = c_bhLMr
            .clone()
            .matmul(chunk_input_state_bhpr.clone().permute([0, 1, 3, 2]));
        let d_da_blue_bhLM: Tensor<B, 3> = (d_y_bhLMp.clone() * ch_bhLMp * exp_da_cumsum_bhLMp)
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        let d_da_blue_bhl: Tensor<B, 3> = d_da_blue_bhLM
            .reshape([batch, nheads, chunk_len, mimo_rank])
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        vec_blue_d_da_bhl.push(d_da_blue_bhl);

        // ── LOWER backward (strict lower-tri + per-column scale) ────────
        let da_target_bhLMLM: Tensor<B, 4> = da_cumsum_bhLM.clone().unsqueeze_dim::<4>(3).expand([
            batch,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank,
        ]);
        let da_source_bhLMLM: Tensor<B, 4> = da_cumsum_bhLM.unsqueeze_dim::<4>(2).expand([
            batch,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank,
        ]);
        let diff_bhLMLM = da_target_bhLMLM - da_source_bhLMLM;

        // Strict-lower MIMO mask: -inf where s_time ≥ t_time — interleaved
        // expansion of the [l, l] strict-upper (triu(0)) base mask.
        let neg_inf_mimo_bhLMLM: Tensor<B, 4> = neg_inf_strict_ll
            .clone()
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([batch, nheads, chunk_len, chunk_len])
            .unsqueeze_dim::<5>(3)
            .expand([batch, nheads, chunk_len, mimo_rank, chunk_len])
            .reshape([batch, nheads, chunk_len * mimo_rank, chunk_len])
            .unsqueeze_dim::<5>(4)
            .expand([batch, nheads, chunk_len * mimo_rank, chunk_len, mimo_rank])
            .reshape([batch, nheads, chunk_len * mimo_rank, chunk_len * mimo_rank]);
        let decay_strict_bhLMLM = (diff_bhLMLM + neg_inf_mimo_bhLMLM).exp();
        san(&decay_strict_bhLMLM);

        let scale_col_bhLMLM: Tensor<B, 4> = scale_bhLM
            .unsqueeze_dim::<4>(2) // [b,h,1,LMs]
            .expand([batch, nheads, chunk_len * mimo_rank, chunk_len * mimo_rank]);

        // w = cb · decay_strict · scale_col
        let prod_bhLMLM = cb_bhLMLM.clone() * decay_strict_bhLMLM.clone();
        let w_bhLMLM = prod_bhLMLM.clone() * scale_col_bhLMLM.clone();

        // d_w = d_y · vᵀ
        let d_w_bhLMLM: Tensor<B, 4> = d_y_bhLMp
            .clone()
            .matmul(v_bhLMp.clone().permute([0, 1, 3, 2]));
        san(&d_w_bhLMLM);

        // d_v_lower = wᵀ · d_y
        let d_v_lower_bhLMp: Tensor<B, 4> =
            w_bhLMLM.permute([0, 1, 3, 2]).matmul(d_y_bhLMp.clone());
        san(&d_v_lower_bhLMp);
        vec_lower_d_v_bhLMp.push(d_v_lower_bhLMp);

        // d_prod = d_w · scale_col ; d_scale_at = d_w · prod
        let d_prod_bhLMLM = d_w_bhLMLM.clone() * scale_col_bhLMLM;
        let d_scale_at_bhLMLM = d_w_bhLMLM * prod_bhLMLM;

        // d_cb_lower = d_prod · decay_strict
        let d_cb_lower_bhLMLM = d_prod_bhLMLM.clone() * decay_strict_bhLMLM.clone();
        vec_d_cb_bhLMLM.push(d_cb_lower_bhLMLM);

        // d_decay_strict = d_prod · cb ; d_diff = d_decay_strict · decay_strict
        let d_decay_strict_bhLMLM = d_prod_bhLMLM * cb_bhLMLM;
        let d_diff_bhLMLM = d_decay_strict_bhLMLM * decay_strict_bhLMLM;

        let d_da_target_bhLM: Tensor<B, 3> = d_diff_bhLMLM.clone().sum_dim(3).squeeze_dim::<3>(3);
        let d_da_source_bhLM: Tensor<B, 3> = d_diff_bhLMLM.sum_dim(2).squeeze_dim::<3>(2);
        let d_da_lower_bhLM = d_da_target_bhLM - d_da_source_bhLM;
        let d_da_lower_bhl: Tensor<B, 3> = d_da_lower_bhLM
            .reshape([batch, nheads, chunk_len, mimo_rank])
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        vec_lower_d_da_bhl.push(d_da_lower_bhl);

        // d_scale[s_time] = Σ_{LMt, s_m} d_scale_at[LMt, LMs]
        let d_scale_lower_bhl: Tensor<B, 3> = d_scale_at_bhLMLM
            .sum_dim(2) // sum over target LMt → [b,h,1,LMs]
            .squeeze_dim::<3>(2) // [b,h,LMs]
            .reshape([batch, nheads, chunk_len, mimo_rank])
            .sum_dim(3) // sum over source mimo → [b,h,l,1]
            .squeeze_dim::<3>(3); // [b,h,l]
        vec_lower_d_scale_bhl.push(d_scale_lower_bhl);

        // ── K4 backward step ───────────────────────────────────────────
        vec_d_intra_bhpr.push(d_running_state_bhpr.clone());

        let decay_chunk_bhpr: Tensor<B, 4> = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk])
            .exp()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, per_head_dim, state_rank]);
        san(&decay_chunk_bhpr);

        let d_decay_chunk_bhpr = d_running_state_bhpr.clone() * chunk_input_state_bhpr;
        let d_da_chunk_end_bh: Tensor<B, 2> = (d_decay_chunk_bhpr * decay_chunk_bhpr.clone())
            .reshape([batch, nheads, per_head_dim * state_rank])
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        vec_d_da_end_bh.push(d_da_chunk_end_bh);

        d_running_state_bhpr = decay_chunk_bhpr * d_running_state_bhpr + d_chunk_input_state_bhpr;
        san(&d_running_state_bhpr);
    }
    let d_initial_state_bhpr = d_running_state_bhpr;

    // ── Restore natural (forward) chunk order ─────────────────────────────
    vec_lower_d_v_bhLMp.reverse();
    vec_blue_d_c_bhLMr.reverse();
    vec_d_cb_bhLMLM.reverse();
    vec_blue_d_da_bhl.reverse();
    vec_lower_d_da_bhl.reverse();
    vec_lower_d_scale_bhl.reverse();
    vec_d_intra_bhpr.reverse();
    vec_d_da_end_bh.reverse();

    // ── Stack per-chunk slices back into batched tensors ──────────────────
    let d_v_lower_bnhLMp: Tensor<B, 5> = Tensor::stack(vec_lower_d_v_bhLMp, 1);
    let d_c_blue_bnhLMr: Tensor<B, 5> = Tensor::stack(vec_blue_d_c_bhLMr, 1);
    let d_cb_bnhLMLM: Tensor<B, 5> = Tensor::stack(vec_d_cb_bhLMLM, 1);
    let d_da_blue_bhnl: Tensor<B, 4> = Tensor::stack(vec_blue_d_da_bhl, 2);
    let d_da_lower_bhnl: Tensor<B, 4> = Tensor::stack(vec_lower_d_da_bhl, 2);
    let d_scale_lower_bhnl: Tensor<B, 4> = Tensor::stack(vec_lower_d_scale_bhl, 2);
    let d_intra_chunk_state_bnhpr: Tensor<B, 5> = Tensor::stack(vec_d_intra_bhpr, 1);
    let d_da_end_bhn: Tensor<B, 3> = Tensor::stack(vec_d_da_end_bh, 2);
    let d_da_cumsum_k4_bhnl: Tensor<B, 4> = {
        let zeros = Tensor::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device);
        let d_da_end_bhn1 = d_da_end_bhn.unsqueeze_dim::<4>(3);
        Tensor::cat(vec![zeros, d_da_end_bhn1], 3)
    };

    // ═══════════════════════════════════════════════════════════════════════
    // K3 BACKWARD (batched) — K_scaled = scaleₜ·B
    //
    // intra_state = decayed_vᵀ @ K_scaled, with decayed_v = decay·V.
    // d_K_scaled = decayed_vᵀ-contraction ; then split into d_b_k3 (·scale) and
    // d_scale_k3 (Σ_{m,r} ·B).
    // ═══════════════════════════════════════════════════════════════════════
    let v_bnLMhp =
        v_bnlmhp
            .clone()
            .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let k_scaled_bnLMhr =
        k_scaled_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
    let k_scaled_bnhLMr = k_scaled_bnLMhr.permute([0, 1, 3, 2, 4]);
    let decayed_v_bnhpLM = k3_decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);

    let d_decayed_v_bnhpLM: Tensor<B, 5> = d_intra_chunk_state_bnhpr
        .clone()
        .matmul(k_scaled_bnhLMr.clone().permute([0, 1, 2, 4, 3])); // k_scaled_bnhrLM
    let d_k_scaled_bnhLMr: Tensor<B, 5> = decayed_v_bnhpLM
        .permute([0, 1, 2, 4, 3]) // decayed_v_bnhLMp
        .matmul(d_intra_chunk_state_bnhpr);

    let d_decayed_v_bnLMhp = d_decayed_v_bnhpLM.permute([0, 1, 4, 2, 3]);
    let d_decay_bhnLM: Tensor<B, 4> = (d_decayed_v_bnLMhp.clone() * v_bnLMhp)
        .sum_dim(4)
        .squeeze_dim::<4>(4)
        .permute([0, 3, 1, 2]);

    let k3_decay_bnLMh1 = k3_decay_bhnLM
        .clone()
        .permute([0, 2, 3, 1])
        .unsqueeze_dim::<5>(4);
    let d_v_k3_bnLMhp: Tensor<B, 5> = d_decayed_v_bnLMhp * k3_decay_bnLMh1;
    let d_v_k3_bnlmhp: Tensor<B, 6> =
        d_v_k3_bnLMhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

    // d(cumA_last − cumA) = d_decay · decay
    let d_decay_times_decay_bhnLM = d_decay_bhnLM * k3_decay_bhnLM;
    let d_a_cumsum_last_bhn: Tensor<B, 3> = d_decay_times_decay_bhnLM
        .clone()
        .sum_dim(3)
        .squeeze_dim::<3>(3);
    let d_da_cumsum_bhnLM = -d_decay_times_decay_bhnLM;

    let d_da_cumsum_k3_from_fused_bhnl: Tensor<B, 4> = d_da_cumsum_bhnLM
        .reshape([batch, nheads, nchunks, chunk_len, mimo_rank])
        .sum_dim(4)
        .squeeze_dim::<4>(4);
    let d_da_cumsum_k3_from_last_bhnl: Tensor<B, 4> = {
        let zeros = Tensor::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device);
        let d_last = d_a_cumsum_last_bhn.unsqueeze_dim::<4>(3);
        Tensor::cat(vec![zeros, d_last], 3)
    };
    let d_da_cumsum_k3_bhnl = d_da_cumsum_k3_from_fused_bhnl + d_da_cumsum_k3_from_last_bhnl;

    // d_K_scaled → bnlmhr, then split into d_b_k3 (·scale) and d_scale_k3 (Σ·B).
    let d_k_scaled_bnlmhr: Tensor<B, 6> = d_k_scaled_bnhLMr
        .permute([0, 1, 3, 2, 4]) // bnLMhr
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
    let d_b_k3_bnlmhr: Tensor<B, 6> = d_k_scaled_bnlmhr.clone() * scale_bnlh11;
    let d_scale_k3_bnlh: Tensor<B, 4> = (d_k_scaled_bnlmhr * b_bnlmhr.clone())
        .sum_dim(5) // sum over state_rank → [b,n,l,m,h,1]
        .squeeze_dim::<5>(5) // [b,n,l,m,h]
        .sum_dim(3) // sum over mimo_rank → [b,n,l,1,h]
        .squeeze_dim::<4>(3); // [b,n,l,h]

    // ═══════════════════════════════════════════════════════════════════════
    // K2 BACKWARD (batched) — cb = C @ Bᵀ
    // ═══════════════════════════════════════════════════════════════════════
    let b_bnLMhr =
        b_bnlmhr
            .clone()
            .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
    let c_bnhLMr = c_bnlmhr
        .clone()
        .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank])
        .permute([0, 1, 3, 2, 4]);
    let b_for_k2_bnhLMr = b_bnLMhr.permute([0, 1, 3, 2, 4]);

    let d_c_k2_bnhLMr: Tensor<B, 5> = d_cb_bnhLMLM.clone().matmul(b_for_k2_bnhLMr);
    let d_b_k2_bnhrLM: Tensor<B, 5> = c_bnhLMr.permute([0, 1, 2, 4, 3]).matmul(d_cb_bnhLMLM);

    let d_c_k2_bnlmhr: Tensor<B, 6> = d_c_k2_bnhLMr
        .permute([0, 1, 3, 2, 4])
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
    let d_b_k2_bnlmhr: Tensor<B, 6> = d_b_k2_bnhrLM
        .permute([0, 1, 4, 2, 3])
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

    // ── Unstack d_c_blue / d_v_lower and reshape back ─────────────────────
    let d_c_blue_bnlmhr: Tensor<B, 6> = d_c_blue_bnhLMr
        .permute([0, 1, 3, 2, 4])
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
    let d_v_lower_bnlmhp: Tensor<B, 6> = d_v_lower_bnhLMp.permute([0, 1, 3, 2, 4]).reshape([
        batch,
        nchunks,
        chunk_len,
        mimo_rank,
        nheads,
        per_head_dim,
    ]);

    // ═══════════════════════════════════════════════════════════════════════
    // K1 BACKWARD + SUM CONTRIBUTIONS
    // ═══════════════════════════════════════════════════════════════════════
    let d_da_cumsum_bhnl =
        d_da_blue_bhnl + d_da_lower_bhnl + d_da_cumsum_k3_bhnl + d_da_cumsum_k4_bhnl;
    san(&d_da_cumsum_bhnl);

    // K1 inverse: suffix sum.
    let d_da_bhnl = {
        let d_total_bhnl = d_da_cumsum_bhnl
            .clone()
            .sum_dim(3)
            .expand([batch, nheads, nchunks, chunk_len]);
        let prefix_bhnl = d_da_cumsum_bhnl.cumsum(3);
        let zeros_bhn1 = Tensor::<B, 4>::zeros([batch, nheads, nchunks, 1], &device);
        let prefix_shifted_bhnl =
            Tensor::cat(vec![zeros_bhn1, prefix_bhnl.narrow(3, 0, chunk_len - 1)], 3);
        d_total_bhnl - prefix_shifted_bhnl
    };
    let d_da_bnlh = d_da_bhnl.permute([0, 2, 3, 1]);

    // ── Combine per-input gradient contributions ──────────────────────────
    let d_v_bnlmhp = d_v_k3_bnlmhp + d_v_lower_bnlmhp + d_v_diag_bnlmhp;
    let d_b_bnlmhr = d_b_k2_bnlmhr + d_b_k3_bnlmhr + d_b_diag_bnlmhr;
    let d_c_bnlmhr = d_c_k2_bnlmhr + d_c_blue_bnlmhr + d_c_diag_bnlmhr;
    let d_scale_bnlh = d_scale_lower_bhnl.permute([0, 2, 3, 1]) + d_scale_k3_bnlh;

    san(&d_v_bnlmhp);
    san(&d_da_bnlh);
    san(&d_b_bnlmhr);
    san(&d_c_bnlmhr);
    san(&d_gamma_bnlh);
    san(&d_scale_bnlh);
    san(&d_initial_state_bhpr);

    CombinedSingleSsdGrads {
        d_v_bnlmhp,
        d_da_bnlh,
        d_b_bnlmhr,
        d_c_bnlmhr,
        d_gamma_bnlh,
        d_scale_bnlh,
        d_initial_state_bhpr,
    }
}
