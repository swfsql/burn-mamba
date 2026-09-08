//! # Recompute-based gradient math for the Mamba-3 single-SSD
//!
//! The analytic backward of the single-pass MIMO-first scan.  Forward
//! intermediates (K1–K4) are recomputed from the saved leaf inputs, then a
//! reverse per-chunk loop fuses the K5 state-to-output (BLUE), the strict
//! lower-triangular intra-chunk (LOWER), and the K4 state-passing backwards; the
//! γ-weighted same-step (DIAG) term is computed batched (no recurrence, tiny
//! `m × m` tensors).  Because this pathway applies the trapezoid weights
//! internally, it additionally returns `d_gamma` and `d_scale`.  The shared K3
//! extended helper (and K1/K2/K4) are reused from the double-SSD module.
//!
//! Everything operates on backend **primitives** through the rank-tagged `F`
//! wrapper: the custom [`Backward`](burn::backend::autodiff::ops::Backward) node
//! runs with a generic backend `B`, so the high-level `Tensor` is unavailable
//! and the math uses `B`'s `float_*` ops.

#![allow(non_snake_case)]

use crate::mamba3::double_ssd::ssd::serial_recalculated::combined_backward::k3_ssd_chunk_state_extended;
use crate::mamba3::double_ssd::ssd::serial_recalculated::{
    k1_ssd_chunk_cumsum, k2_ssd_bmm, k4_ssd_state_passing,
};
use crate::mamba3::helpers::prim::{read_causal_mask, read_rows, scatter_read_rows};
use crate::mamba3::single_ssd::ssd::serial_recalculated::diag::{
    DiagGrads, y_diag_correction_backward,
};
use crate::mamba3::ssd_path::Mamba3SsdPath;
use burn_stack::utils::fprim::{F, san};
use burn::backend::Backend;
use burn::tensor::s;

/// Per-input gradients produced by [`combined_backward`] for the Single-SSD.
/// Adds `d_gamma_bnlh` and `d_scale_bnlh` over the double-ssd form
/// [`crate::mamba3::double_ssd::ssd::serial_recalculated::combined_backward::CombinedGrads`].
#[non_exhaustive]
pub struct CombinedSingleSsdGrads<B: Backend> {
    /// Gradient of the raw input `v`.
    pub d_v_bnlmhp: F<B, 6>,
    /// Gradient of `Δ·A` (`da`).
    pub d_da_bnlh: F<B, 4>,
    /// Gradient of the input projection `B`.
    pub d_b_bnlmhr: F<B, 6>,
    /// Gradient of the output projection `C`, on the chunk's read axis.
    pub d_c_bntmhr: F<B, 6>,
    /// Gradient of the same-step trapezoid weight `γ`, on the read axis.
    pub d_gamma_bnth: F<B, 4>,
    /// Gradient of the key scale `scale = γ + (1−λ₊₁)·Δ₊₁`.
    pub d_scale_bnlh: F<B, 4>,
    /// Gradient of the initial SSM state.
    pub d_initial_state_bhpr: F<B, 4>,
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
/// - `d_y_bntmhp` — upstream gradient of the SSD output (read axis)
/// - `d_final_bhpr` — upstream gradient of the final SSM state
/// - `v_bnlmhp`, `da_bnlh`, `b_bnlmhr`, `c_bntmhr`, `gamma_bnth`, `scale_bnlh`,
///   `initial_state_bhpr` — the seven saved forward inputs
/// - `read_stride` — folded positions per read row (see
///   [`Mamba3SingleSsdInput::read_stride`](crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput))
/// - `siso_specialization` — the forward's γ-correction branch choice, replayed
///   here so the backward matches it (performance-only; both agree)
///
/// # Returns
/// One [`CombinedSingleSsdGrads`] with gradients for all 7 inputs.
#[allow(clippy::too_many_arguments)]
pub fn combined_backward<B: Backend>(
    d_y_bntmhp: F<B, 6>,
    d_final_bhpr: F<B, 4>,
    //
    v_bnlmhp: F<B, 6>,
    da_bnlh: F<B, 4>,
    b_bnlmhr: F<B, 6>,
    c_bntmhr: F<B, 6>,
    gamma_bnth: F<B, 4>,
    scale_bnlh: F<B, 4>,
    initial_state_bhpr: F<B, 4>,
    read_stride: usize,
    siso_specialization: bool,
) -> CombinedSingleSsdGrads<B> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();
    let device = v_bnlmhp.device();
    let dtype = v_bnlmhp.dtype();
    // The chunk's two axes: `fused` is what it writes, `read` what it is read
    // at. They coincide unless `micro_steps > 1`.
    let chunk_tokens = Mamba3SsdPath::chunk_tokens(chunk_len, read_stride);
    let fused = chunk_len * mimo_rank;
    let read = chunk_tokens * mimo_rank;

    san(&d_y_bntmhp);
    san(&d_final_bhpr);
    san(&v_bnlmhp);
    san(&da_bnlh);
    san(&b_bnlmhr);
    san(&c_bntmhr);
    san(&gamma_bnth);
    san(&scale_bnlh);
    san(&initial_state_bhpr);

    // ═══════════════════════════════════════════════════════════════════════
    // RECOMPUTE FORWARD INTERMEDIATES (K1–K4, single-ssd form)
    // ═══════════════════════════════════════════════════════════════════════

    // K1
    let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum(da_bnlh.clone());
    san(&da_cumsum_bhnl);

    // K2 — CB matrix (unscaled), used by LOWER.
    let cb_bnhTMLM = k2_ssd_bmm(c_bntmhr.clone(), b_bnlmhr.clone());
    san(&cb_bnhTMLM);

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
    let (chunk_input_state_bnhpr, _final_state_bhpr) = k4_ssd_state_passing(
        intra_chunk_state_bnhpr,
        da_chunk_end_bhn.clone(),
        initial_state_bhpr,
    );

    // Fused-position cumulative decay, on the write axis and on the read one.
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .clone()
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
        .reshape([batch, nheads, nchunks, fused]);
    let da_cumsum_bhnTM = read_rows::<B, 4, 5>(da_cumsum_bhnl.clone(), 3, read_stride)
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_tokens, mimo_rank])
        .reshape([batch, nheads, nchunks, read]);

    // d_y in (batch, nchunks, nheads, chunk_tokens·mimo_rank, per_head_dim) ordering.
    let d_y_bnhTMp = d_y_bntmhp
        .clone()
        .reshape([batch, nchunks, read, nheads, per_head_dim])
        .swap_dims(2, 3);
    san(&d_y_bnhTMp);

    // ═══════════════════════════════════════════════════════════════════════
    // DIAG BACKWARD (batched — no recurrence; m × m working set is tiny)
    //
    // Forward (per (b,n,l,h)):
    //   qk_dot[m_out, m_in] = Σ_r C[m_out, r] · B[m_in, r]
    //   y_diag[m_out, p]    = γ · Σ_{m_in} qk_dot[m_out, m_in] · V[m_in, p]
    // ═══════════════════════════════════════════════════════════════════════
    // The same step is the *read* row, so V and B enter at those positions and
    // their gradients are scattered back onto the folded axis.
    let DiagGrads {
        d_v_bnlmhp: d_v_diag_bntmhp,
        d_c_bnlmhr: d_c_diag_bntmhr,
        d_b_bnlmhr: d_b_diag_bntmhr,
        d_gamma_bnlh: d_gamma_bnth,
    } = y_diag_correction_backward(
        d_y_bntmhp.clone(),
        read_rows::<B, 6, 7>(v_bnlmhp.clone(), 2, read_stride),
        read_rows::<B, 6, 7>(b_bnlmhr.clone(), 2, read_stride),
        c_bntmhr.clone(),
        gamma_bnth.clone(),
        siso_specialization,
    );
    let d_v_diag_bnlmhp = scatter_read_rows::<B, 6, 7>(d_v_diag_bntmhp, 2, read_stride);
    let d_b_diag_bnlmhr = scatter_read_rows::<B, 6, 7>(d_b_diag_bntmhr, 2, read_stride);
    san(&d_gamma_bnth);

    // Reusable [chunk_tokens, chunk_len] -inf strict-upper mask (triu(0) on the
    // folded grid: on+above diagonal → -inf) for the LOWER (strict
    // lower-triangular) path.
    let neg_inf_strict_tl: F<B, 2> =
        read_causal_mask::<B>(chunk_len, read_stride, 0, &device, dtype);

    // ═══════════════════════════════════════════════════════════════════════
    // REVERSE PER-CHUNK LOOP — K5 (BLUE + LOWER) + K4 fused
    // ═══════════════════════════════════════════════════════════════════════
    let mut vec_lower_d_v_bhLMp: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_c_bhTMr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_d_cb_bhTMLM: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_da_bhl: Vec<F<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_lower_d_da_bhl: Vec<F<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_lower_d_scale_bhl: Vec<F<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_d_intra_bhpr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_d_da_end_bh: Vec<F<B, 2>> = Vec::with_capacity(nchunks);

    let mut d_running_state_bhpr: F<B, 4> = d_final_bhpr;

    for i_chunk in (0..nchunks).rev() {
        // ── Per-chunk slices (fused chunk_len · mimo_rank) ─────────────
        let v_bhLMp: F<B, 4> = v_bnlmhp
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..])
            .squeeze_dim::<5>(1)
            .reshape([batch, fused, nheads, per_head_dim])
            .swap_dims(1, 2);

        let c_bhTMr: F<B, 4> = c_bntmhr
            .clone()
            .slice(s![.., i_chunk, .., .., .., ..])
            .squeeze_dim::<5>(1)
            .reshape([batch, read, nheads, state_rank])
            .swap_dims(1, 2);

        let cb_bhTMLM: F<B, 4> = cb_bnhTMLM
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);

        let da_cumsum_bhLM: F<B, 3> = da_cumsum_bhnLM
            .clone()
            .slice(s![.., .., i_chunk, ..])
            .squeeze_dim::<3>(2);
        let da_cumsum_bhTM: F<B, 3> = da_cumsum_bhnTM
            .clone()
            .slice(s![.., .., i_chunk, ..])
            .squeeze_dim::<3>(2);

        // scaleₜ per fused source position: scale[s_time] broadcast over s_m.
        let scale_bhLM: F<B, 3> = scale_bnlh
            .clone()
            .slice(s![.., i_chunk, .., ..]) // [b, l, h]
            .squeeze_dim::<3>(1)
            .swap_dims(1, 2) // [b, h, l]
            .unsqueeze_dim::<4>(3) // [b, h, l, 1]
            .expand([batch, nheads, chunk_len, mimo_rank])
            .reshape([batch, nheads, fused]);

        let chunk_input_state_bhpr: F<B, 4> = chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);
        san(&chunk_input_state_bhpr);

        let d_y_bhTMp: F<B, 4> = d_y_bnhTMp
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);

        // ── BLUE backward (identical to double-ssd form) ─────────────────
        let exp_da_cumsum_bhTM: F<B, 3> = da_cumsum_bhTM.clone().exp();
        let exp_da_cumsum_bhTMp: F<B, 4> = exp_da_cumsum_bhTM
            .clone()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, read, per_head_dim]);
        let d_ch_bhTMp: F<B, 4> = d_y_bhTMp.clone() * exp_da_cumsum_bhTMp.clone();
        san(&d_ch_bhTMp);

        let d_chunk_input_state_bhpr: F<B, 4> = c_bhTMr
            .clone()
            .transpose() // c_bhrTM
            .matmul(d_ch_bhTMp.clone()) // bhrp
            .transpose(); // bhpr
        san(&d_chunk_input_state_bhpr);

        let d_c_blue_bhTMr: F<B, 4> = d_ch_bhTMp.clone().matmul(chunk_input_state_bhpr.clone());
        vec_blue_d_c_bhTMr.push(d_c_blue_bhTMr);

        let ch_bhTMp: F<B, 4> = c_bhTMr
            .clone()
            .matmul(chunk_input_state_bhpr.clone().transpose());
        let d_da_blue_bhTM: F<B, 3> = (d_y_bhTMp.clone() * ch_bhTMp * exp_da_cumsum_bhTMp)
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        // The row's own `da` is one of the folded positions, so its gradient
        // goes back to that one and nowhere else.
        let d_da_blue_bhl: F<B, 3> = scatter_read_rows::<B, 3, 4>(
            d_da_blue_bhTM
                .reshape([batch, nheads, chunk_tokens, mimo_rank])
                .sum_dim(3)
                .squeeze_dim::<3>(3),
            2,
            read_stride,
        );
        vec_blue_d_da_bhl.push(d_da_blue_bhl);

        // ── LOWER backward (strict lower-tri + per-column scale) ────────
        let da_target_bhTMLM: F<B, 4> = da_cumsum_bhTM
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, read, fused]);
        let da_source_bhTMLM: F<B, 4> = da_cumsum_bhLM
            .unsqueeze_dim::<4>(2)
            .expand([batch, nheads, read, fused]);
        let diff_bhTMLM = da_target_bhTMLM - da_source_bhTMLM;

        // Strict-lower MIMO mask: -inf where s_time ≥ t_time — interleaved
        // expansion of the [l, l] strict-upper (triu(0)) base mask.
        let neg_inf_mimo_bhTMLM: F<B, 4> = neg_inf_strict_tl
            .clone()
            .unsqueeze_dims::<4>(&[0, 1])
            .expand([batch, nheads, chunk_tokens, chunk_len])
            .unsqueeze_dim::<5>(3)
            .expand([batch, nheads, chunk_tokens, mimo_rank, chunk_len])
            .reshape([batch, nheads, read, chunk_len])
            .unsqueeze_dim::<5>(4)
            .expand([batch, nheads, read, chunk_len, mimo_rank])
            .reshape([batch, nheads, read, fused]);
        let decay_strict_bhTMLM = (diff_bhTMLM + neg_inf_mimo_bhTMLM).exp();
        san(&decay_strict_bhTMLM);

        let scale_col_bhTMLM: F<B, 4> = scale_bhLM
            .unsqueeze_dim::<4>(2) // [b,h,1,LMs]
            .expand([batch, nheads, read, fused]);

        // w = cb · decay_strict · scale_col
        let prod_bhTMLM = cb_bhTMLM.clone() * decay_strict_bhTMLM.clone();
        let w_bhTMLM = prod_bhTMLM.clone() * scale_col_bhTMLM.clone();

        // d_w = d_y · vᵀ
        let d_w_bhTMLM: F<B, 4> = d_y_bhTMp.clone().matmul(v_bhLMp.clone().transpose());
        san(&d_w_bhTMLM);

        // d_v_lower = wᵀ · d_y
        let d_v_lower_bhLMp: F<B, 4> = w_bhTMLM.transpose().matmul(d_y_bhTMp.clone());
        san(&d_v_lower_bhLMp);
        vec_lower_d_v_bhLMp.push(d_v_lower_bhLMp);

        // d_prod = d_w · scale_col ; d_scale_at = d_w · prod
        let d_prod_bhTMLM = d_w_bhTMLM.clone() * scale_col_bhTMLM;
        let d_scale_at_bhTMLM = d_w_bhTMLM * prod_bhTMLM;

        // d_cb_lower = d_prod · decay_strict
        let d_cb_lower_bhTMLM = d_prod_bhTMLM.clone() * decay_strict_bhTMLM.clone();
        vec_d_cb_bhTMLM.push(d_cb_lower_bhTMLM);

        // d_decay_strict = d_prod · cb ; d_diff = d_decay_strict · decay_strict
        let d_decay_strict_bhTMLM = d_prod_bhTMLM * cb_bhTMLM;
        let d_diff_bhTMLM = d_decay_strict_bhTMLM * decay_strict_bhTMLM;

        // The target term lands on the read rows (scattered back onto the
        // folded axis), the source term on every folded position.
        let d_da_target_bhTM: F<B, 3> = d_diff_bhTMLM.clone().sum_dim(3).squeeze_dim::<3>(3);
        let d_da_source_bhLM: F<B, 3> = d_diff_bhTMLM.sum_dim(2).squeeze_dim::<3>(2);
        let d_da_target_bhl: F<B, 3> = scatter_read_rows::<B, 3, 4>(
            d_da_target_bhTM
                .reshape([batch, nheads, chunk_tokens, mimo_rank])
                .sum_dim(3)
                .squeeze_dim::<3>(3),
            2,
            read_stride,
        );
        let d_da_source_bhl: F<B, 3> = d_da_source_bhLM
            .reshape([batch, nheads, chunk_len, mimo_rank])
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        vec_lower_d_da_bhl.push(d_da_target_bhl - d_da_source_bhl);

        // d_scale[s_time] = Σ_{TMt, s_m} d_scale_at[TMt, LMs]
        let d_scale_lower_bhl: F<B, 3> = d_scale_at_bhTMLM
            .sum_dim(2) // sum over target TMt → [b,h,1,LMs]
            .squeeze_dim::<3>(2) // [b,h,LMs]
            .reshape([batch, nheads, chunk_len, mimo_rank])
            .sum_dim(3) // sum over source mimo → [b,h,l,1]
            .squeeze_dim::<3>(3); // [b,h,l]
        vec_lower_d_scale_bhl.push(d_scale_lower_bhl);

        // ── K4 backward step ───────────────────────────────────────────
        vec_d_intra_bhpr.push(d_running_state_bhpr.clone());

        let decay_chunk_bhpr: F<B, 4> = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk])
            .exp()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, per_head_dim, state_rank]);
        san(&decay_chunk_bhpr);

        let d_decay_chunk_bhpr = d_running_state_bhpr.clone() * chunk_input_state_bhpr;
        let d_da_chunk_end_bh: F<B, 2> = (d_decay_chunk_bhpr * decay_chunk_bhpr.clone())
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
    vec_blue_d_c_bhTMr.reverse();
    vec_d_cb_bhTMLM.reverse();
    vec_blue_d_da_bhl.reverse();
    vec_lower_d_da_bhl.reverse();
    vec_lower_d_scale_bhl.reverse();
    vec_d_intra_bhpr.reverse();
    vec_d_da_end_bh.reverse();

    // ── Stack per-chunk slices back into batched tensors ──────────────────
    let d_v_lower_bnhLMp: F<B, 5> = F::stack(vec_lower_d_v_bhLMp, 1);
    let d_c_blue_bnhTMr: F<B, 5> = F::stack(vec_blue_d_c_bhTMr, 1);
    let d_cb_bnhTMLM: F<B, 5> = F::stack(vec_d_cb_bhTMLM, 1);
    let d_da_blue_bhnl: F<B, 4> = F::stack(vec_blue_d_da_bhl, 2);
    let d_da_lower_bhnl: F<B, 4> = F::stack(vec_lower_d_da_bhl, 2);
    let d_scale_lower_bhnl: F<B, 4> = F::stack(vec_lower_d_scale_bhl, 2);
    let d_intra_chunk_state_bnhpr: F<B, 5> = F::stack(vec_d_intra_bhpr, 1);
    let d_da_end_bhn: F<B, 3> = F::stack(vec_d_da_end_bh, 2);
    let d_da_cumsum_k4_bhnl: F<B, 4> = {
        let zeros = F::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device, dtype);
        let d_da_end_bhn1 = d_da_end_bhn.unsqueeze_dim::<4>(3);
        F::cat(vec![zeros, d_da_end_bhn1], 3)
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
    let k_scaled_bnhLMr = k_scaled_bnLMhr.swap_dims(2, 3);
    let decayed_v_bnhpLM = k3_decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);

    let d_decayed_v_bnhpLM: F<B, 5> = d_intra_chunk_state_bnhpr
        .clone()
        .matmul(k_scaled_bnhLMr.clone().transpose()); // k_scaled_bnhrLM
    let d_k_scaled_bnhLMr: F<B, 5> = decayed_v_bnhpLM
        .transpose() // decayed_v_bnhLMp
        .matmul(d_intra_chunk_state_bnhpr);

    let d_decayed_v_bnLMhp = d_decayed_v_bnhpLM.permute([0, 1, 4, 2, 3]);
    let d_decay_bhnLM: F<B, 4> = (d_decayed_v_bnLMhp.clone() * v_bnLMhp)
        .sum_dim(4)
        .squeeze_dim::<4>(4)
        .permute([0, 3, 1, 2]);

    let k3_decay_bnLMh1 = k3_decay_bhnLM
        .clone()
        .permute([0, 2, 3, 1])
        .unsqueeze_dim::<5>(4);
    let d_v_k3_bnLMhp: F<B, 5> = d_decayed_v_bnLMhp * k3_decay_bnLMh1;
    let d_v_k3_bnlmhp: F<B, 6> =
        d_v_k3_bnLMhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

    // d(cumA_last − cumA) = d_decay · decay
    let d_decay_times_decay_bhnLM = d_decay_bhnLM * k3_decay_bhnLM;
    let d_a_cumsum_last_bhn: F<B, 3> = d_decay_times_decay_bhnLM
        .clone()
        .sum_dim(3)
        .squeeze_dim::<3>(3);
    let d_da_cumsum_bhnLM = -d_decay_times_decay_bhnLM;

    let d_da_cumsum_k3_from_fused_bhnl: F<B, 4> = d_da_cumsum_bhnLM
        .reshape([batch, nheads, nchunks, chunk_len, mimo_rank])
        .sum_dim(4)
        .squeeze_dim::<4>(4);
    let d_da_cumsum_k3_from_last_bhnl: F<B, 4> = {
        let zeros = F::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device, dtype);
        let d_last = d_a_cumsum_last_bhn.unsqueeze_dim::<4>(3);
        F::cat(vec![zeros, d_last], 3)
    };
    let d_da_cumsum_k3_bhnl = d_da_cumsum_k3_from_fused_bhnl + d_da_cumsum_k3_from_last_bhnl;

    // d_K_scaled → bnlmhr, then split into d_b_k3 (·scale) and d_scale_k3 (Σ·B).
    let d_k_scaled_bnlmhr: F<B, 6> = d_k_scaled_bnhLMr
        .swap_dims(2, 3) // bnLMhr
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
    let d_b_k3_bnlmhr: F<B, 6> = d_k_scaled_bnlmhr.clone() * scale_bnlh11;
    let d_scale_k3_bnlh: F<B, 4> = (d_k_scaled_bnlmhr * b_bnlmhr.clone())
        .sum_dim(5) // sum over state_rank → [b,n,l,m,h,1]
        .squeeze_dim::<5>(5) // [b,n,l,m,h]
        .sum_dim(3) // sum over mimo_rank → [b,n,l,1,h]
        .squeeze_dim::<4>(3); // [b,n,l,h]

    // ═══════════════════════════════════════════════════════════════════════
    // K2 BACKWARD (batched) — cb = C @ Bᵀ
    // ═══════════════════════════════════════════════════════════════════════
    let b_bnLMhr = b_bnlmhr
        .clone()
        .reshape([batch, nchunks, fused, nheads, state_rank]);
    let c_bnhTMr = c_bntmhr
        .clone()
        .reshape([batch, nchunks, read, nheads, state_rank])
        .swap_dims(2, 3);
    let b_for_k2_bnhLMr = b_bnLMhr.swap_dims(2, 3);

    let d_c_k2_bnhTMr: F<B, 5> = d_cb_bnhTMLM.clone().matmul(b_for_k2_bnhLMr);
    let d_b_k2_bnhrLM: F<B, 5> = c_bnhTMr.transpose().matmul(d_cb_bnhTMLM);

    let d_c_k2_bntmhr: F<B, 6> = d_c_k2_bnhTMr
        .swap_dims(2, 3)
        .reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]);
    let d_b_k2_bnlmhr: F<B, 6> = d_b_k2_bnhrLM
        .permute([0, 1, 4, 2, 3])
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

    // ── Unstack d_c_blue / d_v_lower and reshape back ─────────────────────
    let d_c_blue_bntmhr: F<B, 6> = d_c_blue_bnhTMr
        .swap_dims(2, 3)
        .reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]);
    let d_v_lower_bnlmhp: F<B, 6> = d_v_lower_bnhLMp.swap_dims(2, 3).reshape([
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
        let zeros_bhn1 = F::<B, 4>::zeros([batch, nheads, nchunks, 1], &device, dtype);
        let prefix_shifted_bhnl =
            F::cat(vec![zeros_bhn1, prefix_bhnl.narrow(3, 0, chunk_len - 1)], 3);
        d_total_bhnl - prefix_shifted_bhnl
    };
    let d_da_bnlh = d_da_bhnl.permute([0, 2, 3, 1]);

    // ── Combine per-input gradient contributions ──────────────────────────
    let d_v_bnlmhp = d_v_k3_bnlmhp + d_v_lower_bnlmhp + d_v_diag_bnlmhp;
    let d_b_bnlmhr = d_b_k2_bnlmhr + d_b_k3_bnlmhr + d_b_diag_bnlmhr;
    let d_c_bntmhr = d_c_k2_bntmhr + d_c_blue_bntmhr + d_c_diag_bntmhr;
    let d_scale_bnlh = d_scale_lower_bhnl.permute([0, 2, 3, 1]) + d_scale_k3_bnlh;

    san(&d_v_bnlmhp);
    san(&d_da_bnlh);
    san(&d_b_bnlmhr);
    san(&d_c_bntmhr);
    san(&d_gamma_bnth);
    san(&d_scale_bnlh);
    san(&d_initial_state_bhpr);

    CombinedSingleSsdGrads {
        d_v_bnlmhp,
        d_da_bnlh,
        d_b_bnlmhr,
        d_c_bntmhr,
        d_gamma_bnth,
        d_scale_bnlh,
        d_initial_state_bhpr,
    }
}
