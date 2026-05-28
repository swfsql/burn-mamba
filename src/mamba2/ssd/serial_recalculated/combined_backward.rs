//! # Recompute-based gradient math for the Mamba-2 SSD
//!
//! The analytic backward of the five-kernel serial scan, mirroring
//! `_mamba_chunk_scan_combined_bwd` in the reference `ssd_combined.py`.  The
//! forward intermediates (K1–K4) are **recomputed** from the saved leaf inputs
//! rather than stashed, then a reverse per-chunk loop fuses the K5 and K4
//! backwards; K1/K2/K3 backwards run as batched ops once the loop has gathered
//! the per-chunk slices.  Comment colours (BLUE / ORANGE / …) tag the
//! corresponding terms of the chunk-scan gradient, matching the reference.
//!
//! Everything here operates on backend **primitives** through the rank-tagged
//! [`F`] wrapper: a custom [`Backward`](burn::backend::autodiff::ops::Backward)
//! node runs with a generic backend `B`, so the high-level `Tensor` (pinned to
//! the global `Dispatch` backend) is unavailable and the math must use `B`'s
//! `float_*` ops.  The recomputed K1/K2/K4 kernels are local primitive ports of
//! the high-level [`crate::mamba2::ssd::serial`] kernels.

#![allow(non_snake_case)]

use super::serial_recalculated::{k1_ssd_chunk_cumsum, k2_ssd_bmm, k4_ssd_state_passing};
use crate::utils::fprim::{F, Mask, san};
use burn::backend::Backend;
use burn::tensor::s;

/// Per-input gradients produced by [`combined_backward`] (one field per
/// differentiable forward input).
#[non_exhaustive]
pub struct CombinedGrads<B: Backend> {
    /// Gradient of the input `x`.
    pub d_x_bnlhp: F<B, 5>,
    /// Gradient of the discretised step `Δ` (`dt`).
    pub d_dt_discretized_bhnl: F<B, 4>,
    /// Gradient of the input projection `B`.
    pub d_b_bnlhr: F<B, 5>,
    /// Gradient of the output projection `C`.
    pub d_c_bnlhr: F<B, 5>,
    /// Gradient of the per-head skip term `D`.
    pub d_d_h: F<B, 1>,
    /// Gradient of the initial SSM state.
    pub d_initial_state_bhpr: F<B, 4>,
    /// Gradient of the per-head decay rate `A` (as `a_decay_h`).
    pub d_a_decay_h: F<B, 1>,
    /// Local same-chunk contribution to `d_da_cumsum` from BLUE+ORANGE only
    /// (excludes K3 and K4 cross-chunk contributions). Exposed for the
    /// `out_x · dout − ddt · dt` oracle test from Tri Dao's reference
    /// (`_chunk_scan_bwd_ddAcs_unstable`). Test-only; absent in release builds.
    #[cfg(test)]
    pub d_da_local_bhnl: F<B, 4>,
    /// Same-chunk d_dt contribution from ORANGE only (= what Tri Dao calls
    /// `ddt` in `_chunk_scan_bwd_ddAcs_unstable`). Test-only; absent in release
    /// builds.
    #[cfg(test)]
    pub d_dt_orange_bhnl: F<B, 4>,
}

// ─── Recomputed forward kernels ──────────────────────────────────────────────
// The recompute backward replays the forward's K1/K2/K4 (imported above from
// [`super::serial_recalculated`]) plus the extended K3 below, which returns the
// extra intermediates the gradient math needs.

/// Same as [`k3_ssd_chunk_state`](super::serial_recalculated::k3_ssd_chunk_state)
/// but also returns intermediates needed by the custom backward:
/// - `intra_chunk_state_bnhpr` — chunk-end state assuming zero initial state
/// - `b_bar_scale_bhnl` — the K3 scaling factor `dt · exp(cumA_last − cumA)`
/// - `forward_decay_to_chunk_end_bhnl` — the decay factor `exp(cumA_last − cumA)`
/// - `b_scaled_bnhlr` — B already scaled by `b_bar_scale`
pub fn k3_ssd_chunk_state_extended<B: Backend>(
    x_bnlhp: F<B, 5>,
    b_bnlhr: F<B, 5>,
    da_cumsum_bhnl: F<B, 4>,
    dt_discretized_bhnl: F<B, 4>,
) -> (F<B, 5>, F<B, 4>, F<B, 4>, F<B, 5>) {
    let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
    let [.., state_rank] = b_bnlhr.dims();

    let x_bnhpl = x_bnlhp.permute([0, 1, 3, 4, 2]);
    let b_bnhlr = b_bnlhr.permute([0, 1, 3, 2, 4]);

    // K3 scaling factor: dt · exp(cumA_last − cumA)
    let da_cumsum_last_bhn1 = da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
    let forward_decay_to_chunk_end_bhnl =
        (da_cumsum_last_bhn1.expand([batch, nheads, nchunks, chunk_len]) - da_cumsum_bhnl).exp();
    san(&forward_decay_to_chunk_end_bhnl);
    let b_bar_scale_bhnl = forward_decay_to_chunk_end_bhnl.clone() * dt_discretized_bhnl;
    san(&b_bar_scale_bhnl);

    let b_bar_scale_bnhlr = b_bar_scale_bhnl
        .clone()
        .permute([0, 2, 1, 3]) // b_bar_scale_bnhl
        .unsqueeze_dim::<5>(4) // b_bar_scale_bnhl1
        .expand([batch, nchunks, nheads, chunk_len, state_rank]); // b_bar_scale_bnhlr
    let b_scaled_bnhlr = b_bnhlr * b_bar_scale_bnhlr;
    san(&b_scaled_bnhlr);

    let intra_chunk_state_bnhpr = x_bnhpl.matmul(b_scaled_bnhlr.clone());
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

/// Memory-efficient backward for the Mamba-2 chunkwise SSD.
///
/// Recomputes the forward intermediates (K1-K4) from the saved inputs, then
/// runs a reverse per-chunk loop that fuses the K5 (BLUE + ORANGE) backward
/// with the K4 state-passing backward. K3/K2/K1 backwards run as single
/// batched ops once the loop has collected all per-chunk slices.
///
/// # Arguments
/// - `d_y_bnlhp` — upstream gradient of the SSD output
/// - `d_final_bhpr` — upstream gradient of the final SSM state
/// - `x_bnlhp`, `dt_discretized_bhnl`, `b_bnlhr`, `c_bnlhr`, `d_h`,
///   `initial_state_bhpr`, `a_decay_h` — the seven saved forward inputs
///
/// # Returns
/// One [`CombinedGrads`] struct containing gradients for all 7 inputs.
#[allow(clippy::too_many_arguments)]
pub fn combined_backward<B: Backend>(
    d_y_bnlhp: F<B, 5>,
    d_final_bhpr: F<B, 4>,
    //
    x_bnlhp: F<B, 5>,
    dt_discretized_bhnl: F<B, 4>,
    b_bnlhr: F<B, 5>,
    c_bnlhr: F<B, 5>,
    d_h: F<B, 1>,
    initial_state_bhpr: F<B, 4>,
    a_decay_h: F<B, 1>,
) -> CombinedGrads<B> {
    let [batch, nheads, nchunks, chunk_len] = dt_discretized_bhnl.dims();
    let [.., per_head_dim] = x_bnlhp.dims();
    let [.., state_rank] = b_bnlhr.dims();
    let device = dt_discretized_bhnl.device();
    let dtype = dt_discretized_bhnl.dtype();

    san(&d_y_bnlhp);
    san(&d_final_bhpr);
    san(&x_bnlhp);
    san(&dt_discretized_bhnl);
    san(&b_bnlhr);
    san(&c_bnlhr);
    san(&d_h);
    san(&initial_state_bhpr);
    san(&a_decay_h);

    // ═══════════════════════════════════════════════════════════════════════
    // RECOMPUTE FORWARD INTERMEDIATES (the memory-saving heart of this op)
    // ═══════════════════════════════════════════════════════════════════════

    // K1 — pre-combined Δ·A → intra-chunk cumsum
    let (da_cumsum_bhnl, da_chunk_end_bhn) =
        k1_ssd_chunk_cumsum(dt_discretized_bhnl.clone(), a_decay_h.clone());
    san(&da_cumsum_bhnl);

    // K2 — CB matrix used in K5 ORANGE
    let cb_bnhll = k2_ssd_bmm(c_bnlhr.clone(), b_bnlhr.clone());
    san(&cb_bnhll);

    // K3 — intra-chunk state + decay/decayed-B intermediates
    let (
        intra_chunk_state_bnhpr,
        b_bar_scale_bhnl,
        forward_decay_to_chunk_end_bhnl,
        b_scaled_bnhlr,
    ) = k3_ssd_chunk_state_extended(
        x_bnlhp.clone(),
        b_bnlhr.clone(),
        da_cumsum_bhnl.clone(),
        dt_discretized_bhnl.clone(),
    );

    // K4 — chunk-input state stream consumed by K5 BLUE
    let (chunk_input_state_bnhpr, _final_state_bhpr) = k4_ssd_state_passing(
        intra_chunk_state_bnhpr.clone(),
        da_chunk_end_bhn.clone(),
        initial_state_bhpr,
    );
    san(&chunk_input_state_bnhpr);

    // ═══════════════════════════════════════════════════════════════════════
    // SKIP backward — y += D · x
    // ═══════════════════════════════════════════════════════════════════════
    let d_d_h = (d_y_bnlhp.clone() * x_bnlhp.clone())
        .permute([3, 0, 1, 2, 4]) // _hbnlp
        .reshape([nheads, batch * nchunks * chunk_len * per_head_dim]) // _hBNLP
        .sum_dim(1) // _h1
        .reshape([nheads]);
    san(&d_d_h);

    let d_x_skip_bnlhp = d_y_bnlhp.clone()
        * d_h
            .clone()
            .unsqueeze_dims::<5>(&[0, 1, 2, 4]) // _111h1
            .expand([batch, nchunks, chunk_len, nheads, per_head_dim]); // _bnlhp
    san(&d_x_skip_bnlhp);

    // ═══════════════════════════════════════════════════════════════════════
    // REVERSE PER-CHUNK LOOP — K5 (BLUE + ORANGE) + K4 fused
    //
    // Per-iteration working set is _bhll (not _bnhll).
    // ═══════════════════════════════════════════════════════════════════════

    // Reusable [chunk_len, chunk_len] upper-triangle base mask for ORANGE.
    let causal_mask_ll: Mask<B> = Mask::tril_mask(chunk_len, chunk_len, 0, &device);

    let mut vec_orange_d_x_bhlp: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_orange_d_dt_bhl: Vec<F<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_orange_d_da_bhl: Vec<F<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_d_cb_bhll: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_c_bhlr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_blue_d_da_bhl: Vec<F<B, 3>> = Vec::with_capacity(nchunks);
    let mut vec_d_intra_bhpr: Vec<F<B, 4>> = Vec::with_capacity(nchunks);
    let mut vec_d_da_end_bh: Vec<F<B, 2>> = Vec::with_capacity(nchunks);

    let mut d_running_state_bhpr: F<B, 4> = d_final_bhpr;

    for i_chunk in (0..nchunks).rev() {
        // ── Per-chunk slices ───────────────────────────────────────────────
        let da_cumsum_bhl: F<B, 3> = da_cumsum_bhnl
            .clone()
            .slice(s![.., .., i_chunk, ..]) // _bh1l
            .squeeze_dim::<3>(2); // _bhl
        let dt_bhl: F<B, 3> = dt_discretized_bhnl
            .clone()
            .slice(s![.., .., i_chunk, ..]) // _bh1l
            .squeeze_dim::<3>(2); // _bhl
        let x_bhlp: F<B, 4> = x_bnlhp
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // _b1lhp
            .squeeze_dim::<4>(1) // _blhp
            .permute([0, 2, 1, 3]); // _bhlp
        let d_y_bhlp: F<B, 4> = d_y_bnlhp
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // _b1lhp
            .squeeze_dim::<4>(1) // _blhp
            .permute([0, 2, 1, 3]); // _bhlp
        let c_bhlr: F<B, 4> = c_bnlhr
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // _b1lhr
            .squeeze_dim::<4>(1) // _blhr
            .permute([0, 2, 1, 3]); // _bhlr
        let cb_bhll: F<B, 4> = cb_bnhll
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // _b1hll
            .squeeze_dim::<4>(1); // _bhll
        let chunk_input_state_bhpr: F<B, 4> = chunk_input_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..]) // _b1hpr
            .squeeze_dim::<4>(1); // _bhpr
        san(&chunk_input_state_bhpr);

        // ── BLUE backward ──────────────────────────────────────────────────
        //   blue[l,p] = exp(cumA[l]) · Σᵣ C[l,r] · state[p,r]
        let exp_da_cumsum_bhl: F<B, 3> = da_cumsum_bhl.clone().exp();
        let exp_da_cumsum_bhlp: F<B, 4> = exp_da_cumsum_bhl
            .clone()
            .unsqueeze_dim::<4>(3) // _bhl1
            .expand([batch, nheads, chunk_len, per_head_dim]); // _bhlp
        let d_ch_bhlp: F<B, 4> = d_y_bhlp.clone() * exp_da_cumsum_bhlp.clone();
        san(&d_ch_bhlp);

        // d_chunk_input_state = C^T @ d_ch
        let d_chunk_input_state_bhpr: F<B, 4> = c_bhlr
            .clone()
            .permute([0, 1, 3, 2]) // c_bhrl
            .matmul(d_ch_bhlp.clone()) // d_chunk_input_state_bhrp
            .permute([0, 1, 3, 2]); // _bhpr
        san(&d_chunk_input_state_bhpr);

        // d_C_blue = d_ch @ state
        let d_c_blue_bhlr: F<B, 4> = d_ch_bhlp.clone().matmul(chunk_input_state_bhpr.clone());
        san(&d_c_blue_bhlr);
        vec_blue_d_c_bhlr.push(d_c_blue_bhlr);

        // d_da from BLUE:  d_da[l] = (Σₚ d_y[l,p] · ch[l,p]) · exp_da[l]
        let ch_bhlp: F<B, 4> = c_bhlr.clone().matmul(
            chunk_input_state_bhpr.clone().permute([0, 1, 3, 2]), // _bhrp
        ); // _bhlp
        let d_da_blue_bhl: F<B, 3> = (d_y_bhlp.clone() * ch_bhlp * exp_da_cumsum_bhlp)
            .sum_dim(3) // _bhl1
            .squeeze_dim::<3>(3); // _bhl
        san(&d_da_blue_bhl);
        vec_blue_d_da_bhl.push(d_da_blue_bhl);

        // ── ORANGE backward ────────────────────────────────────────────────
        //   w[lₜ,lₛ] = CB[lₜ,lₛ] · exp(cumA[lₜ] − cumA[lₛ]) · dt[lₛ]   (causal)
        //   orange[lₜ,p] = Σ_{lₛ} w[lₜ,lₛ] · x[lₛ,p]
        let diff_bhll = {
            let target_bhll = da_cumsum_bhl
                .clone()
                .unsqueeze_dim::<4>(3) // _bhlt1
                .expand([batch, nheads, chunk_len, chunk_len]); // _bhltls
            let source_bhll = da_cumsum_bhl
                .unsqueeze_dim::<4>(2) // _bh1ls
                .expand([batch, nheads, chunk_len, chunk_len]); // _bhltls
            target_bhll - source_bhll
        };
        let causal_mask_bhll: Mask<B> = causal_mask_ll
            .clone()
            .reshape([1, 1, chunk_len, chunk_len]) // _11ll
            .expand([batch, nheads, chunk_len, chunk_len]); // _bhll
        let decay_bhll = diff_bhll
            .mask_fill(causal_mask_bhll.clone(), f32::NEG_INFINITY)
            .exp();
        san(&decay_bhll);

        let dt_source_bhll: F<B, 4> = dt_bhl
            .unsqueeze_dim::<4>(2) // _bh1l
            .expand([batch, nheads, chunk_len, chunk_len]); // _bhll
        let cb_decay_bhll = cb_bhll.clone() * decay_bhll.clone();
        let w_bhll = cb_decay_bhll.clone() * dt_source_bhll.clone();

        let d_orange_bhlp = d_y_bhlp; // = d_y_partial
        // d_w = d_orange @ x^T
        let d_w_bhll: F<B, 4> = d_orange_bhlp.clone().matmul(
            x_bhlp.permute([0, 1, 3, 2]), // x_bhpl
        );
        san(&d_w_bhll);

        // d_x = w^T @ d_orange
        let d_x_orange_bhlp: F<B, 4> = w_bhll
            .permute([0, 1, 3, 2]) // w_bhlslt
            .matmul(d_orange_bhlp);
        san(&d_x_orange_bhlp);
        vec_orange_d_x_bhlp.push(d_x_orange_bhlp);

        // Mask off above-diagonal contributions, then split d_w into its factors.
        let d_w_masked_bhll = d_w_bhll.mask_fill(causal_mask_bhll, 0.);
        let d_cb_decay_bhll = d_w_masked_bhll.clone() * dt_source_bhll;
        san(&d_cb_decay_bhll);

        // d_dt[s] = Σ_{lₜ ≥ lₛ} d_w[lₜ,lₛ] · CB[lₜ,lₛ] · decay[lₜ,lₛ]
        let d_dt_orange_bhl: F<B, 3> = (d_w_masked_bhll * cb_decay_bhll)
            .sum_dim(2) // _bh1ls
            .squeeze_dim::<3>(2); // _bhls
        san(&d_dt_orange_bhl);
        vec_orange_d_dt_bhl.push(d_dt_orange_bhl);

        // d_cb = d_cb_decay · decay ;  d_decay = d_cb_decay · cb
        let d_cb_bhll = d_cb_decay_bhll.clone() * decay_bhll.clone();
        vec_d_cb_bhll.push(d_cb_bhll);
        let d_decay_bhll = d_cb_decay_bhll * cb_bhll;
        let d_diff_bhll = d_decay_bhll * decay_bhll;

        // d_da from ORANGE:  d_da_tgt[l] += Σₛ d_diff[l,s]; d_da_src[s] −= Σₗ d_diff[l,s].
        let d_da_tgt_bhl: F<B, 3> = d_diff_bhll
            .clone()
            .sum_dim(3) // _bhlt1
            .squeeze_dim::<3>(3); // _bhlt
        let d_da_src_bhl: F<B, 3> = d_diff_bhll
            .sum_dim(2) // _bh1ls
            .squeeze_dim::<3>(2); // _bhls
        let d_da_orange_bhl = d_da_tgt_bhl - d_da_src_bhl;
        san(&d_da_orange_bhl);
        vec_orange_d_da_bhl.push(d_da_orange_bhl);

        // ── K4 backward step for chunk i ───────────────────────────────────
        //   Forward: sᵢ₊₁ = decayᵢ · sᵢ + intra_stateᵢ
        //   - d_intra_stateᵢ      = d_sᵢ₊₁  (current d_running_state)
        //   - d_decayᵢ            = d_sᵢ₊₁ · sᵢ
        //   - d_sᵢ (propagated)   = decayᵢ · d_sᵢ₊₁ + d_chunk_input_state
        vec_d_intra_bhpr.push(d_running_state_bhpr.clone());

        let decay_chunk_bhpr: F<B, 4> = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk]) // _bh1
            .exp() // _bh
            .unsqueeze_dim::<4>(3) // _bh11
            .expand([batch, nheads, per_head_dim, state_rank]); // _bhpr
        san(&decay_chunk_bhpr);

        let d_decay_chunk_bhpr = d_running_state_bhpr.clone() * chunk_input_state_bhpr;
        // d_da_chunk_end[b,h] = Σ_{p,r} d_decay · decay   (decay = exp(da_chunk_end))
        let d_da_chunk_end_bh: F<B, 2> = (d_decay_chunk_bhpr * decay_chunk_bhpr.clone())
            .reshape([batch, nheads, per_head_dim * state_rank]) // _bhPR
            .sum_dim(2) // _bh1
            .squeeze_dim::<2>(2); // _bh
        san(&d_da_chunk_end_bh);
        vec_d_da_end_bh.push(d_da_chunk_end_bh);

        d_running_state_bhpr = decay_chunk_bhpr * d_running_state_bhpr + d_chunk_input_state_bhpr;
        san(&d_running_state_bhpr);
    }
    // d_initial_state = the trailing d_running_state after the reverse loop.
    let d_initial_state_bhpr = d_running_state_bhpr;

    // ── Restore natural (forward) chunk order ─────────────────────────────
    vec_orange_d_x_bhlp.reverse();
    vec_orange_d_dt_bhl.reverse();
    vec_orange_d_da_bhl.reverse();
    vec_d_cb_bhll.reverse();
    vec_blue_d_c_bhlr.reverse();
    vec_blue_d_da_bhl.reverse();
    vec_d_intra_bhpr.reverse();
    vec_d_da_end_bh.reverse();

    // ── Stack per-chunk slices back into batched tensors ──────────────────
    let d_x_orange_bnlhp: F<B, 5> =
        F::stack::<5>(vec_orange_d_x_bhlp, 1).permute([0, 1, 3, 2, 4]);
    let d_dt_orange_bhnl: F<B, 4> = F::stack(vec_orange_d_dt_bhl, 2);
    let d_da_orange_bhnl: F<B, 4> = F::stack(vec_orange_d_da_bhl, 2);
    let d_cb_bnhll: F<B, 5> = F::stack(vec_d_cb_bhll, 1);
    let d_da_blue_bhnl: F<B, 4> = F::stack(vec_blue_d_da_bhl, 2);
    let d_intra_chunk_state_bnhpr: F<B, 5> = F::stack(vec_d_intra_bhpr, 1);
    let d_c_blue_bnhlr: F<B, 5> = F::stack(vec_blue_d_c_bhlr, 1);
    let d_da_end_bhn: F<B, 3> = F::stack(vec_d_da_end_bh, 2);
    san(&d_x_orange_bnlhp);
    san(&d_dt_orange_bhnl);
    san(&d_da_orange_bhnl);
    san(&d_cb_bnhll);
    san(&d_da_blue_bhnl);
    san(&d_intra_chunk_state_bnhpr);
    san(&d_c_blue_bnhlr);

    // d_da_cumsum from K4: only the last-l position of each chunk gets the
    // d_da_chunk_end contribution (da_chunk_end = cumA[chunk_len-1]).
    let d_da_cumsum_k4_bhnl = {
        let zeros = F::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device, dtype);
        F::cat(vec![zeros, d_da_end_bhn.unsqueeze_dim::<4>(3)], 3)
    };

    let d_c_blue_bnlhr = d_c_blue_bnhlr.permute([0, 1, 3, 2, 4]);

    // ═══════════════════════════════════════════════════════════════════════
    // K3 BACKWARD
    //
    // Forward (recap):
    //   b_bnhlr = b_bnlhr.permute
    //   x_bnhpl = x_bnlhp.permute
    //   forward_decay_to_chunk_end = exp(cumA_last − cumA)
    //   b_bar_scale = forward_decay · dt
    //   b_scaled = b · b_bar_scale_broadcast
    //   intra_state = x_bnhpl @ b_scaled
    // ═══════════════════════════════════════════════════════════════════════
    let x_bnhpl = x_bnlhp.permute([0, 1, 3, 4, 2]);

    // d_x_k3 = d_intra_state @ b_scaled^T
    let d_x_k3_bnlhp = d_intra_chunk_state_bnhpr
        .clone()
        .matmul(
            b_scaled_bnhlr.clone().permute([0, 1, 2, 4, 3]), // b_scaled_bnhrl
        ) // d_x_k3_bnhpl
        .permute([0, 1, 4, 2, 3]); // d_x_k3_bnlhp
    san(&d_x_k3_bnlhp);

    // d_b_scaled = x^T @ d_intra_state
    let d_b_scaled_bnhlr = x_bnhpl
        .permute([0, 1, 2, 4, 3]) // x_bnhlp
        .matmul(d_intra_chunk_state_bnhpr);
    san(&d_b_scaled_bnhlr);

    // Split d_b_scaled via the product rule: d_b = d_b_scaled · b_bar_scale ; d_b_bar_scale = d_b_scaled · b
    let b_bar_scale_bnhlr = b_bar_scale_bhnl
        .clone()
        .permute([0, 2, 1, 3]) // _bnhl
        .unsqueeze_dim::<5>(4) // _bnhl1
        .expand([batch, nchunks, nheads, chunk_len, state_rank]);
    let d_b_k3_bnhlr = d_b_scaled_bnhlr.clone() * b_bar_scale_bnhlr;
    let d_b_k3_bnlhr = d_b_k3_bnhlr.permute([0, 1, 3, 2, 4]);
    san(&d_b_k3_bnlhr);

    let b_bnhlr = b_bnlhr.clone().permute([0, 1, 3, 2, 4]);
    let d_b_bar_scale_bhnl = (d_b_scaled_bnhlr * b_bnhlr)
        .sum_dim(4) // _bnhl1
        .squeeze_dim::<4>(4) // _bnhl
        .permute([0, 2, 1, 3]); // _bhnl
    san(&d_b_bar_scale_bhnl);

    // Through the product rule for b_bar_scale = forward_decay · dt:
    //   d_forward_decay = d_b_bar_scale · dt
    //   d_dt_k3         = d_b_bar_scale · forward_decay
    let d_forward_decay_bhnl = d_b_bar_scale_bhnl.clone() * dt_discretized_bhnl.clone();
    let d_dt_discretized_k3_bhnl = d_b_bar_scale_bhnl * forward_decay_to_chunk_end_bhnl.clone();

    // Through exp: d_(cumA_last − cumA) = d_forward_decay · forward_decay
    let d_da_delta_bhnl = d_forward_decay_bhnl * forward_decay_to_chunk_end_bhnl;
    // Subtraction splits into:
    //   d_cumA_last = +Σ_l d_da_delta (it was broadcast over l), scattered into last-l position.
    //   d_cumA      = −d_da_delta
    let d_da_cumsum_sub_bhnl = -d_da_delta_bhnl.clone();
    let d_da_cumsum_last_bhn = d_da_delta_bhnl
        .sum_dim(3) // _bhn1
        .squeeze_dim::<3>(3); // _bhn
    let d_da_cumsum_k3_bhnl = {
        let zeros = F::<B, 4>::zeros([batch, nheads, nchunks, chunk_len - 1], &device, dtype);
        d_da_cumsum_sub_bhnl
            + F::cat(vec![zeros, d_da_cumsum_last_bhn.unsqueeze_dim::<4>(3)], 3)
    };
    san(&d_da_cumsum_k3_bhnl);

    // ═══════════════════════════════════════════════════════════════════════
    // K2 BACKWARD (from d_cb_bnhll)
    //
    //   cb_bnhll = c_bnhlr @ b_bnhrl
    //   d_c = d_cb @ b ;  d_b = d_cb^T @ c
    // ═══════════════════════════════════════════════════════════════════════
    let c_bnhlr = c_bnlhr.permute([0, 1, 3, 2, 4]);
    let b_bnhlr = b_bnlhr.permute([0, 1, 3, 2, 4]);

    let d_c_k2_bnhlr = d_cb_bnhll.clone().matmul(b_bnhlr.clone());
    let d_c_k2_bnlhr = d_c_k2_bnhlr.permute([0, 1, 3, 2, 4]);
    san(&d_c_k2_bnlhr);

    let d_b_k2_bnhlr = d_cb_bnhll
        .permute([0, 1, 2, 4, 3]) // d_cb_bnhsltl (target/source swap)
        .matmul(c_bnhlr);
    let d_b_k2_bnlhr = d_b_k2_bnhlr.permute([0, 1, 3, 2, 4]);
    san(&d_b_k2_bnlhr);

    // ═══════════════════════════════════════════════════════════════════════
    // SUM GRADIENT CONTRIBUTIONS + K1 BACKWARD
    // ═══════════════════════════════════════════════════════════════════════

    // Test-only: local same-chunk d_da contribution (BLUE + ORANGE) snapshot
    // for the `out_x · dout − ddt · dt` oracle. Production builds skip the
    // extra add and the retained _bhnl tensor.
    #[cfg(test)]
    let d_da_local_bhnl = d_da_blue_bhnl.clone() + d_da_orange_bhnl.clone();
    #[cfg(test)]
    san(&d_da_local_bhnl);

    let d_da_cumsum_bhnl =
        d_da_blue_bhnl + d_da_orange_bhnl + d_da_cumsum_k3_bhnl + d_da_cumsum_k4_bhnl;
    san(&d_da_cumsum_bhnl);

    // K1 forward:  da_cumsum[l] = cumsumₗ(dt[l] · a_decay)
    // Reverse:     d_da[l] = Σ_{k ≥ l} d_da_cumsum[k]   (suffix sum)
    //                      = total − cumsum(d_da_cumsum)[l−1]   (cumsum[−1] = 0)
    let d_da_bhnl = {
        let d_total_bhnl = d_da_cumsum_bhnl
            .clone()
            .sum_dim(3) // _bhn1
            .expand([batch, nheads, nchunks, chunk_len]);
        let prefix_bhnl = d_da_cumsum_bhnl.cumsum(3);
        let zeros_bhn1 = F::<B, 4>::zeros([batch, nheads, nchunks, 1], &device, dtype);
        let prefix_shifted_bhnl =
            F::cat(vec![zeros_bhn1, prefix_bhnl.narrow(3, 0, chunk_len - 1)], 3);
        d_total_bhnl - prefix_shifted_bhnl
    };
    san(&d_da_bhnl);

    // d_dt from K1: d_dt = d_da · a_decay
    let a_decay_111h1 = a_decay_h
        .unsqueeze_dims::<4>(&[0, 2, 3])
        .expand([batch, nheads, nchunks, chunk_len]);
    let d_dt_k1_bhnl = d_da_bhnl.clone() * a_decay_111h1;
    san(&d_dt_k1_bhnl);

    // d_a_decay[h] = Σ_{b,n,l} d_da[b,h,n,l] · dt[b,h,n,l]
    let d_a_decay_h = (d_da_bhnl * dt_discretized_bhnl.clone())
        .permute([1, 0, 2, 3]) // _hbnl
        .reshape([nheads, batch * nchunks * chunk_len]) // _hBNL
        .sum_dim(1) // _h1
        .squeeze_dim::<1>(1); // _h
    san(&d_a_decay_h);

    // ── Combine per-input gradient contributions ──────────────────────────
    #[cfg(test)]
    let d_dt_orange_bhnl_save = d_dt_orange_bhnl.clone();
    let d_dt_discretized_bhnl = d_dt_orange_bhnl + d_dt_discretized_k3_bhnl + d_dt_k1_bhnl;
    san(&d_dt_discretized_bhnl);

    let d_x_bnlhp = d_x_skip_bnlhp + d_x_k3_bnlhp + d_x_orange_bnlhp;
    san(&d_x_bnlhp);

    let d_b_bnlhr = d_b_k2_bnlhr + d_b_k3_bnlhr;
    san(&d_b_bnlhr);
    let d_c_bnlhr = d_c_k2_bnlhr + d_c_blue_bnlhr;
    san(&d_c_bnlhr);

    CombinedGrads {
        d_a_decay_h,
        d_dt_discretized_bhnl,
        d_x_bnlhp,
        d_b_bnlhr,
        d_c_bnlhr,
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
mod tests;
