//! # The analytic backward of the standard SSD scan
//!
//! The steps of the backward of one SSD call:
//!
//! 1. Recompute the forward intermediates (K1–K4) from the saved leaf inputs.
//!    They are not stored.
//! 2. Run the chunk-**local** K5 backward batched, one chunk group at a time.
//! 3. Walk the K4 state-passing backward: the one recurrence, and the only
//!    walk.
//! 4. Run the K3, K2 and K1 backwards batched.
//!
//! The backward of each kernel is in [`super::kernels`], beside its forward.
//! This file assembles them. The single-SSD pathway assembles the same
//! functions, with its own terms.
//!
//! The fused `L·M` length carries the `mimo_rank` axis through the intra-chunk
//! products.
//!
//! All the math is on backend **primitives**, through the rank-tagged `F`
//! wrapper. The [recompute node](burn_stack::utils::recompute_node)
//! runs with a generic backend `B`, where the high-level `Tensor` is not
//! available, so the math uses the `float_*` ops of `B`.

#![allow(non_snake_case)]

use super::chunk_tokens;
use super::kernels::{
    K5Grads, k1_ssd_chunk_cumsum, k1_ssd_chunk_cumsum_backward, k2_ssd_bmm, k2_ssd_bmm_backward,
    k3_ssd_chunk_state_backward, k3_ssd_chunk_state_extended, k4_ssd_state_passing,
    k4_ssd_state_passing_backward, k5_da_cumsum_on_chunk_axes, k5_ssd_chunk_scan_backward,
    scatter_chunk_end, split_fused_axis,
};
use burn_stack::utils::fprim::{F, san};
use burn::backend::Backend;

/// Per-input gradients produced by [`combined_backward`] (one field per
/// differentiable forward input of one SSD call).
#[non_exhaustive]
pub struct CombinedGrads<B: Backend> {
    /// Gradient of the (pre-scaled) input `v`.
    pub d_v_bnlmhp: F<B, 6>,
    /// Gradient of `Δ·A` (`da`).
    pub d_da_bnlh: F<B, 4>,
    /// Gradient of the input projection `B`.
    pub d_b_bnlmhr: F<B, 6>,
    /// Gradient of the output projection `C`, on the read axis of the chunk.
    pub d_c_bntmhr: F<B, 6>,
    /// Gradient of the initial state.
    pub d_initial_state_bhpr: F<B, 4>,
    /// Gradient of the origin state of the resets (`None` without resets).
    pub d_origin_bhpr: Option<F<B, 4>>,
    /// Local same-chunk part of `d_da_cumsum`, from BLUE and ORANGE only (no K3
    /// and no K4 contribution). Test builds only.
    #[cfg(test)]
    pub d_da_local_bhnl: F<B, 4>,
    /// The ORANGE part of `d_v`. Test builds only.
    #[cfg(test)]
    pub d_v_orange_bnlmhp: F<B, 6>,
}

/// Memory-efficient backward of one SSD call.
///
/// Recomputes the forward intermediates (K1-K4) from the saved inputs. Then
/// the chunk-local K5 (BLUE + ORANGE) backward runs batched over chunk groups,
/// and the K4 state-passing backward walks the chunks in reverse. The K3/K2/K1
/// backwards run as single batched ops.
///
/// # Arguments
/// - `d_y_bntmhp` — upstream gradient of the SSD output (read axis)
/// - `d_final_bhpr` — upstream gradient of the final state
/// - `v_bnlmhp`, `da_bnlh`, `b_bnlmhr`, `c_bntmhr`, `initial_state_bhpr` —
///   the five saved forward inputs
/// - `read_stride` — write positions per read position (`micro_steps`)
/// - `restarts` — `(keep_bn, origin_bhpr)`, or `None`. `keep_bn` is the
///   constant reset factor of the forward (`0` at a chunk that starts a new
///   segment). `origin_bhpr` is the state that those chunks start from.
///
/// # Returns
/// One [`CombinedGrads`] struct with the gradients of all 5 inputs, and of
/// the origin state.
#[allow(clippy::too_many_arguments)]
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
    restarts: Option<(F<B, 2>, F<B, 4>)>,
) -> CombinedGrads<B> {
    let keep_bn = restarts.as_ref().map(|(keep_bn, _)| keep_bn.clone());
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();
    // The chunk's two axes: `fused` is what it writes, `read` what it is read at.
    let chunk_tokens = chunk_tokens(chunk_len, read_stride);
    let fused = chunk_len * mimo_rank;
    let read = chunk_tokens * mimo_rank;

    san(&d_y_bntmhp);
    san(&d_final_bhpr);
    san(&v_bnlmhp);
    san(&da_bnlh);
    san(&b_bnlmhr);
    san(&c_bntmhr);
    san(&initial_state_bhpr);

    // ── Recompute the forward intermediates ───────────────────────────────

    // K1: pre-combined Δ·A to the intra-chunk cumsum.
    let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum(da_bnlh);
    san(&da_cumsum_bhnl);

    // K2: the CB matrix of ORANGE.
    let cb_bnhTMLM = k2_ssd_bmm(c_bntmhr.clone(), b_bnlmhr.clone());
    san(&cb_bnhTMLM);

    // K3: the intra-chunk state, and the intermediates of its backward.
    let (intra_chunk_state_bnhpr, k3_decay_bhnLM, k3_decayed_v_bnLMhp) =
        k3_ssd_chunk_state_extended(v_bnlmhp.clone(), b_bnlmhr.clone(), da_cumsum_bhnl.clone());

    // K4: the chunk-input state stream of BLUE.
    let (chunk_input_state_bnhpr, _final_state_bhpr) = k4_ssd_state_passing(
        intra_chunk_state_bnhpr,
        da_chunk_end_bhn.clone(),
        initial_state_bhpr,
        restarts,
        None,
    );

    let (da_cumsum_bhnLM, da_cumsum_bhnTM) =
        k5_da_cumsum_on_chunk_axes(da_cumsum_bhnl, mimo_rank, read_stride);
    let d_y_bnhTMp = d_y_bntmhp
        .reshape([batch, nchunks, read, nheads, per_head_dim]) // d_y_bnTMhp
        .swap_dims(2, 3); // d_y_bnhTMp
    san(&d_y_bnhTMp);

    // ── K5 backward (BLUE + ORANGE) ───────────────────────────────────────
    let (
        K5Grads {
            d_chunk_input_state_bnhpr,
            d_c_blue_bnhTMr,
            d_da_blue_bhnl,
            d_v_intra_bnhLMp: d_v_orange_bnhLMp,
            d_cb_bnhTMLM,
            d_da_intra_bhnl: d_da_orange_bhnl,
        },
        _,
    ) = k5_ssd_chunk_scan_backward(
        d_y_bnhTMp,
        v_bnlmhp.clone(),
        c_bntmhr.clone(),
        cb_bnhTMLM,
        da_cumsum_bhnLM,
        da_cumsum_bhnTM,
        chunk_input_state_bnhpr.clone(),
        None,
        1, // `triu(1)`: the sum of this scan includes the same position.
        read_stride,
    );

    // ── K4 backward: the reverse of the state-passing scan ────────────────
    let (d_intra_chunk_state_bnhpr, d_da_end_bhn, d_initial_state_bhpr, d_restart_bnhpr) =
        k4_ssd_state_passing_backward(
            d_chunk_input_state_bnhpr,
            chunk_input_state_bnhpr,
            da_chunk_end_bhn,
            d_final_bhpr,
            keep_bn,
        );
    // Every chunk restarts from the same origin state.
    let d_origin_bhpr = d_restart_bnhpr.map(|d_restart_bnhpr| d_restart_bnhpr.sum_dim(1).squeeze_dim::<4>(1));
    let d_da_cumsum_k4_bhnl = scatter_chunk_end(d_da_end_bhn, chunk_len);

    // ── K3 and K2 backwards ───────────────────────────────────────────────
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, fused, nheads, state_rank]);
    let (d_v_k3_bnlmhp, d_b_k3_bnlmhr, d_da_cumsum_k3_bhnl) = k3_ssd_chunk_state_backward(
        d_intra_chunk_state_bnhpr,
        k3_decay_bhnLM,
        k3_decayed_v_bnLMhp,
        v_bnlmhp,
        b_bnLMhr.clone(),
    );
    let (d_c_k2_bntmhr, d_b_k2_bnlmhr) = k2_ssd_bmm_backward(d_cb_bnhTMLM, c_bntmhr, b_bnLMhr);

    let d_c_blue_bntmhr = split_fused_axis(d_c_blue_bnhTMr, chunk_tokens);
    let d_v_orange_bnlmhp = split_fused_axis(d_v_orange_bnhLMp, chunk_len);

    // Test-only copies for the oracle of the local same-chunk `d_da`.
    // Production builds compute neither the add nor the copy.
    #[cfg(test)]
    let d_da_local_bhnl = d_da_blue_bhnl.clone() + d_da_orange_bhnl.clone();
    #[cfg(test)]
    let d_v_orange_save_bnlmhp = d_v_orange_bnlmhp.clone();

    // ── K1 backward and the sum of the contributions ──────────────────────
    let d_da_cumsum_bhnl =
        d_da_blue_bhnl + d_da_orange_bhnl + d_da_cumsum_k3_bhnl + d_da_cumsum_k4_bhnl;
    san(&d_da_cumsum_bhnl);
    let d_da_bnlh = k1_ssd_chunk_cumsum_backward(d_da_cumsum_bhnl);

    let d_v_bnlmhp = d_v_k3_bnlmhp + d_v_orange_bnlmhp;
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
        d_origin_bhpr,
        #[cfg(test)]
        d_da_local_bhnl,
        #[cfg(test)]
        d_v_orange_bnlmhp: d_v_orange_save_bnlmhp,
    }
}
