//! # Recompute-based gradient math for the Mamba-3 single-SSD
//!
//! The analytic backward of the single-pass MIMO-first scan:
//!
//! 1. Recompute the forward intermediates (K1–K4) from the saved leaf inputs.
//! 2. Compute every chunk-**local** gradient batched: the K5 state-to-output
//!    (BLUE), the strict lower-triangular intra-chunk (LOWER) and the
//!    γ-weighted diagonal (DIAG) terms. BLUE and LOWER run one chunk group at
//!    a time.
//! 3. Walk the K4 state-passing backward, the one recurrence.
//!
//! This pathway applies the trapezoid weights inside the kernel, so it also
//! returns `d_gamma` and `d_scale`.
//!
//! The backwards of K1–K4, BLUE and LOWER are in the
//! [kernels of the standard SSD scan](crate::standard_ssd::kernels). This file
//! assembles them. It adds the terms of this pathway: the diagonal DIAG, the
//! split of the K3 gradient into `d_b` and `d_scale`, and the boundary seeds.
//!
//! All the math is on backend **primitives**, through the rank-tagged `F`
//! wrapper. The [recompute node](burn_stack::utils::recompute_node)
//! runs with a generic backend `B`, where the high-level `Tensor` is not
//! available, so the math uses the `float_*` ops of `B`.

#![allow(non_snake_case)]

use crate::mamba3::single_ssd::ssd::serial_recalculated::diag::{
    DiagGrads, y_diag_correction_backward,
};
use crate::standard_ssd::chunk_tokens;
use crate::standard_ssd::kernels::{
    K5Grads, k1_ssd_chunk_cumsum, k1_ssd_chunk_cumsum_backward, k2_ssd_bmm, k2_ssd_bmm_backward,
    k3_ssd_chunk_state_backward, k3_ssd_chunk_state_extended, k4_ssd_state_passing,
    k4_ssd_state_passing_backward, k5_da_cumsum_on_chunk_axes, k5_ssd_chunk_scan_backward,
    scatter_chunk_end, split_fused_axis,
};
use crate::standard_ssd::read_axis::{read_positions, scatter_read_positions};
use burn_stack::utils::fprim::{F, san};
use burn::backend::Backend;

/// Per-input gradients produced by [`combined_backward`] for the Single-SSD.
/// It adds `d_gamma_bnth` and `d_scale_bnlh` to the fields of the
/// [`CombinedGrads`](crate::standard_ssd::CombinedGrads) of the standard SSD scan.
#[non_exhaustive]
pub struct CombinedSingleSsdGrads<B: Backend> {
    /// Gradient of the raw input `v`.
    pub d_v_bnlmhp: F<B, 6>,
    /// Gradient of the log-decay `da`.
    pub d_da_bnlh: F<B, 4>,
    /// Gradient of the input projection `B`.
    pub d_b_bnlmhr: F<B, 6>,
    /// Gradient of the output projection `C`, on the read axis of the chunk.
    pub d_c_bntmhr: F<B, 6>,
    /// Gradient of the diagonal trapezoid weight `γ`, on the read axis.
    pub d_gamma_bnth: F<B, 4>,
    /// Gradient of the key scale `scale` (at lag 1, `γ + (1−λ₊₁)·Δ₊₁`).
    pub d_scale_bnlh: F<B, 4>,
    /// Gradient of the initial state.
    pub d_initial_state_bhpr: F<B, 4>,
    /// Gradient of the origin state of the resets (packed rows only).
    pub d_origin_bhpr: Option<F<B, 4>>,
    /// Gradient of the per-chunk boundary seeds (packed rows only).
    pub d_seed_bnhpr: Option<F<B, 5>>,
}

/// Memory-efficient backward for the Mamba-3 MIMO-first chunkwise Single-SSD.
///
/// Recomputes the forward intermediates (K1–K4) from the saved inputs, then:
/// - computes the K5 BLUE (state-to-output) and strict lower-triangular LOWER
///   (intra-chunk) backwards one chunk group at a time (the group size sets
///   the memory of the score, see
///   [`backward_chunk_group`](crate::standard_ssd::backward_chunk_group)),
/// - computes the γ-weighted diagonal DIAG backward batched (its `m × m`
///   working tensors are tiny),
/// - walks the K4 state-passing backward (the one recurrence) over the
///   chunk-input-state gradient from the first item.
///
/// K3/K2/K1 backwards then run as single batched ops.
///
/// # Arguments
/// - `d_y_bntmhp` — upstream gradient of the SSD output (read axis)
/// - `d_final_bhpr` — upstream gradient of the final state
/// - `v_bnlmhp`, `da_bnlh`, `b_bnlmhr`, `c_bntmhr`, `gamma_bnth`, `scale_bnlh`,
///   `initial_state_bhpr` — the seven saved forward inputs
/// - `read_stride` — write positions per read position (see
///   [`Mamba3SingleSsdInput::read_stride`](crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput))
/// - `siso_specialization` — the forward's γ-correction branch choice, replayed
///   here so the backward matches it (performance-only; both agree)
/// - `resets` — `(keep_bn, origin_bhpr)`, or `None`. `keep_bn` is the
///   constant reset factor of the forward (`0` at a chunk that starts a new
///   segment). `origin_bhpr` is the state that those chunks start from.
/// - `seed_bnhpr` — the per-chunk boundary seeds of a packed row, or `None`
///
/// # Returns
/// One [`CombinedSingleSsdGrads`] with gradients for all 7 inputs (and the
/// origin state and the seeds).
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
    resets: Option<(F<B, 2>, F<B, 4>)>,
    seed_bnhpr: Option<F<B, 5>>,
) -> CombinedSingleSsdGrads<B> {
    let seeded = seed_bnhpr.is_some();
    let keep_bn = resets.as_ref().map(|(keep_bn, _)| keep_bn.clone());
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();
    // The chunk's two axes: `fused` is what it writes, `read` what it is read
    // at. They coincide unless `micro_steps > 1`.
    let chunk_tokens = chunk_tokens(chunk_len, read_stride);
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

    // ── Recompute the forward intermediates (K1–K4, single-SSD form) ──────

    // K1
    let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum(da_bnlh);
    san(&da_cumsum_bhnl);

    // K2: the CB matrix (unscaled) of LOWER.
    let cb_bnhTMLM = k2_ssd_bmm(c_bntmhr.clone(), b_bnlmhr.clone());
    san(&cb_bnhTMLM);

    // K3: the chunk state from K_scaled = scaleₜ·B.
    let scale_bnlh11 = scale_bnlh.clone().unsqueeze_dims::<6>(&[3, 5]);
    let k_scaled_bnlmhr = b_bnlmhr.clone() * scale_bnlh11.clone();
    let (intra_chunk_state_bnhpr, k3_decay_bhnLM, k3_decayed_v_bnLMhp) =
        k3_ssd_chunk_state_extended(
            v_bnlmhp.clone(),
            k_scaled_bnlmhr.clone(),
            da_cumsum_bhnl.clone(),
        );

    // K4: the chunk-input state stream of BLUE.
    let (chunk_input_state_bnhpr, _final_state_bhpr) = k4_ssd_state_passing(
        intra_chunk_state_bnhpr,
        da_chunk_end_bhn.clone(),
        initial_state_bhpr,
        resets,
        seed_bnhpr,
    );

    let (da_cumsum_bhnLM, da_cumsum_bhnTM) =
        k5_da_cumsum_on_chunk_axes(da_cumsum_bhnl, mimo_rank, read_stride);

    // d_y in (batch, nchunks, nheads, chunk_tokens·mimo_rank, per_head_dim) ordering.
    let d_y_bnhTMp = d_y_bntmhp
        .clone()
        .reshape([batch, nchunks, read, nheads, per_head_dim])
        .swap_dims(2, 3);
    san(&d_y_bnhTMp);

    // ═══════════════════════════════════════════════════════════════════════
    // DIAG BACKWARD (batched, with no recurrence, on a small m × m working set)
    //
    // Forward (per (b,n,l,h)):
    //   qk_dot[m_out, m_in] = Σ_r C[m_out, r] · B[m_in, r]
    //   y_diag[m_out, p]    = γ · Σ_{m_in} qk_dot[m_out, m_in] · V[m_in, p]
    // ═══════════════════════════════════════════════════════════════════════
    // The same position is the *read* position, so V and B enter at those
    // positions and their gradients are scattered back onto the write axis.
    let DiagGrads {
        d_v_bnlmhp: d_v_diag_bntmhp,
        d_c_bnlmhr: d_c_diag_bntmhr,
        d_b_bnlmhr: d_b_diag_bntmhr,
        d_gamma_bnlh: d_gamma_bnth,
    } = y_diag_correction_backward(
        d_y_bntmhp,
        read_positions::<B, 6, 7>(v_bnlmhp.clone(), 2, read_stride),
        read_positions::<B, 6, 7>(b_bnlmhr.clone(), 2, read_stride),
        c_bntmhr.clone(),
        gamma_bnth,
        siso_specialization,
    );
    let d_v_diag_bnlmhp = scatter_read_positions::<B, 6, 7>(d_v_diag_bntmhp, 2, read_stride);
    let d_b_diag_bnlmhr = scatter_read_positions::<B, 6, 7>(d_b_diag_bntmhr, 2, read_stride);
    san(&d_gamma_bnth);

    // ── K5 backward (BLUE + LOWER) ────────────────────────────────────────
    let (
        K5Grads {
            d_chunk_input_state_bnhpr,
            d_c_blue_bnhTMr,
            d_da_blue_bhnl,
            d_v_intra_bnhLMp: d_v_lower_bnhLMp,
            d_cb_bnhTMLM,
            d_da_intra_bhnl: d_da_lower_bhnl,
        },
        d_scale_lower_bhnl,
    ) = k5_ssd_chunk_scan_backward(
        d_y_bnhTMp,
        v_bnlmhp.clone(),
        c_bntmhr.clone(),
        cb_bnhTMLM,
        da_cumsum_bhnLM,
        da_cumsum_bhnTM,
        chunk_input_state_bnhpr.clone(),
        Some(scale_bnlh),
        0, // `triu(0)`: the strict lower triangle excludes the same position.
        read_stride,
    );
    let d_scale_lower_bhnl = d_scale_lower_bhnl.expect("K5 returns `d_scale` when it gets a scale");

    // ── K4 backward: the reverse of the state-passing scan ────────────────
    let (d_intra_chunk_state_bnhpr, d_da_end_bhn, d_initial_state_bhpr, d_start_state_bnhpr) =
        k4_ssd_state_passing_backward(
            d_chunk_input_state_bnhpr,
            chunk_input_state_bnhpr,
            da_chunk_end_bhn,
            d_final_bhpr,
            keep_bn,
        );
    // The start state of chunk `c` is `origin + seedᶜ`: the origin gets the
    // sum over the chunks, and each seed its own chunk.
    let d_origin_bhpr = d_start_state_bnhpr
        .clone()
        .map(|d_start_state_bnhpr| d_start_state_bnhpr.sum_dim(1).squeeze_dim::<4>(1));
    let d_seed_bnhpr = d_start_state_bnhpr.filter(|_| seeded);
    let d_da_cumsum_k4_bhnl = scatter_chunk_end(d_da_end_bhn, chunk_len);

    // ── K3 backward on K_scaled = scaleₜ·B ────────────────────────────────
    let k_scaled_bnLMhr = k_scaled_bnlmhr.reshape([batch, nchunks, fused, nheads, state_rank]);
    let (d_v_k3_bnlmhp, d_k_scaled_bnlmhr, d_da_cumsum_k3_bhnl) = k3_ssd_chunk_state_backward(
        d_intra_chunk_state_bnhpr,
        k3_decay_bhnLM,
        k3_decayed_v_bnLMhp,
        v_bnlmhp,
        k_scaled_bnLMhr,
    );

    // Split d_K_scaled into d_b_k3 (·scale) and d_scale_k3 (Σ_{m,r} ·B).
    let d_b_k3_bnlmhr: F<B, 6> = d_k_scaled_bnlmhr.clone() * scale_bnlh11;
    let d_scale_k3_bnlh: F<B, 4> = (d_k_scaled_bnlmhr * b_bnlmhr.clone())
        .sum_dim(5) // sum over state_rank → [b,n,l,m,h,1]
        .squeeze_dim::<5>(5) // [b,n,l,m,h]
        .sum_dim(3) // sum over mimo_rank → [b,n,l,1,h]
        .squeeze_dim::<4>(3); // [b,n,l,h]

    // ── K2 backward: cb = C @ Bᵀ ──────────────────────────────────────────
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, fused, nheads, state_rank]);
    let (d_c_k2_bntmhr, d_b_k2_bnlmhr) = k2_ssd_bmm_backward(d_cb_bnhTMLM, c_bntmhr, b_bnLMhr);

    let d_c_blue_bntmhr = split_fused_axis(d_c_blue_bnhTMr, chunk_tokens);
    let d_v_lower_bnlmhp = split_fused_axis(d_v_lower_bnhLMp, chunk_len);

    // ── K1 backward and the sum of the contributions ──────────────────────
    let d_da_cumsum_bhnl =
        d_da_blue_bhnl + d_da_lower_bhnl + d_da_cumsum_k3_bhnl + d_da_cumsum_k4_bhnl;
    san(&d_da_cumsum_bhnl);
    let d_da_bnlh = k1_ssd_chunk_cumsum_backward(d_da_cumsum_bhnl);

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
        d_origin_bhpr,
        d_seed_bnhpr,
    }
}
