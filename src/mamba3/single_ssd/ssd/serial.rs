//! # SingleSsd Serial (K1–K5) SSD
//!
//! Chunk-serial counterpart to [`crate::mamba3::single_ssd::ssd::minimal`].
//! Whereas the Minimal variant uses a segsum-based quadratic state passing,
//! this one reuses the K1–K4 helpers from [`crate::mamba3::double_ssd::ssd::serial`]
//! (which run a sequential loop for K4) and supplies a **new K5** that bakes
//! in the single-ssd logic:
//!
//! - Strict lower-triangular intra-chunk path (the same-time-step block is
//!   excluded from the SSM sum; it is the “diagonal correction” territory).
//! - K is scaled by `scaleₜ = γₜ + (1−λₜ₊₁) Δₜ₊₁` per source-time column.
//! - Same-time-step block contributes via an explicit `γₜ · (C·Bᵀ at t) · Vₜ`
//!   correction term, restoring the right diagonal weighting.
//!
//! K1–K4 are identical to the double-SSD because:
//! - K1 (`da_cumsum`, `da_chunk_end`) depends only on `da = Δ·A`.
//! - K2 (`cb = C · Bᵀ`) is computed on **unscaled** B / C; the single-ssd
//!   algorithm wants the unscaled CB so it can apply `scaleₜ` per-column
//!   (lower triangular) and reuse the same-step block for the γ-correction.
//! - K3 (chunk-end state from V·decay·K) is form-invariant: passing the
//!   scale-multiplied K (`K_scaled = scaleₜ · B`) recovers the single-ssd
//!   chunk state, with no other changes needed.
//! - K4 (sequential state passing across chunks) operates on a `[H, P, R]`
//!   per-chunk state and a per-chunk decay total; both are mode-agnostic.
//!
//! Reference kernels (same as `single_ssd_minimal`):
//! - `refs/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`
//! - `refs/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`

#![allow(non_snake_case)]

pub use crate::mamba3::double_ssd::ssd::serial::{
    k1_ssd_chunk_cumsum, k2_ssd_bmm, k3_ssd_chunk_state, k4_ssd_state_passing,
};
use crate::mamba3::single_ssd::prelude::*;
use burn::prelude::*;

impl<B: Backend> Mamba3SingleSsdInput<B> {
    /// MIMO-first Single-SSD — chunk-serial (K1–K5) variant.
    ///
    /// Sequence of kernels (matches the double-ssd `ssd_serial`):
    /// 1. **K1**: intra-chunk cumulative log-decay and per-chunk decay totals.
    /// 2. **K2**: `cb = C · Bᵀ` block matrix (unscaled).
    /// 3. **K3**: per-chunk hidden state assuming zero initial state, fed
    ///    `K_scaled = scaleₜ · B`.
    /// 4. **K4**: sequential state passing across chunks (loop over chunks).
    /// 5. **K5** (this module's new function): single-ssd chunk scan with
    ///    strict lower-triangular masking, scale broadcasting, and the
    ///    `γₜ`-weighted same-step diagonal correction.
    ///
    /// # Returns
    /// - `y_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]` —
    ///   the single-ssd accumulator at the last token.
    pub fn single_ssd_serial(self) -> (Tensor<B, 6>, Tensor<B, 4>) {
        let input = self;
        input.sanity();
        let [batch, nchunks, chunk_len, _mimo_rank, nheads, per_head_dim] = input.v_bnlmhp.dims();
        let [.., state_rank] = input.b_bnlmhr.dims();

        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr is not yet supported in single_ssd_serial; use single_ssd_minimal instead"
        );
        assert!(nchunks > 0, "sequence length must be at least 1");
        assert_eq!(
            [batch, nchunks, chunk_len, nheads],
            input.gamma_bnlh.dims(),
            "gamma must align with da"
        );
        assert_eq!(
            [batch, nchunks, chunk_len, nheads],
            input.scale_bnlh.dims(),
            "scale must align with da"
        );

        // ── K1: chunk cumulative decay ────────────────────────────────────────
        let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum(input.da_bnlh.clone());

        // ── K2: CB matrix on unscaled B/C ─────────────────────────────────────
        // SingleSsd K5 applies the `scale` and `gamma` weights post-hoc, so K2 is
        // identical to the double-ssd K2.
        let cb_bnhLMLM: Tensor<B, 5> = k2_ssd_bmm(input.c_bnlmhr.clone(), input.b_bnlmhr.clone());

        // ── K3: chunk state using K_scaled = scaleₜ · B ───────────────────────
        // The existing K3 computes `state = (V * decay)^T @ B_input`, so passing
        // `B_input = K_scaled` recovers the single-ssd per-chunk state.
        let scale_bnlh11 = input.scale_bnlh.clone().unsqueeze_dims::<6>(&[3, 5]);
        let k_scaled_bnlmhr = input.b_bnlmhr.clone() * scale_bnlh11;
        let intra_chunk_state_bnhpr: Tensor<B, 5> = k3_ssd_chunk_state(
            input.v_bnlmhp.clone(),
            k_scaled_bnlmhr,
            da_cumsum_bhnl.clone(),
        );

        // ── K4: sequential state passing across chunks ────────────────────────
        let (chunk_input_state_bnhpr, final_state_bhpr): (Tensor<B, 5>, Tensor<B, 4>) =
            k4_ssd_state_passing(
                intra_chunk_state_bnhpr,
                da_chunk_end_bhn,
                input.initial_state_bhpr,
            );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            chunk_input_state_bnhpr.dims()
        );

        // ── K5: single-ssd chunk scan (strict-lower + diag γ-correction + Y_off)
        let y_bnlmhp = k5_single_ssd_chunk_scan(
            da_cumsum_bhnl,
            input.v_bnlmhp,
            input.c_bnlmhr,
            input.b_bnlmhr,
            cb_bnhLMLM,
            input.gamma_bnlh,
            input.scale_bnlh,
            chunk_input_state_bnhpr,
        );

        (y_bnlmhp, final_state_bhpr)
    }
}

// ---------------------------------------------------------------------------
// K5 (single-ssd) — strict-lower intra-chunk + γ-correction + state-to-output
// ---------------------------------------------------------------------------

/// SingleSsd chunk scan.
///
/// Computes the per-chunk output from three contributions:
/// - **Strict lower triangular intra-chunk** (`t1 > t2`):
///   `(cb[i,j] · scale[t2] · exp(cumA[t1] − cumA[t2])) · V[t2]`
/// - **Same-time-step (`t1 == t2`) γ-correction**:
///   `γ[t] · (Σₙ C[t,r_out,n] · B[t,r_in,n]) · V[t,r_in,p]`
/// - **State-to-output (Y_off)** — same formula as the double-ssd K5:
///   `exp(cumA[t]) · C[t] · h'[n-1]`
///
/// `cb_bnhLMLM` is the unscaled `C · Bᵀ` matrix from K2; `b_bnlmhr` is the
/// unscaled K/B tensor (used for the γ-correction matmul). The strict-lower
/// MIMO mask excludes the same-step `R × R` block, leaving only `t1 > t2`
/// contributions in the masked CB.
///
/// # Shapes
/// - `da_cumsum_bhnl`: `[B, H, N, L]` (base time grid, not fused)
/// - `v_bnlmhp`: `[B, N, L, M, H, P]`
/// - `c_bnlmhr`, `b_bnlmhr`: `[B, N, L, M, H, R]`
/// - `cb_bnhLMLM`: `[B, N, H, L·M, L·M]` (output of K2)
/// - `gamma_bnlh`, `scale_bnlh`: `[B, N, L, H]`
/// - `chunk_input_state_bnhpr`: `[B, N, H, P, R]` (h' at chunk start)
///
/// # Returns
/// - `y_bnlmhp`: `[B, N, L, M, H, P]`
#[allow(clippy::too_many_arguments)]
pub fn k5_single_ssd_chunk_scan<B: Backend>(
    da_cumsum_bhnl: Tensor<B, 4>,
    v_bnlmhp: Tensor<B, 6>,
    c_bnlmhr: Tensor<B, 6>,
    b_bnlmhr: Tensor<B, 6>,
    cb_bnhLMLM: Tensor<B, 5>,
    gamma_bnlh: Tensor<B, 4>,
    scale_bnlh: Tensor<B, 4>,
    chunk_input_state_bnhpr: Tensor<B, 5>,
) -> Tensor<B, 6> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = c_bnlmhr.dims();
    let device = v_bnlmhp.device();
    let fused = chunk_len * mimo_rank;

    // Fuse mimo_rank into chunk_len for the SSM-style matmul.
    let v_bnLMhp = v_bnlmhp
        .clone()
        .reshape([batch, nchunks, fused, nheads, per_head_dim]);
    let c_bnLMhr = c_bnlmhr
        .clone()
        .reshape([batch, nchunks, fused, nheads, state_rank]);

    // Per-fused-step cumulative decay (interleave-expand the base grid).
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
        .reshape([batch, nheads, nchunks, fused]);

    // ── Y_off: exp(cumA[t]) · C[t] · h'[n-1]  (same form as double-ssd K5) ──
    let exp_da_bnhLMp = da_cumsum_bhnLM
        .clone()
        .exp()
        .permute([0, 2, 1, 3]) // bnhLM
        .unsqueeze_dim::<5>(4) // bnhLM1
        .expand([batch, nchunks, nheads, fused, per_head_dim]);

    let c_bnhLMr = c_bnLMhr.permute([0, 1, 3, 2, 4]);
    let chunk_input_state_bnhrp = chunk_input_state_bnhpr.permute([0, 1, 2, 4, 3]);
    let ch_bnhLMp = c_bnhLMr.matmul(chunk_input_state_bnhrp);
    let y_off_bnhLMp = ch_bnhLMp * exp_da_bnhLMp;

    // ── Y_lower: strict lower-tri intra-chunk with scale and decay ────────
    //
    // Mask `cb` to keep only `t1 > t2`, multiply by `exp(cumA[t1] - cumA[t2])`
    // and by `scale[t2]` along the source axis, then matmul with V.
    let da_cumsum_bnhLM = da_cumsum_bhnLM.permute([0, 2, 1, 3]); // bnhLM
    let target_da_cumsum_bnhLMLM = da_cumsum_bnhLM
        .clone()
        .unsqueeze_dim::<5>(4) // bnhLM1
        .expand([batch, nchunks, nheads, fused, fused]);
    let source_da_cumsum_bnhLMLM = da_cumsum_bnhLM
        .unsqueeze_dim::<5>(3) // bnh1LM
        .expand([batch, nchunks, nheads, fused, fused]);
    let diff_bnhLMLM = target_da_cumsum_bnhLMLM - source_da_cumsum_bnhLMLM;

    // Strict-upper -inf mask on the base time grid (`t1 <= t2` → -inf),
    // then interleave-expand to fused length so that MIMO same-time blocks
    // are zeroed out.
    let inf_upper_ll =
        Tensor::<B, 2>::full([chunk_len, chunk_len], f32::NEG_INFINITY, &device).triu(0); // upper triangle INCLUDING diagonal
    let inf_upper_bnhll = inf_upper_ll
        .unsqueeze_dims::<5>(&[0, 1, 2])
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    let inf_upper_bnhLMLM = inf_upper_bnhll
        .unsqueeze_dim::<6>(4)
        .expand([batch, nchunks, nheads, chunk_len, mimo_rank, chunk_len])
        .reshape([batch, nchunks, nheads, fused, chunk_len])
        .unsqueeze_dim::<6>(5)
        .expand([batch, nchunks, nheads, fused, chunk_len, mimo_rank])
        .reshape([batch, nchunks, nheads, fused, fused]);
    let decay_strict_bnhLMLM = (diff_bnhLMLM + inf_upper_bnhLMLM).exp();

    // Per-column scale: `scale[t2]` lives on the source axis (column).
    let scale_bnhLM = scale_bnlh
        .permute([0, 1, 3, 2]) // bnhl
        .unsqueeze_dim::<5>(4) // bnhl1
        .expand([batch, nchunks, nheads, chunk_len, mimo_rank])
        .reshape([batch, nchunks, nheads, fused]);
    let scale_col_bnhLMLM = scale_bnhLM
        .unsqueeze_dim::<5>(3) // bnh1LM
        .expand([batch, nchunks, nheads, fused, fused]);

    let kernel_bnhLMLM = decay_strict_bnhLMLM * scale_col_bnhLMLM;
    let masked_cb_bnhLMLM = cb_bnhLMLM * kernel_bnhLMLM;

    let v_bnhLMp = v_bnLMhp.permute([0, 1, 3, 2, 4]);
    let y_lower_bnhLMp = masked_cb_bnhLMLM.matmul(v_bnhLMp);

    // ── Y_diag: γ-weighted same-step correction ───────────────────────────
    //
    // y_diag[t, m_out, h, p] = γ[t] · Σ_{m_in} (Σ_n C[t,m_out,n] · B[t,m_in,n]) · V[t,m_in,p]
    //
    // Computed fresh (small same-step matmul) rather than extracting the
    // block-diagonal from `cb_bnhLMLM` (which would require a fiddly reshape).
    let c_bnlhmr = c_bnlmhr.permute([0, 1, 2, 4, 3, 5]);
    let b_bnlhrm = b_bnlmhr.permute([0, 1, 2, 4, 5, 3]);
    let qk_dot_bnlhmM = c_bnlhmr.matmul(b_bnlhrm); // bnlhm_outm_in
    let v_bnlhmp = v_bnlmhp.permute([0, 1, 2, 4, 3, 5]);
    let y_d_bnlhmp = qk_dot_bnlhmM.matmul(v_bnlhmp); // bnlhm_outp
    let gamma_bnlh11 = gamma_bnlh.unsqueeze_dims::<6>(&[4, 5]);
    let y_d_bnlhmp_scaled = y_d_bnlhmp * gamma_bnlh11;

    // Back to fused layout `[B, N, H, L·M, P]` to match y_lower / y_off.
    let y_diag_bnlmhp = y_d_bnlhmp_scaled.permute([0, 1, 2, 4, 3, 5]);
    let y_diag_bnLMhp = y_diag_bnlmhp.reshape([batch, nchunks, fused, nheads, per_head_dim]);
    let y_diag_bnhLMp = y_diag_bnLMhp.permute([0, 1, 3, 2, 4]);

    // ── Combine and reshape ───────────────────────────────────────────────
    let y_bnhLMp = y_off_bnhLMp + y_lower_bnhLMp + y_diag_bnhLMp;
    let y_bnLMhp = y_bnhLMp.permute([0, 1, 3, 2, 4]);
    y_bnLMhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim])
}
