//! # Single-Pass SSD (Minimal / segsum variant)
//!
//! This is the MIMO-first, single SSD pass implementation of the
//! Mamba-3 trapezoid recurrence. It is the Burn analogue of the official
//! Tilelang MIMO kernel and Triton SISO kernel; SISO is the `mimo_rank = 1`
//! degenerate case.
//!
//! ## Background — the single-ssd recurrence
//!
//! The double-ssd trapezoid hidden state is
//!
//! ```text
//!   hₜ = αₜ hₜ₋₁ + βₜ (Bₜ₋₁ ⊗ xₜ₋₁) + γₜ (Bₜ ⊗ xₜ)
//! ```
//!
//! Expanding the recurrence and grouping by `(Bₛ ⊗ xₛ)` gives the coefficient
//! `(Πᵣ₌ₛ₊₁ᵗ αᵣ) · [γₛ + (1−λₛ₊₁)·Δₛ₊₁]` for the contribution of step `s` to
//! state `t` (for `s < t`). At `s = t` the coefficient is just `γₜ`.
//!
//! Define `scaleₜ = γₜ + (1−λₜ₊₁)·Δₜ₊₁` (with `scaleₜ = γₜ` at the last
//! position). The single-SSD
//!
//! ```text
//!   h'ₜ = αₜ h'ₜ₋₁ + scaleₜ (Bₜ ⊗ xₜ)
//! ```
//!
//! produces the same outputs `yₜ = Cₜᵀ h'ₜ` as the double-ssd one **except**
//! at the same-step diagonal (`s = t`), where the single-ssd form has `scaleₜ`
//! instead of `γₜ`. We compensate by:
//!
//! 1. Using a **strict** lower-triangular mask in the intra-chunk path (the
//!    `s = t` block is excluded from the trapezoid sum).
//! 2. Adding a separate γ-weighted same-step term `γₜ · (Cₜᵀ Bₜ) · xₜ`.
//!
//! ## Algorithm (per chunk, MIMO-first)
//!
//! ```text
//!   K_scaled[t, m, h, n] = scaleₜ · B[t, m, h, n]     // K scaled inside the SSD
//!
//!   y_lower  = (C ⊗ K_scaledᵀ ⊙ L_strict) · PsiV     // strict lower-tri
//!   y_diag   = γₜ · (C ⊗ Bᵀ at same step) · PsiV      // diagonal correction
//!   y_off    = C · h'_chunk_in · exp(da_cs)           // state-to-output
//!
//!   y        = y_lower + y_diag + y_off
//!
//!   h'_chunk_out = exp(da_cs_last) · h'_chunk_in
//!                + K_scaled · exp(da_cs_rev)ᵀ · PsiV   // standard state update
//! ```
//!
//! The MIMO causal mask is identical to [`crate::mamba3::double_ssd::ssd::minimal`] but
//! with a stricter inequality (`i_time > j_time` rather than `i_time ≥ j_time`).
//!
//! Reference implementations:
//! - SISO: `refs/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`
//! - MIMO: `refs/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`

use crate::mamba3::single_ssd::prelude::*;
use crate::modules::segsum;
use burn::prelude::*;

impl Mamba3SingleSsdInput {
    /// MIMO-first single-SSD — segsum variant.
    ///
    /// See module documentation for the algorithm. Returns the chunked outputs
    /// and the final single-ssd accumulator.
    ///
    /// # Shapes
    /// - input: see [`Mamba3SingleSsdInput`]
    /// - output `(y_bnlmhp, final_state_bhpr)`:
    ///   - `y_bnlmhp`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    ///   - `final_state_bhpr`:   `[batch, nheads, per_head_dim, state_rank]`
    #[allow(non_snake_case)]
    pub fn single_ssd_minimal(self) -> (Tensor<6>, Tensor<4>) {
        let input = self;
        input.sanity();
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = input.v_bnlmhp.dims();
        let [.., state_rank] = input.b_bnlmhr.dims();
        let device = &input.v_bnlmhp.device();

        assert!(nchunks >= 1, "sequence must be non-empty");
        assert!(chunk_len > 0, "chunk_len must be positive");
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

        // ── Fuse mimo_rank into chunk_len (matches `ssd_minimal`) ─────────────
        let c_bnLMhr = input.c_bnlmhr.clone().reshape([
            batch,
            nchunks,
            chunk_len * mimo_rank,
            nheads,
            state_rank,
        ]);
        let v_bnLMhp = input.v_bnlmhp.clone().reshape([
            batch,
            nchunks,
            chunk_len * mimo_rank,
            nheads,
            per_head_dim,
        ]);

        // Per-time-step cumulative log-decay (used for L_strict, decay_states, y_off)
        let a_bhnl = input.da_bnlh.permute([0, 3, 1, 2]);
        let a_cumsum_bhnl = a_bhnl.clone().cumsum(3);

        // K scaled for lower-triangular and state recurrence paths
        // (the diagonal correction reuses the unscaled `b_bnlmhr`).
        // scale_bnlh broadcast over (mimo_rank, state_rank):
        let scale_bnlh11 = input
            .scale_bnlh
            .clone()
            .unsqueeze_dims::<6>(&[3, 5]) // scale_bnlh -> scale_bnl1h1
            ;
        let k_scaled_bnlmhr = input.b_bnlmhr.clone() * scale_bnlh11;
        let k_scaled_bnLMhr =
            k_scaled_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

        // =============================================================
        // STEP 1a: Strict lower-triangular intra-chunk output (y_lower)
        //
        // y_lower[t1] = Σ_{t2 < t1}  (C[t1] · K_scaled[t2]^T)
        //              · exp(cumA[t1] - cumA[t2])  · PsiV[t2]
        // (block-diagonal in time t1 = t2 is excluded — handled by y_diag.)
        // =============================================================
        let y_lower_bnLMhp = {
            let c_bnhLMr = c_bnLMhr.clone().permute([0, 1, 3, 2, 4]);
            let k_bnhrLM = k_scaled_bnLMhr.clone().permute([0, 1, 3, 4, 2]);
            // [batch, nchunks, nheads, chunk_len*mimo_rank, chunk_len*mimo_rank]
            let cb_bnhLMLM = c_bnhLMr.matmul(k_bnhrLM);

            // L_strict_base[i, j] = exp(cumA[i] - cumA[j]) for i > j, else 0.
            //
            // Like `segsum` but with -inf on the diagonal as well (so exp = 0
            // there). Replaces the existing `triu(1)` masking with `triu(0)`.
            let l_strict_base_bhnll = {
                let x_cumsum = a_bhnl.clone().cumsum(3);
                let row: Tensor<5> = x_cumsum.clone().unsqueeze_dim(4); // [..., l, 1]
                let col: Tensor<5> = x_cumsum.unsqueeze_dim(3); // [..., 1, l]
                let diff = row - col; // [..., l, l]
                let neg_inf_strict = Tensor::full_like(&diff, f32::NEG_INFINITY).triu(0);
                (diff + neg_inf_strict).exp()
            };

            // Interleave-expand to fused length (L_strict[i,j] = L_strict_base[i//m, j//m]):
            let l_strict_bhnLMLM = l_strict_base_bhnll
                .unsqueeze_dim::<6>(4)
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank, chunk_len])
                .reshape([batch, nheads, nchunks, chunk_len * mimo_rank, chunk_len])
                .unsqueeze_dim::<6>(5)
                .expand([
                    batch,
                    nheads,
                    nchunks,
                    chunk_len * mimo_rank,
                    chunk_len,
                    mimo_rank,
                ])
                .reshape([
                    batch,
                    nheads,
                    nchunks,
                    chunk_len * mimo_rank,
                    chunk_len * mimo_rank,
                ]);

            // (CB ⊙ L_strict) · V    (back in MIMO-fused layout)
            let cb_bnLMhLM = cb_bnhLMLM.permute([0, 1, 3, 2, 4]);
            let l_bnLMhLM = l_strict_bhnLMLM.permute([0, 2, 3, 1, 4]);
            let masked_cb_bnhLMLM = (cb_bnLMhLM * l_bnLMhLM).permute([0, 1, 3, 2, 4]);

            let v_bnhLMp = v_bnLMhp.clone().permute([0, 1, 3, 2, 4]);
            let y_lower_bnhLMp = masked_cb_bnhLMLM.matmul(v_bnhLMp);

            y_lower_bnhLMp.permute([0, 1, 3, 2, 4]) // y_lower_bnLMhp
        };

        // =============================================================
        // STEP 1b: γ-weighted same-step diagonal correction (y_diag)
        //
        // y_diag[t, m_out, h, p] = γₜ · Σ_{m_in} (C[t, m_out, h, ·] · B[t, m_in, h, ·]) · PsiV[t, m_in, h, p]
        // =============================================================
        let y_diag_bnlmhp = {
            // C @ B^T contracts over state_rank, leaving mimo_rank on both sides.
            // c_bnlmhr  [b, n, l, m, h, r] -> c_bnlhmr  [b, n, l, h, m, r]
            // b_bnlmhr  [b, n, l, m, h, r] -> b_bnlhrm  [b, n, l, h, r, m]
            let c_bnlhmr = input.c_bnlmhr.permute([0, 1, 2, 4, 3, 5]);
            let b_bnlhrm = input.b_bnlmhr.permute([0, 1, 2, 4, 5, 3]);
            // qk_dot_bnlhmM [b, n, l, h, m_out, m_in]
            let qk_dot_bnlhmM = c_bnlhmr.matmul(b_bnlhrm);

            // V in [b, n, l, h, m_in, p] layout for the next matmul:
            let v_bnlhmp = input.v_bnlmhp.permute([0, 1, 2, 4, 3, 5]);
            // (qk_dot) · V → [b, n, l, h, m_out, p]
            let y_d_bnlhmp = qk_dot_bnlhmM.matmul(v_bnlhmp);

            // Multiply by γₜ (per (batch, nchunks, chunk_len, nheads)):
            let gamma_bnlh11 = input.gamma_bnlh.clone().unsqueeze_dims::<6>(&[4, 5]);
            let y_d_bnlhmp_scaled = y_d_bnlhmp * gamma_bnlh11;

            // Back to [b, n, l, m, h, p]:
            y_d_bnlhmp_scaled.permute([0, 1, 2, 4, 3, 5])
        };
        // Reshape to fused layout for combination with y_lower / y_off.
        let y_diag_bnLMhp =
            y_diag_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);

        // =============================================================
        // STEP 2: Per-chunk single-ssd state (standard SSD with K_scaled)
        //
        // s[n] = Σ_{t,m} exp(cumA[n,-1] - cumA[n,t]) · V[n,t*M+m] · K_scaled[n,t*M+m]^T
        // =============================================================
        let state_bnhpr = {
            let a_cumsum_last_bhn1 = a_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
            let a_cumsum_bhnLM = a_cumsum_bhnl
                .clone()
                .unsqueeze_dim::<5>(4)
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
                .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]);
            let decay_bhnLM = (a_cumsum_last_bhn1 - a_cumsum_bhnLM).exp();

            let decay_bnLMh1 = decay_bhnLM.permute([0, 2, 3, 1]).unsqueeze_dim(4);
            let decayed_v_bnLMhp = decay_bnLMh1 * v_bnLMhp.clone();

            let decayed_v_bnhpLM = decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);
            let k_scaled_bnhLMr = k_scaled_bnLMhr.permute([0, 1, 3, 2, 4]);
            decayed_v_bnhpLM.matmul(k_scaled_bnhLMr) // state_bnhpr
        };

        // =============================================================
        // STEP 3: Inter-chunk state scan (segsum-based state passing)
        //
        // h'[n] = Ā_chunk[n] · h'[n-1] + s[n]
        // =============================================================
        let (state_bnhpr, final_state_bhpr) = {
            let initial_state_b1hpr = input.initial_state_bhpr.unsqueeze_dim(1);
            let initial_state_b1hpr = if let Some(init_hpr) = input.init_state_hpr {
                let init_b1hpr = init_hpr.unsqueeze_dim::<4>(0).expand([
                    batch,
                    1,
                    nheads,
                    per_head_dim,
                    state_rank,
                ]);
                initial_state_b1hpr + init_b1hpr
            } else {
                initial_state_b1hpr
            };

            let state_bNhpr = Tensor::cat(vec![initial_state_b1hpr, state_bnhpr], 1);

            let a_cumsum_last_bhn: Tensor<3> = a_cumsum_bhnl
                .clone()
                .slice(s![.., .., .., -1])
                .squeeze_dim(3);
            let a_chunk_pad_bhN = Tensor::cat(
                vec![Tensor::zeros([batch, nheads, 1], device), a_cumsum_last_bhn],
                2,
            );
            let decay_chunk_bhNN = segsum::<3, 4>(a_chunk_pad_bhN).exp();

            let flat = per_head_dim * state_rank;
            let state_bhNPR = state_bNhpr.clone().permute([0, 2, 1, 3, 4]).reshape([
                batch,
                nheads,
                1 + nchunks,
                flat,
            ]);

            let new_state_bhNPR = decay_chunk_bhNN.matmul(state_bhNPR);
            let new_state_bhNpr =
                new_state_bhNPR.reshape([batch, nheads, 1 + nchunks, per_head_dim, state_rank]);

            let new_state_bnhpr = new_state_bhNpr
                .clone()
                .slice(s![.., .., 0..nchunks, .., ..])
                .permute([0, 2, 1, 3, 4]);
            let last_state_bhpr: Tensor<4> = new_state_bhNpr
                .slice(s![.., .., nchunks, .., ..])
                .squeeze_dim(2);

            (new_state_bnhpr, last_state_bhpr)
        };

        // =============================================================
        // STEP 4: State-to-output (y_off)
        //
        // y_off[n, t*M+m] = C[t*M+m]^T · exp(cumA[t]) · h'[n-1]
        // =============================================================
        let y_off_bnLMhp = {
            let state_decay_bhnLM = a_cumsum_bhnl
                .unsqueeze_dim::<5>(4)
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
                .reshape([batch, nheads, nchunks, chunk_len * mimo_rank])
                .exp();

            let c_bnhLMr = c_bnLMhr.permute([0, 1, 3, 2, 4]);
            let state_bnhrp = state_bnhpr.permute([0, 1, 2, 4, 3]);
            let ch_bnhLMp = c_bnhLMr.matmul(state_bnhrp);

            let decay_bnhLM1 = state_decay_bhnLM.permute([0, 2, 1, 3]).unsqueeze_dim(4);
            let y_off_bnhLMp = ch_bnhLMp * decay_bnhLM1;
            y_off_bnhLMp.permute([0, 1, 3, 2, 4])
        };

        // ── Combine and reshape ───────────────────────────────────────────────
        let y_bnLMhp = y_lower_bnLMhp + y_diag_bnLMhp + y_off_bnLMhp;
        let y_bnlmhp =
            y_bnLMhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

        (y_bnlmhp, final_state_bhpr)
    }
}
