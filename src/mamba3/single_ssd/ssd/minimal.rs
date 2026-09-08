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

use crate::mamba3::helpers;
use crate::mamba3::prelude::Mamba3SsdPath;
use crate::mamba3::single_ssd::prelude::*;
use crate::mamba3::single_ssd::ssd::diag::y_diag_correction;
use burn_stack::modules::segsum;
use burn::prelude::*;

impl Mamba3SingleSsdInput {
    /// MIMO-first single-SSD — segsum variant.
    ///
    /// See module documentation for the algorithm. Returns the chunked outputs
    /// and the final single-ssd accumulator.
    ///
    /// # Shapes
    /// `T = chunk_tokens = chunk_len / read_stride` is the chunk's read axis
    /// (see [`Mamba3SingleSsdInput::c_bntmhr`]); at `read_stride = 1` it is
    /// `chunk_len` and every shape here is the stock one.
    /// - input: see [`Mamba3SingleSsdInput`]
    /// - output `(y_bntmhp, final_state_bhpr)`:
    ///   - `y_bntmhp`:           `[batch, nchunks, T, mimo_rank, nheads, per_head_dim]`
    ///   - `final_state_bhpr`:   `[batch, nheads, per_head_dim, state_rank]`
    #[allow(non_snake_case)]
    pub fn single_ssd_minimal(self) -> (Tensor<6>, Tensor<4>) {
        let input = self;
        input.sanity();
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = input.v_bnlmhp.dims();
        let [.., state_rank] = input.b_bnlmhr.dims();
        let device = &input.v_bnlmhp.device();
        let read_stride = input.read_stride;
        let chunk_tokens = Mamba3SsdPath::chunk_tokens(chunk_len, read_stride);
        let fused = chunk_len * mimo_rank; // write (source) axis
        let read = chunk_tokens * mimo_rank; // read (target) axis

        assert!(nchunks >= 1, "sequence must be non-empty");
        assert!(chunk_len > 0, "chunk_len must be positive");
        assert_eq!(
            [batch, nchunks, chunk_tokens, nheads],
            input.gamma_bnth.dims(),
            "gamma must align with the read axis"
        );
        assert_eq!(
            [batch, nchunks, chunk_len, nheads],
            input.scale_bnlh.dims(),
            "scale must align with da"
        );

        // ── Fuse mimo_rank into the chunk axes (matches `ssd_minimal`) ────────
        let c_bnTMhr =
            input
                .c_bntmhr
                .clone()
                .reshape([batch, nchunks, read, nheads, state_rank]);
        let v_bnLMhp =
            input
                .v_bnlmhp
                .clone()
                .reshape([batch, nchunks, fused, nheads, per_head_dim]);

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
            k_scaled_bnlmhr.reshape([batch, nchunks, fused, nheads, state_rank]);

        // =============================================================
        // STEP 1a: Strict lower-triangular intra-chunk output (y_lower)
        //
        // y_lower[t1] = Σ_{t2 < t1}  (C[t1] · K_scaled[t2]^T)
        //              · exp(cumA[t1] - cumA[t2])  · PsiV[t2]
        // (block-diagonal in time t1 = t2 is excluded — handled by y_diag.)
        // =============================================================
        let y_lower_bnTMhp = {
            let c_bnhTMr = c_bnTMhr.clone().swap_dims(2, 3);
            let k_bnhrLM = k_scaled_bnLMhr.clone().permute([0, 1, 3, 4, 2]);
            // [batch, nchunks, nheads, T*mimo_rank, chunk_len*mimo_rank]
            let cb_bnhTMLM = c_bnhTMr.matmul(k_bnhrLM);

            // L_strict_base[i, j] = exp(cumA[row(i)] - cumA[j]) for row(i) > j,
            // else 0 — rectangular, the read rows against every source.
            //
            // Like `segsum` but with -inf on the diagonal as well (so exp = 0
            // there), i.e. `triu(0)` rather than `triu(1)`.
            let l_strict_base_bhntl = {
                let x_cumsum = a_bhnl.clone().cumsum(3);
                let row: Tensor<5> =
                    helpers::read_rows::<4, 5>(x_cumsum.clone(), 3, read_stride).unsqueeze_dim(4); // [..., t, 1]
                let col: Tensor<5> = x_cumsum.unsqueeze_dim(3); // [..., 1, l]
                let diff = row - col; // [..., t, l]
                let neg_inf_strict =
                    helpers::read_causal_mask(chunk_len, read_stride, 0, device).unsqueeze::<5>();
                (diff + neg_inf_strict).exp()
            };

            // Interleave-expand both axes (L_strict[i,j] = L_strict_base[i//m, j//m]):
            let l_strict_bhnTMLM = l_strict_base_bhntl
                .unsqueeze_dim::<6>(4)
                .expand([batch, nheads, nchunks, chunk_tokens, mimo_rank, chunk_len])
                .reshape([batch, nheads, nchunks, read, chunk_len])
                .unsqueeze_dim::<6>(5)
                .expand([batch, nheads, nchunks, read, chunk_len, mimo_rank])
                .reshape([batch, nheads, nchunks, read, fused]);

            // (CB ⊙ L_strict) · V    (back in MIMO-fused layout)
            let cb_bnTMhLM = cb_bnhTMLM.swap_dims(2, 3);
            let l_bnTMhLM = l_strict_bhnTMLM.permute([0, 2, 3, 1, 4]);
            let masked_cb_bnhTMLM = (cb_bnTMhLM * l_bnTMhLM).swap_dims(2, 3);

            let v_bnhLMp = v_bnLMhp.clone().swap_dims(2, 3);
            let y_lower_bnhTMp = masked_cb_bnhTMLM.matmul(v_bnhLMp);

            y_lower_bnhTMp.swap_dims(2, 3) // y_lower_bnTMhp
        };

        // =============================================================
        // STEP 1b: γ-weighted same-step diagonal correction (y_diag)
        //
        // y_diag[t, m_out, h, p] = γₜ · Σ_{m_in} (C[t, m_out, h, ·] · B[t, m_in, h, ·]) · PsiV[t, m_in, h, p]
        // =============================================================
        // Same step means the *read* row, so V and B enter at those positions.
        let y_diag_bntmhp = y_diag_correction(
            helpers::read_rows::<6, 7>(input.v_bnlmhp, 2, read_stride),
            helpers::read_rows::<6, 7>(input.b_bnlmhr, 2, read_stride),
            input.c_bntmhr,
            input.gamma_bnth,
            input.siso_specialization,
        );
        // Reshape to fused layout for combination with y_lower / y_off.
        let y_diag_bnTMhp =
            y_diag_bntmhp.reshape([batch, nchunks, read, nheads, per_head_dim]);

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
            let k_scaled_bnhLMr = k_scaled_bnLMhr.swap_dims(2, 3);
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
            let state_bhNPR =
                state_bNhpr
                    .clone()
                    .swap_dims(1, 2)
                    .reshape([batch, nheads, 1 + nchunks, flat]);

            let new_state_bhNPR = decay_chunk_bhNN.matmul(state_bhNPR);
            let new_state_bhNpr =
                new_state_bhNPR.reshape([batch, nheads, 1 + nchunks, per_head_dim, state_rank]);

            let new_state_bnhpr = new_state_bhNpr
                .clone()
                .slice(s![.., .., 0..nchunks, .., ..])
                .swap_dims(1, 2);
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
        let y_off_bnTMhp = {
            let state_decay_bhnTM = helpers::read_rows::<4, 5>(a_cumsum_bhnl, 3, read_stride)
                .unsqueeze_dim::<5>(4)
                .expand([batch, nheads, nchunks, chunk_tokens, mimo_rank])
                .reshape([batch, nheads, nchunks, read])
                .exp();

            let c_bnhTMr = c_bnTMhr.swap_dims(2, 3);
            let state_bnhrp = state_bnhpr.transpose();
            let ch_bnhTMp = c_bnhTMr.matmul(state_bnhrp);

            let decay_bnhTM1 = state_decay_bhnTM.swap_dims(1, 2).unsqueeze_dim(4);
            let y_off_bnhTMp = ch_bnhTMp * decay_bnhTM1;
            y_off_bnhTMp.swap_dims(2, 3)
        };

        // ── Combine and reshape ───────────────────────────────────────────────
        let y_bnTMhp = y_lower_bnTMhp + y_diag_bnTMhp + y_off_bnTMhp;
        let y_bntmhp =
            y_bnTMhp.reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, per_head_dim]);

        (y_bntmhp, final_state_bhpr)
    }
}
