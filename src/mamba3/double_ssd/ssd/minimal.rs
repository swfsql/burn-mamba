//! ## The Chunkwise MIMO-SSD Algorithm (Minimal/Segsum variant)
//!
//! During training (and prefill), a naive sequential recurrence cannot
//! exploit GPU tensor cores.  The **chunkwise SSD algorithm** achieves this
//! by splitting the sequence into chunks of length chunk_len and decomposing the
//! computation into four steps.
//!
//! ```text
//!   Step 1  (intra-chunk, MIMO quadratic form)  →  Y_diag   [batch, nchunks, chunk_len*mimo_rank, nheads, per_head_dim]
//!   Step 2  (input → chunk state)               →  state    [batch, nchunks, nheads, per_head_dim, state_rank]
//!   Step 3  (inter-chunk state scan)            →  state    [batch, nchunks, nheads, per_head_dim, state_rank], final_state
//!   Step 4  (chunk state → output)              →  Y_off    [batch, nchunks, chunk_len*mimo_rank, nheads, per_head_dim]
//!
//!   Y = Y_diag + Y_off   →  reshape to [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]
//! ```
//!
//! The MIMO causal mask `LM_mimo[i,j] = exp(cumA[i//m] - cumA[j//m])` for `i//m >= j//m`
//! allows all mimo_ranks ranks at the same time step to attend to each other while
//! maintaining causal ordering across time steps.

use crate::mamba3::double_ssd::prelude::*;
use crate::utils::segsum::segsum;
use burn::prelude::*;
use burn::backend::Backend;

impl Mamba3DoubleSsdInput {
    /// MIMO-first chunkwise SSD — minimal/segsum variant.
    ///
    /// Implements the four-step decomposition for the MIMO (double-ssd) trapezoidal recurrence.
    /// SISO (mimo_rank=1) is the degenerate case where the fused length equals the chunk length.
    ///
    /// No D skip is applied here — the caller handles it.
    ///
    /// # Shapes
    /// - input: see [`Mamba3DoubleSsdInput`]
    /// - output.0 `y_bnlrhp`:       `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
    /// - output.1 `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    #[allow(non_snake_case)]
    pub fn double_ssd_minimal(self) -> (Tensor<6>, Tensor<4>) {
        let input = self;
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = input.v_bnlmhp.dims();
        let [.., state_rank] = input.b_bnlmhr.dims();
        // note: L above denotes the chunk_len
        let device = &input.v_bnlmhp.device();

        assert!(nchunks >= 1, "sequence must be non-empty");
        assert!(chunk_len > 0, "chunk_len must be positive");

        // ── Fuse mimo_rank into chunk_len ────────────────────────────────────────
        let b_bnLMhr =
            input
                .b_bnlmhr
                .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
        let c_bnLMhr =
            input
                .c_bnlmhr
                .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
        let v_bnLMhp =
            input
                .v_bnlmhp
                .reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);

        // Base per-time-step cumulative log-decay
        let a_bhnl = input.da_bnlh.clone().permute([0, 3, 1, 2]);
        let a_cumsum_bhnl = a_bhnl.clone().cumsum(3);

        // =============================================================
        // STEP 1: Intra-chunk outputs (Y_diag)
        //
        // Y_diag[m] = (L_mimo[m] ∘ C[m] B[m]ᵀ) · V[m]
        // note: L above does not denote the chunk_len, but L in the Mamba-3 paper.
        //
        // MIMO mask: L_mimo[i,j] = exp(cumA[i//m] - cumA[j//m]) if i//m >= j//m, else 0
        // =============================================================
        let y_diag_bnLMhp = {
            // CB = C @ B^T: contract over state_rank
            let c_bnhLMr = c_bnLMhr.clone().permute([0, 1, 3, 2, 4]);
            let b_bnhrLM = b_bnLMhr.clone().permute([0, 1, 3, 4, 2]);
            // [batch, nchunks, nheads, chunk_len*mimo_rank, chunk_len*mimo_rank]
            let cb_bnhLMLM = c_bnhLMr.matmul(b_bnhrLM);

            // Build MIMO causal mask from segsum on base dimension, then interleave-expand.
            // l_base_bhnll[i,j] = exp(cumA[i] - cumA[j]) if i >= j, else 0
            let l_base_bhnll = segsum::<4, 5>(a_bhnl.clone()).exp();

            // Interleave-expand
            // L_mimo[i, j] = L_base[i//m, j//m]  (same decay for all ranks at a given time)
            let l_mimo_bhnLMLM = l_base_bhnll
                // row interleaving: insert mimo_rank copies of each l-row
                .unsqueeze_dim::<6>(4) // l_base_bhnl1l
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank, chunk_len]) // l_base_bhnlml
                .reshape([batch, nheads, nchunks, chunk_len * mimo_rank, chunk_len]) // l_base_bhnLMl
                // col interleaving: insert mimo_rank copies of each l-col
                .unsqueeze_dim::<6>(5) // l_base_bhnLMl1
                .expand([
                    batch,
                    nheads,
                    nchunks,
                    chunk_len * mimo_rank,
                    chunk_len,
                    mimo_rank,
                ]) // l_base_bhnLMlm
                .reshape([
                    batch,
                    nheads,
                    nchunks,
                    chunk_len * mimo_rank,
                    chunk_len * mimo_rank,
                ]); // l_base_bhnLMLM

            // Apply mask: (CB ∘ L_mimo) · V
            let cb_bnLMhLM = cb_bnhLMLM.permute([0, 1, 3, 2, 4]);
            let l_bnLMhLM = l_mimo_bhnLMLM.permute([0, 2, 3, 1, 4]);
            let masked_cb_bnhLMLM = (cb_bnLMhLM * l_bnLMhLM).permute([0, 1, 3, 2, 4]);

            let v_bnhLMp = v_bnLMhp.clone().permute([0, 1, 3, 2, 4]);
            let y_diag_bnhLMp = masked_cb_bnhLMLM.matmul(v_bnhLMp);

            y_diag_bnhLMp.permute([0, 1, 3, 2, 4])
        };

        // =============================================================
        // STEP 2: Chunk state (input → state, zero initial state)
        //
        // s[n] = Σ_{t,r} exp(cumA[n,-1] - cumA[n,t]) · V[n,t*m+r] · B[n,t*m+r]ᵀ
        //      (outer product over per_head_dim and state_rank)
        // =============================================================
        let state_bnhpr = {
            // Decay from each fused position to end of chunk:
            let a_cumsum_last_bhn1 = a_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
            // Expand base cumsum to fused length (each time repeated mimo_rank times):
            // [b, nheads, n, l] → [b, nheads, n, l, R] → [b, nheads, n, L]
            let a_cumsum_bhnLM = a_cumsum_bhnl
                .clone()
                .unsqueeze_dim::<5>(4) // a_cumsum_bhnl1
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // a_cumsum_bhnlm
                .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // a_cumsum_bhnLM
            let decay_bhnLM = (a_cumsum_last_bhn1 - a_cumsum_bhnLM).exp();

            // Multiply decay into V
            let decay_bnLMh1 = decay_bhnLM
                .permute([0, 2, 3, 1]) // decay_bnLMh
                .unsqueeze_dim(4); // decay_bnLMh1
            let decayed_v_bnLMhp = decay_bnLMh1 * v_bnLMhp.clone();

            // state = decayed_V^T @ B
            let decayed_v_bnhpLM = decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);
            let b_bnhLMr = b_bnLMhr.permute([0, 1, 3, 2, 4]);
            decayed_v_bnhpLM.matmul(b_bnhLMr) // state_bnhpr
        };

        // =============================================================
        // STEP 3: Inter-chunk state scan (state passing via segsum)
        //
        // h[n] = Ā_chunk[n] · h[n-1] + s[n]
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

            // Prepend initial state: [batch, 1+nchunks, nheads, per_head_dim, state_rank]
            let state_bNhpr = Tensor::cat(vec![initial_state_b1hpr, state_bnhpr], 1);

            // Per-chunk cumulative decay (last position of each chunk)
            let a_cumsum_last_bhn: Tensor<3> = a_cumsum_bhnl
                .clone()
                .slice(s![.., .., .., -1]) // a_cumsum_last_bhn1
                .squeeze_dim(3); // a_cumsum_last_bhn
            // Prepend zero for the initial state (no decay before chunk 0):
            let a_chunk_pad_bhN = Tensor::cat(
                vec![Tensor::zeros([batch, nheads, 1], device), a_cumsum_last_bhn],
                2,
            ); // [batch, nheads, 1+nchunks]

            // Inter-chunk decay matrix via segsum: [batch, nheads, 1+nchunks, 1+nchunks]
            let decay_chunk_bhNN = segsum::<3, 4>(a_chunk_pad_bhN).exp();

            // Flatten (per_head_dim, state_rank) for matmul
            let flat = per_head_dim * state_rank;
            let state_bhNPR = state_bNhpr
                .clone()
                .permute([0, 2, 1, 3, 4]) // state_bhNpr
                .reshape([batch, nheads, 1 + nchunks, flat]); // [batch, nheads, 1+nchunks, per_head_dim*state_rank]

            let new_state_bhNPR = decay_chunk_bhNN.matmul(state_bhNPR);
            let new_state_bhNpr =
                new_state_bhNPR.reshape([batch, nheads, 1 + nchunks, per_head_dim, state_rank]);

            // Split: chunk input states [0..n], final state [n]
            let new_state_bnhpr = new_state_bhNpr
                .clone()
                .slice(s![.., .., 0..nchunks, .., ..]) // new_state_bhnpr
                .permute([0, 2, 1, 3, 4]); // new_state_bnhpr
            let last_state_bhpr: Tensor<4> = new_state_bhNpr
                .slice(s![.., .., nchunks, .., ..]) // new_state_bh1pr
                .squeeze_dim(2); // last_state_bhpr

            (new_state_bnhpr, last_state_bhpr)
        };

        // =============================================================
        // STEP 4: State-to-output (Y_off)
        //
        // Y_off[n, t*m+r] = C[t*m+r]ᵀ · exp(cumA[t]) · h[n-1]
        // =============================================================
        let y_off_bnLMhp = {
            // Expand base cumsum to fused, then exp:
            let state_decay_bhnLM = a_cumsum_bhnl
                .clone()
                .unsqueeze_dim::<5>(4) // a_cumsum_bhnl1
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // a_cumsum_bhnlm
                .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]) // a_cumsum_bhnLM
                .exp();

            // C
            let c_bnhLMr = c_bnLMhr.permute([0, 1, 3, 2, 4]);
            let state_bnhrp = state_bnhpr.permute([0, 1, 2, 4, 3]);
            let ch_bnhLMp = c_bnhLMr.matmul(state_bnhrp);

            // Multiply by intra-chunk decay
            let decay_bnhLM1 = state_decay_bhnLM
                .permute([0, 2, 1, 3]) // state_decay_bnhLM
                .unsqueeze_dim(4); // state_decay_bnhLM1
            let y_off_bnhLMp = ch_bnhLMp * decay_bnhLM1;
            y_off_bnhLMp.permute([0, 1, 3, 2, 4]) // y_off_bnLMhp
        };

        // ── Combine and reshape ───────────────────────────────────────────────
        let y_bnLMhp = y_diag_bnLMhp + y_off_bnLMhp;
        let y_bnlmhp =
            y_bnLMhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

        (y_bnlmhp, final_state_bhpr)
    }
}
