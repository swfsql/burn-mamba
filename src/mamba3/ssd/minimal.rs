//! ## The Chunkwise MIMO-SSD Algorithm (Minimal/Segsum variant)
//!
//! During training (and prefill), a naive sequential recurrence cannot
//! exploit GPU tensor cores.  The **chunkwise SSD algorithm** achieves this
//! by splitting the sequence into chunks of length Q and decomposing the
//! computation into four steps.
//!
//! For MIMO (mimo_rank=R>1), the rank dimension is fused into the chunk_len
//! dimension via an interleaved reshape: position `t*R+r` represents
//! (time=t, rank=r).  This gives a fused sequence length `L = Q·R` per chunk.
//! SISO (R=1) is the special case where `L = Q`.
//!
//! ```text
//!   Step 1  (intra-chunk, MIMO quadratic form)  →  Y_diag   [b, n, L, H, P]
//!   Step 2  (input → chunk state)               →  state    [b, n, H, P, N]
//!   Step 3  (inter-chunk state scan)            →  state    [b, n, H, P, N], final_state
//!   Step 4  (chunk state → output)              →  Y_off    [b, n, L, H, P]
//!
//!   Y = Y_diag + Y_off   →  reshape to [b, n, l, R, H, P]
//! ```
//!
//! The MIMO causal mask `L_mimo[i,j] = exp(cumA[i//R] - cumA[j//R])` for `i//R >= j//R`
//! allows all R ranks at the same time step to attend to each other while
//! maintaining causal ordering across time steps.

use crate::mamba3::prelude::*;
use burn::prelude::*;

impl<B: Backend> Mamba3<B> {
    /// MIMO-first chunkwise SSD — minimal/segsum variant.
    ///
    /// Implements the four-step decomposition for the MIMO trapezoidal recurrence.
    /// SISO (R=1) is the degenerate case where the fused length equals the chunk length.
    ///
    /// No D skip is applied here — the caller handles it.
    ///
    /// # Shapes
    /// - input: see [`Mamba3SsdInput`]
    /// - output.0 `y_bnlrhp`:       `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
    /// - output.1 `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    #[allow(non_snake_case)]
    pub fn ssd_minimal(input: super::Mamba3SsdInput<B>) -> (Tensor<B, 6>, Tensor<B, 4>) {
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = input.v_bnlrhp.dims();
        let [.., state_rank] = input.b_bnlrhn.dims();
        let fused_len = chunk_len * mimo_rank; // L = Q·R
        let device = &input.v_bnlrhp.device();

        assert!(nchunks >= 1, "sequence must be non-empty");
        assert!(chunk_len > 0, "chunk_len must be positive");

        // ── Fuse R into chunk_len ─────────────────────────────────────────────
        // [b, n, l, R, H, N] → [b, n, L, H, N]  where L = l*R
        let b_bnLhn = input
            .b_bnlrhn
            .reshape([batch, nchunks, fused_len, nheads, state_rank]);
        let c_bnLhn = input
            .c_bnlrhn
            .reshape([batch, nchunks, fused_len, nheads, state_rank]);
        // [b, n, l, R, H, P] → [b, n, L, H, P]
        let v_bnLhp = input
            .v_bnlrhp
            .reshape([batch, nchunks, fused_len, nheads, per_head_dim]);

        // Base per-time-step cumulative log-decay: [b, H, n, l]
        let a_bhnl = input.da_bnlh.clone().permute([0, 3, 1, 2]);
        let a_cumsum_bhnl = a_bhnl.clone().cumsum(3);

        // =============================================================
        // STEP 1: Intra-chunk outputs (Y_diag)
        //
        // Y_diag[n] = (L_mimo[n] ∘ C[n] B[n]ᵀ) · V[n]
        //
        // MIMO mask: L_mimo[i,j] = exp(cumA[i//R] - cumA[j//R]) if i//R >= j//R, else 0
        // =============================================================
        let y_diag_bnLhp = {
            // CB = C @ B^T: contract over state_rank N
            let c_bnhLr = c_bnLhn.clone().permute([0, 1, 3, 2, 4]); // [b, n, H, L, N]
            let b_bnhLr = b_bnLhn.clone().permute([0, 1, 3, 2, 4]); // [b, n, H, L, N]
            let b_bnhrL = b_bnhLr.permute([0, 1, 2, 4, 3]); // [b, n, H, N, L]
            let cb_bnhLL = c_bnhLr.matmul(b_bnhrL); // [b, n, H, L, L]

            // Build MIMO causal mask from segsum on base dimension, then interleave-expand.
            // l_base_bhnll[i,j] = exp(cumA[i] - cumA[j]) if i >= j, else 0
            let l_base_bhnll = segsum::<B, 4, 5>(a_bhnl.clone()).exp(); // [b, H, n, l, l]

            // Interleave-expand: [b, H, n, l, l] → [b, H, n, L, L]
            // L_mimo[i, j] = L_base[i//R, j//R]  (same decay for all ranks at a given time)
            let l_mimo_bhnLL = l_base_bhnll
                // row interleaving: insert R copies of each l-row
                .unsqueeze_dim::<6>(4)
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank, chunk_len])
                .reshape([batch, nheads, nchunks, fused_len, chunk_len])
                // col interleaving: insert R copies of each l-col
                .unsqueeze_dim::<6>(5)
                .expand([batch, nheads, nchunks, fused_len, chunk_len, mimo_rank])
                .reshape([batch, nheads, nchunks, fused_len, fused_len]);

            // Apply mask: (CB ∘ L_mimo) · V
            let cb_bnLhL = cb_bnhLL.permute([0, 1, 3, 2, 4]); // [b, n, L, H, L]
            let l_bnLhL = l_mimo_bhnLL.permute([0, 2, 3, 1, 4]); // [b, n, L, H, L]
            let masked_cb_bnhLL = (cb_bnLhL * l_bnLhL).permute([0, 1, 3, 2, 4]); // [b, n, H, L, L]

            let v_bnhLp = v_bnLhp.clone().permute([0, 1, 3, 2, 4]); // [b, n, H, L, P]
            let y_diag_bnhLp = masked_cb_bnhLL.matmul(v_bnhLp); // [b, n, H, L, P]

            y_diag_bnhLp.permute([0, 1, 3, 2, 4]) // [b, n, L, H, P]
        };

        // =============================================================
        // STEP 2: Chunk state (input → state, zero initial state)
        //
        // s[n] = Σ_{t,r} exp(cumA[n,-1] - cumA[n,t]) · V[n,t*R+r] · B[n,t*R+r]ᵀ
        //      (outer product over P and N)
        // =============================================================
        let state_bnhpr = {
            // Decay from each fused position to end of chunk:
            //   decay_fused[t*R+r] = exp(cumA_last - cumA_base[t])
            let a_cumsum_last_bhn1 = a_cumsum_bhnl.clone().slice(s![.., .., .., -1]); // [b,H,n,1]
            // Expand base cumsum to fused length (each time repeated R times):
            // [b, H, n, l] → [b, H, n, l, R] → [b, H, n, L]
            let a_cumsum_fused_bhnL = a_cumsum_bhnl
                .clone()
                .unsqueeze_dim::<5>(4)
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
                .reshape([batch, nheads, nchunks, fused_len]);
            // (cumA_last - cumA_fused): broadcasts [b,H,n,1] - [b,H,n,L]
            let decay_bhnL = (a_cumsum_last_bhn1 - a_cumsum_fused_bhnL).exp();

            // Multiply decay into V: [b, n, L, H, 1] * [b, n, L, H, P]
            let decay_bnLh1 = decay_bhnL.permute([0, 2, 3, 1]).unsqueeze_dim(4);
            let decayed_v_bnLhp = decay_bnLh1 * v_bnLhp.clone();

            // state = decayed_V^T @ B:  [b, n, H, P, L] × [b, n, H, L, N] → [b, n, H, P, N]
            let decayed_v_bnhpL = decayed_v_bnLhp.permute([0, 1, 3, 4, 2]);
            let b_bnhLN = b_bnLhn.permute([0, 1, 3, 2, 4]);
            decayed_v_bnhpL.matmul(b_bnhLN) // [b, n, H, P, N]
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

            // Prepend initial state: [b, 1+n, H, P, N]
            let state_bNhpr = Tensor::cat(vec![initial_state_b1hpr, state_bnhpr], 1);

            // Per-chunk cumulative decay (last position of each chunk): [b, H, n]
            let a_cumsum_last_bhn: Tensor<B, 3> = a_cumsum_bhnl
                .clone()
                .slice(s![.., .., .., -1])
                .squeeze_dim(3);
            // Prepend zero for the initial state (no decay before chunk 0):
            let a_chunk_pad_bhN = Tensor::cat(
                vec![Tensor::zeros([batch, nheads, 1], device), a_cumsum_last_bhn],
                2,
            ); // [b, H, 1+n]

            // Inter-chunk decay matrix via segsum: [b, H, 1+n, 1+n]
            let decay_chunk_bhNN = segsum::<B, 3, 4>(a_chunk_pad_bhN).exp();

            // Flatten (P, N) for matmul
            let flat = per_head_dim * state_rank;
            let state_bhNf = state_bNhpr.clone().permute([0, 2, 1, 3, 4]).reshape([
                batch,
                nheads,
                1 + nchunks,
                flat,
            ]);

            // [b, H, 1+n, 1+n] × [b, H, 1+n, P·N] → [b, H, 1+n, P·N]
            let new_state_bhNf = decay_chunk_bhNN.matmul(state_bhNf);
            let new_state_bhNpr =
                new_state_bhNf.reshape([batch, nheads, 1 + nchunks, per_head_dim, state_rank]);

            // Split: chunk input states [0..n], final state [n]
            let s_bhnpr = new_state_bhNpr
                .clone()
                .slice(s![.., .., 0..nchunks, .., ..]);
            let f_bhpr: Tensor<B, 4> = new_state_bhNpr
                .slice(s![.., .., nchunks, .., ..])
                .squeeze_dim(2);

            (s_bhnpr.permute([0, 2, 1, 3, 4]), f_bhpr) // [b, n, H, P, N], [b, H, P, N]
        };

        // =============================================================
        // STEP 4: State-to-output (Y_off)
        //
        // Y_off[n, t*R+r] = C[t*R+r]ᵀ · exp(cumA[t]) · h[n-1]
        // =============================================================
        let y_off_bnLhp = {
            // Expand base cumsum to fused, then exp:
            let state_decay_bhnL = a_cumsum_bhnl
                .clone()
                .unsqueeze_dim::<5>(4)
                .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
                .reshape([batch, nheads, nchunks, fused_len])
                .exp(); // [b, H, n, L]

            // C: [b, n, H, L, N], state: [b, n, H, N, P]
            let c_bnhLr = c_bnLhn.permute([0, 1, 3, 2, 4]);
            let state_bnhrp = state_bnhpr.permute([0, 1, 2, 4, 3]);
            let ch_bnhLp = c_bnhLr.matmul(state_bnhrp); // [b, n, H, L, P]

            // Multiply by intra-chunk decay: [b, n, H, L, 1]
            let decay_bnhL1 = state_decay_bhnL.permute([0, 2, 1, 3]).unsqueeze_dim(4);
            let y_off_bnhLp = ch_bnhLp * decay_bnhL1;
            y_off_bnhLp.permute([0, 1, 3, 2, 4]) // [b, n, L, H, P]
        };

        // ── Combine and reshape ───────────────────────────────────────────────
        let y_bnLhp = y_diag_bnLhp + y_off_bnLhp; // [b, n, L, H, P]
        // Reshape: [b, n, L, H, P] = [b, n, l*R, H, P] → [b, n, l, R, H, P]
        let y_bnlrhp =
            y_bnLhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);

        (y_bnlrhp, final_state_bhpr)
    }
}

// ---------------------------------------------------------------------------
// segsum  (stable segment sum for the 1-SS mask)
// ---------------------------------------------------------------------------

/// Compute stable segment sums for constructing the 1-semiseparable mask.
///
/// Given a tensor `x` of shape `[..., T]`, returns a tensor of shape `[..., T, T]` where:
///
/// ```text
///   out[..., i, j] = Σ_{k=j+1}^{i} x[..., k]   for i ≥ j  (lower triangle)
///   out[..., i, j] = -∞                           for i < j  (upper triangle)
/// ```
pub(super) fn segsum<B: Backend, const D: usize, const D2: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D2> {
    assert_eq!(D + 1, D2);

    let x_cumsum = x.cumsum(D - 1);
    let x_cumsum_row = x_cumsum.clone().unsqueeze_dim(D); // [..., T, 1]
    let x_cumsum_col = x_cumsum.unsqueeze_dim(D - 1); // [..., 1, T]

    let diff = x_cumsum_row - x_cumsum_col; // [..., T, T]
    let neg_inf_mask = Tensor::full_like(&diff, f32::NEG_INFINITY).triu(1);
    diff + neg_inf_mask
}
