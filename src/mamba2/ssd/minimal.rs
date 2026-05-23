//! ## The Chunkwise SSD Algorithm
//!
//! During training (and prefill), a naive sequential recurrence cannot
//! exploit GPU tensor cores.  The **chunkwise SSD algorithm** (§4 of the
//! paper) achieves this by splitting the sequence into chunks of length chunk_len
//! and decomposing the computation into four steps:
//!
//! ```text
//!   Step 1  (intra-chunk, quadratic form)   →  Y_diag
//!   Step 2  (input → chunk state)           →  state_bnhpr
//!   Step 3  (inter-chunk state scan)        →  state_bnhpr, final_state
//!   Step 4  (chunk state → output)          →  Y_off
//!
//!   Y = Y_diag + Y_off
//! ```
//!
//! Steps 1, 2, 4 are fully parallel across chunks and use batched matrix
//! multiplications (exploiting tensor cores).  Step 3 is a short sequential
//! scan over `sequence/chunk_len` elements rather than `sequence`.

use crate::mamba2::prelude::*;
use crate::utils::{sanity::sanity as san, segsum::segsum};
use burn::prelude::*;

impl<B: Backend> Mamba2<B> {
    // -----------------------------------------------------------------------
    // chunked_selective_scan
    // -----------------------------------------------------------------------

    /// Minimal chunkwise SSD algorithm.
    ///
    /// Implements the four-step decomposition of the semiseparable matrix
    /// multiplication described in §4 of the paper.  The sequence of length
    /// is split into `nchunks = ⌈sequence/chunk_len⌉` chunks of length chunk_len.
    ///
    /// ## The four steps
    ///
    /// ### Step 1 — Intra-chunk outputs (Y_diag)
    ///
    /// Within each chunk, compute the output assuming the initial hidden state
    /// is zero.  This is the *quadratic attention form* of the SSD layer
    /// restricted to a window of chunk_len tokens (§4.1):
    ///
    /// ```text
    ///   Y_diag[n] = (L[n] ∘ C[n] B[n]ᵀ) · X[n]
    /// ```
    ///
    /// where `L[n]` is the chunk_len×chunk_len 1-semiseparable mask for chunk n.
    /// This step is a batched GEMM (exploits tensor cores).
    ///
    /// ### Step 2 — Chunk state (state_bnhpr)
    ///
    /// Compute the final SSM state of each chunk assuming zero initial state
    /// (§4.1, Eq. 20):
    ///
    /// ```text
    ///   s[n] = Σ_{t ∈ chunk n}  exp(A_cum[end] - A_cum[t]) · B̄[t] · x[t]ᵀ
    /// ```
    ///
    /// This is also a batched GEMM and is fully parallel across chunks.
    ///
    /// ### Step 3 — Inter-chunk state scan (state passing)
    ///
    /// Propagate the true hidden state across chunk boundaries using the
    /// recurrence (§4.1, Eq. 22):
    ///
    /// ```text
    ///   h[n] = Ā[n]_chunk · h[n-1] + s[n]
    /// ```
    ///
    /// where `Ā[n]_chunk = exp(Σ_{t ∈ chunk n} Δₜ · A)` is the cumulative
    /// decay over the whole chunk.  This step is implemented as a single
    /// batched matrix multiplication using the 1-semiseparable structure of
    /// the inter-chunk decay matrix (same `segsum` trick, now over chunks).
    /// The scan has length `nchunks = sequence/chunk_len` rather than sequence, so its cost is
    /// negligible for typical chunk sizes.
    ///
    /// ### Step 4 — State-to-output (Y_off)
    ///
    /// For each chunk n, compute the contribution of the true initial state
    /// `h[n-1]` to the outputs within that chunk (§4.1, Eq. 23):
    ///
    /// ```text
    ///   Y_off[n, t] = C[n, t]ᵀ · exp(A_cum[t]) · h[n-1]
    /// ```
    ///
    /// This is again a batched GEMM.
    ///
    /// ### Final output (with D skip-connection)
    ///
    /// ```text
    ///   Y = Y_diag + Y_off + D · X
    /// ```
    #[allow(non_snake_case)]
    pub fn ssd_minimal(input: super::Mamba2SsdInput<B>) -> (Tensor<B, 5>, Tensor<B, 4>) {
        let [batch, nchunks, chunk_len, nheads, per_head_dim] = input.x_bnlhp.dims();
        let [.., ngroups, state_rank] = input.b_bnlgr.dims();
        let device = &input.x_bnlhp.device();

        assert_eq!(nheads % ngroups, 0);
        assert!(nchunks >= 1, "sequence must be non-empty");
        assert!(chunk_len > 0, "chunk_len must be positive");

        // ── Compute discretised parameters ────────────────────────────────────
        // Ā = exp(Δ · A)   stored in log-space as  a_bnlh = Δ · A  (negative)
        // B̄ = Δ · B        (Euler/ZOH approximation)

        // Expand B and C from ngroups to nheads by repeating each group's
        // projection across all heads_per_group heads in that group.
        let heads_per_group = nheads / ngroups;

        // b_bnlgr → b_bnlhr
        let b_bnlhr = input
            .b_bnlgr
            .clone()
            .unsqueeze_dim::<6>(4) // b_bnlg1r
            .expand([
                batch,
                nchunks,
                chunk_len,
                ngroups,
                heads_per_group,
                state_rank,
            ]) // b_bnlgHr
            .reshape([batch, nchunks, chunk_len, nheads, state_rank]); // b_bnlhr

        // c_bnlgr → c_bnlhr
        let c_bnlhr = input
            .c_bnlgr
            .clone()
            .unsqueeze_dim::<6>(4) // c_bnlg1r
            .expand([
                batch,
                nchunks,
                chunk_len,
                ngroups,
                heads_per_group,
                state_rank,
            ]) // c_bnlgHr
            .reshape([batch, nchunks, chunk_len, nheads, state_rank]); // c_bnlhr

        // B̄ₜ = Δₜ · Bₜ
        let delta_b_bnlhr = input.dt_bnlh.clone().unsqueeze_dim(4) * b_bnlhr.clone();
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, state_rank],
            delta_b_bnlhr.dims()
        );
        san(&delta_b_bnlhr);

        // Ā in log-space: a_bnlh = Δₜ · A
        let a_bnlh = input.dt_bnlh.clone()
            * input
                .a_decay_h
                .clone()
                .unsqueeze_dims::<4>(&[0, 1, 2]) // a_head_decay_111h
                .expand([batch, nchunks, chunk_len, nheads]) // a_decay_bnlh
            ;
        san(&a_bnlh);

        // ── Reshape ───────────────────────────────────────────────────────────
        // a (log-decay)
        let a_bhnl = a_bnlh.permute([0, 3, 1, 2]);
        assert_eq!([batch, nheads, nchunks, chunk_len], a_bhnl.dims());

        // Cumulative sum of log-decays within each chunk.
        // a_cumsum_bhnl[b, h, n, t] = Σ_{k=0..t} Δ_{n,k} · A
        // This is the log of the cumulative decay factor from the start of the
        // chunk to position t (inclusive).
        let a_cumsum_bhnl = a_bhnl.clone().cumsum(3);
        assert_eq!([batch, nheads, nchunks, chunk_len], a_cumsum_bhnl.dims());
        san(&a_cumsum_bhnl);

        // =============================================================
        // STEP 1: Intra-chunk outputs (diagonal blocks, Y_diag)
        // =============================================================
        //
        // For each chunk n, compute Y_diag[n] = (L[n] ∘ C[n] B[n]ᵀ) · X[n]
        // where L[n] ∈ ℝ^{chunk_len×chunk_len} is the 1-semiseparable mask for the chunk.
        //
        // L[n]_{i,j} = exp(Σ_{k=j+1..i} a_{n,k})  for i ≥ j
        //            = exp(a_cumsum[n,i] - a_cumsum[n,j])   (using segsum trick)
        //
        // Implementation uses three batched matmuls:
        //   (a) C[n] · B[n]ᵀ  (contract over state_rank state_rank)  → temp1
        //   (b) temp1 ∘ L[n]                                 → temp2
        //   (c) temp2 · X[n]  (contract over chunk_len)              → Y_diag
        let y_diag_bnlhp = {
            // Permute for the matmul along chunk_len and state_rank.
            let b_bnhlr = delta_b_bnlhr.clone().permute([0, 1, 3, 2, 4]);
            let c_bnhlr = c_bnlhr.clone().permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                b_bnhlr.dims()
            );
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                c_bnhlr.dims()
            );

            // (a) C[n] · B[n]ᵀ
            //     Contracts over state_rank.
            let b_bnhrl = b_bnhlr.permute([0, 1, 2, 4, 3]);
            let cb_bnhll = c_bnhlr.matmul(b_bnhrl);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, chunk_len],
                cb_bnhll.dims()
            );
            san(&cb_bnhll);

            // (b) Element-wise multiply with the 1-SS mask L.
            //     L = exp(segsum(a_bhnl))
            //     Lᵢⱼ = exp(a_cumsum[n,i] - a_cumsum[n,j])  (Eq. 4–5)
            let l_bhnll = segsum(a_bhnl.clone()).exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len, chunk_len],
                l_bhnll.dims()
            );
            san(&l_bhnll);

            // Permute both for the broadcast multiply.
            let cb_bnlhl = cb_bnhll.permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, chunk_len],
                cb_bnlhl.dims()
            );
            let l_bnlhl = l_bhnll.permute([0, 2, 3, 1, 4]);
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, chunk_len],
                l_bnlhl.dims()
            );
            san(&cb_bnlhl);
            san(&l_bnlhl);
            let masked_cb_bnlhl = cb_bnlhl * l_bnlhl;
            san(&masked_cb_bnlhl);

            // (c) masked_CB · X → Y_diag.
            //     Contract over the last chunk_len dimension.
            let masked_cb_bnhll = masked_cb_bnlhl.permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, chunk_len],
                masked_cb_bnhll.dims()
            );

            let x_bnhlp = input.x_bnlhp.clone().permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                x_bnhlp.dims()
            );

            let y_diag_bnhlp = masked_cb_bnhll.matmul(x_bnhlp);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                y_diag_bnhlp.dims()
            );
            san(&y_diag_bnhlp);

            y_diag_bnhlp.permute([0, 1, 3, 2, 4]) // y_diag_bnlhp
        };
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_diag_bnlhp.dims()
        );

        // =============================================================
        // STEP 2: Compute chunk state (input → state)
        // =============================================================
        //
        // For each chunk n, compute the SSM state at the end of the chunk
        // assuming the initial state is zero (Eq. 20):
        //
        //   s[n] = Σ_{t ∈ [0, chunk_len)} exp(a_cumsum[n,-1] - a_cumsum[n,t]) · B̄[n,t] · x[n,t]ᵀ
        //
        // Equivalently:
        //   decay_state[n, t] = exp(a_cum_last[n] - a_cum[n, t])
        //   s[n] = Σₜ  decay_state[n, t] · x[n, t]ᵀ · B̄[n, t]     (outer product over per_head_dim and state_rank)
        //
        // This is a batched GEMM, fully parallel across n and b.
        let state_bnhpr = {
            // Decay from each position t to the end of the chunk:
            //   decay_state[n, t] = exp(a_cum[n, chunk_len-1] - a_cum[n, t])
            let a_cumsum_last_bhn1 = a_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
            assert_eq!([batch, nheads, nchunks, 1], a_cumsum_last_bhn1.dims());

            let decay_state_bhnl = (a_cumsum_last_bhn1 - a_cumsum_bhnl.clone()).exp();
            assert_eq!([batch, nheads, nchunks, chunk_len], decay_state_bhnl.dims());
            san(&decay_state_bhnl);

            // Multiply decay into x: decay[n, t] · x[n, t]
            let decay_state_bnlh1 = decay_state_bhnl.permute([0, 2, 3, 1]).unsqueeze_dim(4);
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, 1],
                decay_state_bnlh1.dims()
            );
            let decayed_x_bnlhp = decay_state_bnlh1 * input.x_bnlhp.clone();
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, per_head_dim],
                decayed_x_bnlhp.dims()
            );
            san(&decayed_x_bnlhp);

            // Contract over chunk_len: (decayed_x[n, :, h, :])ᵀ · B̄[n, :, h, :]
            let decayed_x_bnhpl = decayed_x_bnlhp.permute([0, 1, 3, 4, 2]);
            assert_eq!(
                [batch, nchunks, nheads, per_head_dim, chunk_len],
                decayed_x_bnhpl.dims()
            );
            let b_bnhlr = delta_b_bnlhr.clone().permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                b_bnhlr.dims()
            );

            decayed_x_bnhpl.matmul(b_bnhlr)
        };
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            state_bnhpr.dims()
        );
        san(&state_bnhpr);

        // =============================================================
        // STEP 3: Inter-chunk state scan (state passing)
        // =============================================================
        //
        // Propagate hidden state across chunk boundaries.  The recurrence is
        //
        //   h[n] = Ā_chunk[n] · h[n-1] + s[n]     (Eq. 22)
        //
        // where Ā_chunk[n] = exp(Σ_{t ∈ chunk n} Δₜ · A) = exp(a_cum[n, chunk_len-1]).
        //
        // Unrolling the recurrence gives a matrix form identical to Step 2 but
        // at the chunk level: each new state is a weighted sum of all previous
        // chunk state.  We implement this with the same 1-SS segsum trick,
        // now applied over the nchunks dimension.
        //
        // The result is `new_state[n]`, the true hidden state entering chunk n,
        // for n ∈ {0, ..., nchunks-1}, plus the final state after all chunks.
        let (state_bnhpr, final_state_bnpr) = {
            // Prepend the initial state h₀ to the array of chunk state.
            let initial_state_b1hpr = input.initial_state_bhpr.unsqueeze_dim(1);
            assert_eq!(
                [batch, 1, nheads, per_head_dim, state_rank],
                initial_state_b1hpr.dims()
            );

            // Optionally add learnable initial state (broadcast over batch).
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
            san(&initial_state_b1hpr);

            let state_bNhpr = Tensor::cat(vec![initial_state_b1hpr, state_bnhpr], 1);
            assert_eq!(
                [batch, 1 + nchunks, nheads, per_head_dim, state_rank],
                state_bNhpr.dims()
            );

            // Build the inter-chunk decay matrix using segsum.
            // a_cum_last[n] = Σ_{t ∈ chunk n} Δₜ · A   (the total log-decay of chunk n)
            let a_cumsum_last_bhn = a_cumsum_bhnl
                .clone()
                .slice(s![.., .., .., -1]) // a_cumsum_bhn1
                .squeeze_dim(3); // a_cumsum_bhn
            assert_eq!([batch, nheads, nchunks], a_cumsum_last_bhn.dims());

            // Prepend a zero for the initial state (no decay before chunk 0).
            let a_chunk_pad_bhN = Tensor::cat(
                vec![
                    Tensor::zeros(Shape::new([batch, nheads, 1]), device),
                    a_cumsum_last_bhn,
                ],
                2,
            );
            assert_eq!([batch, nheads, 1 + nchunks], a_chunk_pad_bhN.dims());

            // 1-SS inter-chunk decay matrix.
            //   decay_chunk[i, j] = exp(Σ_{k=j+1..i} a_cum_last[k])  (i ≥ j)
            // Row i of this matrix, when multiplied by the state vector,
            // gives the true hidden state entering chunk i.
            let decay_chunk_bhNN = segsum(a_chunk_pad_bhN).exp();
            assert_eq!(
                [batch, nheads, 1 + nchunks, 1 + nchunks],
                decay_chunk_bhNN.dims()
            );
            san(&decay_chunk_bhNN);

            // Flatten the state's (per_head_dim, state_rank) dimensions for the matmul.
            let flat_state_dim = per_head_dim * state_rank; // f = per_head_dim·state_rank
            let state_bhNf = state_bNhpr
                .clone()
                .permute([0, 2, 1, 3, 4]) // state_bhNpr
                .reshape([batch, nheads, 1 + nchunks, flat_state_dim]); // state_bhNf
            assert_eq!(
                [batch, nheads, 1 + nchunks, flat_state_dim],
                state_bhNf.dims()
            );

            let new_state_bhNf = decay_chunk_bhNN.matmul(state_bhNf);
            assert_eq!(
                [batch, nheads, 1 + nchunks, flat_state_dim],
                new_state_bhNf.dims()
            );
            san(&new_state_bhNf);

            let new_state_bhNpr =
                new_state_bhNf.reshape([batch, nheads, 1 + nchunks, per_head_dim, state_rank]);

            // Slice to get:
            //   state[0..nchunks]  — the initial state entering each chunk
            //   state[nchunks]     — the final state after the last real token
            //
            // For padded sequences the padding steps are identity operations
            // (Δ=0 ⇒ Ā=1, B̄=0), so the state is carried unchanged through the
            // pad region, and `state[nchunks]` is the correct final state.
            let state_bhnpr = new_state_bhNpr
                .clone()
                .slice(s![.., .., 0..nchunks, .., ..]);
            let final_state_bhpr = new_state_bhNpr
                .slice(s![.., .., nchunks, .., ..])
                .squeeze_dim(2);

            (
                state_bhnpr.permute([0, 2, 1, 3, 4]), // state_bnhpr
                final_state_bhpr,
            )
        };
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            state_bnhpr.dims()
        );
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            final_state_bnpr.dims()
        );

        // =============================================================
        // STEP 4: State-to-output contribution (Y_off)
        // =============================================================
        //
        // For each chunk n, compute the contribution of the true initial state
        // h[n-1] to the outputs within that chunk (Eq. 23):
        //
        //   Y_off[n, t] = C[n, t]ᵀ · exp(a_cumsum[n, t]) · h[n-1]
        //               = exp(a_cum[n,t]) · (C[n,t]ᵀ · h[n-1])
        //
        // where the scalar `exp(a_cum[n,t])` is the cumulative decay from the
        // start of the chunk to position t.
        //
        // Implementation:
        //   (a) C[n] · h[n-1]ᵀ  (contract over state_rank)
        //   (b) element-wise multiply with exp(a_cum)
        let y_off_bnlhp = {
            // exp(a_cumsum[n, t]): decay from start of chunk to position t.
            let state_decay_out_bhnl = a_cumsum_bhnl.exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len],
                state_decay_out_bhnl.dims()
            );
            san(&state_decay_out_bhnl);

            // (a) C[n] · h[n-1]ᵀ
            let c_bnhlr = c_bnlhr.permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                c_bnhlr.dims()
            );

            let state_bnhrp = state_bnhpr.permute([0, 1, 2, 4, 3]);
            assert_eq!(
                [batch, nchunks, nheads, state_rank, per_head_dim],
                state_bnhrp.dims()
            );

            let ch_bnhlp = c_bnhlr.matmul(state_bnhrp);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                ch_bnhlp.dims()
            );
            san(&ch_bnhlp);

            // (b) Multiply by the intra-chunk cumulative decay.
            let state_decay_out_bnhl1 = state_decay_out_bhnl.permute([0, 2, 1, 3]).unsqueeze_dim(4);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, 1],
                state_decay_out_bnhl1.dims()
            );

            let y_off_bnhlp = ch_bnhlp * state_decay_out_bnhl1;
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                y_off_bnhlp.dims()
            );
            san(&y_off_bnhlp);

            y_off_bnhlp.permute([0, 1, 3, 2, 4]) // y_off_bnlhp
        };
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_off_bnlhp.dims()
        );

        // ── Combine Y_diag and Y_off, undo padding ────────────────────────────
        let y_bnlhp = y_diag_bnlhp + y_off_bnlhp;
        san(&y_bnlhp);

        // ── D skip connection ─────────────────────────────────────────────────
        // yₜ += D · xₜ
        // D is a per-head scalar; broadcast over batch, sequence, and per_head_dim.
        let d_bnlhp = input
            .d_h
            .unsqueeze_dims::<5>(&[0, 1, 2, 4]) // d_111h1
            .expand([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let y_bnlhp = y_bnlhp + d_bnlhp * input.x_bnlhp;
        san(&y_bnlhp);

        (y_bnlhp, final_state_bnpr)
    }
}