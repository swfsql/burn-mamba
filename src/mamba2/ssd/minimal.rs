//! ## The Chunkwise SSD Algorithm
//!
//! During training (and prefill), a naive sequential recurrence cannot
//! exploit GPU tensor cores.  The **chunkwise SSD algorithm** (§4 of the
//! paper) achieves this by splitting the sequence into chunks of length Q
//! and decomposing the computation into four steps:
//!
//! ```text
//!   Step 1  (intra-chunk, quadratic form)   →  Y_diag
//!   Step 2  (input → chunk state)           →  states_bnhpr
//!   Step 3  (inter-chunk state scan)        →  states_bnhpr, final_state
//!   Step 4  (chunk state → output)          →  Y_off
//!
//!   Y = Y_diag + Y_off
//! ```
//!
//! Steps 1, 2, 4 are fully parallel across chunks and use batched matrix
//! multiplications (exploiting tensor cores).  Step 3 is a short sequential
//! scan over `T/Q` elements rather than `T`.

use crate::mamba2::prelude::*;
use burn::prelude::*;

impl<B: Backend> Mamba2<B> {
    // -----------------------------------------------------------------------
    // chunked_selective_scan
    // -----------------------------------------------------------------------

    /// Minimal chunkwise SSD algorithm.
    ///
    /// Implements the four-step decomposition of the semiseparable matrix
    /// multiplication described in §4 of the paper.  The sequence of length T
    /// is split into `nchunks = ⌈T/Q⌉` chunks of length Q.
    ///
    /// ## The four steps
    ///
    /// ### Step 1 — Intra-chunk outputs (Y_diag)
    ///
    /// Within each chunk, compute the output assuming the initial hidden state
    /// is zero.  This is the *quadratic attention form* of the SSD layer
    /// restricted to a window of Q tokens (§4.1):
    ///
    /// ```text
    ///   Y_diag[n] = (L[n] ∘ C[n] B[n]ᵀ) · X[n]
    /// ```
    ///
    /// where `L[n]` is the Q×Q 1-semiseparable mask for chunk n.
    /// This step is a batched GEMM (exploits tensor cores).
    ///
    /// ### Step 2 — Chunk states (states_bnhpr)
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
    /// The scan has length `nchunks = T/Q` rather than T, so its cost is
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
    ///
    /// # Shapes
    /// - `x_bshp`                  : `[batch, sequence, nheads, per_head_dim]`
    /// - `dt_bsh`     (Δ)          : `[batch, sequence, nheads]`
    /// - `a_head_decay_h` (A < 0)  : `[nheads]`
    /// - `b_bsgr`     (B)          : `[batch, sequence, ngroups, state_rank]`
    /// - `c_bsgr`     (C)          : `[batch, sequence, ngroups, state_rank]`
    /// - `d_h`        (D)          : `[nheads]`
    /// - `ssm_initial_state_bhpr`  : `[batch, nheads, per_head_dim, state_rank]`
    /// - `init_states_hpr`         : `[nheads, per_head_dim, state_rank]`
    /// - output.0 `y_bshp`         : `[batch, sequence, nheads, per_head_dim]`
    /// - output.1 `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    #[allow(non_snake_case)]
    pub fn ssd_minimal(
        x_bshp: Tensor<B, 4>,
        dt_bsh: Tensor<B, 3>,
        a_head_decay_h: Tensor<B, 1>,
        b_bsgr: Tensor<B, 4>,
        c_bsgr: Tensor<B, 4>,
        d_h: Tensor<B, 1>,
        ssm_initial_state_bhpr: Tensor<B, 4>,
        init_states_hpr: Option<Tensor<B, 3>>,
        ngroups: usize,
        state_rank: usize,
        chunk_len: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, sequence, nheads, per_head_dim] = x_bshp.dims();
        let device = &x_bshp.device();

        assert_eq!(nheads % ngroups, 0);
        assert!(sequence >= 1, "sequence must be non-empty");
        assert!(chunk_len > 0, "chunk_len must be positive");

        assert_eq!(sequence % chunk_len, 0);
        let nchunks = sequence / chunk_len;

        // ── Compute discretised parameters ────────────────────────────────────
        // Ā = exp(Δ · A)   stored in log-space as  a_bsh = Δ · A  (negative)
        // B̄ = Δ · B        (Euler/ZOH approximation)

        // Expand B and C from ngroups to nheads by repeating each group's
        // projection across all heads_per_group heads in that group.
        let heads_per_group = nheads / ngroups;

        // b_bsgr → b_bshr  [batch, sequence, nheads, state_rank]
        let b_bshr = b_bsgr
            .clone()
            .unsqueeze_dim::<5>(3) // [B, T, G, 1, N]
            .expand([batch, sequence, ngroups, heads_per_group, state_rank])
            .reshape([batch, sequence, nheads, state_rank]);

        // c_bsgr → c_bshr  [batch, sequence, nheads, state_rank]
        let c_bshr = c_bsgr
            .clone()
            .unsqueeze_dim::<5>(3) // [B, T, G, 1, N]
            .expand([batch, sequence, ngroups, heads_per_group, state_rank])
            .reshape([batch, sequence, nheads, state_rank]);

        // B̄ₜ = Δₜ · Bₜ   [batch, sequence, nheads, state_rank]
        let delta_b_bshr = dt_bsh.clone().unsqueeze_dim(3) * b_bshr;
        assert_eq!([batch, sequence, nheads, state_rank], delta_b_bshr.dims());

        // Ā in log-space: a_bsh = Δₜ · A  (negative scalar per [B, T, H])
        let a_bsh = dt_bsh.clone()
            * a_head_decay_h
                .clone()
                .unsqueeze_dims::<3>(&[0, 1]) // [H] → [1, 1, H]
                .expand([batch, sequence, nheads]);

        // ── Reshape into chunks ───────────────────────────────────────────────
        // All tensors are reshaped from [B, T, ...] to [B, nchunks, chunk_len, ...]
        // (denoted `n` and `l` in the suffix convention).

        // x: [B, T, H, P] → [B, nchunks, chunk_len, H, P]
        let x_bnlhp = x_bshp
            .clone()
            .reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        // a (log-decay): [B, T, H] → [B, nchunks, chunk_len, H] → [B, H, nchunks, chunk_len]
        let a_bnlh = a_bsh.reshape([batch, nchunks, chunk_len, nheads]);
        let a_bhnl = a_bnlh.permute([0, 3, 1, 2]);
        assert_eq!([batch, nheads, nchunks, chunk_len], a_bhnl.dims());

        // B̄: [B, T, H, N] → [B, nchunks, chunk_len, H, N]
        let b_bnlhr = delta_b_bshr.reshape([batch, nchunks, chunk_len, nheads, state_rank]);
        // C: [B, T, H, N] → [B, nchunks, chunk_len, H, N]
        let c_bnlhr = c_bshr
            .clone()
            .reshape([batch, nchunks, chunk_len, nheads, state_rank]);

        // Cumulative sum of log-decays within each chunk.
        // a_cumsum_bhnl[b, h, n, t] = Σ_{k=0..t} Δ_{n,k} · A
        // This is the log of the cumulative decay factor from the start of the
        // chunk to position t (inclusive).
        let a_cumsum_bhnl = a_bhnl.clone().cumsum(3);
        assert_eq!([batch, nheads, nchunks, chunk_len], a_cumsum_bhnl.dims());

        // =============================================================
        // STEP 1: Intra-chunk outputs (diagonal blocks, Y_diag)
        // =============================================================
        //
        // For each chunk n, compute Y_diag[n] = (L[n] ∘ C[n] B[n]ᵀ) · X[n]
        // where L[n] ∈ ℝ^{Q×Q} is the 1-semiseparable mask for the chunk.
        //
        // L[n]_{i,j} = exp(Σ_{k=j+1..i} a_{n,k})  for i ≥ j
        //            = exp(a_cumsum[n,i] - a_cumsum[n,j])   (using segsum trick)
        //
        // Implementation uses three batched matmuls:
        //   (a) C[n] · B[n]ᵀ  (contract over state_rank N)  → temp1 [B, nchunks, H, Q, Q]
        //   (b) temp1 ∘ L[n]                                 → temp2 [B, nchunks, H, Q, Q]
        //   (c) temp2 · X[n]  (contract over Q)              → Y_diag [B, nchunks, Q, H, P]
        let y_diag_bnlhp = {
            // Permute to [B, nchunks, H, Q, N] for the matmul along Q and N.
            let b_bnhlr = b_bnlhr.clone().permute([0, 1, 3, 2, 4]);
            let c_bnhlr = c_bnlhr.clone().permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                b_bnhlr.dims()
            );
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                c_bnhlr.dims()
            );

            // (a) C[n] · B[n]ᵀ → [B, nchunks, H, Q, Q]
            //     Contracts over state_rank N.
            let b_bnhrl = b_bnhlr.permute([0, 1, 2, 4, 3]); // [B, n, H, N, Q]
            let cb_bnhll = c_bnhlr.matmul(b_bnhrl); // [B, n, H, Q, Q]
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, chunk_len],
                cb_bnhll.dims()
            );

            // (b) Element-wise multiply with the 1-SS mask L.
            //     L = exp(segsum(a_bhnl))  [B, H, nchunks, Q, Q]
            //     Lᵢⱼ = exp(a_cumsum[n,i] - a_cumsum[n,j])  (Eq. 4–5)
            let l_bhnll = segsum(a_bhnl.clone()).exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len, chunk_len],
                l_bhnll.dims()
            );

            // Permute both to [B, n, Q, H, Q] for the broadcast multiply.
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
            let masked_cb_bnlhl = cb_bnlhl * l_bnlhl;

            // (c) masked_CB · X → Y_diag.
            //     Contract over the last Q dimension.
            let masked_cb_bnhll = masked_cb_bnlhl.permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, chunk_len],
                masked_cb_bnhll.dims()
            );

            let x_bnhlp = x_bnlhp.clone().permute([0, 1, 3, 2, 4]); // [B, n, H, Q, P]
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                x_bnhlp.dims()
            );

            let y_diag_bnhlp = masked_cb_bnhll.matmul(x_bnhlp);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                y_diag_bnhlp.dims()
            );

            y_diag_bnhlp.permute([0, 1, 3, 2, 4]) // → [B, n, Q, H, P]
        };
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_diag_bnlhp.dims()
        );

        // =============================================================
        // STEP 2: Compute chunk states (input → state)
        // =============================================================
        //
        // For each chunk n, compute the SSM state at the end of the chunk
        // assuming the initial state is zero (Eq. 20):
        //
        //   s[n] = Σ_{t ∈ [0, Q)} exp(a_cumsum[n,-1] - a_cumsum[n,t]) · B̄[n,t] · x[n,t]ᵀ
        //
        // Equivalently:
        //   decay_states[n, t] = exp(a_cum_last[n] - a_cum[n, t])
        //   s[n] = Σ_t  decay_states[n, t] · x[n, t]ᵀ · B̄[n, t]     (outer product over P and N)
        //
        // This is a batched GEMM, fully parallel across n and b.
        let states_bnhpr = {
            // Decay from each position t to the end of the chunk:
            //   decay_states[n, t] = exp(a_cum[n, Q-1] - a_cum[n, t])
            let a_cumsum_last_bhn1 = a_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
            assert_eq!([batch, nheads, nchunks, 1], a_cumsum_last_bhn1.dims());

            let decay_states_bhnl = (a_cumsum_last_bhn1 - a_cumsum_bhnl.clone()).exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len],
                decay_states_bhnl.dims()
            );

            // Multiply decay into x: decay[n, t] · x[n, t]  → [B, n, Q, H, P]
            let decay_states_bnlh1 = decay_states_bhnl.permute([0, 2, 3, 1]).unsqueeze_dim(4);
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, 1],
                decay_states_bnlh1.dims()
            );
            let decayed_x_bnlhp = decay_states_bnlh1 * x_bnlhp.clone();
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, per_head_dim],
                decayed_x_bnlhp.dims()
            );

            // Contract over Q: (decayed_x[n, :, h, :])ᵀ · B̄[n, :, h, :]
            //   [B, n, H, P, Q] × [B, n, H, Q, N] → [B, n, H, P, N]
            let decayed_x_bnhpl = decayed_x_bnlhp.permute([0, 1, 3, 4, 2]);
            assert_eq!(
                [batch, nchunks, nheads, per_head_dim, chunk_len],
                decayed_x_bnhpl.dims()
            );
            let b_bnhlr = b_bnlhr.clone().permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                b_bnhlr.dims()
            );

            // TODO: issue for needing to ensure decayed_x_bnhpl is contiguous
            let decayed_x_bnhpl_data = decayed_x_bnhpl.into_data();
            let decayed_x_bnhpl: Tensor<B, 5> = Tensor::from_data(decayed_x_bnhpl_data, device);

            decayed_x_bnhpl.matmul(b_bnhlr)
        };
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            states_bnhpr.dims()
        );

        // =============================================================
        // STEP 3: Inter-chunk state scan (state passing)
        // =============================================================
        //
        // Propagate hidden states across chunk boundaries.  The recurrence is
        //
        //   h[n] = Ā_chunk[n] · h[n-1] + s[n]     (Eq. 22)
        //
        // where Ā_chunk[n] = exp(Σ_{t ∈ chunk n} Δₜ · A) = exp(a_cum[n, Q-1]).
        //
        // Unrolling the recurrence gives a matrix form identical to Step 2 but
        // at the chunk level: each new state is a weighted sum of all previous
        // chunk states.  We implement this with the same 1-SS segsum trick,
        // now applied over the nchunks dimension.
        //
        // The result is `new_states[n]`, the true hidden state entering chunk n,
        // for n ∈ {0, ..., nchunks-1}, plus the final state after all chunks.
        let (states_bnhpr, final_state_bnpr) = {
            // Prepend the initial state h₀ to the array of chunk states.
            // Shape: [B, 1+nchunks, H, P, N]
            let initial_states_b1hpr = ssm_initial_state_bhpr.unsqueeze_dim(1);
            assert_eq!(
                [batch, 1, nheads, per_head_dim, state_rank],
                initial_states_b1hpr.dims()
            );

            // Optionally add learnable initial states (broadcast over batch).
            let initial_states_b1hpr = if let Some(init_hpr) = init_states_hpr {
                let init_b1hpr = init_hpr.unsqueeze_dim::<4>(0).expand([
                    batch,
                    1,
                    nheads,
                    per_head_dim,
                    state_rank,
                ]);
                initial_states_b1hpr + init_b1hpr
            } else {
                initial_states_b1hpr
            };

            let states_bNhpr = Tensor::cat(vec![initial_states_b1hpr, states_bnhpr], 1);
            assert_eq!(
                [batch, 1 + nchunks, nheads, per_head_dim, state_rank],
                states_bNhpr.dims()
            );

            // Build the inter-chunk decay matrix using segsum.
            // a_cum_last[n] = Σ_{t ∈ chunk n} Δₜ · A   (the total log-decay of chunk n)
            let a_cumsum_last_bhn = a_cumsum_bhnl
                .clone()
                .slice(s![.., .., .., -1])
                .squeeze_dim(3); // [B, H, nchunks]
            assert_eq!([batch, nheads, nchunks], a_cumsum_last_bhn.dims());

            // Prepend a zero for the initial state (no decay before chunk 0).
            let a_chunk_pad_bhN = Tensor::cat(
                vec![
                    Tensor::zeros(Shape::new([batch, nheads, 1]), device),
                    a_cumsum_last_bhn,
                ],
                2,
            ); // [B, H, 1+nchunks]
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

            // Flatten the state's (P, N) dimensions for the matmul.
            let flat_state_dim = per_head_dim * state_rank; // f = P·N
            let states_bhNf = states_bNhpr
                .clone()
                .permute([0, 2, 1, 3, 4]) // [B, H, 1+n, P, N]
                .reshape([batch, nheads, 1 + nchunks, flat_state_dim]);
            assert_eq!(
                [batch, nheads, 1 + nchunks, flat_state_dim],
                states_bhNf.dims()
            );

            // Matmul: [B, H, 1+n, 1+n] × [B, H, 1+n, f] → [B, H, 1+n, f]
            let new_states_bhNf = decay_chunk_bhNN.matmul(states_bhNf);
            assert_eq!(
                [batch, nheads, 1 + nchunks, flat_state_dim],
                new_states_bhNf.dims()
            );

            let new_states_bhNpr =
                new_states_bhNf.reshape([batch, nheads, 1 + nchunks, per_head_dim, state_rank]);

            // Slice to get:
            //   states[0..nchunks]  — the initial states entering each chunk
            //   states[nchunks]     — the final state after the last real token
            //
            // For padded sequences the padding steps are identity operations
            // (Δ=0 ⇒ Ā=1, B̄=0), so the state is carried unchanged through the
            // pad region, and `states[nchunks]` is the correct final state.
            let states_bhnpr = new_states_bhNpr
                .clone()
                .slice(s![.., .., 0..nchunks, .., ..]);
            let final_state_bhpr = new_states_bhNpr
                .slice(s![.., .., nchunks, .., ..])
                .squeeze_dim(2);

            (
                states_bhnpr.permute([0, 2, 1, 3, 4]), // → [B, n, H, P, N]
                final_state_bhpr,
            )
        };
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            states_bnhpr.dims()
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
        //   (a) C[n] · h[n-1]ᵀ  (contract over N)  → [B, n, H, Q, P]
        //   (b) element-wise multiply with exp(a_cum)
        let y_off_bnlhp = {
            // exp(a_cumsum[n, t]): decay from start of chunk to position t.
            let state_decay_out_bhnl = a_cumsum_bhnl.exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len],
                state_decay_out_bhnl.dims()
            );

            // (a) C[n] · h[n-1]ᵀ  → [B, n, H, Q, P]
            //   C: [B, n, H, Q, N],  h: [B, n, H, N, P]  (transposed from [B,n,H,P,N])
            let c_bnhlr = c_bnlhr.permute([0, 1, 3, 2, 4]); // [B, n, H, Q, N]
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                c_bnhlr.dims()
            );

            let states_bnhrp = states_bnhpr.permute([0, 1, 2, 4, 3]); // [B, n, H, N, P]
            assert_eq!(
                [batch, nchunks, nheads, state_rank, per_head_dim],
                states_bnhrp.dims()
            );

            let ch_bnhlp = c_bnhlr.matmul(states_bnhrp); // [B, n, H, Q, P]
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                ch_bnhlp.dims()
            );

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

            y_off_bnhlp.permute([0, 1, 3, 2, 4]) // → [B, n, Q, H, P]
        };
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_off_bnlhp.dims()
        );

        // ── Combine Y_diag and Y_off, undo padding ────────────────────────────
        let y_bnlhp = y_diag_bnlhp + y_off_bnlhp;

        // Flatten chunks back into the sequence dimension.
        let y_bshp = y_bnlhp.reshape([batch, sequence, nheads, per_head_dim]);

        // ── D skip connection ─────────────────────────────────────────────────
        // yₜ += D · xₜ
        // D is a per-head scalar; broadcast over batch, sequence, and per_head_dim.
        let d_bsh1 = d_h
            .unsqueeze_dims::<4>(&[0, 1, 3]) // [H] → [1, 1, H, 1]
            .expand([batch, sequence, nheads, 1]);
        let y_bshp = y_bshp + d_bsh1 * x_bshp;

        (y_bshp, final_state_bnpr)
    }
}

// ---------------------------------------------------------------------------
// segsum  (stable segment sum for the 1-SS mask)
// ---------------------------------------------------------------------------

/// Compute stable segment sums for constructing the 1-semiseparable mask.
///
/// Given a tensor `x` of shape `[..., T]`, returns a tensor of shape
/// `[..., T, T]` where:
///
/// ```text
///   out[..., i, j] = Σ_{k=j+1}^{i} x[..., k]     for i ≥ j   (lower triangle)
///   out[..., i, j] = -∞                             for i < j   (upper triangle)
/// ```
///
/// The 1-semiseparable mask is then obtained by exponentiating:
///
/// ```text
///   L = exp(segsum(log_A))
///   L[i, j] = exp(log_A[j+1] + ... + log_A[i])
///            = A[j+1] · A[j+2] · ... · A[i]       (Eq. 4–5 in the paper)
/// ```
///
/// ## Implementation
///
/// A naive computation of all pairwise products `A[j+1]·...·A[i]` would
/// suffer from underflow for long sequences (e.g. `0.9^1000 ≈ 2.6×10⁻⁴⁶`).
/// Working in log-space and computing differences of prefix sums avoids this:
///
/// ```text
///   segsum(x)[i, j] = cumsum(x)[i] - cumsum(x)[j]
/// ```
///
/// The upper triangle is masked to -∞ so that `exp(segsum(...))` gives 0
/// for non-causal positions (the strict upper triangle of L must be zero).
///
/// ## Const-generic dimension handling
///
/// This function is generic over the input rank `D` and returns a tensor of
/// rank `D + 1`.  Burn requires the output rank to be known at compile time,
/// which is achieved through the const generic expression `{ D + 1 }`.
fn segsum<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, { D + 1 }> {
    assert!(D > 0);

    // cumsum[..., t] = x[..., 0] + x[..., 1] + ... + x[..., t]
    let x_cumsum = x.cumsum(D - 1);

    // Broadcast along two different axes to compute all pairwise differences:
    //   x_cumsum_row[..., i, j] = cumsum[..., i]   (i varies along axis D)
    //   x_cumsum_col[..., i, j] = cumsum[..., j]   (j varies along axis D-1)
    let x_cumsum_row = x_cumsum.clone().unsqueeze_dim(D); // [..., T, 1]
    let x_cumsum_col = x_cumsum.unsqueeze_dim(D - 1); // [..., 1, T]

    // diff[..., i, j] = cumsum[i] - cumsum[j]
    //                 = x[j+1] + ... + x[i]    for i ≥ j
    let diff = x_cumsum_row - x_cumsum_col; // [..., T, T]

    // Mask the strict upper triangle (i < j) with -∞.
    // triu(1) returns a tensor that is -∞ above the main diagonal and 0
    // elsewhere; adding it to `diff` zeroes out the upper triangle of exp(diff).
    let neg_inf_mask = Tensor::full_like(&diff, f32::NEG_INFINITY).triu(1);
    diff + neg_inf_mask
}
