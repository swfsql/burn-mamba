// mamba2-pre-gpu.rs
//
// GPU-naive but Triton-faithful forward implementation of the Mamba-2 SSD module.
//
// This file mirrors `_mamba_chunk_scan_combined_fwd` from `ssd_combined.py`,
// which orchestrates the following five GPU kernels, each described in detail below:
//
//   Kernel 1: _chunk_cumsum_fwd_kernel  (ssd_chunk_state.py)
//   Kernel 2: _bmm_chunk_fwd_kernel     (ssd_bmm.py)
//   Kernel 3: _chunk_state_fwd_kernel   (ssd_chunk_state.py)
//   Kernel 4: _state_passing_fwd_kernel (ssd_state_passing.py)
//   Kernel 5: _chunk_scan_fwd_kernel    (ssd_chunk_scan.py)
//
// Parallelism conventions used in comments:
//   blockIdx.{x,y,z} — the outer for-loops in this file; in a real GPU each iteration
//                       would be a separate thread block running concurrently.
//   threadIdx / SRAM  — the inner slice/matmul operations inside a loop body; in a real GPU
//                       these are the per-warp register operations within a thread block.
//   autotune(N)       — a Triton `@triton.autotune` BLOCK_SIZE_* constant; N is the typical
//                       best value so a GPU translation can use it as a starting point.
//
// Each for-loop carries a `// blockIdx.*` or `// threadIdx` comment to make the
// mapping explicit. Tile-size variables are given full descriptive names instead of
// single letters so a reader can immediately understand what dimension they tile.

use crate::mamba2::prelude::*;
use burn::prelude::*;

/// Returns the Triton autotune fallback value for a BLOCK_SIZE_* constant.
///
/// In a real GPU kernel, Triton picks the best block size at JIT-compile time
/// by benchmarking configurations listed in `@triton.autotune(configs=[...])`.
/// Here we record the typical winning value so that CUDA/WGSL/Metal translations
/// can use it as a sensible starting tiling without re-running the search.
fn autotune(fallback: usize) -> usize {
    fallback
}

impl<B: Backend> Mamba2<B> {
    /// GPU-naive, Triton-faithful forward pass for the Mamba-2 SSD module.
    ///
    /// Mirrors `_mamba_chunk_scan_combined_fwd` from `ssd_combined.py`,
    /// which is called from `Mamba2.forward()` via `mamba_chunk_scan_combined()`.
    ///
    /// # Tensor shapes flowing through the function
    ///
    /// ```text
    /// x_bshp:                  [batch, sequence, nheads, per_head_dim] OK
    /// dt_bsh:                  [batch, sequence, nheads] OK
    /// a_decay_h:               [nheads] OK
    /// b_bsgr:                  [batch, sequence, ngroups, state_rank]
    /// c_bsgr:                  [batch, sequence, ngroups, state_rank]
    /// d_h:                     [nheads]
    /// ssm_initial_state_bhpr   [batch, nheads, per_head_dim, state_rank]
    /// _init_states_hpr         [nheads, per_head_dim, state_rank]
    ///
    /// K1 outputs:
    ///   dA_cumsum          [batch, nheads, nchunks, chunk_len]  (intra-chunk cumsum)
    ///   dA_chunk_end       [batch, nheads, nchunks]              (last dA_cumsum per chunk)
    ///
    /// K2 output:
    ///   cb_mat             [batch, nchunks, ngroups, chunk_len, chunk_len]
    ///                      cb_mat[b,c,g,l,s] = sum_k C[b,c*cs+l,g,k] * B[b,c*cs+s,g,k]
    ///
    /// K3 output:
    ///   intra_chunk_states [batch, nchunks, nheads, per_head_dim, d_state]
    ///                      state assuming zero initial state at each chunk boundary
    ///
    /// K4 outputs:
    ///   chunk_input_states [batch, nchunks, nheads, per_head_dim, d_state]
    ///                      chunk_input_states[c] = true initial state entering chunk c
    ///   final_ssm_state    [batch, nheads, per_head_dim, d_state]
    ///
    /// K5 output:
    ///   y_chunked          [batch, nchunks, chunk_len, nheads, per_head_dim]
    ///   → reshaped →      [batch, sequence, nheads, per_head_dim]
    /// ```
    #[allow(non_snake_case)]
    pub fn chunked_selective_scan_hybrid_naive(
        x_bshp: Tensor<B, 4>,
        dt_bsh: Tensor<B, 3>,
        a_decay_h: Tensor<B, 1>,
        b_bsgr: Tensor<B, 4>,
        c_bsgr: Tensor<B, 4>,
        d_h: Tensor<B, 1>,
        ssm_initial_state_bhpr: Tensor<B, 4>,
        _init_states_hpr: Option<Tensor<B, 3>>,
        ngroups: usize,
        state_rank: usize,
        // currently must be set by caller
        chunk_len: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, sequence, nheads, per_head_dim] = x_bshp.dims();
        let device = x_bshp.device();
        assert_ne!(ngroups, 0);
        assert_eq!(nheads % ngroups, 0);
        assert!(sequence > 0, "sequence length must be at least 1");
        // `heads_per_group` is called `nheads_ngroups_ratio` in every Triton kernel.
        // It is the compile-time constant used by GQA (Grouped Query Attention) to map
        // a head index to its B/C group: `group_idx = head_idx / heads_per_group`.
        let heads_per_group = nheads / ngroups;

        // We require seqlen % chunk_len == 0 for simplicity (the Triton kernel handles a
        // partial last chunk via `chunk_len_limit = min(chunk_len, seqlen - pid_c * chunk_len)`).
        debug_assert_eq!(sequence % chunk_len, 0);
        let nchunks = sequence / chunk_len;

        assert!(
            _init_states_hpr.is_none(),
            "init_states_hpr not yet implemented"
        );

        // ── Reshape into chunks ───────────────────────────────────────────────────────
        let x_bnlhp = x_bshp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let dt_bnlh = dt_bsh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlgr = b_bsgr.reshape([batch, nchunks, chunk_len, ngroups, state_rank]);
        let c_bnlgr = c_bsgr.reshape([batch, nchunks, chunk_len, ngroups, state_rank]);

        // ================================================================
        // KERNEL 1: _chunk_cumsum_fwd_kernel  (ssd_chunk_state.py)
        //
        // Triton compile-time constants for this call (from ssd_combined.py):
        //   DT_SOFTPLUS = True
        //   HAS_DT_BIAS = True  (dt_bias is provided)
        //
        // Grid: (batch, nchunks, ceil(nheads / BLOCK_SIZE_H))
        //   blockIdx.x = pid_b         — batch index
        //   blockIdx.y = pid_c         — chunk index
        //   blockIdx.z = pid_h_block   — block of BLOCK_SIZE_H heads
        //
        // threadIdx work per block:
        //   - Loads BLOCK_SIZE_H heads × BLOCK_SIZE_CHUNK positions from dt_raw.
        //   - Adds dt_bias, applies softplus, clamps to dt_limit.
        //   - Multiplies by A (scalar per head) to get dA per position.
        //   - Computes cumulative sum of dA along the chunk_len axis (axis=1 in Triton).
        //   - Stores dt_discretized and dA_cumsum back to VRAM.
        //
        // Outputs are stored in Triton's transposed layout [batch, nheads, nchunks, chunk_len],
        // which puts the heads dimension before chunks to allow coalesced per-head access
        // in the downstream kernels.
        // ================================================================

        // BLOCK_SIZE_H: number of heads processed by one thread block in K1.
        // The autotune key is ['chunk_len', 'nheads'].
        let _heads_per_k1_block = autotune(8);
        // BLOCK_SIZE_CHUNK: positions within a chunk processed by one thread block.
        // Must be a power-of-two ≥ chunk_len (next_power_of_2 in Triton).
        let _chunk_pos_k1_block = autotune(128);
        // These tile sizes are used only for GPU layout annotation; the Rust loops below
        // iterate per-head per-chunk for clarity.
        let _ = (_heads_per_k1_block, _chunk_pos_k1_block);

        // Outputs in Triton layout [batch, nheads, nchunks, chunk_len].
        // Note: dt_bnlh calculation moved to Step 4 (before padding).
        let dt_discretized_bhnl = dt_bnlh.permute([0, 3, 1, 2]);
        assert_eq!(
            [batch, nheads, nchunks, chunk_len],
            dt_discretized_bhnl.dims()
        );
        let mut da_cumsum_bhnl =
            Tensor::<B, 4>::zeros([batch, nheads, nchunks, chunk_len], &device);

        // blockIdx.x = pid_b
        for i_batch in 0..batch {
            // blockIdx.y = pid_c
            for i_chunk in 0..nchunks {
                // blockIdx.z = pid_h_block (one block per BLOCK_SIZE_H heads).
                // We expand one-per-head here; a GPU translation would use:
                //   offs_h = pid_h_block * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
                for i_head in 0..nheads {
                    // threadIdx: load dt_raw tile for this (batch, chunk, head).
                    // Triton ptr: dt_ptr + pid_b*stride_dt_batch + pid_c*chunk_len*stride_dt_seqlen
                    //             + pid_h_block*BLOCK_H*stride_dt_head
                    //   offs_c spans BLOCK_CHUNK positions (threadIdx.x covers seqlen within block)
                    // Note: dt_l calculation moved to Step 4 (before padding).
                    let dt_l = dt_discretized_bhnl
                        .clone()
                        .slice(s![i_batch, i_head, i_chunk, ..,])
                        .squeeze_dims::<1>(&[0, 1, 2]); // tile
                    assert_eq!([chunk_len], dt_l.dims());

                    let da_cumsum_l = {
                        let a_decay_1 = a_decay_h.clone().slice(s![i_head]);
                        assert_eq!([1], a_decay_1.dims());
                        // Per-position decay increment: dA[s] = dt_clamped[s] * A[h].
                        // A[h] < 0 so dA[s] < 0; exp(dA[s]) ∈ (0, 1) is a damping factor.
                        let da_decay_increment_l = dt_l.clone() * a_decay_1.expand([chunk_len]);
                        assert_eq!([chunk_len], da_decay_increment_l.dims());
                        // Cumulative sum along chunk positions (Triton axis=1).
                        // dA_cumsum[s] = Σ_{t=0}^{s} dA[t] = total log-decay from chunk start to s.
                        // Triton: `dA_cs = tl.cumsum(dA, axis=1)`
                        da_decay_increment_l.cumsum(0) // tile
                    };
                    assert_eq!([chunk_len], da_cumsum_l.dims());

                    // threadIdx: write dt_discretized and dA_cumsum to VRAM.
                    // Triton: tl.store(dt_out_ptrs, dt, mask=(offs_h < nheads) & (offs_c < chunk_len))
                    //         tl.store(dA_cs_ptrs,  dA_cs, ...)
                    // Note: dt_discretized already calculated at Step 4.
                    da_cumsum_bhnl = da_cumsum_bhnl.slice_assign(
                        s![i_batch, i_head, i_chunk, ..],
                        da_cumsum_l.unsqueeze_dims::<4>(&[0, 1, 2]),
                    );
                } // n_heads loop
            } // nchunks loop
        } // batch loop

        // dA_chunk_end[b, h, c] = dA_cumsum[b, h, c, chunk_len-1]
        // = total log-decay accumulated over the entire chunk c for head h.
        // Used in K4 as the inter-chunk decay exponent.
        // Triton in _mamba_chunk_scan_combined_fwd: `dA_cumsum[:, :, :, -1]`
        let da_chunk_end_bhn = da_cumsum_bhnl
            .clone()
            .slice(s![.., .., .., -1])
            .squeeze_dim::<3>(3);
        assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());

        // ================================================================
        // KERNEL 2: _bmm_chunk_fwd_kernel  (ssd_bmm.py)
        //
        // Triton compile-time constants:
        //   IS_CAUSAL = False  (full chunk_len × chunk_len matrix, causal mask applied in K5)
        //   HAS_SEQ_IDX = False  (single-sequence batch, no packing)
        //
        // Computes: cb_mat[b, c, g, target_pos, source_pos]
        //         = Σ_k  C[b, c·cs + target_pos, g, k] · B[b, c·cs + source_pos, g, k]
        //         = C[target, :] · B[source, :]   (dot product over d_state)
        //
        // This is the "CB" product whose (l, s) entry weights the intra-chunk contribution
        // of source position s to output position l (the off-diagonal orange blocks and the
        // diagonal blocks of the semiseparable matrix M from Part II of the paper).
        //
        // Grid: (ceil(cs/BM) * ceil(cs/BN), batch * nchunks, ngroups)
        //   blockIdx.x = pid_tile = pid_m * n_tiles_N + pid_n
        //   blockIdx.y = pid_b * nchunks + pid_c
        //   blockIdx.z = pid_g   — group index
        //
        // threadIdx: standard Tensor Core GEMM tile:
        //   accumulates a [BLOCK_M, BLOCK_N] output sub-tile over BLOCK_K steps of d_state.
        //
        // Output dtype: float32  (`output_dtype=torch.float32` in the Python call).
        // ================================================================

        // autotune key: ['chunk_len', 'K', 'IS_CAUSAL']
        let cb_target_pos_tile = autotune(64); // BLOCK_SIZE_M  (target chunk_pos dimension)
        let cb_source_pos_tile = autotune(64); // BLOCK_SIZE_N  (source chunk_pos dimension)
        let cb_state_rank_tile = autotune(32); // BLOCK_SIZE_K  (inner state_rank dimension)

        // outputs
        let mut cb_bngll =
            Tensor::<B, 5>::zeros([batch, nchunks, ngroups, chunk_len, chunk_len], &device);

        // blockIdx.y part-1: pid_b
        for i_batch in 0..batch {
            // blockIdx.y part-2: pid_c
            for i_chunk in 0..nchunks {
                // blockIdx.z: pid_g  — one group per block
                for i_group in 0..ngroups {
                    // Full [chunk_len, chunk_len] accumulator for this (batch, chunk, group).
                    // In the GPU, each (pid_m, pid_n) pair is a separate thread block;
                    // here we accumulate into a single tensor for clarity.
                    let mut cb_tile_ll = Tensor::<B, 2>::zeros([chunk_len, chunk_len], &device);

                    // blockIdx.x: pid_m — output target-position tile
                    for target_start in (0..chunk_len).step_by(cb_target_pos_tile) {
                        let target_end = (target_start + cb_target_pos_tile).min(chunk_len);
                        let target_tile_width = target_end - target_start;
                        let target_range = target_start..target_end;
                        assert_ne!(target_tile_width, 0);

                        // blockIdx.x: pid_n — output source-position tile
                        for source_start in (0..chunk_len).step_by(cb_source_pos_tile) {
                            let source_end = (source_start + cb_source_pos_tile).min(chunk_len);
                            let source_tile_width = source_end - source_start;
                            let source_range = source_start..source_end;
                            assert_ne!(source_tile_width, 0);

                            // threadIdx: accumulator for this [target_tile × source_tile] output block.
                            // Triton: acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
                            let mut cb_subtile_accum_LL = Tensor::<B, 2>::zeros(
                                [target_tile_width, source_tile_width],
                                &device,
                            );

                            // threadIdx K-loop: stream through d_state in BLOCK_K steps.
                            // Triton: `for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):`
                            //           acc += tl.dot(c_tile, b_tile_transposed)
                            for state_rank_start in (0..state_rank).step_by(cb_state_rank_tile) {
                                let state_rank_end =
                                    (state_rank_start + cb_state_rank_tile).min(state_rank);
                                let state_rank_width = state_rank_end - state_rank_start;
                                let state_rank_range = state_rank_start..state_rank_end;
                                assert_ne!(state_rank_width, 0);

                                // SRAM load: C tile [target_tile_width, state_rank_block_width]
                                // Triton ptr: a_ptr + offs_m * stride_a_seqlen + offs_k * stride_ak
                                //   offs_m → target chunk positions, offs_k → d_state indices
                                let c_LR = c_bnlgr
                                    .clone()
                                    .slice(s![
                                        i_batch,
                                        i_chunk,
                                        target_range.clone(),
                                        i_group,
                                        state_rank_range.clone()
                                    ])
                                    .squeeze_dims::<2>(&[0, 1, 3]); // SRAM subtile
                                assert_eq!([target_tile_width, state_rank_width], c_LR.dims());

                                // SRAM load: B tile, loaded transposed as [state_rank_block_width, source_tile_width]
                                // Triton ptr: b_ptr + offs_k * stride_bk + offs_n * stride_b_seqlen
                                //   offs_k → d_state (fast stride), offs_n → source chunk positions
                                // The Triton layout means b[state_rank, source_pos], i.e. B is stored
                                // with d_state as the fast axis, yielding B^T naturally in the matmul.
                                let b_RL = b_bnlgr
                                    .clone()
                                    .slice(s![
                                        i_batch,
                                        i_chunk,
                                        source_range.clone(),
                                        i_group,
                                        state_rank_range.clone()
                                    ])
                                    .squeeze_dims::<2>(&[0, 1, 3])
                                    // transpose
                                    .permute([1, 0]);
                                assert_eq!([state_rank_width, source_tile_width], b_RL.dims());

                                // Tensor Core: C[target, state_rank] @ B^T[state_rank, source] = [target, source]
                                // Triton: `acc += tl.dot(a, b)`
                                cb_subtile_accum_LL = cb_subtile_accum_LL + c_LR.matmul(b_RL);
                            } // state_rank tile loop

                            // threadIdx: store completed subtile back (SRAM → VRAM).
                            // Triton: `tl.store(out_ptrs, out, mask=...)`
                            cb_tile_ll = cb_tile_ll.slice_assign(
                                s![target_range.clone(), source_range.clone()],
                                cb_subtile_accum_LL,
                            );
                        } // source tile loop
                    } // target tile loop

                    // Write the full [chunk_len, chunk_len] group-chunk tile to VRAM.
                    cb_bngll = cb_bngll.slice_assign(
                        s![i_batch, i_chunk, i_group, .., ..],
                        cb_tile_ll.unsqueeze_dims::<5>(&[0, 1, 2]),
                    ); // tile to VRAM
                } // ngroups loop
            } // nchunks loop
        } // batch loop

        // ================================================================
        // KERNEL 3: _chunk_state_fwd_kernel  (ssd_chunk_state.py)
        //
        // Triton compile-time constants:
        //   HAS_SEQ_IDX = False
        //
        // Computes the SSM state at the END of each chunk, assuming the state
        // was ZERO at the START of that chunk (the zero-initial-state assumption).
        // Kernel 4 will later correct for the actual non-zero initial state.
        //
        //   intra_state[b, c, h, per_head_dim_p, state_rank_n]
        //     = Σ_{s=0}^{chunk_len-1}  X[b, c, s, h, p]
        //                                * exp(dA_cumsum[b,h,c,chunk_len-1] - dA_cumsum[b,h,c,s])
        //                                * dt_discretized[b,h,c,s]
        //                                * B[b, c, s, group(h), n]
        //
        // The factor  exp(dA_last - dA[s]) * dt[s]  is the "scale" per position s:
        //   - exp(dA_last - dA[s]): decay from position s forward to end-of-chunk
        //   - dt[s]: discretization factor converting continuous B into discrete B-bar
        //
        // Grid: (ceil(per_head_dim/BM)*ceil(d_state/BN), batch*nchunks, nheads)
        //   blockIdx.x = pid_m * n_tiles_N + pid_n  — tile over per_head_dim × d_state
        //   blockIdx.y = pid_b * nchunks + pid_c    — batch-chunk pair
        //   blockIdx.z = pid_h                      — head index
        //
        // GQA: B is indexed as B[..., group(h), :] where group(h) = pid_h / heads_per_group.
        //
        // states_in_fp32=True in the Python call → output is float32 accumulation.
        // ================================================================

        // autotune key: ['hdim', 'state_rank', 'chunk_len']
        let k3_per_head_dim_tile = autotune(64); // BLOCK_SIZE_M  (per_head_dim output dimension)
        let k3_state_rank_tile = autotune(64); // BLOCK_SIZE_N  (state_rank output dimension)
        let k3_chunk_pos_tile = autotune(64); // BLOCK_SIZE_K  (chunk_len inner dimension)

        // output
        let mut intra_chunk_states_bnhpr =
            Tensor::<B, 5>::zeros([batch, nchunks, nheads, per_head_dim, state_rank], &device);

        // blockIdx.y part-1: pid_b
        for i_batch in 0..batch {
            // blockIdx.y part-2: pid_c
            for i_chunk in 0..nchunks {
                // blockIdx.z: pid_h
                for i_head in 0..nheads {
                    // GQA: map head → group.
                    // Triton: `b_ptr += (pid_h // nheads_ngroups_ratio) * stride_b_head`
                    let i_group = i_head / heads_per_group;

                    // dA_cumsum at the LAST position within this chunk (scalar).
                    // Triton: `dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_len - 1) * stride)`
                    let da_cumsum_last_in_chunk_1 = da_cumsum_bhnl
                        .clone()
                        .slice(s![i_batch, i_head, i_chunk, -1])
                        .squeeze_dims::<1>(&[0, 1, 2]);
                    assert_eq!([1], da_cumsum_last_in_chunk_1.dims());

                    // dA_cumsum at every position within this chunk.
                    // Triton: `dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize`
                    let da_cumsum_within_chunk_l = da_cumsum_bhnl
                        .clone()
                        .slice(s![i_batch, i_head, i_chunk, ..])
                        .squeeze_dims::<1>(&[0, 1, 2]);
                    assert_eq!([chunk_len], da_cumsum_within_chunk_l.dims());

                    // dt_discretized at every position within this chunk.
                    // Triton: `dt_ptrs = dt_ptr + offs_k * stride_dt_csize`
                    let dt_within_chunk_l = dt_discretized_bhnl
                        .clone()
                        .slice(s![i_batch, i_head, i_chunk, ..])
                        .squeeze_dims::<1>(&[0, 1, 2]);
                    assert_eq!([chunk_len], dt_within_chunk_l.dims());

                    // scale[s] = exp(dA_last - dA[s]) * dt[s]
                    // = "forward decay from position s to end-of-chunk, times B-bar factor"
                    // Triton: `scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k`
                    let forward_decay_to_chunk_end_l =
                        (da_cumsum_last_in_chunk_1.expand([chunk_len]) - da_cumsum_within_chunk_l)
                            .exp();
                    assert_eq!([chunk_len], forward_decay_to_chunk_end_l.dims());

                    let b_bar_scale_l = forward_decay_to_chunk_end_l * dt_within_chunk_l;
                    assert_eq!([chunk_len], b_bar_scale_l.dims());

                    // Tiled matmul: X[per_head_dim, chunk_len] @ (B ⊙ scale)[chunk_len, d_state]
                    // where ⊙ denotes element-wise multiplication over the chunk_len axis.
                    //
                    // blockIdx.x: pid_m — per_head_dim output tile
                    for per_head_dim_start in (0..per_head_dim).step_by(k3_per_head_dim_tile) {
                        let per_head_dim_end =
                            (per_head_dim_start + k3_per_head_dim_tile).min(per_head_dim);
                        let per_head_dim_tile_width = per_head_dim_end - per_head_dim_start;
                        let per_head_dim_range = per_head_dim_start..per_head_dim_end;
                        assert_ne!(per_head_dim_tile_width, 0);

                        // blockIdx.x: pid_n — d_state output tile
                        for state_rank_start in (0..state_rank).step_by(k3_state_rank_tile) {
                            let state_rank_end =
                                (state_rank_start + k3_state_rank_tile).min(state_rank);
                            let state_rank_tile_width = state_rank_end - state_rank_start;
                            let state_rank_range = state_rank_start..state_rank_end;
                            assert_ne!(state_rank_tile_width, 0);

                            // threadIdx: accumulator [per_head_dim_tile, state_rank_tile].
                            // Triton: `acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), float32)`
                            let mut state_subtile_accum_PR = Tensor::<B, 2>::zeros(
                                [per_head_dim_tile_width, state_rank_tile_width],
                                &device,
                            );

                            // threadIdx K-loop: tile the chunk_len inner dimension.
                            // Triton: `for k in range(0, chunk_len_limit, BLOCK_SIZE_K):`
                            for chunk_start in (0..chunk_len).step_by(k3_chunk_pos_tile) {
                                let chunk_end = (chunk_start + k3_chunk_pos_tile).min(chunk_len);
                                let chunk_width = chunk_end - chunk_start;
                                let chunk_range = chunk_start..chunk_end;
                                assert_ne!(chunk_width, 0);

                                // SRAM load: X tile [per_head_dim_tile, chunk_block]
                                // Triton ptr: x_ptrs = offs_m * stride_x_hdim + offs_k * stride_x_seqlen
                                //   (offs_m → per_head_dim, offs_k → seqlen within chunk)
                                let x_PL = x_bnlhp
                                    .clone()
                                    .slice(s![
                                        i_batch,
                                        i_chunk,
                                        chunk_range.clone(),
                                        i_head,
                                        per_head_dim_range.clone()
                                    ]) // SRAM load
                                    .squeeze_dims::<2>(&[0, 1, 3])
                                    .permute([1, 0]);
                                assert_eq!([per_head_dim_tile_width, chunk_width], x_PL.dims());

                                // SRAM load: B tile [chunk_block, state_rank_tile]
                                // Triton ptr: b_ptrs = offs_n * stride_b_state_rank + offs_k * stride_b_seqlen
                                //   GQA: group_idx = pid_h / nheads_ngroups_ratio
                                let b_LR = b_bnlgr
                                    .clone()
                                    .slice(s![
                                        i_batch,
                                        i_chunk,
                                        chunk_range.clone(),
                                        i_group,
                                        state_rank_range.clone()
                                    ])
                                    .squeeze_dims::<2>(&[0, 1, 3]); // SRAM load
                                assert_eq!([chunk_width, state_rank_tile_width], b_LR.dims());

                                // Scale B by b_bar_scale[s] (broadcast over d_state columns).
                                // Triton: `b *= scale[:, None]`
                                let scale_for_block_LR = b_bar_scale_l
                                    .clone()
                                    .slice(s![chunk_range.clone()])
                                    .unsqueeze_dim::<2>(1)
                                    .expand([chunk_width, state_rank_tile_width]);

                                let b_scaled_LR = b_LR * scale_for_block_LR;
                                assert_eq!(
                                    [chunk_width, state_rank_tile_width],
                                    b_scaled_LR.dims()
                                );

                                // Tensor Core: X^T[per_head_dim, chunk] @ B_scaled[chunk, state_rank]
                                // Triton: `acc += tl.dot(x, b_scaled)`
                                state_subtile_accum_PR =
                                    state_subtile_accum_PR + x_PL.matmul(b_scaled_LR);
                            } // chunk tile loop

                            // threadIdx: store completed subtile to VRAM.
                            // Triton: `tl.store(states_ptrs, states, mask=...)`
                            intra_chunk_states_bnhpr = intra_chunk_states_bnhpr.slice_assign(
                                s![
                                    i_batch,
                                    i_chunk,
                                    i_head,
                                    per_head_dim_range.clone(),
                                    state_rank_range.clone()
                                ],
                                state_subtile_accum_PR.unsqueeze_dims::<5>(&[0, 0, 0]),
                            );
                        } // state_rank tile loop
                    } // per_head_dim tile loop
                } // nheads loop
            } // chunk loop
        } // batch loop

        // ================================================================
        // KERNEL 4: _state_passing_fwd_kernel  (ssd_state_passing.py)
        //
        // This is the ONLY inherently serial kernel in the SSD pipeline.
        // It performs the SSM recurrence BETWEEN chunks, accumulating history.
        //
        // Triton compile-time constants:
        //   HAS_INITSTATES = True   (initial states from cache.ssm)
        //   HAS_SEQ_IDX    = False
        //
        // The kernel works on the state flattened to a 1D vector per (batch, head):
        //   flat_state_dim = per_head_dim * d_state
        //
        // In the Python call:
        //   states_flat, _ = _state_passing_fwd(
        //       rearrange(intra_chunk_states, "... p n -> ... (p n)"),
        //       dA_chunk_end,
        //       initial_states=rearrange(cache.ssm, "... p n -> ... (p n)"),
        //       out_dtype=C.dtype)    ← output cast to C.dtype (typically fp16/bf16)
        //   chunk_input_states = rearrange(states_flat, "... (p n) -> ... p n", n=d_state)
        //
        // Algorithm (per batch-head pair, serial over chunks):
        //   prev_state = initial_state
        //   out[0] = prev_state                    ← state entering chunk 0
        //   for c in 0..nchunks:
        //     decay = exp(dA_chunk_end[c])
        //     prev_state = decay * prev_state + intra_chunk_states[c]
        //     if c < nchunks - 1:
        //         out[c+1] = prev_state            ← state entering chunk c+1
        //     else:
        //         final_ssm_state = prev_state     ← state after all chunks
        //
        // Grid: (ceil(flat_state_dim / BLOCK_SIZE), batch, nheads)
        //   blockIdx.x = pid_dim_block — tile over the flat state dimension
        //   blockIdx.y = pid_b
        //   blockIdx.z = pid_h
        //
        // threadIdx: each block owns BLOCK_SIZE contiguous elements of the flat state.
        // The serial loop over chunks is INSIDE the thread block (not a new grid dispatch),
        // which is what makes this kernel inherently serial.
        // ================================================================

        // autotune key: ['dim']  where dim = per_head_dim * d_state
        let flat_state_dim = per_head_dim * state_rank;
        // BLOCK_SIZE tiles the flat_state_dim dimension across thread blocks.
        let _state_flat_block_size = autotune(256); // BLOCK_SIZE
        // The Rust loop below processes the entire flat state at once, corresponding to
        // all pid_dim_block blocks combined.
        let _ = _state_flat_block_size;

        // Flatten intra_chunk_states for state-passing arithmetic.
        // Python: `rearrange(states, "... p n -> ... (p n)")`
        let intra_states_bnhf =
            intra_chunk_states_bnhpr.reshape([batch, nchunks, nheads, flat_state_dim]);

        // Initial state from cache, also flattened.
        // Python: `rearrange(initial_states, "... p n -> ... (p n)")`
        let initial_ssm_state_bhf =
            ssm_initial_state_bhpr
                .clone()
                .reshape([batch, nheads, flat_state_dim]);

        // outputs
        // chunk_input_states_flat[b, c, h, :] = state entering chunk c for (batch b, head h).
        let mut chunk_input_states_bnhf =
            Tensor::<B, 4>::zeros([batch, nchunks, nheads, flat_state_dim], &device);
        let mut final_ssm_state_bnf =
            Tensor::<B, 3>::zeros([batch, nheads, flat_state_dim], &device);

        // blockIdx.y = pid_b
        for i_batch in 0..batch {
            // blockIdx.z = pid_h
            for i_head in 0..nheads {
                // blockIdx.x = pid_dim_block tiles the flat state dimension.
                // In a GPU translation, each thread block processes
                //   offs_m = pid_dim_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                // Here we process the full flat state to preserve algorithmic clarity.

                // threadIdx: load initial state for this (batch, head) pair.
                // Triton: `if HAS_INITSTATES: states = tl.load(initstates_ptrs, ...)`
                let mut running_state_f = initial_ssm_state_bhf
                    .clone()
                    .slice(s![i_batch, i_head, ..])
                    .squeeze_dims::<1>(&[0, 1]);
                assert_eq!([flat_state_dim], running_state_f.dims());

                // Store the initial state as the state entering chunk 0.
                // Triton: `tl.store(out_ptrs, states, ...)` BEFORE the loop,
                //         then `out_ptrs += stride_out_chunk`.
                chunk_input_states_bnhf = chunk_input_states_bnhf.slice_assign(
                    s![i_batch, 0, i_head, ..],
                    running_state_f.clone().unsqueeze_dims::<4>(&[0, 1, 2]),
                );

                // Serial loop over chunks — this loop CANNOT be parallelised.
                // Triton: `for c in range(nchunks):`
                for i_chunk in 0..nchunks {
                    // threadIdx: load intra-chunk state for this chunk.
                    // Triton: `new_states = tl.load(states_ptrs, ...)`
                    let intra_state_for_chunk_f = intra_states_bnhf
                        .clone()
                        .slice(s![i_batch, i_chunk, i_head, ..])
                        .squeeze_dims::<1>(&[0, 1, 2]);
                    assert_eq!([flat_state_dim], intra_state_for_chunk_f.dims());

                    // Inter-chunk decay factor: exp(dA_chunk_end[b, h, c]).
                    // Triton: `dA_cs = tl.load(dA_cs_ptr); scale = tl.exp(dA_cs)`
                    // dA_chunk_end is a scalar per (batch, head, chunk).
                    let chunk_decay_f = da_chunk_end_bhn
                        .clone()
                        .slice(s![i_batch, i_head, i_chunk])
                        .squeeze_dims::<1>(&[0, 1]) // [1]
                        .exp()
                        .expand([flat_state_dim]);

                    // SSM recurrence: running_state = decay * running_state + intra_state
                    // Triton: `states = scale * states + new_states`
                    running_state_f = chunk_decay_f * running_state_f + intra_state_for_chunk_f;

                    if i_chunk < nchunks - 1 {
                        // Store the state entering chunk pid_c + 1.
                        // Triton: `if c < nchunks - 1: tl.store(out_ptrs, states, ...)`
                        chunk_input_states_bnhf = chunk_input_states_bnhf.slice_assign(
                            s![i_batch, i_chunk + 1, i_head, ..],
                            running_state_f.clone().unsqueeze_dims::<4>(&[0, 1, 2]),
                        );
                    } else {
                        // Last chunk: store as the final SSM state.
                        // Triton: `else: tl.store(final_states_ptrs, states, ...)`
                        final_ssm_state_bnf = final_ssm_state_bnf.slice_assign(
                            s![i_batch, i_head, ..],
                            running_state_f.clone().unsqueeze_dims::<3>(&[0, 1]),
                        );
                    }
                } // nchunks serial loop
            } // nheads loop
        } // batch loop

        // Restore the 5D shape after state-passing.
        // Python: `rearrange(states_flat, "... (p n) -> ... p n", n=state_rank)`
        let chunk_input_states_bnhpr =
            chunk_input_states_bnhf.reshape([batch, nchunks, nheads, per_head_dim, state_rank]);
        let final_ssm_state_bnpr =
            final_ssm_state_bnf.reshape([batch, nheads, per_head_dim, state_rank]);

        // ================================================================
        // KERNEL 5: _chunk_scan_fwd_kernel  (ssd_chunk_scan.py)
        //
        // Triton compile-time constants for this call (from ssd_combined.py):
        //   IS_CAUSAL     = True   — output pos l only sees source pos s ≤ l
        //   HAS_D         = True   — the skip connection D·x is included
        //   D_HAS_HDIM    = False  — D is [nheads] (scalar per head, not per channel)
        //   HAS_Z         = False  — z gating is NOT done here; it is applied separately
        //                            via self.norm.forward(y, z) after this kernel
        //   HAS_SEQ_IDX   = False  — single-sequence batches
        //
        // For each output position l in chunk c, this kernel accumulates THREE contributions
        // corresponding to the three colour regions of the semiseparable matrix M
        // (see Part III of the paper, "Semiseparable Matrix M Block Decomposition"):
        //
        //   BLUE  — history contribution (off-diagonal, state entering from previous chunks):
        //     y_blue[l, p] = exp(dA_cumsum[l]) * Σ_n  C[l, n] * chunk_input_states[c, p, n]
        //     Triton code:
        //       scale_m = tl.exp(dA_cs_m)
        //       acc = tl.dot(C, prev_states) * scale_m[:, None]
        //
        //   ORANGE — intra-chunk causal contribution (diagonal blocks of M):
        //     y_orange[l, p] = Σ_{s ≤ l}  CB[l,s] · exp(dA[l] - dA[s]) · dt[s] · x[s, p]
        //     Triton code (IS_CAUSAL=True):
        //       K_MAX = min((pid_m + 1) * BLOCK_SIZE_M, chunk_len_limit)
        //       for k in range(0, K_MAX, BLOCK_SIZE_K):
        //           cb *= tl.exp(dA_cs_k - dA_cs_m)  ← note sign: source minus target
        //           cb *= dt_k
        //           mask = offs_m[:, None] >= k + offs_k[None, :]
        //           cb = tl.where(mask, cb, 0.0)
        //           acc += tl.dot(cb, x)
        //     Note: the Triton sign convention for the decay in the orange part is
        //     exp(dA_cs_k - dA_cs_m) which equals exp(dA[source] - dA[target]).
        //     This is the INVERSE of the blue convention and encodes "decay from target
        //     back to source" — equivalent to exp(-(dA[target] - dA[source])).
        //
        //   SKIP  — D residual (outside the state-space recurrence):
        //     y_skip[l, p] = D[h] · x[l, p]
        //     D_HAS_HDIM=False means D is a scalar per head (shape [nheads]).
        //
        // Grid: (ceil(chunk_len/BM) * ceil(per_head_dim/BN), batch * nchunks, nheads)
        //   blockIdx.x = pid_m * n_tiles_N + pid_n  — tile over (chunk_len × per_head_dim)
        //   blockIdx.y = pid_b * nchunks + pid_c    — batch-chunk pair
        //   blockIdx.z = pid_h                      — head index
        // ================================================================

        // autotune key: ['chunk_len', 'hdim', 'state_rank', 'IS_CAUSAL']
        let k5_output_pos_tile = autotune(64); // BLOCK_SIZE_M  (output chunk_pos dimension)
        let k5_per_head_dim_tile = autotune(32); // BLOCK_SIZE_N  (per_head_dim output dimension)
        let k5_state_rank_tile = autotune(64); // BLOCK_SIZE_state_rank (d_state for blue part)
        let k5_source_pos_tile = autotune(32); // BLOCK_SIZE_K  (source chunk_pos for orange)

        // output
        let mut y_bnlhp =
            Tensor::<B, 5>::zeros([batch, nchunks, chunk_len, nheads, per_head_dim], &device);

        // blockIdx.y part-1: pid_b
        for i_batch in 0..batch {
            // blockIdx.y part-2: pid_c
            for i_chunk in 0..nchunks {
                // blockIdx.z: pid_h
                for i_head in 0..nheads {
                    // GQA: CB and C use group_idx for the head dimension.
                    // Triton: `(pid_h // nheads_ngroups_ratio) * stride_cb_head`
                    let i_group = i_head / heads_per_group;

                    // Load dA_cumsum for this (batch, head, chunk) — shape [chunk_len].
                    // Triton ptr: dA_cumsum_ptr + pid_b*stride_batch + pid_c*stride_chunk + pid_h*stride_head
                    let da_cumsum_l = da_cumsum_bhnl
                        .clone()
                        .slice(s![i_batch, i_head, i_chunk, ..])
                        .squeeze_dims::<1>(&[0, 1, 2]);
                    assert_eq!([chunk_len], da_cumsum_l.dims());

                    // Load dt_discretized for this (batch, head, chunk) — shape [chunk_len].
                    // Triton ptr: dt_ptr + pid_b*stride + pid_c*stride + pid_h*stride
                    let dt_l = dt_discretized_bhnl
                        .clone()
                        .slice(s![i_batch, i_head, i_chunk, ..])
                        .squeeze_dims::<1>(&[0, 1, 2]);
                    assert_eq!([chunk_len], dt_l.dims());

                    // Load the state entering this chunk: shape [per_head_dim, d_state].
                    // Triton ptr: prev_states_ptr + pid_b*stride_batch + pid_c*stride_chunk + pid_h*stride_head
                    // Layout: prev_states[per_head_dim_p, state_rank_n] (row=per_head_dim, col=d_state)
                    let state_entering_pr = chunk_input_states_bnhpr
                        .clone()
                        .slice(s![i_batch, i_chunk, i_head, .., ..])
                        .squeeze_dims::<2>(&[0, 1, 2]);
                    assert_eq!([per_head_dim, state_rank], state_entering_pr.dims());

                    // Load x for this (batch, chunk, head): shape [chunk_len, per_head_dim].
                    // Triton ptr: x_ptr + pid_b*stride + pid_c*chunk_len*stride_seqlen + pid_h*stride_head
                    let x_lp = x_bnlhp
                        .clone()
                        .slice(s![i_batch, i_chunk, .., i_head, ..])
                        .squeeze_dims::<2>(&[0, 1, 3]);
                    assert_eq!([chunk_len, per_head_dim], x_lp.dims());

                    // Load CB for this (batch, chunk, group): shape [chunk_len, chunk_len].
                    // Triton ptr: cb_ptr + ... (pid_h // ratio) * stride_cb_head
                    let cb_ll = cb_bngll
                        .clone()
                        .slice(s![i_batch, i_chunk, i_group, .., ..])
                        .squeeze_dims::<2>(&[0, 1, 2]);
                    assert_eq!([chunk_len, chunk_len], cb_ll.dims());
                    // [target, source]

                    // D skip coefficient for this head.
                    // D_HAS_HDIM=False: D is [nheads], one scalar per head.
                    // Triton: `D = tl.load(D_ptr + pid_h * stride_D_head)` (single scalar load)
                    let d_skip_1 = d_h.clone().slice(s![i_head]);
                    assert_eq!([1], d_skip_1.dims());

                    // Accumulate the full output for this (batch, chunk, head).
                    // In the GPU, each (pid_m, pid_n) sub-block is a separate thread block.
                    // Here we use a single [chunk_len, per_head_dim] tensor.
                    let mut y_lp = Tensor::<B, 2>::zeros([chunk_len, per_head_dim], &device);

                    // blockIdx.x: pid_m — output target-position tile
                    for output_pos_start in (0..chunk_len).step_by(k5_output_pos_tile) {
                        let output_pos_end = (output_pos_start + k5_output_pos_tile).min(chunk_len);
                        let output_pos_tile_width = output_pos_end - output_pos_start;
                        let output_pos_range = output_pos_start..output_pos_end;
                        assert_ne!(output_pos_tile_width, 0);

                        // blockIdx.x: pid_n — per_head_dim output tile
                        for per_head_dim_start in (0..per_head_dim).step_by(k5_per_head_dim_tile) {
                            let per_head_dim_end =
                                (per_head_dim_start + k5_per_head_dim_tile).min(per_head_dim);
                            let per_head_dim_tile_width = per_head_dim_end - per_head_dim_start;
                            let per_head_dim_range = per_head_dim_start..per_head_dim_end;
                            assert_ne!(per_head_dim_tile_width, 0);

                            // threadIdx: accumulator for this [output_pos_tile × per_head_dim_tile] block.
                            // Triton: `acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)`
                            let mut output_subtile_accum_LP = Tensor::<B, 2>::zeros(
                                [output_pos_tile_width, per_head_dim_tile_width],
                                &device,
                            );

                            // ----------------------------------------------------------------
                            // BLUE contribution — prev-state propagated forward to each output pos.
                            //
                            // Mathematical form:
                            //   y_blue[l, p] = exp(dA_cumsum[l]) * Σ_n  C[l, n] * state[p, n]
                            //
                            // Triton loads and stores:
                            //   C_ptrs      = offs_m * stride_C_seqlen + offs_k_state_rank * stride_C_state_rank
                            //   prev_states_ptrs = offs_n * stride_states_hdim + offs_k_state_rank * stride_states_state_rank
                            //
                            // Note the index roles:
                            //   offs_m  → output chunk positions (l),  shapes C as [output_pos, d_state]
                            //   offs_n  → per_head_dim output (p),          shapes prev_states as [d_state, per_head_dim]
                            //   offs_k_state_rank → inner d_state dim (n)
                            //
                            // Matmul: C[l, n] @ prev_states[n, p] = [output_pos_tile, per_head_dim_tile]
                            // ----------------------------------------------------------------
                            // threadIdx K-loop over d_state (BLOCK_SIZE_state_rank steps).
                            for state_rank_start in (0..state_rank).step_by(k5_state_rank_tile) {
                                let state_rank_end =
                                    (state_rank_start + k5_state_rank_tile).min(state_rank);
                                let state_rank_block_width = state_rank_end - state_rank_start;
                                let state_rank_range = state_rank_start..state_rank_end;
                                assert_ne!(state_rank_block_width, 0);

                                // SRAM load: C tile [output_pos_tile, state_rank_block]
                                // Triton: C_ptrs = C_ptr + offs_m * stride_C_seqlen + offs_k * stride_C_state_rank
                                //         GQA: group_idx applied
                                let c_LR = c_bnlgr
                                    .clone()
                                    .slice(s![
                                        i_batch,
                                        i_chunk,
                                        output_pos_range.clone(),
                                        i_group,
                                        state_rank_range.clone()
                                    ])
                                    .squeeze_dims::<2>(&[0, 1, 3]);
                                assert_eq!(
                                    [output_pos_tile_width, state_rank_block_width],
                                    c_LR.dims()
                                );

                                // SRAM load: prev_state tile [state_rank_block, per_head_dim_tile]
                                // Triton: prev_states_ptrs = offs_n * stride_states_hdim
                                //                           + offs_k * stride_states_state_rank
                                // state_entering_this_chunk has shape [per_head_dim, d_state].
                                // We slice [per_head_dim_start..per_head_dim_end, state_rank_start..state_rank_end]
                                // = [per_head_dim_tile, state_rank_block], then transpose to [state_rank_block, per_head_dim_tile].
                                // This matches the Triton access pattern where offs_n indexes per_head_dim
                                // and offs_k indexes d_state.
                                let state_entering_RP = state_entering_pr
                                    .clone()
                                    .slice(s![per_head_dim_range.clone(), state_rank_range.clone()])
                                    .permute([1, 0]); // transpose
                                assert_eq!(
                                    [state_rank_block_width, per_head_dim_tile_width],
                                    state_entering_RP.dims()
                                );

                                // Tensor Core: C[output_pos, state_rank] @ prev_state^T[state_rank, per_head_dim]
                                // Triton: `acc = tl.dot(C, prev_states) * scale_m[:, None]`
                                // (scale is applied after the d_state accumulation loop)
                                output_subtile_accum_LP =
                                    output_subtile_accum_LP + c_LR.matmul(state_entering_RP);
                            } // state_rank tile loop

                            // Apply the state decay to the blue term.
                            // exp(dA_cumsum[l]) = how much the initial state has decayed
                            // by the time we reach output position l in this chunk.
                            // Triton: `acc = tl.dot(...) * scale_m[:, None]`
                            //   where `scale_m = tl.exp(dA_cs_m)`
                            let state_decay_LP = da_cumsum_l
                                .clone()
                                .slice(s![output_pos_range.clone()])
                                .exp()
                                .unsqueeze_dim::<2>(1)
                                .expand([output_pos_tile_width, per_head_dim_tile_width]);
                            output_subtile_accum_LP = output_subtile_accum_LP * state_decay_LP;

                            // ----------------------------------------------------------------
                            // ORANGE contribution — intra-chunk causal attention.
                            //
                            // IS_CAUSAL=True: source position s can only contribute to output
                            // position l when s ≤ l.
                            //
                            // K_MAX (Triton): `min((pid_m + 1) * BLOCK_SIZE_M, chunk_len_limit)`
                            //   = output_pos_end in our notation.
                            // Source positions beyond output_pos_end cannot causally contribute
                            // to any output in [output_pos_start, output_pos_end), so we skip them.
                            //
                            // Triton orange loop (from _chunk_scan_fwd_kernel in ssd_chunk_scan.py):
                            //   for k in range(0, K_MAX, BLOCK_SIZE_K):
                            //     cb = tl.load(cb_ptrs, ...)
                            //     dA_cs_k = tl.load(dA_cumsum_ptrs, ...)   ← source positions
                            //     cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
                            //                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            //                  target_cumsum - source_cumsum
                            //                  Since dA < 0 and target ≥ source in causal region,
                            //                  dA_cumsum[target] ≤ dA_cumsum[source], so the
                            //                  difference is ≤ 0 and exp(⋅) ∈ (0, 1].
                            //     dt_k = tl.load(dt_ptrs, ...)
                            //     cb *= dt_k
                            //     if IS_CAUSAL:
                            //         mask = offs_m[:, None] >= k + offs_k[None, :]
                            //         cb = tl.where(mask, cb, 0.0)
                            //     acc += tl.dot(cb, x)
                            // ----------------------------------------------------------------
                            let source_pos_upper_limit = output_pos_end; // K_MAX = (pid_m+1)*BM
                            for source_pos_start in
                                (0..source_pos_upper_limit).step_by(k5_source_pos_tile)
                            {
                                let source_pos_end = (source_pos_start + k5_source_pos_tile)
                                    .min(source_pos_upper_limit);
                                let source_pos_tile_width = source_pos_end - source_pos_start;
                                let source_pos_range = source_pos_start..source_pos_end;
                                assert_ne!(source_pos_tile_width, 0);

                                // SRAM load: CB tile [output_pos_tile, source_pos_tile]
                                // Triton: cb_ptrs = cb_ptr + offs_m * stride_cb_csize_m
                                //                          + offs_k * stride_cb_csize_k
                                //         GQA: (pid_h // ratio) * stride_cb_head
                                let cb_LL = cb_ll
                                    .clone()
                                    .slice(s![output_pos_range.clone(), source_pos_range.clone()]);
                                assert_eq!(
                                    [output_pos_tile_width, source_pos_tile_width],
                                    cb_LL.dims()
                                );

                                // Inter-position decay: exp(dA_cumsum[target] - dA_cumsum[source])
                                //
                                // Triton: `cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])`
                                //   dA_cs_m → dA_cumsum at target (output) positions  (loaded outside K-loop)
                                //   dA_cs_k → dA_cumsum at source positions            (loaded inside K-loop)
                                //
                                // Physical meaning: the state at source position s has decayed
                                // by factor exp(sum of dA[s+1..=l]) when it reaches target l.
                                // Using the inclusive cumsum identity:
                                //   dA_cumsum[l] - dA_cumsum[s] = sum(dA[s+1..=l])
                                // So exp(dA_cumsum[target] - dA_cumsum[source]) encodes exactly that
                                // decay.  Because dA < 0 and target ≥ source (causal), the difference
                                // is ≤ 0 and the factor lies in (0, 1]. This is consistent with the
                                // BLUE contribution which uses exp(dA_cumsum[l]) = decay from chunk
                                // start (before position 0) to position l.
                                let da_cumsum_at_target_LL = da_cumsum_l
                                    .clone()
                                    .slice(s![output_pos_range.clone()])
                                    .unsqueeze_dim::<2>(1)
                                    .expand([output_pos_tile_width, source_pos_tile_width]);
                                let da_cumsum_at_source_LL = da_cumsum_l
                                    .clone()
                                    .slice(s![source_pos_range.clone()])
                                    .unsqueeze_dim::<2>(0)
                                    .expand([output_pos_tile_width, source_pos_tile_width]);

                                // dA_at_source: [output_pos_tile_width, source_pos_tile_width]
                                //
                                // IMPORTANT: sign is (target - source), NOT (source - target).
                                // target_cumsum ≤ source_cumsum in the causal region (both are
                                // non-positive and target has accumulated more negative dA steps),
                                // so the difference is ≤ 0 and exp(⋅) ∈ (0, 1] as required.
                                let causal_intra_chunk_decay_LL =
                                    (da_cumsum_at_target_LL - da_cumsum_at_source_LL).exp();
                                assert_eq!(
                                    [output_pos_tile_width, source_pos_tile_width],
                                    causal_intra_chunk_decay_LL.dims()
                                );

                                // Discretization scale at each source position.
                                // Triton: `dt_k = tl.load(dt_ptrs, ...); cb *= dt_k`
                                let dt_at_source_LL = dt_l
                                    .clone()
                                    .slice([source_pos_range.clone()])
                                    .unsqueeze_dim::<2>(0)
                                    .expand([output_pos_tile_width, source_pos_tile_width]);

                                let cb_weighted_LL =
                                    cb_LL * causal_intra_chunk_decay_LL * dt_at_source_LL;
                                assert_eq!(
                                    [output_pos_tile_width, source_pos_tile_width],
                                    cb_weighted_LL.dims()
                                );

                                // Causal mask: output position (absolute) >= source position (absolute).
                                // Triton: `mask = offs_m[:, None] >= k + offs_k[None, :]`
                                //   where k = source_pos_start in our notation.
                                // In CUDA this would be a per-thread integer comparison:
                                //   valid = (output_pos_start + threadIdx_m) >= (source_pos_start + threadIdx_k)
                                // We approximate it with float broadcasting:
                                let causal_mask_LL = {
                                    let output_abs_indices: Vec<f32> =
                                        output_pos_range.clone().map(|i| i as f32).collect();
                                    let source_abs_indices: Vec<f32> =
                                        source_pos_range.clone().map(|j| j as f32).collect();
                                    let output_abs_float_LL = Tensor::<B, 1>::from_floats(
                                        output_abs_indices.as_slice(),
                                        &device,
                                    )
                                    .unsqueeze_dim::<2>(1)
                                    .expand([output_pos_tile_width, source_pos_tile_width]);
                                    let source_abs_float_LL = Tensor::<B, 1>::from_floats(
                                        source_abs_indices.as_slice(),
                                        &device,
                                    )
                                    .unsqueeze_dim::<2>(0)
                                    .expand([output_pos_tile_width, source_pos_tile_width]);
                                    // causal_mask[target, source] = True when target_abs >= source_abs
                                    output_abs_float_LL.greater_equal(source_abs_float_LL)
                                };

                                // Zero out anti-causal entries.
                                // Triton: `cb = tl.where(mask, cb, 0.0)`
                                let cb_causal_LL =
                                    cb_weighted_LL.mask_fill(causal_mask_LL.bool_not(), 0.0);

                                // SRAM load: X tile [source_pos_tile, per_head_dim_tile]
                                // Triton: x_ptrs = offs_k * stride_x_seqlen + offs_n * stride_x_hdim
                                let x_LP = x_lp
                                    .clone()
                                    .slice([source_pos_range.clone(), per_head_dim_range.clone()]);
                                assert_eq!(
                                    [source_pos_tile_width, per_head_dim_tile_width],
                                    x_LP.dims()
                                );

                                // Tensor Core: CB_causal[output, source] @ X[source, per_head_dim]
                                // Triton: `acc += tl.dot(cb_causal.to(x.dtype), x)`
                                output_subtile_accum_LP =
                                    output_subtile_accum_LP + cb_causal_LL.matmul(x_LP);
                            } // source_pos tile loop

                            // ----------------------------------------------------------------
                            // SKIP contribution — D residual connection.
                            //
                            // HAS_D=True, D_HAS_HDIM=False: D is a scalar per head.
                            // Mathematical form: y_skip[l, p] = D[h] · x[l, p]
                            //
                            // Triton:
                            //   D = tl.load(D_ptr + pid_h * stride_D_head)  ← scalar
                            //   x_residual = tl.load(x_ptr + offs_m * stride_seqlen
                            //                                + offs_n * stride_hdim)
                            //   acc += x_residual * D
                            // ----------------------------------------------------------------
                            let x_skip_LP = x_lp
                                .clone()
                                .slice(s![output_pos_range.clone(), per_head_dim_range.clone()]);
                            assert_eq!(
                                [output_pos_tile_width, per_head_dim_tile_width],
                                x_skip_LP.dims()
                            );

                            let d_skip_LP = d_skip_1
                                .clone()
                                .unsqueeze_dim::<2>(0)
                                .expand([output_pos_tile_width, per_head_dim_tile_width]);

                            output_subtile_accum_LP =
                                output_subtile_accum_LP + d_skip_LP * x_skip_LP;

                            // threadIdx: store the completed [output_pos_tile × per_head_dim_tile]
                            // subtile back to the y_for_chunk accumulator (SRAM → registers).
                            // Triton: `tl.store(out_ptrs, acc, mask=...)`
                            y_lp = y_lp.slice_assign(
                                s![output_pos_range.clone(), per_head_dim_range.clone()],
                                output_subtile_accum_LP,
                            );
                        } // per_head_dim tile loop
                    } // output_pos tile loop

                    // Write this (batch, chunk, head) output tile to VRAM.
                    // Triton: out_ptr += pid_b*stride_batch + pid_c*chunk_len*stride_seqlen
                    //                  + pid_h*stride_head
                    y_bnlhp = y_bnlhp.slice_assign(
                        s![i_batch, i_chunk, 0..chunk_len, i_head, 0..per_head_dim],
                        y_lp
                            // TOD0: unsqueeze_dims::<5>(&[0, 1, 3]) currently panics
                            .unsqueeze_dim::<3>(0)
                            .unsqueeze_dim::<4>(1)
                            .unsqueeze_dim::<5>(3),
                    );
                } // nheads loop
            } // nchunks loop
        } // batch loop

        // Reshape from chunked layout back to flat sequence.
        let y_bshp = y_bnlhp.reshape([batch, sequence, nheads, per_head_dim]);

        (y_bshp, final_ssm_state_bnpr)
    }
}
