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

impl<B: Backend> Mamba2<B>
where
    B: crate::mamba2::gpu::BackendExt,
{
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

        // ── Permutes ──────────────────────────────────────────────────────────────────
        // Note: dt_bnlh calculation (originally in Kernel 1) moved to Step 4 (before padding).
        let dt_discretized_bhnl = dt_bnlh.permute([0, 3, 1, 2]);
        assert_eq!(
            [batch, nheads, nchunks, chunk_len],
            dt_discretized_bhnl.dims()
        );

        // ── Kernel 1 ──────────────────────────────────────────────────────────────────
        let da_cumsum_bhnl: Tensor<B, 4> = crate::mamba2::gpu::k1_ssd_chunk_cumsum::naive::forward(
            dt_discretized_bhnl.clone(),
            a_decay_h.clone(),
        );
        assert_eq!([batch, nheads, nchunks, chunk_len], da_cumsum_bhnl.dims());

        // dA_chunk_end[b, h, c] = dA_cumsum[b, h, c, chunk_len-1]
        // = total log-decay accumulated over the entire chunk c for head h.
        // Used in K4 as the inter-chunk decay exponent.
        // Triton in _mamba_chunk_scan_combined_fwd: `dA_cumsum[:, :, :, -1]`
        let da_chunk_end_bhn = da_cumsum_bhnl
            .clone()
            .slice(s![.., .., .., -1])
            .squeeze_dim::<3>(3);
        assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());

        // ── Kernel 2 ──────────────────────────────────────────────────────────────────
        let c_bnglr = c_bnlgr.clone().permute([0, 1, 3, 2, 4]);
        let b_bngrl = b_bnlgr.clone().permute([0, 1, 3, 4, 2]);
        let cb_bngll: Tensor<B, 5> = c_bnglr.matmul(b_bngrl);
        assert_eq!(
            [batch, nchunks, ngroups, chunk_len, chunk_len],
            cb_bngll.dims()
        );
        // Note: cb_bngll is then only used by Kernel 5.

        // ── Kernel 3 ──────────────────────────────────────────────────────────────────
        let intra_chunk_states_bnhpr: Tensor<B, 5> = {
            use burn::tensor::s;

            let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
            let [_batch, _nchunks, _chunk_len, ngroups, state_rank] = b_bnlgr.dims();

            // permute b and x to prepare them for the mamtul
            let x_bnhpl = x_bnlhp.clone().permute([0, 1, 3, 4, 2]);
            assert_eq!(
                [batch, nchunks, nheads, per_head_dim, chunk_len],
                x_bnhpl.dims()
            );
            let b_bnglr = b_bnlgr.permute([0, 1, 3, 2, 4]); // note: still in groups instead of heads
            assert_eq!(
                [batch, nchunks, ngroups, chunk_len, state_rank],
                b_bnglr.dims()
            );

            // Expand B from ngroups to nheads by repeating each group's
            // projection across all heads_per_group heads in that group.
            let heads_per_group = nheads / ngroups;
            let b_bnhlr = b_bnglr
                .unsqueeze_dim::<6>(3) // bng1lr
                .expand([
                    batch,
                    nchunks,
                    ngroups,
                    heads_per_group,
                    chunk_len,
                    state_rank,
                ])
                .reshape([batch, nchunks, nheads, chunk_len, state_rank]);

            // scale b
            let b_scaled_bnhlr = {
                let b_bar_scale_bhnl = {
                    let da_cumsum_last_in_chunk_bhn1 =
                        da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
                    assert_eq!(
                        [batch, nheads, nchunks, 1],
                        da_cumsum_last_in_chunk_bhn1.dims()
                    );

                    let forward_decay_to_chunk_end_bhnl = (da_cumsum_last_in_chunk_bhn1
                        .expand([batch, nheads, nchunks, chunk_len])
                        - da_cumsum_bhnl.clone())
                    .exp();
                    assert_eq!(
                        [batch, nheads, nchunks, chunk_len],
                        forward_decay_to_chunk_end_bhnl.dims()
                    );

                    forward_decay_to_chunk_end_bhnl * dt_discretized_bhnl.clone()
                };
                assert_eq!([batch, nheads, nchunks, chunk_len], b_bar_scale_bhnl.dims());

                let b_bar_scale_bnhl = b_bar_scale_bhnl.permute([0, 2, 1, 3]);
                assert_eq!([batch, nchunks, nheads, chunk_len], b_bar_scale_bnhl.dims());
                let b_bar_scale_bnhlr = b_bar_scale_bnhl
                    .unsqueeze_dim::<5>(4) // bnhl1
                    .expand([batch, nchunks, nheads, chunk_len, state_rank]);
                b_bnhlr * b_bar_scale_bnhlr
            };
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                b_scaled_bnhlr.dims()
            );

            x_bnhpl.matmul(b_scaled_bnhlr) // intra_chunk_states_bnhpr  
        };
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            intra_chunk_states_bnhpr.dims()
        );

        // ── Kernel 4 ──────────────────────────────────────────────────────────────────
        let flat_state_dim = per_head_dim * state_rank;
        let (chunk_input_states_bnhpr, final_ssm_state_bhpr): (Tensor<B, 5>, Tensor<B, 4>) =
            crate::mamba2::gpu::k4_ssd_state_passing::naive::forward(
                intra_chunk_states_bnhpr.clone(),
                da_chunk_end_bhn.clone(),
                ssm_initial_state_bhpr.clone(),
            );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            chunk_input_states_bnhpr.dims()
        );
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            final_ssm_state_bhpr.dims()
        );

        // ── Kernel 5 ──────────────────────────────────────────────────────────────────
        let y_bnlhp: Tensor<B, 5> = crate::mamba2::gpu::k5_ssd_chunk_scan::naive::forward(
            da_cumsum_bhnl.clone(),
            dt_discretized_bhnl.clone(),
            x_bnlhp.clone(),
            c_bnlgr.clone(),
            cb_bngll.clone(),
            chunk_input_states_bnhpr.clone(),
            d_h.clone(),
        );
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_bnlhp.dims()
        );

        // Reshape from chunked layout back to flat sequence.
        let y_bshp = y_bnlhp.reshape([batch, sequence, nheads, per_head_dim]);

        (y_bshp, final_ssm_state_bhpr)
    }
}
