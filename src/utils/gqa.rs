//! Grouped-Query Attention (GQA) dimension expansion.
//!
//! Mamba-2 and Mamba-3 produce B and C projections per-group (size `ngroups`),
//! but the chunkwise SSD algorithms consume them per-head (size `nheads`).
//! This helper bridges the two by replicating each group's vector across the
//! `heads_per_group = nheads / ngroups` heads of that group.

use burn::prelude::*;

/// Expand a tensor's `ngroups` dim at `group_dim` into an `nheads` dim, by
/// replicating each group's slice across `heads_per_group = nheads / ngroups`
/// heads of that group.
///
/// The const generic `DP1` must equal `D + 1` (the rank used during the
/// intermediate `unsqueeze`+`expand`). Rust cannot yet express that constraint
/// directly, so it is the caller's responsibility — supplying a wrong value
/// produces a compile-time rank mismatch from `unsqueeze_dim::<DP1>` / `reshape`.
///
/// # Panics
/// Panics if `nheads % ngroups != 0` (i.e. `nheads` is not a multiple of the
/// current group count).
///
/// # Example
/// ```ignore
/// // b_bnlgr: [batch, nchunks, chunk_len, ngroups, state_rank] (D=5)
/// // group_dim = 3 (the ngroups axis)
/// // result:  [batch, nchunks, chunk_len, nheads,  state_rank]
/// let b_bnlhr = gqa_expand_to_heads::<_, 5, 6>(b_bnlgr, 3, nheads);
/// ```
pub fn gqa_expand_to_heads<B: Backend, const D: usize, const DP1: usize>(
    t: Tensor<B, D>,
    group_dim: usize,
    nheads: usize,
) -> Tensor<B, D> {
    let dims = t.dims();
    let ngroups = dims[group_dim];
    assert!(
        nheads.is_multiple_of(ngroups),
        "nheads ({nheads}) must be a multiple of ngroups ({ngroups})"
    );
    let heads_per_group = nheads / ngroups;

    // Expanded shape: insert `heads_per_group` immediately after `group_dim`.
    let mut expanded = [0usize; DP1];
    expanded[..=group_dim].copy_from_slice(&dims[..=group_dim]);
    expanded[group_dim + 1] = heads_per_group;
    expanded[group_dim + 2..].copy_from_slice(&dims[group_dim + 1..]);

    // Final shape: collapse `(ngroups, heads_per_group)` back into `nheads`.
    let mut final_shape = dims;
    final_shape[group_dim] = nheads;

    t.unsqueeze_dim::<DP1>(group_dim + 1)
        .expand(expanded)
        .reshape(final_shape)
}
