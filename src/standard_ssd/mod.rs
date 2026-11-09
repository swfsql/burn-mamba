//! # The standard SSD scan on primitives
//!
//! One SSD call computes the scan of one recurrence with a scalar decay, chunk
//! by chunk:
//!
//! ```text
//! hₜ = exp(daₜ)·hₜ₋₁ + Bₜ vₜᵀ
//! yₜ = Cₜ hₜ
//! ```
//!
//! `v` is the input after its scale, and `da` is the log-decay `Δ·A`. This
//! module holds the forward K1–K5 and the analytic backward of that call. They
//! use backend primitives, so that a
//! [recompute node](burn_stack::utils::recompute_node) can run them.
//!
//! The standard SSD scan has two callers, in their `SerialRecalculated` paths:
//!
//! - **Mamba-2** makes one SSD call for each call of its block. It forms
//!   `v = Δ·x` and `da = Δ·A`, and it adds the `D` skip and the chain rule back
//!   to its own inputs (`crate::mamba2::ssd::serial_recalculated`).
//! - The **double-SSD pathway** of Mamba-3 makes one SSD call for each term of
//!   the trapezoid (`crate::mamba3::double_ssd::ssd::serial_recalculated`).
//!
//! The single-SSD pathway makes one SSD call for all terms, with its own
//! forward K5 and its own diagonal term. It assembles its backward from the
//! backwards of K1–K5 in `kernels.rs`, and it uses K1–K4 and the helpers of
//! this module.
//!
//! The scan has the two axes that the callers need: `mimo_rank` and the read
//! axis. A caller without such an axis passes a length-1 `m` axis and
//! `read_stride = 1`. Each math statement of the scan is in the header of
//! `src/mamba2/mamba2.rs` or `src/mamba3/mamba3.rs`. The decision to keep one
//! scan is ADR-0010.
//!
//! This module compiles when the `mamba2` or the `mamba3` feature is on.

pub mod combined_backward;
pub mod kernels;
pub mod read_axis;

pub use combined_backward::{CombinedGrads, combined_backward};
pub use kernels::forward;

/// Tokens per chunk: `chunk_len / micro_steps`, the **read** axis of the
/// chunk.
///
/// `chunk_len` counts write positions (one per micro-step). The SSD reads each
/// token once, at its last micro-step. So this is the row count of the score
/// and of the `y` that the kernels return.
///
/// # Panics
/// If `micro_steps` does not divide `chunk_len`.
pub fn chunk_tokens(chunk_len: usize, micro_steps: usize) -> usize {
    let micro_steps = micro_steps.max(1);
    assert_eq!(
        chunk_len % micro_steps,
        0,
        "chunk_len {chunk_len} must be a whole number of tokens at micro_steps {micro_steps}",
    );
    chunk_len / micro_steps
}

/// Chunks per iteration of the chunk-local pass of [`combined_backward()`]: how
/// many chunks that pass batches.
///
/// The recompute backward does not need to *walk* the chunk axis. The
/// chunk-local gradients are the state-to-output term and the intra-chunk
/// triangular term. Each needs only the slices of its chunk and its input
/// state, and the recomputed K4 already produced that state, batched. Only the
/// state gradient is a scan, and it is a few ops per chunk.
///
/// The cost of batching the rest is memory. The score
/// `[batch, group, nheads, read, fused]` is live `group`-wide. At its peak this
/// pass holds about six score-shaped tensors, where the forward K5 holds about
/// four at **full** width. So **half** stays under the peak that the forward
/// already reaches, and the walk is two iterations at any chunk count. The
/// bound is a ratio of live tensors, so it needs no dimension and no absolute
/// budget. In both cases, the SSD path carries no intermediates across the
/// forward/backward boundary. When this pass runs, the forward intermediates
/// are already free.
pub fn backward_chunk_group(nchunks: usize) -> usize {
    nchunks.div_ceil(2).max(1)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
