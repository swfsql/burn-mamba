//! # Pathway-agnostic SSD algorithm selection (Mamba-3)
//!
//! [`Mamba3SsdPath`] picks the chunkwise SSD *algorithm* (Minimal / Serial /
//! SerialRecalculated) and chunk length, independent of the double-vs-single
//! *pathway* (which the supplied cache variant selects).  It converts into the
//! per-pathway path types via `From` and is threaded by
//! [`Mamba3::forward`](crate::mamba3::mamba3::Mamba3::forward) into whichever
//! pathway the cache implies.

use crate::mamba3::prelude::*;

/// Algorithm selection for the Mamba-3 chunkwise SSD.
///
/// This selects the chunkwise SSD *algorithm*. The *pathway* (double- vs
/// single-ssd) is selected separately, by the supplied cache variant (see
/// [`crate::mamba3::cache::Mamba3Caches`]); [`Mamba3::forward`] threads this
/// same selection into whichever pathway the cache implies, converting it into
/// the per-pathway input bundle ([`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput`]
/// or [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput`]) and calling
/// that bundle's `run`.
///
/// Each variant carries an optional chunk length. Larger values increase the
/// intra-chunk GEMM work and reduce the inter-chunk scan length; the optimal
/// value is approximately `√(state_rank · per_head_dim)`, divided by whatever
/// `mimo_rank` and `micro_steps` already widen the chunk by (see
/// [`Self::optimal_chunk_len`]). `None` falls back to that optimal value.
///
/// If no path is specified, the cache defaults to
/// [`crate::mamba3::cache::Mamba3Caches::SingleSsd`] with [`Self::default`]
/// (i.e. [`Self::SerialRecalculated`] with an unset chunk length).
#[derive(Debug, Clone)]
pub enum Mamba3SsdPath {
    /// Minimal/segsum SSD: mostly batched matmuls; backward via autodiff.
    ///
    /// See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_minimal`]
    /// / [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_minimal`].
    /// For training, prefer [`Self::SerialRecalculated`].
    Minimal(Option<usize>),

    /// (Hybrid) serial SSD: a serial loop over the chunks plus batched matmuls;
    /// backward via autodiff.
    ///
    /// See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_serial`]
    /// / [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_serial`].
    /// For a memory-saving custom backward, see [`Self::SerialRecalculated`].
    Serial(Option<usize>),

    /// (Hybrid) serial SSD with a custom, memory-efficient backward that
    /// recomputes the forward intermediates instead of storing them.
    ///
    /// See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_serial_recalculated`]
    /// / [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_serial_recalculated`].
    /// For a plain autodiff backward, see [`Self::Serial`].
    SerialRecalculated(Option<usize>),
}

impl Mamba3SsdPath {
    /// Optimal chunk length, in **folded positions**: `√(state_rank ·
    /// per_head_dim)` divided by `mimo_rank`, on the 32 grid, clamped to
    /// `32 ..= 512` — then held there while [`Self::chunk_tokens`] absorbs
    /// `micro_steps`.
    ///
    /// The square root is the SISO rule of thumb — it balances the intra-chunk
    /// GEMMs against the inter-chunk scan. Two of the block's dials widen a chunk
    /// without appearing in it, and they are **not** the same widening:
    ///
    /// - **`mimo_rank`** (`m`) widens *both* of a chunk's axes: every rank writes
    ///   and every rank reads, so the intra-chunk matmuls run on the fused
    ///   `chunk_len · m`. The quantity the rule of thumb is about is that
    ///   product, hence the divisor. Leaving `chunk_len` alone costs `m²` where
    ///   this costs `m` — the source's own advice, carried on its `chunk_size`
    ///   argument as "64 for SISO, 64/mimo_rank for MIMO".
    /// - **`micro_steps`** (`u`) widens only the **write** axis. A token's `u`
    ///   micro-steps all write to the state, but the token is *read* once, at its
    ///   last one, so the score is `[batch, nchunks, heads, T·m, T·u·m]` for
    ///   `T` = [`Self::chunk_tokens`] and holds `batch · sequence · heads · T · u
    ///   · m²` elements. Keeping that flat in `u` means keeping `T · u` — the
    ///   *folded* chunk — flat, and shrinking the token count instead. Memory is
    ///   the binding constraint here, there being no fused kernel to hide it.
    ///
    /// So `u` no longer shortens the chunk, only subdivides it, and `nchunks`
    /// grows like `u` rather than `u²`. The returned length is a multiple of
    /// `micro_steps` (which is what makes each chunk a whole number of tokens,
    /// and their rows a contiguous run); it is on the 32 grid exactly when
    /// `micro_steps` divides it.
    ///
    /// `info/architecture-deltas.md` §8.
    pub fn optimal_chunk_len(
        state_rank: usize,
        per_head_dim: usize,
        mimo_rank: usize,
        micro_steps: usize,
    ) -> usize {
        let micro_steps = micro_steps.max(1);
        let folded = (state_rank * per_head_dim)
            .isqrt()
            .div_ceil(mimo_rank.max(1)) // the fused read axis is `chunk_tokens · m`.
            .next_multiple_of(32) // rule-of-thumb: common plane dimension.
            .clamp(32, 512); // rule-of-thumb: one plane minimum, ceiling at 512.
        // A whole number of tokens per chunk, at (very nearly) that folded width.
        folded.div_ceil(micro_steps) * micro_steps
    }

    /// Tokens per chunk: `chunk_len / micro_steps`, the chunk's **read** axis.
    ///
    /// `chunk_len` counts folded positions (one per micro-step); the SSD reads
    /// each token once, at its last micro-step, so this is the row count of the
    /// score and of the `y` the kernels return. See [`Self::optimal_chunk_len`].
    ///
    /// # Panics
    /// If `micro_steps` does not divide `chunk_len` — [`Self::chunk_len_or_optimal`]
    /// is what guarantees it does.
    pub fn chunk_tokens(chunk_len: usize, micro_steps: usize) -> usize {
        let micro_steps = micro_steps.max(1);
        assert_eq!(
            chunk_len % micro_steps,
            0,
            "chunk_len {chunk_len} must be a whole number of tokens at micro_steps {micro_steps}",
        );
        chunk_len / micro_steps
    }

    /// The chunk length carried by this variant, if any.
    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            Self::Minimal(chunk_len)
            | Self::Serial(chunk_len)
            | Self::SerialRecalculated(chunk_len) => *chunk_len,
        }
    }

    /// The chunk length carried by this variant, or [`Self::optimal_chunk_len`]
    /// for `block`'s dimensions when unset — in either case rounded **up to a
    /// multiple of `micro_steps`**, so a chunk is a whole number of tokens.
    ///
    /// That rounding is what lets the read axis be a plain reshape: with
    /// `chunk_len = T·u` a chunk covers folded positions `[cL, (c+1)L)`, i.e.
    /// tokens `[cT, (c+1)T)`, and the surviving rows (`≡ u−1 mod u`) are a
    /// contiguous run of them. At `micro_steps = 1` it is the identity.
    pub fn chunk_len_or_optimal(&self, block: &Mamba3) -> usize {
        match self.chunk_len() {
            Some(chunk_len) => chunk_len.next_multiple_of(block.micro_steps.max(1)),
            None => Self::optimal_chunk_len(
                block.state_rank,
                block.per_head_dim(),
                block.mimo_rank,
                block.micro_steps,
            ),
        }
    }

    /// Chunks per iteration of the [`Self::SerialRecalculated`] backward's
    /// chunk-local pass — how far that pass batches the chunk axis.
    ///
    /// The recompute backward has no reason to *walk* that axis: every
    /// chunk-local gradient (the state-to-output term and the intra-chunk
    /// triangular one) needs the chunk's own slices plus its input state, which
    /// the recomputed K4 already produced batched. Only the state gradient
    /// itself is a scan, and it is a few ops per chunk.
    ///
    /// What batching the rest costs is memory: the score `[batch, group,
    /// nheads, read, fused]` is live `group`-wide, and this pass holds about six
    /// score-shaped tensors at its peak where the forward's K5 holds about four
    /// at **full** width. **Half** therefore keeps it under the peak the forward
    /// already reaches, while still cutting the walk to two iterations at any
    /// chunk count. Being a ratio of live tensors, the bound needs neither a
    /// dimension nor an absolute budget. The path's actual promise — not
    /// carrying intermediates across the forward/backward boundary — is
    /// untouched either way: by the time this runs, the forward's own are freed.
    pub fn backward_chunk_group(nchunks: usize) -> usize {
        nchunks.div_ceil(2).max(1)
    }

    /// The recommended default path for a given block: [`Self::SerialRecalculated`]
    /// with [`Self::optimal_chunk_len`] for the block's dimensions.
    pub fn default_optimal_from_block(block: &Mamba3) -> Self {
        let chunk_len = Self::optimal_chunk_len(
            block.state_rank,
            block.per_head_dim(),
            block.mimo_rank,
            block.micro_steps,
        );
        Self::SerialRecalculated(Some(chunk_len))
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;

impl Default for Mamba3SsdPath {
    fn default() -> Self {
        // Defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Self::SerialRecalculated(None)
    }
}
