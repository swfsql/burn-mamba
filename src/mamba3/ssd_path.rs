//! # Pathway-agnostic SSD algorithm selection (Mamba-3)
//!
//! [`Mamba3SsdPath`] selects the chunkwise SSD *algorithm* (Minimal / Serial /
//! SerialRecalculated) and the chunk length. It is independent of the
//! double-vs-single *pathway*, which the cache variant selects. It converts
//! into the per-pathway path types with `From`, and
//! [`Mamba3::forward`](crate::mamba3::mamba3::Mamba3::forward) sends it to the
//! pathway of the cache.

use crate::mamba3::prelude::*;

/// Algorithm selection for the Mamba-3 chunkwise SSD.
///
/// This selects the chunkwise SSD *algorithm*. The cache variant selects the
/// *pathway* (double- vs single-ssd, see
/// [`crate::mamba3::cache::Mamba3Caches`]). [`Mamba3::forward`] converts this
/// selection into the input bundle of that pathway
/// ([`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput`] or
/// [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput`]) and calls its
/// `run`.
///
/// Each variant carries an optional chunk length. A larger value gives more
/// intra-chunk GEMM work and a shorter inter-chunk scan. The optimal value is
/// approximately `√(state_rank · per_head_dim)`, divided by the widening that
/// `mimo_rank` and `micro_steps` already give the chunk (see
/// [`Self::optimal_chunk_len`]). `None` uses that optimal value.
///
/// The default is [`Self::SerialRecalculated`] with an unset chunk length.
#[derive(Debug, Clone)]
pub enum Mamba3SsdPath {
    /// Minimal/segsum SSD: mostly batched matmuls, with the autodiff backward.
    ///
    /// See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_minimal`]
    /// / [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_minimal`].
    /// For training, prefer [`Self::SerialRecalculated`]. This is the only path
    /// that supports a learnable initial state.
    Minimal(Option<usize>),

    /// (Hybrid) serial SSD: a serial loop over the chunks plus batched
    /// matmuls, with the autodiff backward.
    ///
    /// See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_serial`]
    /// / [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_serial`].
    /// For a memory-saving custom backward, see [`Self::SerialRecalculated`].
    Serial(Option<usize>),

    /// (Hybrid) serial SSD with a custom, memory-efficient backward. The
    /// backward recomputes the forward intermediates instead of storing them.
    ///
    /// See [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput::double_ssd_serial_recalculated`]
    /// / [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput::single_ssd_serial_recalculated`].
    /// For a plain autodiff backward, see [`Self::Serial`].
    SerialRecalculated(Option<usize>),
}

impl Mamba3SsdPath {
    /// Optimal chunk length, in **folded positions**: `√(state_rank ·
    /// per_head_dim)` divided by `mimo_rank`, on the 32 grid, clamped to
    /// `32 ..= 512`. `micro_steps` does not change this width:
    /// [`Self::chunk_tokens`] absorbs it.
    ///
    /// The square root is the SISO rule of thumb: it balances the intra-chunk
    /// GEMMs against the inter-chunk scan. Two dials of the block widen a chunk
    /// without a place in this formula, and they are **not** the same widening:
    ///
    /// - **`mimo_rank`** (`m`) widens *both* axes of a chunk. Every rank writes
    ///   and every rank reads, so the intra-chunk matmuls run on the fused
    ///   `chunk_len · m`. The rule of thumb is about that product, hence the
    ///   divisor. An unchanged `chunk_len` would cost `m²`, and this costs `m`.
    ///   The reference gives the same advice on its `chunk_size` argument: "64
    ///   for SISO, 64/mimo_rank for MIMO".
    /// - **`micro_steps`** (`u`) widens only the **write** axis. All `u`
    ///   micro-steps of a token write to the state, but the block *reads* the
    ///   token once, at its last micro-step. So the score is
    ///   `[batch, nchunks, heads, T·m, T·u·m]` for `T` = [`Self::chunk_tokens`],
    ///   with `batch · sequence · heads · T · u · m²` elements. To keep that flat
    ///   in `u`, keep `T · u` (the *folded* chunk) flat and use fewer tokens.
    ///   Memory is the binding constraint here, because no fused kernel hides it.
    ///
    /// So `u` subdivides the chunk and does not shorten it, and `nchunks` grows
    /// like `u`, not `u²`. The returned length is a multiple of `micro_steps`.
    /// So each chunk is a whole number of tokens, and their rows are a
    /// contiguous run. It is on the 32 grid exactly when `micro_steps` divides
    /// it.
    ///
    /// `info/mamba-3/architecture-deltas.md` §8.
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

    /// Tokens per chunk: `chunk_len / micro_steps`, the **read** axis of the
    /// chunk.
    ///
    /// `chunk_len` counts folded positions (one per micro-step). The SSD reads
    /// each token once, at its last micro-step. So this is the row count of the
    /// score and of the `y` that the kernels return. See
    /// [`Self::optimal_chunk_len`].
    ///
    /// # Panics
    /// If `micro_steps` does not divide `chunk_len`.
    /// [`Self::chunk_len_or_optimal`] makes sure that it does.
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
    /// for the dimensions of `block` when unset. In both cases the length is
    /// rounded **up to a multiple of `micro_steps`**, so a chunk is a whole
    /// number of tokens.
    ///
    /// With that rounding, the read axis is a plain reshape. With
    /// `chunk_len = T·u`, a chunk covers folded positions `[cL, (c+1)L)`, that
    /// is tokens `[cT, (c+1)T)`, and the surviving rows (`≡ u−1 mod u`) are a
    /// contiguous run of them. At `micro_steps = 1` the rounding does nothing.
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

    /// Chunks per iteration of the chunk-local pass of the
    /// [`Self::SerialRecalculated`] backward: how many chunks that pass
    /// batches.
    ///
    /// The recompute backward does not need to *walk* the chunk axis. Every
    /// chunk-local gradient (the state-to-output term and the intra-chunk
    /// triangular term) needs only the slices of its chunk and its input
    /// state, which the recomputed K4 already produced, batched. Only the
    /// state gradient is a scan, and it is a few ops per chunk.
    ///
    /// The cost of batching the rest is memory. The score
    /// `[batch, group, nheads, read, fused]` is live `group`-wide. At its peak
    /// this pass holds about six score-shaped tensors, where the forward K5
    /// holds about four at **full** width. So **half** stays under the peak
    /// that the forward already reaches, and the walk is two iterations at any
    /// chunk count. The bound is a ratio of live tensors, so it needs no
    /// dimension and no absolute budget. The promise of the path (no
    /// intermediates carried across the forward/backward boundary) holds in
    /// both cases: when this runs, the forward intermediates are already free.
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

impl Default for Mamba3SsdPath {
    fn default() -> Self {
        // Defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Self::SerialRecalculated(None)
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
