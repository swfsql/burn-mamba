//! # Pathway-agnostic SSD algorithm selection (Mamba-3)
//!
//! [`Mamba3SsdPath`] picks the chunkwise SSD *algorithm* (Minimal / Serial /
//! SerialRecalculated) and chunk length, independent of the double-vs-single
//! *pathway* (which the supplied cache variant selects).  It converts into the
//! per-pathway path types via `From` and is threaded by
//! [`Mamba3::forward`](crate::mamba3::mamba3::Mamba3::forward) into whichever
//! pathway the cache implies.

use crate::mamba3::prelude::*;
use burn::prelude::*;

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
/// value is approximately `√(state_rank · per_head_dim)` (see
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
    /// Optimal chunk length, approximately `√(state_rank · per_head_dim)`,
    /// rounded up to a multiple of 32 and capped at 512.
    pub fn optimal_chunk_len(state_rank: usize, per_head_dim: usize) -> usize {
        (state_rank * per_head_dim)
            .isqrt()
            .next_multiple_of(32) // rule-of-thumb: common plane dimension.
            .min(512) // rule-of-thumb: ceiling at 512.
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
    /// when unset.
    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        self.chunk_len()
            .unwrap_or_else(|| Self::optimal_chunk_len(state_rank, per_head_dim))
    }

    /// The recommended default path for a given block: [`Self::SerialRecalculated`]
    /// with [`Self::optimal_chunk_len`] for the block's dimensions.
    pub fn default_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
        let chunk_len = Self::optimal_chunk_len(block.state_rank, block.per_head_dim());
        Self::SerialRecalculated(Some(chunk_len))
    }
}

impl Default for Mamba3SsdPath {
    fn default() -> Self {
        // Defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Self::SerialRecalculated(None)
    }
}
