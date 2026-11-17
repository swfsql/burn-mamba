//! # The scan options of Mamba-2
//!
//! [`Mamba2ScanOptions`] holds the [SSD path](crate::ssd_path::SsdPath) and the
//! chunk length of the chunkwise SSD scan.
//! [`Mamba2::forward_packed`](crate::mamba2::mamba2::Mamba2::forward_packed),
//! which `forward` calls, computes the chunk length and builds the
//! [`Mamba2SsdInput`] at that length. Then it gives the SSD path to
//! [`Mamba2SsdInput::run`].

use crate::mamba2::prelude::*;
use burn::config::Config;

/// The scan options of a Mamba-2 block: the SSD path and the chunk length.
///
/// All scan options give the same values and gradients. The default is
/// [`SsdPath::SerialRecalculated`] with an unset chunk length.
#[derive(Config, Debug)]
pub struct Mamba2ScanOptions {
    /// The method that computes the chunkwise SSD scan.
    pub ssd_path: SsdPath,
    /// The chunk length. A larger value gives more intra-chunk GEMM work and a
    /// shorter inter-chunk scan.
    ///
    /// `None` uses the optimal value, [`Self::optimal_chunk_len`]. That value
    /// is approximately `√(state_rank · per_head_dim)`.
    pub chunk_len: Option<usize>,
}

impl Default for Mamba2ScanOptions {
    fn default() -> Self {
        Self::new(SsdPath::default())
    }
}

impl Mamba2ScanOptions {
    /// Optimal chunk length, approximately `√(state_rank · per_head_dim)`,
    /// rounded up to a multiple of 32 and capped at 512.
    pub fn optimal_chunk_len(state_rank: usize, per_head_dim: usize) -> usize {
        (state_rank * per_head_dim)
            .isqrt()
            .next_multiple_of(32) // rule-of-thumb: common plane dimension.
            .min(512) // rule-of-thumb: ceiling at 512.
    }

    /// The chunk length of these scan options, or [`Self::optimal_chunk_len`]
    /// when unset.
    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        self.chunk_len
            .unwrap_or_else(|| Self::optimal_chunk_len(state_rank, per_head_dim))
    }
}
