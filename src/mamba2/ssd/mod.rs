use crate::mamba2::prelude::*;
use burn::prelude::*;

pub mod minimal;
pub mod serial;
pub mod serial_recalculated;

/// Ssd algorithm selection.
///
/// Each variant carries the chunk length Q for the SSD algorithm.  
/// Larger values increase the intra-chunk GEMM work and reduce the
/// inter-chunk scan length.  
/// Optimal value is approximately `√(state_rank · per_head_dim)`.
#[derive(Debug, Clone)]
pub enum SsdPath {
    /// Minimal SSD.
    ///
    /// This algorithm mostly uses batched matmuls. For the backward operation, this relies on autodiff.  
    /// See [`chunked_selective_scan`] for more info.
    ///
    /// For training, you may prefer using [SerialRecalculated](Self::SerialRecalculated) instead.
    ///
    /// Based on `/mamba_ssm/modules/ssd_minimal.py` from the `state-spaces/mamba` github reference.
    Minimal(Option<usize>),
    /// (Hybrid) Serial SSD.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// For the backward operation, this relies on autodiff.  
    /// For a custom backwards that saves memory, see [SerialRecalculated](Self::SerialRecalculated).
    ///
    /// Based on 5 kernels on `/mamba_ssm/ops/triton/` from the `state-spaces/mamba` github reference:
    /// - `ssd_chunk_state.py` (K1, K3).
    /// - `ssd_bmm.py` (K2).
    /// - `ssd_state_passing.py` (K4).
    /// - `ssd_chunk_scan.py` (K5).
    Serial(Option<usize>),
    /// (Hybrid) Serial SSD that triggers recalculations for the backward pass.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// Contains a custom backward operation that saves memory.  
    /// For an autodiff backwards, see [Serial](Self::Serial).
    ///
    /// Based on the combined kernel `/mamba_ssm/ops/triton/ssd_combined.py` from the `state-spaces/mamba`
    /// github reference.
    SerialRecalculated(Option<usize>),
}

impl SsdPath {
    /// Optimal chunk length is approximately `√(state_rank · per_head_dim)`.
    pub fn optimal_default(state_rank: usize, per_head_dim: usize) -> usize {
        (state_rank * per_head_dim)
            .isqrt()
            .next_multiple_of(32) // rule-of-thumb: common plane dimension.
            .min(512) // rule-of-thumb: ceiling at 512.
    }

    /// Optimal Minimal variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn core_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Minimal(Some(optim))
    }

    /// Optimal Minimal variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn core_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::core_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn chunked_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Serial(Some(optim))
    }

    /// Optimal Serial variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn chunked_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::chunked_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn chunked_recalculated_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::SerialRecalculated(Some(optim))
    }

    /// Optimal Serial Recalculated variant.
    ///
    /// See [optimal_default](Self::optimal_default) for more info.
    pub fn chunked_recalculated_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::chunked_recalculated_optimal(block.state_rank, block.per_head_dim())
    }

    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            SsdPath::Minimal(chunk_len) => chunk_len.clone(),
            SsdPath::Serial(chunk_len) => chunk_len.clone(),
            SsdPath::SerialRecalculated(chunk_len) => chunk_len.clone(),
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        match self {
            SsdPath::Minimal(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
            SsdPath::Serial(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
            SsdPath::SerialRecalculated(chunk_len) => {
                chunk_len.unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
            }
        }
    }
}

impl Default for SsdPath {
    fn default() -> SsdPath {
        // SsdPath defaults to the SerialRecalculated algorithm with the optimal chunk length.
        SsdPath::SerialRecalculated(None)
    }
}
