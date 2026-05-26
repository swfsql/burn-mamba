use crate::mamba2::prelude::*;
use burn::prelude::*;

/// Ssd algorithm selection.
///
/// Each variant carries the chunk length Q for the SSD algorithm.  
/// Larger values increase the intra-chunk GEMM work and reduce the
/// inter-chunk scan length.  
/// Optimal value is approximately `√(state_rank · per_head_dim)`.
#[derive(Debug, Clone)]
pub enum Mamba2SsdPath {
    /// Minimal SSD.
    ///
    /// This algorithm mostly uses batched matmuls. For the backward operation, this relies on autodiff.  
    /// See [`Mamba2SsdInput::ssd_minimal`] for more info.
    ///
    /// For training, you may prefer using [`Self::SerialRecalculated`] instead.
    ///
    /// Based on `/mamba_ssm/modules/ssd_minimal.py` from the `state-spaces/mamba` github reference.
    Minimal(Option<usize>),
    /// (Hybrid) Serial SSD.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// See [`Mamba2SsdInput::ssd_serial`] for more info.  
    /// For the backward operation, this relies on autodiff.  
    /// For a custom backwards that saves memory, see [`Self::SerialRecalculated`].
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
    /// See [`Mamba2SsdInput::ssd_serial_recalculated`] for more info.  
    /// Contains a custom backward operation that saves memory.  
    /// For an autodiff backwards, see [`Self::Serial`].
    ///
    /// Based on the combined kernel `/mamba_ssm/ops/triton/ssd_combined.py` from the `state-spaces/mamba`
    /// github reference.
    SerialRecalculated(Option<usize>),
}

/// SSD input.
///
/// All tensors are pre-processed: B/C are already GQA-expanded to per-head.
pub struct Mamba2SsdInput<B: Backend> {
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, per_head_dim]`
    pub x_bnlhp: Tensor<B, 5>,
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub dt_bnlh: Tensor<B, 4>,
    /// # Shape
    /// - `[nheads]`
    pub a_decay_h: Tensor<B, 1>,
    /// B tensor, expanded to per-head.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, state_rank]`
    pub b_bnlhr: Tensor<B, 5>,
    /// C tensor, expanded to per-head.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, state_rank]`
    pub c_bnlhr: Tensor<B, 5>,
    /// # Shape
    /// - `[nheads]`
    pub d_h: Tensor<B, 1>,
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<B, 4>,
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<B, 3>>,
}

impl<B: Backend> Mamba2SsdInput<B> {
    pub fn sanity(&self) {
        use crate::utils::sanity::sanity as san;
        san(&self.x_bnlhp);
        san(&self.dt_bnlh);
        san(&self.a_decay_h);
        san(&self.b_bnlhr);
        san(&self.c_bnlhr);
        san(&self.d_h);
        san(&self.initial_state_bhpr);
        if let Some(ref init_state_hpr) = self.init_state_hpr {
            san(init_state_hpr);
        }
    }
}

impl Mamba2SsdPath {
    /// Optimal chunk length is approximately `√(state_rank · per_head_dim)`.
    pub fn optimal_default(state_rank: usize, per_head_dim: usize) -> usize {
        (state_rank * per_head_dim)
            .isqrt()
            .next_multiple_of(32) // rule-of-thumb: common plane dimension.
            .min(512) // rule-of-thumb: ceiling at 512.
    }

    /// Optimal Minimal variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn core_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Minimal(Some(optim))
    }

    /// Optimal Minimal variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn core_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::core_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn chunked_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Serial(Some(optim))
    }

    /// Optimal Serial variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn chunked_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::chunked_optimal(block.state_rank, block.per_head_dim())
    }

    /// Optimal Serial variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn chunked_recalculated_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::SerialRecalculated(Some(optim))
    }

    /// Optimal Serial Recalculated variant.
    ///
    /// See [`Self::optimal_default`] for more info.
    pub fn chunked_recalculated_optimal_from_block<B: Backend>(block: &Mamba2<B>) -> Self {
        Self::chunked_recalculated_optimal(block.state_rank, block.per_head_dim())
    }

    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            Mamba2SsdPath::Minimal(chunk_len)
            | Mamba2SsdPath::Serial(chunk_len)
            | Mamba2SsdPath::SerialRecalculated(chunk_len) => *chunk_len,
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        self.chunk_len()
            .unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
    }

    /// Run the SSD algorithm on the given input.
    ///
    /// Dispatches to `ssd_minimal`, `ssd_serial`, or `ssd_serial_recalculated` based on the variant.
    ///
    /// # Returns
    /// - `y_bnlhp`: `[batch, nchunks, chunk_len, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn run<B: Backend + Mamba2BackendExt>(
        &self,
        input: Mamba2SsdInput<B>,
    ) -> (Tensor<B, 5>, Tensor<B, 4>) {
        match self {
            Mamba2SsdPath::Minimal(_) => input.ssd_minimal(),
            Mamba2SsdPath::Serial(_) => input.ssd_serial(),
            Mamba2SsdPath::SerialRecalculated(_) => input.ssd_serial_recalculated(),
        }
    }
}

impl Default for Mamba2SsdPath {
    fn default() -> Mamba2SsdPath {
        // SSD Path defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Mamba2SsdPath::SerialRecalculated(None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests;
