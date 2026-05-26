use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::prelude::*;
use burn::prelude::*;

/// Ssd algorithm selection.
///
/// Each variant carries the chunk length for the SSD algorithm.
/// Larger values increase the intra-chunk GEMM work and reduce the
/// inter-chunk scan length.
/// Optimal value is approximately `√(state_rank · per_head_dim)`.
#[derive(Debug, Clone)]
pub enum Mamba3DoubleSsdPath {
    // TODO: add specific mamba-3 python files references. Currently the references are for mamba-2.
    //
    /// Minimal SSD.
    ///
    /// This algorithm mostly uses batched matmuls. For the backward operation, this relies on autodiff.
    /// See [`Mamba3DoubleSsdInput::double_ssd_minimal`] for more info.
    ///
    /// For training, you may prefer using [`Self::SerialRecalculated`] instead.
    ///
    /// Based on `/mamba_ssm/modules/ssd_minimal.py` from the `state-spaces/mamba` github reference,
    /// adapted to Mamba-3.
    Minimal(Option<usize>),
    // TODO: add specific mamba-3 python files references. Currently the references are for mamba-2.
    //
    /// (Hybrid) Serial SSD.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// See [`Mamba3DoubleSsdInput::double_ssd_serial`] for more info.  
    /// For the backward operation, this relies on autodiff.
    /// For a custom backwards that saves memory, see [`Self::SerialRecalculated`].
    ///
    /// Based on 5 kernels on `/mamba_ssm/ops/triton/`, adapted to mamba-3,
    /// from the `state-spaces/mamba` github reference:
    /// - `ssd_chunk_state.py` (K1, K3).
    /// - `ssd_bmm.py` (K2).
    /// - `ssd_state_passing.py` (K4).
    /// - `ssd_chunk_scan.py` (K5).
    Serial(Option<usize>),
    // TODO: add specific mamba-3 python files references. Currently the references are for mamba-2.
    //
    /// (Hybrid) Serial SSD that triggers recalculations for the backward pass.
    ///
    /// This algorithm uses a serial loop over the nchunks, besides batched matmuls.
    /// See [`Mamba3DoubleSsdInput::double_ssd_serial_recalculated`] for more info.  
    /// Contains a custom backward operation that saves memory.
    /// For an autodiff backwards, see [`Self::Serial`].
    ///
    /// Based on the combined kernel `/mamba_ssm/ops/triton/ssd_combined.py`, adapted to Mamba-3,
    /// from the `state-spaces/mamba` github reference.
    SerialRecalculated(Option<usize>),
}

/// MIMO-first SSD input.
///
/// All tensors are pre-processed: B/C are already QK-normed, RoPE-applied, bias-added, and
/// expanded to per-head (not per-group). V is already scaled by the (double-ssd) trapezoidal
/// coefficient (γ or β). The combined log-decay `da = Δ·A` is pre-computed. D skip is handled
/// by the caller.
pub struct Mamba3DoubleSsdInput<B: Backend> {
    /// Value tensor, already scaled by (double-ssd) trapezoidal coefficient (γ or β).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    pub v_bnlmhp: Tensor<B, 6>,

    /// Pre-combined log-decay `Δ·A` (negative).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<B, 4>,

    /// Key/B tensor: QK-normed, RoPE-applied, bias-added, expanded to per-head, per-rank.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub b_bnlmhr: Tensor<B, 6>,

    /// Query/C tensor: same processing as B.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub c_bnlmhr: Tensor<B, 6>,

    /// Initial SSM hidden state.
    ///
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<B, 4>,

    /// Optional learnable initial state (broadcast over batch).
    ///
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<B, 3>>,
}

impl<B: Backend> Mamba3DoubleSsdInput<B> {
    pub fn sanity(&self) {
        use crate::utils::sanity::sanity as san;
        san(&self.v_bnlmhp);
        san(&self.da_bnlh);
        san(&self.b_bnlmhr);
        san(&self.c_bnlmhr);
        san(&self.initial_state_bhpr);
        if let Some(ref init_state_hpr) = self.init_state_hpr {
            san(init_state_hpr);
        }
    }
}

impl Mamba3DoubleSsdPath {
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
    pub fn core_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
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
    pub fn chunked_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
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
    pub fn chunked_recalculated_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
        Self::chunked_recalculated_optimal(block.state_rank, block.per_head_dim())
    }

    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            Mamba3DoubleSsdPath::Minimal(chunk_len)
            | Mamba3DoubleSsdPath::Serial(chunk_len)
            | Mamba3DoubleSsdPath::SerialRecalculated(chunk_len) => *chunk_len,
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        self.chunk_len()
            .unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
    }

    /// Run the SSD algorithm on the given MIMO-first input.
    ///
    /// Dispatches to `ssd_minimal`, `ssd_serial`, or `ssd_serial_recalculated` based on the variant.
    ///
    /// # Returns
    /// - `y_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn run<B: Backend + Mamba3DoubleSsdBackendExt>(
        &self,
        input: Mamba3DoubleSsdInput<B>,
    ) -> (Tensor<B, 6>, Tensor<B, 4>) {
        match self {
            Mamba3DoubleSsdPath::Minimal(_) => input.double_ssd_minimal(),
            Mamba3DoubleSsdPath::Serial(_) => input.double_ssd_serial(),
            Mamba3DoubleSsdPath::SerialRecalculated(_) => input.double_ssd_serial_recalculated(),
        }
    }
}

impl Default for Mamba3DoubleSsdPath {
    fn default() -> Mamba3DoubleSsdPath {
        // The SSD Path defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Mamba3DoubleSsdPath::SerialRecalculated(None)
    }
}

impl From<Mamba3SsdPath> for Mamba3DoubleSsdPath {
    fn from(path: Mamba3SsdPath) -> Self {
        match path {
            Mamba3SsdPath::Minimal(chunk_len) => Mamba3DoubleSsdPath::Minimal(chunk_len),
            Mamba3SsdPath::Serial(chunk_len) => Mamba3DoubleSsdPath::Serial(chunk_len),
            Mamba3SsdPath::SerialRecalculated(chunk_len) => {
                Mamba3DoubleSsdPath::SerialRecalculated(chunk_len)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests;
