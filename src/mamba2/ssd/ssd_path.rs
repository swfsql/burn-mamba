//! # SSD algorithm selection and input bundle (Mamba-2)
//!
//! [`Mamba2SsdPath`] chooses which of the three exact SSD reformulations
//! ([`super::minimal`] / [`super::serial`] / [`super::serial_recalculated`])
//! runs, and at what chunk length.  [`Mamba2SsdInput`] bundles the pre-processed
//! tensors the scan consumes (B/C already GQA-expanded to per-head); its
//! [`Mamba2SsdInput::run`] dispatches to the path-selected algorithm.

use crate::mamba2::prelude::*;
use burn::backend::Backend;
use burn::prelude::*;

/// Algorithm selection for the Mamba-2 chunkwise SSD.
///
/// Each variant carries an optional chunk length. Larger values increase the
/// intra-chunk GEMM work and reduce the inter-chunk scan length; the optimal
/// value is approximately `√(state_rank · per_head_dim)` (see
/// [`Self::optimal_chunk_len`]). `None` falls back to that optimal value.
#[derive(Debug, Clone)]
pub enum Mamba2SsdPath {
    /// Minimal SSD: mostly batched matmuls; backward via autodiff.
    ///
    /// See [`Mamba2SsdInput::ssd_minimal`]. For training, prefer
    /// [`Self::SerialRecalculated`].
    ///
    /// Based on `/mamba_ssm/modules/ssd_minimal.py` from the `state-spaces/mamba`
    /// github reference.
    Minimal(Option<usize>),

    /// (Hybrid) serial SSD: a serial loop over the chunks plus batched matmuls;
    /// backward via autodiff.
    ///
    /// See [`Mamba2SsdInput::ssd_serial`]. For a memory-saving custom backward,
    /// see [`Self::SerialRecalculated`].
    ///
    /// Based on 5 kernels under `/mamba_ssm/ops/triton/` from the
    /// `state-spaces/mamba` github reference:
    /// - `ssd_chunk_state.py` (K1, K3).
    /// - `ssd_bmm.py` (K2).
    /// - `ssd_state_passing.py` (K4).
    /// - `ssd_chunk_scan.py` (K5).
    Serial(Option<usize>),

    /// (Hybrid) serial SSD with a custom, memory-efficient backward that
    /// recomputes the forward intermediates instead of storing them.
    ///
    /// See [`Mamba2SsdInput::ssd_serial_recalculated`]. For a plain autodiff
    /// backward, see [`Self::Serial`].
    ///
    /// Based on the combined kernel `/mamba_ssm/ops/triton/ssd_combined.py` from
    /// the `state-spaces/mamba` github reference.
    SerialRecalculated(Option<usize>),
}

/// SSD input.
///
/// All tensors are pre-processed: B/C are already GQA-expanded to per-head.
pub struct Mamba2SsdInput {
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, per_head_dim]`
    pub x_bnlhp: Tensor<5>,
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub dt_bnlh: Tensor<4>,
    /// # Shape
    /// - `[nheads]`
    pub a_decay_h: Tensor<1>,
    /// B tensor, expanded to per-head.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, state_rank]`
    pub b_bnlhr: Tensor<5>,
    /// C tensor, expanded to per-head.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads, state_rank]`
    pub c_bnlhr: Tensor<5>,
    /// # Shape
    /// - `[nheads]`
    pub d_h: Tensor<1>,
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<4>,
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<3>>,
}

impl Mamba2SsdInput {
    /// Run the [`NaN`/`Inf` guards](crate::utils::sanity) on every input tensor.
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
    pub fn default_optimal_from_block<B: Backend>(block: &Mamba2) -> Self {
        let chunk_len = Self::optimal_chunk_len(block.state_rank, block.per_head_dim());
        Self::SerialRecalculated(Some(chunk_len))
    }
}

impl Mamba2SsdInput {
    /// Run the selected SSD algorithm on this input.
    ///
    /// Dispatches by [`Mamba2SsdPath`] variant to `ssd_minimal`, `ssd_serial`,
    /// or `ssd_serial_recalculated`.
    ///
    /// # Returns
    /// - `y_bnlhp`: `[batch, nchunks, chunk_len, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn run(self, path: &Mamba2SsdPath) -> (Tensor<5>, Tensor<4>) {
        match path {
            Mamba2SsdPath::Minimal(_) => self.ssd_minimal(),
            Mamba2SsdPath::Serial(_) => self.ssd_serial(),
            Mamba2SsdPath::SerialRecalculated(_) => self.ssd_serial_recalculated(),
        }
    }
}

impl Default for Mamba2SsdPath {
    fn default() -> Mamba2SsdPath {
        // Defaults to the SerialRecalculated algorithm with the optimal chunk length.
        Mamba2SsdPath::SerialRecalculated(None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
