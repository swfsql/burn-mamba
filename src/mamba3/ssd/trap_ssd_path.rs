//! # Trapezoidal-Merged (Single-Pass) SSD — Path Dispatcher
//!
//! Sibling to [`crate::mamba3::ssd::ssd_path::Mamba3SsdPath`]. Where the existing
//! [`Mamba3SsdPath`] runs the *standard* SSD twice (γ-term and β-term), this
//! module's [`Mamba3TrapSsdPath`] runs **one** merged SSD pass that absorbs both
//! contributions by scaling `K` with `scaleₜ = γₜ + (1−λₜ₊₁) Δₜ₊₁`. The same-step
//! diagonal contribution differs (it must use `γₜ`, not `scaleₜ`) and is patched
//! via an explicit correction term inside each variant.
//!
//! Reference kernels:
//! - `refs/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`
//! - `refs/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`
//!
//! The interface is MIMO-first (matches the existing burn-mamba SSD inputs),
//! with `R = 1` collapsing to the SISO case.

use crate::mamba3::prelude::*;
use burn::prelude::*;

/// Algorithm selection for the trapezoidal-merged (single-pass) SSD.
///
/// Mirrors [`Mamba3SsdPath`] but each variant computes the trapezoidal
/// recurrence with **one** chunkwise pass.
#[derive(Debug, Clone)]
pub enum Mamba3TrapSsdPath {
    /// Minimal/segsum variant.
    ///
    /// Mostly batched matmuls; the backward pass relies on autodiff. The
    /// algorithm is the merged-form analogue of [`Mamba3SsdPath::Minimal`]:
    /// strict lower-triangular intra-chunk (excludes same-step block),
    /// a separate γ-scaled diagonal correction, and a state recurrence using
    /// the `scaleₜ`-scaled K. See [`Mamba3TrapSsdInput::ssd_trap_minimal`].
    Minimal(Option<usize>),
}

/// MIMO-first input bundle for the merged-form SSD.
///
/// All tensors are pre-processed by the caller (`Mamba3::forward2`): B/C are
/// already QK-normed, RoPE-applied, bias-added, and expanded to per-head; V is
/// the raw, *unscaled* MIMO-expanded value. The combined log-decay `da = Δ·A`
/// is pre-computed. The two trapezoidal coefficients `gammaₜ` and `scaleₜ` are
/// supplied separately because the SSD itself does the K-scaling and γ-weighted
/// diagonal correction internally. D-skip and Z-gating are handled by the
/// caller.
pub struct Mamba3TrapSsdInput<B: Backend> {
    /// Value tensor, MIMO-expanded but **not** trapezoidally scaled.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    pub v_bnlmhp: Tensor<B, 6>,

    /// K/B tensor: QK-normed, RoPE-applied, bias-added, expanded to per-head.
    /// Not pre-scaled — the SSD multiplies by `scaleₜ` internally for the
    /// lower-triangular and state-recurrence paths, while the diagonal
    /// correction reuses the unscaled tensor.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub b_bnlmhr: Tensor<B, 6>,

    /// Q/C tensor: same processing as `b_bnlmhr`.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub c_bnlmhr: Tensor<B, 6>,

    /// Pre-combined log-decay `Δ·A` (negative).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<B, 4>,

    /// `γₜ = λₜ · Δₜ` — used as the per-token diagonal multiplier.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub gamma_bnlh: Tensor<B, 4>,

    /// `scaleₜ = γₜ + (1 − λₜ₊₁) · Δₜ₊₁` — K is multiplied by this for the
    /// lower-triangular and state recurrence paths. The shifted term is zero
    /// at the very last sequence position (no future token exists).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub scale_bnlh: Tensor<B, 4>,

    /// Initial SSM hidden state (merged-form accumulator).
    ///
    /// When continuing from a prior call, this should already include the
    /// boundary β contribution `(1 − λ₀) · Δ₀ · Σₘ Kₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_xₘ)`
    /// (which the previous call could not yet add because it did not know
    /// `λ₀, Δ₀`).
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

impl<B: Backend> Mamba3TrapSsdInput<B> {
    pub fn sanity(&self) {
        use crate::utils::sanity::sanity as san;
        san(&self.v_bnlmhp);
        san(&self.b_bnlmhr);
        san(&self.c_bnlmhr);
        san(&self.da_bnlh);
        san(&self.gamma_bnlh);
        san(&self.scale_bnlh);
        san(&self.initial_state_bhpr);
        if let Some(ref init_state_hpr) = self.init_state_hpr {
            san(init_state_hpr);
        }
    }
}

impl Mamba3TrapSsdPath {
    /// Optimal chunk length — same heuristic as [`Mamba3SsdPath::optimal_default`].
    pub fn optimal_default(state_rank: usize, per_head_dim: usize) -> usize {
        Mamba3SsdPath::optimal_default(state_rank, per_head_dim)
    }

    /// Optimal Minimal variant.
    pub fn core_optimal(state_rank: usize, per_head_dim: usize) -> Self {
        let optim = Self::optimal_default(state_rank, per_head_dim);
        Self::Minimal(Some(optim))
    }

    /// Optimal Minimal variant from a block.
    pub fn core_optimal_from_block<B: Backend>(block: &Mamba3<B>) -> Self {
        Self::core_optimal(block.state_rank, block.per_head_dim())
    }

    pub fn chunk_len(&self) -> Option<usize> {
        match self {
            Mamba3TrapSsdPath::Minimal(chunk_len) => *chunk_len,
        }
    }

    pub fn chunk_len_or_optimal(&self, state_rank: usize, per_head_dim: usize) -> usize {
        self.chunk_len()
            .unwrap_or_else(|| Self::optimal_default(state_rank, per_head_dim))
    }

    /// Run the merged-form SSD on the given MIMO-first input.
    ///
    /// # Returns
    /// - `y_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]` —
    ///   the merged-form accumulator at the last token (to be stored in the
    ///   cache for streaming).
    pub fn run<B: Backend>(&self, input: Mamba3TrapSsdInput<B>) -> (Tensor<B, 6>, Tensor<B, 4>) {
        match self {
            Mamba3TrapSsdPath::Minimal(_) => input.ssd_trap_minimal(),
        }
    }
}

impl Default for Mamba3TrapSsdPath {
    fn default() -> Mamba3TrapSsdPath {
        Mamba3TrapSsdPath::Minimal(None)
    }
}
