//! # Single-Pass SSD — Input Bundle
//!
//! Sibling to [`crate::mamba3::double_ssd::ssd::ssd_path`]. Where the double-ssd
//! pathway runs the *standard* SSD twice (γ-term and β-term), this module's
//! [`Mamba3SingleSsdInput`] runs **one** merged SSD pass that absorbs both
//! contributions by scaling `K` with `scaleₜ = γₜ + (1−λₜ₊₁) Δₜ₊₁`. The same-step
//! diagonal contribution differs (it must use `γₜ`, not `scaleₜ`) and is patched
//! via an explicit correction term inside each variant.
//
//! Reference kernels:
//! - `refs/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`
//! - `refs/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`
//!
//! The interface is MIMO-first (matches the other burn-mamba SSD inputs),
//! with `mimo_rank = 1` collapsing to the SISO case. The algorithm is selected
//! by [`Mamba3SsdPath`], shared with the double-ssd pathway.

use crate::mamba3::prelude::*;
use crate::mamba3::single_ssd::prelude::*;
use burn::prelude::*;
use burn::backend::Backend;

/// MIMO-first input bundle for the merged-form SSD.
///
/// All tensors are pre-processed by the caller (`Mamba3::forward_single_ssd`): B/C are
/// already QK-normed, RoPE-applied, bias-added, and expanded to per-head; V is
/// the raw, *unscaled* MIMO-expanded value. The combined log-decay `da = Δ·A`
/// is pre-computed. The two trapezoidal coefficients `gammaₜ` and `scaleₜ` are
/// supplied separately because the SSD itself does the K-scaling and γ-weighted
/// diagonal correction internally. D-skip and Z-gating are handled by the
/// caller.
pub struct Mamba3SingleSsdInput {
    /// Value tensor, MIMO-expanded but **not** trapezoidally scaled.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    pub v_bnlmhp: Tensor<6>,

    /// K/B tensor: QK-normed, RoPE-applied, bias-added, expanded to per-head.
    /// Not pre-scaled — the SSD multiplies by `scaleₜ` internally for the
    /// lower-triangular and state-recurrence paths, while the diagonal
    /// correction reuses the unscaled tensor.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub b_bnlmhr: Tensor<6>,

    /// Q/C tensor: same processing as `b_bnlmhr`.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub c_bnlmhr: Tensor<6>,

    /// Pre-combined log-decay `Δ·A` (negative).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<4>,

    /// `γₜ = λₜ · Δₜ` — used as the per-token diagonal multiplier.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub gamma_bnlh: Tensor<4>,

    /// `scaleₜ = γₜ + (1 − λₜ₊₁) · Δₜ₊₁` — K is multiplied by this for the
    /// lower-triangular and state recurrence paths. The shifted term is zero
    /// at the very last sequence position (no future token exists).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub scale_bnlh: Tensor<4>,

    /// Initial SSM hidden state (merged-form accumulator).
    ///
    /// When continuing from a prior call, this should already include the
    /// boundary β contribution `(1 − λ₀) · Δ₀ · Σₘ Kₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_xₘ)`
    /// (which the previous call could not yet add because it did not know
    /// `λ₀, Δ₀`).
    ///
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<4>,

    /// Optional learnable initial state (broadcast over batch).
    ///
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<3>>,
}

impl Mamba3SingleSsdInput {
    /// Run the [`NaN`/`Inf` guards](crate::utils::sanity) on every input tensor.
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

impl Mamba3SingleSsdInput {
    /// Run the selected merged-form (single-ssd) algorithm on this MIMO-first input.
    ///
    /// Dispatches by [`Mamba3SsdPath`] variant to `single_ssd_minimal`,
    /// `single_ssd_serial`, or `single_ssd_serial_recalculated`.
    ///
    /// # Returns
    /// - `y_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]` —
    ///   the merged-form accumulator at the last token (to be stored in the
    ///   cache for streaming).
    pub fn run(self, path: &Mamba3SsdPath) -> (Tensor<6>, Tensor<4>) {
        match path {
            Mamba3SsdPath::Minimal(_) => self.single_ssd_minimal(),
            Mamba3SsdPath::Serial(_) => self.single_ssd_serial(),
            Mamba3SsdPath::SerialRecalculated(_) => self.single_ssd_serial_recalculated(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — Minimal ≡ Serial (forward outputs + input gradients)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
