//! # Single-Pass SSD — Input Bundle
//!
//! Sibling of [`crate::mamba3::double_ssd::ssd::ssd_path`]. The double-ssd
//! pathway runs the *standard* SSD once per term (γ-term and β-terms). The
//! [`Mamba3SingleSsdInput`] of this module runs **one** merged SSD pass that
//! absorbs all terms: it scales `K` by `scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)`. The
//! same-step diagonal contribution is different (it must use `γₜ`, not
//! `scaleₜ`), and an explicit correction term inside each variant patches it.
//!
//! Reference kernels, in `state-spaces/mamba`:
//! - `mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`
//! - `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`
//!
//! The interface is MIMO-first (as the other burn-mamba SSD inputs), and
//! `mimo_rank = 1` is the SISO case.
//! [`Mamba3SsdPath`](crate::mamba3::ssd_path::Mamba3SsdPath), shared with the
//! double-ssd pathway, selects the algorithm.

use crate::mamba3::prelude::*;
use burn::prelude::*;

/// MIMO-first input bundle for the merged-form SSD.
///
/// The caller (`Mamba3::forward_single_ssd`) pre-processes all tensors:
///
/// - B/C are QK-normed, bias-added, rotated, and expanded to per-head,
/// - V is the raw, *unscaled* MIMO-expanded value,
/// - the log-decay `da` (`Δ·A`, or the Kalman decay) is pre-computed,
/// - the two trapezoidal coefficients `gammaₜ` and `scaleₜ` are separate
///   inputs, because the SSD does the K-scaling and the γ-weighted diagonal
///   correction itself.
///
/// The caller adds the D skip and the Z gate.
pub struct Mamba3SingleSsdInput {
    /// Value tensor, MIMO-expanded but **not** trapezoidally scaled.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    pub v_bnlmhp: Tensor<6>,

    /// K/B tensor: QK-normed, bias-added, rotated, expanded to per-head. Not
    /// pre-scaled: the SSD multiplies it by `scaleₜ` for the lower-triangular
    /// and state-recurrence paths, and the diagonal correction uses the
    /// unscaled tensor.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub b_bnlmhr: Tensor<6>,

    /// Q/C tensor: the same processing as `b_bnlmhr`, but on the **read** axis
    /// of the chunk: one row per token, not per folded position.
    ///
    /// Every micro-step of a token writes to the state. The token is read
    /// once, at its last micro-step
    /// ([`micro_steps`](crate::mamba3::mamba3::Mamba3Config::micro_steps) is
    /// [`Self::read_stride`]). So `C` is necessary only there, and the `y` that
    /// this bundle returns is already at token resolution. See
    /// [the read axis](crate::mamba3::product).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]`,
    ///   `chunk_tokens = chunk_len / read_stride`
    pub c_bntmhr: Tensor<6>,

    /// Pre-computed log-decay (negative): `Δ·A`, or the Kalman decay.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<4>,

    /// `γₜ = λₜ · Δₜ`: the per-token diagonal multiplier.
    ///
    /// On the **read** axis, like `c_bntmhr`: γ weights only the same-step term
    /// of the output of a row (the state recurrence runs on `scale_bnlh`). So
    /// the folded positions between the reads never use it.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_tokens, nheads]`
    pub gamma_bnth: Tensor<4>,

    /// `scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)`: the factor on K for the
    /// lower-triangular and state-recurrence paths. A shifted term is zero at
    /// the last positions that its lag reaches past (the next call pays them).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub scale_bnlh: Tensor<4>,

    /// Initial SSM hidden state (merged-form accumulator).
    ///
    /// When the call continues a prior call, this must already hold the
    /// boundary β seed `Σⱼ νⱼ · Σₘ K_prev[j, m] ⊗ (x_prev[j] ⊙ mimo_xₘ)`. The
    /// previous call could not add it, because it did not know the `ν` of the
    /// first `lag` positions of this call.
    ///
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<4>,

    /// Optional learnable initial state (broadcast over batch). Only the
    /// `Minimal` path supports it: the serial paths panic when it is `Some`.
    ///
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<3>>,

    /// The read stride of the chunk: `micro_steps`, the number of folded
    /// positions (writes) in a token. `1` for stock Mamba-3, where the read
    /// axis *is* the chunk and every shape below reduces to `chunk_len`.
    ///
    /// `chunk_len` is a multiple of it. So a chunk holds a whole number of
    /// tokens, and their rows are the contiguous run
    /// `[chunk · chunk_tokens, (chunk+1) · chunk_tokens)`.
    pub read_stride: usize,

    /// Whether the specialized `mimo_rank == 1` γ-correction can be used
    /// ([`Mamba3Config::siso_specialization`](crate::mamba3::mamba3::Mamba3Config::siso_specialization),
    /// the chunkwise flag; the per-token sites have their own).
    ///
    /// Performance-only: both branches of
    /// [`y_diag_correction`](crate::mamba3::single_ssd::ssd::diag::y_diag_correction)
    /// compute the same values and gradients. No effect at `mimo_rank > 1`.
    pub siso_specialization: bool,
}

impl Mamba3SingleSsdInput {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every input tensor.
    pub fn sanity(&self) {
        use burn_stack::modules::sanity as san;
        san(&self.v_bnlmhp);
        san(&self.b_bnlmhr);
        san(&self.c_bntmhr);
        san(&self.da_bnlh);
        san(&self.gamma_bnth);
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
    /// - `y_bntmhp`: `[batch, nchunks, chunk_tokens, mimo_rank, nheads, per_head_dim]`
    ///   — token resolution, the readout's own (see [`Self::c_bntmhr`])
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
