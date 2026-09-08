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
//! by [`Mamba3SsdPath`](crate::mamba3::ssd_path::Mamba3SsdPath), shared with the double-ssd pathway.

use crate::mamba3::prelude::*;
use burn::prelude::*;

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

    /// Q/C tensor: same processing as `b_bnlmhr`, but on the chunk's **read**
    /// axis — one row per token rather than per folded position.
    ///
    /// Every micro-step of a token writes to the state; the token is read once,
    /// at its last one ([`micro_steps`](crate::mamba3::mamba3::Mamba3Config::micro_steps)
    /// is [`Self::read_stride`]), so `C` is only ever needed there and the `y`
    /// this bundle returns is already at token resolution. See
    /// [the read axis](crate::mamba3::product).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]`,
    ///   `chunk_tokens = chunk_len / read_stride`
    pub c_bntmhr: Tensor<6>,

    /// Pre-combined log-decay `Δ·A` (negative).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<4>,

    /// `γₜ = λₜ · Δₜ` — used as the per-token diagonal multiplier.
    ///
    /// On the **read** axis, like `c_bntmhr`: γ weights the same-step term of a
    /// row's output and nothing else (the state recurrence runs on
    /// `scale_bnlh`), so the folded positions in between never spend it.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_tokens, nheads]`
    pub gamma_bnth: Tensor<4>,

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

    /// The chunk's read stride: `micro_steps`, i.e. how many folded positions
    /// (writes) a token spans. `1` for stock Mamba-3, where the read axis *is*
    /// the chunk and every shape below reduces to `chunk_len`.
    ///
    /// `chunk_len` is a multiple of it, so a chunk holds a whole number of
    /// tokens and their rows are the contiguous run
    /// `[chunk · chunk_tokens, (chunk+1) · chunk_tokens)`.
    pub read_stride: usize,

    /// Whether the specialized `mimo_rank == 1` γ-correction may be used
    /// ([`Mamba3Config::siso_specialization`](crate::mamba3::mamba3::Mamba3Config::siso_specialization)
    /// — the chunkwise flag; the per-token sites have their own).
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
