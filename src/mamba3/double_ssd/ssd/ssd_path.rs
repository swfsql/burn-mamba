//! # SSD input bundle for the Mamba-3 double-SSD pathway
//!
//! [`Mamba3DoubleSsdInput`] gathers the pre-processed tensors a single standard
//! SSD pass consumes (B/C already QK-normed, RoPE-applied, bias-added, and
//! GQA-expanded to per-head; `v` already scaled by the trapezoid coefficient γ
//! or β; `da = Δ·A` pre-combined).  [`Mamba3DoubleSsdInput::run`] dispatches to
//! the algorithm chosen by the shared [`Mamba3SsdPath`].
//!
//! [`Mamba3SsdPath`]: crate::mamba3::ssd_path::Mamba3SsdPath

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::prelude::*;
use burn::prelude::*;
use burn::backend::Backend;

/// MIMO-first SSD input.
///
/// All tensors are pre-processed: B/C are already QK-normed, RoPE-applied, bias-added, and
/// expanded to per-head (not per-group). V is already scaled by the (double-ssd) trapezoidal
/// coefficient (γ or β). The combined log-decay `da = Δ·A` is pre-computed. D skip is handled
/// by the caller.
pub struct Mamba3DoubleSsdInput {
    /// Value tensor, already scaled by (double-ssd) trapezoidal coefficient (γ or β).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    pub v_bnlmhp: Tensor<6>,

    /// Pre-combined log-decay `Δ·A` (negative).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<4>,

    /// Key/B tensor: QK-normed, RoPE-applied, bias-added, expanded to per-head, per-rank.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub b_bnlmhr: Tensor<6>,

    /// Query/C tensor: same processing as B.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub c_bnlmhr: Tensor<6>,

    /// Initial SSM hidden state.
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

impl Mamba3DoubleSsdInput {
    /// Run the [`NaN`/`Inf` guards](crate::utils::sanity) on every input tensor.
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

impl Mamba3DoubleSsdInput {
    /// Run the selected double-ssd algorithm on this MIMO-first input.
    ///
    /// Dispatches by [`Mamba3SsdPath`] variant to `double_ssd_minimal`,
    /// `double_ssd_serial`, or `double_ssd_serial_recalculated`.
    ///
    /// # Returns
    /// - `y_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn run(self, path: &Mamba3SsdPath) -> (Tensor<6>, Tensor<4>) {
        match path {
            Mamba3SsdPath::Minimal(_) => self.double_ssd_minimal(),
            Mamba3SsdPath::Serial(_) => self.double_ssd_serial(),
            Mamba3SsdPath::SerialRecalculated(_) => self.double_ssd_serial_recalculated(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests;
