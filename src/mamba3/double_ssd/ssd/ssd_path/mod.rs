//! # SSD input bundle for the Mamba-3 double-SSD pathway
//!
//! [`Mamba3DoubleSsdInput`] bundles the pre-processed tensors that one standard
//! SSD pass uses. [`Mamba3DoubleSsdInput::run`] calls the algorithm that the
//! shared [`Mamba3SsdPath`] selects.
//!
//! [`Mamba3SsdPath`]: crate::mamba3::ssd_path::Mamba3SsdPath

use crate::mamba3::prelude::*;
use burn::prelude::*;

/// MIMO-first SSD input.
///
/// All tensors are pre-processed:
///
/// - B/C are QK-normed, bias-added, rotated, and expanded to per-head (not
///   per-group),
/// - V is scaled by the trapezoidal coefficient (γ or β) of its pass,
/// - the log-decay `da` (`Δ·A`, or the Kalman decay) is pre-computed.
///
/// The caller adds the D skip.
pub struct Mamba3DoubleSsdInput {
    /// Value tensor, already scaled by the trapezoidal coefficient (γ or β).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    pub v_bnlmhp: Tensor<6>,

    /// Pre-computed log-decay (negative): `Δ·A`, or the Kalman decay.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<4>,

    /// Key/B tensor: QK-normed, bias-added, rotated, expanded to per-head,
    /// per-rank.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    pub b_bnlmhr: Tensor<6>,

    /// Query/C tensor: the same processing as B, but on the **read** axis of
    /// the chunk. That is one row per token, not per folded position, because
    /// a token is read once, at its last micro-step. See
    /// [`Mamba3SingleSsdInput::c_bntmhr`](crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput)
    /// and [the read axis](crate::mamba3::product).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]`,
    ///   `chunk_tokens = chunk_len / read_stride`
    pub c_bntmhr: Tensor<6>,

    /// The read stride of the chunk: `micro_steps`, the folded positions per
    /// read row. `1` for stock Mamba-3, where the two axes are equal.
    pub read_stride: usize,

    /// Initial SSM hidden state.
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
}

impl Mamba3DoubleSsdInput {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every input tensor.
    pub fn sanity(&self) {
        use burn_stack::modules::sanity as san;
        san(&self.v_bnlmhp);
        san(&self.da_bnlh);
        san(&self.b_bnlmhr);
        san(&self.c_bntmhr);
        san(&self.initial_state_bhpr);
        if let Some(ref init_state_hpr) = self.init_state_hpr {
            san(init_state_hpr);
        }
    }
}

impl Mamba3DoubleSsdInput {
    /// Run the selected double-ssd algorithm on this MIMO-first input:
    /// `double_ssd_minimal`, `double_ssd_serial`, or
    /// `double_ssd_serial_recalculated`, by [`Mamba3SsdPath`] variant.
    ///
    /// # Returns
    /// - `y_bntmhp`: `[batch, nchunks, chunk_tokens, mimo_rank, nheads, per_head_dim]`,
    ///   at token resolution, the resolution of the readout (see [`Self::c_bntmhr`])
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

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
