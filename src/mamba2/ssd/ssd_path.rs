//! # The input bundle of the SSD paths (Mamba-2)
//!
//! [`Mamba2SsdInput`] bundles the pre-processed tensors that the scan uses
//! (B/C already GQA-expanded to per-head). [`Mamba2SsdInput::run`] runs one of
//! the three exact SSD reformulations ([`minimal`] / [`serial`] /
//! [`serial_recalculated`]): the one that its
//! [`SsdPath`](crate::ssd_path::SsdPath) names.

use crate::mamba2::prelude::*;
use burn::prelude::*;

/// SSD input.
///
/// All tensors are pre-processed: B/C are already GQA-expanded to per-head.
/// `a_decay_h` is `A < 0`, not yet multiplied by `Δ`.
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
    /// The incoming state (from the cache).
    ///
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<4>,
    /// The learnable initial state, added to `initial_state_bhpr`. Only
    /// [`SsdPath::Minimal`] supports it. The two serial paths panic when
    /// it is `Some`.
    ///
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<3>>,
    /// The chunks that start a new segment of a packed row, and the origin
    /// state. The start state of each of these chunks is the origin state
    /// (see [`crate::packing`]). The call itself starts from
    /// `initial_state_bhpr`.
    /// `None` ⇒ no reset. Only the two serial paths take it.
    /// [`SsdPath::Minimal`] panics when it is `Some`.
    pub resets: Option<crate::packing::SsdResets>,
}

impl Mamba2SsdInput {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every input tensor.
    pub fn sanity(&self) {
        use burn_stack::modules::sanity as san;
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

impl Mamba2SsdInput {
    /// Run the selected SSD path on this input: `ssd_minimal`,
    /// `ssd_serial`, or `ssd_serial_recalculated`, by [`SsdPath`]
    /// variant. The chunk length is already in the tensor shapes.
    ///
    /// # Returns
    /// - `y_bnlhp`: `[batch, nchunks, chunk_len, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn run(self, ssd_path: SsdPath) -> (Tensor<5>, Tensor<4>) {
        match ssd_path {
            SsdPath::Minimal => {
                assert!(
                    self.resets.is_none(),
                    "Minimal does not take resets: use Serial or SerialRecalculated"
                );
                self.ssd_minimal()
            }
            SsdPath::Serial => self.ssd_serial(),
            SsdPath::SerialRecalculated => self.ssd_serial_recalculated(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
