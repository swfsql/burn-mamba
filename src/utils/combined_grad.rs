//! Helpers for the "two-output, one autodiff node" pattern used by both
//! [`Mamba2BackendExt::ssd_serial_recalculated`](crate::mamba2::ssd::Mamba2BackendExt::ssd_serial_recalculated)
//! and
//! [`Mamba3BackendExt::ssd_serial_recalculated`](crate::mamba3::ssd::Mamba3BackendExt::ssd_serial_recalculated).
//!
//! Burn's `prep.finish` accepts only a single tracked tensor, so the two
//! outputs (`y` and `final_state`) are flattened and concatenated into a
//! single 1-D tracked tensor; the caller then `narrow`s it back into two
//! reshaped views. Burn's autodiff accumulates the upstream gradients of those
//! views into the combined gradient vector which the custom backward consumes.

use burn::prelude::*;
use burn::tensor::Shape;

/// Flatten the two outputs (`y` and `final_state`) and concatenate them along a
/// fresh axis-0 into a single 1-D tensor. Returns the combined tensor and the
/// per-output flat lengths needed to split it later.
pub fn flatten_pair<B: Backend, const DA: usize, const DB: usize>(
    y: Tensor<B, DA>,
    final_state: Tensor<B, DB>,
) -> (Tensor<B, 1>, usize, usize) {
    let flat_y_len = Shape::from(y.shape()).num_elements();
    let flat_s_len = Shape::from(final_state.shape()).num_elements();
    let combined = Tensor::cat(
        vec![y.reshape([flat_y_len]), final_state.reshape([flat_s_len])],
        0,
    );
    (combined, flat_y_len, flat_s_len)
}

/// Inverse of [`flatten_pair`]: split a 1-D combined tensor back into the two
/// outputs at their original ranks/shapes.
pub fn unflatten_pair<B: Backend, const DA: usize, const DB: usize>(
    combined: Tensor<B, 1>,
    flat_y_len: usize,
    flat_s_len: usize,
    shape_y: [usize; DA],
    shape_s: [usize; DB],
) -> (Tensor<B, DA>, Tensor<B, DB>) {
    let y = combined.clone().narrow(0, 0, flat_y_len).reshape(shape_y);
    let s = combined.narrow(0, flat_y_len, flat_s_len).reshape(shape_s);
    (y, s)
}
