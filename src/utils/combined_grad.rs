//! Helpers for the "two-output, one autodiff node" pattern used by both
//! [`crate::mamba2::ssd::Mamba2BackendExt::ssd_serial_recalculated`]
//! and
//! [`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdBackendExt::double_ssd_serial_recalculated`]/[`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdBackendExt::single_ssd_serial_recalculated`].
//!
//! Burn's `prep.finish` accepts only a single tracked tensor, so the two
//! outputs (`y` and `final_state`) are flattened and concatenated into a
//! single 1-D tracked tensor; the caller then `narrow`s it back into two
//! reshaped views. Burn's autodiff accumulates the upstream gradients of those
//! views into the combined gradient vector which the custom backward consumes.

use burn::backend::Autodiff;
use burn::backend::Backend;
use burn::backend::BackendTypes;
use burn::backend::TensorMetadata;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::ops::FloatTensorOps;
use burn::backend::tensor::FloatTensor;
use burn::prelude::*;

/// Flatten the two outputs (`y` and `final_state`) and concatenate them along a
/// fresh axis-0 into a single 1-D tensor. Returns the combined tensor and the
/// per-output flat lengths needed to split it later.
pub fn flatten_pair<B: Backend>(
    y: <B as BackendTypes>::FloatTensorPrimitive,
    final_state: <B as BackendTypes>::FloatTensorPrimitive,
) -> (<B as BackendTypes>::FloatTensorPrimitive, usize, usize) {
    let flat_y_len = y.shape().num_elements();
    let flat_s_len = final_state.shape().num_elements();
    let flat_y = B::float_reshape(y, Shape::new([flat_y_len]));
    let flat_s = B::float_reshape(final_state, Shape::new([flat_s_len]));
    let combined = B::float_cat(vec![flat_y, flat_s], 0);
    (combined, flat_y_len, flat_s_len)
}

/// Inverse of [`flatten_pair`]: split a 1-D combined tensor back into the two
/// outputs at their original ranks/shapes.
pub fn unflatten_pair<B: Backend, const DA: usize, const DB: usize>(
    combined: <B as BackendTypes>::FloatTensorPrimitive,
    flat_y_len: usize,
    flat_s_len: usize,
    shape_y: [usize; DA],
    shape_s: [usize; DB],
) -> (
    <B as BackendTypes>::FloatTensorPrimitive,
    <B as BackendTypes>::FloatTensorPrimitive,
) {
    let flat_y = B::float_slice(combined.clone(), &[s![0..flat_y_len]]);
    let y = B::float_reshape(flat_y, Shape::new(shape_y));
    let flat_s = B::float_slice(combined, &[s![flat_y_len..flat_y_len + flat_s_len]]);
    let s = B::float_reshape(flat_s, Shape::new(shape_s));
    (y, s)
}

/// Inverse of [`flatten_pair`]: split a 1-D combined tensor back into the two
/// outputs at their original ranks/shapes.
pub fn autodiff_unflatten_pair<
    B: Backend,
    C: CheckpointStrategy,
    const DA: usize,
    const DB: usize,
>(
    combined: FloatTensor<Autodiff<B, C>>,
    flat_y_len: usize,
    flat_s_len: usize,
    shape_y: [usize; DA],
    shape_s: [usize; DB],
) -> (FloatTensor<Autodiff<B, C>>, FloatTensor<Autodiff<B, C>>) {
    let flat_y = Autodiff::<B, C>::float_slice(combined.clone(), &[s![0..flat_y_len]]);
    let y = Autodiff::<B, C>::float_reshape(flat_y, Shape::new(shape_y));
    let flat_s =
        Autodiff::<B, C>::float_slice(combined, &[s![flat_y_len..flat_y_len + flat_s_len]]);
    let s = Autodiff::<B, C>::float_reshape(flat_s, Shape::new(shape_s));
    (y, s)
}
