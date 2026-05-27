//! # Rank-tagged primitive tensor wrapper for the custom backward math
//!
//! [`F`] is a thin newtype over a backend's [`FloatTensor`] primitive that
//! mirrors the subset of the high-level [`Tensor`](burn::tensor::Tensor) method
//! API used by the recompute-backward gradient math
//! (`*/serial_recalculated/combined_backward.rs`).
//!
//! Why it exists: in Burn 0.22 the high-level `Tensor` is pinned to the global
//! `Dispatch` backend, so it cannot be built from an arbitrary backend `B`'s
//! primitive.  A custom [`Backward`](burn::backend::autodiff::ops::Backward)
//! node runs with a *generic* `B`, so its gradient math must operate directly on
//! `B`'s primitives via the `B::float_*` ops.  This wrapper keeps that
//! primitive-level math reading like the original `Tensor` code (method
//! chaining, shape-suffixed names) instead of deeply nested free-function calls.
//!
//! The rank `D` is a compile-time tag for parity with the ported code and to
//! catch rank mistakes; every operation ultimately defers to `B`'s
//! runtime-shaped primitive ops.

use burn::backend::Backend;
use burn::backend::ops::{BoolTensorOps, FloatTensorOps};
use burn::backend::tensor::{BoolTensor, Device, FloatTensor};
use burn::backend::{FloatDType, Scalar, Shape, Slice, SliceArg, TensorData, TensorMetadata};

/// A backend float-tensor primitive tagged with a compile-time rank `D`.
///
/// Mirrors the slice of [`Tensor`](burn::tensor::Tensor)'s method API needed by
/// the custom backward gradient math, operating directly on `B`'s primitives.
pub(crate) struct F<B: Backend, const D: usize>(pub FloatTensor<B>);

impl<B: Backend, const D: usize> Clone for F<B, D> {
    fn clone(&self) -> Self {
        F(self.0.clone())
    }
}

impl<B: Backend, const D: usize> F<B, D> {
    /// Wrap a raw primitive.
    pub fn new(p: FloatTensor<B>) -> Self {
        F(p)
    }

    /// Unwrap to the raw primitive.
    pub fn inner(self) -> FloatTensor<B> {
        self.0
    }

    /// Runtime shape as a `[usize; D]` array.
    pub fn dims(&self) -> [usize; D] {
        self.0.shape().dims()
    }

    /// Device the tensor lives on.
    pub fn device(&self) -> Device<B> {
        B::float_device(&self.0)
    }

    /// Float dtype of the tensor.
    pub fn dtype(&self) -> FloatDType {
        self.0.dtype().into()
    }

    /// Batched matrix multiplication over the last two dims.
    pub fn matmul(self, rhs: Self) -> Self {
        F(B::float_matmul(self.0, rhs.0))
    }

    /// Permute the axes (rank-preserving).
    pub fn permute(self, axes: [usize; D]) -> Self {
        F(B::float_permute(self.0, &axes))
    }

    /// Swap two axes.
    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        F(B::float_swap_dims(self.0, dim1, dim2))
    }

    /// Element-wise `exp`.
    pub fn exp(self) -> Self {
        F(B::float_exp(self.0))
    }

    /// Sum along `dim`, keeping it as a size-1 axis (rank-preserving).
    pub fn sum_dim(self, dim: usize) -> Self {
        F(B::float_sum_dim(self.0, dim))
    }

    /// Cumulative sum along `dim`.
    pub fn cumsum(self, dim: usize) -> Self {
        F(B::float_cumsum(self.0, dim))
    }

    /// Slice the tensor (rank-preserving), accepting the same `s![..]` args as
    /// the high-level API.
    pub fn slice<S: SliceArg>(self, slices: S) -> Self {
        let shape = self.0.shape();
        let slices = slices.into_slices(&shape);
        F(B::float_slice(self.0, &slices))
    }

    /// Narrow `dim` to `[start, start + length)` (rank-preserving).
    pub fn narrow(self, dim: usize, start: usize, length: usize) -> Self {
        let mut slices: Vec<Slice> = (0..dim).map(|_| Slice::from(..)).collect();
        slices.push(Slice::from(start..start + length));
        let shape = self.0.shape();
        let slices = (&slices[..]).into_slices(&shape);
        F(B::float_slice(self.0, &slices))
    }

    /// Reshape to a new rank `D2`.
    pub fn reshape<const D2: usize>(self, shape: [usize; D2]) -> F<B, D2> {
        F(B::float_reshape(self.0, Shape::new(shape)))
    }

    /// Broadcast-expand to `shape` (rank-preserving).
    pub fn expand(self, shape: [usize; D]) -> Self {
        F(B::float_expand(self.0, Shape::new(shape)))
    }

    /// Remove the size-1 axis at `dim`, yielding rank `D2 = D - 1`.
    pub fn squeeze_dim<const D2: usize>(self, dim: usize) -> F<B, D2> {
        let current = self.0.shape().dims::<D>();
        let mut new_dims = [0usize; D2];
        new_dims[..dim].copy_from_slice(&current[..dim]);
        new_dims[dim..].copy_from_slice(&current[dim + 1..]);
        F(B::float_reshape(self.0, Shape::new(new_dims)))
    }

    /// Insert a size-1 axis at `dim`, yielding rank `D2 = D + 1`.
    pub fn unsqueeze_dim<const D2: usize>(self, dim: usize) -> F<B, D2> {
        let shape = self.0.shape().dims::<D>();
        let mut dims = [1usize; D2];
        dims[0..dim].copy_from_slice(&shape[0..dim]);
        if dim < D {
            dims[dim] = 1;
            dims[(dim + 1)..].copy_from_slice(&shape[dim..]);
        } else {
            dims[dim] = 1;
        }
        F(B::float_reshape(self.0, Shape::new(dims)))
    }

    /// Insert size-1 axes at the given output positions, yielding rank `D2`.
    ///
    /// Mirrors [`Tensor::unsqueeze_dims`](burn::tensor::Tensor::unsqueeze_dims):
    /// negative indices count from the back and duplicates insert multiple axes.
    pub fn unsqueeze_dims<const D2: usize>(self, axes: &[isize]) -> F<B, D2> {
        let old_dims = self.0.shape().dims::<D>();
        let mut new_dims = [1usize; D2];

        // Resolve negative indices (counting from the back, in reverse order).
        let mut neg_offset = D2;
        let mut dim_indices = axes
            .iter()
            .map(|&d| {
                (if d < 0 {
                    neg_offset -= 1;
                    d + neg_offset as isize + 1
                } else {
                    d
                }) as usize
            })
            .collect::<Vec<usize>>();
        dim_indices.sort_unstable();
        // Duplicate axes mean "insert N dims at that index": bump duplicates.
        for i in 1..dim_indices.len() {
            if dim_indices[i] <= dim_indices[i - 1] {
                dim_indices[i] = dim_indices[i - 1] + 1;
            }
        }

        let mut dim_indices_curr = 0usize;
        let mut old_dims_curr = 0usize;
        for new_dims_curr in 0..D2 {
            if dim_indices_curr == dim_indices.len() {
                new_dims[new_dims_curr..].copy_from_slice(&old_dims[old_dims_curr..]);
                break;
            }
            if new_dims_curr == dim_indices[dim_indices_curr] {
                dim_indices_curr += 1;
            } else {
                new_dims[new_dims_curr] = old_dims[old_dims_curr];
                old_dims_curr += 1;
            }
        }

        F(B::float_reshape(self.0, Shape::new(new_dims)))
    }

    /// Zero everything strictly below the `diagonal` (keeps the upper triangle).
    ///
    /// Equivalent to [`Tensor::triu`](burn::tensor::Tensor::triu): builds the
    /// triangular bool mask over the last two dims and fills the masked region
    /// with `0`.
    pub fn triu(self, diagonal: i64) -> Self {
        let dims = self.0.shape().dims::<D>();
        let rows = dims[D - 2];
        let cols = dims[D - 1];
        let device = B::float_device(&self.0);

        let mask2 = tri_bool::<B>(rows, cols, diagonal, false, &device);
        let mut lead = [1usize; D];
        lead[D - 2] = rows;
        lead[D - 1] = cols;
        let mask = B::bool_reshape(mask2, Shape::new(lead));
        let mask = B::bool_expand(mask, Shape::new(dims));
        F(B::float_mask_fill(self.0, mask, Scalar::from(0.0f32)))
    }

    /// Fill the positions where `mask` is `true` with `value`.
    pub fn mask_fill(self, mask: Mask<B>, value: f32) -> Self {
        F(B::float_mask_fill(self.0, mask.0, Scalar::from(value)))
    }

    /// Concatenate same-rank tensors along `dim` (rank-preserving).
    pub fn cat(tensors: Vec<F<B, D>>, dim: usize) -> Self {
        F(B::float_cat(tensors.into_iter().map(|t| t.0).collect(), dim))
    }

    /// Stack same-rank tensors along a fresh axis `dim`, yielding rank `D2 = D + 1`.
    pub fn stack<const D2: usize>(tensors: Vec<F<B, D>>, dim: usize) -> F<B, D2> {
        let unsqueezed = tensors
            .into_iter()
            .map(|t| {
                let current = t.0.shape().dims::<D>();
                let mut new_dims = [1usize; D2];
                new_dims[0..dim].copy_from_slice(&current[0..dim]);
                new_dims[dim] = 1;
                new_dims[(dim + 1)..].copy_from_slice(&current[dim..]);
                B::float_reshape(t.0, Shape::new(new_dims))
            })
            .collect::<Vec<_>>();
        F(B::float_cat(unsqueezed, dim))
    }

    /// All-zeros tensor of the given shape, dtype and device.
    pub fn zeros(shape: [usize; D], device: &Device<B>, dtype: FloatDType) -> Self {
        F(B::float_zeros(Shape::new(shape), device, dtype))
    }

    /// Constant-filled tensor of the given shape, dtype and device.
    pub fn full(shape: [usize; D], value: f32, device: &Device<B>, dtype: FloatDType) -> Self {
        F(B::float_full(
            Shape::new(shape),
            Scalar::from(value),
            device,
            dtype,
        ))
    }
}

impl<B: Backend, const D: usize> core::ops::Add for F<B, D> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        F(B::float_add(self.0, rhs.0))
    }
}

impl<B: Backend, const D: usize> core::ops::Sub for F<B, D> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        F(B::float_sub(self.0, rhs.0))
    }
}

impl<B: Backend, const D: usize> core::ops::Mul for F<B, D> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        F(B::float_mul(self.0, rhs.0))
    }
}

impl<B: Backend, const D: usize> core::ops::Neg for F<B, D> {
    type Output = Self;
    fn neg(self) -> Self {
        F(B::float_neg(self.0))
    }
}

/// A boolean mask primitive used with [`F::mask_fill`].
///
/// Mirrors the slice of the bool-tensor API needed to build and broadcast the
/// causal masks in the custom backward (construct → reshape → expand).
pub(crate) struct Mask<B: Backend>(pub BoolTensor<B>);

impl<B: Backend> Clone for Mask<B> {
    fn clone(&self) -> Self {
        Mask(self.0.clone())
    }
}

impl<B: Backend> Mask<B> {
    /// `[rows, cols]` mask that is `true` strictly above the `offset` diagonal.
    ///
    /// Matches [`Tensor::tril_mask`](burn::tensor::Tensor::tril_mask): the
    /// `true` entries are the region a `tril` would fill.
    pub fn tril_mask(rows: usize, cols: usize, offset: i64, device: &Device<B>) -> Self {
        Mask(tri_bool::<B>(rows, cols, offset, true, device))
    }

    /// Reshape the mask to a new rank.
    pub fn reshape<const N: usize>(self, shape: [usize; N]) -> Self {
        Mask(B::bool_reshape(self.0, Shape::new(shape)))
    }

    /// Broadcast-expand the mask.
    pub fn expand<const N: usize>(self, shape: [usize; N]) -> Self {
        Mask(B::bool_expand(self.0, Shape::new(shape)))
    }
}

/// Build a `[rows, cols]` triangular boolean mask on the host.
///
/// Following `tri_mask` in Burn: with `matrix = row - col + offset`, the result
/// is `matrix < 0` when `lower` (the `tril_mask`/lower-triangle region) and
/// `matrix > 0` otherwise (the `triu_mask`/upper-triangle region).
fn tri_bool<B: Backend>(
    rows: usize,
    cols: usize,
    offset: i64,
    lower: bool,
    device: &Device<B>,
) -> BoolTensor<B> {
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let m = r as i64 - c as i64 + offset;
            data.push(if lower { m < 0 } else { m > 0 });
        }
    }
    B::bool_from_data(TensorData::new(data, [rows, cols]), device)
}

/// Primitive analogue of [`crate::utils::sanity::sanity`] for [`F`].
///
/// Panics if `t` contains a `NaN` (when [`crate::DENY_NAN`] is set) or an `Inf`
/// (when [`crate::DENY_INF`] is set).  A no-op — with no device read — when both
/// flags are `false` (the default), so it can be sprinkled through the backward
/// math at no release-build cost.
pub(crate) fn san<B: Backend, const D: usize>(t: &F<B, D>) {
    if !crate::DENY_NAN && !crate::DENY_INF {
        return;
    }
    let data = burn::tensor::read_sync(B::float_into_data(t.0.clone()))
        .expect("sanity check: failed to read tensor data");
    let mut has_nan = false;
    let mut has_inf = false;
    for v in data.iter::<f64>() {
        if crate::DENY_NAN && v.is_nan() {
            has_nan = true;
        }
        if crate::DENY_INF && v.is_infinite() {
            has_inf = true;
        }
    }
    if has_nan {
        eprintln!("got a NaN");
    }
    if has_inf {
        eprintln!("got a INF");
    }
    if has_nan || has_inf {
        panic!("sanity check failed");
    }
}
