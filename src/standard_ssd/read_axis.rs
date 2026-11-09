//! # The read axis on primitives
//!
//! The helpers of the read axis for the recompute-backward math. A chunk has
//! `chunk_len` write positions, and a read position is the last of each run of
//! `read_stride` write positions. Each helper is the identity at
//! `read_stride = 1`.
//!
//! The `Tensor` twins of these helpers are in `src/mamba3/helpers.rs`. Every
//! kernel of this crate has this split: a `Tensor` body, and an `F<B, _>` body
//! that a [`Backward`](burn::backend::autodiff::ops::Backward) node can run
//! (see [`fprim`](burn_stack::utils::fprim)). Both are reshape-and-narrow, so
//! neither needs an op that `F` does not have.

use burn_stack::utils::fprim::F;
use burn::backend::Backend;
use burn::backend::tensor::Device;
use burn::backend::FloatDType;

/// The read positions of a write axis: index `i·stride + (stride − 1)` for each
/// `i`, that is the last position of every run of `stride`.
///
/// # Shapes
/// - `t`   : `[…, len, …]` at `dim`, `len` a multiple of `stride`
/// - out   : `[…, len / stride, …]`
///
/// `DP1 = D + 1`.
pub fn read_positions<B: Backend, const D: usize, const DP1: usize>(
    t: F<B, D>,
    dim: usize,
    stride: usize,
) -> F<B, D> {
    if stride <= 1 {
        return t;
    }
    let dims = t.dims();
    let len = dims[dim];
    assert_eq!(
        len % stride,
        0,
        "a write axis of {len} is not a whole number of {stride}-micro-step tokens",
    );
    let mut split = [0usize; DP1];
    split[..dim].copy_from_slice(&dims[..dim]);
    split[dim] = len / stride;
    split[dim + 1] = stride;
    split[dim + 2..].copy_from_slice(&dims[dim + 1..]);

    let mut out = dims;
    out[dim] = len / stride;
    t.reshape::<DP1>(split)
        .narrow(dim + 1, stride - 1, 1)
        .reshape::<D>(out)
}

/// The transpose of [`read_positions`]: scatter a read-position tensor back
/// onto the write axis, with zeros at the positions that have no readout.
///
/// This is the gradient of `read_positions`. The hand-written backward uses it
/// where autodiff does the same for the `Minimal` / `Serial` paths.
///
/// # Shapes
/// - `t`  : `[…, len / stride, …]` at `dim`
/// - out  : `[…, len, …]`
pub fn scatter_read_positions<B: Backend, const D: usize, const DP1: usize>(
    t: F<B, D>,
    dim: usize,
    stride: usize,
) -> F<B, D> {
    if stride <= 1 {
        return t;
    }
    let dims = t.dims();
    let device = t.device();
    let dtype = t.dtype();

    let mut split = [0usize; DP1];
    split[..dim].copy_from_slice(&dims[..dim]);
    split[dim] = dims[dim];
    split[dim + 1] = 1;
    split[dim + 2..].copy_from_slice(&dims[dim + 1..]);
    let mut pad = split;
    pad[dim + 1] = stride - 1;

    let mut out = dims;
    out[dim] = dims[dim] * stride;
    // The row is the *last* of its run, so the zeros go in front of it.
    F::cat(
        vec![F::<B, DP1>::zeros(pad, &device, dtype), t.reshape::<DP1>(split)],
        dim + 1,
    )
    .reshape::<D>(out)
}

/// The additive causal mask over `(read position, write source)`:
/// `0` where the source is early enough to contribute, `−∞` where it is not.
///
/// `diagonal` goes to `triu` on the **write** grid. `0` excludes the same
/// position (the single-SSD strict mask). `1` keeps it (the mask of the
/// standard SSD scan).
///
/// # Shapes
/// - out : `[chunk_len / stride, chunk_len]`
pub fn read_causal_mask<B: Backend>(
    chunk_len: usize,
    stride: usize,
    diagonal: i64,
    device: &Device<B>,
    dtype: FloatDType,
) -> F<B, 2> {
    let full = F::<B, 2>::full(
        [chunk_len, chunk_len],
        f32::NEG_INFINITY,
        device,
        dtype,
    )
    .triu(diagonal);
    read_positions::<B, 2, 3>(full, 0, stride)
}
