//! Right padding inside a block — the half of
//! [`burn_stack::utils::padding`]'s contract a family keeps.
//!
//! A padded row is absent: the outputs of the real rows and the returned cache
//! are those of each slot's real rows run alone. Every family meets it the same
//! two ways:
//!
//! - **the state** is carried through a padded row untouched, by zeroing its
//!   step's decay and writes where the discretisation forms them (`Δ = 0` gives
//!   `Ā = 1, B̄ = 0` — the same transparency the internal chunk padding relies
//!   on). Right padding means no real row reads a padded one, so nothing else
//!   has to know;
//! - **every cache field that is a slot's last few samples** (a conv window, a
//!   tap FIFO, a scan's carry) is read at each slot's own end — a gather at a
//!   per-slot offset ([`window`]), fixed-shape whatever the lengths.

use burn::prelude::*;

/// Real rows per slot, `[batch]`.
pub(crate) fn real_len_b(pad_bs: &Tensor<2, Bool>) -> Tensor<1, Int> {
    let [batch, _] = pad_bs.dims();
    pad_bs.clone().bool_not().int().sum_dim(1).reshape([batch])
}

/// `width` consecutive positions of `x` along `axis`, from `start_b` in each
/// slot (axis 0 is the batch): the per-slot `narrow(axis, start, width)`.
pub(crate) fn window<const D: usize>(
    x: Tensor<D>,
    axis: usize,
    start_b: Tensor<1, Int>,
    width: usize,
) -> Tensor<D> {
    let [batch] = start_b.dims();
    let device = x.device();
    let idx_bw = Tensor::<1, Int>::arange(0..width as i64, &device)
        .reshape([1, width])
        .expand([batch, width])
        .add(start_b.reshape([batch, 1]).expand([batch, width]));
    let mut shape = [1usize; D];
    shape[0] = batch;
    shape[axis] = width;
    let mut dims = x.dims();
    dims[axis] = width;
    x.gather(axis, idx_bw.reshape(shape).expand(dims))
}

/// `pad_bs` with every row repeated `u` times in place — a token's padding
/// spread over its `u` folded micro-steps (`[batch, tokens]` →
/// `[batch, tokens·u]`).
pub(crate) fn repeat_rows(pad_bt: Tensor<2, Bool>, u: usize) -> Tensor<2, Bool> {
    let [batch, tokens] = pad_bt.dims();
    pad_bt
        .unsqueeze_dim::<3>(2)
        .expand([batch, tokens, u])
        .reshape([batch, tokens * u])
}

/// `x` with every padded row's entries replaced by `value` — `x` is
/// `[batch, sequence, …]` and `pad_bs` is `[batch, sequence]`.
pub(crate) fn fill_padded<const D: usize>(x: Tensor<D>, pad_bs: &Tensor<2, Bool>, value: f32) -> Tensor<D> {
    let [batch, sequence] = pad_bs.dims();
    let mut shape = [1usize; D];
    shape[0] = batch;
    shape[1] = sequence;
    let dims = x.dims();
    x.mask_fill(pad_bs.clone().reshape(shape).expand(dims), value)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
