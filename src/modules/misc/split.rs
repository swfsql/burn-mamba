//! Typed-array variants of [`Tensor::split_with_sizes`].

use burn::prelude::*;

/// Like [`Tensor::split_with_sizes`] but returns a fixed-size array, enabling
/// `let [a, b, c, ...] = split_into::<…, N>(t, [size_a, size_b, size_c, ...], dim);`
/// destructuring at the call site instead of a fragile `parts.next().unwrap()` chain.
///
/// Panics if the underlying split does not produce exactly `N` parts (which
/// would indicate that the requested sizes do not cover the dimension).
pub fn split_into<const D: usize, const N: usize>(
    t: Tensor<D>,
    sizes: [usize; N],
    dim: usize,
) -> [Tensor<D>; N] {
    let parts = t.split_with_sizes(sizes.to_vec(), dim);
    let got = parts.len();
    parts
        .try_into()
        .unwrap_or_else(|_| panic!("split_into: expected {N} parts, got {got}"))
}
