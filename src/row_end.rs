//! The row end: where a row reads each cache field that keeps its last
//! samples.
//!
//! Some cache fields keep the last `width` samples of a row, along one axis:
//!
//! - the conv window of Mamba-1 and Mamba-2 (`width = conv_kernel`),
//! - the tap FIFO of Mamba-3 (`width = lag`),
//! - the `ln Λ` and `c` carries of the Mamba-3 positive systems
//!   (`width = 1`).
//!
//! A call gives these fields for the next call. Each field is the last `width`
//! entries before the row end of each row, in the samples of the call after a
//! `start` field:
//!
//! ```text
//!   [ start | samples of the call ]  →  the `width` entries before the row end
//! ```
//!
//! `start` has `width` entries. It holds the samples from before the call, so
//! a call that is shorter than `width` also gives `width` entries. A
//! [`RowEnd`] does each such read of one call. The code builds it once for
//! each call, from the padding and the resets of the call.
//!
//! # Padding
//!
//! Under right padding ([`crate::padding`]), the row end of a row is one past
//! its last real position. A row with no real position reads `start`. Each row
//! has its own row end. So the read is a gather at an offset for each row,
//! with a fixed shape for all lengths.
//!
//! # Resets
//!
//! A packed call ([`crate::packing`]) returns the cache of the last segment of
//! each row, and that segment starts from the origin cache. So for a row with
//! a reset in the call, `start` is the origin field. Each entry from before the
//! start of the last segment comes from `start`. For a row with no reset,
//! `start` is the incoming field.
//!
//! Let `s` be the first position of the last segment of a row, or `0` for a
//! row with no reset. Let `e` be the row end. Then:
//!
//! - In `[start | last segment]`, the read starts at `e′ = max(e − s, 0)`, so
//!   its entry `j < width` is at `i = e′ + j`.
//! - In `[start | samples]`, that entry is at `i` for `i < width` (an entry of
//!   `start`), else at `i + s`.
//! - With no reset, `s = 0`, and the read is `[e, e + width)` of
//!   `[start | samples]`.
//!
//! A last segment that holds only padding has `e ≤ s`, and its read is
//! `start`: the field of the origin.
//!
//! # One `narrow`
//!
//! The gather and the `cat` cost memory. A read is one `narrow`, with no
//! gather, when no row of the call has padding and one of these conditions is
//! true:
//!
//! - The call has no resets.
//! - `width` is not more than the positions of one token.
//!
//! The row end of each row is then the end of the call. A segment holds at
//! least one token of the call, so a read of one token or less stays in the
//! last segment. A Mamba-3 read has `width ≤ u`, because a tap reads at most
//! `u` positions back (`lag ≤ u`) and a positive carry reads 1. A packed
//! Mamba-2 call still gathers. Its last segment can be shorter than
//! `conv_kernel`, because a packed call requires only
//! `chunk_len ≥ conv_kernel − 1`.
//!
//! When the call holds at least `width` positions, that `narrow` reads the
//! samples alone, with no `cat`.
//!
//! In the shape suffixes of this module, `w` is the `width` of a read.

use burn::prelude::*;

/// The row-end reader of one call: it reads each "last samples" cache field
/// at the row end of each row. See the module header.
pub(crate) struct RowEnd {
    /// `[batch]`: the row end of each row. `None` when no row of the call has
    /// padding. Each row end is then the end of the call.
    end_b: Option<Tensor<1, Int>>,
    /// `[batch, nchunks]` (`true` at a chunk that starts a segment), and the
    /// positions of one chunk. `None` for a call with no resets.
    resets: Option<(Tensor<2, Bool>, usize)>,
    /// The positions of one token: `u` for Mamba-3, `1` for Mamba-1 and
    /// Mamba-2.
    micro_steps: usize,
}

impl RowEnd {
    /// The reader of one call, from its padding (`pad_bs`, `true` at a padded
    /// position) and its resets. `resets` is `(reset_bn, chunk_len)`: `true`
    /// at a chunk that starts a segment, and the positions of one chunk.
    /// `micro_steps` is the positions of one token.
    ///
    /// # Shapes
    /// - `pad_bs`   : `[batch, len]`
    /// - `reset_bn` : `[batch, nchunks]`
    pub fn new(
        pad_bs: Option<&Tensor<2, Bool>>,
        resets: Option<(&Tensor<2, Bool>, usize)>,
        micro_steps: usize,
    ) -> Self {
        Self {
            end_b: pad_bs.map(row_end_b),
            resets: resets.map(|(reset_bn, chunk_len)| (reset_bn.clone(), chunk_len)),
            micro_steps,
        }
    }

    /// The last `width` entries of each row before its row end, along `axis`,
    /// from `[start | samples]`. Axis 0 is the batch.
    ///
    /// `start` is `origin` for a row with a reset in the call, else
    /// `incoming`. Each one has `width` entries along `axis`. A call with no
    /// resets does not read `origin`.
    ///
    /// # Panics
    ///
    /// If the call has resets and `origin` is `None`.
    pub fn read<const D: usize>(
        &self,
        samples: Tensor<D>,
        axis: usize,
        width: usize,
        incoming: Tensor<D>,
        origin: Option<Tensor<D>>,
    ) -> Tensor<D> {
        let reset_bn = self.resets.as_ref().map(|(reset_bn, _)| reset_bn);
        assert!(
            reset_bn.is_none() || origin.is_some(),
            "a row-end read in a call with resets needs the origin field: the last segment of a row \
             with a reset starts from it"
        );
        self.read_after(samples, axis, width, || match reset_bn.zip(origin) {
            None => incoming,
            Some((reset_bn, origin)) => origin_where_reset(incoming, origin, reset_bn),
        })
    }

    /// [`Self::read`] with zeros for `start` in each row. The decay of the
    /// slots of the tap FIFO reads this way, because the incoming slots
    /// already carry their decay.
    #[cfg_attr(not(feature = "mamba3"), allow(dead_code))]
    pub fn read_after_zeros<const D: usize>(&self, samples: Tensor<D>, axis: usize, width: usize) -> Tensor<D> {
        let mut dims = samples.dims();
        dims[axis] = width;
        let options = (&samples.device(), samples.dtype());
        self.read_after(samples, axis, width, || Tensor::zeros(dims, options))
    }

    /// The read from `[start() | samples]`. Only a read that holds an entry
    /// of `start` calls `start`.
    fn read_after<const D: usize>(
        &self,
        samples: Tensor<D>,
        axis: usize,
        width: usize,
        start: impl FnOnce() -> Tensor<D>,
    ) -> Tensor<D> {
        let dims = samples.dims();
        let [batch, len] = [dims[0], dims[axis]];
        if self.end_b.is_none() && (self.resets.is_none() || width <= self.micro_steps) {
            if len >= width {
                return samples.narrow(axis, len - width, width);
            }
            // A call is at least one token. So a call that is shorter than
            // `width` has no resets here, and `start` is the incoming field.
            return Tensor::cat(vec![start(), samples], axis).narrow(axis, len, width);
        }
        let device = samples.device();
        let end_bw = match &self.end_b {
            Some(end_b) => end_b.clone(),
            None => Tensor::full([batch], len as i64, &device),
        }
        .reshape([batch, 1])
        .expand([batch, width]);
        let offset_bw = Tensor::<1, Int>::arange(0..width as i64, &device)
            .reshape([1, width])
            .expand([batch, width]);
        let idx_bw = match &self.resets {
            None => offset_bw + end_bw,
            Some((reset_bn, chunk_len)) => {
                let first_bw = last_segment_start_b(reset_bn, *chunk_len)
                    .reshape([batch, 1])
                    .expand([batch, width]);
                // The index in `[start | last segment]`, then in
                // `[start | samples]`.
                let local_bw = offset_bw + (end_bw - first_bw.clone()).clamp_min(0);
                let in_start_bw = local_bw.clone().lower_elem(width as i64);
                local_bw + first_bw.mask_fill(in_start_bw, 0)
            }
        };
        assert_eq!([batch, width], idx_bw.dims());
        gather_rows(Tensor::cat(vec![start(), samples], axis), axis, idx_bw)
    }
}

/// `incoming`, where each row with a reset takes its row of `origin`: the
/// field that the last segment of each row starts from. Axis 0 is the batch.
fn origin_where_reset<const D: usize>(
    incoming: Tensor<D>,
    origin: Tensor<D>,
    reset_bn: &Tensor<2, Bool>,
) -> Tensor<D> {
    let dims = incoming.dims();
    let mut shape = [1usize; D];
    shape[0] = dims[0];
    let reset_rows = reset_bn.clone().any_dim(1).reshape(shape).expand(dims);
    incoming.mask_where(reset_rows, origin)
}

/// The first position of the last segment of each row, `[batch]`. A row with
/// no reset gives `0`.
fn last_segment_start_b(reset_bn: &Tensor<2, Bool>, chunk_len: usize) -> Tensor<1, Int> {
    let [batch, nchunks] = reset_bn.dims();
    (Tensor::<1, Int>::arange(0..nchunks as i64, &reset_bn.device()) * chunk_len as i64)
        .reshape([1, nchunks])
        .expand([batch, nchunks])
        .mask_fill(reset_bn.clone().bool_not(), 0)
        .max_dim(1)
        .reshape([batch])
}

/// The row end of each row, `[batch]`: one past its last real position. A row
/// with no real position gives `0`. In a packed row, padded positions can end
/// each segment, and only the last real position counts.
fn row_end_b(pad_bs: &Tensor<2, Bool>) -> Tensor<1, Int> {
    let [batch, len] = pad_bs.dims();
    Tensor::<1, Int>::arange(1..len as i64 + 1, &pad_bs.device())
        .reshape([1, len])
        .expand([batch, len])
        .mask_fill(pad_bs.clone(), 0)
        .max_dim(1)
        .reshape([batch])
}

/// The entries `idx_bw` of `x` along `axis`, one set for each row. Axis 0 is
/// the batch.
fn gather_rows<const D: usize>(x: Tensor<D>, axis: usize, idx_bw: Tensor<2, Int>) -> Tensor<D> {
    let [batch, width] = idx_bw.dims();
    let mut shape = [1usize; D];
    shape[0] = batch;
    shape[axis] = width;
    let mut dims = x.dims();
    dims[axis] = width;
    x.gather(axis, idx_bw.reshape(shape).expand(dims))
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
