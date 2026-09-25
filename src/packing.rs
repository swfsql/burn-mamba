//! Packed segments inside a block: several sequences one after another in one
//! row, each from one origin cache.
//!
//! `reset_bt` (`[batch, tokens]`, `true` at the first token of a segment)
//! marks where a row starts a new sequence. At that token, the block restarts
//! from the **origin** cache (`origin`, a fresh one for `origin = None`). The
//! tokens before the first reset of a row continue from the incoming cache
//! (`cache`), as in a plain call. So a stream of packed calls can continue a
//! sequence across a call boundary. A reset at the first token of a call
//! starts a new sequence.
//!
//! Each sequence gives what its own row gives in a batch of plain calls from
//! the origin: the outputs, the last cache and the gradients. This is true for
//! any split of the stream into calls. The previous segment gets no gradient
//! through a restart. The origin gets the gradients of every segment that
//! restarts from it. `origin = cache` restarts every segment from the incoming
//! cache (for example, a shared prefix).
//!
//! # Only at a chunk start
//!
//! A reset must be at the first token of a chunk (`chunk_len` tokens, or
//! `chunk_len / micro_steps` for Mamba-3). Then the chunkwise scan changes at
//! one place only, the carry between chunks (K4). Every other site that reads
//! an earlier sample (a conv tap, a trapezoid tap) crosses a reset exactly when
//! two conditions are true:
//!
//! - its chunk starts with a reset,
//! - it reads further back than its own index in the chunk.
//!
//! Such a read takes the value of the origin instead (`restart_heads`). So a
//! per-chunk flag is all that those sites need, and no scan runs over the row
//! for them. This needs a read that reaches back one chunk at most:
//!
//! - A Mamba-2 conv tap reads `conv_kernel − 1` rows back. So
//!   `Mamba2::forward_packed` asserts `chunk_len ≥ conv_kernel − 1`.
//! - A Mamba-3 tap reads `u` positions (one token) back at most.
//!
//! A segment can end with pad rows (`pad`), up to the next chunk start. The official Mamba-3 varlen kernels also start each
//! sequence at a chunk start, with a padded last chunk.
//!
//! Some scans run over the whole row: the cumulative rotation of Mamba-3 and
//! its positive systems. They restart at the first position of each segment,
//! from the carry of the origin.
//!
//! The cache that a call returns is that of the last segment of each row. Its
//! "last samples" fields (conv window, tap FIFO, carries) read the origin for
//! the entries before the start of that segment (`last_origin`,
//! `restart_window`).
//!
//! Only the serial SSD paths take resets. The inter-chunk decay of `Minimal`
//! is a `nchunks × nchunks` matrix, which a long packed row makes large.
//!
//! The entry points are `Mamba2::forward_packed` and
//! `Mamba3::forward_packed`.

use burn::prelude::*;

/// The resets of a packed row as an SSD pass takes them: the chunks that
/// restart, and the state that they restart from.
#[derive(Clone, Debug)]
pub struct Restarts {
    /// `[batch, nchunks]`: `true` at a chunk that starts a new segment.
    pub reset_bn: Tensor<2, Bool>,
    /// `[batch, nheads, per_head_dim, state_rank]`: the state of the origin
    /// cache. Each chunk of `reset_bn` starts from it.
    pub origin_bhpr: Tensor<4>,
}

impl Restarts {
    /// The restarts as a recompute op takes them. The extension macro takes
    /// no `Option` tensor, so the op gets `(keep_bn, origin_bhpr, resets)`:
    ///
    /// - `keep_bn`: `0` at a chunk that starts a new segment, `1` elsewhere.
    ///   It is a constant (no gradient), with the dtype of `like`.
    /// - `origin_bhpr`: the state that those chunks start from.
    /// - `resets`: `false` for `None`. Then the two tensors are placeholders
    ///   (each dimension 1) that the op does not read.
    ///
    /// `like` is `[batch, nchunks, …]`.
    pub(crate) fn op_inputs<const D: usize>(
        restarts: Option<Self>,
        like: &Tensor<D>,
    ) -> (Tensor<2>, Tensor<4>, bool) {
        let dims = like.dims();
        let options = (&like.device(), like.dtype());
        match restarts {
            Some(Restarts { reset_bn, origin_bhpr }) => (
                Tensor::ones([dims[0], dims[1]], options).mask_fill(reset_bn, 0.0),
                origin_bhpr,
                true,
            ),
            None => (
                Tensor::ones([1, 1], options),
                Tensor::zeros([1, 1, 1, 1], options),
                false,
            ),
        }
    }
}

/// `cache`, where each row with a reset takes the row of `origin` instead:
/// the field that the last segment of each row starts from. Axis 0 is the
/// batch.
pub(crate) fn last_origin<const D: usize>(
    cache: Tensor<D>,
    origin: Tensor<D>,
    reset_bn: &Tensor<2, Bool>,
) -> Tensor<D> {
    let dims = cache.dims();
    let mut shape = [1usize; D];
    shape[0] = dims[0];
    let restarted = reset_bn.clone().any_dim(1).reshape(shape).expand(dims);
    cache.mask_where(restarted, origin)
}

/// The reset flag of each chunk, `[batch, nchunks]`: the flag of its first
/// token. `nchunks = ⌈tokens / chunk_tokens⌉`, so a partial last chunk also
/// has a flag.
///
/// With `debug_assertions`, it panics if a reset is not at a chunk start (this
/// check reads the device).
pub(crate) fn chunk_resets(reset_bt: Tensor<2, Bool>, chunk_tokens: usize) -> Tensor<2, Bool> {
    let [batch, tokens] = reset_bt.dims();
    let nchunks = tokens.div_ceil(chunk_tokens);
    let pad = nchunks * chunk_tokens - tokens;
    // The partial last chunk is padded with "no reset".
    let whole_bt = if pad == 0 {
        reset_bt
    } else {
        let device = reset_bt.device();
        let none_bt = Tensor::<2, Int>::zeros([batch, pad], &device).bool();
        Tensor::cat(vec![reset_bt, none_bt], 1)
    };
    let reset_bnt = whole_bt.reshape([batch, nchunks, chunk_tokens]);
    if cfg!(debug_assertions) && chunk_tokens > 1 {
        let inside = reset_bnt.clone().narrow(2, 1, chunk_tokens - 1).any();
        assert!(
            !inside.into_scalar::<bool>().to_bool(),
            "a reset must be at the first token of a chunk (every {chunk_tokens} tokens)"
        );
    }
    reset_bnt.narrow(2, 0, 1).reshape([batch, nchunks])
}

/// `[batch, len]`: `true` at the first `width` positions of every chunk that
/// starts with a reset. `len ≤ nchunks · chunk_len`.
///
/// These are the positions whose reads `width` positions back cross the reset.
pub(crate) fn chunk_heads(
    reset_bn: &Tensor<2, Bool>,
    chunk_len: usize,
    width: usize,
    len: usize,
) -> Tensor<2, Bool> {
    let [batch, nchunks] = reset_bn.dims();
    let device = reset_bn.device();
    let head_bnl = Tensor::<1, Int>::arange(0..chunk_len as i64, &device)
        .lower_elem(width as i64)
        .reshape([1, 1, chunk_len])
        .expand([batch, nchunks, chunk_len]);
    reset_bn
        .clone()
        .unsqueeze_dim::<3>(2)
        .expand([batch, nchunks, chunk_len])
        .bool_and(head_bnl)
        .reshape([batch, nchunks * chunk_len])
        .narrow(1, 0, len)
}

/// `x`, where `head` replaces the first `width` positions (along `axis`) of
/// every chunk that starts with a reset. `head` holds the values of the origin
/// cache that those positions read at a restart: `x` with `width` entries
/// along `axis`, one set per row. Axis 0 is the batch.
pub(crate) fn restart_heads<const D: usize>(
    x: Tensor<D>,
    axis: usize,
    head: Tensor<D>,
    reset_bn: &Tensor<2, Bool>,
    chunk_len: usize,
) -> Tensor<D> {
    let dims = x.dims();
    let len = dims[axis];
    let width = head.dims()[axis];
    // `head` at every position, by its index in the chunk (clamped: the
    // positions past `width` are not selected). A gather, because its
    // backward is a scatter-add, and the backward of `repeat_dim` is wrong in
    // this Burn.
    let mut line = [1usize; D];
    line[axis] = len;
    let idx = Tensor::<1, Int>::arange(0..len as i64, &x.device())
        .remainder_scalar(chunk_len as i64)
        .clamp_max(width as i64 - 1)
        .reshape(line)
        .expand(dims);
    let tiled = head.gather(axis, idx);
    let mut shape = [1usize; D];
    shape[0] = dims[0];
    shape[axis] = len;
    let heads = chunk_heads(reset_bn, chunk_len, width, len)
        .reshape(shape)
        .expand(dims);
    x.mask_where(heads, tiled)
}

/// `[batch, len]`: the first position of the segment that holds each
/// position. It is `-1` where no reset comes before the position in its row:
/// the incoming cache continues there.
pub(crate) fn segment_start(reset_bn: &Tensor<2, Bool>, chunk_len: usize, len: usize) -> Tensor<2, Int> {
    let [batch, nchunks] = reset_bn.dims();
    let device = reset_bn.device();
    let first_bn = (Tensor::<1, Int>::arange(0..nchunks as i64, &device) * chunk_len as i64)
        .reshape([1, nchunks])
        .expand([batch, nchunks]);
    Tensor::<2, Int>::full([batch, nchunks], -1, &device)
        .mask_where(reset_bn.clone(), first_bn)
        .cummax(1)
        .unsqueeze_dim::<3>(2)
        .expand([batch, nchunks, chunk_len])
        .reshape([batch, nchunks * chunk_len])
        .narrow(1, 0, len)
}

/// `[batch]`: the first position of the last segment of each row, or `0` for
/// a row with no reset.
pub(crate) fn last_start_b(reset_bn: &Tensor<2, Bool>, chunk_len: usize) -> Tensor<1, Int> {
    let [batch, nchunks] = reset_bn.dims();
    let device = reset_bn.device();
    (Tensor::<1, Int>::arange(0..nchunks as i64, &device) * chunk_len as i64)
        .reshape([1, nchunks])
        .expand([batch, nchunks])
        .mask_fill(reset_bn.clone().bool_not(), 0)
        .max_dim(1)
        .reshape([batch])
}

/// The resets of a packed call on its folded axis (Mamba-3: `micro_steps`
/// positions per token), with what the sites that read the past need.
pub(crate) struct Segments {
    /// `[batch, nchunks]`: the chunks that start a segment.
    pub reset_bn: Tensor<2, Bool>,
    /// Positions per chunk.
    pub chunk_len: usize,
    /// `[batch, len]`: see [`segment_start`].
    pub start_bs: Tensor<2, Int>,
}

impl Segments {
    /// `len` is the number of positions of the call (not padded to a chunk).
    pub fn new(reset_bn: Tensor<2, Bool>, chunk_len: usize, len: usize) -> Self {
        let start_bs = segment_start(&reset_bn, chunk_len, len);
        Self {
            reset_bn,
            chunk_len,
            start_bs,
        }
    }

    /// `[batch, len]`: `true` at the first `width` positions of each segment
    /// (see [`chunk_heads`]).
    pub fn heads(&self, width: usize) -> Tensor<2, Bool> {
        let len = self.start_bs.dims()[1];
        chunk_heads(&self.reset_bn, self.chunk_len, width, len)
    }

    /// See [`restart_heads`].
    pub fn restart_heads<const D: usize>(&self, x: Tensor<D>, axis: usize, head: Tensor<D>) -> Tensor<D> {
        restart_heads(x, axis, head, &self.reset_bn, self.chunk_len)
    }

    /// See [`last_start_b`].
    pub fn last_start_b(&self) -> Tensor<1, Int> {
        last_start_b(&self.reset_bn, self.chunk_len)
    }

    /// See [`last_origin`].
    pub fn last_origin<const D: usize>(&self, cache: Tensor<D>, origin: Tensor<D>) -> Tensor<D> {
        last_origin(cache, origin, &self.reset_bn)
    }
}

/// [`crate::padding::window`] of `x_all` (`[start | row]` along `axis`, where
/// the `start` field is `width` entries) at `end_b`, as the last segment of
/// each row sees it. That segment starts from the `start` field, so an entry
/// from before its start is read from that field instead. `start` is the
/// field of [`last_origin`]: the origin for a row with a reset, else the
/// incoming cache. `last_start_b` is `0` for a row with no reset, which gives
/// the plain window.
pub(crate) fn restart_window<const D: usize>(
    x_all: Tensor<D>,
    axis: usize,
    end_b: Tensor<1, Int>,
    width: usize,
    last_start_b: Tensor<1, Int>,
) -> Tensor<D> {
    let [batch] = end_b.dims();
    let device = x_all.device();
    let start_bw = last_start_b.reshape([batch, 1]).expand([batch, width]);
    // In `[start | last segment]`, the window starts at the real length of
    // the segment (zero for a segment with no real row).
    let local_end_bw = (end_b.reshape([batch, 1]).expand([batch, width]) - start_bw.clone()).clamp_min(0);
    let local_bw = Tensor::<1, Int>::arange(0..width as i64, &device)
        .reshape([1, width])
        .expand([batch, width])
        + local_end_bw;
    // Back to `[start | row]`: a `start` entry stays, a segment entry moves
    // by the start of the segment.
    let in_cache_bw = local_bw.clone().lower_elem(width as i64);
    let idx_bw = local_bw + start_bw.mask_fill(in_cache_bw, 0);
    let mut shape = [1usize; D];
    shape[0] = batch;
    shape[axis] = width;
    let mut dims = x_all.dims();
    dims[axis] = width;
    x_all.gather(axis, idx_bw.reshape(shape).expand(dims))
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
