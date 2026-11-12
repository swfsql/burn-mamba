//! Packed segments inside a block: several sequences one after another in one
//! row, each from one origin cache.
//!
//! `reset_bt` (`[batch, tokens]`, `true` at the first token of a segment)
//! marks where a row starts a new sequence. At that token, the block starts
//! again from the **origin** cache (`origin`, a fresh one for `origin = None`). The
//! tokens before the first reset of a row continue from the incoming cache
//! (`cache`), as in a plain call. So a stream of packed calls can continue a
//! sequence across a call boundary. A reset at the first token of a call
//! starts a new sequence.
//!
//! Each sequence gives what its own row gives in a batch of plain calls from
//! the origin: the outputs, the last cache and the gradients. This is true for
//! any split of the stream into calls. The previous segment gets no gradient
//! through a reset. The origin gets the gradients of every segment that
//! starts from it. `origin = cache` starts every segment from the incoming
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
//! Such a read takes the value of the origin instead (`origin_heads`). So a
//! per-chunk flag is all that those sites need, and no scan runs over the row
//! for them. This needs a read that reaches back one chunk at most:
//!
//! - A Mamba-2 conv tap reads `conv_kernel − 1` positions back. So
//!   `Mamba2::forward_packed` asserts `chunk_len ≥ conv_kernel − 1`.
//! - A Mamba-3 tap reads `u` positions (one token) back at most.
//!
//! A segment can end with padded positions (`pad`), up to the next chunk start. The official Mamba-3 varlen kernels also start each
//! sequence at a chunk start, with a padded last chunk.
//!
//! Some scans run over the whole row: the cumulative rotation of Mamba-3 and
//! its positive systems. They start again at the first position of each segment,
//! from the carry of the origin.
//!
//! The cache that a call returns is that of the last segment of each row. Its
//! "last samples" fields (conv window, tap FIFO, carries) read the origin for
//! the entries before the start of that segment (`crate::row_end`).
//!
//! Only the serial SSD paths take resets. The inter-chunk decay of `Minimal`
//! is a `nchunks × nchunks` matrix, which a long packed row makes large.
//!
//! The entry points are `Mamba2::forward_packed` and
//! `Mamba3::forward_packed`.

// A build without Mamba-3 reads only a part of this module.
#![cfg_attr(not(feature = "mamba3"), allow(dead_code))]

use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Dispatch, ExtensionType};
#[cfg(feature = "autodiff")]
use burn::backend::{Autodiff, autodiff::checkpoint::strategy::CheckpointStrategy};
use burn::prelude::*;
use burn_stack::utils::fprim::F;

/// The resets of a packed row as an SSD pass takes them: the chunks with a
/// reset, and the state that they start from.
#[derive(Clone, Debug)]
pub struct SsdResets {
    /// `[batch, nchunks]`: `true` at a chunk that starts a new segment.
    pub reset_bn: Tensor<2, Bool>,
    /// `[batch, nheads, per_head_dim, state_rank]`: the state of the origin
    /// cache. Each chunk of `reset_bn` starts from it.
    pub origin_bhpr: Tensor<4>,
}

impl SsdResets {
    /// The resets as an SSD op takes them ([`OpResets`]). `keep_bn` gets the
    /// dtype of `like`, and `like` is `[batch, nchunks, …]`.
    pub(crate) fn op_inputs<const D: usize>(
        resets: Option<Self>,
        like: &Tensor<D>,
    ) -> OpResets<Dispatch> {
        let Some(SsdResets { reset_bn, origin_bhpr }) = resets else {
            return OpResets::None;
        };
        let dims = like.dims();
        let options = (&like.device(), like.dtype());
        let keep_bn = Tensor::<2>::ones([dims[0], dims[1]], options).mask_fill(reset_bn, 0.0);
        OpResets::Some {
            keep_bn: keep_bn.into_dispatch(),
            origin_bhpr: origin_bhpr.into_dispatch(),
        }
    }
}

/// The resets of a packed row as an SSD op takes them, on the primitives of
/// `B`. An op argument of this type has `#[extension_type]`.
///
/// This enum takes the place of an `Option` of the two tensors. Burn has no
/// `ExtensionType` for `Option`, and the orphan rule stops this crate from
/// adding one.
#[derive(ExtensionType)]
pub enum OpResets<B: Backend> {
    /// No chunk of the call has a reset.
    None,
    /// The chunks with a reset, and the state that they start from.
    Some {
        /// `[batch, nchunks]`: `0` at a chunk that starts a new segment, `1`
        /// elsewhere. It is a constant: the op registers no gradient for it.
        keep_bn: FloatTensor<B>,
        /// `[batch, nheads, per_head_dim, state_rank]`: the state that each
        /// such chunk starts from.
        origin_bhpr: FloatTensor<B>,
    },
}

impl<B: Backend> OpResets<B> {
    /// The pair `(keep_bn, origin_bhpr)` that the standard SSD scan takes.
    pub(crate) fn into_option(self) -> Option<(F<B, 2>, F<B, 4>)> {
        match self {
            OpResets::None => None,
            OpResets::Some { keep_bn, origin_bhpr } => {
                Some((F::new(keep_bn), F::new(origin_bhpr)))
            }
        }
    }

    /// The resets from the two parts that a recompute node keeps (see
    /// [`OpResets::into_recompute`]). `origin_bhpr` stays unread when
    /// `keep_bn` is `None`.
    #[cfg(feature = "autodiff")]
    pub(crate) fn from_recompute(keep_bn: Option<FloatTensor<B>>, origin_bhpr: FloatTensor<B>) -> Self {
        match keep_bn {
            Some(keep_bn) => OpResets::Some { keep_bn, origin_bhpr },
            None => OpResets::None,
        }
    }
}

#[cfg(feature = "autodiff")]
impl<B: Backend, C: CheckpointStrategy> OpResets<Autodiff<B, C>> {
    /// The resets as the `Autodiff` impl of an SSD op gives them to its
    /// recompute node:
    ///
    /// - `keep_bn`, on the primitives of `B`: a constant, with no gradient.
    /// - `origin_bhpr`: a tracked input. For `None`, it is a [`placeholder`].
    pub(crate) fn into_recompute(
        self,
        like: &FloatTensor<Autodiff<B, C>>,
    ) -> (Option<FloatTensor<B>>, FloatTensor<Autodiff<B, C>>) {
        match self {
            OpResets::Some { keep_bn, origin_bhpr } => (Some(keep_bn.into_primitive()), origin_bhpr),
            OpResets::None => (None, placeholder::<B, C, 4>(like)),
        }
    }
}

/// The placeholder of a missing tracked input of a recompute node: the
/// `origin_bhpr` or the `seed_bnhpr` of a call with no resets. The node takes
/// a fixed set of tracked inputs. So the `Autodiff` impl of an SSD op gives it
/// this tensor, which the op does not read. Each dimension is 1, and the
/// device and the dtype are those of `like`.
#[cfg(feature = "autodiff")]
pub(crate) fn placeholder<B: Backend, C: CheckpointStrategy, const D: usize>(
    like: &FloatTensor<Autodiff<B, C>>,
) -> FloatTensor<Autodiff<B, C>> {
    use burn::backend::TensorMetadata;
    F::<Autodiff<B, C>, D>::zeros([1; D], &like.device(), like.dtype().into()).inner()
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
/// cache that those positions read at a reset: `x` with `width` entries
/// along `axis`, one set per row. Axis 0 is the batch.
pub(crate) fn origin_heads<const D: usize>(
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

/// The resets of a packed call on its write axis (Mamba-3: `micro_steps`
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

    /// See [`origin_heads`].
    pub fn origin_heads<const D: usize>(&self, x: Tensor<D>, axis: usize, head: Tensor<D>) -> Tensor<D> {
        origin_heads(x, axis, head, &self.reset_bn, self.chunk_len)
    }
}

#[cfg(all(test, feature = "_dev-test", any(feature = "mamba2", feature = "mamba3")))]
mod tests;
