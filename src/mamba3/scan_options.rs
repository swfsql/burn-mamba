//! # The scan options of Mamba-3
//!
//! [`Mamba3ScanOptions`] holds the [SSD path](crate::ssd_path::SsdPath), the
//! chunk length of the chunkwise SSD scan and the [`Pathway`].
//! [`Mamba3::forward_packed`](crate::mamba3::mamba3::Mamba3::forward_packed),
//! which `forward` calls, sends the scan options to the pathway that they
//! select. Both pathways read the SSD path and the chunk length.

use crate::mamba3::prelude::*;
use burn::config::Config;

/// The pathway of a Mamba-3 forward: how the forward maps the trapezoidal
/// recurrence onto SSD calls. Both pathways give the same outputs, final
/// caches and gradients (ADR-0003).
///
/// [`Mamba3::step`] has no pathway. It runs the double-SSD recurrence.
///
/// The default is [`Self::SingleSsd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Pathway {
    /// One standard SSD call for each term of the trapezoid: a γ-SSD for the
    /// current sample, and one β-SSD for each `β` tap. That is two calls at
    /// the default tap pattern, three under a two-tap pattern, and one under
    /// [`TapPattern::None`].
    ///
    /// It follows the "two-SSD decomposition" of mamba3-minimal. See
    /// [`crate::mamba3::double_ssd`].
    DoubleSsd,

    /// One SSD call, at every tap pattern. It follows the form of the official
    /// kernels: Triton for SISO and Tilelang for MIMO. See
    /// [`crate::mamba3::single_ssd`].
    ///
    /// It uses about half the training memory of [`Self::DoubleSsd`] at the
    /// default tap pattern, and less than a third under a two-tap pattern.
    #[default]
    SingleSsd,
}

/// The scan options of a Mamba-3 block: the SSD path, the chunk length and
/// the pathway.
///
/// All scan options give the same values and gradients.
/// [`Mamba3::forward_packed`] runs the packed forward of the pathway of
/// [`Self::pathway`]. That forward computes the chunk length. It builds the
/// input bundle of the pathway at that length
/// ([`crate::mamba3::double_ssd::ssd::Mamba3DoubleSsdInput`] or
/// [`crate::mamba3::single_ssd::ssd::Mamba3SingleSsdInput`]). Then it gives
/// the SSD path to the `run` of the bundle.
///
/// The default is [`SsdPath::SerialRecalculated`] with an unset chunk length,
/// on [`Pathway::SingleSsd`]. For example,
/// `Mamba3ScanOptions::new(SsdPath::Serial).with_pathway(Pathway::DoubleSsd)`
/// selects the `Serial` SSD path on double-SSD.
#[derive(Config, Debug)]
pub struct Mamba3ScanOptions {
    /// The method that computes the chunkwise SSD scan.
    pub ssd_path: SsdPath,
    /// The chunk length, in write positions. A larger value gives more
    /// intra-chunk GEMM work and a shorter inter-chunk scan.
    ///
    /// `None` uses the optimal value, [`Self::optimal_chunk_len`]. That value
    /// is approximately `√(state_rank · per_head_dim)`, divided by the widening
    /// that `mimo_rank` and `micro_steps` already give the chunk.
    pub chunk_len: Option<usize>,
    /// The pathway of [`Mamba3::forward`] and [`Mamba3::forward_packed`].
    #[config(default = "Pathway::SingleSsd")]
    pub pathway: Pathway,
}

impl Default for Mamba3ScanOptions {
    fn default() -> Self {
        Self::new(SsdPath::default())
    }
}

impl Mamba3ScanOptions {
    /// Optimal chunk length, in **write positions**: `√(state_rank ·
    /// per_head_dim)` divided by `mimo_rank`, rounded up to a multiple of 32,
    /// clamped to `32 ..= 512`. `micro_steps` does not change this width:
    /// [`chunk_tokens`](crate::standard_ssd::chunk_tokens) absorbs it.
    ///
    /// The square root is the SISO rule of thumb: it balances the intra-chunk
    /// GEMMs against the inter-chunk scan. Two dials of the block widen a chunk
    /// without a place in this formula, and they are **not** the same widening:
    ///
    /// - **`mimo_rank`** (`m`) widens *both* axes of a chunk. Every rank writes
    ///   and every rank reads, so the intra-chunk matmuls run on the fused
    ///   `chunk_len · m`. The rule of thumb is about that product, hence the
    ///   divisor. An unchanged `chunk_len` would cost `m²`, and this costs `m`.
    ///   The reference gives the same advice on its `chunk_size` argument: "64
    ///   for SISO, 64/mimo_rank for MIMO".
    /// - **`micro_steps`** (`u`) widens only the **write** axis. All `u`
    ///   micro-steps of a token write to the state, but the block *reads* the
    ///   token once, at its last micro-step. So the score is
    ///   `[batch, nchunks, heads, T·m, T·u·m]` for `T` =
    ///   [`chunk_tokens`](crate::standard_ssd::chunk_tokens),
    ///   with `batch · sequence · heads · T · u · m²` elements. To keep that flat
    ///   in `u`, keep `T · u` (the chunk on the *write* axis) flat and use fewer
    ///   tokens.
    ///   Memory is the binding constraint here, because no fused kernel hides it.
    ///
    /// So `u` subdivides the chunk and does not shorten it, and `nchunks` grows
    /// like `u`, not `u²`. The returned length is a multiple of `micro_steps`.
    /// So each chunk is a whole number of tokens, and their positions are a
    /// contiguous run. It is on the 32 grid exactly when `micro_steps` divides
    /// it.
    ///
    /// `info/mamba-3/architecture-deltas.md` §8.
    pub fn optimal_chunk_len(
        state_rank: usize,
        per_head_dim: usize,
        mimo_rank: usize,
        micro_steps: usize,
    ) -> usize {
        let micro_steps = micro_steps.max(1);
        let chunk_len = (state_rank * per_head_dim)
            .isqrt()
            .div_ceil(mimo_rank.max(1)) // the fused read axis is `chunk_tokens · m`.
            .next_multiple_of(32) // rule-of-thumb: common plane dimension.
            .clamp(32, 512); // rule-of-thumb: one plane minimum, ceiling at 512.
        // A whole number of tokens per chunk, at (very nearly) that chunk length.
        chunk_len.div_ceil(micro_steps) * micro_steps
    }

    /// The chunk length of these scan options, or [`Self::optimal_chunk_len`]
    /// for the dimensions of `block` when unset. In both cases the function
    /// rounds the length **up to a multiple of `micro_steps`**, so a chunk is a
    /// whole number of tokens.
    ///
    /// With that rounding, the read axis is a plain reshape. With
    /// `chunk_len = T·u`, a chunk covers write positions `[cL, (c+1)L)`, that
    /// is tokens `[cT, (c+1)T)`, and the surviving positions (`≡ u−1 mod u`) are a
    /// contiguous run of them. At `micro_steps = 1` the rounding does nothing.
    pub fn chunk_len_or_optimal(&self, block: &Mamba3) -> usize {
        match self.chunk_len {
            Some(chunk_len) => chunk_len.next_multiple_of(block.micro_steps.max(1)),
            None => Self::optimal_chunk_len(
                block.state_rank,
                block.per_head_dim(),
                block.mimo_rank,
                block.micro_steps,
            ),
        }
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
