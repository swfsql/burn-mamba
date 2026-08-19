//! [`Segmented`]: run a different optimizer on each column block of a fused
//! projection weight.
//!
//! Burn's [`Muon`] orthogonalises the *whole* matrix it is handed. Handing it a
//! fused `in_proj` would couple the singular values of maps that have nothing to
//! do with each other (the gate `z`, the values `x`, the SSM keys/queries `B`/`C`,
//! the per-head Δ/`A`/`λ` scalars, …). [`Segmented`] slices the weight and its
//! gradient along the fused axis, steps each block with its own optimizer, and
//! concatenates the results — so each sub-matrix is orthogonalised (and
//! shape-LR-adjusted) on its own, exactly as if it had been a separate `Linear`.
//!
//! Nothing about the *model* changes: the forward pass keeps one fused GEMM.
//!
//! Slicing along columns is exact for every optimizer used here: AdamW is
//! elementwise, and Muon's Newton–Schulz is per-matrix. A [`Segmented`] whose
//! blocks are all AdamW is therefore bit-comparable to plain AdamW on the whole
//! tensor (asserted in `tests.rs`).

use burn::optim::{
    AdamW, AdamWState, LearningRate, Muon, MuonState, Optimizer, RecordState, StateSink,
    StateSource, join_index,
};
use burn::prelude::*;

use super::spec::ProjSpec;

/// The optimizer owning one column block.
#[derive(Clone)]
enum BlockOptim {
    /// Muon — orthogonalised momentum-SGD, for genuine feature maps.
    Muon(Muon),
    /// AdamW — the fallback for scalar-producing (or otherwise unsuitable) blocks.
    AdamW(AdamW),
}

/// One column block's optimizer state.
// The AdamW variant is the bigger one; a handful of these exist per fused
// weight, so the padding is not worth an extra indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum BlockState<const D: usize> {
    /// State of a [`Muon`] block.
    Muon(MuonState<D>),
    /// State of an [`AdamW`] block.
    AdamW(AdamWState<D>),
}

/// State of a [`Segmented`] optimizer: one entry per column block, in order.
#[derive(Clone)]
pub struct SegmentedState<const D: usize> {
    /// Per-block states, in the same order as the [`ProjSpec`] segments.
    pub blocks: Vec<BlockState<D>>,
}

/// Per-column-block optimizer over a fused projection weight.
///
/// Built by [`ProjSpec`]-driven [`MuonPlan`](super::MuonPlan) assembly; the
/// blocks' widths must sum to the parameter's size along [`Self::dim`].
#[derive(Clone)]
pub struct Segmented {
    optims: Vec<BlockOptim>,
    widths: Vec<usize>,
    dim: usize,
}

impl Segmented {
    /// Build the per-block optimizers for `spec`, splitting along `dim`
    /// (`1` for a Burn `Linear` weight, whose layout is `[d_input, d_output]`).
    pub fn new(spec: &ProjSpec, muon: Muon, adamw: AdamW, dim: usize) -> Self {
        let optims = spec
            .segments
            .iter()
            .map(|s| {
                if s.muon {
                    BlockOptim::Muon(muon.clone())
                } else {
                    BlockOptim::AdamW(adamw.clone())
                }
            })
            .collect();
        let widths = spec.segments.iter().map(|s| s.width).collect();
        Self { optims, widths, dim }
    }

    /// The axis the blocks are laid out along.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Optimizer for Segmented {
    type State<const D: usize> = SegmentedState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        assert!(
            self.dim < D,
            "Segmented: split dim {} out of range for a {D}D parameter",
            self.dim
        );
        let total: usize = self.widths.iter().sum();
        assert_eq!(
            tensor.shape().dims::<D>()[self.dim],
            total,
            "Segmented: the parameter's dim-{} width does not match the projection spec",
            self.dim
        );

        let tensors = tensor.split_with_sizes(self.widths.clone(), self.dim);
        let grads = grad.split_with_sizes(self.widths.clone(), self.dim);

        // A missing (first-step) state, or one whose length drifted from the
        // spec, restarts every block from scratch rather than mis-pairing them.
        let mut prev: Vec<Option<BlockState<D>>> = match state {
            Some(s) if s.blocks.len() == self.optims.len() => {
                s.blocks.into_iter().map(Some).collect()
            }
            _ => (0..self.optims.len()).map(|_| None).collect(),
        };

        let mut out = Vec::with_capacity(self.optims.len());
        let mut blocks = Vec::with_capacity(self.optims.len());

        for (i, optim) in self.optims.iter().enumerate() {
            let (t, g) = (tensors[i].clone(), grads[i].clone());
            match optim {
                BlockOptim::Muon(muon) => {
                    let prev = match prev[i].take() {
                        Some(BlockState::Muon(s)) => Some(s),
                        _ => None,
                    };
                    let (t, s) = muon.step(lr, t, g, prev);
                    out.push(t);
                    blocks.extend(s.map(BlockState::Muon));
                }
                BlockOptim::AdamW(adamw) => {
                    let prev = match prev[i].take() {
                        Some(BlockState::AdamW(s)) => Some(s),
                        _ => None,
                    };
                    let (t, s) = adamw.step(lr, t, g, prev);
                    out.push(t);
                    blocks.extend(s.map(BlockState::AdamW));
                }
            }
        }

        // Both inner optimizers always return a state, so `blocks` is complete.
        assert_eq!(blocks.len(), self.optims.len());
        (Tensor::cat(out, self.dim), Some(SegmentedState { blocks }))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        let blocks = state
            .blocks
            .into_iter()
            .map(|b| match b {
                BlockState::Muon(s) => BlockState::Muon(Muon::to_device(s, device)),
                BlockState::AdamW(s) => BlockState::AdamW(AdamW::to_device(s, device)),
            })
            .collect();
        SegmentedState { blocks }
    }
}

/// Hand-written because the `RecordState` derive covers `Vec<Tensor>` but not a
/// `Vec` of nested states, and because the reload has no access to the spec: the
/// two block kinds are told apart by their leaf names (`momentum.velocity` for
/// Muon, `momentum.moment_1`/`moment_2` for AdamW), which never overlap.
impl<const D: usize> RecordState for SegmentedState<D> {
    fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = join_index(prefix, i);
            match block {
                BlockState::Muon(s) => s.state_flatten(&prefix, out),
                BlockState::AdamW(s) => s.state_flatten(&prefix, out),
            }
        }
    }

    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        let mut blocks = Vec::new();
        for i in 0.. {
            let prefix = join_index(prefix, i);
            if !src.has_under(&prefix) {
                break;
            }
            // A failed attempt consumes nothing (the leaf it looks for is
            // absent), so trying Muon first is safe.
            let block = MuonState::state_unflatten(&prefix, src, device)
                .map(BlockState::Muon)
                .or_else(|| {
                    AdamWState::state_unflatten(&prefix, src, device).map(BlockState::AdamW)
                })?;
            blocks.push(block);
        }

        (!blocks.is_empty()).then_some(SegmentedState { blocks })
    }
}
