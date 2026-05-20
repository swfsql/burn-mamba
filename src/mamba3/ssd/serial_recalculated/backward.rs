#![allow(non_snake_case)]

use crate::mamba3::ssd::serial_recalculated::{
    Mamba3BackendExt,
    combined_backward::{self, CombinedGrads},
};
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::prelude::*;
use burn::tensor::{TensorPrimitive, ops::FloatTensor};

impl<B: Backend + Mamba3BackendExt, C: CheckpointStrategy> Mamba3BackendExt for Autodiff<B, C> {
    /// Memory-efficient combined forward+backward for the Mamba-3 MIMO SSD.
    ///
    /// The two output tensors (`y_bnlrhp`, `final_state_bhpr`) are flattened
    /// and concatenated into a single 1-D tracked tensor so a single
    /// `Backward<B, 5>` node covers both. The caller receives split+reshaped
    /// slices of that combined tensor; burn's autodiff accumulates their
    /// upstream gradients back into one gradient vector before invoking
    /// `backward`.
    fn ssd_serial_recalculated(
        v_bnlrhp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnlrhn: FloatTensor<Self>,
        c_bnlrhn: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // ── Backward node definition ─────────────────────────────────────────
        #[derive(Debug)]
        struct K1K2K3K4K5CombinedBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            // Saved forward inputs
            v_bnlrhp: FloatTensor<B>,
            da_bnlh: FloatTensor<B>,
            b_bnlrhn: FloatTensor<B>,
            c_bnlrhn: FloatTensor<B>,
            initial_state_bhpr: FloatTensor<B>,
            // Flat lengths for splitting the combined upstream gradient
            flat_len_y_BNLRHP: usize,
            flat_len_final_state_BHPR: usize,
            // Shapes needed to reconstruct tensors at the right ranks
            shape_v_bnlrhp: [usize; 6],
            shape_da_bnlh: [usize; 4],
            shape_b_bnlrhn: [usize; 6],
            shape_c_bnlrhn: [usize; 6],
            shape_initial_state_bhpr: [usize; 4],
            shape_y_bnlrhp: [usize; 6],
            shape_final_state_bhpr: [usize; 4],
        }

        impl<B: Backend + Mamba3BackendExt> Backward<B, 5> for K1K2K3K4K5CombinedBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 5>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_v_bnlrhp,
                    node_da_bnlh,
                    node_b_bnlrhn,
                    node_c_bnlrhn,
                    node_initial_state_bhpr,
                ] = ops.parents;

                // Upstream gradient of the combined 1-D output.
                let d_combined: Tensor<B, 1> =
                    Tensor::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));

                let State {
                    v_bnlrhp,
                    da_bnlh,
                    b_bnlrhn,
                    c_bnlrhn,
                    initial_state_bhpr,
                    flat_len_y_BNLRHP,
                    flat_len_final_state_BHPR,
                    shape_v_bnlrhp,
                    shape_da_bnlh,
                    shape_b_bnlrhn,
                    shape_c_bnlrhn,
                    shape_initial_state_bhpr,
                    shape_y_bnlrhp,
                    shape_final_state_bhpr,
                } = ops.state;

                // ── Reconstruct saved tensors ───────────────────────────
                use super::serial_recalculated::mk;
                let v_bnlrhp = mk::<_, 6>(v_bnlrhp).reshape(shape_v_bnlrhp);
                let da_bnlh = mk::<_, 4>(da_bnlh).reshape(shape_da_bnlh);
                let b_bnlrhn = mk::<_, 6>(b_bnlrhn).reshape(shape_b_bnlrhn);
                let c_bnlrhn = mk::<_, 6>(c_bnlrhn).reshape(shape_c_bnlrhn);
                let initial_state_bhpr =
                    mk::<_, 4>(initial_state_bhpr).reshape(shape_initial_state_bhpr);

                // ── Split combined gradient vector ──────────────────────
                let flat_d_y_BNLRHP = d_combined.clone().narrow(0, 0, flat_len_y_BNLRHP);
                let flat_d_final_state_BHPR =
                    d_combined.narrow(0, flat_len_y_BNLRHP, flat_len_final_state_BHPR);
                let d_y_bnlrhp: Tensor<B, 6> = flat_d_y_BNLRHP.reshape(shape_y_bnlrhp);
                let d_final_state_bhpr: Tensor<B, 4> =
                    flat_d_final_state_BHPR.reshape(shape_final_state_bhpr);

                // ── Core gradient computation ───────────────────────────
                let CombinedGrads {
                    d_v_bnlrhp,
                    d_da_bnlh,
                    d_b_bnlrhn,
                    d_c_bnlrhn,
                    d_initial_state_bhpr,
                    ..
                } = combined_backward::combined_backward(
                    d_y_bnlrhp,
                    d_final_state_bhpr,
                    v_bnlrhp,
                    da_bnlh,
                    b_bnlrhn,
                    c_bnlrhn,
                    initial_state_bhpr,
                );

                // ── Register gradients with autodiff ────────────────────
                if let Some(n) = node_v_bnlrhp {
                    grads.register::<B>(n.id, d_v_bnlrhp.into_primitive().tensor());
                }
                if let Some(n) = node_da_bnlh {
                    grads.register::<B>(n.id, d_da_bnlh.into_primitive().tensor());
                }
                if let Some(n) = node_b_bnlrhn {
                    grads.register::<B>(n.id, d_b_bnlrhn.into_primitive().tensor());
                }
                if let Some(n) = node_c_bnlrhn {
                    grads.register::<B>(n.id, d_c_bnlrhn.into_primitive().tensor());
                }
                if let Some(n) = node_initial_state_bhpr {
                    grads.register::<B>(n.id, d_initial_state_bhpr.into_primitive().tensor());
                }
            }
        }

        // ── Shape extraction (via the AutodiffTensor wrappers) ─────────────
        use burn::tensor::TensorMetadata;
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] =
            v_bnlrhp.primitive.shape().dims();
        let [.., state_rank] = b_bnlrhn.primitive.shape().dims::<6>();

        let flat_len_y_BNLRHP = batch * nchunks * chunk_len * mimo_rank * nheads * per_head_dim;
        let flat_len_final_state_BHPR = batch * nheads * per_head_dim * state_rank;

        let shape_v_bnlrhp: [usize; 6] =
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim];
        let shape_da_bnlh: [usize; 4] = [batch, nchunks, chunk_len, nheads];
        let shape_b_bnlrhn: [usize; 6] = [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank];
        let shape_c_bnlrhn: [usize; 6] = [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank];
        let shape_initial_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];
        let shape_y_bnlrhp: [usize; 6] =
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim];
        let shape_final_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];

        // ── Register backward / run forward ───────────────────────────────
        match K1K2K3K4K5CombinedBackward
            .prepare::<C>([
                v_bnlrhp.node.clone(),
                da_bnlh.node.clone(),
                b_bnlrhn.node.clone(),
                c_bnlrhn.node.clone(),
                initial_state_bhpr.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (prim_y_bnlrhp, prim_final_state_bhpr) = B::ssd_serial_recalculated(
                    v_bnlrhp.primitive.clone(),
                    da_bnlh.primitive.clone(),
                    b_bnlrhn.primitive.clone(),
                    c_bnlrhn.primitive.clone(),
                    initial_state_bhpr.primitive.clone(),
                );

                // Flatten + cat both outputs into one tracked 1-D tensor so a
                // single backward node covers both.
                let flat_y_BNLRHP: Tensor<B, 1> =
                    Tensor::<B, 6>::from_primitive(TensorPrimitive::Float(prim_y_bnlrhp))
                        .reshape([flat_len_y_BNLRHP]);
                let flat_final_state_BHPR: Tensor<B, 1> =
                    Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(prim_final_state_bhpr))
                        .reshape([flat_len_final_state_BHPR]);
                let combined: Tensor<B, 1> =
                    Tensor::cat(vec![flat_y_BNLRHP, flat_final_state_BHPR], 0);

                let state = State {
                    v_bnlrhp: v_bnlrhp.primitive.clone(),
                    da_bnlh: da_bnlh.primitive.clone(),
                    b_bnlrhn: b_bnlrhn.primitive.clone(),
                    c_bnlrhn: c_bnlrhn.primitive.clone(),
                    initial_state_bhpr: initial_state_bhpr.primitive.clone(),
                    flat_len_y_BNLRHP,
                    flat_len_final_state_BHPR,
                    shape_v_bnlrhp,
                    shape_da_bnlh,
                    shape_b_bnlrhn,
                    shape_c_bnlrhn,
                    shape_initial_state_bhpr,
                    shape_y_bnlrhp,
                    shape_final_state_bhpr,
                };
                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(state, combined.into_primitive().tensor());

                // Split the tracked combined tensor back into the two outputs.
                // The narrow/reshape ops are thin autodiff pass-throughs whose
                // backwards accumulate into the combined gradient vector that
                // `backward` above consumes.
                let tracked_combined: Tensor<Autodiff<B, C>, 1> =
                    Tensor::from_primitive(TensorPrimitive::Float(tracked_combined));

                let tracked_y_bnlrhp: Tensor<Autodiff<B, C>, 6> = tracked_combined
                    .clone()
                    .narrow(0, 0, flat_len_y_BNLRHP)
                    .reshape(shape_y_bnlrhp);
                let tracked_final_state_bhpr: Tensor<Autodiff<B, C>, 4> = tracked_combined
                    .narrow(0, flat_len_y_BNLRHP, flat_len_final_state_BHPR)
                    .reshape(shape_final_state_bhpr);

                (
                    tracked_y_bnlrhp.into_primitive().tensor(),
                    tracked_final_state_bhpr.into_primitive().tensor(),
                )
            }

            OpsKind::UnTracked(prep) => {
                // No gradient tracking — just run the bare forward.
                let (prim_y_bnlrhp, prim_final_state_bhpr) = B::ssd_serial_recalculated(
                    v_bnlrhp.primitive,
                    da_bnlh.primitive,
                    b_bnlrhn.primitive,
                    c_bnlrhn.primitive,
                    initial_state_bhpr.primitive,
                );

                let flat_y_BNLRHP: Tensor<B, 1> =
                    Tensor::<B, 6>::from_primitive(TensorPrimitive::Float(prim_y_bnlrhp))
                        .reshape([flat_len_y_BNLRHP]);
                let flat_final_state_BHPR: Tensor<B, 1> =
                    Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(prim_final_state_bhpr))
                        .reshape([flat_len_final_state_BHPR]);
                let combined: Tensor<B, 1> =
                    Tensor::cat(vec![flat_y_BNLRHP, flat_final_state_BHPR], 0);

                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(combined.into_primitive().tensor());

                let tracked_combined: Tensor<Autodiff<B, C>, 1> =
                    Tensor::from_primitive(TensorPrimitive::Float(tracked_combined));
                let tracked_y_bnlrhp: Tensor<Autodiff<B, C>, 6> = tracked_combined
                    .clone()
                    .narrow(0, 0, flat_len_y_BNLRHP)
                    .reshape(shape_y_bnlrhp);
                let tracked_final_state_bhpr: Tensor<Autodiff<B, C>, 4> = tracked_combined
                    .narrow(0, flat_len_y_BNLRHP, flat_len_final_state_BHPR)
                    .reshape(shape_final_state_bhpr);

                (
                    tracked_y_bnlrhp.into_primitive().tensor(),
                    tracked_final_state_bhpr.into_primitive().tensor(),
                )
            }
        }
    }
}
