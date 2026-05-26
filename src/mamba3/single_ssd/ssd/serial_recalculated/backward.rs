#![allow(non_snake_case)]

use crate::mamba3::single_ssd::ssd;
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::prelude::*;
use burn::tensor::{TensorPrimitive, ops::FloatTensor};
use ssd::serial_recalculated::{
    Mamba3SingleSsdBackendExt,
    combined_backward::{self, CombinedSingleSsdGrads},
};

impl<B: Backend + Mamba3SingleSsdBackendExt, C: CheckpointStrategy> Mamba3SingleSsdBackendExt
    for Autodiff<B, C>
{
    /// Memory-efficient combined forward+backward for the Mamba-3 MIMO
    /// Single-SSD.
    ///
    /// The two outputs (`y_bnlmhp`, `final_state_bhpr`) are flattened and
    /// concatenated into a single 1-D tracked tensor so one `Backward<B, 7>`
    /// node covers both. The seven differentiable inputs are `v, da, b, c,
    /// gamma, scale, initial_state`.
    fn single_ssd_serial_recalculated(
        v_bnlmhp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnlmhr: FloatTensor<Self>,
        c_bnlmhr: FloatTensor<Self>,
        gamma_bnlh: FloatTensor<Self>,
        scale_bnlh: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct CombinedKernelsBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            v_bnlmhp: FloatTensor<B>,
            da_bnlh: FloatTensor<B>,
            b_bnlmhr: FloatTensor<B>,
            c_bnlmhr: FloatTensor<B>,
            gamma_bnlh: FloatTensor<B>,
            scale_bnlh: FloatTensor<B>,
            initial_state_bhpr: FloatTensor<B>,
            flat_len_y_BNLMHP: usize,
            flat_len_final_state_BHPR: usize,
            shape_v_bnlmhp: [usize; 6],
            shape_da_bnlh: [usize; 4],
            shape_b_bnlmhr: [usize; 6],
            shape_c_bnlmhr: [usize; 6],
            shape_gamma_bnlh: [usize; 4],
            shape_scale_bnlh: [usize; 4],
            shape_initial_state_bhpr: [usize; 4],
            shape_y_bnlmhp: [usize; 6],
            shape_final_state_bhpr: [usize; 4],
        }

        impl<B: Backend + Mamba3SingleSsdBackendExt> Backward<B, 7> for CombinedKernelsBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_v_bnlmhp,
                    node_da_bnlh,
                    node_b_bnlmhr,
                    node_c_bnlmhr,
                    node_gamma_bnlh,
                    node_scale_bnlh,
                    node_initial_state_bhpr,
                ] = ops.parents;

                let d_combined: Tensor<B, 1> =
                    Tensor::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));

                let State {
                    v_bnlmhp,
                    da_bnlh,
                    b_bnlmhr,
                    c_bnlmhr,
                    gamma_bnlh,
                    scale_bnlh,
                    initial_state_bhpr,
                    flat_len_y_BNLMHP,
                    flat_len_final_state_BHPR,
                    shape_v_bnlmhp,
                    shape_da_bnlh,
                    shape_b_bnlmhr,
                    shape_c_bnlmhr,
                    shape_gamma_bnlh,
                    shape_scale_bnlh,
                    shape_initial_state_bhpr,
                    shape_y_bnlmhp,
                    shape_final_state_bhpr,
                } = ops.state;

                use crate::utils::primitive::mk;
                let v_bnlmhp = mk::<_, 6>(v_bnlmhp).reshape(shape_v_bnlmhp);
                let da_bnlh = mk::<_, 4>(da_bnlh).reshape(shape_da_bnlh);
                let b_bnlmhr = mk::<_, 6>(b_bnlmhr).reshape(shape_b_bnlmhr);
                let c_bnlmhr = mk::<_, 6>(c_bnlmhr).reshape(shape_c_bnlmhr);
                let gamma_bnlh = mk::<_, 4>(gamma_bnlh).reshape(shape_gamma_bnlh);
                let scale_bnlh = mk::<_, 4>(scale_bnlh).reshape(shape_scale_bnlh);
                let initial_state_bhpr =
                    mk::<_, 4>(initial_state_bhpr).reshape(shape_initial_state_bhpr);

                let (d_y_bnlmhp, d_final_state_bhpr) = crate::utils::combined_grad::unflatten_pair(
                    d_combined,
                    flat_len_y_BNLMHP,
                    flat_len_final_state_BHPR,
                    shape_y_bnlmhp,
                    shape_final_state_bhpr,
                );

                let CombinedSingleSsdGrads {
                    d_v_bnlmhp,
                    d_da_bnlh,
                    d_b_bnlmhr,
                    d_c_bnlmhr,
                    d_gamma_bnlh,
                    d_scale_bnlh,
                    d_initial_state_bhpr,
                } = combined_backward::combined_backward(
                    d_y_bnlmhp,
                    d_final_state_bhpr,
                    v_bnlmhp,
                    da_bnlh,
                    b_bnlmhr,
                    c_bnlmhr,
                    gamma_bnlh,
                    scale_bnlh,
                    initial_state_bhpr,
                );

                if let Some(n) = node_v_bnlmhp {
                    grads.register::<B>(n.id, d_v_bnlmhp.into_primitive().tensor());
                }
                if let Some(n) = node_da_bnlh {
                    grads.register::<B>(n.id, d_da_bnlh.into_primitive().tensor());
                }
                if let Some(n) = node_b_bnlmhr {
                    grads.register::<B>(n.id, d_b_bnlmhr.into_primitive().tensor());
                }
                if let Some(n) = node_c_bnlmhr {
                    grads.register::<B>(n.id, d_c_bnlmhr.into_primitive().tensor());
                }
                if let Some(n) = node_gamma_bnlh {
                    grads.register::<B>(n.id, d_gamma_bnlh.into_primitive().tensor());
                }
                if let Some(n) = node_scale_bnlh {
                    grads.register::<B>(n.id, d_scale_bnlh.into_primitive().tensor());
                }
                if let Some(n) = node_initial_state_bhpr {
                    grads.register::<B>(n.id, d_initial_state_bhpr.into_primitive().tensor());
                }
            }
        }

        use burn::tensor::TensorMetadata;
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] =
            v_bnlmhp.primitive.shape().dims();
        let [.., state_rank] = b_bnlmhr.primitive.shape().dims::<6>();

        let flat_len_y_BNLMHP = batch * nchunks * chunk_len * mimo_rank * nheads * per_head_dim;
        let flat_len_final_state_BHPR = batch * nheads * per_head_dim * state_rank;

        let shape_v_bnlmhp: [usize; 6] =
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim];
        let shape_da_bnlh: [usize; 4] = [batch, nchunks, chunk_len, nheads];
        let shape_b_bnlmhr: [usize; 6] = [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank];
        let shape_c_bnlmhr: [usize; 6] = [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank];
        let shape_gamma_bnlh: [usize; 4] = [batch, nchunks, chunk_len, nheads];
        let shape_scale_bnlh: [usize; 4] = [batch, nchunks, chunk_len, nheads];
        let shape_initial_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];
        let shape_y_bnlmhp: [usize; 6] =
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim];
        let shape_final_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];

        match CombinedKernelsBackward
            .prepare::<C>([
                v_bnlmhp.node.clone(),
                da_bnlh.node.clone(),
                b_bnlmhr.node.clone(),
                c_bnlmhr.node.clone(),
                gamma_bnlh.node.clone(),
                scale_bnlh.node.clone(),
                initial_state_bhpr.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (prim_y_bnlmhp, prim_final_state_bhpr) = B::single_ssd_serial_recalculated(
                    v_bnlmhp.primitive.clone(),
                    da_bnlh.primitive.clone(),
                    b_bnlmhr.primitive.clone(),
                    c_bnlmhr.primitive.clone(),
                    gamma_bnlh.primitive.clone(),
                    scale_bnlh.primitive.clone(),
                    initial_state_bhpr.primitive.clone(),
                );

                let (combined, _, _) = crate::utils::combined_grad::flatten_pair(
                    crate::utils::primitive::mk::<B, 6>(prim_y_bnlmhp),
                    crate::utils::primitive::mk::<B, 4>(prim_final_state_bhpr),
                );

                let state = State {
                    v_bnlmhp: v_bnlmhp.primitive.clone(),
                    da_bnlh: da_bnlh.primitive.clone(),
                    b_bnlmhr: b_bnlmhr.primitive.clone(),
                    c_bnlmhr: c_bnlmhr.primitive.clone(),
                    gamma_bnlh: gamma_bnlh.primitive.clone(),
                    scale_bnlh: scale_bnlh.primitive.clone(),
                    initial_state_bhpr: initial_state_bhpr.primitive.clone(),
                    flat_len_y_BNLMHP,
                    flat_len_final_state_BHPR,
                    shape_v_bnlmhp,
                    shape_da_bnlh,
                    shape_b_bnlmhr,
                    shape_c_bnlmhr,
                    shape_gamma_bnlh,
                    shape_scale_bnlh,
                    shape_initial_state_bhpr,
                    shape_y_bnlmhp,
                    shape_final_state_bhpr,
                };
                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(state, combined.into_primitive().tensor());

                let (tracked_y_bnlmhp, tracked_final_state_bhpr) =
                    crate::utils::combined_grad::unflatten_pair::<Autodiff<B, C>, 6, 4>(
                        crate::utils::primitive::mk(tracked_combined),
                        flat_len_y_BNLMHP,
                        flat_len_final_state_BHPR,
                        shape_y_bnlmhp,
                        shape_final_state_bhpr,
                    );

                (
                    tracked_y_bnlmhp.into_primitive().tensor(),
                    tracked_final_state_bhpr.into_primitive().tensor(),
                )
            }

            OpsKind::UnTracked(prep) => {
                let (prim_y_bnlmhp, prim_final_state_bhpr) = B::single_ssd_serial_recalculated(
                    v_bnlmhp.primitive,
                    da_bnlh.primitive,
                    b_bnlmhr.primitive,
                    c_bnlmhr.primitive,
                    gamma_bnlh.primitive,
                    scale_bnlh.primitive,
                    initial_state_bhpr.primitive,
                );

                let (combined, _, _) = crate::utils::combined_grad::flatten_pair(
                    crate::utils::primitive::mk::<B, 6>(prim_y_bnlmhp),
                    crate::utils::primitive::mk::<B, 4>(prim_final_state_bhpr),
                );

                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(combined.into_primitive().tensor());

                let (tracked_y_bnlmhp, tracked_final_state_bhpr) =
                    crate::utils::combined_grad::unflatten_pair::<Autodiff<B, C>, 6, 4>(
                        crate::utils::primitive::mk(tracked_combined),
                        flat_len_y_BNLMHP,
                        flat_len_final_state_BHPR,
                        shape_y_bnlmhp,
                        shape_final_state_bhpr,
                    );

                (
                    tracked_y_bnlmhp.into_primitive().tensor(),
                    tracked_final_state_bhpr.into_primitive().tensor(),
                )
            }
        }
    }
}
