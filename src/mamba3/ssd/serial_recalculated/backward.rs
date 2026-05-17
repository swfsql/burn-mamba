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
    /// Memory-efficient combined forward+backward.
    ///
    /// The two output tensors are concatenated into a single 1-D tracked tensor
    /// so that one `Backward<B, 7>` node covers both outputs.  The caller
    /// receives split+reshaped slices of that combined tensor; burn's autodiff
    /// accumulates their upstream gradients back into a single gradient vector
    /// before firing this backward.
    fn ssd_serial_recalculated(
        // AI init interface:
        //
        // da_cumsum_bhnl: FloatTensor<Self>,
        // dt_discretized_bhnl: FloatTensor<Self>,
        // x_bnlhp: FloatTensor<Self>,
        // b_bnlgr: FloatTensor<Self>,
        // c_bnlgr: FloatTensor<Self>,
        // d_h: FloatTensor<Self>,
        // initial_state_bhpr: FloatTensor<Self>,

        // actual required interface:
        //
        x_bnlhp: FloatTensor<Self>,
        dt_discretized_bhnl: FloatTensor<Self>,
        b_bnlgr: FloatTensor<Self>,
        c_bnlgr: FloatTensor<Self>,
        d_h: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        da_cumsum_bhnl: FloatTensor<Self>,
        // current inner interface:
        //
        // d_da_cumsum_bhnl: Tensor<B, 4>,
        // d_dt_discretized_bhnl: Tensor<B, 4>,
        // d_x_bnlhp: Tensor<B, 5>,
        // d_b_bnlgr: Tensor<B, 5>,
        // d_c_bnlgr: Tensor<B, 5>,
        // d_d_h: Tensor<B, 1>,
        // d_initial_state_bhpr: Tensor<B, 4>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // ── Backward struct ──────────────────────────────────────────────────
        #[derive(Debug)]
        struct K2K3K4K5CombinedBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            x_bnlhp: FloatTensor<B>,
            dt_discretized_bhnl: FloatTensor<B>,
            b_bnlgr: FloatTensor<B>,
            c_bnlgr: FloatTensor<B>,
            d_h: FloatTensor<B>,
            initial_state_bhpr: FloatTensor<B>,
            da_cumsum_bhnl: FloatTensor<B>, // (from K1)
            // flat byte-sizes for splitting the combined gradient vector
            flat_len_y_BNLHP: usize,
            flat_len_final_state_BHPR: usize,
            // shapes needed to reconstruct tensors in the right ranks
            shape_x_bnlhp: [usize; 5],
            shape_dt_discretized_bhnl: [usize; 4],
            shape_b_bnlgr: [usize; 5],
            shape_c_bnlgr: [usize; 5],
            shape_d_h: [usize; 1],
            shape_initial_state_bhpr: [usize; 4],
            shape_da_cumsum_bhnl: [usize; 4],   // (from K1)
            shape_y_bnlhp: [usize; 5],          // (output 1)
            shape_final_state_bhpr: [usize; 4], // (output 2)
        }

        /// State carried across the forward→backward boundary.
        ///
        /// Only the 7 original inputs are saved; all intermediates (cb, intra
        /// state, chunk_input_state) are recomputed during `backward`.
        #[allow(clippy::type_complexity)]
        impl<B: Backend + Mamba3BackendExt> Backward<B, 7> for K2K3K4K5CombinedBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_x_bnlhp,
                    node_dt_discretized_bhnl,
                    node_b_bnlgr,
                    node_c_bnlgr,
                    node_d_h,
                    node_initial_state_bhpr,
                    node_da_cumsum_bhnl,
                ] = ops.parents;

                // Retrieve the gradient of the combined 1-D output.
                let d_combined: Tensor<B, 1> =
                    Tensor::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));

                let State {
                    x_bnlhp,
                    dt_discretized_bhnl,
                    b_bnlgr,
                    c_bnlgr,
                    d_h,
                    initial_state_bhpr,
                    da_cumsum_bhnl,
                    //
                    flat_len_y_BNLHP,
                    flat_len_final_state_BHPR,
                    //
                    shape_x_bnlhp,
                    shape_dt_discretized_bhnl,
                    shape_b_bnlgr,
                    shape_c_bnlgr,
                    shape_d_h,
                    shape_initial_state_bhpr,
                    shape_da_cumsum_bhnl,
                    //
                    shape_y_bnlhp,
                    shape_final_state_bhpr,
                } = ops.state;

                // ── Reconstruct saved tensors ──────────────────────────────
                use super::serial_recalculated::mk;

                let x_bnlhp = mk::<_, 5>(x_bnlhp).reshape(shape_x_bnlhp);
                let dt_discretized_bhnl =
                    mk::<_, 4>(dt_discretized_bhnl).reshape(shape_dt_discretized_bhnl);
                let b_bnlgr = mk::<_, 5>(b_bnlgr).reshape(shape_b_bnlgr);
                let c_bnlgr = mk::<_, 5>(c_bnlgr).reshape(shape_c_bnlgr);
                let d_h = mk::<_, 1>(d_h).reshape(shape_d_h);
                let initial_state_bhpr =
                    mk::<_, 4>(initial_state_bhpr).reshape(shape_initial_state_bhpr);
                let da_cumsum_bhnl = mk::<_, 4>(da_cumsum_bhnl).reshape(shape_da_cumsum_bhnl);

                // ── Split incoming combined gradient ───────────────────────
                // d_combined : [y_flat_len_BNLHP + fs_flat_len_BHPR]
                let flat_d_y_BNLHP = d_combined.clone().narrow(0, 0, flat_len_y_BNLHP);
                let flat_d_final_state_BHPR =
                    d_combined.narrow(0, flat_len_y_BNLHP, flat_len_final_state_BHPR);

                let d_y_bnlhp: Tensor<B, 5> = flat_d_y_BNLHP.reshape(shape_y_bnlhp);
                let d_final_state_bhpr: Tensor<B, 4> =
                    flat_d_final_state_BHPR.reshape(shape_final_state_bhpr);

                // ── Core gradient computation ──────────────────────────────
                let CombinedGrads {
                    d_x_bnlhp,
                    d_dt_discretized_bhnl,
                    d_b_bnlgr,
                    d_c_bnlgr,
                    d_d_h,
                    d_initial_state_bhpr,
                    d_da_cumsum_bhnl,
                } = combined_backward::combined_backward(
                    d_y_bnlhp,
                    d_final_state_bhpr,
                    //
                    x_bnlhp,
                    dt_discretized_bhnl,
                    b_bnlgr,
                    c_bnlgr,
                    d_h,
                    initial_state_bhpr,
                    da_cumsum_bhnl,
                );

                // ── Register gradients ─────────────────────────────────────
                // TODO: request Node to be re-exported.
                //
                // use burn::cubecl::stub::Arc;
                // use burn::backend::autodiff::Node;
                // let reg = |node: Option<Arc<_>>, grad: Tensor<B, _>| {
                //     if let Some(n) = node {
                //         grads.register::<B>(n.id, grad.into_primitive().tensor());
                //     }
                // };
                // let () = reg(node_x_bnlhp, d_x_bnlhp);
                // let () = reg(node_dt_discretized_bhnl, d_dt_discretized_bhnl);
                // let () = reg(node_b_bnlgr, d_b_bnlgr);
                // let () = reg(node_c_bnlgr, d_c_bnlgr);
                // let () = reg(node_d_h, d_d_h);
                // let () = reg(node_initial_state_bhpr, d_initial_state_bhpr);
                // let () = reg(node_da_cumsum_bhnl, d_da_cumsum_bhnl);

                if let Some(n) = node_x_bnlhp {
                    grads.register::<B>(n.id, d_x_bnlhp.into_primitive().tensor());
                }
                if let Some(n) = node_dt_discretized_bhnl {
                    grads.register::<B>(n.id, d_dt_discretized_bhnl.into_primitive().tensor());
                }
                if let Some(n) = node_b_bnlgr {
                    grads.register::<B>(n.id, d_b_bnlgr.into_primitive().tensor());
                }
                if let Some(n) = node_c_bnlgr {
                    grads.register::<B>(n.id, d_c_bnlgr.into_primitive().tensor());
                }
                if let Some(n) = node_d_h {
                    grads.register::<B>(n.id, d_d_h.into_primitive().tensor());
                }
                if let Some(n) = node_initial_state_bhpr {
                    grads.register::<B>(n.id, d_initial_state_bhpr.into_primitive().tensor());
                }
                if let Some(n) = node_da_cumsum_bhnl {
                    grads.register::<B>(n.id, d_da_cumsum_bhnl.into_primitive().tensor());
                }
            }
        } // end impl Backward

        // ── Shape extraction helpers ───────────────────────────────────────
        // Accessed via the AutodiffTensor wrappers (which own both .node
        // and .primitive).
        use burn::tensor::TensorMetadata;
        let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.primitive.shape().dims();
        let [_, _, _, ngroups, state_rank] = b_bnlgr.primitive.shape().dims();

        let flat_len_y_BNLHP = batch * nchunks * chunk_len * nheads * per_head_dim;
        let flat_len_final_state_BHPR = batch * nheads * per_head_dim * state_rank;

        let shape_x_bnlhp: [usize; 5] = [batch, nchunks, chunk_len, nheads, per_head_dim];
        let shape_dt_discretized_bhnl: [usize; 4] = [batch, nheads, nchunks, chunk_len];
        let shape_b_bnlgr: [usize; 5] = [batch, nchunks, chunk_len, ngroups, state_rank];
        let shape_c_bnlgr: [usize; 5] = [batch, nchunks, chunk_len, ngroups, state_rank];
        let shape_d_h: [usize; 1] = [nheads];
        let shape_initial_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];
        let shape_da_cumsum_bhnl: [usize; 4] = [batch, nheads, nchunks, chunk_len];
        let shape_y_bnlhp: [usize; 5] = [batch, nchunks, chunk_len, nheads, per_head_dim];
        let shape_final_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];

        // ── Register backward / run forward ───────────────────────────────
        match K2K3K4K5CombinedBackward
            .prepare::<C>([
                x_bnlhp.node.clone(),
                dt_discretized_bhnl.node.clone(),
                b_bnlgr.node.clone(),
                c_bnlgr.node.clone(),
                d_h.node.clone(),
                initial_state_bhpr.node.clone(),
                da_cumsum_bhnl.node.clone(),
            ])
            .compute_bound()
            .stateful() // requires compute_bound
        {
            OpsKind::Tracked(prep) => {
                // Run the inner (non-autodiff) forward pass.
                let (prim_y_bnlhp, prim_final_state_bhpr) = B::ssd_serial_recalculated(
                    x_bnlhp.primitive.clone(),
                    dt_discretized_bhnl.primitive.clone(),
                    b_bnlgr.primitive.clone(),
                    c_bnlgr.primitive.clone(),
                    d_h.primitive.clone(),
                    initial_state_bhpr.primitive.clone(),
                    da_cumsum_bhnl.primitive.clone(),
                );

                // Note: prep.finish accepts only a single tensor.
                // Flatten both outputs and cat into one 1-D tensor so that
                // one Backward node covers both.
                let flat_y_BNLHP: Tensor<B, 1> =
                    Tensor::<B, 5>::from_primitive(TensorPrimitive::Float(prim_y_bnlhp))
                        .reshape([flat_len_y_BNLHP]);
                let flat_final_state_BHPR: Tensor<B, 1> =
                    Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(prim_final_state_bhpr))
                        .reshape([flat_len_final_state_BHPR]);
                let combined: Tensor<B, 1> =
                    Tensor::cat(vec![flat_y_BNLHP, flat_final_state_BHPR], 0);

                let state = State {
                    x_bnlhp: x_bnlhp.primitive.clone(),
                    dt_discretized_bhnl: dt_discretized_bhnl.primitive.clone(),
                    b_bnlgr: b_bnlgr.primitive.clone(),
                    c_bnlgr: c_bnlgr.primitive.clone(),
                    d_h: d_h.primitive.clone(),
                    initial_state_bhpr: initial_state_bhpr.primitive.clone(),
                    da_cumsum_bhnl: da_cumsum_bhnl.primitive.clone(),
                    //
                    flat_len_y_BNLHP,
                    flat_len_final_state_BHPR,
                    //
                    shape_x_bnlhp, shape_dt_discretized_bhnl, shape_b_bnlgr, shape_c_bnlgr, shape_d_h, shape_initial_state_bhpr, shape_da_cumsum_bhnl,
                    shape_y_bnlhp, shape_final_state_bhpr,
                };
                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(state, combined.into_primitive().tensor());

                // Split the tracked 1-D tensor back into the two outputs.
                // The narrow / reshape ops create thin pass-through autodiff
                // nodes; their backward accumulates gradients into the
                // combined gradient vector consumed above.
                let tracked_combined: Tensor<Autodiff<B, C>, 1> =
                    Tensor::from_primitive(TensorPrimitive::Float(tracked_combined));

                let tracked_y_bnlhp: Tensor<Autodiff<B, C>, 5> = tracked_combined
                    .clone()
                    .narrow(0, 0, flat_len_y_BNLHP)
                    .reshape(shape_y_bnlhp);
                let tracked_final_state_bhpr: Tensor<Autodiff<B, C>, 4> = tracked_combined
                    .narrow(0, flat_len_y_BNLHP, flat_len_final_state_BHPR)
                    .reshape(shape_final_state_bhpr);

                (
                    tracked_y_bnlhp.into_primitive().tensor(),
                    tracked_final_state_bhpr.into_primitive().tensor(),
                )
            }

            OpsKind::UnTracked(prep) => {
                // No gradient tracking needed — run bare forward and return.
                let (prim_y_bnlhp, prim_final_state_bhpr) = B::ssd_serial_recalculated(
                    x_bnlhp.primitive,
                    dt_discretized_bhnl.primitive,
                    b_bnlgr.primitive,
                    c_bnlgr.primitive,
                    d_h.primitive,
                    initial_state_bhpr.primitive,
                    da_cumsum_bhnl.primitive,
                );

                // Note: prep.finish accepts only a single tensor.
                let flat_y_BNLHP: Tensor<B, 1> =
                    Tensor::<B, 5>::from_primitive(TensorPrimitive::Float(prim_y_bnlhp))
                        .reshape([flat_len_y_BNLHP]);
                let flat_final_state_BHPR: Tensor<B, 1> =
                    Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(prim_final_state_bhpr))
                        .reshape([flat_len_final_state_BHPR]);
                let combined: Tensor<B, 1> = Tensor::cat(vec![flat_y_BNLHP, flat_final_state_BHPR], 0);

                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(combined.into_primitive().tensor());

                let tracked_combined: Tensor<Autodiff<B, C>, 1> =
                    Tensor::from_primitive(TensorPrimitive::Float(tracked_combined));
                let tracked_y_bnlhp: Tensor<Autodiff<B, C>, 5> = tracked_combined
                    .clone()
                    .narrow(0, 0, flat_len_y_BNLHP)
                    .reshape(shape_y_bnlhp);
                let tracked_final_state_bhpr: Tensor<Autodiff<B, C>, 4> = tracked_combined
                    .narrow(0, flat_len_y_BNLHP, flat_len_final_state_BHPR)
                    .reshape(shape_final_state_bhpr);

                (
                    tracked_y_bnlhp.into_primitive().tensor(),
                    tracked_final_state_bhpr.into_primitive().tensor(),
                )
            }
        } // end match
    } // end fn ssd_serial_recalculated on Autodiff<B, C>
} // end impl Mamba3BackendExt for Autodiff<B, C>
