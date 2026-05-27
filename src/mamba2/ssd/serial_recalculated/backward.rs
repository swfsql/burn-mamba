//! # Custom autodiff node for the Mamba-2 recompute backward
//!
//! Implements [`Mamba2BackendExt`] for `Autodiff<B>` by registering a single
//! Burn [`Backward`] node.  The forward stores only its (small) leaf inputs;
//! during backprop those are replayed through the K1–K5 kernels and the
//! analytic gradient math in [`combined_backward`], so the large intermediate
//! tensors never have to be kept alive — the ~⅓ training-memory saving of the
//! `SerialRecalculated` path.
//!
//! The two forward outputs (`y` and `final_state`) are flattened into one
//! tracked 1-D tensor (via [`crate::utils::combined_grad`]) so that a single
//! `Backward<B, 7>` node — one per the 7 differentiable inputs — covers both.

#![allow(non_snake_case)]

use crate::mamba2::ssd::serial_recalculated::{
    Mamba2BackendExt,
    combined_backward::{self, CombinedGrads},
};
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::prelude::*;
use burn::backend::{TensorPrimitive, tensor::FloatTensor};
use burn::backend::{Backend, BackendTypes, DispatchTensor};
use burn::backend::ops::FloatTensorOps;


impl<B: Backend + Mamba2BackendExt, C: CheckpointStrategy> Mamba2BackendExt for Autodiff<B, C> 
// where 
//     B: BackendTypes<FloatTensorPrimitive = DispatchTensor>
// where 
//     B: BackendTypes<FloatTensorPrimitive = DispatchTensor>
{
    /// Memory-efficient combined forward+backward.
    ///
    /// The two output tensors are concatenated into a single 1-dimensional tracked tensor
    /// so that one `Backward<B, 7>` node covers both outputs.  The caller
    /// receives split+reshaped slices of that combined tensor; burn's autodiff
    /// accumulates their upstream gradients back into a single gradient vector
    /// before firing this backward.
    fn ssd_serial_recalculated(
        x_bnlhp: FloatTensor<Self>,
        dt_discretized_bhnl: FloatTensor<Self>,
        b_bnlhr: FloatTensor<Self>,
        c_bnlhr: FloatTensor<Self>,
        d_h: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        a_decay_h: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // ── Backward struct ──────────────────────────────────────────────────
        #[derive(Debug)]
        struct CombinedKernelsBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            x_bnlhp: <B as BackendTypes>::FloatTensorPrimitive,
            dt_discretized_bhnl: <B as BackendTypes>::FloatTensorPrimitive,
            b_bnlhr: <B as BackendTypes>::FloatTensorPrimitive,
            c_bnlhr: <B as BackendTypes>::FloatTensorPrimitive,
            d_h: <B as BackendTypes>::FloatTensorPrimitive,
            initial_state_bhpr: <B as BackendTypes>::FloatTensorPrimitive,
            a_decay_h: <B as BackendTypes>::FloatTensorPrimitive,
            // flat byte-sizes for splitting the combined gradient vector
            flat_len_y_BNLHP: usize,
            flat_len_final_state_BHPR: usize,
            // shapes needed to reconstruct tensors in the right ranks
            shape_x_bnlhp: [usize; 5],
            shape_dt_discretized_bhnl: [usize; 4],
            shape_b_bnlhr: [usize; 5],
            shape_c_bnlhr: [usize; 5],
            shape_d_h: [usize; 1],
            shape_initial_state_bhpr: [usize; 4],
            shape_a_decay_h: [usize; 1],
            shape_y_bnlhp: [usize; 5],          // (output 1)
            shape_final_state_bhpr: [usize; 4], // (output 2)
        }

        /// State carried across the forward→backward boundary.
        ///
        /// Only the 7 original inputs are saved; all intermediates (cb, intra
        /// state, chunk_input_state) are recomputed during `backward`.
        #[allow(clippy::type_complexity)]
        impl<B: Backend + Mamba2BackendExt> Backward<B, 7> for CombinedKernelsBackward {
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
                    node_b_bnlhr,
                    node_c_bnlhr,
                    node_d_h,
                    node_initial_state_bhpr,
                    node_a_decay_h,
                ] = ops.parents;

                let d_combined: <B as BackendTypes>::FloatTensorPrimitive = grads.consume::<B>(&ops.node);

                let State {
                    x_bnlhp,
                    dt_discretized_bhnl,
                    b_bnlhr,
                    c_bnlhr,
                    d_h,
                    initial_state_bhpr,
                    a_decay_h,
                    //
                    flat_len_y_BNLHP,
                    flat_len_final_state_BHPR,
                    //
                    shape_x_bnlhp,
                    shape_dt_discretized_bhnl,
                    shape_b_bnlhr,
                    shape_c_bnlhr,
                    shape_d_h,
                    shape_initial_state_bhpr,
                    shape_a_decay_h,
                    //
                    shape_y_bnlhp,
                    shape_final_state_bhpr,
                } = ops.state;

                // ── Reconstruct saved tensors ──────────────────────────────
                use crate::utils::primitive::mk;

                let x_bnlhp = mk::<_, 5>(x_bnlhp).reshape(shape_x_bnlhp);
                let dt_discretized_bhnl =
                    mk::<_, 4>(dt_discretized_bhnl).reshape(shape_dt_discretized_bhnl);
                let b_bnlhr = mk::<_, 5>(b_bnlhr).reshape(shape_b_bnlhr);
                let c_bnlhr = mk::<_, 5>(c_bnlhr).reshape(shape_c_bnlhr);
                let d_h = mk::<_, 1>(d_h).reshape(shape_d_h);
                let initial_state_bhpr =
                    mk::<_, 4>(initial_state_bhpr).reshape(shape_initial_state_bhpr);
                let a_decay_h = mk::<_, 1>(a_decay_h).reshape(shape_a_decay_h);

                // ── Split incoming combined gradient ───────────────────────
                let (d_y_bnlhp, d_final_state_bhpr) = crate::utils::combined_grad::unflatten_pair(
                    d_combined,
                    flat_len_y_BNLHP,
                    flat_len_final_state_BHPR,
                    shape_y_bnlhp,
                    shape_final_state_bhpr,
                );

                // ── Core gradient computation ──────────────────────────────
                let CombinedGrads {
                    d_x_bnlhp,
                    d_dt_discretized_bhnl,
                    d_b_bnlhr,
                    d_c_bnlhr,
                    d_d_h,
                    d_initial_state_bhpr,
                    d_a_decay_h,
                    ..
                } = combined_backward::combined_backward(
                    d_y_bnlhp,
                    d_final_state_bhpr,
                    //
                    x_bnlhp,
                    dt_discretized_bhnl,
                    b_bnlhr,
                    c_bnlhr,
                    d_h,
                    initial_state_bhpr,
                    a_decay_h,
                );

                // ── Register gradients ─────────────────────────────────────
                // TODO: request Node to be re-exported.
                //
                // use burn::cubecl::stub::Arc;
                // use burn::backend::autodiff::Node;
                // let reg = |node: Option<Arc<_>>, grad: Tensor<_>| {
                //     if let Some(n) = node {
                //         grads.register::<B>(n.id, grad.into_primitive().tensor());
                //     }
                // };
                // let () = reg(node_x_bnlhp, d_x_bnlhp);
                // let () = reg(node_dt_discretized_bhnl, d_dt_discretized_bhnl);
                // let () = reg(node_b_bnlhr, d_b_bnlhr);
                // let () = reg(node_c_bnlhr, d_c_bnlhr);
                // let () = reg(node_d_h, d_d_h);
                // let () = reg(node_initial_state_bhpr, d_initial_state_bhpr);
                // let () = reg(node_da_cumsum_bhnl, d_da_cumsum_bhnl);

                if let Some(n) = node_x_bnlhp {
                    grads.register::<B>(n.id, d_x_bnlhp);
                }
                if let Some(n) = node_dt_discretized_bhnl {
                    grads.register::<B>(n.id, d_dt_discretized_bhnl.into_primitive().tensor());
                }
                if let Some(n) = node_b_bnlhr {
                    grads.register::<B>(n.id, d_b_bnlhr.into_primitive().tensor());
                }
                if let Some(n) = node_c_bnlhr {
                    grads.register::<B>(n.id, d_c_bnlhr.into_primitive().tensor());
                }
                if let Some(n) = node_d_h {
                    grads.register::<B>(n.id, d_d_h.into_primitive().tensor());
                }
                if let Some(n) = node_initial_state_bhpr {
                    grads.register::<B>(n.id, d_initial_state_bhpr.into_primitive().tensor());
                }
                if let Some(n) = node_a_decay_h {
                    grads.register::<B>(n.id, d_a_decay_h.into_primitive().tensor());
                }
            }
        } // end impl Backward

        // ── Shape extraction helpers ───────────────────────────────────────
        // Accessed via the AutodiffTensor wrappers (which own both .node
        // and .primitive).
        use burn::backend::TensorMetadata;
        let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.primitive.shape().dims();
        let [_, _, _, _nheads_b, state_rank] = b_bnlhr.primitive.shape().dims();

        let flat_len_y_BNLHP = batch * nchunks * chunk_len * nheads * per_head_dim;
        let flat_len_final_state_BHPR = batch * nheads * per_head_dim * state_rank;

        let shape_x_bnlhp: [usize; 5] = [batch, nchunks, chunk_len, nheads, per_head_dim];
        let shape_dt_discretized_bhnl: [usize; 4] = [batch, nheads, nchunks, chunk_len];
        let shape_b_bnlhr: [usize; 5] = [batch, nchunks, chunk_len, nheads, state_rank];
        let shape_c_bnlhr: [usize; 5] = [batch, nchunks, chunk_len, nheads, state_rank];
        let shape_d_h: [usize; 1] = [nheads];
        let shape_initial_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];
        let shape_a_decay_h: [usize; 1] = [nheads];
        let shape_y_bnlhp: [usize; 5] = [batch, nchunks, chunk_len, nheads, per_head_dim];
        let shape_final_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];

        // ── Register backward / run forward ───────────────────────────────
        match CombinedKernelsBackward
            .prepare::<C>([
                x_bnlhp.node.clone(),
                dt_discretized_bhnl.node.clone(),
                b_bnlhr.node.clone(),
                c_bnlhr.node.clone(),
                d_h.node.clone(),
                initial_state_bhpr.node.clone(),
                a_decay_h.node.clone(),
            ])
            .compute_bound()
            .stateful() // requires compute_bound
        {
            OpsKind::Tracked(prep) => {
                // Run the inner (non-autodiff) forward pass.
                let (prim_y_bnlhp, prim_final_state_bhpr) = B::ssd_serial_recalculated(
                    x_bnlhp.primitive.clone(),
                    dt_discretized_bhnl.primitive.clone(),
                    b_bnlhr.primitive.clone(),
                    c_bnlhr.primitive.clone(),
                    d_h.primitive.clone(),
                    initial_state_bhpr.primitive.clone(),
                    a_decay_h.primitive.clone(),
                );

                // prep.finish takes a single tensor, so pack both outputs into a
                // single 1-D tensor; one Backward node then covers both.
                let (prim_combined, _, _) = crate::utils::combined_grad::flatten_pair::<B>(
                    prim_y_bnlhp,
                    prim_final_state_bhpr,
                );

                let state = State {
                    x_bnlhp: x_bnlhp.primitive.clone(),
                    dt_discretized_bhnl: dt_discretized_bhnl.primitive.clone(),
                    b_bnlhr: b_bnlhr.primitive.clone(),
                    c_bnlhr: c_bnlhr.primitive.clone(),
                    d_h: d_h.primitive.clone(),
                    initial_state_bhpr: initial_state_bhpr.primitive.clone(),
                    a_decay_h: a_decay_h.primitive.clone(),
                    //
                    flat_len_y_BNLHP,
                    flat_len_final_state_BHPR,
                    //
                    shape_x_bnlhp, shape_dt_discretized_bhnl, shape_b_bnlhr, shape_c_bnlhr, shape_d_h, shape_initial_state_bhpr, shape_a_decay_h,
                    shape_y_bnlhp, shape_final_state_bhpr,
                };
                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(state, prim_combined);

                // Split the tracked combined tensor back into the two outputs.
                // The narrow/reshape ops are thin autodiff pass-throughs whose
                // backwards accumulate into the combined gradient vector that
                // `backward` above consumes.
                let (tracked_y_bnlhp, tracked_final_state_bhpr) =
                    crate::utils::combined_grad::autodiff_unflatten_pair::<B, C, 5, 4>(
                    tracked_combined,
                    flat_len_y_BNLHP,
                    flat_len_final_state_BHPR,
                    shape_y_bnlhp,
                    shape_final_state_bhpr,
                );

                (
                    tracked_y_bnlhp,
                    tracked_final_state_bhpr,
                )
            }

            OpsKind::UnTracked(prep) => {
                // No gradient tracking — just run the bare forward.
                let (prim_y_bnlhp, prim_final_state_bhpr) = B::ssd_serial_recalculated(
                    x_bnlhp.primitive,
                    dt_discretized_bhnl.primitive,
                    b_bnlhr.primitive,
                    c_bnlhr.primitive,
                    d_h.primitive,
                    initial_state_bhpr.primitive,
                    a_decay_h.primitive,
                );

                let (combined, _, _) = crate::utils::combined_grad::flatten_pair::<B>(
                    prim_y_bnlhp,
                    prim_final_state_bhpr,
                );

                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(combined);

                let (tracked_y_bnlhp, tracked_final_state_bhpr) = crate::utils::combined_grad::autodiff_unflatten_pair::<B, C, 5, 4>(
                    tracked_combined,
                    flat_len_y_BNLHP,
                    flat_len_final_state_BHPR,
                    shape_y_bnlhp,
                    shape_final_state_bhpr,
                );

                (
                    tracked_y_bnlhp,
                    tracked_final_state_bhpr,
                )
            }
        } // end match
    } // end fn ssd_serial_recalculated on Autodiff<B, C>
} // end impl Mamba2BackendExt for Autodiff<B, C>
