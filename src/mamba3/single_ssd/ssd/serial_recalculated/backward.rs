//! # Custom autodiff node for the Mamba-3 single-SSD recompute backward
//!
//! Implements [`Mamba3SingleSsdBackendExt`](crate::mamba3::single_ssd::ssd::Mamba3SingleSsdBackendExt)
//! for `Autodiff<B>` via a single Burn [`Backward`](burn::backend::autodiff::ops::Backward) node.  The forward keeps only
//! its leaf inputs; backprop replays the serial kernels and the gradient math in
//! [`super::combined_backward`](crate::mamba3::single_ssd::ssd::serial_recalculated::combined_backward), so the large intermediates are never retained.
//! The two outputs (`y`, `final_state`) are flattened into one tracked tensor
//! (via [`burn_stack::utils::combined_grad`]) so one node covers both.

#![allow(non_snake_case)]

use crate::mamba3::single_ssd::ssd;
use burn_stack::utils::fprim::F;
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, BackendTypes};
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
    /// The two outputs (`y_bntmhp`, `final_state_bhpr`) are flattened and
    /// concatenated into a single 1-D tracked tensor so one `Backward<B, 7>`
    /// node covers both. The seven differentiable inputs are `v, da, b, c,
    /// gamma, scale, initial_state`.
    fn single_ssd_serial_recalculated(
        v_bnlmhp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnlmhr: FloatTensor<Self>,
        c_bntmhr: FloatTensor<Self>,
        gamma_bnth: FloatTensor<Self>,
        scale_bnlh: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        read_stride: usize,
        siso_specialization: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct CombinedKernelsBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            v_bnlmhp: <B as BackendTypes>::FloatTensorPrimitive,
            da_bnlh: <B as BackendTypes>::FloatTensorPrimitive,
            b_bnlmhr: <B as BackendTypes>::FloatTensorPrimitive,
            c_bntmhr: <B as BackendTypes>::FloatTensorPrimitive,
            gamma_bnth: <B as BackendTypes>::FloatTensorPrimitive,
            scale_bnlh: <B as BackendTypes>::FloatTensorPrimitive,
            initial_state_bhpr: <B as BackendTypes>::FloatTensorPrimitive,
            read_stride: usize,
            siso_specialization: bool,
            flat_len_y_BNTMHP: usize,
            flat_len_final_state_BHPR: usize,
            shape_v_bnlmhp: [usize; 6],
            shape_da_bnlh: [usize; 4],
            shape_b_bnlmhr: [usize; 6],
            shape_c_bntmhr: [usize; 6],
            shape_gamma_bnth: [usize; 4],
            shape_scale_bnlh: [usize; 4],
            shape_initial_state_bhpr: [usize; 4],
            shape_y_bntmhp: [usize; 6],
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
                    node_c_bntmhr,
                    node_gamma_bnth,
                    node_scale_bnlh,
                    node_initial_state_bhpr,
                ] = ops.parents;

                let d_combined: <B as BackendTypes>::FloatTensorPrimitive =
                    grads.consume::<B>(&ops.node);

                let State {
                    v_bnlmhp,
                    da_bnlh,
                    b_bnlmhr,
                    c_bntmhr,
                    gamma_bnth,
                    scale_bnlh,
                    initial_state_bhpr,
                    read_stride,
                    siso_specialization,
                    flat_len_y_BNTMHP,
                    flat_len_final_state_BHPR,
                    shape_v_bnlmhp,
                    shape_da_bnlh,
                    shape_b_bnlmhr,
                    shape_c_bntmhr,
                    shape_gamma_bnth,
                    shape_scale_bnlh,
                    shape_initial_state_bhpr,
                    shape_y_bntmhp,
                    shape_final_state_bhpr,
                } = ops.state;

                // ── Reconstruct saved tensors as rank-tagged primitives ──
                let v_bnlmhp = F::<B, 6>::new(v_bnlmhp).reshape(shape_v_bnlmhp);
                let da_bnlh = F::<B, 4>::new(da_bnlh).reshape(shape_da_bnlh);
                let b_bnlmhr = F::<B, 6>::new(b_bnlmhr).reshape(shape_b_bnlmhr);
                let c_bntmhr = F::<B, 6>::new(c_bntmhr).reshape(shape_c_bntmhr);
                let gamma_bnth = F::<B, 4>::new(gamma_bnth).reshape(shape_gamma_bnth);
                let scale_bnlh = F::<B, 4>::new(scale_bnlh).reshape(shape_scale_bnlh);
                let initial_state_bhpr =
                    F::<B, 4>::new(initial_state_bhpr).reshape(shape_initial_state_bhpr);

                let (d_y_bntmhp, d_final_state_bhpr) =
                    burn_stack::utils::combined_grad::unflatten_pair::<B, 6, 4>(
                        d_combined,
                        flat_len_y_BNTMHP,
                        flat_len_final_state_BHPR,
                        shape_y_bntmhp,
                        shape_final_state_bhpr,
                    );

                let CombinedSingleSsdGrads {
                    d_v_bnlmhp,
                    d_da_bnlh,
                    d_b_bnlmhr,
                    d_c_bntmhr,
                    d_gamma_bnth,
                    d_scale_bnlh,
                    d_initial_state_bhpr,
                } = combined_backward::combined_backward(
                    F::<B, 6>::new(d_y_bntmhp),
                    F::<B, 4>::new(d_final_state_bhpr),
                    v_bnlmhp,
                    da_bnlh,
                    b_bnlmhr,
                    c_bntmhr,
                    gamma_bnth,
                    scale_bnlh,
                    initial_state_bhpr,
                    read_stride,
                    siso_specialization,
                );

                if let Some(n) = node_v_bnlmhp {
                    grads.register::<B>(n.id, d_v_bnlmhp.inner());
                }
                if let Some(n) = node_da_bnlh {
                    grads.register::<B>(n.id, d_da_bnlh.inner());
                }
                if let Some(n) = node_b_bnlmhr {
                    grads.register::<B>(n.id, d_b_bnlmhr.inner());
                }
                if let Some(n) = node_c_bntmhr {
                    grads.register::<B>(n.id, d_c_bntmhr.inner());
                }
                if let Some(n) = node_gamma_bnth {
                    grads.register::<B>(n.id, d_gamma_bnth.inner());
                }
                if let Some(n) = node_scale_bnlh {
                    grads.register::<B>(n.id, d_scale_bnlh.inner());
                }
                if let Some(n) = node_initial_state_bhpr {
                    grads.register::<B>(n.id, d_initial_state_bhpr.inner());
                }
            }
        }

        use burn::backend::TensorMetadata;
        let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] =
            v_bnlmhp.primitive.shape().dims();
        let [.., state_rank] = b_bnlmhr.primitive.shape().dims::<6>();

        // `C`, `γ` and `y` live on the chunk's read axis; everything else on its
        // write axis. See [`Mamba3SingleSsdInput::read_stride`].
        let chunk_tokens =
            crate::mamba3::ssd_path::Mamba3SsdPath::chunk_tokens(chunk_len, read_stride);

        let flat_len_y_BNTMHP = batch * nchunks * chunk_tokens * mimo_rank * nheads * per_head_dim;
        let flat_len_final_state_BHPR = batch * nheads * per_head_dim * state_rank;

        let shape_v_bnlmhp: [usize; 6] =
            [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim];
        let shape_da_bnlh: [usize; 4] = [batch, nchunks, chunk_len, nheads];
        let shape_b_bnlmhr: [usize; 6] = [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank];
        let shape_c_bntmhr: [usize; 6] =
            [batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank];
        let shape_gamma_bnth: [usize; 4] = [batch, nchunks, chunk_tokens, nheads];
        let shape_scale_bnlh: [usize; 4] = [batch, nchunks, chunk_len, nheads];
        let shape_initial_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];
        let shape_y_bntmhp: [usize; 6] =
            [batch, nchunks, chunk_tokens, mimo_rank, nheads, per_head_dim];
        let shape_final_state_bhpr: [usize; 4] = [batch, nheads, per_head_dim, state_rank];

        match CombinedKernelsBackward
            .prepare::<C>([
                v_bnlmhp.node.clone(),
                da_bnlh.node.clone(),
                b_bnlmhr.node.clone(),
                c_bntmhr.node.clone(),
                gamma_bnth.node.clone(),
                scale_bnlh.node.clone(),
                initial_state_bhpr.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (prim_y_bntmhp, prim_final_state_bhpr) = B::single_ssd_serial_recalculated(
                    v_bnlmhp.primitive.clone(),
                    da_bnlh.primitive.clone(),
                    b_bnlmhr.primitive.clone(),
                    c_bntmhr.primitive.clone(),
                    gamma_bnth.primitive.clone(),
                    scale_bnlh.primitive.clone(),
                    initial_state_bhpr.primitive.clone(),
                    read_stride,
                    siso_specialization,
                );

                let (prim_combined, _, _) = burn_stack::utils::combined_grad::flatten_pair::<B>(
                    prim_y_bntmhp,
                    prim_final_state_bhpr,
                );

                let state = State {
                    v_bnlmhp: v_bnlmhp.primitive.clone(),
                    da_bnlh: da_bnlh.primitive.clone(),
                    b_bnlmhr: b_bnlmhr.primitive.clone(),
                    c_bntmhr: c_bntmhr.primitive.clone(),
                    gamma_bnth: gamma_bnth.primitive.clone(),
                    scale_bnlh: scale_bnlh.primitive.clone(),
                    initial_state_bhpr: initial_state_bhpr.primitive.clone(),
                    read_stride,
                    siso_specialization,
                    flat_len_y_BNTMHP,
                    flat_len_final_state_BHPR,
                    shape_v_bnlmhp,
                    shape_da_bnlh,
                    shape_b_bnlmhr,
                    shape_c_bntmhr,
                    shape_gamma_bnth,
                    shape_scale_bnlh,
                    shape_initial_state_bhpr,
                    shape_y_bntmhp,
                    shape_final_state_bhpr,
                };
                let tracked_combined: FloatTensor<Autodiff<B, C>> =
                    prep.finish(state, prim_combined);

                let (tracked_y_bntmhp, tracked_final_state_bhpr) =
                    burn_stack::utils::combined_grad::autodiff_unflatten_pair::<B, C, 6, 4>(
                        tracked_combined,
                        flat_len_y_BNTMHP,
                        flat_len_final_state_BHPR,
                        shape_y_bntmhp,
                        shape_final_state_bhpr,
                    );

                (tracked_y_bntmhp, tracked_final_state_bhpr)
            }

            OpsKind::UnTracked(prep) => {
                let (prim_y_bntmhp, prim_final_state_bhpr) = B::single_ssd_serial_recalculated(
                    v_bnlmhp.primitive,
                    da_bnlh.primitive,
                    b_bnlmhr.primitive,
                    c_bntmhr.primitive,
                    gamma_bnth.primitive,
                    scale_bnlh.primitive,
                    initial_state_bhpr.primitive,
                    read_stride,
                    siso_specialization,
                );

                let (prim_combined, _, _) = burn_stack::utils::combined_grad::flatten_pair::<B>(
                    prim_y_bntmhp,
                    prim_final_state_bhpr,
                );

                let tracked_combined: FloatTensor<Autodiff<B, C>> = prep.finish(prim_combined);

                let (tracked_y_bntmhp, tracked_final_state_bhpr) =
                    burn_stack::utils::combined_grad::autodiff_unflatten_pair::<B, C, 6, 4>(
                        tracked_combined,
                        flat_len_y_BNTMHP,
                        flat_len_final_state_BHPR,
                        shape_y_bntmhp,
                        shape_final_state_bhpr,
                    );

                (tracked_y_bntmhp, tracked_final_state_bhpr)
            }
        }
    }
}
