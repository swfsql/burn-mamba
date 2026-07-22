//! Custom autodiff nodes for the Mamba-3 K1 and K4 scans.

use super::state_passing::Mamba3StatePassingBackendExt;
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, BackendTypes};

impl<B: Backend + Mamba3StatePassingBackendExt, C: CheckpointStrategy> Mamba3StatePassingBackendExt
    for Autodiff<B, C>
{
    fn mamba3_chunk_cumsum(da_bnlh: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct ChunkCumsumBackward;

        impl<B: Backend + Mamba3StatePassingBackendExt> Backward<B, 1> for ChunkCumsumBackward {
            type State = ();

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_da] = ops.parents;
                let d_prefix = grads.consume::<B>(&ops.node);
                let d_da = B::mamba3_chunk_cumsum_backward(d_prefix);
                if let Some(node) = node_da {
                    grads.register::<B>(node.id, d_da);
                }
            }
        }

        match ChunkCumsumBackward
            .prepare::<C>([da_bnlh.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let prefix = B::mamba3_chunk_cumsum(da_bnlh.primitive);
                prep.finish((), prefix)
            }
            OpsKind::UnTracked(prep) => {
                let prefix = B::mamba3_chunk_cumsum(da_bnlh.primitive);
                prep.finish(prefix)
            }
        }
    }

    fn mamba3_state_passing(
        intra_bnhpr: FloatTensor<Self>,
        decay_bhn: FloatTensor<Self>,
        initial_bhpr: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct StatePassingBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            states_bn1hpr: <B as BackendTypes>::FloatTensorPrimitive,
            decay_bhn: <B as BackendTypes>::FloatTensorPrimitive,
        }

        impl<B: Backend + Mamba3StatePassingBackendExt> Backward<B, 3> for StatePassingBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_intra, node_decay, node_initial] = ops.parents;
                let d_states = grads.consume::<B>(&ops.node);
                let (d_intra, d_decay, d_initial) = B::mamba3_state_passing_backward(
                    ops.state.states_bn1hpr,
                    ops.state.decay_bhn,
                    d_states,
                );

                if let Some(node) = node_intra {
                    grads.register::<B>(node.id, d_intra);
                }
                if let Some(node) = node_decay {
                    grads.register::<B>(node.id, d_decay);
                }
                if let Some(node) = node_initial {
                    grads.register::<B>(node.id, d_initial);
                }
            }
        }

        match StatePassingBackward
            .prepare::<C>([
                intra_bnhpr.node.clone(),
                decay_bhn.node.clone(),
                initial_bhpr.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let states = B::mamba3_state_passing(
                    intra_bnhpr.primitive,
                    decay_bhn.primitive.clone(),
                    initial_bhpr.primitive,
                );
                let state = State {
                    states_bn1hpr: states.clone(),
                    decay_bhn: decay_bhn.primitive,
                };
                prep.finish(state, states)
            }
            OpsKind::UnTracked(prep) => {
                let states = B::mamba3_state_passing(
                    intra_bnhpr.primitive,
                    decay_bhn.primitive,
                    initial_bhpr.primitive,
                );
                prep.finish(states)
            }
        }
    }
}
