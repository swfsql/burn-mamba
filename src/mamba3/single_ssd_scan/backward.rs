//! Memory-bounded exact custom backward for the fused rank-one SSD scan.

use super::single_ssd_scan::Mamba3SingleSsdScanBackendExt;
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, BackendTypes};

impl<B: Backend + Mamba3SingleSsdScanBackendExt, C: CheckpointStrategy>
    Mamba3SingleSsdScanBackendExt for Autodiff<B, C>
{
    fn mamba3_single_ssd_scan(
        v_bnl1hp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnl1hr: FloatTensor<Self>,
        c_bnl1hr: FloatTensor<Self>,
        gamma_bnlh: FloatTensor<Self>,
        scale_bnlh: FloatTensor<Self>,
        initial_bhpr: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SingleSsdScanBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            v: <B as BackendTypes>::FloatTensorPrimitive,
            da: <B as BackendTypes>::FloatTensorPrimitive,
            b: <B as BackendTypes>::FloatTensorPrimitive,
            c: <B as BackendTypes>::FloatTensorPrimitive,
            gamma: <B as BackendTypes>::FloatTensorPrimitive,
            scale: <B as BackendTypes>::FloatTensorPrimitive,
            packed: <B as BackendTypes>::FloatTensorPrimitive,
        }

        impl<B: Backend + Mamba3SingleSsdScanBackendExt> Backward<B, 7> for SingleSsdScanBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_v,
                    node_da,
                    node_b,
                    node_c,
                    node_gamma,
                    node_scale,
                    node_initial,
                ] = ops.parents;
                let d_packed = grads.consume::<B>(&ops.node);
                let (d_v, d_da, d_b, d_c, d_gamma, d_scale, d_initial) =
                    B::mamba3_single_ssd_scan_backward(
                        ops.state.v,
                        ops.state.da,
                        ops.state.b,
                        ops.state.c,
                        ops.state.gamma,
                        ops.state.scale,
                        ops.state.packed,
                        d_packed,
                    );
                for (node, grad) in [
                    (node_v, d_v),
                    (node_da, d_da),
                    (node_b, d_b),
                    (node_c, d_c),
                    (node_gamma, d_gamma),
                    (node_scale, d_scale),
                    (node_initial, d_initial),
                ] {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, grad);
                    }
                }
            }
        }

        match SingleSsdScanBackward
            .prepare::<C>([
                v_bnl1hp.node.clone(),
                da_bnlh.node.clone(),
                b_bnl1hr.node.clone(),
                c_bnl1hr.node.clone(),
                gamma_bnlh.node.clone(),
                scale_bnlh.node.clone(),
                initial_bhpr.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let packed = B::mamba3_single_ssd_scan(
                    v_bnl1hp.primitive.clone(),
                    da_bnlh.primitive.clone(),
                    b_bnl1hr.primitive.clone(),
                    c_bnl1hr.primitive.clone(),
                    gamma_bnlh.primitive.clone(),
                    scale_bnlh.primitive.clone(),
                    initial_bhpr.primitive,
                );
                prep.finish(
                    State {
                        v: v_bnl1hp.primitive,
                        da: da_bnlh.primitive,
                        b: b_bnl1hr.primitive,
                        c: c_bnl1hr.primitive,
                        gamma: gamma_bnlh.primitive,
                        scale: scale_bnlh.primitive,
                        packed: packed.clone(),
                    },
                    packed,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(B::mamba3_single_ssd_scan(
                v_bnl1hp.primitive,
                da_bnlh.primitive,
                b_bnl1hr.primitive,
                c_bnl1hr.primitive,
                gamma_bnlh.primitive,
                scale_bnlh.primitive,
                initial_bhpr.primitive,
            )),
        }
    }
}
