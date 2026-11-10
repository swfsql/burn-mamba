//! # `RecomputeOp` of the Mamba-3 double-SSD
//!
//! Implements [`Mamba3DoubleSsdBackendExt`] for `Autodiff<B>` with one
//! [recompute node](burn_stack::utils::recompute_node). The node saves the
//! inputs of the forward and not its large intermediates. The backward replays
//! the serial kernels and the gradient math of the
//! [standard SSD scan](crate::standard_ssd).
//!
//! Without resets, `origin_bhpr` is a placeholder with one element
//! (`packing::placeholder`), which the node saves at no real cost.

use crate::mamba3::double_ssd::ssd::serial_recalculated::Mamba3DoubleSsdBackendExt;
use crate::packing::OpRestarts;
use crate::standard_ssd::combined_backward;
use burn::backend::Backend;
use burn::backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};
use burn::backend::tensor::FloatTensor;
use burn_stack::utils::fprim::F;
use burn_stack::utils::recompute_node::{Prim, RecomputeOp, recompute_pair};

burn_stack::recompute_inputs!(Inputs {
    v_bnlmhp,
    da_bnlh,
    b_bnlmhr,
    c_bntmhr,
    initial_state_bhpr,
    origin_bhpr,
});

/// The values that the op takes with no gradient.
#[derive(Clone, Debug)]
struct Extra<B: Backend> {
    read_stride: usize,
    /// `None` without resets.
    keep_bn: Option<Prim<B>>,
}

/// The [`RecomputeOp`] of the Mamba-3 double-SSD.
struct SsdRecompute;

impl<B: Backend + Mamba3DoubleSsdBackendExt> RecomputeOp<B, 6> for SsdRecompute {
    type Inputs<T> = Inputs<T>;
    type Extra = Extra<B>;

    fn forward(inputs: Inputs<Prim<B>>, extra: &Extra<B>) -> (Prim<B>, Prim<B>) {
        B::double_ssd_serial_recalculated(
            inputs.v_bnlmhp,
            inputs.da_bnlh,
            inputs.b_bnlmhr,
            inputs.c_bntmhr,
            inputs.initial_state_bhpr,
            extra.read_stride,
            OpRestarts::from_recompute(extra.keep_bn.clone(), inputs.origin_bhpr),
        )
    }

    fn backward(
        d_y_bntmhp: Prim<B>,
        d_final_state_bhpr: Prim<B>,
        inputs: Inputs<Prim<B>>,
        extra: Extra<B>,
    ) -> Inputs<Option<Prim<B>>> {
        let grads = combined_backward::<B>(
            F::new(d_y_bntmhp),
            F::new(d_final_state_bhpr),
            F::new(inputs.v_bnlmhp),
            F::new(inputs.da_bnlh),
            F::new(inputs.b_bnlmhr),
            F::new(inputs.c_bntmhr),
            F::new(inputs.initial_state_bhpr),
            extra.read_stride,
            OpRestarts::from_recompute(extra.keep_bn, inputs.origin_bhpr).into_option(),
        );
        Inputs {
            v_bnlmhp: Some(grads.d_v_bnlmhp.inner()),
            da_bnlh: Some(grads.d_da_bnlh.inner()),
            b_bnlmhr: Some(grads.d_b_bnlmhr.inner()),
            c_bntmhr: Some(grads.d_c_bntmhr.inner()),
            initial_state_bhpr: Some(grads.d_initial_state_bhpr.inner()),
            origin_bhpr: grads.d_origin_bhpr.map(F::inner),
        }
    }
}

impl<B: Backend + Mamba3DoubleSsdBackendExt, C: CheckpointStrategy> Mamba3DoubleSsdBackendExt
    for Autodiff<B, C>
{
    fn double_ssd_serial_recalculated(
        v_bnlmhp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnlmhr: FloatTensor<Self>,
        c_bntmhr: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        read_stride: usize,
        restarts: OpRestarts<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (keep_bn, origin_bhpr) = restarts.into_recompute(&v_bnlmhp);
        recompute_pair::<B, C, SsdRecompute, _>(
            Inputs { v_bnlmhp, da_bnlh, b_bnlmhr, c_bntmhr, initial_state_bhpr, origin_bhpr },
            Extra { read_stride, keep_bn },
        )
    }
}
