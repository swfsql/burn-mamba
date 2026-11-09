//! # `RecomputeOp` of the Mamba-2 SSD
//!
//! Implements [`Mamba2BackendExt`] for `Autodiff<B>` with one
//! [recompute node](burn_stack::utils::recompute_node). The node saves the
//! inputs of the forward and not its large intermediates. The backward replays
//! the inputs through [`combined_backward`], which calls the
//! [standard SSD scan](crate::standard_ssd). This is the ~⅓ training-memory
//! saving of the `SerialRecalculated` path.
//!
//! Without resets, `keep_bn` and `origin_bhpr` are placeholders with one
//! element, so the node saves them at no real cost.

use crate::mamba2::ssd::serial_recalculated::{
    Mamba2BackendExt, combined_backward::combined_backward,
};
use burn::backend::Backend;
use burn::backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};
use burn::backend::tensor::FloatTensor;
use burn_stack::utils::fprim::F;
use burn_stack::utils::recompute_node::{Prim, RecomputeOp, recompute_pair};

burn_stack::recompute_inputs!(Inputs {
    x_bnlhp,
    dt_discretized_bhnl,
    b_bnlhr,
    c_bnlhr,
    d_h,
    initial_state_bhpr,
    a_decay_h,
    origin_bhpr,
});

/// The values that the op takes with no gradient.
#[derive(Clone, Debug)]
struct Extra<B: Backend> {
    keep_bn: Prim<B>,
    resets: bool,
}

/// The [`RecomputeOp`] of the Mamba-2 SSD.
struct SsdRecompute;

impl<B: Backend + Mamba2BackendExt> RecomputeOp<B, 8> for SsdRecompute {
    type Inputs<T> = Inputs<T>;
    type Extra = Extra<B>;

    fn forward(inputs: Inputs<Prim<B>>, extra: &Extra<B>) -> (Prim<B>, Prim<B>) {
        B::ssd_serial_recalculated(
            inputs.x_bnlhp,
            inputs.dt_discretized_bhnl,
            inputs.b_bnlhr,
            inputs.c_bnlhr,
            inputs.d_h,
            inputs.initial_state_bhpr,
            inputs.a_decay_h,
            extra.keep_bn.clone(),
            inputs.origin_bhpr,
            extra.resets,
        )
    }

    fn backward(
        d_y_bnlhp: Prim<B>,
        d_final_state_bhpr: Prim<B>,
        inputs: Inputs<Prim<B>>,
        extra: Extra<B>,
    ) -> Inputs<Option<Prim<B>>> {
        let grads = combined_backward::<B>(
            F::new(d_y_bnlhp),
            F::new(d_final_state_bhpr),
            F::new(inputs.x_bnlhp),
            F::new(inputs.dt_discretized_bhnl),
            F::new(inputs.b_bnlhr),
            F::new(inputs.c_bnlhr),
            F::new(inputs.d_h),
            F::new(inputs.initial_state_bhpr),
            F::new(inputs.a_decay_h),
            extra.resets.then(|| (F::new(extra.keep_bn), F::new(inputs.origin_bhpr))),
        );
        Inputs {
            x_bnlhp: Some(grads.d_x_bnlhp.inner()),
            dt_discretized_bhnl: Some(grads.d_dt_discretized_bhnl.inner()),
            b_bnlhr: Some(grads.d_b_bnlhr.inner()),
            c_bnlhr: Some(grads.d_c_bnlhr.inner()),
            d_h: Some(grads.d_d_h.inner()),
            initial_state_bhpr: Some(grads.d_initial_state_bhpr.inner()),
            a_decay_h: Some(grads.d_a_decay_h.inner()),
            origin_bhpr: grads.d_origin_bhpr.map(F::inner),
        }
    }
}

impl<B: Backend + Mamba2BackendExt, C: CheckpointStrategy> Mamba2BackendExt for Autodiff<B, C> {
    fn ssd_serial_recalculated(
        x_bnlhp: FloatTensor<Self>,
        dt_discretized_bhnl: FloatTensor<Self>,
        b_bnlhr: FloatTensor<Self>,
        c_bnlhr: FloatTensor<Self>,
        d_h: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        a_decay_h: FloatTensor<Self>,
        keep_bn: FloatTensor<Self>,
        origin_bhpr: FloatTensor<Self>,
        resets: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        recompute_pair::<B, C, SsdRecompute, _>(
            Inputs {
                x_bnlhp,
                dt_discretized_bhnl,
                b_bnlhr,
                c_bnlhr,
                d_h,
                initial_state_bhpr,
                a_decay_h,
                origin_bhpr,
            },
            Extra { keep_bn: keep_bn.into_primitive(), resets },
        )
    }
}
