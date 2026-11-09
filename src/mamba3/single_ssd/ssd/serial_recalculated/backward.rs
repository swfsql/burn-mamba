//! # `RecomputeOp` of the Mamba-3 single-SSD
//!
//! Implements [`Mamba3SingleSsdBackendExt`] for `Autodiff<B>` with one
//! [recompute node](burn_stack::utils::recompute_node). The node saves the
//! inputs of the forward and not its large intermediates. The backward replays
//! the serial kernels and the gradient math in
//! [`super::combined_backward`](crate::mamba3::single_ssd::ssd::serial_recalculated::combined_backward).
//!
//! Without resets, `keep_bn` and `origin_bhpr` are placeholders with one
//! element. Without seeds, `seed_bnhpr` is also a placeholder with one element.
//! So the node saves them at no real cost.

use crate::mamba3::single_ssd::ssd::serial_recalculated::{
    Mamba3SingleSsdBackendExt, combined_backward::combined_backward,
};
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
    gamma_bnth,
    scale_bnlh,
    initial_state_bhpr,
    seed_bnhpr,
    origin_bhpr,
});

/// The values that the op takes with no gradient.
#[derive(Clone, Debug)]
struct Extra<B: Backend> {
    read_stride: usize,
    siso_specialization: bool,
    keep_bn: Prim<B>,
    resets: bool,
    seeded: bool,
}

/// The [`RecomputeOp`] of the Mamba-3 single-SSD.
struct SsdRecompute;

impl<B: Backend + Mamba3SingleSsdBackendExt> RecomputeOp<B, 9> for SsdRecompute {
    type Inputs<T> = Inputs<T>;
    type Extra = Extra<B>;

    fn forward(inputs: Inputs<Prim<B>>, extra: &Extra<B>) -> (Prim<B>, Prim<B>) {
        B::single_ssd_serial_recalculated(
            inputs.v_bnlmhp,
            inputs.da_bnlh,
            inputs.b_bnlmhr,
            inputs.c_bntmhr,
            inputs.gamma_bnth,
            inputs.scale_bnlh,
            inputs.initial_state_bhpr,
            extra.read_stride,
            extra.siso_specialization,
            extra.keep_bn.clone(),
            inputs.origin_bhpr,
            extra.resets,
            inputs.seed_bnhpr,
            extra.seeded,
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
            F::new(inputs.gamma_bnth),
            F::new(inputs.scale_bnlh),
            F::new(inputs.initial_state_bhpr),
            extra.read_stride,
            extra.siso_specialization,
            extra.resets.then(|| (F::new(extra.keep_bn), F::new(inputs.origin_bhpr))),
            extra.seeded.then(|| F::new(inputs.seed_bnhpr)),
        );
        Inputs {
            v_bnlmhp: Some(grads.d_v_bnlmhp.inner()),
            da_bnlh: Some(grads.d_da_bnlh.inner()),
            b_bnlmhr: Some(grads.d_b_bnlmhr.inner()),
            c_bntmhr: Some(grads.d_c_bntmhr.inner()),
            gamma_bnth: Some(grads.d_gamma_bnth.inner()),
            scale_bnlh: Some(grads.d_scale_bnlh.inner()),
            initial_state_bhpr: Some(grads.d_initial_state_bhpr.inner()),
            seed_bnhpr: grads.d_seed_bnhpr.map(F::inner),
            origin_bhpr: grads.d_origin_bhpr.map(F::inner),
        }
    }
}

impl<B: Backend + Mamba3SingleSsdBackendExt, C: CheckpointStrategy> Mamba3SingleSsdBackendExt
    for Autodiff<B, C>
{
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
        keep_bn: FloatTensor<Self>,
        origin_bhpr: FloatTensor<Self>,
        resets: bool,
        seed_bnhpr: FloatTensor<Self>,
        seeded: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        recompute_pair::<B, C, SsdRecompute, _>(
            Inputs {
                v_bnlmhp,
                da_bnlh,
                b_bnlmhr,
                c_bntmhr,
                gamma_bnth,
                scale_bnlh,
                initial_state_bhpr,
                seed_bnhpr,
                origin_bhpr,
            },
            Extra {
                read_stride,
                siso_specialization,
                keep_bn: keep_bn.into_primitive(),
                resets,
                seeded,
            },
        )
    }
}
