//! Backend extension and primitive reference implementation for a fused M=1 SSD scan.

use super::RECONSTRUCTION_INTERVAL;
use crate::utils::fprim::F;
use burn::backend::tensor::FloatTensor;
use burn::backend::*;
use burn::backend::{Dispatch, backend_extension};
use burn::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
fn single_ssd_scan_forward<B: Backend>(
    v_bnl1hp: F<B, 6>,
    da_bnlh: F<B, 4>,
    b_bnl1hr: F<B, 6>,
    c_bnl1hr: F<B, 6>,
    gamma_bnlh: F<B, 4>,
    scale_bnlh: F<B, 4>,
    initial_bhpr: F<B, 4>,
) -> F<B, 3> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnl1hp.dims();
    let [.., state_rank] = b_bnl1hr.dims();
    let tokens = nchunks * chunk_len;
    let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
    assert_eq!(mimo_rank, 1, "fused single-SSD scan requires MIMO rank one");
    assert!(
        tokens > 0,
        "fused single-SSD scan requires at least one token"
    );
    assert_eq!(
        [batch, nchunks, chunk_len, 1, nheads, state_rank],
        c_bnl1hr.dims()
    );
    assert_eq!([batch, nchunks, chunk_len, nheads], da_bnlh.dims());
    assert_eq!([batch, nchunks, chunk_len, nheads], gamma_bnlh.dims());
    assert_eq!([batch, nchunks, chunk_len, nheads], scale_bnlh.dims());
    assert_eq!(
        [batch, nheads, per_head_dim, state_rank],
        initial_bhpr.dims()
    );

    let v_bthp = v_bnl1hp.reshape([batch, tokens, nheads, per_head_dim]);
    let b_bthr = b_bnl1hr.reshape([batch, tokens, nheads, state_rank]);
    let c_bthr = c_bnl1hr.reshape([batch, tokens, nheads, state_rank]);
    let da_bth = da_bnlh.reshape([batch, tokens, nheads]);
    let gamma_bth = gamma_bnlh.reshape([batch, tokens, nheads]);
    let scale_bth = scale_bnlh.reshape([batch, tokens, nheads]);
    let mut state_bhpr = initial_bhpr;
    let mut outputs = Vec::with_capacity(tokens);
    let mut checkpoints = Vec::with_capacity(checkpoint_count);

    for token in 0..tokens {
        let v_bhp = v_bthp.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let b_bhr = b_bthr.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let c_bhr = c_bthr.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let da_bh11 = da_bth
            .clone()
            .narrow(1, token, 1)
            .squeeze_dim::<2>(1)
            .unsqueeze_dims::<4>(&[2, 3]);
        let gamma_bh11 = gamma_bth
            .clone()
            .narrow(1, token, 1)
            .squeeze_dim::<2>(1)
            .unsqueeze_dims::<4>(&[2, 3]);
        let scale_bh11 = scale_bth
            .clone()
            .narrow(1, token, 1)
            .squeeze_dim::<2>(1)
            .unsqueeze_dims::<4>(&[2, 3]);
        let injection_bhpr = v_bhp.unsqueeze_dim::<4>(3) * b_bhr.unsqueeze_dim::<4>(2);
        let pre_bhpr = da_bh11.exp() * state_bhpr;
        let y_bhp = (c_bhr.unsqueeze_dim::<4>(2)
            * (pre_bhpr.clone() + gamma_bh11 * injection_bhpr.clone()))
        .sum_dim(3)
        .squeeze_dim::<3>(3);
        outputs.push(y_bhp);
        state_bhpr = pre_bhpr + scale_bh11 * injection_bhpr;
        if (token + 1).is_multiple_of(RECONSTRUCTION_INTERVAL) || token + 1 == tokens {
            checkpoints.push(state_bhpr.clone());
        }
    }

    let y_bhtp = F::stack::<4>(outputs, 2);
    let checkpoints_bhnpr = F::stack::<5>(checkpoints, 2);
    F::cat(
        vec![
            y_bhtp.reshape([batch, nheads, tokens * per_head_dim]),
            checkpoints_bhnpr.reshape([
                batch,
                nheads,
                checkpoint_count * per_head_dim * state_rank,
            ]),
        ],
        2,
    )
}

#[allow(clippy::too_many_arguments)]
fn single_ssd_scan_backward<B: Backend>(
    v_bnl1hp: F<B, 6>,
    da_bnlh: F<B, 4>,
    b_bnl1hr: F<B, 6>,
    c_bnl1hr: F<B, 6>,
    gamma_bnlh: F<B, 4>,
    scale_bnlh: F<B, 4>,
    packed_bh_tnpr: F<B, 3>,
    d_packed_bh_tnpr: F<B, 3>,
) -> (
    F<B, 6>,
    F<B, 4>,
    F<B, 6>,
    F<B, 6>,
    F<B, 4>,
    F<B, 4>,
    F<B, 4>,
) {
    let [batch, nchunks, chunk_len, _, nheads, per_head_dim] = v_bnl1hp.dims();
    let [.., state_rank] = b_bnl1hr.dims();
    let tokens = nchunks * chunk_len;
    let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
    let v_bthp = v_bnl1hp.reshape([batch, tokens, nheads, per_head_dim]);
    let b_bthr = b_bnl1hr.reshape([batch, tokens, nheads, state_rank]);
    let c_bthr = c_bnl1hr.reshape([batch, tokens, nheads, state_rank]);
    let da_bth = da_bnlh.reshape([batch, tokens, nheads]);
    let gamma_bth = gamma_bnlh.reshape([batch, tokens, nheads]);
    let scale_bth = scale_bnlh.reshape([batch, tokens, nheads]);
    let d_y_bhpt = d_packed_bh_tnpr
        .clone()
        .narrow(2, 0, tokens * per_head_dim)
        .reshape([batch, nheads, tokens, per_head_dim])
        .permute([0, 1, 3, 2]);
    let final_checkpoint_offset =
        tokens * per_head_dim + (checkpoint_count - 1) * per_head_dim * state_rank;
    let mut state_post_bhpr = packed_bh_tnpr
        .clone()
        .narrow(2, final_checkpoint_offset, per_head_dim * state_rank)
        .reshape([batch, nheads, per_head_dim, state_rank]);
    let mut g_post_bhpr = d_packed_bh_tnpr
        .narrow(2, final_checkpoint_offset, per_head_dim * state_rank)
        .reshape([batch, nheads, per_head_dim, state_rank]);
    let mut d_v_rev = Vec::with_capacity(tokens);
    let mut d_b_rev = Vec::with_capacity(tokens);
    let mut d_c_rev = Vec::with_capacity(tokens);
    let mut d_da_rev = Vec::with_capacity(tokens);
    let mut d_gamma_rev = Vec::with_capacity(tokens);
    let mut d_scale_rev = Vec::with_capacity(tokens);

    for token in (0..tokens).rev() {
        if (token + 1).is_multiple_of(RECONSTRUCTION_INTERVAL) || token + 1 == tokens {
            let checkpoint = token / RECONSTRUCTION_INTERVAL;
            state_post_bhpr = packed_bh_tnpr
                .clone()
                .narrow(
                    2,
                    tokens * per_head_dim + checkpoint * per_head_dim * state_rank,
                    per_head_dim * state_rank,
                )
                .reshape([batch, nheads, per_head_dim, state_rank]);
        }
        let v_bhp = v_bthp.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let b_bhr = b_bthr.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let c_bhr = c_bthr.clone().narrow(1, token, 1).squeeze_dim::<3>(1);
        let da_bh = da_bth.clone().narrow(1, token, 1).squeeze_dim::<2>(1);
        let gamma_bh = gamma_bth.clone().narrow(1, token, 1).squeeze_dim::<2>(1);
        let scale_bh = scale_bth.clone().narrow(1, token, 1).squeeze_dim::<2>(1);
        let dy_bhp = d_y_bhpt.clone().narrow(3, token, 1).squeeze_dim::<3>(3);
        let v_bhp1 = v_bhp.clone().unsqueeze_dim::<4>(3);
        let b_bh1r = b_bhr.clone().unsqueeze_dim::<4>(2);
        let c_bh1r = c_bhr.clone().unsqueeze_dim::<4>(2);
        let injection_bhpr = v_bhp1.clone() * b_bh1r.clone();
        let scale_bh11 = scale_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
        let gamma_bh11 = gamma_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
        let state_pre_bhpr = state_post_bhpr - scale_bh11 * injection_bhpr.clone();
        let g_pre_bhpr =
            g_post_bhpr.clone() + dy_bhp.clone().unsqueeze_dim::<4>(3) * c_bh1r.clone();

        let dy_v_bh = (dy_bhp.clone() * v_bhp.clone())
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let cb_bh = (c_bhr.clone() * b_bhr.clone())
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let d_v_bhp = scale_bh.clone().unsqueeze_dim::<3>(2)
            * (g_post_bhpr.clone() * b_bh1r.clone())
                .sum_dim(3)
                .squeeze_dim::<3>(3)
            + gamma_bh.clone().unsqueeze_dim::<3>(2)
                * dy_bhp.clone()
                * cb_bh.clone().unsqueeze_dim::<3>(2);
        let d_b_bhr = scale_bh.clone().unsqueeze_dim::<3>(2)
            * (g_post_bhpr.clone() * v_bhp1.clone())
                .sum_dim(2)
                .squeeze_dim::<3>(2)
            + gamma_bh.clone().unsqueeze_dim::<3>(2)
                * c_bhr.clone()
                * dy_v_bh.clone().unsqueeze_dim::<3>(2);
        let d_c_bhr = (dy_bhp.clone().unsqueeze_dim::<4>(3)
            * (state_pre_bhpr.clone() + gamma_bh11 * injection_bhpr.clone()))
        .sum_dim(2)
        .squeeze_dim::<3>(2);
        let d_scale_bh = (g_post_bhpr.clone() * injection_bhpr.clone())
            .sum_dim(3)
            .sum_dim(2)
            .squeeze_dim::<3>(3)
            .squeeze_dim::<2>(2);
        let d_gamma_bh = dy_v_bh * cb_bh;
        let d_da_bh = (g_pre_bhpr.clone() * state_pre_bhpr.clone())
            .sum_dim(3)
            .sum_dim(2)
            .squeeze_dim::<3>(3)
            .squeeze_dim::<2>(2);

        d_v_rev.push(d_v_bhp);
        d_b_rev.push(d_b_bhr);
        d_c_rev.push(d_c_bhr);
        d_da_rev.push(d_da_bh);
        d_gamma_rev.push(d_gamma_bh);
        d_scale_rev.push(d_scale_bh);

        let decay_bh11 = da_bh.clone().exp().unsqueeze_dims::<4>(&[2, 3]);
        state_post_bhpr = (-da_bh).exp().unsqueeze_dims::<4>(&[2, 3]) * state_pre_bhpr;
        g_post_bhpr = decay_bh11 * g_pre_bhpr;
    }

    d_v_rev.reverse();
    d_b_rev.reverse();
    d_c_rev.reverse();
    d_da_rev.reverse();
    d_gamma_rev.reverse();
    d_scale_rev.reverse();
    (
        F::stack::<4>(d_v_rev, 1).reshape([batch, nchunks, chunk_len, 1, nheads, per_head_dim]),
        F::stack::<3>(d_da_rev, 1).reshape([batch, nchunks, chunk_len, nheads]),
        F::stack::<4>(d_b_rev, 1).reshape([batch, nchunks, chunk_len, 1, nheads, state_rank]),
        F::stack::<4>(d_c_rev, 1).reshape([batch, nchunks, chunk_len, 1, nheads, state_rank]),
        F::stack::<3>(d_gamma_rev, 1).reshape([batch, nchunks, chunk_len, nheads]),
        F::stack::<3>(d_scale_rev, 1).reshape([batch, nchunks, chunk_len, nheads]),
        g_post_bhpr,
    )
}

#[backend_extension(
    Cpu: cfg(feature = "backend-cpu"),
    Cuda: cfg(feature = "backend-cuda"),
    Rocm: cfg(feature = "backend-rocm"),
    Metal: cfg(feature = "backend-metal"),
    Vulkan: cfg(feature = "backend-vulkan"),
    Wgpu: cfg(feature = "backend-wgpu"),
    WebGpu: cfg(feature = "backend-webgpu"),
    Flex: cfg(feature = "backend-flex"),
    NdArray: cfg(feature = "backend-ndarray"),
    LibTorch: cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Autodiff: cfg(feature = "autodiff"),
)]
pub trait Mamba3SingleSsdScanBackendExt: Backend {
    #[allow(clippy::too_many_arguments)]
    fn mamba3_single_ssd_scan(
        v_bnl1hp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnl1hr: FloatTensor<Self>,
        c_bnl1hr: FloatTensor<Self>,
        gamma_bnlh: FloatTensor<Self>,
        scale_bnlh: FloatTensor<Self>,
        initial_bhpr: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        single_ssd_scan_forward::<Self>(
            F::new(v_bnl1hp),
            F::new(da_bnlh),
            F::new(b_bnl1hr),
            F::new(c_bnl1hr),
            F::new(gamma_bnlh),
            F::new(scale_bnlh),
            F::new(initial_bhpr),
        )
        .inner()
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn mamba3_single_ssd_scan_backward(
        v_bnl1hp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnl1hr: FloatTensor<Self>,
        c_bnl1hr: FloatTensor<Self>,
        gamma_bnlh: FloatTensor<Self>,
        scale_bnlh: FloatTensor<Self>,
        packed_bh_tnpr: FloatTensor<Self>,
        d_packed_bh_tnpr: FloatTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let grads = single_ssd_scan_backward::<Self>(
            F::new(v_bnl1hp),
            F::new(da_bnlh),
            F::new(b_bnl1hr),
            F::new(c_bnl1hr),
            F::new(gamma_bnlh),
            F::new(scale_bnlh),
            F::new(packed_bh_tnpr),
            F::new(d_packed_bh_tnpr),
        );
        (
            grads.0.inner(),
            grads.1.inner(),
            grads.2.inner(),
            grads.3.inner(),
            grads.4.inner(),
            grads.5.inner(),
            grads.6.inner(),
        )
    }
}

#[cfg(feature = "backend-ndarray")]
impl<F, I> Mamba3SingleSsdScanBackendExt for burn::backend::NdArray<F, I> {}

#[cfg(feature = "backend-flex")]
impl Mamba3SingleSsdScanBackendExt for burn::backend::Flex {}

#[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
impl<F, I> Mamba3SingleSsdScanBackendExt for burn::backend::libtorch::LibTorch<F, I> {}

#[cfg(feature = "backend-remote")]
impl<F, I> Mamba3SingleSsdScanBackendExt for burn::backend::RemoteBackend<F, I> {}

/// Run the rank-one single-SSD recurrence as one backend operation.
#[allow(clippy::too_many_arguments)]
pub fn single_ssd_scan(
    v_bnl1hp: Tensor<6>,
    da_bnlh: Tensor<4>,
    b_bnl1hr: Tensor<6>,
    c_bnl1hr: Tensor<6>,
    gamma_bnlh: Tensor<4>,
    scale_bnlh: Tensor<4>,
    initial_bhpr: Tensor<4>,
) -> (Tensor<6>, Tensor<4>) {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnl1hp.dims();
    let [.., state_rank] = b_bnl1hr.dims();
    assert_eq!(mimo_rank, 1, "fused single-SSD scan requires MIMO rank one");
    let tokens = nchunks * chunk_len;
    let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
    let packed = Tensor::<3>::from_dispatch(
        <Dispatch as Mamba3SingleSsdScanBackendExt>::mamba3_single_ssd_scan(
            v_bnl1hp.into_dispatch(),
            da_bnlh.into_dispatch(),
            b_bnl1hr.into_dispatch(),
            c_bnl1hr.into_dispatch(),
            gamma_bnlh.into_dispatch(),
            scale_bnlh.into_dispatch(),
            initial_bhpr.into_dispatch(),
        ),
    );
    let y = packed
        .clone()
        .narrow(2, 0, tokens * per_head_dim)
        .reshape([batch, nheads, tokens, per_head_dim])
        .permute([0, 2, 1, 3])
        .reshape([batch, nchunks, chunk_len, 1, nheads, per_head_dim]);
    let final_state = packed
        .narrow(
            2,
            tokens * per_head_dim + (checkpoint_count - 1) * per_head_dim * state_rank,
            per_head_dim * state_rank,
        )
        .reshape([batch, nheads, per_head_dim, state_rank]);
    (y, final_state)
}

/// Exercise the primitive fallback without selecting a specialized backend implementation.
#[cfg(all(test, feature = "_dev-test"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn single_ssd_scan_reference(
    v_bnl1hp: Tensor<6>,
    da_bnlh: Tensor<4>,
    b_bnl1hr: Tensor<6>,
    c_bnl1hr: Tensor<6>,
    gamma_bnlh: Tensor<4>,
    scale_bnlh: Tensor<4>,
    initial_bhpr: Tensor<4>,
) -> (Tensor<6>, Tensor<4>) {
    let [batch, nchunks, chunk_len, _, nheads, per_head_dim] = v_bnl1hp.dims();
    let state_rank = b_bnl1hr.dims()[5];
    let tokens = nchunks * chunk_len;
    let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
    let packed = Tensor::<3>::from_dispatch(
        single_ssd_scan_forward::<Dispatch>(
            F::new(v_bnl1hp.into_dispatch()),
            F::new(da_bnlh.into_dispatch()),
            F::new(b_bnl1hr.into_dispatch()),
            F::new(c_bnl1hr.into_dispatch()),
            F::new(gamma_bnlh.into_dispatch()),
            F::new(scale_bnlh.into_dispatch()),
            F::new(initial_bhpr.into_dispatch()),
        )
        .inner(),
    );
    let y = packed
        .clone()
        .narrow(2, 0, tokens * per_head_dim)
        .reshape([batch, nheads, tokens, per_head_dim])
        .permute([0, 2, 1, 3])
        .reshape([batch, nchunks, chunk_len, 1, nheads, per_head_dim]);
    let final_state = packed
        .narrow(
            2,
            tokens * per_head_dim + (checkpoint_count - 1) * per_head_dim * state_rank,
            per_head_dim * state_rank,
        )
        .reshape([batch, nheads, per_head_dim, state_rank]);
    (y, final_state)
}
