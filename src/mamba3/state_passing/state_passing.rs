//! Backend extension for the serial SSD K1 and K4 scans.

use crate::utils::fprim::F;
use burn::backend::tensor::FloatTensor;
use burn::backend::*;
use burn::backend::{Dispatch, backend_extension};
use burn::tensor::Tensor;

/// Primitive reference implementation of the per-chunk prefix sum.
fn chunk_cumsum_forward<B: Backend>(da_bnlh: F<B, 4>) -> F<B, 4> {
    let [batch, nchunks, chunk_len, nheads] = da_bnlh.dims();
    let prefix_bhnl = da_bnlh.permute([0, 3, 1, 2]).cumsum(3);
    assert_eq!([batch, nheads, nchunks, chunk_len], prefix_bhnl.dims());
    prefix_bhnl
}

/// Exact VJP of [`chunk_cumsum_forward`].
fn chunk_cumsum_backward<B: Backend>(d_prefix_bhnl: F<B, 4>) -> F<B, 4> {
    d_prefix_bhnl
        .flip(&[3])
        .cumsum(3)
        .flip(&[3])
        .permute([0, 2, 3, 1])
}

/// Primitive reference implementation of the forward recurrence.
fn state_passing_forward<B: Backend>(
    intra_bnhpr: F<B, 5>,
    decay_bhn: F<B, 3>,
    initial_bhpr: F<B, 4>,
) -> F<B, 5> {
    let [batch, nchunks, nheads, per_head_dim, state_rank] = intra_bnhpr.dims();
    assert!(nchunks > 0, "state passing requires at least one chunk");
    assert_eq!([batch, nheads, nchunks], decay_bhn.dims());
    assert_eq!(
        [batch, nheads, per_head_dim, state_rank],
        initial_bhpr.dims()
    );

    let mut running_bhpr = initial_bhpr;
    let mut states = Vec::with_capacity(nchunks + 1);
    states.push(running_bhpr.clone());

    for chunk in 0..nchunks {
        let intra_bhpr = intra_bnhpr.clone().narrow(1, chunk, 1).squeeze_dim::<4>(1);
        let decay_bhpr = decay_bhn
            .clone()
            .narrow(2, chunk, 1)
            .squeeze_dim::<2>(2)
            .unsqueeze_dims::<4>(&[2, 3])
            .expand([batch, nheads, per_head_dim, state_rank]);
        running_bhpr = decay_bhpr * running_bhpr + intra_bhpr;
        states.push(running_bhpr.clone());
    }

    F::stack::<5>(states, 1)
}

/// Exact VJP of [`state_passing_forward`].
fn state_passing_backward<B: Backend>(
    states_bn1hpr: F<B, 5>,
    decay_bhn: F<B, 3>,
    d_states_bn1hpr: F<B, 5>,
) -> (F<B, 5>, F<B, 3>, F<B, 4>) {
    let [batch, states_len, nheads, per_head_dim, state_rank] = states_bn1hpr.dims();
    let nchunks = states_len - 1;
    assert!(nchunks > 0, "state passing requires at least one chunk");

    let mut g_bhpr = d_states_bn1hpr
        .clone()
        .narrow(1, nchunks, 1)
        .squeeze_dim::<4>(1);
    let mut d_intra_rev = Vec::with_capacity(nchunks);
    let mut d_decay_rev = Vec::with_capacity(nchunks);

    for chunk in (0..nchunks).rev() {
        let state_before_bhpr = states_bn1hpr
            .clone()
            .narrow(1, chunk, 1)
            .squeeze_dim::<4>(1);
        let d_decay_bh = (g_bhpr.clone() * state_before_bhpr)
            .sum_dim(3)
            .sum_dim(2)
            .squeeze_dim::<3>(3)
            .squeeze_dim::<2>(2);
        d_decay_rev.push(d_decay_bh);
        d_intra_rev.push(g_bhpr.clone());

        let decay_bhpr = decay_bhn
            .clone()
            .narrow(2, chunk, 1)
            .squeeze_dim::<2>(2)
            .unsqueeze_dims::<4>(&[2, 3])
            .expand([batch, nheads, per_head_dim, state_rank]);
        let direct_bhpr = d_states_bn1hpr
            .clone()
            .narrow(1, chunk, 1)
            .squeeze_dim::<4>(1);
        g_bhpr = direct_bhpr + decay_bhpr * g_bhpr;
    }

    d_intra_rev.reverse();
    d_decay_rev.reverse();
    (
        F::stack::<5>(d_intra_rev, 1),
        F::stack::<3>(d_decay_rev, 2),
        g_bhpr,
    )
}

/// Backend operations for the serial SSD scan primitives.
///
/// K4 returns the initial state at index zero and every post-chunk state after
/// it: `[batch, nchunks + 1, heads, p, r]`.
#[backend_extension(
    Cpu:  cfg(feature = "backend-cpu"),
    Cuda: cfg(feature = "backend-cuda"),
    Rocm:  cfg(feature = "backend-rocm"),
    Metal:  cfg(feature = "backend-metal"),
    Vulkan:  cfg(feature = "backend-vulkan"),
    Wgpu:  cfg(feature = "backend-wgpu"),
    WebGpu:  cfg(feature = "backend-webgpu"),
    Flex:  cfg(feature = "backend-flex"),
    NdArray:  cfg(feature = "backend-ndarray"),
    LibTorch:  cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Autodiff:  cfg(feature = "autodiff"),
)]
pub trait Mamba3StatePassingBackendExt: Backend {
    /// Compute prefix sums independently inside every chunk.
    fn mamba3_chunk_cumsum(da_bnlh: FloatTensor<Self>) -> FloatTensor<Self> {
        chunk_cumsum_forward::<Self>(F::new(da_bnlh)).inner()
    }

    /// Run the suffix-sum VJP for the chunk cumulative sum.
    fn mamba3_chunk_cumsum_backward(d_prefix_bhnl: FloatTensor<Self>) -> FloatTensor<Self> {
        chunk_cumsum_backward::<Self>(F::new(d_prefix_bhnl)).inner()
    }

    /// Run the forward recurrence and return all boundary states.
    fn mamba3_state_passing(
        intra_bnhpr: FloatTensor<Self>,
        decay_bhn: FloatTensor<Self>,
        initial_bhpr: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        state_passing_forward::<Self>(F::new(intra_bnhpr), F::new(decay_bhn), F::new(initial_bhpr))
            .inner()
    }

    /// Run the exact reverse recurrence for the custom autodiff node.
    fn mamba3_state_passing_backward(
        states_bn1hpr: FloatTensor<Self>,
        decay_bhn: FloatTensor<Self>,
        d_states_bn1hpr: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (d_intra, d_decay, d_initial) = state_passing_backward::<Self>(
            F::new(states_bn1hpr),
            F::new(decay_bhn),
            F::new(d_states_bn1hpr),
        );
        (d_intra.inner(), d_decay.inner(), d_initial.inner())
    }
}

#[cfg(feature = "backend-ndarray")]
impl<F, I> Mamba3StatePassingBackendExt for burn::backend::NdArray<F, I> {}

#[cfg(feature = "backend-flex")]
impl Mamba3StatePassingBackendExt for burn::backend::Flex {}

#[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
impl<F, I> Mamba3StatePassingBackendExt for burn::backend::libtorch::LibTorch<F, I> {}

#[cfg(feature = "backend-remote")]
impl<F, I> Mamba3StatePassingBackendExt for burn::backend::RemoteBackend<F, I> {}

/// Run the backend state-passing operation from high-level tensor code.
pub fn state_passing(
    intra_bnhpr: Tensor<5>,
    decay_bhn: Tensor<3>,
    initial_bhpr: Tensor<4>,
) -> Tensor<5> {
    Tensor::from_dispatch(
        <Dispatch as Mamba3StatePassingBackendExt>::mamba3_state_passing(
            intra_bnhpr.into_dispatch(),
            decay_bhn.into_dispatch(),
            initial_bhpr.into_dispatch(),
        ),
    )
}

/// Run the backend chunk-cumsum operation from high-level tensor code.
pub fn chunk_cumsum(da_bnlh: Tensor<4>) -> Tensor<4> {
    Tensor::from_dispatch(
        <Dispatch as Mamba3StatePassingBackendExt>::mamba3_chunk_cumsum(da_bnlh.into_dispatch()),
    )
}
