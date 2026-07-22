//! Fusion registration for the fused Mamba-3 K1 and K4 backend operations.

use super::state_passing::Mamba3StatePassingBackendExt;
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Shape};
use burn_fusion::{
    Fusion, FusionBackend, FusionRuntime,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use core::marker::PhantomData;

#[derive(Clone, Debug)]
struct ChunkCumsumForward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B: FusionBackend + Mamba3StatePassingBackendExt> Operation<B::FusionRuntime>
    for ChunkCumsumForward<B>
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([da], [prefix]) = self.desc.as_fixed();
        let result = B::mamba3_chunk_cumsum(handles.get_float_tensor::<B>(da));
        handles.register_float_tensor::<B>(&prefix.id, result);
    }
}

#[derive(Clone, Debug)]
struct ChunkCumsumBackward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B: FusionBackend + Mamba3StatePassingBackendExt> Operation<B::FusionRuntime>
    for ChunkCumsumBackward<B>
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([d_prefix], [d_da]) = self.desc.as_fixed();
        let result = B::mamba3_chunk_cumsum_backward(handles.get_float_tensor::<B>(d_prefix));
        handles.register_float_tensor::<B>(&d_da.id, result);
    }
}

#[derive(Clone, Debug)]
struct StatePassingForward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B: FusionBackend + Mamba3StatePassingBackendExt> Operation<B::FusionRuntime>
    for StatePassingForward<B>
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([intra, decay, initial], [output]) = self.desc.as_fixed();
        let result = B::mamba3_state_passing(
            handles.get_float_tensor::<B>(intra),
            handles.get_float_tensor::<B>(decay),
            handles.get_float_tensor::<B>(initial),
        );
        handles.register_float_tensor::<B>(&output.id, result);
    }
}

#[derive(Clone, Debug)]
struct StatePassingBackward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B: FusionBackend + Mamba3StatePassingBackendExt> Operation<B::FusionRuntime>
    for StatePassingBackward<B>
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([states, decay, d_states], [d_intra, d_decay, d_initial]) = self.desc.as_fixed();
        let (intra_result, decay_result, initial_result) = B::mamba3_state_passing_backward(
            handles.get_float_tensor::<B>(states),
            handles.get_float_tensor::<B>(decay),
            handles.get_float_tensor::<B>(d_states),
        );
        handles.register_float_tensor::<B>(&d_intra.id, intra_result);
        handles.register_float_tensor::<B>(&d_decay.id, decay_result);
        handles.register_float_tensor::<B>(&d_initial.id, initial_result);
    }
}

impl<B: FusionBackend + Mamba3StatePassingBackendExt> Mamba3StatePassingBackendExt for Fusion<B> {
    fn mamba3_chunk_cumsum(da_bnlh: FloatTensor<Self>) -> FloatTensor<Self> {
        let [batch, nchunks, chunk_len, nheads] = da_bnlh.shape.dims::<4>();
        let client = da_bnlh.client.clone();
        let prefix = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, nheads, nchunks, chunk_len]),
            da_bnlh.dtype,
        );
        let desc = CustomOpIr::new(
            "mamba3_chunk_cumsum_forward",
            &[da_bnlh.into_ir()],
            &[prefix],
        );
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                ChunkCumsumForward::<B> {
                    desc,
                    backend: PhantomData,
                },
            )
            .output()
    }

    fn mamba3_chunk_cumsum_backward(d_prefix_bhnl: FloatTensor<Self>) -> FloatTensor<Self> {
        let [batch, nheads, nchunks, chunk_len] = d_prefix_bhnl.shape.dims::<4>();
        let client = d_prefix_bhnl.client.clone();
        let d_da = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, nchunks, chunk_len, nheads]),
            d_prefix_bhnl.dtype,
        );
        let desc = CustomOpIr::new(
            "mamba3_chunk_cumsum_backward",
            &[d_prefix_bhnl.into_ir()],
            &[d_da],
        );
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                ChunkCumsumBackward::<B> {
                    desc,
                    backend: PhantomData,
                },
            )
            .output()
    }

    fn mamba3_state_passing(
        intra_bnhpr: FloatTensor<Self>,
        decay_bhn: FloatTensor<Self>,
        initial_bhpr: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let [batch, nchunks, nheads, per_head_dim, state_rank] = intra_bnhpr.shape.dims::<5>();
        let client = intra_bnhpr.client.clone();
        let output = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, nchunks + 1, nheads, per_head_dim, state_rank]),
            intra_bnhpr.dtype,
        );
        let desc = CustomOpIr::new(
            "mamba3_state_passing_forward",
            &[
                intra_bnhpr.into_ir(),
                decay_bhn.into_ir(),
                initial_bhpr.into_ir(),
            ],
            &[output],
        );
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                StatePassingForward::<B> {
                    desc,
                    backend: PhantomData,
                },
            )
            .output()
    }

    fn mamba3_state_passing_backward(
        states_bn1hpr: FloatTensor<Self>,
        decay_bhn: FloatTensor<Self>,
        d_states_bn1hpr: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let [batch, states_len, nheads, per_head_dim, state_rank] = states_bn1hpr.shape.dims::<5>();
        let nchunks = states_len - 1;
        let client = states_bn1hpr.client.clone();
        let dtype = states_bn1hpr.dtype;
        let d_intra = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, nchunks, nheads, per_head_dim, state_rank]),
            dtype,
        );
        let d_decay = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, nheads, nchunks]),
            dtype,
        );
        let d_initial = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, nheads, per_head_dim, state_rank]),
            dtype,
        );
        let desc = CustomOpIr::new(
            "mamba3_state_passing_backward",
            &[
                states_bn1hpr.into_ir(),
                decay_bhn.into_ir(),
                d_states_bn1hpr.into_ir(),
            ],
            &[d_intra, d_decay, d_initial],
        );
        let [d_intra, d_decay, d_initial] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                StatePassingBackward::<B> {
                    desc,
                    backend: PhantomData,
                },
            )
            .try_into()
            .expect("state passing backward registers three outputs");
        (d_intra, d_decay, d_initial)
    }
}
