//! Fusion registration for the fused rank-one single-SSD operation.

use super::{RECONSTRUCTION_INTERVAL, single_ssd_scan::Mamba3SingleSsdScanBackendExt};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Shape};
use burn_fusion::{
    Fusion, FusionBackend, FusionRuntime,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use core::marker::PhantomData;

#[derive(Clone, Debug)]
struct SingleSsdScanForward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B: FusionBackend + Mamba3SingleSsdScanBackendExt> Operation<B::FusionRuntime>
    for SingleSsdScanForward<B>
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([v, da, b, c, gamma, scale, initial], [packed]) = self.desc.as_fixed();
        let result = B::mamba3_single_ssd_scan(
            handles.get_float_tensor::<B>(v),
            handles.get_float_tensor::<B>(da),
            handles.get_float_tensor::<B>(b),
            handles.get_float_tensor::<B>(c),
            handles.get_float_tensor::<B>(gamma),
            handles.get_float_tensor::<B>(scale),
            handles.get_float_tensor::<B>(initial),
        );
        handles.register_float_tensor::<B>(&packed.id, result);
    }
}

#[derive(Clone, Debug)]
struct SingleSsdScanBackward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B: FusionBackend + Mamba3SingleSsdScanBackendExt> Operation<B::FusionRuntime>
    for SingleSsdScanBackward<B>
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let (
            [v, da, b, c, gamma, scale, packed, d_packed],
            [d_v, d_da, d_b, d_c, d_gamma, d_scale, d_initial],
        ) = self.desc.as_fixed();
        let results = B::mamba3_single_ssd_scan_backward(
            handles.get_float_tensor::<B>(v),
            handles.get_float_tensor::<B>(da),
            handles.get_float_tensor::<B>(b),
            handles.get_float_tensor::<B>(c),
            handles.get_float_tensor::<B>(gamma),
            handles.get_float_tensor::<B>(scale),
            handles.get_float_tensor::<B>(packed),
            handles.get_float_tensor::<B>(d_packed),
        );
        handles.register_float_tensor::<B>(&d_v.id, results.0);
        handles.register_float_tensor::<B>(&d_da.id, results.1);
        handles.register_float_tensor::<B>(&d_b.id, results.2);
        handles.register_float_tensor::<B>(&d_c.id, results.3);
        handles.register_float_tensor::<B>(&d_gamma.id, results.4);
        handles.register_float_tensor::<B>(&d_scale.id, results.5);
        handles.register_float_tensor::<B>(&d_initial.id, results.6);
    }
}

impl<B: FusionBackend + Mamba3SingleSsdScanBackendExt> Mamba3SingleSsdScanBackendExt for Fusion<B> {
    fn mamba3_single_ssd_scan(
        v_bnl1hp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnl1hr: FloatTensor<Self>,
        c_bnl1hr: FloatTensor<Self>,
        gamma_bnlh: FloatTensor<Self>,
        scale_bnlh: FloatTensor<Self>,
        initial_bhpr: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let [batch, nchunks, chunk_len, _, nheads, per_head_dim] = v_bnl1hp.shape.dims::<6>();
        let state_rank = b_bnl1hr.shape.dims::<6>()[5];
        let tokens = nchunks * chunk_len;
        let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
        let client = v_bnl1hp.client.clone();
        let packed = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([
                batch,
                nheads,
                per_head_dim * (tokens + checkpoint_count * state_rank),
            ]),
            v_bnl1hp.dtype,
        );
        let desc = CustomOpIr::new(
            "mamba3_single_ssd_scan_forward",
            &[
                v_bnl1hp.into_ir(),
                da_bnlh.into_ir(),
                b_bnl1hr.into_ir(),
                c_bnl1hr.into_ir(),
                gamma_bnlh.into_ir(),
                scale_bnlh.into_ir(),
                initial_bhpr.into_ir(),
            ],
            &[packed],
        );
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                SingleSsdScanForward::<B> {
                    desc,
                    backend: PhantomData,
                },
            )
            .output()
    }

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
        let client = v_bnl1hp.client.clone();
        let dtype = v_bnl1hp.dtype;
        let d_v = TensorIr::uninit(client.create_empty_handle(), v_bnl1hp.shape.clone(), dtype);
        let d_da = TensorIr::uninit(client.create_empty_handle(), da_bnlh.shape.clone(), dtype);
        let d_b = TensorIr::uninit(client.create_empty_handle(), b_bnl1hr.shape.clone(), dtype);
        let d_c = TensorIr::uninit(client.create_empty_handle(), c_bnl1hr.shape.clone(), dtype);
        let d_gamma = TensorIr::uninit(
            client.create_empty_handle(),
            gamma_bnlh.shape.clone(),
            dtype,
        );
        let d_scale = TensorIr::uninit(
            client.create_empty_handle(),
            scale_bnlh.shape.clone(),
            dtype,
        );
        let [batch, _, _, _, nheads, per_head_dim] = v_bnl1hp.shape.dims::<6>();
        let state_rank = b_bnl1hr.shape.dims::<6>()[5];
        let d_initial = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, nheads, per_head_dim, state_rank]),
            dtype,
        );
        let desc = CustomOpIr::new(
            "mamba3_single_ssd_scan_backward",
            &[
                v_bnl1hp.into_ir(),
                da_bnlh.into_ir(),
                b_bnl1hr.into_ir(),
                c_bnl1hr.into_ir(),
                gamma_bnlh.into_ir(),
                scale_bnlh.into_ir(),
                packed_bh_tnpr.into_ir(),
                d_packed_bh_tnpr.into_ir(),
            ],
            &[d_v, d_da, d_b, d_c, d_gamma, d_scale, d_initial],
        );
        let [d_v, d_da, d_b, d_c, d_gamma, d_scale, d_initial] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                SingleSsdScanBackward::<B> {
                    desc,
                    backend: PhantomData,
                },
            )
            .try_into()
            .expect("single SSD scan backward registers seven outputs");
        (d_v, d_da, d_b, d_c, d_gamma, d_scale, d_initial)
    }
}
