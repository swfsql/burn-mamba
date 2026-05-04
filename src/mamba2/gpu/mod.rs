// TODO: remove
#![allow(unused_variables)]
#![allow(non_snake_case)]

pub mod k1_ssd_chunk_cumsum;
pub mod k4_ssd_state_passing;
pub mod k5_ssd_chunk_scan;
pub mod naive;

use burn::tensor::backend::{AutodiffBackend, Backend};

// Currently set as a conservative default for CUDA/WebGPU/Metal.
//
// TODO: decide how to generally and properly fetch and react to the client's settings.
pub const USUAL_PLANE_DIM: u32 = 32;

pub trait BackendExt: Backend
where
    Self: k1_ssd_chunk_cumsum::backend_ext::BackendExt
        + k4_ssd_state_passing::backend_ext::BackendExt
        + k5_ssd_chunk_scan::backend_ext::BackendExt,
{
}
impl<T> BackendExt for T where
    T: Backend
        + k1_ssd_chunk_cumsum::backend_ext::BackendExt
        + k4_ssd_state_passing::backend_ext::BackendExt
        + k5_ssd_chunk_scan::backend_ext::BackendExt
{
}

// TODO: after backwards is implemented, check if the bounds requirements are correct.
// i.e. they may require for than necessary.
pub trait AutodiffBackendExt: AutodiffBackend
where
    Self: k1_ssd_chunk_cumsum::backend_ext::BackendExt
        + k4_ssd_state_passing::backend_ext::BackendExt
        + k5_ssd_chunk_scan::backend_ext::BackendExt,
    <Self as AutodiffBackend>::InnerBackend: k1_ssd_chunk_cumsum::backend_ext::BackendExt
        + k4_ssd_state_passing::backend_ext::BackendExt
        + k5_ssd_chunk_scan::backend_ext::BackendExt,
{
}

impl<T> AutodiffBackendExt for T
where
    T: AutodiffBackend
        + k1_ssd_chunk_cumsum::backend_ext::BackendExt
        + k4_ssd_state_passing::backend_ext::BackendExt
        + k5_ssd_chunk_scan::backend_ext::BackendExt,
    <T as AutodiffBackend>::InnerBackend: k1_ssd_chunk_cumsum::backend_ext::BackendExt
        + k4_ssd_state_passing::backend_ext::BackendExt
        + k5_ssd_chunk_scan::backend_ext::BackendExt,
{
}
