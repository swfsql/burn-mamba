/// Extends the backend (for `burn_cubecl`) and wraps it for `burn`.
pub mod backend_ext {
    use burn::tensor::{Tensor, TensorPrimitive, ops::FloatTensor};

    // Wraps the extended backend for `burn`.
    //
    /// Returns `da_cumsum_bhnl`.
    #[cfg(feature = "cubecl")]
    pub fn k1_ssd_chunk_cumsum<B: BackendExt>(
        dt_discretized_bhnl: Tensor<B, 4>,
        a_decay_h: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let da_cumsum_bhnl = B::k1_ssd_chunk_cumsum(
            dt_discretized_bhnl.into_primitive().tensor(),
            a_decay_h.into_primitive().tensor(),
        );

        Tensor::from_primitive(TensorPrimitive::Float(da_cumsum_bhnl))
    }

    // Wraps the extended backend for `burn` (naive version).
    //
    /// Returns `da_cumsum_bhnl`.
    pub fn k1_ssd_chunk_cumsum_naive<B: BackendExt>(
        dt_discretized_bhnl: Tensor<B, 4>,
        a_decay_h: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let da_cumsum_bhnl = B::k1_ssd_chunk_cumsum_naive(
            dt_discretized_bhnl.into_primitive().tensor(),
            a_decay_h.into_primitive().tensor(),
        );

        Tensor::from_primitive(TensorPrimitive::Float(da_cumsum_bhnl))
    }

    // Backend extended with a new op.
    pub trait BackendExt: burn::tensor::backend::Backend {
        /// Returns `da_cumsum_bhnl`.
        #[cfg(feature = "cubecl")]
        fn k1_ssd_chunk_cumsum(
            dt_discretized_bhnl: FloatTensor<Self>,
            a_decay_h: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            unimplemented!("Backend not supported")
        }

        // Naive fallback.
        /// Returns `da_cumsum_bhnl`.
        fn k1_ssd_chunk_cumsum_naive(
            dt_discretized_bhnl: FloatTensor<Self>,
            a_decay_h: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            unimplemented!("Backend not supported")
        }
    }
    // Note: non-cubecl devices should hit the unimpl message.

    // Backend extended with a new op.
    #[cfg(feature = "autodiff")]
    pub trait AutodiffBackendExt: BackendExt + burn::tensor::backend::AutodiffBackend {}
}
#[cfg(feature = "autodiff")]
pub use backend_ext::AutodiffBackendExt;
pub use backend_ext::BackendExt;
#[cfg(feature = "cubecl")]
pub use backend_ext::k1_ssd_chunk_cumsum;

/// Backend implementation.
///
/// Wraps the `cubecl` kernel for `burn_cubecl`.
#[cfg(feature = "cubecl")]
pub mod forward {
    use super::{BackendExt, kernel};
    use burn::cubecl::{CubeCount, CubeDim};
    use burn::tensor::{Shape, ops::FloatTensor};
    use burn_cubecl::{
        CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement,
        kernel::into_contiguous, tensor::CubeTensor,
    };

    // Implement our custom backend trait for the generic `CubeBackend`.
    //
    // TODO: see if burn/crates/burn-vision/src/backends/cube/connected_components/prefix_sum.rs
    // can be used here
    impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> BackendExt
        for CubeBackend<R, F, I, BT>
    {
        /// Returns `da_cumsum_bhnl`.
        fn k1_ssd_chunk_cumsum(
            dt_discretized_bhnl: FloatTensor<Self>,
            a_decay_h: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            // Note: `burn::tensor::ops::FloatTensor` is concretized to `burn::cubecl::tensor::CubeTensor`.

            let client = dt_discretized_bhnl.client.clone();
            let device = dt_discretized_bhnl.device.clone();
            dt_discretized_bhnl.assert_is_on_same_device(&a_decay_h);

            // For simplicity, make sure each tensor is contiguous.
            let dt_discretized_bhnl = into_contiguous(dt_discretized_bhnl);
            let a_decay_h = into_contiguous(a_decay_h);

            // Get relevant shapes.
            let [batch, nheads, nchunks, chunk_len] = dt_discretized_bhnl.meta.shape().dims();

            // Create output tensor.
            let da_cumsum_shape_bhnl = Shape::from([batch, nheads, nchunks, chunk_len]);
            let da_cumsum_buffer_bhnl =
                client.empty(da_cumsum_shape_bhnl.num_elements() * core::mem::size_of::<F>());
            let da_cumsum_bhnl = CubeTensor::new_contiguous(
                client.clone(),
                device.clone(),
                da_cumsum_shape_bhnl,
                da_cumsum_buffer_bhnl,
                F::dtype(),
            );

            // Compute number of planes per cube.
            let num_planes = {
                use crate::mamba2::gpu::USUAL_PLANE_DIM; // 32
                assert_eq!(
                    chunk_len as u32 % USUAL_PLANE_DIM,
                    0,
                    "only filled lanes are supported. chunk_len % USUAL_PLANE_DIM = {chunk_len} % {USUAL_PLANE_DIM} != 0"
                );
                assert!(
                    chunk_len as u32 <= USUAL_PLANE_DIM * USUAL_PLANE_DIM,
                    "chunk_len ({}) exceeds supported limit ({}). Current implementation requires chunk_len <= USUAL_PLANE_DIM * USUAL_PLANE_DIM",
                    chunk_len,
                    USUAL_PLANE_DIM * USUAL_PLANE_DIM
                );
                chunk_len as u32 / USUAL_PLANE_DIM
            };

            unsafe {
                kernel::forward::launch_unchecked::<F, R>(
                    &client,
                    CubeCount::Static(batch as u32, nheads as u32, nchunks as u32),
                    CubeDim::new_1d(chunk_len as u32),
                    dt_discretized_bhnl.into_tensor_arg(),
                    a_decay_h.into_tensor_arg(),
                    da_cumsum_bhnl.clone().into_tensor_arg(),
                    num_planes, // jit time
                )
            };

            da_cumsum_bhnl
        }
    }
}
#[cfg(feature = "backend-ndarray")]
impl<F, I> BackendExt for burn::backend::NdArray<F, I> {}
#[cfg(feature = "backend-flex")]
impl BackendExt for burn::backend::Flex {}
#[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
impl<F, I> BackendExt for burn::backend::libtorch::LibTorch<F, I> {}
#[cfg(feature = "backend-remote")]
impl<F, I> BackendExt for burn::backend::RemoteBackend<F, I> {}
// TODO: avoid leaking backend-* features into the library

#[cfg(feature = "autodiff")]
pub mod backward {
    use super::{AutodiffBackendExt, BackendExt};
    use burn::backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};
    use burn::prelude::*;
    use burn::tensor::ops::FloatTensor;
    use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement};

    impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> AutodiffBackendExt
        for Autodiff<CubeBackend<R, F, I, BT>>
    {
    }

    impl<B: Backend + BackendExt, C: CheckpointStrategy> BackendExt for Autodiff<B, C> {
        fn k1_ssd_chunk_cumsum(
            dt_discretized_bhnl: FloatTensor<Self>,
            a_decay_h: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            unimplemented!("AutodiffBackend not supported")
        }

        // Naive fallback.
        fn k1_ssd_chunk_cumsum_naive(
            dt_discretized_bhnl: FloatTensor<Self>,
            a_decay_h: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            unimplemented!("AutodiffBackend not supported")
        }
    }

    // blank impl for non-cubecl backends
    #[cfg(feature = "backend-ndarray")]
    impl<F, I> AutodiffBackendExt for Autodiff<burn::backend::NdArray<F, I>> {}
    #[cfg(feature = "backend-flex")]
    impl AutodiffBackendExt for Autodiff<burn::backend::Flex> {}
    #[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
    impl<F, I> AutodiffBackendExt for Autodiff<burn::backend::libtorch::LibTorch<F, I>> {}
    #[cfg(feature = "backend-remote")]
    impl<F, I> AutodiffBackendExt for Autodiff<burn::backend::RemoteBackend<F, I>> {}
    // TODO: avoid leaking backend-* features into the library
}

/// Kernel implementation in `cubecl`.
#[cfg(feature = "cubecl")]
pub mod kernel {
    use burn::cubecl;
    use cubecl::prelude::*;

    /// # Shapes
    /// - `dt_discretized_bhnl`   : `[batch, nheads, nchunks, chunk_len]`
    /// - `a_decay_h`             : `[nheads]`
    /// - `da_cumsum_bhnl`        : `[batch, nheads, nchunks, chunk_len]`
    #[cube(launch_unchecked)]
    pub fn forward<F: Float>(
        dt_discretized_bhnl: &Tensor<F>,
        a_decay_h: &Tensor<F>,
        da_cumsum_bhnl: &mut Tensor<F>,
        #[comptime] num_planes: u32, // jit time
    ) {
        // Note: all plane operations require filled planes.
        // Operation examples: plane_exclusive_sum, plane_shuffle.

        let dt = dt_discretized_bhnl;
        let a = a_decay_h;
        let num_planes = num_planes as usize;

        let batch_idx = CUBE_POS_X as usize; // 0..batch
        let nheads_idx = CUBE_POS_Y as usize; // 0..nheads
        let nchunks_idx = CUBE_POS_Z as usize; // 0..nchunks
        let thread_id = UNIT_POS_X as usize; // 0..chunk_len

        let dt_batch_stride = dt.stride(0);
        let dt_nheads_stride = dt.stride(1);
        let dt_nchunks_stride = dt.stride(2);
        let dt_chunk_len_stride = dt.stride(3);
        let a_nheads_stride = a.stride(0);

        // Base linear offset for the current (batch, head, chunk) slice.
        let base_pos = batch_idx * dt_batch_stride
            + nheads_idx * dt_nheads_stride
            + nchunks_idx * dt_nchunks_stride;

        // Scalar value a[h] (identical for all units in this cube).
        let a_offset = nheads_idx * a_nheads_stride;
        let a_1 = a[a_offset];

        // Local position within the chunk
        let plane_idx = thread_id / PLANE_DIM as usize; // 0..num_planes
        let plane_thread_idx = UNIT_POS_PLANE as usize; // 0..PLANE_DIM

        // Position of the element processed by this unit.
        let my_pos = base_pos + (thread_id as usize) * dt_chunk_len_stride;

        // Per-position increment for the prefix sum.
        // eg. increment        = 01 02 03 04 ; 05 06 07 08 ; 09 10 11 12 ; 013 014 015 016
        let increment = dt[my_pos] * a_1;

        // Shared memory for plane totals / offsets (size known at jit time).
        let mut shared_totals = SharedMemory::<F>::new(num_planes as usize);

        // 1. Exclusive prefix sum within the current plane.
        // e.g. local_scan       = 00 01 03 06 ; 00 05 11 18 ; 00 09 19 30 ; 000 013 027 042
        let local_scan = plane_exclusive_sum(increment);

        // 2. Compute inclusive sum of the entire plane and store in shared memory
        //    (only the first thread of each plane writes).
        // e.g. plane_last_scan  =          06 ;          18 ;          30 ;             042
        let plane_last_scan = plane_shuffle(local_scan, PLANE_DIM - 1);
        // e.g. plane_last_inc   =          04 ;          08 ;          12 ;             016
        let plane_last_increment = plane_shuffle(increment, PLANE_DIM - 1);
        // e.g. plane_total      =          10 ;          26 ;          42 ;             058
        let plane_total = plane_last_scan + plane_last_increment;
        // e.g. shared_totals    =          10 ;          26 ;          42 ;             058
        if plane_thread_idx == 0 {
            shared_totals[plane_idx] = plane_total;
        }
        let () = sync_cube(); // for shared_totals

        // 3. Exclusive prefix sum over the plane totals (performed by plane 0).
        comptime! { // jit time
            if num_planes > PLANE_DIM as usize {
                panic!("num_planes exceeds PLANE_DIM. The current plane-based reduction assumes PLANE_DIM >= num_planes.");
            }
            // i.e. requires chunk_len <= PLANE_DIM * PLANE_DIM.
        }
        // e.g. offset           =           0 ;          10 ;          36 ;             078
        // e.g. shared_totals    =           0 ;          10 ;          36 ;             078
        if plane_idx == 0 {
            // The number of threads capacity in plane 0 may exceed the required num_planes,
            // so the extra threads need to compute the identity operation.
            let val = if plane_thread_idx < num_planes {
                shared_totals[plane_thread_idx]
            } else {
                F::new(0.0)
            };
            // Note: All threads in plane 0 must call plane_exclusive_sum together (no intra-plane divergence).
            let offset = plane_exclusive_sum(val);
            // Only write back for slots that actually exist.
            if plane_thread_idx < num_planes {
                shared_totals[plane_thread_idx] = offset;
            }
        }
        let () = sync_cube(); // for shared_totals

        // 4. Full-cube exclusive prefix sum.
        // e.g. exclusive_result = 00 01 03 06 ; 10 15 21 28 ; 36 45 55 66 ; 078 091 105 120
        let exclusive_result = local_scan + shared_totals[plane_idx];

        // Convert to inclusive prefix sum (required by the original algorithm).
        // e.g. inclusive_result = 01 03 06 10 ; 15 21 28 36 ; 45 55 66 78 ; 091 105 120 136
        let inclusive_result = exclusive_result + increment;

        // Write result.
        da_cumsum_bhnl[my_pos] = inclusive_result;
    }
}

/// Naive (forward) implementation in `burn`.
pub mod naive {
    use burn::prelude::*;

    /// Returns `da_cumsum_bhnl`.
    pub fn forward<B: Backend>(
        dt_discretized_bhnl: Tensor<B, 4>,
        a_decay_h: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let [batch, nheads, nchunks, chunk_len] = dt_discretized_bhnl.dims();
        let a_decay_bhnl = a_decay_h
            .unsqueeze_dims::<4>(&[0, 1, 2]) // 111h
            .expand([batch, nheads, nchunks, chunk_len]);
        (dt_discretized_bhnl * a_decay_bhnl).cumsum(3) // da_cumsum_bhnl
    }
}

#[cfg(test)]
#[cfg(feature = "backend-cuda")]
mod test {
    use super::*;
    pub type TestBackend = burn::backend::Cuda<f32, i32>;
    use burn::tensor::Tolerance;
    use burn::tensor::{Distribution, Tensor};

    #[test]
    fn kernel_naive_comparison_01() {
        let device = Default::default();
        let (batch, nheads, nchunks, chunk_len) = (2, 2, 2, 32);

        // input setup
        let dt_discretized_bhnl: Tensor<TestBackend, 1, burn::prelude::Int> = Tensor::arange(
            1..(1 + batch * nheads * nchunks * chunk_len) as i64,
            &device,
        );
        let dt_discretized_bhnl: Tensor<TestBackend, 4> = dt_discretized_bhnl
            .float()
            .reshape([batch, nheads, nchunks, chunk_len]);
        let a_decay_h: Tensor<TestBackend, 1> = Tensor::ones([nheads], &device);

        // naive
        let da_cumsum_naive_bhnl = naive::forward(dt_discretized_bhnl.clone(), a_decay_h.clone());
        assert_eq!(
            [batch, nheads, nchunks, chunk_len],
            da_cumsum_naive_bhnl.dims()
        );
        println!("naive: {}", &da_cumsum_naive_bhnl);

        // kernel
        let da_cumsum_cubecl_bhnl: Tensor<TestBackend, 4> =
            backend_ext::k1_ssd_chunk_cumsum(dt_discretized_bhnl.clone(), a_decay_h.clone());
        assert_eq!(
            [batch, nheads, nchunks, chunk_len],
            da_cumsum_cubecl_bhnl.dims()
        );
        println!("cubecl: {}", &da_cumsum_cubecl_bhnl);

        // comparison
        da_cumsum_naive_bhnl
            .to_data()
            .assert_approx_eq::<f32>(&da_cumsum_cubecl_bhnl.to_data(), Tolerance::default());
    }
}
