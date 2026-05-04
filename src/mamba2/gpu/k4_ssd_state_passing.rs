// snippet, other info omitted

/// Extends the backend (for `burn_cubecl`) and wraps it for `burn`.
pub mod backend_ext {
    use burn::tensor::{Tensor, TensorPrimitive, ops::FloatTensor};

    // Wraps the extended backend for `burn`.
    //
    /// Returns `(chunk_input_states_bnhpr, final_ssm_state_bhpr)`.
    ///
    /// # Shapes
    /// - `intra_chunk_states_bnhpr` : `[batch, nchunks, nheads, per_head_dim, state_rank]`
    /// - `da_chunk_end_bhn`         : `[batch, nheads, nchunks]`
    /// - `initial_states_bhpr`      : `[batch, nheads, per_head_dim, state_rank]`
    #[cfg(feature = "cubecl")]
    pub fn k4_ssd_state_passing<B: BackendExt>(
        intra_chunk_states_bnhpr: Tensor<B, 5>,
        da_chunk_end_bhn: Tensor<B, 3>,
        initial_states_bhpr: Tensor<B, 4>,
    ) -> (Tensor<B, 5>, Tensor<B, 4>) {
        let [batch, nchunks, nheads, per_head_dim, state_rank] = intra_chunk_states_bnhpr.dims();
        let flat_state_dim = per_head_dim * state_rank;

        // Flatten the (per_head_dim, state_rank) trailing dims into one flat_state_dim axis.
        // Python: `rearrange(states, "... p n -> ... (p n)")`
        let intra_chunk_states_bnhf =
            intra_chunk_states_bnhpr.reshape([batch, nchunks, nheads, flat_state_dim]);
        let initial_states_bhf = initial_states_bhpr.reshape([batch, nheads, flat_state_dim]);

        let (chunk_input_states_bnhf, final_ssm_state_bhf) = B::k4_ssd_state_passing(
            intra_chunk_states_bnhf.into_primitive().tensor(),
            da_chunk_end_bhn.into_primitive().tensor(),
            initial_states_bhf.into_primitive().tensor(),
        );

        let chunk_input_states_bnhf: Tensor<B, 4> =
            Tensor::from_primitive(TensorPrimitive::Float(chunk_input_states_bnhf));
        let final_ssm_state_bhf: Tensor<B, 5> =
            Tensor::from_primitive(TensorPrimitive::Float(final_ssm_state_bhf));

        // Restore the 5D shape after state-passing.
        // Python: `rearrange(states_flat, "... (p n) -> ... p n", n=state_rank)`
        let chunk_input_states_bnhpr: Tensor<B, 5> =
            chunk_input_states_bnhf.reshape([batch, nchunks, nheads, per_head_dim, state_rank]);
        let final_ssm_state_bhpr: Tensor<B, 4> =
            final_ssm_state_bhf.reshape([batch, nheads, per_head_dim, state_rank]);

        (chunk_input_states_bnhpr, final_ssm_state_bhpr)
    }

    // Wraps the extended backend for `burn` (naive version).
    //
    /// Returns `(chunk_input_states_bnhpr, final_ssm_state_bhpr)`.
    pub fn k4_ssd_state_passing_naive<B: BackendExt>(
        intra_chunk_states_bnhpr: Tensor<B, 5>,
        da_chunk_end_bhn: Tensor<B, 3>,
        initial_states_bhpr: Tensor<B, 4>,
    ) -> (Tensor<B, 5>, Tensor<B, 4>) {
        let (chunk_input_states_bnhpr, final_ssm_state_bhpr) = B::k4_ssd_state_passing_naive(
            intra_chunk_states_bnhpr.into_primitive().tensor(),
            da_chunk_end_bhn.into_primitive().tensor(),
            initial_states_bhpr.into_primitive().tensor(),
        );

        (
            Tensor::from_primitive(TensorPrimitive::Float(chunk_input_states_bnhpr)),
            Tensor::from_primitive(TensorPrimitive::Float(final_ssm_state_bhpr)),
        )
    }

    // Backend extended with a new op.
    pub trait BackendExt: burn::tensor::backend::Backend {
        // Note: tensors are pre-flattened to [..., flat_state_dim] before this call.
        // The host wrapper (`k4_ssd_state_passing`) handles flatten/reshape around this.
        /// Returns `(chunk_input_states_bnhf, final_ssm_state_bhf)`.
        #[cfg(feature = "cubecl")]
        fn k4_ssd_state_passing(
            intra_chunk_states_bnhf: FloatTensor<Self>,
            da_chunk_end_bhn: FloatTensor<Self>,
            initial_states_bhf: FloatTensor<Self>,
        ) -> (FloatTensor<Self>, FloatTensor<Self>) {
            unimplemented!("Backend not supported")
        }

        // Naive fallback.
        /// Returns `(chunk_input_states_bnhpr, final_ssm_state_bhpr)`.
        fn k4_ssd_state_passing_naive(
            intra_chunk_states_bnhpr: FloatTensor<Self>,
            da_chunk_end_bhn: FloatTensor<Self>,
            initial_states_bhpr: FloatTensor<Self>,
        ) -> (FloatTensor<Self>, FloatTensor<Self>) {
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
pub use backend_ext::k4_ssd_state_passing;

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
    impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> BackendExt
        for CubeBackend<R, F, I, BT>
    {
        /// Returns `(chunk_input_states_bnhf, final_states_bhf)`.
        ///
        /// # Shapes (flat representation — caller handles reshape)
        /// - `intra_chunk_states_bnhf` : `[batch, nchunks, nheads, flat_state_dim]`
        /// - `da_chunk_end_bhn`        : `[batch, nheads, nchunks]`
        /// - `initial_states_bhf`      : `[batch, nheads, flat_state_dim]`
        fn k4_ssd_state_passing(
            intra_chunk_states_bnhf: FloatTensor<Self>,
            da_chunk_end_bhn: FloatTensor<Self>,
            initial_states_bhf: FloatTensor<Self>,
        ) -> (FloatTensor<Self>, FloatTensor<Self>) {
            let client = intra_chunk_states_bnhf.client.clone();
            let device = intra_chunk_states_bnhf.device.clone();
            intra_chunk_states_bnhf.assert_is_on_same_device(&da_chunk_end_bhn);
            intra_chunk_states_bnhf.assert_is_on_same_device(&initial_states_bhf);

            // For simplicity, make sure each tensor is contiguous.
            let intra = into_contiguous(intra_chunk_states_bnhf);
            let da = into_contiguous(da_chunk_end_bhn);
            let init = into_contiguous(initial_states_bhf);

            // Get relevant shapes.
            let [batch, nchunks, nheads, flat_state_dim] = intra.meta.shape().dims();

            // Tile the flat_state_dim across cubes; each unit owns one element.
            //
            //   CUBE_POS_X → tile over flat_state_dim   (0 .. ceil(flat / BLOCK_SIZE))
            //   CUBE_POS_Y → batch index                (0 .. batch)
            //   CUBE_POS_Z → head index                 (0 .. nheads)
            //   UNIT_POS_X → position within tile       (0 .. BLOCK_SIZE)
            //
            // The serial loop over `nchunks` lives inside each unit; it cannot be
            // parallelised because each iteration depends on the previous.
            //
            // BLOCK_SIZE = 256 matches the Triton autotune default.
            pub const STATE_FLAT_BLOCK_SIZE: u32 = 256;
            let block_size = STATE_FLAT_BLOCK_SIZE;
            let num_tiles =
                flat_state_dim.next_multiple_of(block_size as usize) as u32 / block_size;

            // Create output tensors.
            //
            // chunk_input_states[b, c, h, f] = SSM state *entering* chunk c.
            let cis_shape = Shape::from([batch, nchunks, nheads, flat_state_dim]);
            let cis_buffer = client.empty(cis_shape.num_elements() * core::mem::size_of::<F>());
            let chunk_input_states = CubeTensor::new_contiguous(
                client.clone(),
                device.clone(),
                cis_shape,
                cis_buffer,
                F::dtype(),
            );

            // final_states[b, h, f] = SSM state *after* the last chunk.
            let fs_shape = Shape::from([batch, nheads, flat_state_dim]);
            let fs_buffer = client.empty(fs_shape.num_elements() * core::mem::size_of::<F>());
            let final_states = CubeTensor::new_contiguous(
                client.clone(),
                device.clone(),
                fs_shape,
                fs_buffer,
                F::dtype(),
            );

            unsafe {
                kernel::forward::launch_unchecked::<F, R>(
                    &client,
                    CubeCount::Static(num_tiles, batch as u32, nheads as u32),
                    CubeDim::new_1d(block_size),
                    intra.into_tensor_arg(),
                    da.into_tensor_arg(),
                    init.into_tensor_arg(),
                    chunk_input_states.clone().into_tensor_arg(),
                    final_states.clone().into_tensor_arg(),
                    block_size, // #[comptime]
                )
            };

            (chunk_input_states, final_states)
        }
    }
}
// blank impl for non-cubecl backends
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
        fn k4_ssd_state_passing(
            intra_chunk_states_bnhf: FloatTensor<Self>,
            da_chunk_end_bhn: FloatTensor<Self>,
            initial_states_bhf: FloatTensor<Self>,
        ) -> (FloatTensor<Self>, FloatTensor<Self>) {
            unimplemented!("AutodiffBackend not supported")
        }

        // Naive fallback.
        fn k4_ssd_state_passing_naive(
            intra_chunk_states_bnhpr: FloatTensor<Self>,
            da_chunk_end_bhn: FloatTensor<Self>,
            initial_states_bhpr: FloatTensor<Self>,
        ) -> (FloatTensor<Self>, FloatTensor<Self>) {
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
    /// - `intra_states`        : `[batch, nchunks, nheads, flat_state_dim]`
    /// - `da_chunk_end`        : `[batch, nheads, nchunks]`
    /// - `initial_states`      : `[batch, nheads, flat_state_dim]`
    /// - `chunk_input_states`  : `[batch, nchunks, nheads, flat_state_dim]` (output)
    /// - `final_states`        : `[batch, nheads, flat_state_dim]`         (output)
    ///
    /// # Dispatch
    /// ```text
    ///   CubeCount  = (ceil(flat_state_dim / BLOCK_SIZE), batch, nheads)
    ///   CubeDim    = (BLOCK_SIZE, 1, 1)
    ///
    ///   CUBE_POS_X  = tile index over flat_state_dim
    ///   CUBE_POS_Y  = batch index
    ///   CUBE_POS_Z  = head index
    ///   UNIT_POS_X  = lane within tile  →  flat_idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X
    /// ```
    ///
    /// Each unit independently walks the serial chunk-loop for its own flat_state_dim
    /// element.  No synchronisation is required because units never share state.
    #[cube(launch_unchecked)]
    pub fn forward<F: Float>(
        intra_states: &Tensor<F>,
        da_chunk_end: &Tensor<F>,
        initial_states: &Tensor<F>,
        chunk_input_states: &mut Tensor<F>,
        final_states: &mut Tensor<F>,
        #[comptime] block_size: u32,
    ) {
        // ── Index derivation ────────────────────────────────────────────────────
        let batch_idx = CUBE_POS_Y as usize;
        let head_idx = CUBE_POS_Z as usize;
        let tile_idx = CUBE_POS_X as usize;
        let local_idx = UNIT_POS_X as usize;

        // Global position in the flat state vector for this unit.
        let flat_idx = tile_idx * block_size as usize + local_idx;

        // Guard: the last tile may be partially filled.
        let flat_state_dim = initial_states.shape(2); // == intra_states.shape(3)
        if flat_idx >= flat_state_dim {
            terminate!();
        }

        // ── Stride pre-computation ──────────────────────────────────────────────
        // intra_states  : [batch, nchunks, nheads, flat_state_dim]
        let is_s0 = intra_states.stride(0); // batch stride
        let is_s1 = intra_states.stride(1); // chunk stride
        let is_s2 = intra_states.stride(2); // head stride
        let is_s3 = intra_states.stride(3); // flat stride

        // da_chunk_end  : [batch, nheads, nchunks]
        let da_s0 = da_chunk_end.stride(0); // batch stride
        let da_s1 = da_chunk_end.stride(1); // head stride
        let da_s2 = da_chunk_end.stride(2); // chunk stride

        // initial_states : [batch, nheads, flat_state_dim]
        let init_s0 = initial_states.stride(0); // batch stride
        let init_s1 = initial_states.stride(1); // head stride
        let init_s2 = initial_states.stride(2); // flat stride

        // chunk_input_states : [batch, nchunks, nheads, flat_state_dim]
        //   (same logical layout as intra_states)
        let cis_s0 = chunk_input_states.stride(0);
        let cis_s1 = chunk_input_states.stride(1);
        let cis_s2 = chunk_input_states.stride(2);
        let cis_s3 = chunk_input_states.stride(3);

        // final_states : [batch, nheads, flat_state_dim]
        let fs_s0 = final_states.stride(0);
        let fs_s1 = final_states.stride(1);
        let fs_s2 = final_states.stride(2);

        // ── Base offsets (shared across all chunks for this unit) ───────────────
        // Using usize throughout to avoid overflow on large tensors.
        let intra_base = batch_idx * is_s0 + head_idx * is_s2 + flat_idx * is_s3;
        let da_base = batch_idx * da_s0 + head_idx * da_s1;
        let init_base = batch_idx * init_s0 + head_idx * init_s1 + flat_idx * init_s2;
        let cis_base = batch_idx * cis_s0 + head_idx * cis_s2 + flat_idx * cis_s3;
        let fs_base = batch_idx * fs_s0 + head_idx * fs_s1 + flat_idx * fs_s2;

        // ── Load initial state ──────────────────────────────────────────────────
        // Triton: `if HAS_INITSTATES: states = tl.load(initstates_ptrs, ...)`
        let mut running_state = initial_states[init_base];

        // Store the initial state as the input for chunk 0.
        // Triton: `tl.store(out_ptrs, states, ...)` BEFORE the chunk loop.
        chunk_input_states[cis_base] = running_state; // chunk index 0 → offset 0

        // ── Serial recurrence over chunks ───────────────────────────────────────
        // This loop is the ONLY inherently serial part of the SSD pipeline.
        // Triton: `for c in range(nchunks):`
        //
        // Each iteration:
        //   decay          = exp(dA_chunk_end[b, h, c])
        //   running_state  = decay * running_state + intra_state[b, c, h, f]
        //
        // Then the updated state is stored as the *input* for the next chunk
        // (or as the final SSM state when c == nchunks - 1).
        let nchunks = intra_states.shape(1); // runtime value

        let mut i_chunk: u32 = 0;
        while i_chunk < nchunks as u32 {
            let i_chunk_us = i_chunk as usize;

            // Triton: `new_states = tl.load(states_ptrs, ...)`
            let intra_pos = intra_base + i_chunk_us * is_s1;
            let intra_state = intra_states[intra_pos];

            // Triton: `dA_cs = tl.load(dA_cs_ptr); scale = tl.exp(dA_cs)`
            // da_chunk_end is a scalar per (batch, head, chunk) — all units in this
            // cube load the same value, which is fine (read-only broadcast).
            let da_pos = da_base + i_chunk_us * da_s2;
            let decay = F::exp(da_chunk_end[da_pos]);

            // Triton: `states = scale * states + new_states`
            running_state = decay * running_state + intra_state;

            // Use `i_chunk + 1 < nchunks` (instead of `i_chunk < nchunks - 1`)
            // to stay safe with u32 arithmetic when nchunks == 1.
            if i_chunk + 1 < nchunks as u32 {
                // Store state entering chunk (i_chunk + 1).
                // Triton: `tl.store(out_ptrs, states, ...); out_ptrs += stride_out_chunk`
                let cis_pos = cis_base + (i_chunk_us + 1) * cis_s1;
                chunk_input_states[cis_pos] = running_state;
            } else {
                // Last chunk: this running_state is the final SSM state.
                // Triton: `tl.store(final_states_ptrs, states, ...)`
                final_states[fs_base] = running_state;
            }

            i_chunk += 1;
        }
    }
}

/// Naive (forward) implementation in `burn`.
pub mod naive {
    use burn::prelude::*;
    use burn::tensor::s;

    /// Returns `(chunk_input_states_bnhpr, final_ssm_state_bhpr)`.
    ///
    /// # Shapes
    /// - `intra_chunk_states_bnhpr` : `[batch, nchunks, nheads, per_head_dim, state_rank]`
    /// - `da_chunk_end_bhn`         : `[batch, nheads, nchunks]`
    /// - `ssm_initial_state_bhpr`   : `[batch, nheads, per_head_dim, state_rank]`
    pub fn forward<B: Backend>(
        intra_chunk_states_bnhpr: Tensor<B, 5>,
        da_chunk_end_bhn: Tensor<B, 3>,
        ssm_initial_state_bhpr: Tensor<B, 4>,
    ) -> (Tensor<B, 5>, Tensor<B, 4>) {
        let [batch, nchunks, nheads, per_head_dim, state_rank] = intra_chunk_states_bnhpr.dims();

        let flat_state_dim = per_head_dim * state_rank;
        let mut running_state_bhpr = ssm_initial_state_bhpr;
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            running_state_bhpr.dims()
        );

        let mut chunk_input_states_vec_bhpr = Vec::with_capacity(nchunks + 1);
        chunk_input_states_vec_bhpr.push(running_state_bhpr.clone());

        // serial loop
        for i_chunk in 0..nchunks {
            let intra_state_bhpr = intra_chunk_states_bnhpr
                .clone()
                .slice(s![.., i_chunk, .., .., ..])
                .squeeze_dim::<4>(1);
            assert_eq!(
                [batch, nheads, per_head_dim, state_rank],
                intra_state_bhpr.dims()
            );

            let decay_bhpr = da_chunk_end_bhn
                .clone()
                .slice(s![.., .., i_chunk]) // bh1
                .exp()
                .unsqueeze_dim::<4>(3) // bh11
                .expand([batch, nheads, per_head_dim, state_rank]);

            // SSM recurrence: running_state = decay * running_state + intra_state
            running_state_bhpr = decay_bhpr * running_state_bhpr + intra_state_bhpr;
            chunk_input_states_vec_bhpr.push(running_state_bhpr.clone());
        }

        let final_ssm_state_bhpr = chunk_input_states_vec_bhpr.pop().unwrap();
        let chunk_input_states_bnhpr = Tensor::stack(chunk_input_states_vec_bhpr, 1);
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            chunk_input_states_bnhpr.dims()
        );

        (chunk_input_states_bnhpr, final_ssm_state_bhpr)
    }
}
