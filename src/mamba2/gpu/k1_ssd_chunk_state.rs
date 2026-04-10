// WIP - code available but not yet being used

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
    #[comptime] num_planes: u32,
) {
    let dt = dt_discretized_bhnl;
    let a = a_decay_h;
    let num_planes = num_planes as usize;

    let batch_stride = dt.stride(0);
    let nheads_stride = dt.stride(1);
    let nchunks_stride = dt.stride(2);
    let chunk_len_stride = dt.stride(3);

    // Base linear offset for the current (batch, head, chunk) slice.
    let base_pos = CUBE_POS_X as usize * batch_stride
        + CUBE_POS_Y as usize * nheads_stride
        + CUBE_POS_Z as usize * nchunks_stride;

    // Scalar value a[h] (identical for all units in this cube).
    let a_offset = CUBE_POS_Y as usize * a.stride(0);
    let a_1 = a[a_offset];

    // Local position within the chunk (0 .. chunk_len-1).
    let thread_id = UNIT_POS_X;
    let plane_thread_idx = UNIT_POS_PLANE as usize;
    let plane_idx = (thread_id / PLANE_DIM) as usize;

    // Effective plane size (handles cases where chunk_len < PLANE_DIM).
    let plane_size = if CUBE_DIM_X < PLANE_DIM {
        CUBE_DIM_X
    } else {
        PLANE_DIM
    };

    // Position of the element processed by this unit.
    let my_pos = base_pos + (thread_id as usize) * chunk_len_stride;

    // Per-position increment for the prefix sum.
    // eg. increment        = 01 02 03 04 ; 05 06 07 08 ; 09 10 11 12 ; 013 014 015 016
    let increment = dt[my_pos] * a_1;

    // Shared memory for plane totals / offsets (size known at compile time).
    let mut shared_totals = SharedMemory::<F>::new(num_planes as usize);

    // 1. Exclusive prefix sum within the current plane.
    // e.g. local_scan       = 00 01 03 06 ; 00 05 11 18 ; 00 09 19 30 ; 000 013 027 042
    let local_scan = plane_exclusive_sum(increment);

    // 2. Compute inclusive sum of the entire plane and store in shared memory
    //    (only the first thread of each plane writes).
    // e.g. plane_last_scan  =          06 ;          18 ;          30 ;             042
    let plane_last_scan = plane_shuffle(local_scan, plane_size - 1);
    // e.g. plane_last_inc   =          04 ;          08 ;          12 ;             016
    let plane_last_increment = plane_shuffle(increment, plane_size - 1);
    // e.g. plane_total      =          10 ;          26 ;          42 ;             058
    let plane_total = plane_last_scan + plane_last_increment;
    // e.g. shared_totals    =          10 ;          26 ;          42 ;             058
    if plane_thread_idx == 0 {
        shared_totals[plane_idx] = plane_total;
    }
    sync_cube();

    // 3. Exclusive prefix sum over the plane totals (performed by plane 0).
    comptime! {
        if num_planes > PLANE_DIM as usize {
            panic!("num_planes exceeds PLANE_DIM. The current plane-based reduction assumes PLANE_DIM >= num_planes.");
        }
    }
    // e.g. offset           =           0 ;          10 ;          36 ;             078
    // e.g. shared_totals    =           0 ;          10 ;          36 ;             078
    if plane_idx == 0 && plane_thread_idx < num_planes {
        let offset = plane_exclusive_sum(shared_totals[plane_thread_idx]);
        shared_totals[plane_thread_idx] = offset;
    }
    sync_cube();

    // 4. Full-cube exclusive prefix sum.
    // e.g. exclusive_result = 00 01 03 06 ; 10 15 21 28 ; 36 45 55 66 ; 078 091 105 120
    let exclusive_result = local_scan + shared_totals[plane_idx];

    // Convert to inclusive prefix sum (required by the original algorithm).
    // e.g. inclusive_result = 01 03 06 10 ; 15 21 28 36 ; 45 55 66 78 ; 091 105 120 136
    let inclusive_result = exclusive_result + increment;

    // Write result.
    da_cumsum_bhnl[my_pos] = inclusive_result;
}

#[allow(unused_mut)]
pub fn launch<R: Runtime, F: Float + CubeElement>(
    device: &R::Device,
    dt_discretized_bhnl: TensorArg<R>,
    a_decay_h: TensorArg<R>,
    mut da_cumsum_bhnl: TensorArg<R>,
    batch: usize,
    nheads: usize,
    nchunks: usize,
    chunk_len: usize,
) {
    let client = R::client(device);

    // Compute number of planes per cube (comptime parameter).
    // Typical plane dimension is 32 on most backends; adjust if a backend-specific
    // query is available (shared memory is over-allocated safely if needed).
    let plane_dim = 32u32; // conservative default for CUDA/WebGPU/Metal
    let num_planes = ((chunk_len as u32) + plane_dim - 1) / plane_dim;

    unsafe {
        forward::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(batch as u32, nheads as u32, nchunks as u32),
            CubeDim::new_1d(chunk_len as u32),
            dt_discretized_bhnl,
            a_decay_h,
            da_cumsum_bhnl,
            num_planes, // #[comptime]
        )
    };
}
