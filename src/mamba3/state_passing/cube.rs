//! CubeCL implementation of the Mamba-3 inter-chunk state recurrence.

use super::state_passing::Mamba3StatePassingBackendExt;
use burn::backend::Shape;
use burn::backend::cubecl::dtype_to_storage_type;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeRuntime};
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, cube};

const WORKGROUP: u32 = 256;

#[cube(launch)]
fn state_passing_forward_kernel<F: Float>(
    intra: &Tensor<F>,
    decay: &Tensor<F>,
    initial: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let state_pos = ABSOLUTE_POS as usize;
    let batch = intra.shape(0);
    let nchunks = intra.shape(1);
    let nheads = intra.shape(2);
    let inner = intra.shape(3) * intra.shape(4);
    let state_elements = batch * nheads * inner;
    if state_pos >= state_elements {
        terminate!();
    }

    let inner_pos = state_pos % inner;
    let bh = state_pos / inner;
    let head = bh % nheads;
    let batch_pos = bh / nheads;
    let mut running = initial[state_pos];
    output[batch_pos * (nchunks + 1) * nheads * inner + head * inner + inner_pos] = running;

    let mut chunk = 0usize;
    while chunk < nchunks {
        let intra_pos = ((batch_pos * nchunks + chunk) * nheads + head) * inner + inner_pos;
        let decay_pos = (batch_pos * nheads + head) * nchunks + chunk;
        running = decay[decay_pos] * running + intra[intra_pos];
        let output_pos = batch_pos * (nchunks + 1) * nheads * inner
            + (chunk + 1) * nheads * inner
            + head * inner
            + inner_pos;
        output[output_pos] = running;
        chunk += 1;
    }
}

#[cube(launch)]
fn state_passing_backward_kernel<F: Float>(
    states: &Tensor<F>,
    decay: &Tensor<F>,
    d_states: &Tensor<F>,
    d_intra: &mut Tensor<F>,
    d_decay_contrib: &mut Tensor<F>,
    d_initial: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let state_pos = ABSOLUTE_POS as usize;
    let batch = d_intra.shape(0);
    let nchunks = d_intra.shape(1);
    let nheads = d_intra.shape(2);
    let inner = d_intra.shape(3) * d_intra.shape(4);
    let state_elements = batch * nheads * inner;
    if state_pos >= state_elements {
        terminate!();
    }

    let inner_pos = state_pos % inner;
    let bh = state_pos / inner;
    let head = bh % nheads;
    let batch_pos = bh / nheads;
    let state_base = batch_pos * (nchunks + 1) * nheads * inner;
    let state_tail = state_base + nchunks * nheads * inner + head * inner + inner_pos;
    let mut g = d_states[state_tail];

    let mut remaining = nchunks;
    while remaining > 0 {
        let chunk = remaining - 1;
        let intra_pos = ((batch_pos * nchunks + chunk) * nheads + head) * inner + inner_pos;
        let state_before = state_base + chunk * nheads * inner + head * inner + inner_pos;
        let decay_pos = (batch_pos * nheads + head) * nchunks + chunk;

        d_intra[intra_pos] = g;
        d_decay_contrib[intra_pos] = g * states[state_before];
        g = d_states[state_before] + decay[decay_pos] * g;
        remaining -= 1;
    }
    d_initial[state_pos] = g;
}

#[cube(launch)]
fn state_passing_decay_reduce_kernel<F: Float>(
    contributions: &Tensor<F>,
    d_decay: &mut Tensor<F>,
    #[define(F)] _dtype: StorageType,
) {
    let decay_pos = ABSOLUTE_POS as usize;
    let batch = contributions.shape(0);
    let nchunks = contributions.shape(1);
    let nheads = contributions.shape(2);
    let inner = contributions.shape(3) * contributions.shape(4);
    let decay_elements = batch * nheads * nchunks;
    if decay_pos >= decay_elements {
        terminate!();
    }

    let chunk = decay_pos % nchunks;
    let bh = decay_pos / nchunks;
    let head = bh % nheads;
    let batch_pos = bh / nheads;
    let mut sum = F::new(0.0);
    let mut inner_pos = 0usize;
    while inner_pos < inner {
        let contribution_pos = ((batch_pos * nchunks + chunk) * nheads + head) * inner + inner_pos;
        sum += contributions[contribution_pos];
        inner_pos += 1;
    }
    d_decay[decay_pos] = sum;
}

fn empty<R: CubeRuntime>(template: &CubeTensor<R>, shape: Shape) -> CubeTensor<R> {
    let buffer = template
        .client
        .empty(shape.num_elements() * template.dtype.size());
    CubeTensor::new_contiguous(
        template.client.clone(),
        template.device.clone(),
        shape,
        buffer,
        template.dtype,
    )
}

fn launch_geometry(elements: usize) -> (CubeCount, CubeDim) {
    let dim = CubeDim {
        x: WORKGROUP,
        y: 1,
        z: 1,
    };
    let cubes = elements.div_ceil(WORKGROUP as usize) as u32;
    (CubeCount::Static(cubes, 1, 1), dim)
}

fn state_passing_forward<R: CubeRuntime>(
    intra: CubeTensor<R>,
    decay: CubeTensor<R>,
    initial: CubeTensor<R>,
) -> CubeTensor<R> {
    let intra = into_contiguous(intra);
    let decay = into_contiguous(decay);
    let initial = into_contiguous(initial);
    let batch = intra.meta.shape()[0];
    let nchunks = intra.meta.shape()[1];
    let nheads = intra.meta.shape()[2];
    let per_head_dim = intra.meta.shape()[3];
    let state_rank = intra.meta.shape()[4];
    let output = empty(
        &intra,
        Shape::new([batch, nchunks + 1, nheads, per_head_dim, state_rank]),
    );
    let (cube_count, cube_dim) = launch_geometry(batch * nheads * per_head_dim * state_rank);
    let dtype = intra.dtype;

    state_passing_forward_kernel::launch::<R>(
        &output.client,
        cube_count,
        cube_dim,
        intra.into_tensor_arg(),
        decay.into_tensor_arg(),
        initial.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        dtype_to_storage_type(dtype),
    );
    output
}

fn state_passing_backward<R: CubeRuntime>(
    states: CubeTensor<R>,
    decay: CubeTensor<R>,
    d_states: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    let states = into_contiguous(states);
    let decay = into_contiguous(decay);
    let d_states = into_contiguous(d_states);
    let batch = states.meta.shape()[0];
    let nchunks = states.meta.shape()[1] - 1;
    let nheads = states.meta.shape()[2];
    let per_head_dim = states.meta.shape()[3];
    let state_rank = states.meta.shape()[4];
    let shape_intra = Shape::new([batch, nchunks, nheads, per_head_dim, state_rank]);
    let d_intra = empty(&states, shape_intra.clone());
    let contributions = empty(&states, shape_intra);
    let d_initial = empty(
        &states,
        Shape::new([batch, nheads, per_head_dim, state_rank]),
    );
    let d_decay = empty(&states, Shape::new([batch, nheads, nchunks]));
    let dtype = states.dtype;
    let client = states.client.clone();
    let (state_count, state_dim) = launch_geometry(batch * nheads * per_head_dim * state_rank);

    state_passing_backward_kernel::launch::<R>(
        &client,
        state_count,
        state_dim,
        states.into_tensor_arg(),
        decay.into_tensor_arg(),
        d_states.into_tensor_arg(),
        d_intra.clone().into_tensor_arg(),
        contributions.clone().into_tensor_arg(),
        d_initial.clone().into_tensor_arg(),
        dtype_to_storage_type(dtype),
    );

    let (decay_count, decay_dim) = launch_geometry(batch * nheads * nchunks);
    state_passing_decay_reduce_kernel::launch::<R>(
        &d_decay.client,
        decay_count,
        decay_dim,
        contributions.into_tensor_arg(),
        d_decay.clone().into_tensor_arg(),
        dtype_to_storage_type(dtype),
    );

    (d_intra, d_decay, d_initial)
}

impl<R: CubeRuntime> Mamba3StatePassingBackendExt for CubeBackend<R> {
    fn mamba3_state_passing(
        intra_bnhpr: CubeTensor<R>,
        decay_bhn: CubeTensor<R>,
        initial_bhpr: CubeTensor<R>,
    ) -> CubeTensor<R> {
        state_passing_forward(intra_bnhpr, decay_bhn, initial_bhpr)
    }

    fn mamba3_state_passing_backward(
        states_bn1hpr: CubeTensor<R>,
        decay_bhn: CubeTensor<R>,
        d_states_bn1hpr: CubeTensor<R>,
    ) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
        state_passing_backward(states_bn1hpr, decay_bhn, d_states_bn1hpr)
    }
}
