//! CubeCL kernels for the fused rank-one single-SSD recurrence.

use super::{RECONSTRUCTION_INTERVAL, single_ssd_scan::Mamba3SingleSsdScanBackendExt};
use burn::backend::Shape;
use burn::backend::cubecl::dtype_to_storage_type;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeRuntime};
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, cube};

const WORKGROUP: u32 = 256;

#[cube(launch)]
fn single_ssd_scan_forward_kernel<F: Float>(
    v: &Tensor<F>,
    da: &Tensor<F>,
    b: &Tensor<F>,
    c: &Tensor<F>,
    gamma: &Tensor<F>,
    scale: &Tensor<F>,
    initial: &Tensor<F>,
    packed: &mut Tensor<F>,
    #[comptime] state_rank: usize,
    #[define(F)] _dtype: StorageType,
) {
    let scan_pos = ABSOLUTE_POS as usize;
    let batch = v.shape(0);
    let nchunks = v.shape(1);
    let chunk_len = v.shape(2);
    let tokens = nchunks * chunk_len;
    let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
    let nheads = v.shape(4);
    let per_head_dim = v.shape(5);
    let scans = batch * nheads * per_head_dim;
    if scan_pos >= scans {
        terminate!();
    }

    let p = scan_pos % per_head_dim;
    let bh = scan_pos / per_head_dim;
    let head = bh % nheads;
    let batch_pos = bh / nheads;
    let mut state = Array::<F>::new(state_rank);
    for r in 0..state_rank {
        let state_pos = ((batch_pos * nheads + head) * per_head_dim + p) * state_rank + r;
        state[r] = initial[state_pos];
    }

    let mut token = 0usize;
    while token < tokens {
        let coef_pos = (batch_pos * tokens + token) * nheads + head;
        let v_pos = ((batch_pos * tokens + token) * nheads + head) * per_head_dim + p;
        let decay = da[coef_pos].exp();
        let gamma_value = gamma[coef_pos];
        let scale_value = scale[coef_pos];
        let value = v[v_pos];
        let mut y = F::new(0.0);
        for r in 0..state_rank {
            let bc_pos = ((batch_pos * tokens + token) * nheads + head) * state_rank + r;
            let key = b[bc_pos];
            let pre = decay * state[r];
            y += c[bc_pos] * (pre + gamma_value * key * value);
            state[r] = pre + scale_value * key * value;
        }
        let packed_pos = (batch_pos * nheads + head)
            * (per_head_dim * (tokens + checkpoint_count * state_rank))
            + token * per_head_dim
            + p;
        packed[packed_pos] = y;
        if (token + 1) % RECONSTRUCTION_INTERVAL == 0 || token + 1 == tokens {
            let checkpoint = token / RECONSTRUCTION_INTERVAL;
            for r in 0..state_rank {
                let checkpoint_pos = (batch_pos * nheads + head)
                    * (per_head_dim * (tokens + checkpoint_count * state_rank))
                    + tokens * per_head_dim
                    + checkpoint * per_head_dim * state_rank
                    + p * state_rank
                    + r;
                packed[checkpoint_pos] = state[r];
            }
        }
        token += 1;
    }
}

/// Per-value stream: reconstructs the state and emits the unique dV/dInitial entries.
#[cube(launch)]
fn single_ssd_scan_backward_value_kernel<F: Float>(
    v: &Tensor<F>,
    da: &Tensor<F>,
    b: &Tensor<F>,
    c: &Tensor<F>,
    gamma: &Tensor<F>,
    scale: &Tensor<F>,
    packed: &Tensor<F>,
    d_packed: &Tensor<F>,
    d_v: &mut Tensor<F>,
    d_initial: &mut Tensor<F>,
    #[comptime] state_rank: usize,
    #[define(F)] _dtype: StorageType,
) {
    let scan_pos = ABSOLUTE_POS as usize;
    let batch = v.shape(0);
    let nchunks = v.shape(1);
    let chunk_len = v.shape(2);
    let tokens = nchunks * chunk_len;
    let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
    let nheads = v.shape(4);
    let per_head_dim = v.shape(5);
    let scans = batch * nheads * per_head_dim;
    if scan_pos >= scans {
        terminate!();
    }

    let p = scan_pos % per_head_dim;
    let bh = scan_pos / per_head_dim;
    let head = bh % nheads;
    let batch_pos = bh / nheads;
    let packed_base =
        (batch_pos * nheads + head) * (per_head_dim * (tokens + checkpoint_count * state_rank));
    let mut state_post = Array::<F>::new(state_rank);
    let mut g_post = Array::<F>::new(state_rank);
    for r in 0..state_rank {
        let state_pos = packed_base
            + tokens * per_head_dim
            + (checkpoint_count - 1) * per_head_dim * state_rank
            + p * state_rank
            + r;
        state_post[r] = packed[state_pos];
        g_post[r] = d_packed[state_pos];
    }

    let mut remaining = tokens;
    while remaining > 0 {
        let token = remaining - 1;
        if (token + 1) % RECONSTRUCTION_INTERVAL == 0 || token + 1 == tokens {
            let checkpoint = token / RECONSTRUCTION_INTERVAL;
            for r in 0..state_rank {
                let state_pos = packed_base
                    + tokens * per_head_dim
                    + checkpoint * per_head_dim * state_rank
                    + p * state_rank
                    + r;
                state_post[r] = packed[state_pos];
            }
        }
        let coef_pos = (batch_pos * tokens + token) * nheads + head;
        let v_pos = ((batch_pos * tokens + token) * nheads + head) * per_head_dim + p;
        let value = v[v_pos];
        let dy = d_packed[packed_base + token * per_head_dim + p];
        let decay = da[coef_pos].exp();
        let inverse_decay = (-da[coef_pos]).exp();
        let mut dv = F::new(0.0);
        for r in 0..state_rank {
            let bc_pos = ((batch_pos * tokens + token) * nheads + head) * state_rank + r;
            let key = b[bc_pos];
            dv += scale[coef_pos] * g_post[r] * key + gamma[coef_pos] * dy * c[bc_pos] * key;
            let pre = state_post[r] - scale[coef_pos] * key * value;
            let g_pre = g_post[r] + dy * c[bc_pos];
            state_post[r] = inverse_decay * pre;
            g_post[r] = decay * g_pre;
        }
        d_v[v_pos] = dv;
        remaining -= 1;
    }

    for r in 0..state_rank {
        let state_pos = ((batch_pos * nheads + head) * per_head_dim + p) * state_rank + r;
        d_initial[state_pos] = g_post[r];
    }
}

/// Per-state-rank stream: emits dB/dC and contention-free scalar contributions.
#[cube(launch)]
fn single_ssd_scan_backward_state_kernel<F: Float>(
    v: &Tensor<F>,
    da: &Tensor<F>,
    b: &Tensor<F>,
    c: &Tensor<F>,
    gamma: &Tensor<F>,
    scale: &Tensor<F>,
    packed: &Tensor<F>,
    d_packed: &Tensor<F>,
    d_b: &mut Tensor<F>,
    d_c: &mut Tensor<F>,
    contributions: &mut Tensor<F>,
    #[comptime] state_rank: usize,
    #[comptime] per_head_dim: usize,
    #[define(F)] _dtype: StorageType,
) {
    let scan_pos = ABSOLUTE_POS as usize;
    let batch = v.shape(0);
    let nchunks = v.shape(1);
    let chunk_len = v.shape(2);
    let tokens = nchunks * chunk_len;
    let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
    let nheads = v.shape(4);
    let scans = batch * nheads * state_rank;
    if scan_pos >= scans {
        terminate!();
    }

    let r = scan_pos % state_rank;
    let bh = scan_pos / state_rank;
    let head = bh % nheads;
    let batch_pos = bh / nheads;
    let packed_base =
        (batch_pos * nheads + head) * (per_head_dim * (tokens + checkpoint_count * state_rank));
    let mut state_post = Array::<F>::new(per_head_dim);
    let mut g_post = Array::<F>::new(per_head_dim);
    for p in 0..per_head_dim {
        let state_pos = packed_base
            + tokens * per_head_dim
            + (checkpoint_count - 1) * per_head_dim * state_rank
            + p * state_rank
            + r;
        state_post[p] = packed[state_pos];
        g_post[p] = d_packed[state_pos];
    }

    let mut remaining = tokens;
    while remaining > 0 {
        let token = remaining - 1;
        if (token + 1) % RECONSTRUCTION_INTERVAL == 0 || token + 1 == tokens {
            let checkpoint = token / RECONSTRUCTION_INTERVAL;
            for p in 0..per_head_dim {
                let state_pos = packed_base
                    + tokens * per_head_dim
                    + checkpoint * per_head_dim * state_rank
                    + p * state_rank
                    + r;
                state_post[p] = packed[state_pos];
            }
        }
        let coef_pos = (batch_pos * tokens + token) * nheads + head;
        let bc_pos = ((batch_pos * tokens + token) * nheads + head) * state_rank + r;
        let key = b[bc_pos];
        let query = c[bc_pos];
        let decay = da[coef_pos].exp();
        let inverse_decay = (-da[coef_pos]).exp();
        let mut gp_v = F::new(0.0);
        let mut dy_v = F::new(0.0);
        let mut dc = F::new(0.0);
        let mut dda = F::new(0.0);
        let mut dscale = F::new(0.0);

        for p in 0..per_head_dim {
            let v_pos = ((batch_pos * tokens + token) * nheads + head) * per_head_dim + p;
            let value = v[v_pos];
            let dy = d_packed[packed_base + token * per_head_dim + p];
            let pre = state_post[p] - scale[coef_pos] * key * value;
            let g_pre = g_post[p] + dy * query;
            gp_v += g_post[p] * value;
            dy_v += dy * value;
            dc += dy * (pre + gamma[coef_pos] * key * value);
            dda += g_pre * pre;
            dscale += g_post[p] * key * value;
            state_post[p] = inverse_decay * pre;
            g_post[p] = decay * g_pre;
        }

        d_b[bc_pos] = scale[coef_pos] * gp_v + gamma[coef_pos] * query * dy_v;
        d_c[bc_pos] = dc;
        let contribution_base = ((batch_pos * tokens + token) * nheads + head) * (3 * state_rank);
        contributions[contribution_base + r] = dda;
        contributions[contribution_base + state_rank + r] = query * key * dy_v;
        contributions[contribution_base + 2 * state_rank + r] = dscale;
        remaining -= 1;
    }
}

#[cube(launch)]
fn single_ssd_scan_backward_reduce_kernel<F: Float>(
    contributions: &Tensor<F>,
    d_da: &mut Tensor<F>,
    d_gamma: &mut Tensor<F>,
    d_scale: &mut Tensor<F>,
    #[comptime] state_rank: usize,
    #[define(F)] _dtype: StorageType,
) {
    let coef_pos = ABSOLUTE_POS as usize;
    let elements = d_da.len();
    if coef_pos >= elements {
        terminate!();
    }
    let contribution_base = coef_pos * (3 * state_rank);
    let mut dda = F::new(0.0);
    let mut dgamma = F::new(0.0);
    let mut dscale = F::new(0.0);
    for r in 0..state_rank {
        dda += contributions[contribution_base + r];
        dgamma += contributions[contribution_base + state_rank + r];
        dscale += contributions[contribution_base + 2 * state_rank + r];
    }
    d_da[coef_pos] = dda;
    d_gamma[coef_pos] = dgamma;
    d_scale[coef_pos] = dscale;
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

#[allow(clippy::too_many_arguments)]
fn single_ssd_scan_forward<R: CubeRuntime>(
    v: CubeTensor<R>,
    da: CubeTensor<R>,
    b: CubeTensor<R>,
    c: CubeTensor<R>,
    gamma: CubeTensor<R>,
    scale: CubeTensor<R>,
    initial: CubeTensor<R>,
) -> CubeTensor<R> {
    let v = into_contiguous(v);
    let da = into_contiguous(da);
    let b = into_contiguous(b);
    let c = into_contiguous(c);
    let gamma = into_contiguous(gamma);
    let scale = into_contiguous(scale);
    let initial = into_contiguous(initial);
    let [batch, nchunks, chunk_len, _, nheads, per_head_dim] = v.meta.shape().dims();
    let state_rank = b.meta.shape()[5];
    let tokens = nchunks * chunk_len;
    let checkpoint_count = tokens.div_ceil(RECONSTRUCTION_INTERVAL);
    let packed = empty(
        &v,
        Shape::new([
            batch,
            nheads,
            per_head_dim * (tokens + checkpoint_count * state_rank),
        ]),
    );
    let (cube_count, cube_dim) = launch_geometry(batch * nheads * per_head_dim);
    let dtype = v.dtype;
    single_ssd_scan_forward_kernel::launch::<R>(
        &packed.client,
        cube_count,
        cube_dim,
        v.into_tensor_arg(),
        da.into_tensor_arg(),
        b.into_tensor_arg(),
        c.into_tensor_arg(),
        gamma.into_tensor_arg(),
        scale.into_tensor_arg(),
        initial.into_tensor_arg(),
        packed.clone().into_tensor_arg(),
        state_rank,
        dtype_to_storage_type(dtype),
    );
    packed
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn single_ssd_scan_backward<R: CubeRuntime>(
    v: CubeTensor<R>,
    da: CubeTensor<R>,
    b: CubeTensor<R>,
    c: CubeTensor<R>,
    gamma: CubeTensor<R>,
    scale: CubeTensor<R>,
    packed: CubeTensor<R>,
    d_packed: CubeTensor<R>,
) -> (
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
) {
    let v = into_contiguous(v);
    let da = into_contiguous(da);
    let b = into_contiguous(b);
    let c = into_contiguous(c);
    let gamma = into_contiguous(gamma);
    let scale = into_contiguous(scale);
    let packed = into_contiguous(packed);
    let d_packed = into_contiguous(d_packed);
    let [batch, nchunks, chunk_len, _, nheads, per_head_dim] = v.meta.shape().dims();
    let state_rank = b.meta.shape()[5];
    let tokens = nchunks * chunk_len;
    let d_v = empty(&v, v.meta.shape().clone());
    let d_b = empty(&b, b.meta.shape().clone());
    let d_c = empty(&c, c.meta.shape().clone());
    let d_da = empty(&da, da.meta.shape().clone());
    let d_gamma = empty(&gamma, gamma.meta.shape().clone());
    let d_scale = empty(&scale, scale.meta.shape().clone());
    let d_initial = empty(&v, Shape::new([batch, nheads, per_head_dim, state_rank]));
    let contributions = empty(&v, Shape::new([batch, tokens, nheads, 3 * state_rank]));
    let dtype = v.dtype;
    let client = v.client.clone();

    let (value_count, value_dim) = launch_geometry(batch * nheads * per_head_dim);
    single_ssd_scan_backward_value_kernel::launch::<R>(
        &client,
        value_count,
        value_dim,
        v.clone().into_tensor_arg(),
        da.clone().into_tensor_arg(),
        b.clone().into_tensor_arg(),
        c.clone().into_tensor_arg(),
        gamma.clone().into_tensor_arg(),
        scale.clone().into_tensor_arg(),
        packed.clone().into_tensor_arg(),
        d_packed.clone().into_tensor_arg(),
        d_v.clone().into_tensor_arg(),
        d_initial.clone().into_tensor_arg(),
        state_rank,
        dtype_to_storage_type(dtype),
    );

    let (state_count, state_dim) = launch_geometry(batch * nheads * state_rank);
    single_ssd_scan_backward_state_kernel::launch::<R>(
        &client,
        state_count,
        state_dim,
        v.into_tensor_arg(),
        da.into_tensor_arg(),
        b.into_tensor_arg(),
        c.into_tensor_arg(),
        gamma.into_tensor_arg(),
        scale.into_tensor_arg(),
        packed.into_tensor_arg(),
        d_packed.into_tensor_arg(),
        d_b.clone().into_tensor_arg(),
        d_c.clone().into_tensor_arg(),
        contributions.clone().into_tensor_arg(),
        state_rank,
        per_head_dim,
        dtype_to_storage_type(dtype),
    );

    let (reduce_count, reduce_dim) = launch_geometry(batch * tokens * nheads);
    single_ssd_scan_backward_reduce_kernel::launch::<R>(
        &client,
        reduce_count,
        reduce_dim,
        contributions.into_tensor_arg(),
        d_da.clone().into_tensor_arg(),
        d_gamma.clone().into_tensor_arg(),
        d_scale.clone().into_tensor_arg(),
        state_rank,
        dtype_to_storage_type(dtype),
    );

    (d_v, d_da, d_b, d_c, d_gamma, d_scale, d_initial)
}

impl<R: CubeRuntime> Mamba3SingleSsdScanBackendExt for CubeBackend<R> {
    fn mamba3_single_ssd_scan(
        v_bnl1hp: CubeTensor<R>,
        da_bnlh: CubeTensor<R>,
        b_bnl1hr: CubeTensor<R>,
        c_bnl1hr: CubeTensor<R>,
        gamma_bnlh: CubeTensor<R>,
        scale_bnlh: CubeTensor<R>,
        initial_bhpr: CubeTensor<R>,
    ) -> CubeTensor<R> {
        single_ssd_scan_forward(
            v_bnl1hp,
            da_bnlh,
            b_bnl1hr,
            c_bnl1hr,
            gamma_bnlh,
            scale_bnlh,
            initial_bhpr,
        )
    }

    fn mamba3_single_ssd_scan_backward(
        v_bnl1hp: CubeTensor<R>,
        da_bnlh: CubeTensor<R>,
        b_bnl1hr: CubeTensor<R>,
        c_bnl1hr: CubeTensor<R>,
        gamma_bnlh: CubeTensor<R>,
        scale_bnlh: CubeTensor<R>,
        packed_bh_tnpr: CubeTensor<R>,
        d_packed_bh_tnpr: CubeTensor<R>,
    ) -> (
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
    ) {
        single_ssd_scan_backward(
            v_bnl1hp,
            da_bnlh,
            b_bnl1hr,
            c_bnl1hr,
            gamma_bnlh,
            scale_bnlh,
            packed_bh_tnpr,
            d_packed_bh_tnpr,
        )
    }
}
