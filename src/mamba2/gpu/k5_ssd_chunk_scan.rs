//! Kernel 5: SSD chunk-scan forward pass.
//!
//! For each output position `l` in chunk `c`, accumulates three contributions
//! corresponding to the three colour regions of the semiseparable matrix M
//! (see Part III of the paper, "Semiseparable Matrix M Block Decomposition"):
//!
//!   **BLUE**   – history from the SSM state entering this chunk:
//!     `y_blue[l, p]  = exp(dA_cumsum[l]) * Σ_n  C[l,n] * state_in[p,n]`
//!
//!   **ORANGE** – causal intra-chunk attention (source positions s ≤ l):
//!     `y_orange[l, p] = Σ_{s≤l} CB[l,s] · exp(dA[l]−dA[s]) · dt[s] · x[s,p]`
//!
//!   **SKIP**   – D residual connection (D_HAS_HDIM=False: scalar per head):
//!     `y_skip[l, p]  = D[h] · x[l,p]`
//!
//! Reference: `_chunk_scan_fwd_kernel` in `ssd_chunk_scan.py`.
//!
//! # Optimisation road-map (first iteration: naive scalar kernel)
//!
//! Both BLUE and ORANGE reduce to batched matrix multiplications that can be
//! accelerated with `cubek::matmul` in a future tiled implementation:
//!
//! * **BLUE** (per batch·chunk·head slice):
//!   `C[L, R]  @  state_in^T[R, P]  →  [L, P]`
//!   Requires a preprocessing step to broadcast C from *ngroups* to *nheads*
//!   (GQA) and to transpose `state_in`.  After the matmul, each row `l` is
//!   scaled by `exp(dA_cumsum[l])`.
//!
//! * **ORANGE** (per batch·chunk·head slice):
//!   `CB_weighted[L, L]  @  X[L, P]  →  [L, P]`
//!   Requires a preprocessing kernel that fills the lower-triangular
//!   `CB_weighted[l, s] = CB[l,s] · exp(dA[l]−dA[s]) · dt[s]  (s ≤ l, else 0)`.
//!   After that the matmul is a standard dense×dense product.
//!
//! Once those kernels exist, the per-element loops in `kernel::forward` can be
//! replaced by calls to `NaiveBatchMatmulFamily::launch_unchecked` (or
//! `cubek::matmul::launch::launch_ref` at the host level), reducing the
//! arithmetic intensity per unit and enabling higher occupancy.

/// Extends the backend (for `burn_cubecl`) and wraps it for `burn`.
pub mod backend_ext {
    use burn::tensor::{Tensor, TensorPrimitive, ops::FloatTensor};

    // Wraps the extended backend for `burn`.
    //
    /// Returns `y_bnlhp`.
    ///
    /// # Shapes
    /// | argument                   | layout                                               |
    /// |----------------------------|------------------------------------------------------|
    /// | `da_cumsum_bhnl`           | `[batch, nheads, nchunks, chunk_len]`                |
    /// | `dt_discretized_bhnl`      | `[batch, nheads, nchunks, chunk_len]`                |
    /// | `x_bnlhp`                  | `[batch, nchunks, chunk_len, nheads, per_head_dim]`  |
    /// | `c_bnlgr`                  | `[batch, nchunks, chunk_len, ngroups, state_rank]`   |
    /// | `cb_bngll`                 | `[batch, nchunks, ngroups, chunk_len, chunk_len]`    |
    /// | `chunk_input_states_bnhpr` | `[batch, nchunks, nheads, per_head_dim, state_rank]` |
    /// | `d_h`                      | `[nheads]`                                           |
    #[cfg(feature = "cubecl")]
    pub fn k5_ssd_chunk_scan<B: BackendExt>(
        da_cumsum_bhnl: Tensor<B, 4>,
        dt_discretized_bhnl: Tensor<B, 4>,
        x_bnlhp: Tensor<B, 5>,
        c_bnlgr: Tensor<B, 5>,
        cb_bngll: Tensor<B, 5>,
        chunk_input_states_bnhpr: Tensor<B, 5>,
        d_h: Tensor<B, 1>,
    ) -> Tensor<B, 5> {
        let y_bnlhp = B::k5_ssd_chunk_scan(
            da_cumsum_bhnl.into_primitive().tensor(),
            dt_discretized_bhnl.into_primitive().tensor(),
            x_bnlhp.into_primitive().tensor(),
            c_bnlgr.into_primitive().tensor(),
            cb_bngll.into_primitive().tensor(),
            chunk_input_states_bnhpr.into_primitive().tensor(),
            d_h.into_primitive().tensor(),
        );
        Tensor::from_primitive(TensorPrimitive::Float(y_bnlhp))
    }

    // Wraps the naive burn reference for `burn`.
    //
    /// Returns `y_bnlhp`.
    pub fn k5_ssd_chunk_scan_naive<B: BackendExt>(
        da_cumsum_bhnl: Tensor<B, 4>,
        dt_discretized_bhnl: Tensor<B, 4>,
        x_bnlhp: Tensor<B, 5>,
        c_bnlgr: Tensor<B, 5>,
        cb_bngll: Tensor<B, 5>,
        chunk_input_states_bnhpr: Tensor<B, 5>,
        d_h: Tensor<B, 1>,
    ) -> Tensor<B, 5> {
        let y_bnlhp = B::k5_ssd_chunk_scan_naive(
            da_cumsum_bhnl.into_primitive().tensor(),
            dt_discretized_bhnl.into_primitive().tensor(),
            x_bnlhp.into_primitive().tensor(),
            c_bnlgr.into_primitive().tensor(),
            cb_bngll.into_primitive().tensor(),
            chunk_input_states_bnhpr.into_primitive().tensor(),
            d_h.into_primitive().tensor(),
        );
        Tensor::from_primitive(TensorPrimitive::Float(y_bnlhp))
    }

    // Backend extended with K5.
    pub trait BackendExt: burn::tensor::backend::Backend {
        /// Returns `y_bnlhp`.
        #[cfg(feature = "cubecl")]
        fn k5_ssd_chunk_scan(
            da_cumsum_bhnl: FloatTensor<Self>,
            dt_discretized_bhnl: FloatTensor<Self>,
            x_bnlhp: FloatTensor<Self>,
            c_bnlgr: FloatTensor<Self>,
            cb_bngll: FloatTensor<Self>,
            chunk_input_states_bnhpr: FloatTensor<Self>,
            d_h: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            unimplemented!("Backend not supported")
        }

        // Naive burn fallback.
        /// Returns `y_bnlhp`.
        fn k5_ssd_chunk_scan_naive(
            da_cumsum_bhnl: FloatTensor<Self>,
            dt_discretized_bhnl: FloatTensor<Self>,
            x_bnlhp: FloatTensor<Self>,
            c_bnlgr: FloatTensor<Self>,
            cb_bngll: FloatTensor<Self>,
            chunk_input_states_bnhpr: FloatTensor<Self>,
            d_h: FloatTensor<Self>,
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
pub use backend_ext::k5_ssd_chunk_scan;

/// Backend implementation.
///
/// Wraps the `cubecl` kernel for `burn_cubecl`.
#[cfg(feature = "cubecl")]
pub mod forward {
    use super::backend_ext::BackendExt;
    use burn::cubecl::{CubeCount, CubeDim};
    use burn::tensor::{Shape, ops::FloatTensor};
    use burn_cubecl::{
        CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement,
        kernel::into_contiguous, tensor::CubeTensor,
    };

    // Number of GPU units (threads) per cube.
    //
    // Each unit computes exactly one element of the output tile, identified by a
    // flat index `flat_idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X`.  The pair
    // `(l, p) = (flat_idx / per_head_dim, flat_idx % per_head_dim)` is the
    // output chunk-position and per-head channel for that unit.
    //
    // FUTURE: switching to a 2-D CubeDim (TILE_M × TILE_N) and using shared
    // memory to cache C, state_in, CB, and X tiles would allow each cube to
    // leverage `NaiveMatmul::execute` for the BLUE and ORANGE reductions,
    // amortising global-memory latency across the tile.
    const BLOCK_SIZE: u32 = 256;

    impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> BackendExt
        for CubeBackend<R, F, I, BT>
    {
        /// Returns `y_bnlhp : [batch, nchunks, chunk_len, nheads, per_head_dim]`.
        fn k5_ssd_chunk_scan(
            da_cumsum_bhnl: FloatTensor<Self>,
            dt_discretized_bhnl: FloatTensor<Self>,
            x_bnlhp: FloatTensor<Self>,
            c_bnlgr: FloatTensor<Self>,
            cb_bngll: FloatTensor<Self>,
            chunk_input_states_bnhpr: FloatTensor<Self>,
            d_h: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            let client = x_bnlhp.client.clone();
            let device = x_bnlhp.device.clone();

            // Assert same device (spot-check a subset).
            x_bnlhp.assert_is_on_same_device(&da_cumsum_bhnl);
            x_bnlhp.assert_is_on_same_device(&chunk_input_states_bnhpr);

            // Ensure all inputs are contiguous before handing to the kernel.
            let da_cumsum_bhnl = into_contiguous(da_cumsum_bhnl);
            let dt_discretized_bhnl = into_contiguous(dt_discretized_bhnl);
            let x_bnlhp = into_contiguous(x_bnlhp);
            let c_bnlgr = into_contiguous(c_bnlgr);
            let cb_bngll = into_contiguous(cb_bngll);
            let chunk_input_states_bnhpr = into_contiguous(chunk_input_states_bnhpr);
            let d_h = into_contiguous(d_h);

            // x_bnlhp : [batch, nchunks, chunk_len, nheads, per_head_dim]
            let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.meta.shape().dims();

            // ── Dispatch ─────────────────────────────────────────────────────────
            //
            // Each unit owns one (l, p) output element for a fixed (batch, chunk, head).
            //
            // flat range   = [0, chunk_len * per_head_dim)
            // CUBE_POS_X   = tile index over that flat range
            // CUBE_POS_Y   = batch_idx * nchunks + chunk_idx
            // CUBE_POS_Z   = head_idx
            let elems_per_bch = (chunk_len * per_head_dim) as u32;
            let cube_count_x = elems_per_bch.div_ceil(BLOCK_SIZE);

            // Allocate output tensor.
            let y_shape = Shape::from([batch, nchunks, chunk_len, nheads, per_head_dim]);
            let y_buffer = client.empty(y_shape.num_elements() * core::mem::size_of::<F>());
            let y_bnlhp = CubeTensor::new_contiguous(
                client.clone(),
                device.clone(),
                y_shape,
                y_buffer,
                F::dtype(),
            );

            unsafe {
                super::kernel::forward::launch_unchecked::<F, R>(
                    &client,
                    CubeCount::Static(cube_count_x, (batch * nchunks) as u32, nheads as u32),
                    CubeDim::new_1d(BLOCK_SIZE),
                    da_cumsum_bhnl.into_tensor_arg(),
                    dt_discretized_bhnl.into_tensor_arg(),
                    x_bnlhp.into_tensor_arg(),
                    c_bnlgr.into_tensor_arg(),
                    cb_bngll.into_tensor_arg(),
                    chunk_input_states_bnhpr.into_tensor_arg(),
                    d_h.into_tensor_arg(),
                    y_bnlhp.clone().into_tensor_arg(),
                    BLOCK_SIZE, // #[comptime]
                )
            };

            y_bnlhp
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
        fn k5_ssd_chunk_scan(
            da_cumsum_bhnl: FloatTensor<Self>,
            dt_discretized_bhnl: FloatTensor<Self>,
            x_bnlhp: FloatTensor<Self>,
            c_bnlgr: FloatTensor<Self>,
            cb_bngll: FloatTensor<Self>,
            chunk_input_states_bnhpr: FloatTensor<Self>,
            d_h: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            unimplemented!("AutodiffBackend not supported")
        }

        // Naive fallback.
        fn k5_ssd_chunk_scan_naive(
            da_cumsum_bhnl: FloatTensor<Self>,
            dt_discretized_bhnl: FloatTensor<Self>,
            x_bnlhp: FloatTensor<Self>,
            c_bnlgr: FloatTensor<Self>,
            cb_bngll: FloatTensor<Self>,
            chunk_input_states_bnhpr: FloatTensor<Self>,
            d_h: FloatTensor<Self>,
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

    /// SSD chunk-scan: compute `y[batch, chunk, l, head, p]` for all elements.
    ///
    /// Each unit independently computes one output element as:
    /// ```text
    ///   y[l, p] = BLUE[l, p] + ORANGE[l, p] + SKIP[l, p]
    /// ```
    ///
    /// # Tensor shapes
    /// | arg                   | layout                                               |
    /// |-----------------------|------------------------------------------------------|
    /// | `da_cumsum`           | `[batch, nheads, nchunks, chunk_len]`                |
    /// | `dt`                  | `[batch, nheads, nchunks, chunk_len]`                |
    /// | `x`                   | `[batch, nchunks, chunk_len, nheads, per_head_dim]`  |
    /// | `c`                   | `[batch, nchunks, chunk_len, ngroups, state_rank]`   |
    /// | `cb`                  | `[batch, nchunks, ngroups, chunk_len, chunk_len]`    |
    /// | `chunk_input_states`  | `[batch, nchunks, nheads, per_head_dim, state_rank]` |
    /// | `d`                   | `[nheads]`                                           |
    /// | `y`                   | `[batch, nchunks, chunk_len, nheads, per_head_dim]`  |
    ///
    /// # Dispatch
    /// ```text
    /// CubeCount  = (ceil(chunk_len * per_head_dim / block_size), batch * nchunks, nheads)
    /// CubeDim    = (block_size, 1, 1)
    ///
    /// flat_idx   = CUBE_POS_X * block_size + UNIT_POS_X
    /// l          = flat_idx / per_head_dim   (output chunk position)
    /// p          = flat_idx % per_head_dim   (per-head channel index)
    ///
    /// batch_idx  = CUBE_POS_Y / nchunks
    /// chunk_idx  = CUBE_POS_Y % nchunks
    /// head_idx   = CUBE_POS_Z
    /// ```
    #[cube(launch_unchecked)]
    pub fn forward<F: Float>(
        da_cumsum: &Tensor<F>,          // [batch, nheads, nchunks, chunk_len]
        dt: &Tensor<F>,                 // [batch, nheads, nchunks, chunk_len]
        x: &Tensor<F>,                  // [batch, nchunks, chunk_len, nheads, per_head_dim]
        c: &Tensor<F>,                  // [batch, nchunks, chunk_len, ngroups, state_rank]
        cb: &Tensor<F>,                 // [batch, nchunks, ngroups, chunk_len(tgt), chunk_len(src)]
        chunk_input_states: &Tensor<F>, // [batch, nchunks, nheads, per_head_dim, state_rank]
        d: &Tensor<F>,                  // [nheads]
        y: &mut Tensor<F>,              // [batch, nchunks, chunk_len, nheads, per_head_dim]
        #[comptime] block_size: u32,
    ) {
        // ── Index derivation ────────────────────────────────────────────────────
        let flat_idx = (CUBE_POS_X * block_size + UNIT_POS_X) as usize;
        let batch_chunk_idx = CUBE_POS_Y as usize;
        let head_idx = CUBE_POS_Z as usize;

        // Runtime shape queries.
        // x            : [batch, nchunks, chunk_len, nheads, per_head_dim]
        // da_cumsum    : [batch, nheads,  nchunks,   chunk_len]
        let chunk_len = x.shape(2);
        let per_head_dim = x.shape(4);
        let state_rank = chunk_input_states.shape(4);
        let nchunks = da_cumsum.shape(2);
        let nheads = da_cumsum.shape(1);
        let ngroups = c.shape(3);

        // Decompose flat_idx → (output chunk position l, per-head channel p).
        let output_pos = flat_idx / per_head_dim; // l ∈ [0, chunk_len)
        let per_head_dim_idx = flat_idx % per_head_dim; // p ∈ [0, per_head_dim)

        // Guard: the last cube may be partially filled.
        if output_pos >= chunk_len {
            terminate!();
        }

        // Decompose CUBE_POS_Y → (batch_idx, chunk_idx).
        let batch_idx = batch_chunk_idx / nchunks;
        let chunk_idx = batch_chunk_idx % nchunks;

        // GQA: map head index → group index.
        let heads_per_group = nheads / ngroups;
        let group_idx = head_idx / heads_per_group;

        // ── Pre-compute stride-scaled base offsets ──────────────────────────────
        //
        // da_cumsum : [batch, nheads, nchunks, chunk_len]
        let da_s0 = da_cumsum.stride(0); // batch
        let da_s1 = da_cumsum.stride(1); // head
        let da_s2 = da_cumsum.stride(2); // chunk
        let da_s3 = da_cumsum.stride(3); // position
        // Base for (batch, head, chunk); position l or s added in loops.
        let da_base = batch_idx * da_s0 + head_idx * da_s1 + chunk_idx * da_s2;

        // dt : [batch, nheads, nchunks, chunk_len]
        let dt_s2 = dt.stride(2);
        let dt_s3 = dt.stride(3);
        let dt_base = batch_idx * dt.stride(0) + head_idx * dt.stride(1) + chunk_idx * dt_s2;

        // x : [batch, nchunks, chunk_len, nheads, per_head_dim]
        let x_s2 = x.stride(2); // chunk-position stride (the inner loop stride)
        // Base for (batch, chunk, head, p); chunk-pos l or s added in loops.
        let x_base_bhp = batch_idx * x.stride(0)
            + chunk_idx * x.stride(1)
            + head_idx * x.stride(3)
            + per_head_dim_idx * x.stride(4);

        // c : [batch, nchunks, chunk_len, ngroups, state_rank]
        let c_s4 = c.stride(4); // state_rank stride
        // Base for (batch, chunk, l, group); state index n added in BLUE loop.
        let c_base = batch_idx * c.stride(0)
            + chunk_idx * c.stride(1)
            + output_pos * c.stride(2)
            + group_idx * c.stride(3);

        // cb : [batch, nchunks, ngroups, chunk_len(target), chunk_len(source)]
        let cb_s4 = cb.stride(4); // source-position stride
        // Base for (batch, chunk, group, target=l); source s added in ORANGE loop.
        let cb_base = batch_idx * cb.stride(0)
            + chunk_idx * cb.stride(1)
            + group_idx * cb.stride(2)
            + output_pos * cb.stride(3);

        // chunk_input_states : [batch, nchunks, nheads, per_head_dim, state_rank]
        let st_s4 = chunk_input_states.stride(4); // state_rank stride
        // Base for (batch, chunk, head, p); state index n added in BLUE loop.
        let st_base = batch_idx * chunk_input_states.stride(0)
            + chunk_idx * chunk_input_states.stride(1)
            + head_idx * chunk_input_states.stride(2)
            + per_head_dim_idx * chunk_input_states.stride(3);

        // y : [batch, nchunks, chunk_len, nheads, per_head_dim]
        let y_idx = batch_idx * y.stride(0)
            + chunk_idx * y.stride(1)
            + output_pos * y.stride(2)
            + head_idx * y.stride(3)
            + per_head_dim_idx * y.stride(4);

        // ── dA_cumsum at output position l ──────────────────────────────────────
        //
        // Used in both BLUE (as exp scale) and ORANGE (as decay reference).
        let da_cumsum_l = da_cumsum[da_base + output_pos * da_s3];

        // ── BLUE contribution ───────────────────────────────────────────────────
        //
        //   y_blue[l, p] = exp(dA_cumsum[l]) * Σ_n  C[l, n] * state_in[p, n]
        //
        // The inner Σ_n is a dot-product of row C[l, :] with column state_in[p, :].
        //
        // FUTURE tiled version: load tiles of C and state_in into shared memory
        // and compute via NaiveMatmul::execute (see module-level doc-comment).
        let mut blue = F::new(0.0);
        let mut n: u32 = 0;
        while n < state_rank as u32 {
            let n_us = n as usize;
            // C[batch, chunk, l, group, n]
            let c_val = c[c_base + n_us * c_s4];
            // state_in[batch, chunk, head, p, n]
            let st_val = chunk_input_states[st_base + n_us * st_s4];
            blue += c_val * st_val;
            n += 1;
        }
        // Scale by the state-decay accumulated from chunk start to position l.
        // Triton: `acc = tl.dot(C, prev_states) * scale_m[:, None]`
        //         where `scale_m = tl.exp(dA_cs_m)`
        blue *= F::exp(da_cumsum_l);

        // ── ORANGE contribution ─────────────────────────────────────────────────
        //
        //   y_orange[l, p] = Σ_{s≤l} CB[l,s] · exp(dA[l]−dA[s]) · dt[s] · x[s,p]
        //
        // IS_CAUSAL=True: only source positions s ≤ l contribute (lower-triangular).
        //
        // Triton inner loop (orange):
        //   for k in range(0, K_MAX, BLOCK_K):        K_MAX = (pid_m+1)*BM
        //     cb     *= exp(dA_cs_m[:,None] - dA_cs_k[None,:])
        //     cb     *= dt_k
        //     mask    = offs_m[:,None] >= offs_k[None,:]
        //     cb      = where(mask, cb, 0)
        //     acc    += dot(cb, x)
        //
        // FUTURE tiled version: precompute cb_weighted in a separate kernel,
        // then call NaiveMatmul::execute on cb_weighted @ X.
        let mut orange = F::new(0.0);
        // Source positions s ∈ [0, l].
        let mut s: u32 = 0;
        while s <= output_pos as u32 {
            let s_us = s as usize;
            // CB[batch, chunk, group, l, s]  (target=l, source=s)
            let cb_val = cb[cb_base + s_us * cb_s4];
            // dA_cumsum[batch, head, chunk, s]
            let da_cumsum_s = da_cumsum[da_base + s_us * da_s3];
            // dt[batch, head, chunk, s]
            let dt_val = dt[dt_base + s_us * dt_s3];
            // x[batch, chunk, s, head, p]
            let x_val = x[x_base_bhp + s_us * x_s2];
            // Decay: exp(dA_cumsum[l] - dA_cumsum[s])
            // Both dA_cumsum values are ≤ 0 with cumsum[l] ≤ cumsum[s]
            // (target accumulated more decay steps), so the difference is ≤ 0
            // and exp(·) ∈ (0, 1] as required by the causal structure.
            orange += cb_val * F::exp(da_cumsum_l - da_cumsum_s) * dt_val * x_val;
            s += 1;
        }

        // ── SKIP contribution ───────────────────────────────────────────────────
        //
        //   y_skip[l, p] = D[h] · x[l, p]
        //
        // D_HAS_HDIM=False: D is a scalar per head, shape [nheads].
        // Triton: `D = tl.load(D_ptr + pid_h * stride_D_head)`
        let x_lp = x[x_base_bhp + output_pos * x_s2];
        // d is 1-D contiguous: d[head_idx] = d[head_idx * stride(0)].
        // For contiguous storage stride(0) == 1, but we use the stride to be safe.
        let skip = d[head_idx * d.stride(0)] * x_lp;

        // ── Write output ────────────────────────────────────────────────────────
        y[y_idx] = blue + orange + skip;
    }
}

/// Naive (forward) implementation in `burn`.
pub mod naive {
    use burn::prelude::*;

    /// Returns `y_bnlhp`.
    ///
    /// # Shapes
    /// | argument                   | layout                                               |
    /// |----------------------------|------------------------------------------------------|
    /// | `da_cumsum_bhnl`           | `[batch, nheads, nchunks, chunk_len]`                |
    /// | `dt_discretized_bhnl`      | `[batch, nheads, nchunks, chunk_len]`                |
    /// | `x_bnlhp`                  | `[batch, nchunks, chunk_len, nheads, per_head_dim]`  |
    /// | `c_bnlgr`                  | `[batch, nchunks, chunk_len, ngroups, state_rank]`   |
    /// | `cb_bngll`                 | `[batch, nchunks, ngroups, chunk_len, chunk_len]`    |
    /// | `chunk_input_states_bnhpr` | `[batch, nchunks, nheads, per_head_dim, state_rank]` |
    /// | `d_h`                      | `[nheads]`                                           |
    pub fn forward<B: Backend>(
        da_cumsum_bhnl: Tensor<B, 4>,
        dt_discretized_bhnl: Tensor<B, 4>,
        x_bnlhp: Tensor<B, 5>,
        c_bnlgr: Tensor<B, 5>,
        cb_bngll: Tensor<B, 5>,
        chunk_input_states_bnhpr: Tensor<B, 5>,
        d_h: Tensor<B, 1>,
    ) -> Tensor<B, 5> {
        let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
        let [_b, _n, _l, ngroups, state_rank] = c_bnlgr.dims();
        let heads_per_group = nheads / ngroups;
        let device = x_bnlhp.device();

        // Rearrange inputs to the common [batch, nchunks, nheads, ...] ordering used below.
        let da_cumsum_bnhl = da_cumsum_bhnl.permute([0, 2, 1, 3]); // [b,n,h,l]
        let dt_bnhl = dt_discretized_bhnl.permute([0, 2, 1, 3]); // [b,n,h,l]
        let x_bnhlp = x_bnlhp.clone().permute([0, 1, 3, 2, 4]); // [b,n,h,l,p]

        // GQA: expand C  [b,n,l,g,r] → [b,n,h,l,r].
        let c_bnhlr = c_bnlgr
            .unsqueeze_dim::<6>(4) // [b,n,l,g,1,r]
            .expand([
                batch,
                nchunks,
                chunk_len,
                ngroups,
                heads_per_group,
                state_rank,
            ])
            .reshape([batch, nchunks, chunk_len, nheads, state_rank])
            .permute([0, 1, 3, 2, 4]); // [b,n,h,l,r]

        // GQA: expand CB [b,n,g,l,l] → [b,n,h,l,l].
        let cb_bnhll = cb_bngll
            .unsqueeze_dim::<6>(3) // [b,n,g,1,l,l]
            .expand([
                batch,
                nchunks,
                ngroups,
                heads_per_group,
                chunk_len,
                chunk_len,
            ])
            .reshape([batch, nchunks, nheads, chunk_len, chunk_len]);

        // ── BLUE: exp(dA[l]) · C[l,:] @ state_in^T ─────────────────────────────
        //
        //   blue[b,n,h,l,p] = exp(da[b,n,h,l]) · Σ_r  c[b,n,h,l,r] · state[b,n,h,p,r]
        //
        //   [b,n,h,l,r] @ [b,n,h,r,p]  →  [b,n,h,l,p]
        let blue_bnhlp = c_bnhlr
            .matmul(chunk_input_states_bnhpr.permute([0, 1, 2, 4, 3]))  // state^T [b,n,h,r,p]
            * da_cumsum_bnhl
                .clone()
                .exp()
                .unsqueeze_dim::<5>(4)
                .expand([batch, nchunks, nheads, chunk_len, per_head_dim]);

        // ── ORANGE: causal CB_weighted @ X ──────────────────────────────────────
        //
        //   orange[b,n,h,l,p] = Σ_{s≤l} CB[l,s] · exp(da[l]-da[s]) · dt[s] · x[s,p]
        //
        // Precompute the full lower-triangular weight matrix, then do a single matmul.
        let da_target_bnhll = da_cumsum_bnhl
            .clone()
            .unsqueeze_dim::<5>(4) // [b,n,h,l,1]
            .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
        let da_source_bnhll = da_cumsum_bnhl
            .unsqueeze_dim::<5>(3) // [b,n,h,1,l]
            .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
        let dt_source_bnhll = dt_bnhl
            .unsqueeze_dim::<5>(3) // [b,n,h,1,l]  (broadcast over target positions)
            .expand([batch, nchunks, nheads, chunk_len, chunk_len]);

        // Causal mask (0 above the main diagonal, 1 elsewhere).
        let causal_mask_bnhll =
            Tensor::ones([batch, nchunks, nheads, chunk_len, chunk_len], &device).tril(0);

        //   [b,n,h,l,l] @ [b,n,h,l,p]  →  [b,n,h,l,p]
        let orange_bnhlp = (cb_bnhll
            * (da_target_bnhll - da_source_bnhll).exp()
            * dt_source_bnhll
            * causal_mask_bnhll)
            .matmul(x_bnhlp);

        // ── SKIP: D[h] · x[l,p] ─────────────────────────────────────────────────
        //
        //   D_HAS_HDIM = False: D is a scalar per head, shape [nheads].
        //   Triton: `acc += x_residual * D`
        let skip_bnlhp = d_h.reshape([1, 1, 1, nheads, 1]).expand([
            batch,
            nchunks,
            chunk_len,
            nheads,
            per_head_dim,
        ]) * x_bnlhp;

        // Permute BLUE + ORANGE from [b,n,h,l,p] back to [b,n,l,h,p], then add SKIP.
        (blue_bnhlp + orange_bnhlp).permute([0, 1, 3, 2, 4]) + skip_bnlhp // y_bnlhp
    }
}

#[cfg(test)]
#[cfg(feature = "backend-cuda")]
mod test {
    use super::*;
    pub type TestBackend = burn::backend::Cuda<f32, i32>;
    use burn::tensor::Tolerance;
    use burn::tensor::{Distribution, Tensor};

    /// Smoke test: cubecl kernel output matches the burn naive reference.
    #[test]
    fn kernel_naive_comparison_k5() {
        let device = Default::default();
        // Small shapes for quick iteration.
        let (batch, nheads, ngroups, nchunks, chunk_len, per_head_dim, state_rank) =
            (2, 4, 2, 3, 16, 8, 4);
        let heads_per_group = nheads / ngroups;
        assert_eq!(heads_per_group * ngroups, nheads);

        let da_cumsum_bhnl: Tensor<TestBackend, 4> = Tensor::random(
            [batch, nheads, nchunks, chunk_len],
            Distribution::Uniform(-1.0, 0.0),
            &device,
        );
        let dt_bhnl: Tensor<TestBackend, 4> = Tensor::random(
            [batch, nheads, nchunks, chunk_len],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let x_bnlhp: Tensor<TestBackend, 5> = Tensor::random(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            Distribution::Default,
            &device,
        );
        let c_bnlgr: Tensor<TestBackend, 5> = Tensor::random(
            [batch, nchunks, chunk_len, ngroups, state_rank],
            Distribution::Default,
            &device,
        );
        let cb_bngll: Tensor<TestBackend, 5> = Tensor::random(
            [batch, nchunks, ngroups, chunk_len, chunk_len],
            Distribution::Default,
            &device,
        );
        let states_bnhpr: Tensor<TestBackend, 5> = Tensor::random(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            Distribution::Default,
            &device,
        );
        let d_h: Tensor<TestBackend, 1> = Tensor::random([nheads], Distribution::Default, &device);

        // Naive (burn reference).
        let y_naive = naive::forward::<TestBackend>(
            da_cumsum_bhnl.clone(),
            dt_bhnl.clone(),
            x_bnlhp.clone(),
            c_bnlgr.clone(),
            cb_bngll.clone(),
            states_bnhpr.clone(),
            d_h.clone(),
        );
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_naive.dims()
        );
        println!("naive: {}", &y_naive);

        // CubeCL kernel.
        let y_cubecl: Tensor<TestBackend, 5> = backend_ext::k5_ssd_chunk_scan(
            da_cumsum_bhnl.clone(),
            dt_bhnl.clone(),
            x_bnlhp.clone(),
            c_bnlgr.clone(),
            cb_bngll.clone(),
            states_bnhpr.clone(),
            d_h.clone(),
        );
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_cubecl.dims()
        );
        println!("cubecl: {}", &y_cubecl);

        // Numerical comparison.
        y_naive
            .to_data()
            .assert_approx_eq::<f32>(&y_cubecl.to_data(), Tolerance::default());
    }
}
