use crate::mamba2::prelude::*;
use crate::mamba2::ssd::serial;
use burn::prelude::*;
use burn::tensor::{Tensor, TensorPrimitive, ops::FloatTensor};

impl<B: Backend + Mamba2BackendExt> Mamba2<B> {
    /// Forward pass for the Mamba-2 SSD module.
    ///
    /// Returns:
    /// - `y_bnlhp`.
    /// - `final_state_bhpr`.
    #[allow(non_snake_case)]
    pub fn ssd_serial_recalculated(
        input: super::super::SsdInput<B>,
    ) -> (Tensor<B, 5>, Tensor<B, 4>) {
        // Must use a backend-dependent method.
        //
        // For inference, this will untimately replicate Mamba2::ssd_serial;
        // For autodiff, this will call the custom implementation.

        let [batch, nchunks, chunk_len, nheads, _per_head_dim] = input.x_bnlhp.dims();
        let [.., ngroups, _state_rank] = input.b_bnlgr.dims();
        assert_ne!(ngroups, 0);
        assert_eq!(nheads % ngroups, 0);
        assert!(nchunks > 0, "sequence length must be at least 1");

        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented"
        );

        // ── Permutes ──────────────────────────────────────────────────────────────────
        // Note: dt_bnlh calculation (originally in Kernel 1) moved to Step 4 (before padding).
        let dt_discretized_bhnl = input.dt_bnlh.permute([0, 3, 1, 2]);
        assert_eq!(
            [batch, nheads, nchunks, chunk_len],
            dt_discretized_bhnl.dims()
        );

        // Always executes K1 in autodiff.
        // ── Kernel 1 ──────────────────────────────────────────────────────────────────
        // IO: (..) -> (da_cumsum_bhnl [used in K3+K5][*], da_chunk_end_bhn [used in K4][omitted][*])
        let (da_cumsum_bhnl, _da_chunk_end_bhn): (Tensor<B, 4>, Tensor<B, 3>) =
            serial::k1_ssd_chunk_cumsum(dt_discretized_bhnl.clone(), input.a_decay_h.clone());
        assert_eq!([batch, nheads, nchunks, chunk_len], da_cumsum_bhnl.dims());
        assert_eq!([batch, nheads, nchunks], _da_chunk_end_bhn.dims());

        let (y_bnlhp, final_state_bhpr) = <B as Mamba2BackendExt>::ssd_serial_recalculated(
            input.x_bnlhp.into_primitive().tensor(),
            dt_discretized_bhnl.into_primitive().tensor(),
            input.b_bnlgr.into_primitive().tensor(),
            input.c_bnlgr.into_primitive().tensor(),
            input.d_h.into_primitive().tensor(),
            input.initial_state_bhpr.into_primitive().tensor(),
            da_cumsum_bhnl.into_primitive().tensor(),
        );
        let y_bnlhp = Tensor::from_primitive(TensorPrimitive::Float(y_bnlhp));
        let final_state_bhpr = Tensor::from_primitive(TensorPrimitive::Float(final_state_bhpr));
        (y_bnlhp, final_state_bhpr)
    }
}

/// Extends the backend and wraps it for `burn`.
pub trait Mamba2BackendExt: burn::tensor::backend::Backend {
    /// Returns:
    /// - `y_bnlhp`.
    /// - `final_state_bhpr`.
    fn ssd_serial_recalculated(
        x_bnlhp: FloatTensor<Self>,
        // dt_bnlh: FloatTensor<Self>,
        dt_discretized_bhnl: FloatTensor<Self>,
        // a_decay_h: FloatTensor<Self>,
        b_bnlgr: FloatTensor<Self>,
        c_bnlgr: FloatTensor<Self>,
        d_h: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        da_cumsum_bhnl: FloatTensor<Self>, // from K1
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Default impl essentially replicates Mamba2::ssd_serial.

        let x_bnlhp: Tensor<Self, 5> = mk(x_bnlhp);
        let dt_discretized_bhnl: Tensor<Self, 4> = mk(dt_discretized_bhnl);
        let b_bnlgr: Tensor<Self, 5> = mk(b_bnlgr);
        let c_bnlgr: Tensor<Self, 5> = mk(c_bnlgr);
        let d_h: Tensor<Self, 1> = mk(d_h);
        let initial_state_bhpr: Tensor<Self, 4> = mk(initial_state_bhpr);
        let da_cumsum_bhnl: Tensor<Self, 4> = mk(da_cumsum_bhnl);

        let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
        let [.., ngroups, state_rank] = b_bnlgr.dims();
        assert_ne!(ngroups, 0);
        assert_eq!(nheads % ngroups, 0);
        assert!(nchunks > 0, "sequence length must be at least 1");
        // `heads_per_group` is called `nheads_ngroups_ratio` in every Triton kernel.
        // It is the compile-time constant used by GQA (Grouped Query Attention) to map
        // a head index to its B/C group: `group_idx = head_idx / heads_per_group`.

        // ── Kernel 1 ──────────────────────────────────────────────────────────
        // Note: This kernel is assumed to already have been called.
        // Just recalculate da_chunk_end_bhn.
        let da_chunk_end_bhn = da_cumsum_bhnl
            .clone()
            .slice(s![.., .., .., -1]) // da_cumsum_bhn1 // replay forward step 5
            .squeeze_dim::<3>(3); // replay forward step 6
        assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());

        // ── Kernel 2 ──────────────────────────────────────────────────────────
        // IO: (..) -> (cb_bngll [used in K5][!])
        let cb_bngll: Tensor<Self, 5> = serial::k2_ssd_bmm(c_bnlgr.clone(), b_bnlgr.clone());
        assert_eq!(
            [batch, nchunks, ngroups, chunk_len, chunk_len],
            cb_bngll.dims()
        );
        // Note: cb_bngll is then only used by Kernel 5.

        // ── Kernel 3 ──────────────────────────────────────────────────────────
        // IO: (..) -> (intra_chunk_state_bnhpr [used in K4][!])
        let intra_chunk_state_bnhpr: Tensor<Self, 5> = serial::k3_ssd_chunk_state(
            x_bnlhp.clone(),
            b_bnlgr.clone(),
            da_cumsum_bhnl.clone(),
            dt_discretized_bhnl.clone(),
        );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            intra_chunk_state_bnhpr.dims()
        );

        // ── Kernel 4 ──────────────────────────────────────────────────────────
        // IO: (..) -> (chunk_input_state_bnhpr [used in K5][!], final_state_bhpr [final output])
        let (chunk_input_state_bnhpr, final_state_bhpr): (Tensor<Self, 5>, Tensor<Self, 4>) =
            serial::k4_ssd_state_passing(
                intra_chunk_state_bnhpr.clone(),
                da_chunk_end_bhn.clone(),
                initial_state_bhpr,
            );
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            chunk_input_state_bnhpr.dims()
        );
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            final_state_bhpr.dims()
        );

        // ── Kernel 5 ──────────────────────────────────────────────────────────
        let y_bnlhp: Tensor<Self, 5> = serial::k5_ssd_chunk_scan(
            da_cumsum_bhnl,
            dt_discretized_bhnl,
            x_bnlhp,
            c_bnlgr,
            cb_bngll,
            chunk_input_state_bnhpr,
            d_h,
        );
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_bnlhp.dims()
        );

        let y_bnlhp = y_bnlhp.into_primitive().tensor();
        let final_state_bhpr = final_state_bhpr.into_primitive().tensor();
        (y_bnlhp, final_state_bhpr)
    }
}

// For inference and for any backend, fallback to the default impl (to Mamba2::ssd_serial).
//
// impl<B: Backend> Mamba2BackendExt for B {}
// Note: cannot generally implement as above as it conflicts with the custom autodiff impl.
// So it's necessary to implement for each backend.
//
// TODO: somehow avoid leaking backend-* features into the library
#[cfg(feature = "backend-ndarray")]
impl<F, I> Mamba2BackendExt for burn::backend::NdArray<F, I> {}
// #[cfg(feature = "backend-flex")]
// impl Mamba2BackendExt for burn::backend::Flex {}
#[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
impl<F, I> Mamba2BackendExt for burn::backend::libtorch::LibTorch<F, I> {}
#[cfg(feature = "backend-remote")]
impl<F, I> Mamba2BackendExt for burn::backend::RemoteBackend<F, I> {}
// impl for cubecl backends
#[cfg(feature = "cubecl")]
mod cubecl {
    use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement};
    impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> super::Mamba2BackendExt
        for CubeBackend<R, F, I, BT>
    {
    }
}

// impl for fusion backends — delegates to the default impl, which runs the serial
// computation using the inner backend's standard tensor operations.
#[cfg(feature = "fusion")]
mod fusion {
    use burn_fusion::{Fusion, FusionBackend};
    impl<B: FusionBackend + super::Mamba2BackendExt> super::Mamba2BackendExt for Fusion<B> {}
}

/// Marker for autodiff-compatible backends that are valid for the custom backward implementation.
#[cfg(feature = "autodiff")]
pub trait Mamba2AutodiffBackendExt:
    Backend + Mamba2BackendExt + burn::tensor::backend::AutodiffBackend
{
}
// Any autodiff-compatible backend is valid with our custom implementation
//
// Note: This is just a marker. The actual custom implementation is at super::serial_recalculated::backward,
// a custom Mamba2BackendExt implementation.
#[cfg(feature = "autodiff")]
impl<B: Backend + Mamba2BackendExt> Mamba2AutodiffBackendExt for burn::backend::Autodiff<B> {}

/// Conversion helper.
pub(crate) fn mk<B: Backend, const D: usize>(p: FloatTensor<B>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(p))
}
