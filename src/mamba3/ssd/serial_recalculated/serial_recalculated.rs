#![allow(non_snake_case)]

use crate::mamba3::prelude::*;
use crate::mamba3::ssd::serial;
use burn::prelude::*;
use burn::tensor::{Tensor, TensorPrimitive, ops::FloatTensor};

impl<B: Backend + Mamba3BackendExt> Mamba3<B> {
    /// MIMO-first Serial SSD with recalculated backward.
    ///
    /// Computes K1 eagerly (so the cumsum is available for the backward pass),
    /// then delegates the remaining computation to [`Mamba3BackendExt::ssd_serial_recalculated`]
    /// which can provide a memory-efficient custom backward for supported backends.
    ///
    /// Falls back to the standard K2-K5 serial computation on unsupported backends.
    ///
    /// # Returns
    /// - `y_bnlrhp`:        `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn ssd_serial_recalculated(
        input: super::super::SsdInput<B>,
    ) -> (Tensor<B, 6>, Tensor<B, 4>) {
        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented for ssd_serial_recalculated"
        );

        // K1 runs on the tracked compute graph so the cumsum is available during backward.
        let (da_cumsum_bhnl, _da_chunk_end_bhn): (Tensor<B, 4>, Tensor<B, 3>) =
            serial::k1_ssd_chunk_cumsum(input.da_bnlh.clone());

        let (y_bnlrhp, final_state_bhpr) = <B as Mamba3BackendExt>::ssd_serial_recalculated(
            input.v_bnlrhp.into_primitive().tensor(),
            input.da_bnlh.into_primitive().tensor(),
            input.b_bnlrhn.into_primitive().tensor(),
            input.c_bnlrhn.into_primitive().tensor(),
            input.initial_state_bhpr.into_primitive().tensor(),
            da_cumsum_bhnl.into_primitive().tensor(),
        );
        let y_bnlrhp = Tensor::from_primitive(TensorPrimitive::Float(y_bnlrhp));
        let final_state_bhpr = Tensor::from_primitive(TensorPrimitive::Float(final_state_bhpr));
        (y_bnlrhp, final_state_bhpr)
    }
}

/// Extends the backend for the memory-efficient serial recalculated SSD.
///
/// The default implementation runs K2-K5 using standard tensor operations.
/// Backends that support a custom memory-efficient backward can override this.
pub trait Mamba3BackendExt: burn::tensor::backend::Backend {
    /// Memory-efficient MIMO serial SSD.
    ///
    /// # Arguments
    /// - `v_bnlrhp`:           `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
    /// - `da_bnlh`:            `[batch, nchunks, chunk_len, nheads]` — pre-combined Δ·A
    /// - `b_bnlrhn`:           `[batch, nchunks, chunk_len, R, nheads, state_rank]`
    /// - `c_bnlrhn`:           `[batch, nchunks, chunk_len, R, nheads, state_rank]`
    /// - `initial_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    /// - `da_cumsum_bhnl`:     `[batch, nheads, nchunks, chunk_len]` — pre-computed by K1
    ///
    /// # Returns
    /// - `y_bnlrhp`:        `[batch, nchunks, chunk_len, R, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    fn ssd_serial_recalculated(
        v_bnlrhp: FloatTensor<Self>,
        _da_bnlh: FloatTensor<Self>,
        b_bnlrhn: FloatTensor<Self>,
        c_bnlrhn: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        da_cumsum_bhnl: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Default impl: run K2-K5 using the pre-computed cumsum.

        let v: Tensor<Self, 6> = mk(v_bnlrhp);
        let b: Tensor<Self, 6> = mk(b_bnlrhn);
        let c: Tensor<Self, 6> = mk(c_bnlrhn);
        let da_cumsum: Tensor<Self, 4> = mk(da_cumsum_bhnl);
        let init_state: Tensor<Self, 4> = mk(initial_state_bhpr);

        // Recalculate da_chunk_end_bhn from the pre-computed cumsum (the "recalculated" part).
        let da_chunk_end_bhn: Tensor<Self, 3> =
            da_cumsum.clone().slice(s![.., .., .., -1]).squeeze_dim(3);

        let cb_bnhLL: Tensor<Self, 5> = serial::k2_ssd_bmm(c.clone(), b.clone());
        let intra_chunk_state_bnhpr: Tensor<Self, 5> =
            serial::k3_ssd_chunk_state(v.clone(), b, da_cumsum.clone());
        let (chunk_input_state_bnhpr, final_state_bhpr): (Tensor<Self, 5>, Tensor<Self, 4>) =
            serial::k4_ssd_state_passing(intra_chunk_state_bnhpr, da_chunk_end_bhn, init_state);
        let y_bnlrhp: Tensor<Self, 6> =
            serial::k5_ssd_chunk_scan(da_cumsum, v, c, cb_bnhLL, chunk_input_state_bnhpr);

        let y_bnlrhp = y_bnlrhp.into_primitive().tensor();
        let final_state_bhpr = final_state_bhpr.into_primitive().tensor();
        (y_bnlrhp, final_state_bhpr)
    }
}

/// Marker for autodiff-compatible backends that support the custom MIMO backward.
#[cfg(feature = "autodiff")]
pub trait Mamba3AutodiffBackendExt:
    Backend + Mamba3BackendExt + burn::tensor::backend::AutodiffBackend
{
}
/// Autodiff-wrapped backends inherit the inner backend's Mamba3BackendExt impl.
#[cfg(feature = "autodiff")]
impl<B: Backend + Mamba3BackendExt> Mamba3BackendExt for burn::backend::Autodiff<B> {}
/// Any autodiff-wrapped backend satisfies the marker.
#[cfg(feature = "autodiff")]
impl<B: Backend + Mamba3BackendExt> Mamba3AutodiffBackendExt for burn::backend::Autodiff<B> {}

// ---------------------------------------------------------------------------
// Backend impls — each backend uses the default (K2-K5) implementation.
// ---------------------------------------------------------------------------

#[cfg(feature = "backend-ndarray")]
impl<F, I> Mamba3BackendExt for burn::backend::NdArray<F, I> {}
#[cfg(feature = "backend-flex")]
impl Mamba3BackendExt for burn::backend::Flex {}
#[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
impl<F, I> Mamba3BackendExt for burn::backend::libtorch::LibTorch<F, I> {}
#[cfg(feature = "backend-remote")]
impl<F, I> Mamba3BackendExt for burn::backend::RemoteBackend<F, I> {}

// CubeCL backends
#[cfg(feature = "cubecl")]
mod cubecl {
    use burn_cubecl::{CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement};
    impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> super::Mamba3BackendExt
        for CubeBackend<R, F, I, BT>
    {
    }
}

// Fusion backends
#[cfg(feature = "fusion")]
mod fusion {
    use burn_fusion::{Fusion, FusionBackend};
    impl<B: FusionBackend + super::Mamba3BackendExt> super::Mamba3BackendExt for Fusion<B> {}
}

/// Conversion helper: `FloatTensor<B>` → `Tensor<B, D>`.
pub(crate) fn mk<B: Backend, const D: usize>(p: FloatTensor<B>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(p))
}
