//! # Serial SSD with a custom, memory-efficient backward (Mamba-3 double-SSD)
//!
//! The `SerialRecalculated` path for the double-SSD pathway.  The forward is the
//! same serial scan as [`super::super::serial`], routed through the
//! [`Mamba3DoubleSsdBackendExt`] trait so that `Autodiff` backends substitute a
//! custom backward that recomputes per-chunk intermediates instead of storing
//! them (see [`super::backward`] / [`super::combined_backward`]).  Plain
//! backends use the trait's default body, which replays the serial kernels.

#![allow(non_snake_case)]

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::double_ssd::ssd;
use crate::utils::primitive::mk;
use burn::prelude::*;
use burn::tensor::{Tensor, TensorPrimitive, ops::FloatTensor};
use ssd::serial;

impl<B: Backend + Mamba3DoubleSsdBackendExt> Mamba3DoubleSsdInput<B> {
    /// MIMO-first Serial SSD with recalculated backward.
    ///
    /// Delegates the full K1-K5 computation to [`Mamba3DoubleSsdBackendExt::double_ssd_serial_recalculated`]
    /// which can provide a memory-efficient custom backward for supported backends.
    ///
    /// Falls back to the standard K1-K5 serial computation on unsupported backends.
    ///
    /// # Returns
    /// - `y_bnlmhp`:         `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn double_ssd_serial_recalculated(self) -> (Tensor<B, 6>, Tensor<B, 4>) {
        let input = self;
        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented for ssd_serial_recalculated"
        );

        let (y_bnlmhp, final_state_bhpr) =
            <B as Mamba3DoubleSsdBackendExt>::double_ssd_serial_recalculated(
                input.v_bnlmhp.into_primitive().tensor(),
                input.da_bnlh.into_primitive().tensor(),
                input.b_bnlmhr.into_primitive().tensor(),
                input.c_bnlmhr.into_primitive().tensor(),
                input.initial_state_bhpr.into_primitive().tensor(),
            );
        let y_bnlmhp = Tensor::from_primitive(TensorPrimitive::Float(y_bnlmhp));
        let final_state_bhpr = Tensor::from_primitive(TensorPrimitive::Float(final_state_bhpr));
        (y_bnlmhp, final_state_bhpr)
    }
}

/// Extends the backend for the memory-efficient serial recalculated SSD.
///
/// The default implementation runs K1-K5 using standard tensor operations.
/// Backends that support a custom memory-efficient backward (specifically the
/// Autodiff wrapper) override this to recompute forward intermediates during
/// the backward pass instead of saving them.
pub trait Mamba3DoubleSsdBackendExt: burn::tensor::backend::Backend {
    /// Memory-efficient MIMO serial SSD.
    ///
    /// # Arguments
    /// - `v_bnlmhp`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `da_bnlh`:            `[batch, nchunks, chunk_len, nheads]` — pre-combined Δ·A
    /// - `b_bnlmhr`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    /// - `c_bnlmhr`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    /// - `initial_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    ///
    /// # Returns
    /// - `y_bnlmhp`:         `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    fn double_ssd_serial_recalculated(
        v_bnlmhp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnlmhr: FloatTensor<Self>,
        c_bnlmhr: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Default impl: replicate Mamba3::ssd_serial (K1-K5).

        let v_bnlmhp: Tensor<Self, 6> = mk(v_bnlmhp);
        let da_bnlh: Tensor<Self, 4> = mk(da_bnlh);
        let b_bnlmhr: Tensor<Self, 6> = mk(b_bnlmhr);
        let c_bnlmhr: Tensor<Self, 6> = mk(c_bnlmhr);
        let initial_state_bhpr: Tensor<Self, 4> = mk(initial_state_bhpr);

        let (da_cumsum_bhnl, da_chunk_end_bhn) = serial::k1_ssd_chunk_cumsum(da_bnlh);
        let cb_bnhLMLM = serial::k2_ssd_bmm(c_bnlmhr.clone(), b_bnlmhr.clone());
        let intra_chunk_state_bnhpr =
            serial::k3_ssd_chunk_state(v_bnlmhp.clone(), b_bnlmhr, da_cumsum_bhnl.clone());
        let (chunk_input_state_bnhpr, final_state_bhpr) = serial::k4_ssd_state_passing(
            intra_chunk_state_bnhpr,
            da_chunk_end_bhn,
            initial_state_bhpr,
        );
        let y_bnlmhp = serial::k5_ssd_chunk_scan(
            da_cumsum_bhnl,
            v_bnlmhp,
            c_bnlmhr,
            cb_bnhLMLM,
            chunk_input_state_bnhpr,
        );

        let y_bnlmhp = y_bnlmhp.into_primitive().tensor();
        let final_state_bhpr = final_state_bhpr.into_primitive().tensor();
        (y_bnlmhp, final_state_bhpr)
    }
}

crate::decl_ssd_autodiff_backend_ext!(Mamba3DoubleSsdAutodiffBackendExt, Mamba3DoubleSsdBackendExt);

// ---------------------------------------------------------------------------
// Per-backend impls: each delegates to the trait's default (K1-K5) body. The
// custom autodiff backward lives in `super::backward` as a separate impl.
// ---------------------------------------------------------------------------
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3DoubleSsdBackendExt);
