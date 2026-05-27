//! # Serial SSD with a custom, memory-efficient backward (Mamba-3 single-SSD)
//!
//! The `SerialRecalculated` path for the single-SSD pathway.  The forward is the
//! same serial scan as [`super::super::serial`], routed through the
//! [`Mamba3SingleSsdBackendExt`] trait so `Autodiff` backends substitute a
//! custom backward that recomputes per-chunk intermediates rather than storing
//! them (see [`super::backward`] / [`super::combined_backward`]).  Unlike the
//! double-SSD form, the kernels here apply the trapezoid `gamma`/`scale` and the
//! boundary-β seed internally, so the backward also returns `d_gamma`/`d_scale`.

#![allow(non_snake_case)]

use crate::mamba3::single_ssd::prelude::*;
use crate::mamba3::single_ssd::ssd;
use crate::utils::primitive::mk;
use burn::prelude::*;
use burn::tensor::{Tensor};
use burn::backend::{TensorPrimitive, tensor::FloatTensor};
use ssd::serial;
use burn::backend::Backend;

impl Mamba3SingleSsdInput {
    /// MIMO-first single-ssd form Serial SSD with recalculated backward.
    ///
    /// Delegates the full K1–K5 (single-ssd) computation to
    /// [`Mamba3SingleSsdBackendExt::single_ssd_serial_recalculated`], which can provide
    /// a memory-efficient custom backward for supported backends (the Autodiff
    /// wrapper) and falls back to the standard K1–K5 forward on others.
    ///
    /// # Returns
    /// - `y_bnlmhp`:         `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn single_ssd_serial_recalculated(self) -> (Tensor<6>, Tensor<4>) {
        let input = self;
        input.sanity();
        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented for single_ssd_serial_recalculated"
        );

        let (y_bnlmhp, final_state_bhpr) = todo!();
            // <B as Mamba3SingleSsdBackendExt>::single_ssd_serial_recalculated(
            //     input.v_bnlmhp.into_primitive().tensor(),
            //     input.da_bnlh.into_primitive().tensor(),
            //     input.b_bnlmhr.into_primitive().tensor(),
            //     input.c_bnlmhr.into_primitive().tensor(),
            //     input.gamma_bnlh.into_primitive().tensor(),
            //     input.scale_bnlh.into_primitive().tensor(),
            //     input.initial_state_bhpr.into_primitive().tensor(),
            // );
        let y_bnlmhp = Tensor::from_primitive(TensorPrimitive::Float(y_bnlmhp));
        let final_state_bhpr = Tensor::from_primitive(TensorPrimitive::Float(final_state_bhpr));
        (y_bnlmhp, final_state_bhpr)
    }
}

/// Extends the backend for the memory-efficient single-ssd form serial SSD.
///
/// The default implementation runs K1–K5 using standard tensor operations,
/// reusing K1/K2/K4 from [`crate::mamba3::double_ssd::ssd::serial`] (mode-agnostic) and the
/// single-ssd form K5 from [`crate::mamba3::single_ssd::ssd::serial`]. Backends that
/// support a custom memory-efficient backward (the Autodiff wrapper) override
/// this to recompute forward intermediates during backward instead of saving
/// them.
pub trait Mamba3SingleSsdBackendExt: Backend {
    /// Memory-efficient MIMO single-ssd form serial SSD.
    ///
    /// # Arguments
    /// - `v_bnlmhp`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `da_bnlh`:            `[batch, nchunks, chunk_len, nheads]` — pre-combined Δ·A
    /// - `b_bnlmhr`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    /// - `c_bnlmhr`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    /// - `gamma_bnlh`:         `[batch, nchunks, chunk_len, nheads]` — `γₜ = λₜ Δₜ`
    /// - `scale_bnlh`:         `[batch, nchunks, chunk_len, nheads]` — `scaleₜ = γₜ + (1−λₜ₊₁)Δₜ₊₁`
    /// - `initial_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    ///
    /// # Returns
    /// - `y_bnlmhp`:         `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    fn single_ssd_serial_recalculated(
        v_bnlmhp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnlmhr: FloatTensor<Self>,
        c_bnlmhr: FloatTensor<Self>,
        gamma_bnlh: FloatTensor<Self>,
        scale_bnlh: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Default impl: replicate the single-ssd form K1–K5 (see `single_ssd_serial`).
        let v_bnlmhp: Tensor<Self, 6> = mk(v_bnlmhp);
        let da_bnlh: Tensor<Self, 4> = mk(da_bnlh);
        let b_bnlmhr: Tensor<Self, 6> = mk(b_bnlmhr);
        let c_bnlmhr: Tensor<Self, 6> = mk(c_bnlmhr);
        let gamma_bnlh: Tensor<Self, 4> = mk(gamma_bnlh);
        let scale_bnlh: Tensor<Self, 4> = mk(scale_bnlh);
        let initial_state_bhpr: Tensor<Self, 4> = mk(initial_state_bhpr);

        // K1 — chunk cumulative decay.
        let (da_cumsum_bhnl, da_chunk_end_bhn) = serial::k1_ssd_chunk_cumsum(da_bnlh);
        // K2 — CB matrix on unscaled B/C.
        let cb_bnhLMLM = serial::k2_ssd_bmm(c_bnlmhr.clone(), b_bnlmhr.clone());
        // K3 — chunk state on K_scaled = scaleₜ · B.
        let scale_bnlh11 = scale_bnlh.clone().unsqueeze_dims::<6>(&[3, 5]);
        let k_scaled_bnlmhr = b_bnlmhr.clone() * scale_bnlh11;
        let intra_chunk_state_bnhpr =
            serial::k3_ssd_chunk_state(v_bnlmhp.clone(), k_scaled_bnlmhr, da_cumsum_bhnl.clone());
        // K4 — sequential state passing.
        let (chunk_input_state_bnhpr, final_state_bhpr) = serial::k4_ssd_state_passing(
            intra_chunk_state_bnhpr,
            da_chunk_end_bhn,
            initial_state_bhpr,
        );
        // K5 — single-ssd form chunk scan.
        let y_bnlmhp = serial::k5_single_ssd_chunk_scan(
            da_cumsum_bhnl,
            v_bnlmhp,
            c_bnlmhr,
            b_bnlmhr,
            cb_bnhLMLM,
            gamma_bnlh,
            scale_bnlh,
            chunk_input_state_bnhpr,
        );

        let y_bnlmhp = y_bnlmhp.into_primitive().tensor();
        let final_state_bhpr = final_state_bhpr.into_primitive().tensor();
        (y_bnlmhp, final_state_bhpr)
    }
}

crate::decl_ssd_autodiff_backend_ext!(Mamba3SingleSsdAutodiffBackendExt, Mamba3SingleSsdBackendExt);

// ---------------------------------------------------------------------------
// Per-backend impls: each delegates to the trait's default (K1–K5) body. The
// custom autodiff backward lives in `super::backward` as a separate impl.
// ---------------------------------------------------------------------------
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3SingleSsdBackendExt);
