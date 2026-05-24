#![allow(non_snake_case)]

use crate::mamba3::prelude::*;
use crate::mamba3::ssd::{serial, trap_serial};
use crate::utils::primitive::mk;
use burn::prelude::*;
use burn::tensor::{Tensor, TensorPrimitive, ops::FloatTensor};

impl<B: Backend + Mamba3TrapBackendExt> Mamba3TrapSsdInput<B> {
    /// MIMO-first merged-form Serial SSD with recalculated backward.
    ///
    /// Delegates the full K1–K5 (trapezoidal) computation to
    /// [`Mamba3TrapBackendExt::ssd_trap_serial_recalculated`], which can provide
    /// a memory-efficient custom backward for supported backends (the Autodiff
    /// wrapper) and falls back to the standard K1–K5 forward on others.
    ///
    /// # Returns
    /// - `y_bnlmhp`:         `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn ssd_trap_serial_recalculated(self) -> (Tensor<B, 6>, Tensor<B, 4>) {
        let input = self;
        input.sanity();
        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented for ssd_trap_serial_recalculated"
        );

        let (y_bnlmhp, final_state_bhpr) =
            <B as Mamba3TrapBackendExt>::ssd_trap_serial_recalculated(
                input.v_bnlmhp.into_primitive().tensor(),
                input.da_bnlh.into_primitive().tensor(),
                input.b_bnlmhr.into_primitive().tensor(),
                input.c_bnlmhr.into_primitive().tensor(),
                input.gamma_bnlh.into_primitive().tensor(),
                input.scale_bnlh.into_primitive().tensor(),
                input.initial_state_bhpr.into_primitive().tensor(),
            );
        let y_bnlmhp = Tensor::from_primitive(TensorPrimitive::Float(y_bnlmhp));
        let final_state_bhpr = Tensor::from_primitive(TensorPrimitive::Float(final_state_bhpr));
        (y_bnlmhp, final_state_bhpr)
    }
}

/// Extends the backend for the memory-efficient merged-form (trapezoidal)
/// serial SSD.
///
/// The default implementation runs K1–K5 using standard tensor operations,
/// reusing K1/K2/K4 from [`crate::mamba3::ssd::serial`] (mode-agnostic) and the
/// merged-form K5 from [`crate::mamba3::ssd::trap_serial`]. Backends that
/// support a custom memory-efficient backward (the Autodiff wrapper) override
/// this to recompute forward intermediates during backward instead of saving
/// them.
pub trait Mamba3TrapBackendExt: burn::tensor::backend::Backend {
    /// Memory-efficient MIMO merged-form serial SSD.
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
    fn ssd_trap_serial_recalculated(
        v_bnlmhp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnlmhr: FloatTensor<Self>,
        c_bnlmhr: FloatTensor<Self>,
        gamma_bnlh: FloatTensor<Self>,
        scale_bnlh: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Default impl: replicate the merged-form K1–K5 (see `ssd_trap_serial`).
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
        // K5 — merged-form chunk scan.
        let y_bnlmhp = trap_serial::k5_trap_ssd_chunk_scan(
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

crate::decl_ssd_autodiff_backend_ext!(Mamba3TrapAutodiffBackendExt, Mamba3TrapBackendExt);

// ---------------------------------------------------------------------------
// Per-backend impls: each delegates to the trait's default (K1–K5) body. The
// custom autodiff backward lives in `super::backward` as a separate impl.
// ---------------------------------------------------------------------------
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3TrapBackendExt);
