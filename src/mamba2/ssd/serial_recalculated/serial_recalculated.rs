//! # Serial SSD with a custom, memory-efficient backward (Mamba-2)
//!
//! This is the `SerialRecalculated` path.  The forward is the same five-kernel
//! serial scan as [`super::super::serial`], but it is routed through the
//! [`Mamba2BackendExt`] trait so that `Autodiff` backends can substitute a
//! **custom backward** that recomputes the per-chunk intermediates instead of
//! storing them (see [`super::backward`] / [`super::combined_backward`]),
//! trading a little extra compute for ~⅓ less training memory.
//!
//! Every plain (non-autodiff) backend uses the trait's default body, which
//! simply replays the [`super::super::serial`] kernels K1–K5.  The
//! [`crate::impl_ssd_backend_ext_for_burn_backends!`] /
//! [`crate::decl_ssd_autodiff_backend_ext!`] macros wire up the per-backend
//! impls and the autodiff marker trait.

use crate::mamba2::prelude::*;
use crate::mamba2::ssd::serial;
use crate::utils::primitive::mk;
use burn::tensor::{Tensor};
use burn::backend::{TensorPrimitive, tensor::FloatTensor};
use burn::backend::{Backend, backend_extension, Dispatch};

impl Mamba2SsdInput {
    /// Forward pass for the Mamba-2 SSD module (recompute-backward path).
    ///
    /// Returns:
    /// - `y_bnlhp`.
    /// - `final_state_bhpr`.
    #[allow(non_snake_case)]
    pub fn ssd_serial_recalculated(self) -> (Tensor<5>, Tensor<4>) {
        let input = self;
        // Must use a backend-dependent method.
        //
        // For inference, this will ultimately replicate Mamba2::ssd_serial;
        // For autodiff, this will call the custom implementation.

        let [batch, nchunks, chunk_len, nheads, _per_head_dim] = input.x_bnlhp.dims();
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

        // K1 is now computed inside the custom op (both forward and backward).
        // a_decay_h is passed directly; da_cumsum is no longer an autodiff-tracked
        // intermediate crossing the boundary.
        let (y_bnlhp, final_state_bhpr) = Dispatch::ssd_serial_recalculated(
            input.x_bnlhp.into_primitive().tensor(),
            dt_discretized_bhnl.into_primitive().tensor(),
            input.b_bnlhr.into_primitive().tensor(),
            input.c_bnlhr.into_primitive().tensor(),
            input.d_h.into_primitive().tensor(),
            input.initial_state_bhpr.into_primitive().tensor(),
            input.a_decay_h.into_primitive().tensor(),
        );
        let y_bnlhp = Tensor::from_primitive(TensorPrimitive::Float(y_bnlhp));
        let final_state_bhpr = Tensor::from_primitive(TensorPrimitive::Float(final_state_bhpr));
        (y_bnlhp, final_state_bhpr)
    }
}

/// Extends the backend and wraps it for `burn`.
pub trait Mamba2BackendExt: Backend {
    /// Returns:
    /// - `y_bnlhp`.
    /// - `final_state_bhpr`.
    fn ssd_serial_recalculated(
        x_bnlhp: FloatTensor<Self>,
        dt_discretized_bhnl: FloatTensor<Self>,
        b_bnlhr: FloatTensor<Self>,
        c_bnlhr: FloatTensor<Self>,
        d_h: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        a_decay_h: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // Default impl essentially replicates Mamba2SsdInput::ssd_serial.

        let x_bnlhp: Tensor<Self, 5> = mk(x_bnlhp);
        let dt_discretized_bhnl: Tensor<Self, 4> = mk(dt_discretized_bhnl);
        let b_bnlhr: Tensor<Self, 5> = mk(b_bnlhr);
        let c_bnlhr: Tensor<Self, 5> = mk(c_bnlhr);
        let d_h: Tensor<Self, 1> = mk(d_h);
        let initial_state_bhpr: Tensor<Self, 4> = mk(initial_state_bhpr);
        let a_decay_h: Tensor<Self, 1> = mk(a_decay_h);

        let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
        let [.., state_rank] = b_bnlhr.dims();
        assert!(nchunks > 0, "sequence length must be at least 1");

        // ── Kernel 1 ──────────────────────────────────────────────────────────
        let (da_cumsum_bhnl, da_chunk_end_bhn) =
            serial::k1_ssd_chunk_cumsum(dt_discretized_bhnl.clone(), a_decay_h);
        assert_eq!([batch, nheads, nchunks, chunk_len], da_cumsum_bhnl.dims());
        assert_eq!([batch, nheads, nchunks], da_chunk_end_bhn.dims());

        // ── Kernel 2 ──────────────────────────────────────────────────────────
        // IO: (..) -> (cb_bnhll [used in K5][!])
        let cb_bnhll: Tensor<Self, 5> = serial::k2_ssd_bmm(c_bnlhr.clone(), b_bnlhr.clone());
        assert_eq!(
            [batch, nchunks, nheads, chunk_len, chunk_len],
            cb_bnhll.dims()
        );

        // ── Kernel 3 ──────────────────────────────────────────────────────────
        // IO: (..) -> (intra_chunk_state_bnhpr [used in K4][!])
        let intra_chunk_state_bnhpr: Tensor<Self, 5> = serial::k3_ssd_chunk_state(
            x_bnlhp.clone(),
            b_bnlhr.clone(),
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
            c_bnlhr,
            cb_bnhll,
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

// Per-backend impls: each delegates to the trait's default body. The custom
// autodiff backward lives in `super::backward` as a separate impl.
//
// TODO: somehow avoid leaking backend-* features into the library.
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba2BackendExt);

crate::decl_ssd_autodiff_backend_ext!(Mamba2AutodiffBackendExt, Mamba2BackendExt);
