//! # Serial SSD with a custom, memory-efficient backward (Mamba-2)
//!
//! This is the `SerialRecalculated` path. The forward goes through the
//! [`Mamba2BackendExt`] trait. Thus an `Autodiff` backend can use a **custom
//! backward** that recomputes the per-chunk intermediates instead of storing
//! them (see [`super::backward`] / [`super::combined_backward`]). This costs a
//! little extra compute and saves ~⅓ of the training memory.
//!
//! The default body of the trait and the custom backward both call the
//! [standard SSD scan](crate::standard_ssd). This file holds what Mamba-2 adds
//! to the forward of that scan: the inputs `v = Δ·x` and `da = Δ·A`, and the
//! `D` skip.
//!
//! Every plain (non-autodiff) backend uses the default body of the trait. The
//! [`burn_stack::impl_backend_ext_for_burn_backends!`] macro makes these
//! per-backend impls.

use crate::mamba2::prelude::*;
use crate::packing::OpResets;
use crate::standard_ssd;
use burn_stack::utils::fprim::{F, san};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::Tensor;

impl Mamba2SsdInput {
    /// Forward pass for the Mamba-2 SSD module (recompute-backward path).
    ///
    /// Returns:
    /// - `y_bnlhp`.
    /// - `final_state_bhpr`.
    #[allow(non_snake_case)]
    pub fn ssd_serial_recalculated(self) -> (Tensor<5>, Tensor<4>) {
        let input = self;
        // This must be a backend-dependent method. On a plain backend, it runs
        // the forward of the standard SSD scan. On an autodiff backend, it
        // calls the custom backward.

        let [batch, nchunks, chunk_len, nheads, _per_head_dim] = input.x_bnlhp.dims();
        assert!(nchunks > 0, "sequence length must be at least 1");

        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented"
        );

        // ── Permutes ──────────────────────────────────────────────────────────────────
        // The reference computes Δ in Kernel 1. Here `Mamba2::forward` computes
        // it (step 4), before the chunk padding.
        let dt_discretized_bhnl = input.dt_bnlh.permute([0, 3, 1, 2]);
        assert_eq!(
            [batch, nheads, nchunks, chunk_len],
            dt_discretized_bhnl.dims()
        );

        // The resets go into the op as a keep factor and the origin state (see
        // `SsdResets::op_inputs`).
        let resets = crate::packing::SsdResets::op_inputs(input.resets, &input.x_bnlhp);

        // The custom op forms `da` itself (in forward and backward). It gets
        // `a_decay_h` directly, so `da` is not an autodiff-tracked intermediate
        // that crosses the op boundary.
        let (y_bnlhp, final_state_bhpr) = <Dispatch as Mamba2BackendExt>::ssd_serial_recalculated(
            input.x_bnlhp.into_dispatch(),
            dt_discretized_bhnl.into_dispatch(),
            input.b_bnlhr.into_dispatch(),
            input.c_bnlhr.into_dispatch(),
            input.d_h.into_dispatch(),
            input.initial_state_bhpr.into_dispatch(),
            input.a_decay_h.into_dispatch(),
            resets,
        );
        let y_bnlhp = Tensor::from_dispatch(y_bnlhp);
        let final_state_bhpr = Tensor::from_dispatch(final_state_bhpr);
        (y_bnlhp, final_state_bhpr)
    }
}

/// Backend extension for the Mamba-2 SSD with a recompute backward.
///
/// The default body runs the forward on backend primitives. The `Autodiff<B>`
/// impl (in [`super::backward`]) adds the recompute node.
#[backend_extension(
    // Every `#[backend_extension]` site of this crate has the same list.
    // Change them together. `build.rs` sets `cube_backend`.
    Cube: cfg(cube_backend),
    Flex:  cfg(feature = "backend-flex"),
    NdArray:  cfg(feature = "backend-ndarray"),
    LibTorch:  cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Autodiff:  cfg(feature = "autodiff"),
)]
pub trait Mamba2BackendExt: Backend {
    /// `resets` gives the chunks that start a new segment of a packed row,
    /// and the state that they start from ([`OpResets`]).
    ///
    /// Returns:
    /// - `y_bnlhp`.
    /// - `final_state_bhpr`.
    #[allow(clippy::too_many_arguments)]
    fn ssd_serial_recalculated(
        x_bnlhp: FloatTensor<Self>,
        dt_discretized_bhnl: FloatTensor<Self>,
        b_bnlhr: FloatTensor<Self>,
        c_bnlhr: FloatTensor<Self>,
        d_h: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        a_decay_h: FloatTensor<Self>,
        #[extension_type] resets: OpResets<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        // This body runs under a generic `B`, where the high-level `Tensor`
        // (pinned to `Dispatch`) is not available. So the math uses the
        // rank-tagged [`F`] primitive wrapper.
        let x_bnlhp = F::<Self, 5>::new(x_bnlhp);
        let dt_bnlh = F::<Self, 4>::new(dt_discretized_bhnl).permute([0, 2, 3, 1]);
        let a_decay_111h = F::<Self, 1>::new(a_decay_h).unsqueeze_dims::<4>(&[0, 1, 2]);
        let resets = resets.into_option();

        let (v_bnlhp, da_bnlh) = scan_inputs(x_bnlhp.clone(), dt_bnlh, a_decay_111h);
        let (y_bnl1hp, final_state_bhpr) = standard_ssd::forward::<Self>(
            v_bnlhp.unsqueeze_dim::<6>(3),
            da_bnlh,
            F::<Self, 5>::new(b_bnlhr).unsqueeze_dim::<6>(3),
            F::<Self, 5>::new(c_bnlhr).unsqueeze_dim::<6>(3),
            F::new(initial_state_bhpr),
            1,
            resets,
        );

        // ── SKIP: D[h] · x[l,p] ─────────────────────────────────────────────────
        let skip_bnlhp = x_bnlhp * F::<Self, 1>::new(d_h).unsqueeze_dims::<5>(&[0, 1, 2, 4]);
        let y_bnlhp = y_bnl1hp.squeeze_dim::<5>(3) + skip_bnlhp;
        san(&y_bnlhp);

        (y_bnlhp.inner(), final_state_bhpr.inner())
    }
}

/// The inputs of the standard SSD scan: `v = Δ·x` and `da = Δ·A`.
pub(crate) fn scan_inputs<B: Backend>(
    x_bnlhp: F<B, 5>,
    dt_bnlh: F<B, 4>,
    a_decay_111h: F<B, 4>,
) -> (F<B, 5>, F<B, 4>) {
    let v_bnlhp = x_bnlhp * dt_bnlh.clone().unsqueeze_dim::<5>(4);
    let da_bnlh = dt_bnlh * a_decay_111h;
    (v_bnlhp, da_bnlh)
}

// Per-backend impls: each one uses the default body of the trait. The custom
// autodiff backward is a separate impl in `super::backward`.
//
// TODO: find a way to keep the backend-* features out of the library.
burn_stack::impl_backend_ext_for_burn_backends!(Mamba2BackendExt);
