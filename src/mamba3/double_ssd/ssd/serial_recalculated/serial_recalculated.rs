//! # Serial SSD with a custom, memory-efficient backward (Mamba-3 double-SSD)
//!
//! The `SerialRecalculated` path of the double-SSD pathway. The forward goes
//! through the [`Mamba3DoubleSsdBackendExt`] trait. So an `Autodiff` backend can
//! use a custom backward that recomputes per-chunk intermediates instead of
//! storing them (see [`super::backward`]). Plain backends use the default body
//! of the trait.
//!
//! Both the default body and the custom backward call the
//! [standard SSD scan](crate::standard_ssd), once for each term of the
//! trapezoid.

use crate::mamba3::double_ssd::prelude::*;
use crate::standard_ssd;
use burn_stack::utils::fprim::F;
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::Tensor;

impl Mamba3DoubleSsdInput {
    /// MIMO-first Serial SSD with recalculated backward.
    ///
    /// Sends the full K1-K5 computation to
    /// [`Mamba3DoubleSsdBackendExt::double_ssd_serial_recalculated`]. An
    /// autodiff backend adds the memory-efficient custom backward. A plain
    /// backend runs the standard K1-K5 serial computation.
    ///
    /// # Returns
    /// - `y_bntmhp`:         `[batch, nchunks, chunk_tokens, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    pub fn double_ssd_serial_recalculated(self) -> (Tensor<6>, Tensor<4>) {
        let input = self;
        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented for ssd_serial_recalculated"
        );

        let (keep_bn, origin_bhpr, resets) =
            crate::packing::Restarts::op_inputs(input.restarts, &input.v_bnlmhp);
        let (y_bntmhp, final_state_bhpr) =
            <Dispatch as Mamba3DoubleSsdBackendExt>::double_ssd_serial_recalculated(
                input.v_bnlmhp.into_dispatch(),
                input.da_bnlh.into_dispatch(),
                input.b_bnlmhr.into_dispatch(),
                input.c_bntmhr.into_dispatch(),
                input.initial_state_bhpr.into_dispatch(),
                input.read_stride,
                keep_bn.into_dispatch(),
                origin_bhpr.into_dispatch(),
                resets,
            );
        let y_bntmhp = Tensor::from_dispatch(y_bntmhp);
        let final_state_bhpr = Tensor::from_dispatch(final_state_bhpr);
        (y_bntmhp, final_state_bhpr)
    }
}

/// Extends the backend for the memory-efficient serial recalculated SSD.
///
/// The default implementation runs K1-K5 with primitive tensor operations.
/// The Autodiff wrapper overrides it: its backward recomputes the forward
/// intermediates instead of saving them.
#[backend_extension(
    // Every `#[backend_extension]` site of this crate has the same list.
    // Change them together. `build.rs` sets `cube_backend`.
    Cube: cfg(cube_backend),
    Flex:  cfg(feature = "backend-flex"),
    NdArray:  cfg(feature = "backend-ndarray"),
    LibTorch:  cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
    Autodiff:  cfg(feature = "autodiff"),
)]
pub trait Mamba3DoubleSsdBackendExt: Backend {
    /// Memory-efficient MIMO serial SSD.
    ///
    /// # Arguments
    /// - `v_bnlmhp`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
    /// - `da_bnlh`:            `[batch, nchunks, chunk_len, nheads]` — pre-combined Δ·A
    /// - `b_bnlmhr`:           `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
    /// - `c_bntmhr`:           `[batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]`
    /// - `initial_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    /// - `read_stride`:        `micro_steps` — write positions per read position, so
    ///   `chunk_tokens = chunk_len / read_stride`
    /// - `keep_bn`:            `[batch, nchunks]`, `0` at a chunk that starts
    ///   a new segment of a packed row. Read only when `resets` is `true`.
    /// - `origin_bhpr`:        `[batch, nheads, per_head_dim, state_rank]`, the
    ///   state that such a chunk starts from. Read only when `resets` is
    ///   `true`.
    ///
    /// # Returns
    /// - `y_bntmhp`:         `[batch, nchunks, chunk_tokens, mimo_rank, nheads, per_head_dim]`
    /// - `final_state_bhpr`: `[batch, nheads, per_head_dim, state_rank]`
    #[allow(clippy::too_many_arguments)]
    fn double_ssd_serial_recalculated(
        v_bnlmhp: FloatTensor<Self>,
        da_bnlh: FloatTensor<Self>,
        b_bnlmhr: FloatTensor<Self>,
        c_bntmhr: FloatTensor<Self>,
        initial_state_bhpr: FloatTensor<Self>,
        read_stride: usize,
        keep_bn: FloatTensor<Self>,
        origin_bhpr: FloatTensor<Self>,
        resets: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let restarts = resets.then(|| (F::<Self, 2>::new(keep_bn), F::<Self, 4>::new(origin_bhpr)));
        let (y_bntmhp, final_state_bhpr) = standard_ssd::forward::<Self>(
            F::new(v_bnlmhp),
            F::new(da_bnlh),
            F::new(b_bnlmhr),
            F::new(c_bntmhr),
            F::new(initial_state_bhpr),
            read_stride,
            restarts,
        );
        (y_bntmhp.inner(), final_state_bhpr.inner())
    }
}

burn_stack::decl_autodiff_backend_ext!(Mamba3DoubleSsdAutodiffBackendExt, Mamba3DoubleSsdBackendExt);

// ---------------------------------------------------------------------------
// Per-backend impls: each one uses the default (K1-K5) body of the trait. The
// custom autodiff backward is a separate impl in `super::backward`.
// ---------------------------------------------------------------------------
burn_stack::impl_backend_ext_for_burn_backends!(Mamba3DoubleSsdBackendExt);
