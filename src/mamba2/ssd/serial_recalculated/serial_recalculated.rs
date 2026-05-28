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
use crate::utils::fprim::{F, Mask, san};
use burn::tensor::Tensor;
use burn::tensor::s;
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, backend_extension, Dispatch};
use burn::backend::*;

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
        let (y_bnlhp, final_state_bhpr) = <Dispatch as Mamba2BackendExt>::ssd_serial_recalculated(
            input.x_bnlhp.into_primitive(),
            dt_discretized_bhnl.into_primitive(),
            input.b_bnlhr.into_primitive(),
            input.c_bnlhr.into_primitive(),
            input.d_h.into_primitive(),
            input.initial_state_bhpr.into_primitive(),
            input.a_decay_h.into_primitive(),
        );
        let y_bnlhp = Tensor::from_primitive(y_bnlhp);
        let final_state_bhpr = Tensor::from_primitive(final_state_bhpr);
        (y_bnlhp, final_state_bhpr)
    }
}

/// Extends the backend and wraps it for `burn`.
#[backend_extension(
    Cpu:  cfg(feature = "backend-cpu"),
    Cuda: cfg(feature = "backend-cuda"),
    Rocm:  cfg(feature = "backend-rocm"),
    Metal:  cfg(feature = "backend-metal"),
    Vulkan:  cfg(feature = "backend-vulkan"),
    Wgpu:  cfg(feature = "backend-wgpu"),
    WebGpu:  cfg(feature = "backend-webgpu"),
    Flex:  cfg(feature = "backend-flex"),
    NdArray:  cfg(feature = "backend-ndarray"),
    LibTorch:  cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu")),
)]
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
        // Default impl essentially replicates Mamba2SsdInput::ssd_serial, but on
        // backend primitives: this body runs under a generic `B`, where the
        // high-level `Tensor` (pinned to `Dispatch`) is unavailable, so the math
        // goes through the rank-tagged [`F`] primitive wrapper and the local
        // primitive K-kernels below.
        let x_bnlhp = F::<Self, 5>::new(x_bnlhp);
        let dt_discretized_bhnl = F::<Self, 4>::new(dt_discretized_bhnl);
        let b_bnlhr = F::<Self, 5>::new(b_bnlhr);
        let c_bnlhr = F::<Self, 5>::new(c_bnlhr);
        let d_h = F::<Self, 1>::new(d_h);
        let initial_state_bhpr = F::<Self, 4>::new(initial_state_bhpr);
        let a_decay_h = F::<Self, 1>::new(a_decay_h);

        let nchunks = x_bnlhp.dims()[1];
        assert!(nchunks > 0, "sequence length must be at least 1");

        // ── Kernel 1 ──────────────────────────────────────────────────────────
        let (da_cumsum_bhnl, da_chunk_end_bhn) =
            k1_ssd_chunk_cumsum::<Self>(dt_discretized_bhnl.clone(), a_decay_h);
        san(&da_cumsum_bhnl);

        // ── Kernel 2 ──────────────────────────────────────────────────────────
        let cb_bnhll = k2_ssd_bmm::<Self>(c_bnlhr.clone(), b_bnlhr.clone());
        san(&cb_bnhll);

        // ── Kernel 3 ──────────────────────────────────────────────────────────
        let intra_chunk_state_bnhpr = k3_ssd_chunk_state::<Self>(
            x_bnlhp.clone(),
            b_bnlhr,
            da_cumsum_bhnl.clone(),
            dt_discretized_bhnl.clone(),
        );
        san(&intra_chunk_state_bnhpr);

        // ── Kernel 4 ──────────────────────────────────────────────────────────
        let (chunk_input_state_bnhpr, final_state_bhpr) = k4_ssd_state_passing::<Self>(
            intra_chunk_state_bnhpr,
            da_chunk_end_bhn,
            initial_state_bhpr,
        );
        san(&chunk_input_state_bnhpr);
        san(&final_state_bhpr);

        // ── Kernel 5 ──────────────────────────────────────────────────────────
        let y_bnlhp = k5_ssd_chunk_scan::<Self>(
            da_cumsum_bhnl,
            dt_discretized_bhnl,
            x_bnlhp,
            c_bnlhr,
            cb_bnhll,
            chunk_input_state_bnhpr,
            d_h,
        );
        san(&y_bnlhp);

        (y_bnlhp.inner(), final_state_bhpr.inner())
    }
}

// ─── Primitive forward kernels (K1–K5) ───────────────────────────────────────
// Primitive ports of the high-level [`crate::mamba2::ssd::serial`] kernels,
// expressed on `B`'s primitives via [`F`] so the trait default body can run
// under a generic backend. K1/K2/K4 are reused by the recompute backward in
// [`super::combined_backward`]; K5 is forward-only (the backward computes K5's
// gradient analytically rather than recomputing it).

/// Primitive port of [`crate::mamba2::ssd::serial::k1_ssd_chunk_cumsum`].
///
/// Returns the intra-chunk cumsum `da_cumsum_bhnl` and the per-chunk last value
/// `da_chunk_end_bhn`.
pub(crate) fn k1_ssd_chunk_cumsum<B: Backend>(
    dt_discretized_bhnl: F<B, 4>,
    a_decay_h: F<B, 1>,
) -> (F<B, 4>, F<B, 3>) {
    let [batch, nheads, nchunks, chunk_len] = dt_discretized_bhnl.dims();
    let da_cumsum_bhnl: F<B, 4> = {
        let a_decay_bhnl = a_decay_h
            .unsqueeze_dims::<4>(&[0, 2, 3])
            .expand([batch, nheads, nchunks, chunk_len]);
        (dt_discretized_bhnl * a_decay_bhnl).cumsum(3)
    };
    let da_chunk_end_bhn = da_cumsum_bhnl
        .clone()
        .slice(s![.., .., .., -1])
        .squeeze_dim::<3>(3);
    (da_cumsum_bhnl, da_chunk_end_bhn)
}

/// Primitive port of [`crate::mamba2::ssd::serial::k2_ssd_bmm`].
///
/// Returns the intra-chunk `C·Bᵀ` block matrix `cb_bnhll`.
pub(crate) fn k2_ssd_bmm<B: Backend>(c_bnlhr: F<B, 5>, b_bnlhr: F<B, 5>) -> F<B, 5> {
    let c_bnhlr = c_bnlhr.permute([0, 1, 3, 2, 4]);
    let b_bnhrl = b_bnlhr.permute([0, 1, 3, 4, 2]);
    c_bnhlr.matmul(b_bnhrl)
}

/// Primitive port of [`crate::mamba2::ssd::serial::k3_ssd_chunk_state`] (lean:
/// returns only the chunk-end state).
///
/// Returns `intra_chunk_state_bnhpr` — each chunk's contribution to its end
/// state assuming a zero state at the chunk's start.
pub(crate) fn k3_ssd_chunk_state<B: Backend>(
    x_bnlhp: F<B, 5>,
    b_bnlhr: F<B, 5>,
    da_cumsum_bhnl: F<B, 4>,
    dt_discretized_bhnl: F<B, 4>,
) -> F<B, 5> {
    let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
    let [.., state_rank] = b_bnlhr.dims();

    let x_bnhpl = x_bnlhp.permute([0, 1, 3, 4, 2]);
    let b_bnhlr = b_bnlhr.permute([0, 1, 3, 2, 4]);

    // K3 scaling factor: dt · exp(cumA_last − cumA)
    let da_cumsum_last_bhn1 = da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
    let forward_decay_to_chunk_end_bhnl =
        (da_cumsum_last_bhn1.expand([batch, nheads, nchunks, chunk_len]) - da_cumsum_bhnl).exp();
    let b_bar_scale_bhnl = forward_decay_to_chunk_end_bhnl * dt_discretized_bhnl;

    let b_bar_scale_bnhlr = b_bar_scale_bhnl
        .permute([0, 2, 1, 3]) // b_bar_scale_bnhl
        .unsqueeze_dim::<5>(4) // b_bar_scale_bnhl1
        .expand([batch, nchunks, nheads, chunk_len, state_rank]);
    let b_scaled_bnhlr = b_bnhlr * b_bar_scale_bnhlr;

    let intra_chunk_state_bnhpr = x_bnhpl.matmul(b_scaled_bnhlr);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        intra_chunk_state_bnhpr.dims()
    );
    intra_chunk_state_bnhpr
}

/// Primitive port of [`crate::mamba2::ssd::serial::k4_ssd_state_passing`].
///
/// Returns the per-chunk input-state stream `chunk_input_state_bnhpr` and the
/// `final_state_bhpr`.
pub(crate) fn k4_ssd_state_passing<B: Backend>(
    intra_chunk_state_bnhpr: F<B, 5>,
    da_chunk_end_bhn: F<B, 3>,
    initial_state_bhpr: F<B, 4>,
) -> (F<B, 5>, F<B, 4>) {
    let [batch, nchunks, nheads, per_head_dim, state_rank] = intra_chunk_state_bnhpr.dims();

    let mut running_state_bhpr = initial_state_bhpr;
    let mut chunk_input_state_vec_bhpr = Vec::with_capacity(nchunks + 1);
    chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());

    for i_chunk in 0..nchunks {
        let intra_state_bhpr = intra_chunk_state_bnhpr
            .clone()
            .slice(s![.., i_chunk, .., .., ..])
            .squeeze_dim::<4>(1);
        let decay_bhpr = da_chunk_end_bhn
            .clone()
            .slice(s![.., .., i_chunk])
            .exp()
            .unsqueeze_dim::<4>(3)
            .expand([batch, nheads, per_head_dim, state_rank]);
        running_state_bhpr = (decay_bhpr * running_state_bhpr) + intra_state_bhpr;
        chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());
    }

    let final_state_bhpr = chunk_input_state_vec_bhpr.pop().unwrap();
    let chunk_input_state_bnhpr = F::stack(chunk_input_state_vec_bhpr, 1);
    (chunk_input_state_bnhpr, final_state_bhpr)
}

/// Primitive port of [`crate::mamba2::ssd::serial::k5_ssd_chunk_scan`].
///
/// Combines the intra-chunk (ORANGE, attention-like) and inter-chunk (BLUE,
/// state-carried) contributions plus the `D` skip term into the output
/// `y_bnlhp`. Forward-only.
#[allow(clippy::too_many_arguments)]
fn k5_ssd_chunk_scan<B: Backend>(
    da_cumsum_bhnl: F<B, 4>,
    dt_discretized_bhnl: F<B, 4>,
    x_bnlhp: F<B, 5>,
    c_bnlhr: F<B, 5>,
    cb_bnhll: F<B, 5>,
    chunk_input_state_bnhpr: F<B, 5>,
    d_h: F<B, 1>,
) -> F<B, 5> {
    let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();
    let device = x_bnlhp.device();

    let da_cumsum_bnhl = da_cumsum_bhnl.permute([0, 2, 1, 3]);
    let dt_bnhl = dt_discretized_bhnl.permute([0, 2, 1, 3]);
    let x_bnhlp = x_bnlhp.clone().permute([0, 1, 3, 2, 4]);
    let c_bnhlr = c_bnlhr.permute([0, 1, 3, 2, 4]);

    // ── BLUE: exp(dA[l]) · C[l,:] @ state_in^T ─────────────────────────────
    let exp_da_cumsum_bnhlp = da_cumsum_bnhl
        .clone()
        .exp()
        .unsqueeze_dim::<5>(4) // exp_da_cumsum_bnhl1
        .expand([batch, nchunks, nheads, chunk_len, per_head_dim]);
    let chunk_input_state_bnhrp = chunk_input_state_bnhpr.permute([0, 1, 2, 4, 3]);
    let blue_scaled_bnhlp = c_bnhlr.matmul(chunk_input_state_bnhrp) * exp_da_cumsum_bnhlp;
    san(&blue_scaled_bnhlp);

    // ── ORANGE: causal CB_weighted @ X ──────────────────────────────────────
    let da_cumsum_target_bnhll = da_cumsum_bnhl
        .clone()
        .unsqueeze_dim::<5>(4) // da_cumsum_bnhl1
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    let da_cumsum_source_bnhll = da_cumsum_bnhl
        .unsqueeze_dim::<5>(3) // da_cumsum_bnh1l
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    let da_cumsum_diff_bnhll = da_cumsum_target_bnhll - da_cumsum_source_bnhll;

    // Causal mask + exp stabiliser: strictly-above-diagonal set to −∞.
    let causal_mask_bnhll: Mask<B> = Mask::tril_mask(chunk_len, chunk_len, 0, &device)
        .reshape([1, 1, 1, chunk_len, chunk_len])
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    let da_cumsum_diff_exp_bnhll = da_cumsum_diff_bnhll
        .mask_fill(causal_mask_bnhll, f32::NEG_INFINITY)
        .exp();
    san(&da_cumsum_diff_exp_bnhll);

    let dt_source_bnhll = dt_bnhl
        .unsqueeze_dim::<5>(3) // dt_bnh1l
        .expand([batch, nchunks, nheads, chunk_len, chunk_len]);
    let orange_lhs_bnhll = cb_bnhll * da_cumsum_diff_exp_bnhll * dt_source_bnhll;
    let orange_bnhlp = orange_lhs_bnhll.matmul(x_bnhlp);
    san(&orange_bnhlp);

    // ── SKIP: D[h] · x[l,p] ─────────────────────────────────────────────────
    let skip_bnlhp = d_h
        .unsqueeze_dims::<5>(&[0, 1, 2, 4]) // d_111h1
        .expand([batch, nchunks, chunk_len, nheads, per_head_dim])
        * x_bnlhp;

    let y_partial_bnhlp = blue_scaled_bnhlp + orange_bnhlp;
    let y_partial_bnlhp = y_partial_bnhlp.permute([0, 1, 3, 2, 4]);
    let y_bnlhp = y_partial_bnlhp + skip_bnlhp;
    assert_eq!(
        [batch, nchunks, chunk_len, nheads, per_head_dim],
        y_bnlhp.dims()
    );
    y_bnlhp
}

// Per-backend impls: each delegates to the trait's default body. The custom
// autodiff backward lives in `super::backward` as a separate impl.
//
// TODO: somehow avoid leaking backend-* features into the library.
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba2BackendExt);

crate::decl_ssd_autodiff_backend_ext!(Mamba2AutodiffBackendExt, Mamba2BackendExt);
