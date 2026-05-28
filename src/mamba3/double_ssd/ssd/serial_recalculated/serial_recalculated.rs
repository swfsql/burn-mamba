//! # Serial SSD with a custom, memory-efficient backward (Mamba-3 double-SSD)
//!
//! The `SerialRecalculated` path for the double-SSD pathway.  The forward is the
//! same serial scan as [`super::super::serial`], routed through the
//! [`Mamba3DoubleSsdBackendExt`] trait so that `Autodiff` backends substitute a
//! custom backward that recomputes per-chunk intermediates instead of storing
//! them (see [`super::backward`] / [`super::combined_backward`]).  Plain
//! backends use the trait's default body, which replays the serial kernels.
//!
//! The default body runs under a generic backend `B`, where the high-level
//! `Tensor` (pinned to `Dispatch`) is unavailable, so the K1–K5 math goes
//! through the rank-tagged [`F`] primitive wrapper.  K1/K2/K4 are reused by the
//! recompute backward in [`super::combined_backward`]; K5 is forward-only.

#![allow(non_snake_case)]

use crate::mamba3::double_ssd::prelude::*;
use crate::utils::fprim::{F, san};
use burn::backend::tensor::FloatTensor;
use burn::backend::*;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::Tensor;
use burn::tensor::s;

impl Mamba3DoubleSsdInput {
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
    pub fn double_ssd_serial_recalculated(self) -> (Tensor<6>, Tensor<4>) {
        let input = self;
        assert!(
            input.init_state_hpr.is_none(),
            "init_state_hpr not yet implemented for ssd_serial_recalculated"
        );

        let (y_bnlmhp, final_state_bhpr) =
            <Dispatch as Mamba3DoubleSsdBackendExt>::double_ssd_serial_recalculated(
                input.v_bnlmhp.into_primitive(),
                input.da_bnlh.into_primitive(),
                input.b_bnlmhr.into_primitive(),
                input.c_bnlmhr.into_primitive(),
                input.initial_state_bhpr.into_primitive(),
            );
        let y_bnlmhp = Tensor::from_primitive(y_bnlmhp);
        let final_state_bhpr = Tensor::from_primitive(final_state_bhpr);
        (y_bnlmhp, final_state_bhpr)
    }
}

/// Extends the backend for the memory-efficient serial recalculated SSD.
///
/// The default implementation runs K1-K5 using primitive tensor operations.
/// Backends that support a custom memory-efficient backward (specifically the
/// Autodiff wrapper) override this to recompute forward intermediates during
/// the backward pass instead of saving them.
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
    Autodiff:  cfg(feature = "autodiff"),
)]
pub trait Mamba3DoubleSsdBackendExt: Backend {
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
        // Default impl: replicate Mamba3::double_ssd_serial (K1-K5) on primitives.
        let v_bnlmhp = F::<Self, 6>::new(v_bnlmhp);
        let da_bnlh = F::<Self, 4>::new(da_bnlh);
        let b_bnlmhr = F::<Self, 6>::new(b_bnlmhr);
        let c_bnlmhr = F::<Self, 6>::new(c_bnlmhr);
        let initial_state_bhpr = F::<Self, 4>::new(initial_state_bhpr);

        let nchunks = v_bnlmhp.dims()[1];
        assert!(nchunks > 0, "sequence length must be at least 1");

        let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum::<Self>(da_bnlh);
        san(&da_cumsum_bhnl);

        let cb_bnhLMLM = k2_ssd_bmm::<Self>(c_bnlmhr.clone(), b_bnlmhr.clone());
        san(&cb_bnhLMLM);

        let intra_chunk_state_bnhpr =
            k3_ssd_chunk_state::<Self>(v_bnlmhp.clone(), b_bnlmhr, da_cumsum_bhnl.clone());
        san(&intra_chunk_state_bnhpr);

        let (chunk_input_state_bnhpr, final_state_bhpr) = k4_ssd_state_passing::<Self>(
            intra_chunk_state_bnhpr,
            da_chunk_end_bhn,
            initial_state_bhpr,
        );
        san(&chunk_input_state_bnhpr);
        san(&final_state_bhpr);

        let y_bnlmhp = k5_ssd_chunk_scan::<Self>(
            da_cumsum_bhnl,
            v_bnlmhp,
            c_bnlmhr,
            cb_bnhLMLM,
            chunk_input_state_bnhpr,
        );
        san(&y_bnlmhp);

        (y_bnlmhp.inner(), final_state_bhpr.inner())
    }
}

crate::decl_ssd_autodiff_backend_ext!(Mamba3DoubleSsdAutodiffBackendExt, Mamba3DoubleSsdBackendExt);

// ---------------------------------------------------------------------------
// Per-backend impls: each delegates to the trait's default (K1-K5) body. The
// custom autodiff backward lives in `super::backward` as a separate impl.
// ---------------------------------------------------------------------------
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3DoubleSsdBackendExt);

// ─── Primitive forward kernels (K1–K5) ───────────────────────────────────────
// Primitive ports of the high-level [`super::super::serial`] kernels, expressed
// on `B`'s primitives via [`F`] so the trait default body can run under a
// generic backend. K1/K2/K4 are reused by the recompute backward in
// [`super::combined_backward`]; K5 is forward-only (the backward computes K5's
// gradient analytically rather than recomputing it).

/// Primitive port of [`super::super::serial::k1_ssd_chunk_cumsum`].
///
/// Returns the intra-chunk cumsum `da_cumsum_bhnl` and the per-chunk last value
/// `da_chunk_end_bhn`.
pub(crate) fn k1_ssd_chunk_cumsum<B: Backend>(da_bnlh: F<B, 4>) -> (F<B, 4>, F<B, 3>) {
    let da_bhnl = da_bnlh.permute([0, 3, 1, 2]);
    let da_cumsum_bhnl = da_bhnl.cumsum(3);
    let da_chunk_end_bhn = da_cumsum_bhnl
        .clone()
        .slice(s![.., .., .., -1])
        .squeeze_dim::<3>(3);
    (da_cumsum_bhnl, da_chunk_end_bhn)
}

/// Primitive port of [`super::super::serial::k2_ssd_bmm`] (fused `L·M`).
///
/// Returns the intra-chunk `C·Bᵀ` block matrix `cb_bnhLMLM`.
pub(crate) fn k2_ssd_bmm<B: Backend>(c_bnlmhr: F<B, 6>, b_bnlmhr: F<B, 6>) -> F<B, 5> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, state_rank] = c_bnlmhr.dims();
    let c_bnLMhr = c_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);
    let c_bnhLMr = c_bnLMhr.permute([0, 1, 3, 2, 4]);
    let b_bnhrLM = b_bnLMhr.permute([0, 1, 3, 4, 2]);
    c_bnhLMr.matmul(b_bnhrLM)
}

/// Primitive port of [`super::super::serial::k3_ssd_chunk_state`] (lean:
/// returns only the chunk-end state).
///
/// Returns `intra_chunk_state_bnhpr` — each chunk's contribution to its end
/// state assuming a zero state at the chunk's start. `v_bnlmhp` is pre-scaled.
pub(crate) fn k3_ssd_chunk_state<B: Backend>(
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    da_cumsum_bhnl: F<B, 4>,
) -> F<B, 5> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = b_bnlmhr.dims();

    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let b_bnLMhr = b_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

    let da_cumsum_last_bhn1 = da_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4) // da_cumsum_bhnl1
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // da_cumsum_bhnlm
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // da_cumsum_bhnLM
    let decay_bhnLM = (da_cumsum_last_bhn1 - da_cumsum_bhnLM).exp();

    let decay_bnLMh1 = decay_bhnLM.permute([0, 2, 3, 1]).unsqueeze_dim::<5>(4); // decay_bnLMh1
    let decayed_v_bnLMhp = decay_bnLMh1 * v_bnLMhp;

    let decayed_v_bnhpLM = decayed_v_bnLMhp.permute([0, 1, 3, 4, 2]);
    let b_bnhLMr = b_bnLMhr.permute([0, 1, 3, 2, 4]);
    let intra_chunk_state_bnhpr = decayed_v_bnhpLM.matmul(b_bnhLMr);
    assert_eq!(
        [batch, nchunks, nheads, per_head_dim, state_rank],
        intra_chunk_state_bnhpr.dims()
    );
    intra_chunk_state_bnhpr
}

/// Primitive port of [`super::super::serial::k4_ssd_state_passing`].
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
            .unsqueeze_dim::<4>(3)
            .exp()
            .expand([batch, nheads, per_head_dim, state_rank]);
        running_state_bhpr = decay_bhpr * running_state_bhpr + intra_state_bhpr;
        chunk_input_state_vec_bhpr.push(running_state_bhpr.clone());
    }

    let final_state_bhpr = chunk_input_state_vec_bhpr.pop().unwrap();
    let chunk_input_state_bnhpr = F::stack(chunk_input_state_vec_bhpr, 1);
    (chunk_input_state_bnhpr, final_state_bhpr)
}

/// Primitive port of [`super::super::serial::k5_ssd_chunk_scan`].
///
/// Combines the intra-chunk (ORANGE, MIMO causal) and inter-chunk (BLUE,
/// state-carried) contributions into the output `y_bnlmhp`. No `D` skip is
/// applied — the caller handles it. Forward-only.
fn k5_ssd_chunk_scan<B: Backend>(
    da_cumsum_bhnl: F<B, 4>,
    v_bnlmhp: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    cb_bnhLMLM: F<B, 5>,
    chunk_input_state_bnhpr: F<B, 5>,
) -> F<B, 6> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = c_bnlmhr.dims();
    let device = v_bnlmhp.device();
    let dtype = v_bnlmhp.dtype();

    let v_bnLMhp = v_bnlmhp.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, per_head_dim]);
    let c_bnLMhr = c_bnlmhr.reshape([batch, nchunks, chunk_len * mimo_rank, nheads, state_rank]);

    // Expand base da_cumsum to fused length: [b, nheads, n, l] → [b, nheads, n, L]
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4) // da_cumsum_bhnl1
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank]) // da_cumsum_bhnlm
        .reshape([batch, nheads, nchunks, chunk_len * mimo_rank]); // da_cumsum_bhnLM

    // ── BLUE (Y_off): exp(cumA[i]) · C[i] · h[n-1] ─────────────────────
    let exp_da_bnhLMp = da_cumsum_bhnLM
        .clone()
        .exp()
        .permute([0, 2, 1, 3]) // exp_da_bnhLM
        .unsqueeze_dim::<5>(4) // exp_da_bnhLM1
        .expand([batch, nchunks, nheads, chunk_len * mimo_rank, per_head_dim]);
    let c_bnhLMr = c_bnLMhr.permute([0, 1, 3, 2, 4]);
    let chunk_input_state_bnhrp = chunk_input_state_bnhpr.permute([0, 1, 2, 4, 3]);
    let ch_bnhLMp = c_bnhLMr.matmul(chunk_input_state_bnhrp);
    let blue_bnhLMp = ch_bnhLMp * exp_da_bnhLMp;

    // ── ORANGE (Y_diag): MIMO causal decay matrix · CB @ V ────────────────────
    let da_cumsum_bnhLM = da_cumsum_bhnLM.permute([0, 2, 1, 3]);
    let target_da_cumsum_bnhLMLM = da_cumsum_bnhLM
        .clone()
        .unsqueeze_dim::<5>(4) // da_cumsum_bnhLM1
        .expand([
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank,
        ]);
    let source_da_cumsum_bnhLMLM = da_cumsum_bnhLM
        .unsqueeze_dim::<5>(3) // da_cumsum_bnh1LM
        .expand([
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank,
        ]);
    let diff_da_cumsum_bnhLMLM = target_da_cumsum_bnhLMLM - source_da_cumsum_bnhLMLM;

    // MIMO causal neg-inf mask: −∞ where j//m > i//m (source strictly ahead of
    // target in time). Built as interleaved expansion of the standard
    // 2-dimensional upper-triangle mask.
    let neg_inf_base_bnhll =
        F::<B, 2>::full([chunk_len, chunk_len], f32::NEG_INFINITY, &device, dtype)
            .triu(1) // [chunk_len, chunk_len]: -inf above diagonal
            .unsqueeze_dims::<5>(&[0, 1, 2]) // neg_inf_base_111ll
            .expand([batch, nchunks, nheads, chunk_len, chunk_len]); // neg_inf_base_bnhll
    let neg_inf_bnhLMLM = neg_inf_base_bnhll
        .unsqueeze_dim::<6>(4) // neg_inf_base_bnhl1l
        .expand([batch, nchunks, nheads, chunk_len, mimo_rank, chunk_len]) // neg_inf_base_bnhlml
        .reshape([batch, nchunks, nheads, chunk_len * mimo_rank, chunk_len]) // neg_inf_base_bnhLMl
        .unsqueeze_dim::<6>(5) // neg_inf_base_bnhLMl1
        .expand([
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len,
            mimo_rank,
        ]) // neg_inf_base_bnhLMlm
        .reshape([
            batch,
            nchunks,
            nheads,
            chunk_len * mimo_rank,
            chunk_len * mimo_rank,
        ]); // neg_inf_bnhLMLM

    let decay_bnhLMLM = (diff_da_cumsum_bnhLMLM + neg_inf_bnhLMLM).exp();

    let v_bnhLMp = v_bnLMhp.permute([0, 1, 3, 2, 4]);
    let orange_bnhLMp = (cb_bnhLMLM * decay_bnhLMLM).matmul(v_bnhLMp);

    // ── Combine and reshape ────────────────────────────────────────────────────
    let y_bnlmhp = (blue_bnhLMp + orange_bnhLMp)
        .permute([0, 1, 3, 2, 4]) // y_bnLMhp
        .reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]); // y_bnlmhp
    y_bnlmhp
}
