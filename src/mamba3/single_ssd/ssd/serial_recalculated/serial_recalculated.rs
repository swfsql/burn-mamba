//! # Serial SSD with a custom, memory-efficient backward (Mamba-3 single-SSD)
//!
//! The `SerialRecalculated` path for the single-SSD pathway.  The forward is the
//! same serial scan as [`super::super::serial`], routed through the
//! [`Mamba3SingleSsdBackendExt`] trait so `Autodiff` backends substitute a
//! custom backward that recomputes per-chunk intermediates rather than storing
//! them (see [`super::backward`] / [`super::combined_backward`]).  Unlike the
//! double-SSD form, the kernels here apply the trapezoid `gamma`/`scale` and the
//! boundary-β seed internally, so the backward also returns `d_gamma`/`d_scale`.
//!
//! The default body runs under a generic backend `B`, where the high-level
//! `Tensor` (pinned to `Dispatch`) is unavailable, so the K1–K5 math goes
//! through the rank-tagged [`F`] primitive wrapper.  K1–K4 are mode-agnostic and
//! reused from the double-SSD forward; only the single-SSD K5 (strict-lower
//! intra-chunk + γ-correction) is owned here, and it is forward-only.

#![allow(non_snake_case)]

use crate::mamba3::double_ssd::ssd::serial_recalculated::{
    k1_ssd_chunk_cumsum, k2_ssd_bmm, k3_ssd_chunk_state, k4_ssd_state_passing,
};
use crate::mamba3::single_ssd::prelude::*;
use crate::utils::fprim::{F, san};
use burn::backend::tensor::FloatTensor;
use burn::backend::*;
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::Tensor;

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

        let (y_bnlmhp, final_state_bhpr) =
            <Dispatch as Mamba3SingleSsdBackendExt>::single_ssd_serial_recalculated(
                input.v_bnlmhp.into_primitive(),
                input.da_bnlh.into_primitive(),
                input.b_bnlmhr.into_primitive(),
                input.c_bnlmhr.into_primitive(),
                input.gamma_bnlh.into_primitive(),
                input.scale_bnlh.into_primitive(),
                input.initial_state_bhpr.into_primitive(),
            );
        let y_bnlmhp = Tensor::from_primitive(y_bnlmhp);
        let final_state_bhpr = Tensor::from_primitive(final_state_bhpr);
        (y_bnlmhp, final_state_bhpr)
    }
}

/// Extends the backend for the memory-efficient single-ssd form serial SSD.
///
/// The default implementation runs K1–K5 using primitive tensor operations,
/// reusing the mode-agnostic K1/K2/K3/K4 from the double-SSD forward and the
/// single-ssd form K5 below. Backends that support a custom memory-efficient
/// backward (the Autodiff wrapper) override this to recompute forward
/// intermediates during backward instead of saving them.
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
        // Default impl: replicate the single-ssd form K1–K5 on primitives.
        let v_bnlmhp = F::<Self, 6>::new(v_bnlmhp);
        let da_bnlh = F::<Self, 4>::new(da_bnlh);
        let b_bnlmhr = F::<Self, 6>::new(b_bnlmhr);
        let c_bnlmhr = F::<Self, 6>::new(c_bnlmhr);
        let gamma_bnlh = F::<Self, 4>::new(gamma_bnlh);
        let scale_bnlh = F::<Self, 4>::new(scale_bnlh);
        let initial_state_bhpr = F::<Self, 4>::new(initial_state_bhpr);

        // K1 — chunk cumulative decay.
        let (da_cumsum_bhnl, da_chunk_end_bhn) = k1_ssd_chunk_cumsum::<Self>(da_bnlh);
        san(&da_cumsum_bhnl);

        // K2 — CB matrix on unscaled B/C.
        let cb_bnhLMLM = k2_ssd_bmm::<Self>(c_bnlmhr.clone(), b_bnlmhr.clone());
        san(&cb_bnhLMLM);

        // K3 — chunk state on K_scaled = scaleₜ · B.
        let scale_bnlh11 = scale_bnlh.clone().unsqueeze_dims::<6>(&[3, 5]);
        let k_scaled_bnlmhr = b_bnlmhr.clone() * scale_bnlh11;
        let intra_chunk_state_bnhpr =
            k3_ssd_chunk_state::<Self>(v_bnlmhp.clone(), k_scaled_bnlmhr, da_cumsum_bhnl.clone());
        san(&intra_chunk_state_bnhpr);

        // K4 — sequential state passing.
        let (chunk_input_state_bnhpr, final_state_bhpr) = k4_ssd_state_passing::<Self>(
            intra_chunk_state_bnhpr,
            da_chunk_end_bhn,
            initial_state_bhpr,
        );
        san(&chunk_input_state_bnhpr);
        san(&final_state_bhpr);

        // K5 — single-ssd form chunk scan.
        let y_bnlmhp = k5_single_ssd_chunk_scan::<Self>(
            da_cumsum_bhnl,
            v_bnlmhp,
            c_bnlmhr,
            b_bnlmhr,
            cb_bnhLMLM,
            gamma_bnlh,
            scale_bnlh,
            chunk_input_state_bnhpr,
        );
        san(&y_bnlmhp);

        (y_bnlmhp.inner(), final_state_bhpr.inner())
    }
}

crate::decl_ssd_autodiff_backend_ext!(Mamba3SingleSsdAutodiffBackendExt, Mamba3SingleSsdBackendExt);

// ---------------------------------------------------------------------------
// Per-backend impls: each delegates to the trait's default (K1–K5) body. The
// custom autodiff backward lives in `super::backward` as a separate impl.
// ---------------------------------------------------------------------------
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3SingleSsdBackendExt);

// ─── Primitive forward K5 (single-ssd) ───────────────────────────────────────
// Primitive port of [`super::super::serial::k5_single_ssd_chunk_scan`]. Combines
// the strict-lower intra-chunk (LOWER), the γ-weighted same-step (DIAG), and the
// state-to-output (BLUE/Y_off) contributions. Forward-only; the backward
// computes these gradients analytically. K1–K4 are reused from the double-SSD
// forward (mode-agnostic).

/// SingleSsd chunk scan, on primitives.
///
/// - **Strict lower triangular intra-chunk** (`t1 > t2`):
///   `(cb[i,j] · scale[t2] · exp(cumA[t1] − cumA[t2])) · V[t2]`
/// - **Same-time-step (`t1 == t2`) γ-correction**:
///   `γ[t] · (Σₙ C[t,r_out,n] · B[t,r_in,n]) · V[t,r_in,p]`
/// - **State-to-output (Y_off)**: `exp(cumA[t]) · C[t] · h'[n-1]`
///
/// `cb_bnhLMLM` is the unscaled `C · Bᵀ` from K2; `b_bnlmhr` is the unscaled
/// K/B tensor (γ-correction matmul).
#[allow(clippy::too_many_arguments)]
fn k5_single_ssd_chunk_scan<B: Backend>(
    da_cumsum_bhnl: F<B, 4>,
    v_bnlmhp: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    cb_bnhLMLM: F<B, 5>,
    gamma_bnlh: F<B, 4>,
    scale_bnlh: F<B, 4>,
    chunk_input_state_bnhpr: F<B, 5>,
) -> F<B, 6> {
    let [batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim] = v_bnlmhp.dims();
    let [.., state_rank] = c_bnlmhr.dims();
    let device = v_bnlmhp.device();
    let dtype = v_bnlmhp.dtype();
    let fused = chunk_len * mimo_rank;

    // Fuse mimo_rank into chunk_len for the SSM-style matmul.
    let v_bnLMhp = v_bnlmhp
        .clone()
        .reshape([batch, nchunks, fused, nheads, per_head_dim]);
    let c_bnLMhr = c_bnlmhr
        .clone()
        .reshape([batch, nchunks, fused, nheads, state_rank]);

    // Per-fused-step cumulative decay (interleave-expand the base grid).
    let da_cumsum_bhnLM = da_cumsum_bhnl
        .unsqueeze_dim::<5>(4)
        .expand([batch, nheads, nchunks, chunk_len, mimo_rank])
        .reshape([batch, nheads, nchunks, fused]);

    // ── Y_off: exp(cumA[t]) · C[t] · h'[n-1] ────────────────────────────────
    let exp_da_bnhLMp = da_cumsum_bhnLM
        .clone()
        .exp()
        .permute([0, 2, 1, 3]) // bnhLM
        .unsqueeze_dim::<5>(4) // bnhLM1
        .expand([batch, nchunks, nheads, fused, per_head_dim]);
    let c_bnhLMr = c_bnLMhr.permute([0, 1, 3, 2, 4]);
    let chunk_input_state_bnhrp = chunk_input_state_bnhpr.permute([0, 1, 2, 4, 3]);
    let ch_bnhLMp = c_bnhLMr.matmul(chunk_input_state_bnhrp);
    let y_off_bnhLMp = ch_bnhLMp * exp_da_bnhLMp;

    // ── Y_lower: strict lower-tri intra-chunk with scale and decay ──────────
    let da_cumsum_bnhLM = da_cumsum_bhnLM.permute([0, 2, 1, 3]); // bnhLM
    let target_da_cumsum_bnhLMLM = da_cumsum_bnhLM
        .clone()
        .unsqueeze_dim::<5>(4) // bnhLM1
        .expand([batch, nchunks, nheads, fused, fused]);
    let source_da_cumsum_bnhLMLM = da_cumsum_bnhLM
        .unsqueeze_dim::<5>(3) // bnh1LM
        .expand([batch, nchunks, nheads, fused, fused]);
    let diff_bnhLMLM = target_da_cumsum_bnhLMLM - source_da_cumsum_bnhLMLM;

    // Strict-upper -inf mask on the base time grid (`t1 <= t2` → -inf), then
    // interleave-expand to fused length so MIMO same-time blocks are zeroed.
    let inf_upper_bnhLMLM =
        F::<B, 2>::full([chunk_len, chunk_len], f32::NEG_INFINITY, &device, dtype)
            .triu(0) // upper triangle INCLUDING diagonal
            .unsqueeze_dims::<5>(&[0, 1, 2])
            .expand([batch, nchunks, nheads, chunk_len, chunk_len])
            .unsqueeze_dim::<6>(4)
            .expand([batch, nchunks, nheads, chunk_len, mimo_rank, chunk_len])
            .reshape([batch, nchunks, nheads, fused, chunk_len])
            .unsqueeze_dim::<6>(5)
            .expand([batch, nchunks, nheads, fused, chunk_len, mimo_rank])
            .reshape([batch, nchunks, nheads, fused, fused]);
    let decay_strict_bnhLMLM = (diff_bnhLMLM + inf_upper_bnhLMLM).exp();

    // Per-column scale: `scale[t2]` lives on the source axis (column).
    let scale_bnhLM = scale_bnlh
        .permute([0, 1, 3, 2]) // bnhl
        .unsqueeze_dim::<5>(4) // bnhl1
        .expand([batch, nchunks, nheads, chunk_len, mimo_rank])
        .reshape([batch, nchunks, nheads, fused]);
    let scale_col_bnhLMLM = scale_bnhLM
        .unsqueeze_dim::<5>(3) // bnh1LM
        .expand([batch, nchunks, nheads, fused, fused]);

    let kernel_bnhLMLM = decay_strict_bnhLMLM * scale_col_bnhLMLM;
    let masked_cb_bnhLMLM = cb_bnhLMLM * kernel_bnhLMLM;
    let v_bnhLMp = v_bnLMhp.permute([0, 1, 3, 2, 4]);
    let y_lower_bnhLMp = masked_cb_bnhLMLM.matmul(v_bnhLMp);

    // ── Y_diag: γ-weighted same-step correction ─────────────────────────────
    let c_bnlhmr = c_bnlmhr.permute([0, 1, 2, 4, 3, 5]);
    let b_bnlhrm = b_bnlmhr.permute([0, 1, 2, 4, 5, 3]);
    let qk_dot_bnlhmM = c_bnlhmr.matmul(b_bnlhrm); // bnlhm_outm_in
    let v_bnlhmp = v_bnlmhp.permute([0, 1, 2, 4, 3, 5]);
    let y_d_bnlhmp = qk_dot_bnlhmM.matmul(v_bnlhmp); // bnlhm_outp
    let gamma_bnlh11 = gamma_bnlh.unsqueeze_dims::<6>(&[4, 5]);
    let y_d_bnlhmp_scaled = y_d_bnlhmp * gamma_bnlh11;

    let y_diag_bnlmhp = y_d_bnlhmp_scaled.permute([0, 1, 2, 4, 3, 5]);
    let y_diag_bnLMhp = y_diag_bnlmhp.reshape([batch, nchunks, fused, nheads, per_head_dim]);
    let y_diag_bnhLMp = y_diag_bnLMhp.permute([0, 1, 3, 2, 4]);

    // ── Combine and reshape ─────────────────────────────────────────────────
    let y_bnhLMp = y_off_bnhLMp + y_lower_bnhLMp + y_diag_bnhLMp;
    let y_bnLMhp = y_bnhLMp.permute([0, 1, 3, 2, 4]);
    y_bnLMhp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim])
}
