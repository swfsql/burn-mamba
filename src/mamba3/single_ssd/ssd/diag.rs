//! # Same-step γ-correction (the single-SSD diagonal term)
//!
//! The single-SSD recurrence scales `K` by `scaleₜ = γₜ + (1−λₜ₊₁)Δₜ₊₁`, which is
//! the right weight for every source step `s < t` but *not* for the same step
//! `s = t`, where the weight must be `γₜ`. The intra-chunk path therefore masks
//! the diagonal out (strict lower triangle) and this module adds it back:
//!
//! ```text
//!   y_diag[t, m_out, h, p] = γₜ · Σ_{m_in} (Σ_r C[t, m_out, h, r] · B[t, m_in, h, r])
//!                                        · V[t, m_in, h, p]
//! ```
//!
//! It is computed fresh (a small same-step product) rather than extracted from
//! the block diagonal of the fused `L·M` CB matrix, which would need a fiddly
//! reshape.
//!
//! ## SISO fast path
//!
//! The inner `m × m` Gram matrix is what makes this a pair of matmuls. At
//! `mimo_rank == 1` it collapses to the **scalar** `Cₜ·Bₜ`, so both matmuls
//! degenerate into `1×r×1` and `1×1×p` GEMMs — thousands of tiny batched
//! products, one per `(batch, nchunks, chunk_len, nheads)`.
//! [`y_diag_correction_siso`] instead contracts `state_rank` with a reduction
//! and folds the result (together with `γₜ`) in as a per-`(b, n, l, h)` scalar
//! broadcast. Both branches compute the same quantity; only the op mix differs,
//! so [`Mamba3Config::siso_specialization`](crate::mamba3::mamba3::Mamba3Config::siso_specialization)
//! can force the general branch at `mimo_rank == 1` to measure the difference.
//!
//! Reference kernels:
//! - SISO: `refs/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py`
//! - MIMO: `refs/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py`

#![allow(non_snake_case)]

use burn::prelude::*;

/// The γ-weighted same-step correction `y_diag`, dispatching to the SISO fast
/// path ([`y_diag_correction_siso`]) or the general MIMO path
/// ([`y_diag_correction_mimo`]) on `mimo_rank`.
///
/// `siso_specialization` is
/// [`Mamba3Config::siso_specialization`](crate::mamba3::mamba3::Mamba3Config::siso_specialization):
/// `false` keeps the general branch even at `mimo_rank == 1`.
///
/// # Shapes
/// - `v_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
/// - `b_bnlmhr`, `c_bnlmhr`: `[batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]`
/// - `gamma_bnlh`: `[batch, nchunks, chunk_len, nheads]`
/// - returns `y_diag_bnlmhp`: `[batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]`
pub fn y_diag_correction(
    v_bnlmhp: Tensor<6>,
    b_bnlmhr: Tensor<6>,
    c_bnlmhr: Tensor<6>,
    gamma_bnlh: Tensor<4>,
    siso_specialization: bool,
) -> Tensor<6> {
    let [.., mimo_rank, _nheads, _per_head_dim] = v_bnlmhp.dims();
    if mimo_rank == 1 && siso_specialization {
        y_diag_correction_siso(v_bnlmhp, b_bnlmhr, c_bnlmhr, gamma_bnlh)
    } else {
        y_diag_correction_mimo(v_bnlmhp, b_bnlmhr, c_bnlmhr, gamma_bnlh)
    }
}

/// SISO (`mimo_rank == 1`) `y_diag`: `qk_dot` is the scalar `Σ_r Cₜ·Bₜ` per
/// `(b, n, l, h)`, so both MIMO matmuls become a reduction plus broadcasts.
pub(crate) fn y_diag_correction_siso(
    v_bnlmhp: Tensor<6>,
    b_bnlmhr: Tensor<6>,
    c_bnlmhr: Tensor<6>,
    gamma_bnlh: Tensor<4>,
) -> Tensor<6> {
    let qk_dot_bnlmh1: Tensor<6> = (c_bnlmhr * b_bnlmhr).sum_dim(5);
    let gamma_bnl1h1 = gamma_bnlh.unsqueeze_dims::<6>(&[3, 5]);
    v_bnlmhp * qk_dot_bnlmh1 * gamma_bnl1h1
}

/// General MIMO `y_diag`: the `m_out × m_in` Gram matrix `C · Bᵀ` (contracted
/// over `state_rank`) applied to `V`, then scaled by `γₜ`.
pub(crate) fn y_diag_correction_mimo(
    v_bnlmhp: Tensor<6>,
    b_bnlmhr: Tensor<6>,
    c_bnlmhr: Tensor<6>,
    gamma_bnlh: Tensor<4>,
) -> Tensor<6> {
    // c_bnlmhr [b, n, l, m, h, r] -> c_bnlhmr [b, n, l, h, m, r]
    // b_bnlmhr [b, n, l, m, h, r] -> b_bnlhrm [b, n, l, h, r, m]
    let c_bnlhmr = c_bnlmhr.swap_dims(3, 4);
    let b_bnlhrm = b_bnlmhr.permute([0, 1, 2, 4, 5, 3]);
    // qk_dot_bnlhmM [b, n, l, h, m_out, m_in]
    let qk_dot_bnlhmM = c_bnlhmr.matmul(b_bnlhrm);

    // V in [b, n, l, h, m_in, p] layout for the next matmul, then (qk_dot) · V.
    let v_bnlhmp = v_bnlmhp.swap_dims(3, 4);
    let y_d_bnlhmp = qk_dot_bnlhmM.matmul(v_bnlhmp);

    // Multiply by γₜ (per (batch, nchunks, chunk_len, nheads)), back to bnlmhp.
    let gamma_bnlh11 = gamma_bnlh.unsqueeze_dims::<6>(&[4, 5]);
    (y_d_bnlhmp * gamma_bnlh11).swap_dims(3, 4)
}

// ---------------------------------------------------------------------------
// Tests — SISO fast path ≡ MIMO-general path
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
