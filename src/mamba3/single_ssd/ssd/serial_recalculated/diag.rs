//! # Same-step γ-correction on primitives — forward and analytic backward
//!
//! Primitive ([`F`]) port of [`super::super::diag`], plus the analytic backward
//! the recompute node needs.  The forward is used by
//! [`super::serial_recalculated`]'s K5; the backward by
//! [`super::combined_backward`].  Both carry the same SISO fast path: at
//! `mimo_rank == 1` the `m × m` Gram matrix is a scalar, so every matmul over
//! the `mimo_rank` axis degenerates into a `1×K×1` / `1×1×N` GEMM and is
//! replaced by a reduction plus broadcast multiplies.
//!
//! Forward (per `(b, n, l, h)`):
//!
//! ```text
//!   qk_dot[m_out, m_in] = Σ_r C[m_out, r] · B[m_in, r]
//!   y_diag[m_out, p]    = γ · Σ_{m_in} qk_dot[m_out, m_in] · V[m_in, p]
//! ```

#![allow(non_snake_case)]

use crate::utils::fprim::F;
use burn::backend::Backend;

/// Gradients produced by [`y_diag_correction_backward`] — the `y_diag` term's
/// contribution to `d_v`, `d_c`, `d_b` and the whole of `d_gamma` (no other
/// term consumes `γ`).
pub struct DiagGrads<B: Backend> {
    /// `y_diag`'s contribution to the gradient of `v`.
    pub d_v_bnlmhp: F<B, 6>,
    /// `y_diag`'s contribution to the gradient of `C`.
    pub d_c_bnlmhr: F<B, 6>,
    /// `y_diag`'s contribution to the gradient of `B`.
    pub d_b_bnlmhr: F<B, 6>,
    /// The gradient of `γ`.
    pub d_gamma_bnlh: F<B, 4>,
}

/// The γ-weighted same-step correction `y_diag`, on primitives.
///
/// Dispatches on `mimo_rank` and `siso_specialization`; see
/// [`super::super::diag::y_diag_correction`].
pub fn y_diag_correction<B: Backend>(
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    gamma_bnlh: F<B, 4>,
    siso_specialization: bool,
) -> F<B, 6> {
    let [.., mimo_rank, _nheads, _per_head_dim] = v_bnlmhp.dims();
    if mimo_rank == 1 && siso_specialization {
        y_diag_correction_siso(v_bnlmhp, b_bnlmhr, c_bnlmhr, gamma_bnlh)
    } else {
        y_diag_correction_mimo(v_bnlmhp, b_bnlmhr, c_bnlmhr, gamma_bnlh)
    }
}

/// SISO (`mimo_rank == 1`) `y_diag` on primitives: a `state_rank` reduction plus
/// a per-`(b, n, l, h)` scalar broadcast.
pub(crate) fn y_diag_correction_siso<B: Backend>(
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    gamma_bnlh: F<B, 4>,
) -> F<B, 6> {
    let qk_dot_bnlmh1 = (c_bnlmhr * b_bnlmhr).sum_dim(5);
    let gamma_bnl1h1 = gamma_bnlh.unsqueeze_dims::<6>(&[3, 5]);
    v_bnlmhp * qk_dot_bnlmh1 * gamma_bnl1h1
}

/// General MIMO `y_diag` on primitives.
pub(crate) fn y_diag_correction_mimo<B: Backend>(
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    gamma_bnlh: F<B, 4>,
) -> F<B, 6> {
    let c_bnlhmr = c_bnlmhr.swap_dims(3, 4);
    let b_bnlhrm = b_bnlmhr.permute([0, 1, 2, 4, 5, 3]);
    let qk_dot_bnlhmM = c_bnlhmr.matmul(b_bnlhrm); // bnlhm_outm_in
    let v_bnlhmp = v_bnlmhp.swap_dims(3, 4);
    let y_d_bnlhmp = qk_dot_bnlhmM.matmul(v_bnlhmp); // bnlhm_outp
    let gamma_bnlh11 = gamma_bnlh.unsqueeze_dims::<6>(&[4, 5]);
    (y_d_bnlhmp * gamma_bnlh11).swap_dims(3, 4)
}

/// Analytic backward of [`y_diag_correction`].
///
/// Has no recurrence, so it runs batched over all chunks at once. Dispatches on
/// `mimo_rank` / `siso_specialization` exactly like the forward.
pub fn y_diag_correction_backward<B: Backend>(
    d_y_bnlmhp: F<B, 6>,
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    gamma_bnlh: F<B, 4>,
    siso_specialization: bool,
) -> DiagGrads<B> {
    let [.., mimo_rank, _nheads, _per_head_dim] = v_bnlmhp.dims();
    if mimo_rank == 1 && siso_specialization {
        y_diag_correction_backward_siso(d_y_bnlmhp, v_bnlmhp, b_bnlmhr, c_bnlmhr, gamma_bnlh)
    } else {
        y_diag_correction_backward_mimo(d_y_bnlmhp, v_bnlmhp, b_bnlmhr, c_bnlmhr, gamma_bnlh)
    }
}

/// SISO (`mimo_rank == 1`) backward.
///
/// With `qk = Σ_r C·B` and `dyv = Σ_p dY·V` both per-`(b, n, l, h)` scalars, the
/// five MIMO matmuls become two reductions and four broadcast multiplies:
///
/// ```text
///   d_gamma   = qk · dyv
///   d_qk_dot  = γ · dyv
///   d_v[p]    = (qk · γ) · dY[p]
///   d_c[r]    = d_qk_dot · B[r]
///   d_b[r]    = d_qk_dot · C[r]
/// ```
pub(crate) fn y_diag_correction_backward_siso<B: Backend>(
    d_y_bnlmhp: F<B, 6>,
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    gamma_bnlh: F<B, 4>,
) -> DiagGrads<B> {
    let qk_dot_bnlmh1 = (c_bnlmhr.clone() * b_bnlmhr.clone()).sum_dim(5);
    let dyv_bnlmh1 = (d_y_bnlmhp.clone() * v_bnlmhp).sum_dim(5);
    let gamma_bnl1h1 = gamma_bnlh.unsqueeze_dims::<6>(&[3, 5]);

    // d_gamma[b,n,l,h] = Σ_p dY · (qk_dot · V) = qk_dot · Σ_p dY·V
    let d_gamma_bnlh: F<B, 4> = (qk_dot_bnlmh1.clone() * dyv_bnlmh1.clone())
        .squeeze_dim::<5>(5) // bnlmh
        .squeeze_dim::<4>(3); // bnlh  (mimo_rank == 1)

    let d_qk_dot_bnlmh1 = dyv_bnlmh1 * gamma_bnl1h1.clone();
    let d_v_bnlmhp = d_y_bnlmhp * (qk_dot_bnlmh1 * gamma_bnl1h1);
    let d_c_bnlmhr = b_bnlmhr * d_qk_dot_bnlmh1.clone();
    let d_b_bnlmhr = c_bnlmhr * d_qk_dot_bnlmh1;

    DiagGrads {
        d_v_bnlmhp,
        d_c_bnlmhr,
        d_b_bnlmhr,
        d_gamma_bnlh,
    }
}

/// General MIMO backward.
pub(crate) fn y_diag_correction_backward_mimo<B: Backend>(
    d_y_bnlmhp: F<B, 6>,
    v_bnlmhp: F<B, 6>,
    b_bnlmhr: F<B, 6>,
    c_bnlmhr: F<B, 6>,
    gamma_bnlh: F<B, 4>,
) -> DiagGrads<B> {
    let c_bnlhmr = c_bnlmhr.swap_dims(3, 4); // [b,n,l,h,m_out,r]
    let b_bnlhmr = b_bnlmhr.swap_dims(3, 4); // [b,n,l,h,m_in,r]
    let v_bnlhmp = v_bnlmhp.swap_dims(3, 4); // [b,n,l,h,m_in,p]
    let d_y_bnlhmp = d_y_bnlmhp.swap_dims(3, 4); // [b,n,l,h,m_out,p]

    // qk_dot[m_out, m_in] = Σ_r C[m_out,r] · B[m_in,r]
    let qk_dot_bnlhmM = c_bnlhmr.clone().matmul(b_bnlhmr.clone().transpose());
    // y_d_unweighted[m_out, p] = Σ_{m_in} qk_dot · V[m_in, p]
    let y_d_unw_bnlhmp = qk_dot_bnlhmM.clone().matmul(v_bnlhmp.clone());

    // d_gamma[b,n,l,h] = Σ_{m_out,p} d_y · y_d_unweighted
    let d_gamma_bnlh: F<B, 4> = (d_y_bnlhmp.clone() * y_d_unw_bnlhmp)
        .sum_dim(5) // bnlhm1
        .squeeze_dim::<5>(5) // bnlhm
        .sum_dim(4) // bnlh1
        .squeeze_dim::<4>(4); // bnlh

    // d_y_d_unweighted = γ · d_y  (γ broadcast over m_out, p)
    let gamma_bnlh11 = gamma_bnlh.unsqueeze_dims::<6>(&[4, 5]);
    let d_y_d_unw_bnlhmp = d_y_bnlhmp * gamma_bnlh11;

    // d_qk_dot[m_out, m_in] = Σ_p d_y_d_unweighted[m_out, p] · V[m_in, p]
    let d_qk_dot_bnlhmM = d_y_d_unw_bnlhmp.clone().matmul(v_bnlhmp.transpose()); // [b,n,l,h,m_out,m_in]

    // d_v[m_in, p] = Σ_{m_out} qk_dot[m_out, m_in] · d_y_d_unweighted[m_out, p]
    let d_v_bnlhmp = qk_dot_bnlhmM
        .transpose() // qk_dotᵀ: [b,n,l,h,m_in,m_out]
        .matmul(d_y_d_unw_bnlhmp); // [b,n,l,h,m_in,p]

    // d_C[m_out, r] = Σ_{m_in} d_qk_dot[m_out, m_in] · B[m_in, r]
    let d_c_bnlhmr = d_qk_dot_bnlhmM.clone().matmul(b_bnlhmr); // [b,n,l,h,m_out,r]
    // d_B[m_in, r] = Σ_{m_out} d_qk_dot[m_out, m_in] · C[m_out, r]
    let d_b_bnlhmr = d_qk_dot_bnlhmM
        .transpose() // d_qk_dotᵀ: [b,n,l,h,m_in,m_out]
        .matmul(c_bnlhmr); // [b,n,l,h,m_in,r]

    DiagGrads {
        d_v_bnlmhp: d_v_bnlhmp.swap_dims(3, 4),
        d_c_bnlmhr: d_c_bnlhmr.swap_dims(3, 4),
        d_b_bnlmhr: d_b_bnlhmr.swap_dims(3, 4),
        d_gamma_bnlh,
    }
}

// ---------------------------------------------------------------------------
// Tests — SISO fast path ≡ MIMO-general path (forward and backward)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
