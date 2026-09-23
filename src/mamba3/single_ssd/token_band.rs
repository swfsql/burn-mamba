//! # The lag-`u` correction band (single-SSD, [`Vertical`])
//!
//! [`Vertical`]: crate::mamba3::trapezoid::Trapezoid::Vertical
//!
//! The single-SSD pathway scales the key of sample `s` by the whole collapsed
//! weight `Δ̃ₛ = γₛ + νₛ₊ₗₐ₉` (`info/mamba-3/trapezoid-as-integration.md` §5).
//! That is right for every read `t` *after* the tap is paid (`t − s ≥ lag`). It
//! is wrong for the `lag` reads before it, where the weight must still be
//! `γₛ`. At lag 1 that exception is only the diagonal, and
//! [`ssd::diag`](crate::mamba3::single_ssd::ssd::diag) corrects it inside the
//! kernel. At lag `u` it is a `u`-wide **band** (§9).
//!
//! The caller passes the excess in. It is not recomputed from `scale − γ`,
//! because the scale of a two-tap pattern also holds a **lag-1** installment.
//! That installment has already landed at every read that the band covers,
//! except the diagonal, which the kernel replaces anyway. So only the lag-`u`
//! mass belongs here.
//!
//! ## Why the band never has to enter the kernel
//!
//! The band would cross chunk boundaries, and the part of it that arrived
//! through the initial state of the chunk could not be un-weighted. But it
//! does not have to enter the kernel. The only outputs of `forward` are at the
//! **last** micro-step of each token (the [read axis](crate::mamba3::product)
//! of the chunk). For a read at folded position `p = τ·u + (u−1)`, the band
//! `{p−u+1 … p}` is *exactly token `τ`*. So the correction is one small
//! contraction per token, applied after the kernel at token resolution. It
//! needs no mask change, no chunk-length constraint and no cross-chunk term.
//!
//! This is the one place where the two pathways do not agree pointwise. They
//! stay equivalent on **everything that a caller can observe**: the output of
//! the block and every field of the returned cache. That is all that
//! `forward_single_ssd` promises. The correction does not apply to the
//! intermediate `y` at the `u−1` folded positions per token that the read axis
//! does not read (and does not compute): partial sums on the way to the read,
//! never a value that the block returns. The **state** is exact at every
//! position in both pathways. The caches carry it, and a split prefill
//! continues from it.
//!
//! A correction of those positions too is possible, but not free. Their band
//! reaches back over a token boundary. So part of it arrives through the
//! initial state of the chunk (or, in the first token of the call, through the
//! boundary β seed), already weighted by `scale`, where it cannot be
//! un-weighted. This module does not pay that cost.
//!
//! ## The term
//!
//! The diagonal of the kernel is already `γ`, so what is left is `j < u−1`:
//!
//! ```text
//!   corr[τ, m_out, h, p] = Σ_{j<u−1} νᵗᵃᵖ[τ,j] · dcy[τ,j]
//!                          · Σ_{m_in} (C[τ,u−1,m_out]·B[τ,j,m_in]) · V[τ,j,m_in,p]
//!
//!   νᵗᵃᵖ[τ,j] = νᶠᵃʳₛ₊ᵤ at s = (τ,j)     dcy[τ,j] = exp(Σ_{r=j+1}^{u−1} da[τ,r])
//! ```
//!
//! `dcy` is the scalar decay from the tapped position to the read. The
//! relative *rotation* needs no factor, because `C̄`/`B̄` already carry it.

use burn::prelude::*;

/// The intra-token part of the `lag`-wide correction band, to be **subtracted**
/// from the single-SSD output at token resolution.
///
/// `None` at `micro_steps == 1` (the band is the diagonal, which the kernel
/// already corrects). Callers use it only at `lag == micro_steps`, under one
/// of the lag-`u` patterns
/// ([`Trapezoid::Vertical`](crate::mamba3::trapezoid::Trapezoid::Vertical) and
/// the two that add a tap to it). `excess_bsh` is the **lag-`u`** mass,
/// shifted back to the sample that owes it (`νₛ₊ᵤ`), never the whole
/// `scaleₛ − γₛ`.
///
/// # Shapes
/// - `b_bsmhr`             : `[batch, sequence, mimo_rank, nheads, state_rank]`
/// - `c_btmhr`             : `[batch, tokens, mimo_rank, nheads, state_rank]` —
///   already on the read axis, like everywhere else `C` appears
/// - `v_bsmhp`             : `[batch, sequence, mimo_rank, nheads, per_head_dim]`
/// - `excess_bsh`, `da_bsh`: `[batch, sequence, nheads]`
/// - out                   : `[batch, tokens, mimo_rank, nheads, per_head_dim]`
///
/// where `sequence = tokens · micro_steps` is the folded axis.
#[allow(non_snake_case)]
pub fn token_band_correction(
    v_bsmhp: Tensor<5>,
    b_bsmhr: Tensor<5>,
    c_btmhr: Tensor<5>,
    excess_bsh: Tensor<3>,
    da_bsh: Tensor<3>,
    micro_steps: usize,
) -> Option<Tensor<5>> {
    let u = micro_steps;
    if u == 1 {
        return None;
    }
    let [batch, sequence, mimo_rank, nheads, per_head_dim] = v_bsmhp.dims();
    let [.., state_rank] = b_bsmhr.dims();
    let tokens = sequence / u;
    debug_assert_eq!(tokens, c_btmhr.dims()[1], "C is on the read axis");
    // The `u−1` tapped positions per token; the `u`-th is the read itself, whose
    // weight the kernel's γ-diagonal already fixed.
    let taps = u - 1;

    // ── The per-tap scalar: the unpaid excess, decayed to the read ────────────
    // `Σ_{r=j+1}^{u−1} da` within the token, i.e. the reverse-exclusive
    // cumulative log-decay of the token's own window.
    let da_btuh = da_bsh.reshape([batch, tokens, u, nheads]);
    let cumulative_btuh = da_btuh.cumsum(2);
    let total_bt1h = cumulative_btuh.clone().narrow(2, u - 1, 1);
    let decay_btJh = (total_bt1h - cumulative_btuh).exp().narrow(2, 0, taps);
    let excess_btJh = excess_bsh
        .reshape([batch, tokens, u, nheads])
        .narrow(2, 0, taps);
    let weight_btJh = excess_btJh * decay_btJh;

    // ── Fuse (tap, mimo_rank): both are outer products into one state ─────────
    let fused = taps * mimo_rank;
    let v_btJmhp = v_bsmhp
        .reshape([batch, tokens, u, mimo_rank, nheads, per_head_dim])
        .narrow(2, 0, taps);
    let v_weighted_btKhp = (v_btJmhp * weight_btJh.unsqueeze_dims::<6>(&[3, 5]))
        .reshape([batch, tokens, fused, nheads, per_head_dim]);
    let b_btKhr = b_bsmhr
        .reshape([batch, tokens, u, mimo_rank, nheads, state_rank])
        .narrow(2, 0, taps)
        .reshape([batch, tokens, fused, nheads, state_rank]);
    // The read `C` is the token's last micro-step — the copy carrying the
    // cumulative rotation the readout happens at, which is the only one it is
    // ever built at.

    // ── (C · Bᵀ) · V, contracting state_rank then the fused axis ──────────────
    let c_bthmr = c_btmhr.swap_dims(2, 3);
    let b_bthrK = b_btKhr.permute([0, 1, 3, 4, 2]);
    let qk_bthmK = c_bthmr.matmul(b_bthrK);
    let v_bthKp = v_weighted_btKhp.swap_dims(2, 3);
    let corr_bthmp = qk_bthmK.matmul(v_bthKp);
    Some(corr_bthmp.swap_dims(2, 3))
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
