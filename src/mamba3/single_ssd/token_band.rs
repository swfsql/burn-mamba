//! # The lag-`u` correction band (single-SSD, [`Vertical`])
//!
//! [`Vertical`]: crate::mamba3::trapezoid::Trapezoid::Vertical
//!
//! The single-SSD pathway scales sample `s`'s key by the whole collapsed weight
//! `Δ̃ₛ = γₛ + νₛ₊ₗₐ₉` (`info/trapezoid-as-integration.md` §5), which is right for
//! every read `t` that happens *after* the tap has been paid, i.e. `t − s ≥ lag`,
//! and wrong for the `lag` reads before it, where the weight must still be `γₛ`.
//! At lag 1 that exception is the diagonal alone and
//! [`ssd::diag`](crate::mamba3::single_ssd::ssd::diag) handles it inside the
//! kernel; at lag `u` it is a `u`-wide **band** (§9).
//!
//! ## Why the band never has to enter the kernel
//!
//! The band would straddle chunk boundaries, and the part of it that arrived
//! through the chunk's initial state could not be un-weighted. It does not have
//! to: the only outputs that survive `forward` are each token's **last**
//! micro-step ([`crate::mamba3::product::last_micro5`]), and for a read at
//! folded position `p = τ·u + (u−1)` the band `{p−u+1 … p}` is *exactly token
//! `τ`*. So the correction is one small contraction per token, applied after the
//! kernel at token resolution, with no mask change, no chunk-length constraint
//! and no cross-chunk term.
//!
//! What this gives up is worth stating exactly, because it is the one place the
//! two pathways stop agreeing pointwise. The two remain equivalent on
//! **everything a caller can observe** — the block's output and every field of
//! the returned cache — and that is all `forward_single_ssd` ever promised. What
//! is not corrected is the intermediate `y` at the `u−1` folded positions per
//! token that [`last_micro5`](crate::mamba3::product::last_micro5) discards:
//! partial sums on the way to the read, never a value the block computes. The
//! **state** is exact at every position in both pathways, which is what the
//! caches carry and what a split prefill continues from.
//!
//! Correcting those positions too is possible, not free: their band reaches back
//! over a token boundary, so part of it arrives through the chunk's initial state
//! (or, in the call's first token, through the boundary β seed) already weighted
//! by `scale`, where it can no longer be un-weighted. That is the cost this
//! module declines to pay.
//!
//! ## The term
//!
//! The kernel's diagonal is already `γ`, so what is left over is `j < u−1`:
//!
//! ```text
//!   corr[τ, m_out, h, p] = Σ_{j<u−1} νᵗᵃᵖ[τ,j] · dcy[τ,j]
//!                          · Σ_{m_in} (C[τ,u−1,m_out]·B[τ,j,m_in]) · V[τ,j,m_in,p]
//!
//!   νᵗᵃᵖ[τ,j] = Δ̃ − γ  at (τ,j)          dcy[τ,j] = exp(Σ_{r=j+1}^{u−1} da[τ,r])
//! ```
//!
//! `dcy` is the scalar decay from the tapped position to the read; the relative
//! *rotation* needs no factor at all, being already carried by `C̄`/`B̄`.

use burn::prelude::*;

/// The intra-token part of the `lag`-wide correction band, to be **subtracted**
/// from the single-SSD output at token resolution.
///
/// `None` at `micro_steps == 1` (the band is the diagonal, already the kernel's)
/// — and callers only reach it at `lag == micro_steps`, the pattern
/// [`Trapezoid::Vertical`](crate::mamba3::trapezoid::Trapezoid::Vertical).
///
/// # Shapes
/// - `b_bsmhr`, `c_bsmhr`  : `[batch, sequence, mimo_rank, nheads, state_rank]`
/// - `v_bsmhp`             : `[batch, sequence, mimo_rank, nheads, per_head_dim]`
/// - `excess_bsh`, `da_bsh`: `[batch, sequence, nheads]`
/// - out                   : `[batch, tokens, mimo_rank, nheads, per_head_dim]`
///
/// where `sequence = tokens · micro_steps` is the folded axis.
#[allow(non_snake_case)]
pub fn token_band_correction(
    v_bsmhp: Tensor<5>,
    b_bsmhr: Tensor<5>,
    c_bsmhr: Tensor<5>,
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
    // cumulative rotation the readout happens at.
    let c_btmhr = c_bsmhr
        .reshape([batch, tokens, u, mimo_rank, nheads, state_rank])
        .narrow(2, u - 1, 1)
        .squeeze_dim::<5>(2);

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
