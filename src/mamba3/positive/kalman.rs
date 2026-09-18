//! The Kalman gate: a per-head precision `Λ`, scanned from the inputs, and the
//! decay `dₜ = αₜ/(1 + qₜ·αₜ·Λₜ₋₁)` it computes (see the module header one
//! level up).
//!
//! Everything is in log coordinates: `ℓ = ln Λ`, and the decay is formed as a
//! log-decay correction, `ln dₜ = ln αₜ − softplus(ln qₜ + ln αₜ + ℓₜ₋₁)`, so
//! that `κ = 0` (`ln q = −∞`, floored to [`LOG_ZERO`]) leaves `ln αₜ`
//! untouched bit for bit.

use super::LOG_ZERO;
use super::scan::{self, Mobius};
use burn::prelude::*;
use burn_stack::modules::softplus;

/// What the gate reads besides the discretisation: the per-head `ln κ`, the
/// projected noise (if any) and the carried `ℓ`.
pub struct GainInput {
    /// `ln κₕ`, `[nheads]`. `−∞` is the stock block.
    pub log_kappa_h: Tensor<1>,
    /// `rₜ`, `[batch, len, nheads]`, under
    /// [`Gain::KalmanProjectedNoise`](super::Gain::KalmanProjectedNoise).
    pub noise_bsh: Option<Tensor<3>>,
    /// `ℓ` before the first position, `[batch, nheads]` — [`LOG_ZERO`] for a
    /// fresh sequence (`Λ₀ = 0`: no evidence yet).
    pub carry_bh: Tensor<2>,
}

/// The gate's per-position quantities.
pub struct GainOutput {
    /// `ln dₜ`, which replaces `Δₜ·Aₜ` wherever the block reads a log-decay.
    pub da_bsh: Tensor<3>,
    /// `ℓₜ = ln Λₜ` after position `t`'s update, `[batch, len, nheads]`; the
    /// last one is the next call's carry.
    pub log_precision_bsh: Tensor<3>,
}

/// The per-position Möbius elements `ln [[α(1+qm), m], [αq, 1]]` of the tied
/// arm (`m = Δ`), and `ln q` for the decay.
///
/// # Shapes
/// - `dt_bsh`, `da_bsh` : `[batch, len, nheads]` (`Δ`, stock `ln α = Δ·A`)
/// - out                : elements and `ln q`, `[batch, len, nheads]`
pub fn elements(dt_bsh: Tensor<3>, da_bsh: Tensor<3>, input: &GainInput) -> (Mobius, Tensor<3>) {
    // `Δ` may clamp to 0 (`dt_limit`), and `ln 0` must not reach a log-sum-exp.
    let ln_m = dt_bsh.log().clamp_min(LOG_ZERO);
    let ln_q = input.log_kappa_h.clone().unsqueeze::<3>() + ln_m.clone();
    let ln_q = match &input.noise_bsh {
        Some(noise_bsh) => ln_q + noise_bsh.clone(),
        None => ln_q,
    };
    let ln_q = ln_q.clamp_min(LOG_ZERO);
    let m00 = (da_bsh.clone() + softplus(ln_q.clone() + ln_m.clone())).clamp_min(LOG_ZERO);
    let m10 = (da_bsh + ln_q.clone()).clamp_min(LOG_ZERO);
    let m11 = m10.zeros_like();
    (
        Mobius {
            m00,
            m01: ln_m,
            m10,
            m11,
        },
        ln_q,
    )
}

/// Run the gate over a folded run of positions (the whole sequence in
/// `forward`, one token's `u` micro-steps in `step`).
///
/// # Shapes
/// - `dt_bsh`, `da_bsh` : `[batch, len, nheads]`
pub fn gate(dt_bsh: Tensor<3>, da_bsh: Tensor<3>, input: GainInput) -> GainOutput {
    let [batch, len, nheads] = da_bsh.dims();
    let (elements, ln_q) = elements(dt_bsh, da_bsh.clone(), &input);
    let log_precision_bsh = scan::prefix(elements).apply(input.carry_bh.clone());
    // ℓₜ₋₁: the carry, then every position but the last.
    let carry_b1h = input.carry_bh.reshape([batch, 1, nheads]);
    let prev_bsh = if len == 1 {
        carry_b1h
    } else {
        Tensor::cat(vec![carry_b1h, log_precision_bsh.clone().narrow(1, 0, len - 1)], 1)
    };
    let da_bsh = da_bsh.clone() - softplus(ln_q + da_bsh + prev_bsh);
    GainOutput {
        da_bsh,
        log_precision_bsh,
    }
}
