//! The Kalman gate: a per-head precision `Λ`, scanned from the inputs, and the
//! decay it computes (see the module header one level up):
//!
//! ```text
//!   Lₜ = Λₜ₋₁ + νₜ                      the left endpoint's late installment
//!   dₜ = αₜ / (1 + qₜ·αₜ·Lₜ)            the predict
//!   Λₜ = dₜ·Lₜ + γₜ                      the right endpoint's measurement
//! ```
//!
//! `γ`/`ν` are the two endpoint masses of the trapezoid (`γ = Δ` and no `ν`
//! under [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None)). The
//! plant pays the left-endpoint installment of a sample one step late,
//! transported by the decay of that step (`β = ν·α`, with `d` for `α`). So `Λ`
//! **is** the recurrence of the plant on ones: the total weight of every
//! sample it wrote (the zero tap slot of a fresh cache included). And `η/Λ` is
//! a weighted mean of what it wrote. Each step is still one Möbius map: shift
//! by `ν`, predict, and shift by `γ` compose to
//! `ln [[α(1+qγ), αν + γ(1+qαν)], [αq, 1+qαν]]`. So the scan is unchanged.
//!
//! That identity is exact for a **lag-1** tap: every pattern at `u = 1`, and
//! the default at any `u`. A lag-`u` tap (the `Vertical*` patterns at `u > 1`)
//! is transported across its whole gap, `∏ d` over `u` steps. That reads `u`
//! earlier precisions, so it is not a 2×2 map. Its installment enters `L` like
//! a lag-1 installment, transported by `dₜ` alone, and `Λ` over-counts the
//! plant by `νₜ·(dₜ − ∏ d)` per step. `Λ` is then an upper bound on the weight
//! of the plant, not the weight itself. Every other property (the ceiling, the
//! contraction, the stock join) holds unchanged.
//!
//! Everything is in log coordinates: `ℓ = ln Λ`. The decay is a log-decay
//! correction, `ln dₜ = ln αₜ − softplus(ln qₜ + ln αₜ + ln Lₜ)`. So `κ = 0`
//! (`ln q = −∞`, floored to [`LOG_ZERO`]) leaves `ln αₜ` unchanged, bit for
//! bit.

use super::LOG_ZERO;
use super::scan::{self, Mobius};
use burn::prelude::*;
use burn::tensor::DType;
use burn_stack::modules::softplus;

/// What the gate reads in addition to the discretisation: the per-head `ln κ`,
/// the projected noise (if any) and the carried `ℓ`.
pub struct GainInput {
    /// `ln κₕ`, `[nheads]`. `−∞` is the stock block.
    pub log_kappa_h: Tensor<1>,
    /// `rₜ`, `[batch, len, nheads]`, under
    /// [`Gain::KalmanProjectedNoise`](super::Gain::KalmanProjectedNoise).
    pub noise_bsh: Option<Tensor<3>>,
    /// `ℓ` before the first position, `[batch, nheads]`: [`LOG_ZERO`] for a
    /// fresh sequence (`Λ₀ = 0`: no evidence yet).
    pub carry_bh: Tensor<2>,
}

/// The per-position quantities of the gate.
pub struct GainOutput {
    /// `ln dₜ`, which replaces `Δₜ·Aₜ` wherever the block reads a log-decay.
    pub da_bsh: Tensor<3>,
    /// `ℓₜ = ln Λₜ` after the update of position `t`, `[batch, len, nheads]`.
    /// The last one is the carry of the next call.
    pub log_precision_bsh: Tensor<3>,
}

/// The masses of the step in log coordinates, each `[batch, len, nheads]` and
/// floored at [`LOG_ZERO`].
///
/// They come from the **pre-activations**, never as `ln` of the masses that
/// the plant multiplies by. A mass underflows to `0` (`softplus`/`σ` of a
/// large negative pre-activation: below ≈ −104 in f32, ≈ −17 in f16). There,
/// the forward of `ln` can be floored, but its backward cannot:
/// `∂ ln x/∂x = 1/x` meets the zero gradient of the floor as `0·∞ = NaN` (and
/// a denormal `x` overflows `1/x` first). `ln softplus` and `ln σ` of the
/// pre-activation have derivatives in `(0, 1]` everywhere.
pub struct LogMasses {
    /// `ln Δₜ`: the elapsed time, which paces the doubt `q`.
    pub dt_bsh: Tensor<3>,
    /// `ln γₜ`: the right endpoint, the own measurement of this position.
    pub gamma_bsh: Tensor<3>,
    /// `ln νₜ`: the left endpoint, the installment of an earlier sample, paid
    /// at this position. All taps together (a two-tap `μ` split sums back to
    /// `(1 − λ)·Δ`). `None` under
    /// [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None).
    pub nu_bsh: Option<Tensor<3>>,
}

impl LogMasses {
    /// The masses of `helpers::trapezoidal_coefficients`, in logs:
    /// `Δ = clamp(softplus(dt_raw), dt_limit)`, `γ = λ'·Δ`, `ν = (1 − λ')·Δ`,
    /// where `λ' = σ(lambda_raw)`, set to `1` where `far_open_1s1` closes the
    /// far tap.
    ///
    /// # Shapes
    /// - `dt_raw_bsh`, `lambda_raw_bsh` : `[batch, len, nheads]` (`dd_dt + dt_bias`; `λ`'s logit)
    /// - `far_open_1s1`                 : `[1, len, 1]`, `0` where the far tap is closed
    pub fn new(
        dt_raw_bsh: Tensor<3>,
        dt_limit: (f64, f64),
        lambda_raw_bsh: Option<Tensor<3>>,
        far_open_1s1: Option<Tensor<3>>,
    ) -> Self {
        // `ln` is monotone, so clamping `ln Δ` is clamping `Δ`.
        let ln_dt = log_softplus(dt_raw_bsh).clamp_max(dt_limit.1.ln());
        let ln_dt = if dt_limit.0 > 0.0 {
            ln_dt.clamp_min(dt_limit.0.ln())
        } else {
            ln_dt
        }
        .clamp_min(LOG_ZERO);
        let Some(lambda_raw_bsh) = lambda_raw_bsh else {
            return LogMasses {
                gamma_bsh: ln_dt.clone(),
                dt_bsh: ln_dt,
                nu_bsh: None,
            };
        };
        // `ln σ(x) = −softplus(−x)` and `ln(1 − σ(x)) = −softplus(x)`.
        let ln_lambda = -softplus(-lambda_raw_bsh.clone());
        let ln_one_minus = -softplus(lambda_raw_bsh);
        // A closed far tap is `λ' = 1`: `ln λ' = 0` and `ln(1 − λ') = ln 0`.
        let (ln_lambda, ln_one_minus) = match far_open_1s1 {
            Some(open_1s1) => (
                ln_lambda * open_1s1.clone(),
                ln_one_minus * open_1s1.clone() + (-open_1s1 + 1.0) * LOG_ZERO,
            ),
            None => (ln_lambda, ln_one_minus),
        };
        LogMasses {
            gamma_bsh: (ln_lambda + ln_dt.clone()).clamp_min(LOG_ZERO),
            nu_bsh: Some((ln_one_minus + ln_dt.clone()).clamp_min(LOG_ZERO)),
            dt_bsh: ln_dt,
        }
    }
}

/// `ln softplus(x)`, finite everywhere, with a derivative
/// `σ(x)/softplus(x)` in `(0, 1]`. This also holds where `softplus(x)` itself
/// underflows to `0`.
///
/// Below a knee, `softplus(x) = eˣ·(1 − eˣ/2 + …)`, so
/// `ln softplus(x) = x − eˣ/2 + O(e²ˣ)`. Above the knee, `softplus` stays a
/// normal number (the smallest f16 normal is `e^−9.7`). Below it, the dropped
/// `e²ˣ` term is under the resolution of the dtype. The knee is `−8` in f16
/// and `−20` otherwise.
pub fn log_softplus<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let knee = match x.dtype() {
        DType::F16 => -8.0,
        _ => -20.0,
    };
    let below = x.clone().lower_elem(knee);
    // Each branch gets only its own side of the knee. A branch that
    // `mask_where` discards still backpropagates, and a `ln 0` or an
    // overflowing `eˣ` there would reach the gradient as `0·∞ = NaN`.
    let above = softplus(x.clone().clamp_min(knee)).log();
    let x_below = x.clamp_max(knee);
    let series = x_below.clone() - x_below.exp() * 0.5;
    above.mask_where(below, series)
}

/// The per-position Möbius elements (shift by `ν`, predict, shift by `γ`):
///
/// ```text
///   ln [[α(1 + qγ),  αν + γ(1 + qαν)],
///       [αq,         1 + qαν       ]]
/// ```
///
/// which is `ln [[α(1+qγ), γ], [αq, 1]]` without a `ν`.
///
/// # Shapes
/// - every input : `[batch, len, nheads]` (`da_bsh` is the stock `ln α = Δ·A`)
fn elements(da_bsh: &Tensor<3>, ln_q_bsh: &Tensor<3>, masses: &LogMasses) -> Mobius {
    let floor = |t: Tensor<3>| t.clamp_min(LOG_ZERO);
    let ln_gamma = masses.gamma_bsh.clone();
    let m00 = floor(da_bsh.clone() + softplus(ln_q_bsh.clone() + ln_gamma.clone()));
    let m10 = floor(da_bsh.clone() + ln_q_bsh.clone());
    let (m01, m11) = match &masses.nu_bsh {
        None => (ln_gamma, m10.zeros_like()),
        Some(ln_nu) => {
            let m11 = softplus(floor(ln_q_bsh.clone() + da_bsh.clone() + ln_nu.clone()));
            let m01 = scan::lse(floor(da_bsh.clone() + ln_nu.clone()), ln_gamma + m11.clone());
            (m01, m11)
        }
    };
    Mobius { m00, m01, m10, m11 }
}

/// Run the gate over a folded run of positions (the whole sequence in
/// `forward`, the `u` micro-steps of one token in `step`).
///
/// # Shapes
/// - `da_bsh` : `[batch, len, nheads]`, the stock `Δ·A`
pub fn gate(da_bsh: Tensor<3>, masses: LogMasses, input: GainInput) -> GainOutput {
    let [batch, len, nheads] = da_bsh.dims();
    let ln_q = input.log_kappa_h.clone().unsqueeze::<3>() + masses.dt_bsh.clone();
    let ln_q = match &input.noise_bsh {
        Some(noise_bsh) => ln_q + noise_bsh.clone(),
        None => ln_q,
    };
    let ln_q = ln_q.clamp_min(LOG_ZERO);
    let elements = elements(&da_bsh, &ln_q, &masses);
    let log_precision_bsh = scan::prefix(elements).apply(input.carry_bh.clone());
    // ℓₜ₋₁: the carry, then every position but the last.
    let carry_b1h = input.carry_bh.reshape([batch, 1, nheads]);
    let prev_bsh = if len == 1 {
        carry_b1h
    } else {
        Tensor::cat(vec![carry_b1h, log_precision_bsh.clone().narrow(1, 0, len - 1)], 1)
    };
    // `ln Lₜ`: the predict acts on all that is known about the samples before
    // this position, including the installment of a sample that is paid only
    // now.
    let ln_l_bsh = match masses.nu_bsh {
        Some(ln_nu) => scan::lse(prev_bsh, ln_nu),
        None => prev_bsh,
    };
    let da_bsh = da_bsh.clone() - softplus(ln_q + da_bsh + ln_l_bsh);
    GainOutput {
        da_bsh,
        log_precision_bsh,
    }
}
