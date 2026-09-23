//! # Positive systems that set the gate
//!
//! A Mamba-3 head is a linear **parameter-varying** plant. Its transition,
//! write and read are linear in the state, with coefficients `ρₜ = g(uₜ)` that
//! the input sets token by token. This module adds a second, smaller system
//! next to it (one scalar per head). Its output sets some of those
//! coefficients:
//!
//! ```text
//!   σₜ = Mₜ ⊗ σₜ₋₁          Mₜ from the in-projection alone
//!   ρₜ = g(uₜ, σₜ₋₁)        read by the plant, never written by it
//! ```
//!
//! The one-way path is the design rule. `Mₜ` reads the token and never the
//! plant, so the cascade still scans: the chunkwise SSD of the plant takes the
//! scheduled coefficients as ordinary tensors, and `σ` has its own associative
//! scan ([`scan`]). A gain that read the state of the plant would close the
//! loop and lose both.
//!
//! ## The members are positive linear systems, in log coordinates
//!
//! Every `Mₜ` is a **nonnegative** 2×2 matrix that acts projectively on a
//! nonnegative 2-vector. So `σ` is a log-vector, and the product is that of
//! the log-semiring, `(X ⊗ Y)ᵢⱼ = log Σₖ exp(Xᵢₖ + Yₖⱼ)` ([`scan::lse`]).
//! Ordinary arithmetic on positives *is* that semiring in log coordinates, so
//! one kernel carries both members:
//!
//! | member | `Mₜ` | carries | port |
//! |---|---|---|---|
//! | [`Gain::Kalman`] | shift by `ν`, predict, shift by `γ` (projective, [`scan::Mobius`]) | `ℓ = ln Λ`, the precision of the head | `A` and `C` |
//! | [`Tropical::MaxPlus`] | `[[a, b], [−∞, 0]]` (affine, [`scan::Affine`]) | `c`, a soft max-plus register | `D` |
//!
//! Log coordinates are necessary, not a convenience. The natural entries of
//! the Kalman member include `1/α`, which overflows at a reset token. The
//! growth of the tropical member (`a > 0`) overflows in any linear form.
//!
//! ### The Kalman gate ([`kalman`])
//!
//! The Mamba update is an exponential moving average of a rank-one setpoint,
//! with a *projected* gain `1 − α`. A Kalman filter has the same form, with
//! the gain *computed* from a variance that accumulates evidence. In
//! information form (`Λ = 1/P`, `η = Λ·S`), the per-head scalar filter is an
//! SSD with a computed decay:
//!
//! ```text
//!   dₜ = αₜ / (1 + qₜ·αₜ·(Λₜ₋₁ + νₜ))
//!   ηₜ = dₜ·ηₜ₋₁ + (the block's own write)        ← the plant, unchanged
//!   Λₜ = dₜ·(Λₜ₋₁ + νₜ) + γₜ                       ← the same recurrence on ones
//! ```
//!
//! The masses of the trapezoid are the evidence. `γₜ` is the right endpoint.
//! `νₜ` is the installment of the left endpoint, which the plant pays a step
//! late and transports by the decay of that step (exact for a lag-1 tap, an
//! upper bound at lag `u`, see [`kalman`]). Without a tap, `γ = Δ` and
//! `ν = 0`. The stock `αₜ` is in the covariance (`a² = 1/α`), and
//! `qₜ = κₕ·Δₜ` is Brownian: doubt grows with elapsed time.
//! [`Gain::KalmanProjectedNoise`] multiplies `q` by a projected `exp(rₜ)`, so a
//! token can add doubt without evidence (a gap).
//!
//! Only the decay changes. `dₜ` replaces `αₜ` where `αₜ` is formed, before any
//! consumer reads it. So the transports of the trapezoid, the tap slots and
//! the single-SSD key scale all follow. At `κ = 0` the block is stock, bit for
//! bit (`log1p(0) = 0`). Two facts carry the design:
//!
//! - **a ceiling:** `Λₜ < 1/qₜ + γₜ` for every history. So strong evidence
//!   dominates for a bounded time, not for a time that grows with its
//!   strength.
//! - **a contraction:** for `q, γ > 0` the matrix is entrywise positive. By
//!   Birkhoff's theorem, it contracts the Hilbert distance `|ln Λ − ln Λ'|` by
//!   at least `tanh(¼·ln(1 + 1/(q·γ)))` per step, with no `α` in the bound. So
//!   the block forgets the initial `Λ` of the cache, and any rounding in it,
//!   geometrically.
//!
//! The block also reads through `(Λₜ + ε)^(−ωₕ)` (`ω = 0` at init). `ω = 1`
//! reads the estimate `S = η/Λ` instead of the information `η`: a mean, not a
//! sum. The scale applies to the SSD readout *before* the `D` skip and the
//! `c·e` of the register. So a per-head output norm does not remove it: the
//! norm removes only what the three terms share, and `ω` sets the share of the
//! SSD against the other two. It cancels only when the other two vanish, and
//! then only up to the `ε` of the norm.
//!
//! ### The tropical register ([`tropical`])
//!
//! The affine member, with a log-decay that can be positive:
//!
//! ```text
//!   cₜ = log(exp(cₜ₋₁ + aₜ) + exp(bₜ))   →   max(cₜ₋₁ + aₜ, bₜ)   as the scale grows
//! ```
//!
//! It is linear in the (max, +) semiring, so exactly scannable. Yet it
//! computes what no linear recurrence does:
//!
//! - `(x, 0)` is the Lindley recursion (a counter clamped at zero),
//! - `(0, v)` is a running maximum,
//! - `(−∞, b)` is a reset to `b`.
//!
//! The soft value overshoots the hard one by at most `ln(T + 1)` in projection
//! units. So the temperature is the scale of the in-projection, not a knob.
//! It adds to the readout, `yₜ,ₕ += cₜ,ₕ·eₕ`, with a zero-initialised
//! `eₕ ∈ ℝᵖ`.
//!
//! ## Resolution
//!
//! Both run on the folded axis, like the recurrence (one matrix per micro-step
//! under [`crate::mamba3::product`]). The `C` and `D` ports read the value at
//! the last micro-step of each token (the read axis). The mimo ranks share one
//! state, thus one transition, thus one `Λ` per head, broadcast over them.
//! (`info/mamba-3/mimo-as-batch.md` §7 makes the same argument for the
//! rotation.)

/// The log-semiring scan that both members run on.
pub mod scan;

/// The Kalman gate: the precision recurrence and the decay it computes.
pub mod kalman;

/// The tropical register.
pub mod tropical;

use burn::prelude::*;

/// `ln 0`, kept finite.
///
/// A log-sum-exp of two `−∞` is `NaN`, and so is `0·(−∞)` in a
/// zero-initialised readout. So structural zeros use this value. It is far
/// enough below every real entry to vanish under `exp`, and `2·LOG_ZERO` still
/// fits in f16, which every combine clamps to again.
pub const LOG_ZERO: f32 = -3.0e4;

/// How the decay of a head is formed: projected from the token, or computed by
/// a per-head Kalman filter on top of that projection. See the module header.
///
/// Structural, like [`Trapezoid`](crate::mamba3::trapezoid::Trapezoid). A
/// Kalman member allocates the per-head `κ` and `ω` and a cache slot for
/// `ln Λ`. Under [`KalmanProjectedNoise`](Self::KalmanProjectedNoise) it also
/// adds one in-projection channel per (head, micro-step).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Gain {
    /// Stock Mamba-3: `αₜ = exp(Δₜ·Aₜ)`, projected.
    #[default]
    Projected,
    /// The tied arm: `qₜ = κₕ·Δₜ`, with the masses of the trapezoid as the
    /// evidence. No in-projection channel. Stock, bit for bit, at `κ = 0`.
    Kalman,
    /// `qₜ = κₕ·Δₜ·exp(rₜ)`, with `rₜ` projected per (head, micro-step). A
    /// token can add doubt without evidence. The tied arm cannot, because its
    /// doubt and its evidence both scale with `Δ`. `r ≡ 0` is
    /// [`Kalman`](Self::Kalman).
    KalmanProjectedNoise,
}

impl Gain {
    /// Whether the decay is computed (a `ln Λ` slot, a `κ` per head).
    pub fn is_kalman(self) -> bool {
        !matches!(self, Gain::Projected)
    }

    /// Whether the in-projection has the noise channel `r`.
    pub fn projects_noise(self) -> bool {
        matches!(self, Gain::KalmanProjectedNoise)
    }
}

/// Whether each head has a tropical register that adds to its readout. See the
/// module header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Tropical {
    /// No register.
    #[default]
    None,
    /// One soft max-plus register per head: two in-projection channels
    /// `(a, b)` per (head, micro-step), a cache slot, and the readout `eₕ`.
    MaxPlus,
}

impl Tropical {
    /// Whether the block has the register.
    pub fn is_on(self) -> bool {
        matches!(self, Tropical::MaxPlus)
    }
}

/// The fresh-sequence slots `(ln Λ, c)`, each `[batch, nheads]` at
/// [`LOG_ZERO`] (no evidence yet, the max of nothing). `None` for a system that
/// the block does not have. Every cache constructor uses this.
pub fn fresh_slots(
    gain: Gain,
    tropical: Tropical,
    batch: usize,
    nheads: usize,
    device: &Device,
) -> (Option<Tensor<2>>, Option<Tensor<2>>) {
    let slot = || Tensor::full([batch, nheads], LOG_ZERO, device);
    (gain.is_kalman().then(slot), tropical.is_on().then(slot))
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
