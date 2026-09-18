//! # Positive systems that set the gate
//!
//! A Mamba-3 head is a linear **parameter-varying** plant: its transition,
//! write and read are linear in the state, with coefficients `ρₜ = g(uₜ)` set
//! token by token from the input. This module adds a second, smaller system
//! next to it — one scalar per head — whose output sets some of those
//! coefficients instead:
//!
//! ```text
//!   σₜ = Mₜ ⊗ σₜ₋₁          Mₜ from the in-projection alone
//!   ρₜ = g(uₜ, σₜ₋₁)        read by the plant, never written by it
//! ```
//!
//! The one-way path is the whole design rule: `Mₜ` reads the token and never
//! the plant, so the cascade still scans — the plant's chunkwise SSD takes the
//! scheduled coefficients as ordinary tensors, and `σ` has an associative scan
//! of its own ([`scan`]). A gain that read the plant's state would close the
//! loop and lose both.
//!
//! ## The members are positive linear systems, in log coordinates
//!
//! Every `Mₜ` is a **nonnegative** 2×2 matrix acting projectively on a
//! nonnegative 2-vector, so `σ` is carried as a log-vector and the product is
//! the log-semiring one, `(X ⊗ Y)ᵢⱼ = log Σₖ exp(Xᵢₖ + Yₖⱼ)` ([`scan::lse`]).
//! Ordinary arithmetic on positives *is* that semiring in log coordinates, so
//! one kernel carries both members:
//!
//! | member | `Mₜ` | carries | port |
//! |---|---|---|---|
//! | [`Gain::Kalman`] | `ln [[α(1+qm), m], [αq, 1]]` (projective, [`scan::Mobius`]) | `ℓ = ln Λ`, the head's precision | `A`, and `C` without the out-norm |
//! | [`Tropical::MaxPlus`] | `[[a, b], [−∞, 0]]` (affine, [`scan::Affine`]) | `c`, a soft max-plus register | `D` |
//!
//! Log coordinates are forced, not convenient: the Kalman member's natural
//! entries include `1/α`, which overflows at a reset token, and the tropical
//! one's growth (`a > 0`) overflows in any linear form.
//!
//! ### The Kalman gate ([`kalman`])
//!
//! Mamba's update is an exponential moving average of a rank-one setpoint whose
//! gain `1 − α` is *projected*. A Kalman filter has the same form with the gain
//! *computed* from a variance that accumulates evidence. In information form
//! (`Λ = 1/P`, `η = Λ·S`) the per-head scalar filter is an SSD whose decay is
//! computed:
//!
//! ```text
//!   dₜ = αₜ / (1 + qₜ·αₜ·Λₜ₋₁)
//!   ηₜ = dₜ·ηₜ₋₁ + (the block's own write)        ← the plant, unchanged
//!   Λₜ = dₜ·Λₜ₋₁ + mₜ                              ← the same recurrence on ones
//! ```
//!
//! with `mₜ = Δₜ` (the step's whole mass, whatever the trapezoid splits it
//! into), stock's `αₜ` placed in the covariance (`a² = 1/α`), and `qₜ = κₕ·Δₜ`
//! — Brownian: doubt grows with elapsed time. [`Gain::KalmanProjectedNoise`]
//! multiplies `q` by a projected `exp(rₜ)`, so a token can add doubt without
//! evidence (a gap). Only the decay changes: `dₜ` replaces `αₜ` where it is
//! formed, before any consumer reads it, so the trapezoid's transports, the tap
//! slots and single-SSD's key scale all follow. At `κ = 0` the block is stock
//! bit for bit (`log1p(0) = 0`). Two facts carry the design:
//!
//! - **a ceiling:** `Λₜ < 1/qₜ + mₜ` whatever the history, so strong evidence
//!   dominates for a bounded time rather than one growing with its strength;
//! - **a contraction:** for `q, m > 0` the matrix is entrywise positive, so by
//!   Birkhoff's theorem it contracts the Hilbert distance `|ln Λ − ln Λ'|` by at
//!   least `tanh(¼·ln(1 + 1/(q·m)))` per step — the cache's initial `Λ`, and any
//!   rounding in it, is forgotten geometrically.
//!
//! Without the output norm the block may also read through `(Λₜ + ε)^(−ωₕ)`
//! (`ω = 0` at init): `ω = 1` reads the estimate `S = η/Λ` instead of the
//! information `η` — a mean rather than a sum. With the norm, that per-(token,
//! head) scale is removed anyway, so the parameter does not exist.
//!
//! ### The tropical register ([`tropical`])
//!
//! The affine member with its log-decay allowed to be positive:
//!
//! ```text
//!   cₜ = log(exp(cₜ₋₁ + aₜ) + exp(bₜ))   →   max(cₜ₋₁ + aₜ, bₜ)   as the scale grows
//! ```
//!
//! Linear in the (max, +) semiring, so exactly scannable, yet it computes what
//! no linear recurrence does: `(x, 0)` is the Lindley recursion (a counter
//! clamped at zero), `(0, v)` a running maximum, `(−∞, b)` a reset to `b`. The
//! soft value overshoots the hard one by at most `ln(T + 1)` in projection
//! units, so the temperature is the in-projection's own scale and not a knob.
//! It feeds the readout, `yₜ,ₕ += cₜ,ₕ·eₕ` with a zero-initialised `eₕ ∈ ℝᵖ`.
//!
//! ## Resolution
//!
//! Both run on the folded axis, like the recurrence (one matrix per micro-step
//! under [`crate::mamba3::product`]); what the `C` and `D` ports read is the
//! value at each token's last micro-step, the read axis. The mimo ranks share
//! one state, hence one transition, hence one `Λ` per head broadcast over them
//! (`info/mamba-3/mimo-as-batch.md` §7 makes the same argument for the rotation).

/// The log-semiring scan both members run on.
pub mod scan;

/// The Kalman gate: the precision recurrence and the decay it computes.
pub mod kalman;

/// The tropical register.
pub mod tropical;

/// `ln 0`, kept finite.
///
/// A log-sum-exp of two `−∞` is `NaN`, and so is `0·(−∞)` in a zero-initialised
/// readout, so structural zeros are this instead. It is far enough below every
/// real entry to vanish under `exp`, and `2·LOG_ZERO` still fits in f16, which
/// every combine re-clamps to.
pub const LOG_ZERO: f32 = -3.0e4;

/// How a head's decay is formed — projected from the token, or computed by a
/// per-head Kalman filter on top of that projection. See the module header.
///
/// Structural, like [`Trapezoid`](crate::mamba3::trapezoid::Trapezoid): a
/// Kalman member allocates the per-head `κ` (and, without the out-norm, `ω`),
/// a cache slot for `ln Λ`, and under
/// [`KalmanProjectedNoise`](Self::KalmanProjectedNoise) one in-projection
/// channel per (head, micro-step).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Gain {
    /// Stock Mamba-3: `αₜ = exp(Δₜ·Aₜ)`, projected.
    #[default]
    Projected,
    /// The tied arm: `qₜ = κₕ·Δₜ`, `mₜ = Δₜ`. No in-projection channel; stock
    /// bit for bit at `κ = 0`.
    Kalman,
    /// `qₜ = κₕ·Δₜ·exp(rₜ)` with `rₜ` projected per (head, micro-step): a token
    /// may add doubt without adding evidence, which the tied arm cannot (its
    /// `q` and `m` both scale with `Δ`). `r ≡ 0` is [`Kalman`](Self::Kalman).
    KalmanProjectedNoise,
}

impl Gain {
    /// Whether the decay is computed (a `ln Λ` slot, a `κ` per head).
    pub fn is_kalman(self) -> bool {
        !matches!(self, Gain::Projected)
    }

    /// Whether the in-projection carries the noise channel `r`.
    pub fn projects_noise(self) -> bool {
        matches!(self, Gain::KalmanProjectedNoise)
    }
}

/// Whether each head carries a tropical register feeding its readout. See the
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
    /// Whether the block carries the register.
    pub fn is_on(self) -> bool {
        matches!(self, Tropical::MaxPlus)
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
