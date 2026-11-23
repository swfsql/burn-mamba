//! # Double-SSD pathway (VikramLex-style)
//!
//! The Mamba-3 trapezoidal recurrence as **one standard SSD call per term**,
//! with Mamba-2-like kernels:
//!
//! - γ-SSM: `hᵞₜ = αₜ hᵞₜ₋₁ + γₜ Bₜ xₜ`   (the current sample)
//! - β-SSM: `hᵝₜ = αₜ hᵝₜ₋₁ + βₜ Bₜ₋ₗ xₜ₋ₗ`, one per `β` tap, at the own lag
//!   `l` = [`TapPattern::tap_lag`](crate::mamba3::tap_pattern::TapPattern::tap_lag)
//!   of that tap, transported across its own gap
//! - `hₜ = hᵞₜ + Σ hᵝₜ`.
//!
//! So the default makes **two** calls.
//! [`TapPattern::None`](crate::mamba3::tap_pattern::TapPattern::None) makes one
//! call (no tap to shift). A two-tap pattern makes three calls: its lags are
//! different, so they cannot share a pass. This pathway is simple and easy to
//! verify. The cost is ~2× the intra-chunk and chunk-state memory of the
//! [`single_ssd`](crate::mamba3::single_ssd) pathway. That pathway stays at one
//! call for any number of taps, and it suits a two-tap pattern better.
//!
//! [`Pathway::DoubleSsd`](crate::mamba3::scan_options::Pathway::DoubleSsd)
//! in the scan options selects this pathway. `step` has no pathway, and it
//! always uses this recurrence.

/// The forward of this pathway, `step_double_ssd` and its step helpers.
pub mod double_ssd;
/// The standard SSD kernels reused by both the γ and β passes.
pub mod ssd;

/// Public re-exports for the double-SSD pathway.
pub mod prelude {
    use super::*;
    pub use ssd::Mamba3DoubleSsdBackendExt;
    pub use ssd::Mamba3DoubleSsdInput;
}
