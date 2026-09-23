//! # Double-SSD pathway (VikramLex-style)
//!
//! The Mamba-3 trapezoidal recurrence as **one standard SSD call per term**,
//! with Mamba-2-like kernels:
//!
//! - γ-SSM: `hᵞₜ = αₜ hᵞₜ₋₁ + γₜ Bₜ xₜ`   (the current sample)
//! - β-SSM: `hᵝₜ = αₜ hᵝₜ₋₁ + βₜ Bₜ₋ₗ xₜ₋ₗ`, one per `β` tap, at the own lag
//!   `l` = [`Trapezoid::tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag)
//!   of that tap, transported across its own gap
//! - `hₜ = hᵞₜ + Σ hᵝₜ`.
//!
//! So there are **two** calls at the default, one under
//! [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None) (no tap to
//! shift), and three under a two-tap pattern (its lags are different, so they
//! cannot share a pass). It is simple and easy to verify. The cost is ~2× the
//! intra-chunk and chunk-state memory of the
//! [`single_ssd`](crate::mamba3::single_ssd) pathway, which stays at one call
//! for any number of taps and suits a two-tap pattern better.
//!
//! `step` always uses this recurrence, for both cache variants.

/// The double-SSD cache (SSM state, tap FIFO, rotation, positive-system slots;
/// no conv cache).
pub mod cache;
/// `forward_double_ssd` / `step_double_ssd` and their step helpers.
pub mod double_ssd;
/// The standard SSD kernels reused by both the γ and β passes.
pub mod ssd;

/// Public re-exports for the double-SSD pathway.
pub mod prelude {
    use super::*;
    pub use cache::{
        Mamba3DoubleSsdCache, Mamba3DoubleSsdCacheConfig, Mamba3DoubleSsdCaches,
        Mamba3DoubleSsdCachesConfig,
    };
    #[cfg(feature = "autodiff")]
    pub use ssd::Mamba3DoubleSsdAutodiffBackendExt;
    pub use ssd::Mamba3DoubleSsdBackendExt;
    pub use ssd::Mamba3DoubleSsdInput;
}
