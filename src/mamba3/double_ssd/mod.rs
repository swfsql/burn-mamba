//! # Double-SSD pathway (VikramLex-style)
//!
//! Realises the Mamba-3 trapezoidal recurrence as **one standard SSD call per
//! term**, reusing the Mamba-2-like kernels:
//!
//! - γ-SSM: `hᵞₜ = αₜ hᵞₜ₋₁ + γₜ Bₜ xₜ`   (the current sample)
//! - β-SSM: `hᵝₜ = αₜ hᵝₜ₋₁ + βₜ Bₜ₋ₗ xₜ₋ₗ`, one per `β` tap, at that tap's own
//!   lag `l` = [`Trapezoid::tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag)
//!   and transported across its own gap
//! - `hₜ = hᵞₜ + Σ hᵝₜ`.
//!
//! So **two** calls at the default, one under
//! [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None) (no tap to
//! shift) and three under a two-tap pattern, whose lags differ and so cannot
//! share a pass. Simple and easy to verify, at the cost of ~2× the intra-chunk
//! and chunk-state memory of the [`single_ssd`](crate::mamba3::single_ssd)
//! pathway — which stays at one call however many taps there are, and is where a
//! two-tap pattern belongs.

/// The double-SSD cache (`ssm`/`k_state`/`v_state`/`cum_angle`; no conv cache).
pub mod cache;
/// `forward_double_ssd` / `step_double_ssd` and the RoPE helpers.
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
