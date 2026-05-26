//! # Double-SSD pathway (VikramLex-style)
//!
//! Realises the Mamba-3 trapezoidal recurrence as **two standard SSD calls**
//! that reuse the Mamba-2-like kernels:
//!
//! - γ-SSM: `hᵞₜ = αₜ hᵞₜ₋₁ + γₜ Bₜ xₜ`   (current token)
//! - β-SSM: `hᵝₜ = αₜ hᵝₜ₋₁ + βₜ Bₜ₋₁ xₜ₋₁` (previous token)
//! - `hₜ = hᵞₜ + hᵝₜ`.
//!
//! Simple and easy to verify, at the cost of ~2× the intra-chunk and
//! chunk-state memory of the [`single_ssd`](crate::mamba3::single_ssd) pathway.

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
