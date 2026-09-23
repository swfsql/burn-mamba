//! # Single-SSD pathway (official-kernel form)
//!
//! The Mamba-3 trapezoidal recurrence as a **single SSD call** (the official
//! Triton-SISO / Tilelang-MIMO form):
//!
//! - a key scale `scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)`, where the term in
//!   parentheses is the second installment of the two-tap members,
//! - a strict lower-triangular intra-chunk mask,
//! - a same-step γ correction,
//! - a boundary-β seed, folded into the initial state.
//!
//! It is one call for every tap pattern: every tap collapses into that one
//! scalar per sample. Under the lag-`u` members the correction widens to a
//! band, which [`token_band`] adds outside the kernel.
//!
//! It uses ≈ half the training memory of the two-call
//! [`double_ssd`](crate::mamba3::double_ssd) pathway, and less than a third of
//! a three-call one. The SSM accumulator `h'` of its cache has **different
//! mid-sequence semantics** from the double-SSD state (so the cache type is
//! distinct). The two are equal at call boundaries and convert with
//! field-identity `From` impls.

/// The single-SSD cache (same fields as double-SSD, different `ssm` semantics).
pub mod cache;
/// `forward_single_ssd` (scale + boundary-β seed) and `step_single_ssd`.
pub mod single_ssd;
/// The standard SSD kernels specialised to the single-pass scale/mask.
pub mod ssd;
/// The lag-`u` correction band, as one intra-token term outside the kernel.
pub mod token_band;

/// Public re-exports for the single-SSD pathway.
pub mod prelude {
    use super::*;
    pub use cache::{
        Mamba3SingleSsdCache, Mamba3SingleSsdCacheConfig, Mamba3SingleSsdCaches,
        Mamba3SingleSsdCachesConfig,
    };
    #[cfg(feature = "autodiff")]
    pub use ssd::Mamba3SingleSsdAutodiffBackendExt;
    pub use ssd::Mamba3SingleSsdBackendExt;
    pub use ssd::Mamba3SingleSsdInput;
}
