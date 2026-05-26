//! # Single-SSD pathway (official-kernel form)
//!
//! Realises the Mamba-3 trapezoidal recurrence as a **single SSD call** (the
//! official Triton-SISO / Tilelang-MIMO form): a key scale
//! `scaleₜ = γₜ + (1−λₜ₊₁)·Δₜ₊₁`, a strict lower-triangular intra-chunk mask, a
//! same-step γ correction, and a boundary-β seed folded into the initial state.
//!
//! Uses ≈ half the training memory of the
//! [`double_ssd`](crate::mamba3::double_ssd) pathway.  Its cache's SSM
//! accumulator `h'` has **different mid-sequence semantics** than the double-SSD
//! state (hence a distinct cache type), but the two coincide at sequence
//! boundaries and inter-convert via field-identity `From` impls.

/// The single-SSD cache (same fields as double-SSD, different `ssm` semantics).
pub mod cache;
/// `forward_single_ssd` (scale + boundary-β seed) and `step_single_ssd`.
pub mod single_ssd;
/// The standard SSD kernels specialised to the single-pass scale/mask.
pub mod ssd;

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
