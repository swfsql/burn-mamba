//! # Single-SSD pathway (official-kernel form)
//!
//! The Mamba-3 trapezoidal recurrence as a **single SSD call** (the official
//! Triton-SISO / Tilelang-MIMO form):
//!
//! - a key scale `scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)`, where the term in
//!   parentheses is the second installment of the two-tap members,
//! - a strict lower-triangular intra-chunk mask,
//! - a diagonal γ-correction,
//! - a boundary-β seed, folded into the initial accumulator.
//!
//! It is one call for every tap pattern: every tap collapses into that one
//! scalar per sample. Under the lag-`u` members the correction widens to a
//! band, which [`token_band`] adds outside the kernel.
//!
//! It uses ≈ half the training memory of the two-call
//! [`double_ssd`](crate::mamba3::double_ssd) pathway, and less than a third of
//! a three-call one. In the middle of a sequence, it carries the accumulator
//! `h'`, which is different from the state. At a call boundary, the two are
//! equal, so both pathways share one cache type
//! ([`Mamba3Cache`](crate::mamba3::cache::Mamba3Cache)).
//!
//! [`Pathway::SingleSsd`](crate::mamba3::scan_options::Pathway::SingleSsd),
//! the default, in the scan options selects this pathway.

/// The forward of this pathway: the scale and the boundary-β seed.
pub mod single_ssd;
/// The standard SSD kernels specialised to the single-pass scale/mask.
pub mod ssd;
/// The lag-`u` correction band, as one intra-token term outside the kernel.
pub mod token_band;

/// Public re-exports for the single-SSD pathway.
pub mod prelude {
    use super::*;
    pub use ssd::Mamba3SingleSsdBackendExt;
    pub use ssd::Mamba3SingleSsdInput;
}
