//! # Mamba-3 Caches (single-SSD pathway)
//!
//! The cache of [`crate::mamba3::mamba3::Mamba3::forward_single_ssd`] (the
//! single-pass SSD pathway of the Triton SISO and Tilelang MIMO reference
//! kernels). Its fields mirror those of
//! [`Mamba3DoubleSsdCache`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache),
//! but `ssm_bhpr` holds the **accumulator**, which has different semantics
//! mid-sequence:
//!
//! - double-SSD: `ssm_bhpr` holds the trapezoidal state
//!   `hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ` (at the default lag 1).
//! - single-SSD: `ssm_bhpr` holds the **accumulator**
//!   `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ ⊗ xₜ`, with
//!   `scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)`. With it, `yₜ = Cₜᵀ h'ₜ` is correct at
//!   all positions except the `lag`-wide band before a tap is paid. The kernel
//!   patches the diagonal (lag 1) with an explicit `γₜ · (Cₜᵀ Bₜ) · xₜ` term,
//!   and `token_band` patches the wider band (lag `u`).
//!
//! The accumulator and the state are equal at call boundaries, where caches are
//! produced and consumed (see [`crate::mamba3::cache`]). Under
//! [`TapPattern::None`](crate::mamba3::tap_pattern::TapPattern::None) the difference
//! disappears everywhere: `scaleₜ = γₜ`, so `h' ≡ h` at every position, and
//! neither cache has a tap FIFO.
//!
//! The distinct type prevents a `forward_double_ssd` cache from going into
//! `forward_single_ssd` (or the reverse) mid-sequence, which would silently
//! corrupt the state.

use crate::mamba3::prelude::*;
use burn_stack::modules::sanity as san;
use burn::module::Module;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba3SingleSsdCaches  (one cache entry per layer)
// ---------------------------------------------------------------------------

/// A collection of per-layer single-ssd caches for a complete Mamba-3 network.
#[derive(Module, Debug)]
pub struct Mamba3SingleSsdCaches {
    /// Per-layer caches. The length is the number of virtual layers.
    pub caches: Vec<Mamba3SingleSsdCache>,
}

// ---------------------------------------------------------------------------
// Mamba3SingleSsdCache  (cache for a single block)
// ---------------------------------------------------------------------------

/// The cache of one Mamba-3 block that runs the single-ssd pathway.
///
/// The fields and shapes match
/// [`Mamba3DoubleSsdCache`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache):
/// the `ssm_bhpr` of the pathway and the frame cache. The semantic difference
/// is only in `ssm_bhpr` (see the module header).
#[derive(Module, Debug)]
pub struct Mamba3SingleSsdCache {
    /// **Accumulator** `h'ₜ`.
    ///
    /// Update rule: `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ · sumₘ Bₜ[m] ⊗ (xₜ ⊙ mimo_xₘ)`.
    /// Mid-sequence, it is different from the double-ssd `ssm_bhpr`.
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<4>,

    /// **The frame cache** ([`Mamba3FrameCache`]), the same value as in the
    /// double-SSD cache. The `From` impls move it across.
    pub frame: Mamba3FrameCache,
}

impl Mamba3SingleSsdCache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        self.frame.sanity();
    }
}
