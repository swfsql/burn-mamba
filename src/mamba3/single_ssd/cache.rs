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
/// [`Mamba3DoubleSsdCache`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache).
/// The semantic difference is only in `ssm_bhpr` (see the module header).
#[derive(Module, Debug)]
pub struct Mamba3SingleSsdCache {
    /// **Accumulator** `h'ₜ`.
    ///
    /// Update rule: `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ · sumₘ Bₜ[m] ⊗ (xₜ ⊙ mimo_xₘ)`.
    /// Mid-sequence, it is different from the double-ssd `ssm_bhpr`.
    ///
    /// Shape: `[batch, nheads, per_head_dim, state_rank]`
    pub ssm_bhpr: Tensor<4>,

    /// **The tap FIFO** ([`TapFifo`]), the same value as in the double-SSD
    /// cache.
    ///
    /// At the start of the next `forward_single_ssd` call, it seeds the
    /// deferred boundary β contributions
    /// `Σⱼ (1 − λⱼ) · Δⱼ · Bₚ₋ₗₐ₉₊ⱼ ⊗ xₚ₋ₗₐ₉₊ⱼ`. The previous call could not add
    /// them, because it did not know the `λ, Δ` of the `lag` positions that pay
    /// them.
    ///
    /// `None` under [`TapPattern::None`], which has no boundary β term to defer.
    pub tap_fifo: Option<TapFifo>,

    /// **Cumulative data-dependent rotation** up to the current position
    /// ([`RotationState`]).
    ///
    /// The same value as the field of the double-ssd cache (the `From` impls
    /// move it across), so the two caches convert by field identity.
    pub rotation: RotationState,

    /// **The log-precision of the Kalman gain**, `ln Λ` after the last position
    /// (the same value as
    /// [`Mamba3DoubleSsdCache::log_precision_bh`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache::log_precision_bh)).
    ///
    /// Shape: `[batch, nheads]`
    pub log_precision_bh: Option<Tensor<2>>,

    /// **The tropical register** `c` after the last position (see
    /// [`Mamba3DoubleSsdCache::tropical_bh`](crate::mamba3::double_ssd::cache::Mamba3DoubleSsdCache::tropical_bh)).
    ///
    /// Shape: `[batch, nheads]`
    pub tropical_bh: Option<Tensor<2>>,
}

impl Mamba3SingleSsdCache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every cached tensor.
    pub fn sanity(&self) {
        san(&self.ssm_bhpr);
        if let Some(tap_fifo) = &self.tap_fifo {
            tap_fifo.sanity();
        }
        self.rotation.sanity();
        for field in [&self.log_precision_bh, &self.tropical_bh].into_iter().flatten() {
            san(field);
        }
    }
}
