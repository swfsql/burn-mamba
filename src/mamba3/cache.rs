//! # Mamba-3 Cache and Pathway Selection
//!
//! [`Mamba3Cache`] / [`Mamba3Caches`] are **enums** that tag the SSD pathway
//! of a cache (`DoubleSsd` | `SingleSsd`). The variant given to
//! [`Mamba3::forward`](crate::mamba3::mamba3::Mamba3::forward) /
//! [`Mamba3::step`](crate::mamba3::mamba3::Mamba3::step) selects the pathway
//! at runtime. A missing cache selects `SingleSsd`.
//!
//! The SSM accumulators of the two pathways are different **mid-sequence**, so
//! the cache types are distinct: this prevents a silent mix inside a chunked
//! pass. They are equal at call boundaries, where caches are produced and
//! consumed. So the `From` impls at the bottom convert between them by a
//! lossless field-by-field move (see the note there).

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::single_ssd::prelude::*;

/// A pathway-tagged bundle of per-layer caches, so one dispatch entry can
/// take and return either cache family.
///
/// The variant selects the Double-SSD or Single-SSD pathway. Without caches,
/// the pathway is [`Self::SingleSsd`].
///
/// See also [`crate::mamba3::ssd_path::Mamba3SsdPath`].
#[derive(Debug, Clone)]
pub enum Mamba3Caches {
    /// Caches for the double-ssd pathway.
    DoubleSsd(Mamba3DoubleSsdCaches),
    /// Caches for the single-ssd pathway.
    SingleSsd(Mamba3SingleSsdCaches),
}

/// A pathway-tagged cache of one block, so one dispatch entry can take and
/// return either cache family.
///
/// The variant selects the Double-SSD or Single-SSD pathway. Without a cache,
/// the pathway is [`Self::SingleSsd`].
///
/// See also [`crate::mamba3::ssd_path::Mamba3SsdPath`].
#[derive(Debug, Clone)]
pub enum Mamba3Cache {
    /// Cache for the double-ssd pathway.
    DoubleSsd(Mamba3DoubleSsdCache),
    /// Cache for the single-ssd pathway.
    SingleSsd(Mamba3SingleSsdCache),
}

impl Mamba3Caches {
    /// Unwrap to the double-SSD caches, or `None` if this is the single-SSD variant.
    pub fn double_ssd(self) -> Option<Mamba3DoubleSsdCaches> {
        match self {
            Self::DoubleSsd(caches) => Some(caches),
            Self::SingleSsd(_caches) => None,
        }
    }

    /// Unwrap to the single-SSD caches, or `None` if this is the double-SSD variant.
    pub fn single_ssd(self) -> Option<Mamba3SingleSsdCaches> {
        match self {
            Self::DoubleSsd(_caches) => None,
            Self::SingleSsd(caches) => Some(caches),
        }
    }

    /// Number of per-layer caches (independent of pathway).
    pub fn caches_len(&self) -> usize {
        match self {
            Self::DoubleSsd(caches) => caches.caches.len(),
            Self::SingleSsd(caches) => caches.caches.len(),
        }
    }

    /// Collect per-layer caches into a pathway-tagged bundle. The first element
    /// gives the pathway (an empty vec gives single-SSD). Panics if the
    /// elements mix pathways.
    pub fn from_vec(vec: Vec<Mamba3Cache>) -> Self {
        // Look at the first element. Empty means single_ssd.
        let is_double = matches!(vec.first(), Some(Mamba3Cache::DoubleSsd(_)));
        if is_double {
            Mamba3DoubleSsdCaches {
                caches: vec
                    .into_iter()
                    .map(Mamba3Cache::double_ssd)
                    .map(Option::unwrap)
                    .collect(),
            }
            .into()
        } else {
            Mamba3SingleSsdCaches {
                caches: vec
                    .into_iter()
                    .map(Mamba3Cache::single_ssd)
                    .map(Option::unwrap)
                    .collect(),
            }
            .into()
        }
    }

    /// Wrap each per-layer cache in `Some`, so the layer loop can `take` it
    /// without a clone.
    pub fn into_options(self) -> Vec<Option<Mamba3Cache>> {
        match self {
            Self::DoubleSsd(caches) => caches
                .caches
                .into_iter()
                .map(Mamba3Cache::from)
                .map(Some)
                .collect(),
            Self::SingleSsd(caches) => caches
                .caches
                .into_iter()
                .map(Mamba3Cache::from)
                .map(Some)
                .collect(),
        }
    }

    /// Inverse of [`Self::into_options`]: unwrap each slot and re-bundle.
    pub fn from_options(options: Vec<Option<Mamba3Cache>>) -> Self {
        let caches = options.into_iter().map(Option::unwrap).collect();
        Self::from_vec(caches)
    }
}

impl Mamba3Cache {
    /// Unwrap to the double-SSD cache, or `None` if this is the single-SSD variant.
    pub fn double_ssd(self) -> Option<Mamba3DoubleSsdCache> {
        match self {
            Self::DoubleSsd(cache) => Some(cache),
            Self::SingleSsd(_cache) => None,
        }
    }

    /// Unwrap to the single-SSD cache, or `None` if this is the double-SSD variant.
    pub fn single_ssd(self) -> Option<Mamba3SingleSsdCache> {
        match self {
            Self::DoubleSsd(_cache) => None,
            Self::SingleSsd(cache) => Some(cache),
        }
    }
}

impl From<Mamba3DoubleSsdCaches> for Mamba3Caches {
    fn from(caches: Mamba3DoubleSsdCaches) -> Self {
        Mamba3Caches::DoubleSsd(caches)
    }
}

impl From<Mamba3SingleSsdCaches> for Mamba3Caches {
    fn from(caches: Mamba3SingleSsdCaches) -> Self {
        Mamba3Caches::SingleSsd(caches)
    }
}

impl From<Mamba3DoubleSsdCache> for Mamba3Cache {
    fn from(cache: Mamba3DoubleSsdCache) -> Self {
        Mamba3Cache::DoubleSsd(cache)
    }
}

impl From<Mamba3SingleSsdCache> for Mamba3Cache {
    fn from(cache: Mamba3SingleSsdCache) -> Self {
        Mamba3Cache::SingleSsd(cache)
    }
}

// ---------------------------------------------------------------------------
// Conversions between the two pathway caches
// ---------------------------------------------------------------------------
//
// At a cache boundary, the look-ahead term `νₜ₊ₗₐ₉` vanishes, so `scaleₜ = γₜ`
// for the final `lag` positions. The *next* call pays their second
// installment, from the tap slots. With this substitution, the single-ssd
// accumulator `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ⊗xₜ` is *exactly* the double-ssd state
// `hₜ = αₜ hₜ₋₁ + Σ_taps βₜ Bₜ₋ₗ⊗xₜ₋ₗ + γₜ Bₜ⊗xₜ`. The next call rebuilds the
// deferred β contribution from the saved `k_state`/`v_state`, identically in
// both forms.
//
// The other fields have the same meaning in both caches: the K/V slots of the
// tap FIFO (the last `lag` positions), the cumulative rotation, and the `ln Λ`
// and `c` of the positive systems (which read only the inputs). So the
// conversion is a field-by-field move.
//
// The accumulators differ only *mid-sequence*. Caches exist only at
// boundaries, so the move is lossless. The distinct types still prevent a
// silent mix of the two accumulators inside one chunked pass.

impl From<Mamba3SingleSsdCache> for Mamba3DoubleSsdCache {
    fn from(cache: Mamba3SingleSsdCache) -> Self {
        Mamba3DoubleSsdCache {
            ssm_bhpr: cache.ssm_bhpr,
            k_state_bumhr: cache.k_state_bumhr,
            v_state_buhp: cache.v_state_buhp,
            rotation: cache.rotation,
            log_precision_bh: cache.log_precision_bh,
            tropical_bh: cache.tropical_bh,
        }
    }
}

impl From<Mamba3DoubleSsdCache> for Mamba3SingleSsdCache {
    fn from(cache: Mamba3DoubleSsdCache) -> Self {
        Mamba3SingleSsdCache {
            ssm_bhpr: cache.ssm_bhpr,
            k_state_bumhr: cache.k_state_bumhr,
            v_state_buhp: cache.v_state_buhp,
            rotation: cache.rotation,
            log_precision_bh: cache.log_precision_bh,
            tropical_bh: cache.tropical_bh,
        }
    }
}

impl From<Mamba3SingleSsdCaches> for Mamba3DoubleSsdCaches {
    fn from(caches: Mamba3SingleSsdCaches) -> Self {
        Mamba3DoubleSsdCaches {
            caches: caches.caches.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<Mamba3DoubleSsdCaches> for Mamba3SingleSsdCaches {
    fn from(caches: Mamba3DoubleSsdCaches) -> Self {
        Mamba3SingleSsdCaches {
            caches: caches.caches.into_iter().map(Into::into).collect(),
        }
    }
}
