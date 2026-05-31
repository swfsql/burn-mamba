//! # Mamba-3 Cache and Pathway Selection
//!
//! [`Mamba3Cache`] / [`Mamba3Caches`] are **enums** tagging which SSD pathway a
//! cache belongs to (`DoubleSsd` | `SingleSsd`).  Supplying one of these to
//! [`Mamba3::forward`](crate::mamba3::mamba3::Mamba3::forward) /
//! [`Mamba3::step`](crate::mamba3::mamba3::Mamba3::step) is what selects the
//! pathway at runtime; a missing cache defaults to `SingleSsd`.
//!
//! The two pathways' SSM accumulators differ **mid-sequence**, so the cache
//! types are kept distinct to prevent silently mixing them inside a chunked
//! pass.  They coincide at sequence boundaries, however — where caches are
//! actually produced and consumed — so the `From` impls at the bottom convert
//! between them by a lossless field-by-field move (see the note there).

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::single_ssd::prelude::*;

/// A pathway-tagged bundle of per-layer caches, so a single dispatch entry can
/// accept / return either cache family.
///
/// The caches selection infers whether Double-SSD or Single-SSD is used.  
/// If none is specified, this defaults to [`Self::SingleSsd`].
///
/// See also [`crate::mamba3::ssd_path::Mamba3SsdPath`].
#[derive(Debug)]
pub enum Mamba3Caches {
    /// Caches for the double-ssd pathway.
    DoubleSsd(Mamba3DoubleSsdCaches),
    /// Caches for the single-ssd pathway.
    SingleSsd(Mamba3SingleSsdCaches),
}

/// A pathway-tagged bundle of per-block cache, so a single dispatch entry can
/// accept / return either cache family.
///
/// The cache selection infers whether Double-SSD or Single-SSD is used.  
/// If none is specified, this defaults to [`Self::SingleSsd`].
///
/// See also [`crate::mamba3::ssd_path::Mamba3SsdPath`].
#[derive(Debug)]
pub enum Mamba3Cache {
    /// Caches for double-ssd pathway.
    DoubleSsd(Mamba3DoubleSsdCache),
    /// Caches for single-ssd pathway.
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

    /// Collect per-layer caches into a pathway-tagged bundle.  The pathway is
    /// inferred from the first element (an empty vec implies single-SSD).
    pub fn from_vec(vec: Vec<Mamba3Cache>) -> Self {
        // peek at first; empty implies single_ssd
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

    /// Wrap each per-layer cache in `Some` so the loop can `take` it without
    /// cloning (Burn tensors are reference-counted).
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
// At a cache boundary (the last token of a `forward` / `step` call) the
// look-ahead term `(1 − λₜ₊₁)·Δₜ₊₁` vanishes, so `scaleₜ = γₜ` for the final
// position. Substituting that into the single-ssd accumulator
// `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ⊗xₜ` makes it coincide *exactly* with the
// double-ssd state `hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁⊗xₜ₋₁ + γₜ Bₜ⊗xₜ` — the deferred β
// contribution of the next token is reconstructed on the following call from
// the saved `k_state`/`v_state`, identically in both forms. The remaining
// three fields (previous-token K/V history and cumulative RoPE angle) carry the
// same meaning in both caches. Hence the conversion is a field-by-field move.
//
// The accumulators differ only *mid-sequence*; since caches are only ever
// produced and consumed at boundaries, the move is lossless there. The distinct
// types still prevent silently mixing the two accumulators inside a single
// chunked pass.

impl From<Mamba3SingleSsdCache> for Mamba3DoubleSsdCache {
    fn from(cache: Mamba3SingleSsdCache) -> Self {
        Mamba3DoubleSsdCache {
            ssm_bhpr: cache.ssm_bhpr,
            k_state_bmhr: cache.k_state_bmhr,
            v_state_bhp: cache.v_state_bhp,
            rotation: cache.rotation,
        }
    }
}

impl From<Mamba3DoubleSsdCache> for Mamba3SingleSsdCache {
    fn from(cache: Mamba3DoubleSsdCache) -> Self {
        Mamba3SingleSsdCache {
            ssm_bhpr: cache.ssm_bhpr,
            k_state_bmhr: cache.k_state_bmhr,
            v_state_bhp: cache.v_state_bhp,
            rotation: cache.rotation,
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
