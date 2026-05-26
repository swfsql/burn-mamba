//! # Mamba-3 Cache and Pathway Selection

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::single_ssd::prelude::*;
use burn::prelude::*;

/// A pathway-tagged bundle of per-layer caches, so a single dispatch entry can
/// accept / return either cache family.
///
/// The caches selection infers whether Double-SSD or Single-SSD is used.  
/// If none is specified, this defaults to [`Self::SingleSsd`].
///
/// See also [`crate::mamba3::ssd_path::Mamba3SsdPath`].
#[derive(Debug)]
pub enum Mamba3Caches<B: Backend> {
    /// Caches for the double-ssd pathway.
    DoubleSsd(Mamba3DoubleSsdCaches<B>),
    /// Caches for the single-ssd pathway.
    SingleSsd(Mamba3SingleSsdCaches<B>),
}

/// A pathway-tagged bundle of per-block cache, so a single dispatch entry can
/// accept / return either cache family.
///
/// The cache selection infers whether Double-SSD or Single-SSD is used.  
/// If none is specified, this defaults to [`Self::SingleSsd`].
///
/// See also [`crate::mamba3::ssd_path::Mamba3SsdPath`].
#[derive(Debug)]
pub enum Mamba3Cache<B: Backend> {
    /// Caches for double-ssd pathway.
    DoubleSsd(Mamba3DoubleSsdCache<B>),
    /// Caches for single-ssd pathway.
    SingleSsd(Mamba3SingleSsdCache<B>),
}

impl<B: Backend> Mamba3Caches<B> {
    pub fn double_ssd(self) -> Option<Mamba3DoubleSsdCaches<B>> {
        match self {
            Self::DoubleSsd(caches) => Some(caches),
            Self::SingleSsd(_caches) => None,
        }
    }

    pub fn single_ssd(self) -> Option<Mamba3SingleSsdCaches<B>> {
        match self {
            Self::DoubleSsd(_caches) => None,
            Self::SingleSsd(caches) => Some(caches),
        }
    }

    pub fn caches_len(&self) -> usize {
        match self {
            Self::DoubleSsd(caches) => caches.caches.len(),
            Self::SingleSsd(caches) => caches.caches.len(),
        }
    }

    pub fn from_vec(vec: Vec<Mamba3Cache<B>>) -> Self {
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

    pub fn into_options(self) -> Vec<Option<Mamba3Cache<B>>> {
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

    pub fn from_options(options: Vec<Option<Mamba3Cache<B>>>) -> Self {
        let caches = options.into_iter().map(Option::unwrap).collect();
        Self::from_vec(caches)
    }
}

impl<B: Backend> Mamba3Cache<B> {
    pub fn double_ssd(self) -> Option<Mamba3DoubleSsdCache<B>> {
        match self {
            Self::DoubleSsd(cache) => Some(cache),
            Self::SingleSsd(_cache) => None,
        }
    }

    pub fn single_ssd(self) -> Option<Mamba3SingleSsdCache<B>> {
        match self {
            Self::DoubleSsd(_cache) => None,
            Self::SingleSsd(cache) => Some(cache),
        }
    }
}

impl<B: Backend> From<Mamba3DoubleSsdCaches<B>> for Mamba3Caches<B> {
    fn from(caches: Mamba3DoubleSsdCaches<B>) -> Self {
        Mamba3Caches::DoubleSsd(caches)
    }
}

impl<B: Backend> From<Mamba3SingleSsdCaches<B>> for Mamba3Caches<B> {
    fn from(caches: Mamba3SingleSsdCaches<B>) -> Self {
        Mamba3Caches::SingleSsd(caches)
    }
}

impl<B: Backend> From<Mamba3DoubleSsdCache<B>> for Mamba3Cache<B> {
    fn from(cache: Mamba3DoubleSsdCache<B>) -> Self {
        Mamba3Cache::DoubleSsd(cache)
    }
}

impl<B: Backend> From<Mamba3SingleSsdCache<B>> for Mamba3Cache<B> {
    fn from(cache: Mamba3SingleSsdCache<B>) -> Self {
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

impl<B: Backend> From<Mamba3SingleSsdCache<B>> for Mamba3DoubleSsdCache<B> {
    fn from(cache: Mamba3SingleSsdCache<B>) -> Self {
        Mamba3DoubleSsdCache {
            ssm_bhpr: cache.ssm_bhpr,
            k_state_bmhr: cache.k_state_bmhr,
            v_state_bhp: cache.v_state_bhp,
            cum_angle_bha: cache.cum_angle_bha,
        }
    }
}

impl<B: Backend> From<Mamba3DoubleSsdCache<B>> for Mamba3SingleSsdCache<B> {
    fn from(cache: Mamba3DoubleSsdCache<B>) -> Self {
        Mamba3SingleSsdCache {
            ssm_bhpr: cache.ssm_bhpr,
            k_state_bmhr: cache.k_state_bmhr,
            v_state_bhp: cache.v_state_bhp,
            cum_angle_bha: cache.cum_angle_bha,
        }
    }
}

impl<B: Backend> From<Mamba3SingleSsdCaches<B>> for Mamba3DoubleSsdCaches<B> {
    fn from(caches: Mamba3SingleSsdCaches<B>) -> Self {
        Mamba3DoubleSsdCaches {
            caches: caches.caches.into_iter().map(Into::into).collect(),
        }
    }
}

impl<B: Backend> From<Mamba3DoubleSsdCaches<B>> for Mamba3SingleSsdCaches<B> {
    fn from(caches: Mamba3DoubleSsdCaches<B>) -> Self {
        Mamba3SingleSsdCaches {
            caches: caches.caches.into_iter().map(Into::into).collect(),
        }
    }
}
