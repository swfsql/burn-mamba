//! # Mamba-3 Cache and Pathway Selection
//!
//! [`Mamba3Cache`] / [`Mamba3Caches`] are **enums** that tag the SSD pathway
//! of a cache (`DoubleSsd` | `SingleSsd`). The variant given to
//! [`Mamba3::forward`](crate::mamba3::mamba3::Mamba3::forward) /
//! [`Mamba3::step`](crate::mamba3::mamba3::Mamba3::step) selects the pathway
//! at runtime. A missing cache selects `SingleSsd`:
//! [`Mamba3::zero_cache`] builds each fresh cache, of that pathway.
//!
//! The accumulator of single-SSD and the state of double-SSD are different
//! **mid-sequence**, so the cache types are distinct: this prevents a silent
//! mix inside a chunked pass. They are equal at call boundaries, where caches
//! are produced and consumed. So the `From` impls at the bottom convert
//! between them by a lossless field-by-field move (see the note there).
//!
//! Both cache types hold the `ssm_bhpr` of their pathway and one
//! [`Mamba3FrameCache`]: the fields that the plant frame writes. Its
//! [`TapFifo`] is the tap FIFO of the trapezoid. It owns its layout: its
//! zeros, and the docs of the slot order and of the decay pre-scale of `x`.

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::mamba3::Mamba3;
use crate::mamba3::single_ssd::prelude::*;
use burn::prelude::*;
use burn_stack::modules::sanity as san;

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

    /// Inverse of [`Self::into_options`]: unwrap each layer cache and re-bundle.
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

// ---------------------------------------------------------------------------
// The tap FIFO
// ---------------------------------------------------------------------------

/// The tap FIFO of the trapezoid: the `B` and the `x` of the last write
/// positions. A cache keeps them for the `β` taps of the next call. One slot
/// holds one write position, and the slots are **oldest first**.
///
/// The frame cache holds it as an option ([`Mamba3FrameCache::tap_fifo`]).
/// It has [`tap_slots`](crate::mamba3::tap_pattern::TapPattern::tap_slots)
/// slots: `1` for the default, `u` for
/// [`TapPattern::Vertical`](crate::mamba3::tap_pattern::TapPattern::Vertical).
#[derive(Module, Debug)]
pub struct TapFifo {
    /// **The `B` of each slot**, after its bias and its rotation.
    ///
    /// It is stored **as rotated**, at its own position. The relative-rotation
    /// factoring `C̄ₜᵀB̄ₛ` then rebuilds the transport, so the next call
    /// applies no rotation again.
    ///
    /// Shape: `[batch, tap_slots, mimo_rank, nheads, state_rank]`
    pub b_bumhr: Tensor<5>,

    /// **The `x` of each slot**, pre-scaled by the decay accumulated since its
    /// own position (`helpers::tail_decay`). This carries the gap transport of
    /// a lag-`u` tap across the call boundary. At `lag = 1` that product is
    /// empty, so the slot is the plain `xₜ₋₁`.
    ///
    /// Shape: `[batch, tap_slots, nheads, per_head_dim]`
    pub x_buhp: Tensor<4>,
}

impl TapFifo {
    /// The FIFO of a fresh sequence: `slots` zero slots. A zero slot adds
    /// nothing to a tap.
    pub(crate) fn zeros(
        batch: usize,
        slots: usize,
        mimo_rank: usize,
        nheads: usize,
        state_rank: usize,
        per_head_dim: usize,
        device: &Device,
    ) -> Self {
        Self {
            b_bumhr: Tensor::zeros([batch, slots, mimo_rank, nheads, state_rank], device),
            x_buhp: Tensor::zeros([batch, slots, nheads, per_head_dim], device),
        }
    }

    /// The newest `slots` slots, oldest first: the prefix of a tap that reads
    /// `slots` write positions back. The tap at
    /// [`tap_lag`](crate::mamba3::tap_pattern::TapPattern::tap_lag) reads the
    /// whole FIFO. The lag-1 tap of a two-tap pattern reads only the newest
    /// slot, which has the empty decay product. So a two-tap pattern reads one
    /// FIFO twice.
    pub(crate) fn newest(&self, slots: usize) -> Self {
        let start = self.x_buhp.dims()[1] - slots;
        Self {
            b_bumhr: self.b_bumhr.clone().narrow(1, start, slots),
            x_buhp: self.x_buhp.clone().narrow(1, start, slots),
        }
    }

    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on both
    /// tensors.
    pub fn sanity(&self) {
        san(&self.b_bumhr);
        san(&self.x_buhp);
    }
}

// ---------------------------------------------------------------------------
// The frame cache
// ---------------------------------------------------------------------------

/// The frame cache: the part of a Mamba-3 cache that the plant frame writes.
/// It has the same meaning in both pathways, and both cache types hold it
/// beside their `ssm_bhpr`.
#[derive(Module, Debug)]
pub struct Mamba3FrameCache {
    /// **The tap FIFO** ([`TapFifo`]). The `β` taps of the next call read it.
    /// Double-SSD rebuilds its β term
    /// `β * sumₘ Bₚ₋ₗₐ₉[m] ⊗ (xₚ₋ₗₐ₉ * mimo_x[m])` from it. Single-SSD seeds
    /// the deferred boundary β contributions
    /// `Σⱼ (1 − λⱼ) · Δⱼ · Bₚ₋ₗₐ₉₊ⱼ ⊗ xₚ₋ₗₐ₉₊ⱼ` from it. The previous call
    /// could not add them, because it did not know the `λ, Δ` of the `lag`
    /// positions that pay them.
    ///
    /// `None` (not allocated, not zeroed) under
    /// [`TapPattern::None`](crate::mamba3::tap_pattern::TapPattern::None),
    /// which has no β term.
    pub tap_fifo: Option<TapFifo>,

    /// **Cumulative data-dependent rotation** up to the current position
    /// ([`RotationState`](crate::mamba3::rotation::RotationState)). It holds
    /// nothing for `Real1D`, and the abelian rotation angle for
    /// [`Complex2D`](crate::mamba3::rotation::RotationKind::Complex2D) (each
    /// micro-step `cum_angleₜ = cum_angleₜ₋₁ + Δₜ · range · π · tanh(θₜ)`). It
    /// holds the cumulative unit quaternion(s) for the quaternion kinds.
    ///
    /// The identity for fresh sequences. It continues across calls for
    /// streaming.
    pub rotation: crate::mamba3::rotation::RotationState,

    /// **The log-precision of the Kalman gain**, `ln Λ` after the last
    /// position ([`crate::mamba3::positive::kalman`]). It is the evidence that
    /// the head holds. The gain computes the decay of the next position from
    /// it.
    ///
    /// [`LOG_ZERO`](crate::mamba3::positive::LOG_ZERO) (`Λ = 0`) for a fresh
    /// sequence. `None` under
    /// [`Gain::Projected`](crate::mamba3::positive::Gain::Projected).
    ///
    /// Shape: `[batch, nheads]`
    pub log_precision_bh: Option<Tensor<2>>,

    /// **The tropical register** `c` after the last position
    /// ([`crate::mamba3::positive::tropical`]).
    ///
    /// [`LOG_ZERO`](crate::mamba3::positive::LOG_ZERO) (the max of nothing) for
    /// a fresh sequence. `None` under
    /// [`Tropical::None`](crate::mamba3::positive::Tropical::None).
    ///
    /// Shape: `[batch, nheads]`
    pub tropical_bh: Option<Tensor<2>>,
}

impl Mamba3FrameCache {
    /// Run the [`NaN`/`Inf` guards](burn_stack::modules::misc::sanity) on every
    /// tensor of the frame cache.
    pub fn sanity(&self) {
        if let Some(tap_fifo) = &self.tap_fifo {
            tap_fifo.sanity();
        }
        self.rotation.sanity();
        for field in [&self.log_precision_bh, &self.tropical_bh].into_iter().flatten() {
            san(field);
        }
    }
}

// ---------------------------------------------------------------------------
// The fresh cache
// ---------------------------------------------------------------------------

impl Mamba3 {
    /// The cache of a fresh sequence, for `batch` rows. It holds a zero state,
    /// a zero tap FIFO, the identity rotation and the fresh registers of the
    /// positive systems. A `forward` or a `step` without a cache starts from it, and
    /// so does each segment of a packed call without an origin. The shapes come
    /// from the block.
    ///
    /// It is a single-SSD cache, because a missing cache selects that pathway.
    /// A caller that needs a fresh double-SSD cache converts it with `.into()`.
    /// A fresh cache is at a call boundary, so this move is lossless (see the
    /// note on the conversions below).
    pub fn zero_cache(&self, batch: usize, device: &Device) -> Mamba3SingleSsdCache {
        let ssm_bhpr = Tensor::zeros(
            [batch, self.nheads(), self.per_head_dim(), self.state_rank],
            device,
        );
        let tap_fifo = self.zero_tap_slots(batch, device);
        let rotation = self.zero_rotation_state(batch, device);
        let (log_precision_bh, tropical_bh) = self.zero_positive_state(batch, device);
        let frame = Mamba3FrameCache { tap_fifo, rotation, log_precision_bh, tropical_bh };
        Mamba3SingleSsdCache { ssm_bhpr, frame }
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
// installment, from the tap FIFO. With this substitution, the single-ssd
// accumulator `h'ₜ = αₜ h'ₜ₋₁ + scaleₜ Bₜ⊗xₜ` is *exactly* the double-ssd state
// `hₜ = αₜ hₜ₋₁ + Σ_taps βₜ Bₜ₋ₗ⊗xₜ₋ₗ + γₜ Bₜ⊗xₜ`. The next call rebuilds the
// deferred β contribution from the saved tap FIFO, identically in both forms.
//
// The frame cache has the same meaning in both caches. It holds the tap FIFO
// (the `B` and `x` of the last `lag` positions) and the cumulative rotation.
// It also holds the `ln Λ` and `c` of the positive systems, which read only
// the inputs. So the conversion moves the two fields.
//
// The accumulators differ only *mid-sequence*. Caches exist only at
// boundaries, so the move is lossless. The distinct types still prevent a
// silent mix of the two accumulators inside one chunked pass.

impl From<Mamba3SingleSsdCache> for Mamba3DoubleSsdCache {
    fn from(cache: Mamba3SingleSsdCache) -> Self {
        Mamba3DoubleSsdCache { ssm_bhpr: cache.ssm_bhpr, frame: cache.frame }
    }
}

impl From<Mamba3DoubleSsdCache> for Mamba3SingleSsdCache {
    fn from(cache: Mamba3DoubleSsdCache) -> Self {
        Mamba3SingleSsdCache { ssm_bhpr: cache.ssm_bhpr, frame: cache.frame }
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
