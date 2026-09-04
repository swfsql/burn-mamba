//! # The trapezoid's tap pattern — which earlier sample the `β` tap reads
//!
//! Mamba-3's write is a two-tap filter on the state input: the current sample
//! at `γₜ = λₜΔₜ` and an earlier one at `βₜ = (1−λₜ)Δₜαₜ`, transported across
//! the gap between them (`helpers::trapezoidal_coefficients`). At
//! [`micro_steps`](crate::mamba3::mamba3::Mamba3Config::micro_steps) `= 1`
//! "earlier" can only mean the previous token and there is nothing to choose;
//! at `u > 1` the folded sequence carries `u` positions per token
//! ([`crate::mamba3::product`]) and the choice is real.
//!
//! [`Trapezoid`] names the members of that lattice.
//! `info/trapezoid-as-integration.md` §§8–9 derives it, prices each member and
//! proves the invariant they all keep (each tap transported across *its own*
//! gap, which is what preserves the single-SSD `Δ̃` collapse) — cite it, it is
//! not restated here.

/// Which earlier sample the trapezoid's `β` tap reads.
///
/// The choice is **structural**, not a knob on one common algorithm: it decides
/// the shift the chunkwise pathways apply before chunking, the width of the
/// single-SSD γ-correction band, whether the in-projection spends `λ` channels
/// at all, and how many `(B, x)` tap slots the cache carries
/// ([`tap_slots`](Self::tap_slots)). Two members are also degenerate at
/// `u = 1`: [`Vertical`](Self::Vertical) *is*
/// [`HorizontalCarryOver`](Self::HorizontalCarryOver) there, and
/// [`HorizontalReset`](Self::HorizontalReset) *is* [`None`](Self::None).
///
/// Only [`HorizontalCarryOver`](Self::HorizontalCarryOver) — the default, and
/// what the crate has always done — is implemented. The others are named so the
/// lattice has a vocabulary and so
/// [`Mamba3Config::init`](crate::mamba3::mamba3::Mamba3Config::init) rejects
/// them loudly ([`assert_implemented`](Self::assert_implemented)) instead of
/// silently running a different recurrence than the one asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Trapezoid {
    /// **No second tap.** `λ ≡ 1`, hence `β = 0` and `γ = Δ`: the write is plain
    /// exponential-Euler, i.e. Mamba-2's (the note's §4 Lie–Trotter row).
    ///
    /// Structural in the same sense as
    /// [`RotationKind::Real1D`](crate::mamba3::rotation::RotationKind::Real1D)
    /// — the in-projection spends no `λ` channels and the cache carries no tap
    /// slots — rather than a learned `λ` that happens to saturate.
    None,

    /// **Lag `u`, always**: the tap reads the *same* micro-step of the previous
    /// **token**, so every tap crosses a token boundary and the pattern is `u`
    /// parallel token-rate filters, one per micro-step channel. Restores the
    /// `u = 1` tap semantics at every micro-step.
    ///
    /// Costs (§9): key scale `γₛ + (1−λₛ₊ᵤ)Δₛ₊ᵤ`; the single-SSD same-step
    /// correction widens from the diagonal to a `u`-wide band; double-SSD
    /// shifts by `u`; the cache's tap gains a `u` axis.
    Vertical,

    /// **Lag 1, suppressed at each token's first micro-step**: taps pair
    /// micro-steps *within* a token and never cross a token boundary. Having no
    /// cross-token path at all, it cannot do the job the trapezoid was
    /// introduced for — it is a component of a pattern rather than an
    /// alternative to one.
    ///
    /// Caches like [`HorizontalCarryOver`](Self::HorizontalCarryOver): the same
    /// one-slot lag-1 buffer, reset at each token's first micro-step. (The reset
    /// is what makes the carried value inert across a call boundary — the slot
    /// is layout, not information.)
    HorizontalReset,

    /// **Lag 1, always** — the default, and the only implemented pattern: one
    /// tap per position of the folded sequence, so `1/u` of the taps cross a
    /// token boundary and `(u−1)/u` pair two projections of the same token
    /// (§8). The cache carries one `(B, x)` slot, the last micro-step of the
    /// last token.
    #[default]
    HorizontalCarryOver,

    /// [`Vertical`](Self::Vertical) **and**
    /// [`HorizontalReset`](Self::HorizontalReset) at once, with its own
    /// coefficient each instead of the two jobs sharing one `λ`. The **least
    /// settled** member, and the only one whose tap graph is not a single lag on
    /// the folded chain: on the `token × micro-step` grid, cell `(t, j)` is
    /// written by *two* taps — `(t−1, j)` above it and `(t, j−1)` beside it.
    ///
    /// What is 2-D is that graph, not the algorithm. Both taps are transported
    /// across their own gap (`1` and `u` positions of the same folded chain), so
    /// §9's collapse still applies term by term: the state stays one matrix and
    /// sample `s` still carries **one** scalar, `γₛ + νʰₛ₊₁ + νᵛₛ₊ᵤ`, into every
    /// later step. Costs over [`Vertical`](Self::Vertical): one more
    /// per-micro-step scalar channel and a third nonzero per row of the mask's
    /// banded factor — the band itself is already `u` wide, and the within-token
    /// tap never crosses a token boundary, so the cache is
    /// [`Vertical`](Self::Vertical)'s unchanged.
    VerticalPlusHorizontalReset,
}

impl Trapezoid {
    /// Whether this crate implements the pattern today — only
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) does.
    pub fn is_implemented(self) -> bool {
        matches!(self, Trapezoid::HorizontalCarryOver)
    }

    /// Panic unless [`is_implemented`](Self::is_implemented).
    ///
    /// Called from
    /// [`Mamba3Config::init`](crate::mamba3::mamba3::Mamba3Config::init), the
    /// block's only constructor, so an unimplemented pattern fails at
    /// configuration time rather than producing plausible numbers from the
    /// wrong recurrence.
    pub fn assert_implemented(self) {
        assert!(
            self.is_implemented(),
            "Trapezoid::{self:?} is not implemented yet — only \
             Trapezoid::HorizontalCarryOver (the default) has an algorithm; see \
             info/trapezoid-as-integration.md §9 for what the others cost"
        );
    }

    /// How many `(B, x)` tap slots a cache carries for this pattern at
    /// `micro_steps`: `0` with no tap at all, `1` for the lag-1 patterns, `u`
    /// for the lag-`u` ones (§9). The SSM state is one matrix whatever this says
    /// — only the trapezoid's tap buffer changes.
    ///
    /// A slot is a *layout*, not a claim that something crosses the call
    /// boundary through it: [`HorizontalReset`](Self::HorizontalReset) shares
    /// the carry-over's slot and resets it per token, and the `u` slots of the
    /// lag-`u` patterns already cover their within-token tap, whose partner is
    /// one of those same `u` positions.
    pub fn tap_slots(self, micro_steps: usize) -> usize {
        match self {
            Trapezoid::None => 0,
            Trapezoid::HorizontalCarryOver | Trapezoid::HorizontalReset => 1,
            Trapezoid::Vertical | Trapezoid::VerticalPlusHorizontalReset => micro_steps,
        }
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests {
    use super::*;
    use crate::mamba3::mamba3::Mamba3Config;

    fn cfg() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
    }

    /// The default must stay the pattern the crate implements, so an untouched
    /// config keeps building.
    #[test]
    fn default_is_the_implemented_pattern() {
        assert_eq!(Trapezoid::default(), Trapezoid::HorizontalCarryOver);
        assert_eq!(cfg().trapezoid, Trapezoid::HorizontalCarryOver);
        assert!(Trapezoid::default().is_implemented());
        let device: burn::prelude::Device = Default::default();
        let _ = cfg().init(&device);
    }

    /// Every other pattern is a different algorithm *and* a different cache, so
    /// selecting one must fail at construction rather than silently run this one.
    #[test]
    fn unimplemented_patterns_panic_at_init() {
        let device: burn::prelude::Device = Default::default();
        for pattern in [
            Trapezoid::None,
            Trapezoid::Vertical,
            Trapezoid::HorizontalReset,
            Trapezoid::VerticalPlusHorizontalReset,
        ] {
            assert!(!pattern.is_implemented(), "{pattern:?}");
            let config = cfg().with_trapezoid(pattern);
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                config.init(&device);
            }))
            .is_err();
            assert!(panicked, "{pattern:?} must be rejected by init");
        }
    }

    /// The tap buffer is what the pattern changes in the cache: none without a
    /// tap, one slot for the lag-1 patterns, `u` for the lag-`u` ones.
    #[test]
    fn tap_slots_follow_the_lag() {
        assert_eq!(Trapezoid::None.tap_slots(3), 0);
        assert_eq!(Trapezoid::HorizontalReset.tap_slots(3), 1);
        assert_eq!(Trapezoid::HorizontalCarryOver.tap_slots(3), 1);
        assert_eq!(Trapezoid::Vertical.tap_slots(3), 3);
        assert_eq!(Trapezoid::VerticalPlusHorizontalReset.tap_slots(3), 3);
        // At `u = 1` the lag-`u` patterns *are* the lag-1 one, and the two
        // patterns that degenerate to each other agree on the buffer too.
        assert_eq!(Trapezoid::Vertical.tap_slots(1), 1);
        assert_eq!(Trapezoid::VerticalPlusHorizontalReset.tap_slots(1), 1);
    }
}
