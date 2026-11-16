//! # The tap pattern of the trapezoid — which earlier sample the `β` tap reads
//!
//! The write of Mamba-3 is a two-tap filter on the samples: the current
//! sample at `γₜ = λₜΔₜ`, and an earlier one at `βₜ = (1−λₜ)Δₜαₜ`, transported
//! across the gap between them (`helpers::trapezoidal_coefficients`). At
//! [`micro_steps`](crate::mamba3::mamba3::Mamba3Config::micro_steps) `= 1`,
//! "earlier" can only mean the previous token, so there is no choice. At
//! `u > 1`, the write axis has `u` positions per token
//! ([`crate::mamba3::product`]), and the choice is real.
//!
//! [`TapPattern`] names the members of that lattice.
//! `info/mamba-3/trapezoid-as-integration.md` §§8–9 derives it, prices each
//! member, and proves the invariant that all members keep: each tap is
//! transported across *its own* gap, which keeps the single-SSD `Δ̃` collapse.
//! This module cites the note and does not restate it.
//!
//! ## One rule for every member
//!
//! A pattern says which earlier samples are **admissible** at each write
//! position. Their *weight* is not a second decision. Two learned per-(head,
//! micro-step) scalars split the mass `Δₚ` of the micro-step. A tap that is not
//! admissible gives its mass back one level. The lag-1 tap of a two-tap
//! member gives it to the lag-`u` tap. The tap at
//! [`tap_lag`](TapPattern::tap_lag) gives it to `γ`. Under
//! [`HorizontalInToken`](TapPattern::HorizontalInToken), that tap is a lag-1 tap.
//! The masses are:
//!
//! ```text
//!   γ = λΔ            the right endpoint, what the read sees   (λ = σ(λ̂))
//!   ν = (1−λ)Δ        the left-endpoint mass                   (μ = σ(μ̂))
//!   νⁱⁿᵗ = ν·μ        at the lag-1 tap  — pairs micro-steps inside a token
//!   νᶠᵃʳ = ν − νⁱⁿᵗ   at the lag-`u` tap — the same micro-step of the previous token
//! ```
//!
//! So `γ + νⁱⁿᵗ + νᶠᵃʳ = Δ` at every position, open or closed. `λ` keeps the
//! meaning that the note gives it (the splitting parameter of §4). `μ`, the
//! one channel that the two-tap members add, says **which earlier sample is
//! the left endpoint**: it interpolates between the two candidates. Thus the
//! quadrature weights stay non-negative and still sum to `Δ`. The "top-up,
//! never a rollback" reading of §5 holds for the wider tap set, and so does the
//! collapse: sample `s` carries the one scalar `γₛ + νⁱⁿᵗₛ₊₁ + νᶠᵃʳₛ₊ᵤ`.
//!
//! Because of this fallback, the members whose lag-1 tap closes at token
//! boundaries, and the two-tap members, are *submodels*, not different models:
//!
//! - [`HorizontalInToken`](TapPattern::HorizontalInToken) is
//!   [`HorizontalCarryOver`](TapPattern::HorizontalCarryOver) with `λ = 1` at
//!   the first micro-step of each token (the limit of §8).
//! - [`VerticalPlusHorizontalCarryOver`](TapPattern::VerticalPlusHorizontalCarryOver)
//!   *contains* both implemented single-tap patterns, at `μ ≡ 1` and `μ ≡ 0`.
//!
//! If a closed tap dropped its mass instead of giving it back, both statements
//! would be false, and the write would silently be weaker at `1/u` of the
//! positions.

/// Which earlier sample(s) the `β` tap of the trapezoid reads.
///
/// The choice is **structural**, not a knob on one common algorithm. It
/// decides:
///
/// - the shift(s) that the chunkwise pathways apply before chunking,
/// - the width of the single-SSD γ-correction band,
/// - whether the in-projection has `λ` and `μ` channels,
/// - how many slots the tap FIFO of the cache holds ([`tap_slots`](Self::tap_slots)).
///
/// The lattice is the product of the settings of the two taps. The lag-1 tap
/// is absent, closed at token boundaries, or unrestricted (the `Horizontal…`
/// part of a name). The lag-`u` tap is absent or present (the `Vertical`
/// part). All six cells exist:
///
/// ```text
///                      no lag-u tap                + lag-u tap
///   no lag-1 tap       None                        Vertical
///   in token (j ≥ 1)   HorizontalInToken           VerticalPlusHorizontalInToken
///   carry (all j)      HorizontalCarryOver         VerticalPlusHorizontalCarryOver
/// ```
///
/// The single-tap members are one algorithm at two lags
/// ([`tap_lag`](Self::tap_lag)). The right column adds one lag-1 tap, mixed in
/// by `μ` (module header). At `u = 1` the lags are equal, so:
///
/// - [`Vertical`](Self::Vertical) *is*
///   [`HorizontalCarryOver`](Self::HorizontalCarryOver),
/// - the two-tap members are also `HorizontalCarryOver` (they fold and have no
///   `μ` channels, see [`has_two_taps`](Self::has_two_taps)),
/// - [`HorizontalInToken`](Self::HorizontalInToken) *is* [`None`](Self::None).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TapPattern {
    /// **No second tap.** `λ ≡ 1`, so `β = 0` and `γ = Δ`: the write is plain
    /// exponential-Euler, as in Mamba-2 (the §4 Lie–Trotter row of the note).
    ///
    /// Structural, as
    /// [`RotationKind::Real1D`](crate::mamba3::rotation::RotationKind::Real1D)
    /// is. The absent term costs nothing:
    ///
    /// - the in-projection has no `λ` channels (so Muon sees no `λ`
    ///   sub-projection),
    /// - no `β` is formed, and the tap FIFO of each cache is `None`,
    /// - `forward` makes **one** SSD call, and `step` has no second outer
    ///   product.
    ///
    /// The two SSD pathways are the same here. With no second pass to fuse, the
    /// composite key scale of single-SSD is `γ`, and its γ-correction is the
    /// whole diagonal. So
    /// [`forward_single_ssd`](crate::mamba3::mamba3::Mamba3::forward_single_ssd)
    /// runs the double-SSD code, and the caches convert by field identity at
    /// *every* position, not only at boundaries.
    None,

    /// **Lag `u`, always**: the tap reads the *same* micro-step of the previous
    /// **token**. So every tap crosses a token boundary, and the pattern is `u`
    /// parallel token-rate filters, one per micro-step channel. It restores the
    /// `u = 1` tap semantics at every micro-step.
    ///
    /// The **tap graph** is "vertical", not the scan order. The state still
    /// runs the one flattened chain ([`crate::mamba3::product`]), which keeps
    /// the pattern causal and `forward` equal to an unrolled `step`. A scan
    /// that really took every token at micro-step `0` before micro-step `1`
    /// would have to read token `t+1` before it finished token `t`. The
    /// alternative, `u` separate states, is a layer stack (`burn_stack::Layers`),
    /// not a tap pattern.
    ///
    /// At token resolution the pattern has a closed form. Let `Aᵗ` be the
    /// token transition and `ṽᵗⱼ` the write of micro-step `j`, transported to
    /// the end of its own token. The transport of the tap,
    /// `M₍ₜ₋₁,ⱼ₊₁₎:₍ₜ,ⱼ₎`, factors through that of the state, and the whole
    /// `β` side lands **at the token boundary**:
    ///
    /// ```text
    ///   hₜ = Aₜ · ( hₜ₋₁ + Σⱼ νₜ,ⱼ ṽₜ₋₁,ⱼ ) + Σⱼ γₜ,ⱼ ṽₜ,ⱼ ,     ν = (1−λ)Δ
    /// ```
    ///
    /// This is **the `u = 1` trapezoid at token resolution, with its left
    /// endpoint promoted from rank 1 to rank `u`**.
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) taps only the *last*
    /// micro-step of the previous token. `A` is unchanged, so it is still
    /// `λ`-free (`info/mamba-3/trapezoid-as-integration.md` §7).
    ///
    /// Costs (§9):
    ///
    /// - the key scale is `γₛ + (1−λₛ₊ᵤ)Δₛ₊ᵤ`,
    /// - the tap is transported across its own `u`-position gap,
    /// - the tap buffer of the cache becomes a `u`-deep FIFO (exactly the
    ///   previous token),
    /// - the single-SSD γ-correction widens from the diagonal to a
    ///   `u`-wide band. At the only positions whose output survives
    ///   (`j = u−1`), that band is **exactly the token**. So it is one small
    ///   intra-token contraction outside the chunked kernel, not a wider mask
    ///   (`crate::mamba3::single_ssd::token_band`).
    ///
    /// **Under a Kalman [`Gain`](crate::mamba3::positive::Gain) at `u > 1`**,
    /// this tap has no exact filter form. Its installment arrives `u`
    /// micro-steps late, after `u − 1` predicts that a filter would have made
    /// with it. The gain counts it as a lag-1 installment, so `Λ`
    /// over-estimates the weight that the plant wrote, and `η/Λ` is only
    /// approximately its weighted mean.
    /// The block still runs, and its pathways still agree. The lag-1 patterns
    /// are exact ([`crate::mamba3::positive::kalman`]).
    Vertical,

    /// **Lag 1, closed at the first micro-step of each token**: taps pair
    /// micro-steps *within* a token and never cross a token boundary. With no
    /// cross-token path, it cannot do the job that the trapezoid exists for. It
    /// is a component of a pattern, not an alternative to one.
    ///
    /// The closed micro-step has no admissible earlier sample, so all its mass
    /// returns to `γ` (`λ = 1` there, module header). Each token **starts** on
    /// plain exponential-Euler and is trapezoidal inside. This is exactly the
    /// limit that `info/mamba-3/trapezoid-as-integration.md` §8 shows learning
    /// can reach under [`HorizontalCarryOver`](Self::HorizontalCarryOver). So
    /// this member is a constrained **submodel** of that pattern, not a rival.
    /// That is also why the mass goes back instead of being dropped.
    ///
    /// Caches like [`HorizontalCarryOver`](Self::HorizontalCarryOver): the same
    /// one-slot lag-1 buffer, ignored at the first micro-step of each token.
    /// So the carried value has no effect across a call boundary: the slot is
    /// layout, not information.
    HorizontalInToken,

    /// **Lag 1, always** (the default): one tap per position of the write
    /// axis. So `1/u` of the taps cross a token boundary, and `(u−1)/u`
    /// pair two projections of the same token (§8). The cache holds one
    /// `(B, x)` slot: the last micro-step of the last token.
    #[default]
    HorizontalCarryOver,

    /// [`Vertical`](Self::Vertical) **and**
    /// [`HorizontalInToken`](Self::HorizontalInToken) at once. `μ` shares the
    /// left-endpoint mass between them, so the two jobs do not share one `λ`.
    /// Its tap graph is not a single lag on the write axis. On the
    /// `token × micro-step` grid, *two* taps write cell `(t, j)`: `(t−1, j)`
    /// above it, and `(t, j−1)` beside it (for `j ≥ 1` only).
    ///
    /// Only the graph is 2-D, not the algorithm. Each tap is transported
    /// across its own gap (`1` and `u` positions of the same write axis), so
    /// the collapse of §9 still applies term by term. The state stays one
    /// matrix, and sample `s` still carries **one** scalar,
    /// `γₛ + νⁱⁿᵗₛ₊₁ + νᶠᵃʳₛ₊ᵤ`, into every later micro-step. So single-SSD still
    /// needs one pass.
    ///
    /// Costs over [`Vertical`](Self::Vertical):
    ///
    /// - one more per-micro-step scalar channel (`μ`),
    /// - a third nonzero per row of the banded factor of the mask.
    ///
    /// The band is already `u` wide, and the within-token tap never crosses a
    /// token boundary. So the cache and the boundary seed are those of
    /// [`Vertical`](Self::Vertical). On the double-SSD pathway, the two taps
    /// have different shifts and cannot share a pass, so that pathway runs
    /// **three** SSD calls. The single pathway suits this pattern better.
    ///
    /// Under a Kalman [`Gain`](crate::mamba3::positive::Gain) at `u > 1`, it is
    /// inexact through its lag-`u` tap, as [`Vertical`](Self::Vertical) is.
    VerticalPlusHorizontalInToken,

    /// [`Vertical`](Self::Vertical) **and**
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) at once. It has the
    /// same two taps as
    /// [`VerticalPlusHorizontalInToken`](Self::VerticalPlusHorizontalInToken), with
    /// the lag-1 tap unrestricted. So at `j = 0`, the lag-1 tap reads the
    /// *last* micro-step of the previous token, and the lag-`u` tap reads its
    /// *first*. Both candidates for the left endpoint of the micro-step are always
    /// available, and `μ` interpolates between them everywhere.
    ///
    /// So this member **contains** the two single-tap patterns: `μ ≡ 1` is
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) and `μ ≡ 0` is
    /// [`Vertical`](Self::Vertical), both exactly. `μ` is per (head,
    /// micro-step), so descent makes here the choice that the rest of this enum
    /// makes at configuration time.
    ///
    /// Costs over [`VerticalPlusHorizontalInToken`](Self::VerticalPlusHorizontalInToken):
    ///
    /// - nothing in the cache: the **newest** slot of the `u`-deep FIFO *is*
    ///   the lag-1 slot, with the empty decay product,
    /// - one extra term in the single-SSD boundary seed: here the lag-1 tap
    ///   crosses a call boundary. Under
    ///   [`VerticalPlusHorizontalInToken`](Self::VerticalPlusHorizontalInToken),
    ///   it never crosses one.
    ///
    /// Under a Kalman [`Gain`](crate::mamba3::positive::Gain) at `u > 1`, it is
    /// inexact through its lag-`u` tap, as [`Vertical`](Self::Vertical) is.
    VerticalPlusHorizontalCarryOver,
}

impl TapPattern {
    /// Whether the pattern has a second (`β`) tap. `false` only for
    /// [`None`](Self::None). Every site branches on this predicate: no tap
    /// means no `λ` channels, no `β` coefficient, no tap FIFO in the cache,
    /// and no previous-sample term in the recurrence.
    pub fn has_beta_tap(self) -> bool {
        self != TapPattern::None
    }

    /// How far back the `β` tap reaches, in **write** positions
    /// ([`crate::mamba3::product`]): `0` with no tap, `1` for the lag-1
    /// patterns, `u` for the lag-`u` ones.
    ///
    /// This is the one knob for the single-tap patterns:
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) is `lag = 1` and
    /// [`Vertical`](Self::Vertical) is `lag = u`. Every site that touches the
    /// tap (the double-SSD shift, the single-SSD key scale, the FIFO depth of
    /// `step`, the cache) reads it instead of branching on the pattern. At
    /// `u = 1` both return `1`, which is why the two are equal there.
    ///
    /// The tap must be transported across *its own* gap, so a lag-`L` tap
    /// carries `Πᵈ⁼⁰..ᴸ⁻¹ αₚ₋ᵈ`, not `αₚ`.
    /// `info/mamba-3/trapezoid-as-integration.md` §9 shows that this condition
    /// keeps the `Δ̃` collapse, and thus the single-SSD pathway.
    pub fn tap_lag(self, micro_steps: usize) -> usize {
        match self {
            TapPattern::None => 0,
            TapPattern::HorizontalCarryOver | TapPattern::HorizontalInToken => 1,
            TapPattern::Vertical
            | TapPattern::VerticalPlusHorizontalInToken
            | TapPattern::VerticalPlusHorizontalCarryOver => micro_steps,
        }
    }

    /// Whether the pattern has **two** taps at `micro_steps`: a lag-1 tap beside
    /// the lag-`u` tap at [`tap_lag`](Self::tap_lag). True for the two-tap
    /// members, and only when the two lags are different.
    ///
    /// At `u = 1` the lags are equal: both taps read the same sample, so the
    /// split is a decomposition of one coefficient, and the pair **folds** back
    /// into one tap. The fold is structural, as the missing `λ` of
    /// [`None`](Self::None) is: no `μ` channels in the in-projection, no `μ`
    /// sub-projection for Muon, one shift, one SSD pass. Thus a two-tap member
    /// at `u = 1` *is* [`HorizontalCarryOver`](Self::HorizontalCarryOver), not
    /// an expensive form of it.
    pub fn has_two_taps(self, micro_steps: usize) -> bool {
        micro_steps > 1
            && matches!(
                self,
                TapPattern::VerticalPlusHorizontalInToken
                    | TapPattern::VerticalPlusHorizontalCarryOver
            )
    }

    /// Whether the tap at [`tap_lag`](Self::tap_lag) is admissible at the
    /// **first** micro-step of a token, where it would cross a token boundary.
    /// `false` only for
    /// [`HorizontalInToken`](Self::HorizontalInToken), whose left-endpoint mass
    /// then returns to `γ` (module header).
    pub fn tap_crosses_tokens(self) -> bool {
        self != TapPattern::HorizontalInToken
    }

    /// Whether the lag-1 tap of a two-tap member is admissible at the first
    /// micro-step of a token, where it would cross a token boundary. This is the
    /// only difference between
    /// [`VerticalPlusHorizontalCarryOver`](Self::VerticalPlusHorizontalCarryOver)
    /// (`true`) and
    /// [`VerticalPlusHorizontalInToken`](Self::VerticalPlusHorizontalInToken)
    /// (`false`). Where it is closed, the share of `μ` returns to the lag-`u`
    /// tap.
    ///
    /// Read only when [`has_two_taps`](Self::has_two_taps).
    pub fn two_tap_lag1_crosses_tokens(self) -> bool {
        self == TapPattern::VerticalPlusHorizontalCarryOver
    }

    /// The number of slots in the tap FIFO of a cache, for this pattern at
    /// `micro_steps`. It is the depth of the FIFO that the tap reads, so
    /// exactly [`tap_lag`](Self::tap_lag). A lag-`L` tap needs the last `L`
    /// positions.
    /// The state is always one matrix. Only the tap FIFO changes.
    ///
    /// A slot is a *layout*. It does not mean that something crosses the call
    /// boundary through it: [`HorizontalInToken`](Self::HorizontalInToken) shares
    /// the slot of the carry-over and ignores it per token. Also, a second tap
    /// needs no second buffer. The **newest** slot of the `u`-deep FIFO *is*
    /// the lag-1 slot, with the empty decay product, so the two-tap members
    /// read one FIFO twice.
    pub fn tap_slots(self, micro_steps: usize) -> usize {
        self.tap_lag(micro_steps)
    }
}

/// Everything the discretisation needs from the block: the tap pattern, the
/// micro-steps (the period of the token boundaries, where a tap can close), and
/// the two clamps on the coefficients.
///
/// [`Mamba3::tap_pattern_spec`](crate::mamba3::mamba3::Mamba3::tap_pattern_spec)
/// makes it, and `helpers::trapezoidal_coefficients` takes it. So `forward`
/// and `step` compute the masses from **one** definition. The same reason
/// makes [`RotationSpec`](crate::mamba3::rotation::RotationSpec) exist.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TapPatternSpec {
    /// Which earlier sample(s) the `β` tap reads ([`TapPattern`]).
    pub pattern: TapPattern,
    /// Recurrence micro-steps per token (`u`). A tap that closes at token
    /// boundaries closes with this period. At `u = 1`, a two-tap pattern folds
    /// ([`TapPattern::has_two_taps`]).
    pub micro_steps: usize,
    /// `Δ`'s clamp
    /// ([`Mamba3Config::dt_limit`](crate::mamba3::mamba3::Mamba3Config::dt_limit)).
    pub dt_limit: (f64, f64),
    /// `A`'s floor
    /// ([`Mamba3Config::a_floor`](crate::mamba3::mamba3::Mamba3Config::a_floor)).
    pub a_floor: f64,
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
