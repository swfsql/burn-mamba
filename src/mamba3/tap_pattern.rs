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
mod tests {
    use super::*;
    use crate::mamba3::cache::Mamba3Cache;
    use crate::mamba3::in_proj_layout::GroupKey;
    use crate::mamba3::mamba3::{Mamba3, Mamba3Config};
    use crate::mamba3::rotation::RotationKind;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    use burn::prelude::*;
    use burn::tensor::Distribution;
    use burn_stack::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff};
    use burn_stack::utils::test_helpers::test_device;

    fn cfg() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
    }

    fn cfg_none(kind: RotationKind, micro_steps: usize) -> Mamba3Config {
        cfg()
            .with_rotation(kind)
            .with_micro_steps(micro_steps)
            .with_tap_pattern(TapPattern::None)
    }

    fn input(config: &Mamba3Config, tokens: usize) -> Tensor<3> {
        Tensor::random(
            [2, tokens, config.d_model],
            Distribution::Normal(0.0, 1.0),
            &test_device(),
        )
    }

    /// Unroll `step` over the sequence, from a fresh cache.
    fn unrolled(model: &Mamba3, input_bsm: &Tensor<3>) -> (Tensor<3>, Mamba3Cache) {
        let tokens = input_bsm.dims()[1];
        let mut cache: Option<Mamba3Cache> = None;
        let mut outs = Vec::new();
        for t in 0..tokens {
            let (o, c) = model.step(input_bsm.clone().narrow(1, t, 1).squeeze_dim(1), cache);
            outs.push(o.unsqueeze_dim::<3>(1));
            cache = Some(c);
        }
        (Tensor::cat(outs, 1), cache.expect("a non-empty sequence"))
    }

    /// Every member of the lattice, at the `u` its taps are distinct at, for the
    /// bodies below to iterate over.
    const PATTERNS: [TapPattern; 6] = [
        TapPattern::None,
        TapPattern::HorizontalInToken,
        TapPattern::HorizontalCarryOver,
        TapPattern::Vertical,
        TapPattern::VerticalPlusHorizontalInToken,
        TapPattern::VerticalPlusHorizontalCarryOver,
    ];

    fn cfg_for(pattern: TapPattern, kind: RotationKind, micro_steps: usize) -> Mamba3Config {
        cfg()
            .with_rotation(kind)
            .with_micro_steps(micro_steps)
            .with_tap_pattern(pattern)
    }

    /// The default is the lag-1 carry-over, and an untouched config builds.
    #[test]
    fn default_is_the_carry_over() {
        assert_eq!(TapPattern::default(), TapPattern::HorizontalCarryOver);
        assert_eq!(cfg().tap_pattern, TapPattern::HorizontalCarryOver);
        let device = test_device();
        let _ = cfg().init(&device);
    }

    /// The lattice is closed: every member builds, and each pays for exactly the
    /// masses its taps need — `λ` unless the tap is absent, `μ` only for the
    /// two-tap members, and neither at a `u` where they would be inert.
    #[test]
    fn every_pattern_builds_and_pays_for_its_own_masses() {
        let device = test_device();
        for pattern in PATTERNS {
            for micro_steps in [1, 3] {
                let config = cfg_for(pattern, RotationKind::Complex2D, micro_steps);
                let block = config.init(&device);
                let scalars = 2
                    + usize::from(pattern.has_beta_tap())
                    + usize::from(pattern.has_two_taps(micro_steps));
                let label = format!("{pattern:?} u={micro_steps}");
                assert_eq!(
                    block.in_proj.weight.dims()[1],
                    config.d_in_proj(),
                    "{label}: in-projection width"
                );
                let layout = block.in_proj_layout();
                assert_eq!(
                    layout.channels_of(GroupKey::LambdaRaw) + layout.channels_of(GroupKey::MuRaw),
                    (scalars - 2) * micro_steps * config.nheads(),
                    "{label}: mass channels"
                );
                #[cfg(feature = "optim")]
                {
                    let names: Vec<String> = config.muon_projections()[0]
                        .sub_projections
                        .iter()
                        .map(|s| s.name.to_string())
                        .collect();
                    assert_eq!(
                        names.iter().any(|n| n == "lambda_raw"),
                        pattern.has_beta_tap(),
                        "{label}: λ sub-projection"
                    );
                    assert_eq!(
                        names.iter().any(|n| n == "mu_raw"),
                        pattern.has_two_taps(micro_steps),
                        "{label}: μ sub-projection"
                    );
                }
            }
        }
    }

    /// `None` is the ablation, and it is structural: no `λ` channels in the
    /// in-projection, and none of Muon's `λ` sub-projections either.
    #[test]
    fn none_drops_the_lambda_channels() {
        let with = cfg();
        let without = cfg().with_tap_pattern(TapPattern::None);
        assert!(!TapPattern::None.has_beta_tap());
        assert_eq!(
            with.d_in_proj() - without.d_in_proj(),
            with.micro_steps * with.nheads(),
            "one λ channel per (head, micro-step)"
        );
        let device = test_device();
        let block = without.init(&device);
        assert_eq!(block.in_proj_layout().channels_of(GroupKey::LambdaRaw), 0);
        assert_eq!(block.in_proj.weight.dims()[1], without.d_in_proj());
        #[cfg(feature = "optim")]
        {
            let names = |c: &Mamba3Config| {
                c.muon_projections()[0]
                    .sub_projections
                    .iter()
                    .map(|s| s.name.to_string())
                    .collect::<Vec<_>>()
            };
            assert!(names(&with).iter().any(|n| n == "lambda_raw"));
            assert!(!names(&without).iter().any(|n| n == "lambda_raw"));
        }
    }

    /// `forward` and an unrolled `step` must still agree with the tap removed —
    /// on both pathways, every rotation kind, and `u > 1` (where the removed tap
    /// would have straddled micro-steps).
    #[test]
    fn none_forward_matches_step() {
        let device = test_device();
        for kind in [
            RotationKind::Real1D,
            RotationKind::Complex2D,
            RotationKind::Quaternion4D,
            RotationKind::Rotor4D,
        ] {
            for micro_steps in [1, 3] {
                let config = cfg_none(kind, micro_steps);
                let model: Mamba3 = config.init(&device);
                let x = input(&config, 5);
                let (out_step, cache_step) = unrolled(&model, &x);
                let label = format!("{kind:?} u={micro_steps}");

                // Single-ssd (the default cache) delegates to the double-ssd
                // form here, so the two must agree *exactly*, not merely closely.
                let (out_single, cache_single) =
                    model.forward_single_ssd(x.clone(), None, &Mamba3SsdPath::default(), None);
                let (out_double, cache_double) =
                    model.forward_double_ssd(x.clone(), None, &Mamba3SsdPath::default(), None);
                assert_eq!(
                    max_abs_diff(out_single.clone(), out_double),
                    0.0,
                    "{label}: the pathways coincide under TapPattern::None"
                );
                assert_eq!(
                    max_abs_diff(cache_single.ssm_bhpr.clone(), cache_double.ssm_bhpr),
                    0.0,
                    "{label}: h' ≡ h under TapPattern::None"
                );

                assert!(
                    max_abs_diff(out_single, out_step) < dtype_tol(1e-4),
                    "{label}: forward vs unrolled step"
                );
                let cache_step = cache_step
                    .single_ssd()
                    .expect("a missing cache defaults to the single-ssd pathway");
                assert!(
                    max_abs_diff(cache_single.ssm_bhpr, cache_step.ssm_bhpr.clone()) < dtype_tol(1e-4),
                    "{label}: final state"
                );
                // Nothing to carry: the tap FIFO is absent, not zeroed.
                assert!(cache_step.frame.tap_fifo.is_none(), "{label}");
            }
        }
    }

    /// Gradients too: one SSD pass instead of two must still match the unrolled
    /// recurrence's, through the default recompute backward.
    #[test]
    fn none_grads_match_step() {
        let device = test_device();
        for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
            let config = cfg_none(kind, 2);
            let model: Mamba3 = config.init(&device.clone().autodiff());
            let x = input(&config, 4);
            let head = Tensor::random(x.dims(), Distribution::Normal(0.0, 1.0), &device);

            let grads = |out: Tensor<3>| {
                let loss = (out * Tensor::from_inner(head.clone())).sum();
                let grads = loss.backward();
                (
                    model
                        .in_proj
                        .weight
                        .val()
                        .grad(&grads)
                        .expect("in_proj.weight"),
                    model.dt_bias_h.val().grad(&grads).expect("dt_bias_h"),
                )
            };
            let x = Tensor::from_inner(x);
            let (fwd_w, fwd_dt) =
                grads(model.forward(x.clone(), None, Mamba3SsdPath::default(), None).0);
            let (step_w, step_dt) = grads(unrolled(&model, &x).0);
            assert!(
                max_rel_diff(step_w, fwd_w) < dtype_tol(1e-4),
                "{kind:?}: d in_proj.weight"
            );
            assert!(
                max_rel_diff(step_dt, fwd_dt) < dtype_tol(1e-4),
                "{kind:?}: d dt_bias_h"
            );
        }
    }

    /// The semantic claim: `TapPattern::None` **is** `λ ≡ 1`.
    ///
    /// `σ(30)` rounds to exactly `1.0` in f32, so a carry-over block whose `λ`
    /// columns are zeroed and whose `λ` bias is `30` has `β = 0` and `γ = Δ`
    /// exactly. Built by appending that dead sub-projection to a `None` block's
    /// own in-projection, so the two blocks agree on every other weight.
    #[test]
    fn none_equals_lambda_saturated_at_one() {
        use burn::module::Param;
        let device = test_device();
        let config = cfg()
            .with_rotation(RotationKind::Real1D)
            .with_has_proj_bias(true);
        let none = config.clone().with_tap_pattern(TapPattern::None).init(&device);
        let mut carry = config.clone().init(&device);

        // Everything but the in-projection is shared outright.
        carry.dt_bias_h = none.dt_bias_h.clone();
        carry.d_h = none.d_h.clone();
        carry.b_norm = none.b_norm.clone();
        carry.c_norm = none.c_norm.clone();
        carry.b_bias_hmr = none.b_bias_hmr.clone();
        carry.c_bias_hmr = none.c_bias_hmr.clone();
        carry.out_proj = none.out_proj.clone();

        // `Real1D` projects no rotation, so `λ` is the trailing sub-projection:
        // append a dead one (zero weights, +30 bias) to reach the carry-over
        // width.
        let nheads = config.nheads();
        let w = none.in_proj.weight.val();
        let [d_model, _] = w.dims();
        carry.in_proj.weight = Param::from_tensor(Tensor::cat(
            vec![w, Tensor::zeros([d_model, nheads], &device)],
            1,
        ));
        let b = none.in_proj.bias.as_ref().expect("has_proj_bias").val();
        carry.in_proj.bias = Some(Param::from_tensor(Tensor::cat(
            vec![b, Tensor::full([nheads], 30.0, &device)],
            0,
        )));

        let x = input(&config, 6);
        let path = Mamba3SsdPath::default();
        let (out_none, _) = none.forward(x.clone(), None, path.clone(), None);
        for (label, out) in [
            ("single", carry.forward_single_ssd(x.clone(), None, &path, None).0),
            ("double", carry.forward_double_ssd(x.clone(), None, &path, None).0),
        ] {
            let d = max_abs_diff(out_none.clone(), out);
            assert!(d < dtype_tol(1e-6), "λ≡1 carry-over ({label}) vs None: {d:.3e}");
        }
        // …and the same through the decode path.
        let d = max_abs_diff(unrolled(&none, &x).0, unrolled(&carry, &x).0);
        assert!(d < dtype_tol(1e-6), "λ≡1 carry-over vs None, stepped: {d:.3e}");
    }

    // ── TapPattern::Vertical ──────────────────────────────────────────────
    //
    // The lag-`u` pattern. `forward` (both pathways) must equal an unrolled
    // `step`, the two pathways must agree with each other, and at `u = 1` the
    // pattern must *be* the carry-over.

    fn cfg_vertical(kind: RotationKind, micro_steps: usize) -> Mamba3Config {
        cfg()
            .with_rotation(kind)
            .with_micro_steps(micro_steps)
            .with_tap_pattern(TapPattern::Vertical)
    }

    /// The lag is what the pattern is, and both implemented patterns read it
    /// from the same accessor.
    #[test]
    fn vertical_is_lag_u() {
        for u in [1, 2, 5] {
            assert_eq!(TapPattern::Vertical.tap_lag(u), u);
            assert_eq!(TapPattern::HorizontalCarryOver.tap_lag(u), 1);
            assert_eq!(TapPattern::None.tap_lag(u), 0);
        }
        assert!(TapPattern::Vertical.has_beta_tap());
        // Same λ channels as the carry-over: the pattern is an algorithm and a
        // cache layout, not a coefficient.
        assert_eq!(
            cfg_vertical(RotationKind::Complex2D, 3).d_in_proj(),
            cfg().with_micro_steps(3).d_in_proj()
        );
    }

    /// The whole point of the design: `forward` from a cache equals `step`
    /// unrolled from that same cache — on both pathways, every rotation kind,
    /// and both `u`s.
    #[test]
    fn vertical_forward_matches_step() {
        let device = test_device();
        for kind in [
            RotationKind::Real1D,
            RotationKind::Complex2D,
            RotationKind::Quaternion4D,
            RotationKind::Rotor4D,
        ] {
            for micro_steps in [2, 3] {
                let config = cfg_vertical(kind, micro_steps);
                let model: Mamba3 = config.init(&device);
                let x = input(&config, 5);
                let label = format!("{kind:?} u={micro_steps}");
                let (out_step, cache_step) = unrolled(&model, &x);
                let cache_step = cache_step
                    .single_ssd()
                    .expect("a missing cache defaults to the single-ssd pathway");

                let (out_single, cache_single) =
                    model.forward_single_ssd(x.clone(), None, &Mamba3SsdPath::default(), None);
                let (out_double, cache_double) =
                    model.forward_double_ssd(x.clone(), None, &Mamba3SsdPath::default(), None);

                for (pathway, out, ssm) in [
                    ("single", out_single, cache_single.ssm_bhpr),
                    ("double", out_double, cache_double.ssm_bhpr),
                ] {
                    let d = max_abs_diff(out, out_step.clone());
                    assert!(d < dtype_tol(1e-4), "{label} {pathway}: forward vs step: {d:.3e}");
                    let d = max_abs_diff(ssm, cache_step.ssm_bhpr.clone());
                    assert!(d < dtype_tol(1e-4), "{label} {pathway}: final state: {d:.3e}");
                }

                // The tap FIFO is the previous token, `u` slots deep.
                let fifo = cache_step.frame.tap_fifo.expect("a β tap keeps its tap FIFO");
                assert_eq!(fifo.b_bumhr.dims()[1], micro_steps, "{label}");
                assert_eq!(fifo.x_buhp.dims()[1], micro_steps, "{label}");
            }
        }
    }

    /// A split prefill must reach the same place as one call — the case the
    /// cached slots' decay pre-scaling exists for, since a lag-`u` tap's gap
    /// then straddles the boundary.
    #[test]
    fn vertical_split_prefill_matches() {
        let device = test_device();
        for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
            let config = cfg_vertical(kind, 3);
            let model: Mamba3 = config.init(&device);
            let x = input(&config, 6);
            let path = Mamba3SsdPath::default();
            let label = format!("{kind:?}");

            let (whole, whole_cache) = model.forward_single_ssd(x.clone(), None, &path, None);
            let (head, mid) = model.forward_single_ssd(x.clone().narrow(1, 0, 2), None, &path, None);
            let (tail, split_cache) =
                model.forward_single_ssd(x.narrow(1, 2, 4), Some(mid), &path, None);
            let split = Tensor::cat(vec![head, tail], 1);

            let d = max_abs_diff(whole, split);
            assert!(d < dtype_tol(1e-4), "{label}: split prefill output: {d:.3e}");
            let d = max_abs_diff(whole_cache.ssm_bhpr, split_cache.ssm_bhpr);
            assert!(d < dtype_tol(1e-4), "{label}: split prefill final state: {d:.3e}");
        }
    }

    /// MIMO end to end: the boundary seed fuses the `lag` slots with the
    /// `mimo_rank` ranks into one contraction, and the correction band fuses
    /// them the same way — both need `m > 1` to be more than a reshape.
    #[test]
    fn vertical_matches_step_with_mimo() {
        let device = test_device();
        for mimo_rank in [2, 3] {
            let config = cfg_vertical(RotationKind::Complex2D, 2).with_mimo_rank(mimo_rank);
            let model: Mamba3 = config.init(&device);
            let x = input(&config, 4);
            let label = format!("m={mimo_rank}");
            let (out_step, cache_step) = unrolled(&model, &x);
            let cache_step = cache_step.single_ssd().expect("single-ssd");
            let path = Mamba3SsdPath::default();

            for (pathway, (out, cache_ssm)) in [
                ("single", {
                    let (o, c) = model.forward_single_ssd(x.clone(), None, &path, None);
                    (o, c.ssm_bhpr)
                }),
                ("double", {
                    let (o, c) = model.forward_double_ssd(x.clone(), None, &path, None);
                    (o, c.ssm_bhpr)
                }),
            ] {
                let d = max_abs_diff(out, out_step.clone());
                assert!(d < dtype_tol(1e-4), "{label} {pathway}: forward vs step: {d:.3e}");
                let d = max_abs_diff(cache_ssm, cache_step.ssm_bhpr.clone());
                assert!(d < dtype_tol(1e-4), "{label} {pathway}: final state: {d:.3e}");
            }
        }
    }

    /// `forward` continued from a cache **`step` wrote** must equal `step` all
    /// the way — the one check that pins the two down to the *same* FIFO
    /// convention (slot order and the decay each slot carries) rather than
    /// merely to two self-consistent ones.
    #[test]
    fn vertical_forward_continues_a_stepped_cache() {
        let device = test_device();
        for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
            for micro_steps in [2, 3] {
                let config = cfg_vertical(kind, micro_steps);
                let model: Mamba3 = config.init(&device);
                let x = input(&config, 5);
                let label = format!("{kind:?} u={micro_steps}");

                // Two tokens by `step`, the remaining three by `forward`.
                let mut cache: Option<Mamba3Cache> = None;
                for t in 0..2 {
                    let (_, c) = model.step(x.clone().narrow(1, t, 1).squeeze_dim(1), cache);
                    cache = Some(c);
                }
                let stepped = cache.expect("two tokens").single_ssd().expect("single-ssd");
                let (out_tail, cache_fwd) = model.forward_single_ssd(
                    x.clone().narrow(1, 2, 3),
                    Some(stepped),
                    &Mamba3SsdPath::default(),
                    None,
                );

                let (out_all, cache_all) = unrolled(&model, &x);
                let cache_all = cache_all.single_ssd().expect("single-ssd");
                let d = max_abs_diff(out_tail, out_all.narrow(1, 2, 3));
                assert!(d < dtype_tol(1e-4), "{label}: forward after step: {d:.3e}");
                let d = max_abs_diff(cache_fwd.ssm_bhpr, cache_all.ssm_bhpr);
                assert!(d < dtype_tol(1e-4), "{label}: final state: {d:.3e}");
                let (fifo_fwd, fifo_all) =
                    (cache_fwd.frame.tap_fifo.expect("fifo"), cache_all.frame.tap_fifo.expect("fifo"));
                let d = max_abs_diff(fifo_fwd.b_bumhr, fifo_all.b_bumhr);
                assert!(d < dtype_tol(1e-4), "{label}: tap FIFO B: {d:.3e}");
                let d = max_abs_diff(fifo_fwd.x_buhp, fifo_all.x_buhp);
                assert!(d < dtype_tol(1e-4), "{label}: tap FIFO x (decay convention): {d:.3e}");
            }
        }
    }

    /// Gradients too, through the default recompute backward — the band
    /// correction and the seed are on the autodiff path like everything else.
    #[test]
    fn vertical_grads_match_step() {
        let device = test_device();
        for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
            let config = cfg_vertical(kind, 2);
            let model: Mamba3 = config.init(&device.clone().autodiff());
            let x = input(&config, 4);
            let head = Tensor::random(x.dims(), Distribution::Normal(0.0, 1.0), &device);

            let grads = |out: Tensor<3>| {
                let loss = (out * Tensor::from_inner(head.clone())).sum();
                let grads = loss.backward();
                (
                    model
                        .in_proj
                        .weight
                        .val()
                        .grad(&grads)
                        .expect("in_proj.weight"),
                    model.dt_bias_h.val().grad(&grads).expect("dt_bias_h"),
                )
            };
            let x = Tensor::from_inner(x);
            let (fwd_w, fwd_dt) =
                grads(model.forward(x.clone(), None, Mamba3SsdPath::default(), None).0);
            let (step_w, step_dt) = grads(unrolled(&model, &x).0);
            assert!(
                max_rel_diff(step_w, fwd_w) < dtype_tol(1e-4),
                "{kind:?}: d in_proj.weight"
            );
            assert!(
                max_rel_diff(step_dt, fwd_dt) < dtype_tol(1e-4),
                "{kind:?}: d dt_bias_h"
            );
        }
    }

    /// At `u = 1` "the same micro-step of the previous token" *is* "the previous
    /// position", so every lag-`u` pattern must run identically to the carry-over
    /// — the same weights, and bit for bit. For the two-tap members that is the
    /// **fold**: with both taps on one sample there is no mix to make, so they
    /// spend no `μ` channels and the pair is one tap
    /// ([`TapPattern::has_two_taps`]).
    #[test]
    fn the_lag_u_patterns_are_the_carry_over_at_one_micro_step() {
        let device = test_device();
        let base = cfg().with_rotation(RotationKind::Complex2D);
        let carry = base.clone().init(&device);
        let x = input(&base, 6);
        let path = Mamba3SsdPath::default();
        for pattern in [
            TapPattern::Vertical,
            TapPattern::VerticalPlusHorizontalInToken,
            TapPattern::VerticalPlusHorizontalCarryOver,
        ] {
            let mut folded: Mamba3 = base.clone().with_tap_pattern(pattern).init(&device);
            // A tap pattern is not a parameterisation: at `u = 1` the blocks have
            // the same weight shapes, so sharing every weight compares them.
            share_weights_except_in_proj(&carry, &mut folded);
            folded.in_proj = carry.in_proj.clone();

            for (label, a, b) in [
                (
                    "single",
                    carry.forward_single_ssd(x.clone(), None, &path, None).0,
                    folded.forward_single_ssd(x.clone(), None, &path, None).0,
                ),
                (
                    "double",
                    carry.forward_double_ssd(x.clone(), None, &path, None).0,
                    folded.forward_double_ssd(x.clone(), None, &path, None).0,
                ),
                ("step", unrolled(&carry, &x).0, unrolled(&folded, &x).0),
            ] {
                assert_eq!(
                    max_abs_diff(a, b),
                    0.0,
                    "{label}: {pattern:?} coincides with HorizontalCarryOver at u = 1"
                );
            }
        }
    }

    // ── The `HorizontalInToken` and two-tap members ───────────────────────
    //
    // `HorizontalInToken` (lag 1, closed at each token's first micro-step) and the
    // two members that add a lag-1 tap beside the lag-`u` one. The invariants
    // are the same for all three — `forward` on either pathway equals an
    // unrolled `step`, from a fresh cache and from a continued one — and the
    // degeneracy tests below are what pin their *semantics*.

    const IN_TOKEN_AND_TWO_TAP: [TapPattern; 3] = [
        TapPattern::HorizontalInToken,
        TapPattern::VerticalPlusHorizontalInToken,
        TapPattern::VerticalPlusHorizontalCarryOver,
    ];

    /// `forward` from a cache equals `step` unrolled from that same cache — both
    /// pathways, every rotation kind, both `u`s, and the tap FIFO as deep as the
    /// pattern says.
    #[test]
    fn lattice_forward_matches_step() {
        let device = test_device();
        for pattern in IN_TOKEN_AND_TWO_TAP {
            for kind in [
                RotationKind::Real1D,
                RotationKind::Complex2D,
                RotationKind::Quaternion4D,
                RotationKind::Rotor4D,
            ] {
                for micro_steps in [2, 3] {
                    let config = cfg_for(pattern, kind, micro_steps);
                    let model: Mamba3 = config.init(&device);
                    let x = input(&config, 5);
                    let label = format!("{pattern:?} {kind:?} u={micro_steps}");
                    let (out_step, cache_step) = unrolled(&model, &x);
                    let cache_step = cache_step.single_ssd().expect("single-ssd by default");

                    let (out_single, cache_single) =
                        model.forward_single_ssd(x.clone(), None, &Mamba3SsdPath::default(), None);
                    let (out_double, cache_double) =
                        model.forward_double_ssd(x.clone(), None, &Mamba3SsdPath::default(), None);

                    for (pathway, out, ssm) in [
                        ("single", out_single, cache_single.ssm_bhpr),
                        ("double", out_double, cache_double.ssm_bhpr),
                    ] {
                        let d = max_abs_diff(out, out_step.clone());
                        assert!(d < dtype_tol(1e-4), "{label} {pathway}: forward vs step: {d:.3e}");
                        let d = max_abs_diff(ssm, cache_step.ssm_bhpr.clone());
                        assert!(d < dtype_tol(1e-4), "{label} {pathway}: final state: {d:.3e}");
                    }

                    let slots = pattern.tap_slots(micro_steps);
                    let fifo = cache_step.frame.tap_fifo.expect("a β tap keeps its tap FIFO");
                    assert_eq!(fifo.b_bumhr.dims()[1], slots, "{label}");
                }
            }
        }
    }

    /// MIMO end to end: the seed fuses `(slots, ranks)` into one contraction and
    /// the band correction fuses `(taps, ranks)` the same way, so `m > 1` is not
    /// a reshape of the SISO case.
    #[test]
    fn lattice_matches_step_with_mimo() {
        let device = test_device();
        for pattern in IN_TOKEN_AND_TWO_TAP {
            for mimo_rank in [2, 3] {
                let config = cfg_for(pattern, RotationKind::Complex2D, 2).with_mimo_rank(mimo_rank);
                let model: Mamba3 = config.init(&device);
                let x = input(&config, 4);
                let label = format!("{pattern:?} m={mimo_rank}");
                let (out_step, cache_step) = unrolled(&model, &x);
                let cache_step = cache_step.single_ssd().expect("single-ssd");
                let path = Mamba3SsdPath::default();

                for (pathway, (out, ssm)) in [
                    ("single", {
                        let (o, c) = model.forward_single_ssd(x.clone(), None, &path, None);
                        (o, c.ssm_bhpr)
                    }),
                    ("double", {
                        let (o, c) = model.forward_double_ssd(x.clone(), None, &path, None);
                        (o, c.ssm_bhpr)
                    }),
                ] {
                    let d = max_abs_diff(out, out_step.clone());
                    assert!(d < dtype_tol(1e-4), "{label} {pathway}: forward vs step: {d:.3e}");
                    let d = max_abs_diff(ssm, cache_step.ssm_bhpr.clone());
                    assert!(d < dtype_tol(1e-4), "{label} {pathway}: final state: {d:.3e}");
                }
            }
        }
    }

    /// A split prefill reaches the same place as one call — the case the cached
    /// slots' decay pre-scaling and the boundary seed exist for, and the one
    /// that would notice an installment paid twice or not at all.
    #[test]
    fn lattice_split_prefill_matches() {
        let device = test_device();
        for pattern in IN_TOKEN_AND_TWO_TAP {
            for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
                let config = cfg_for(pattern, kind, 3);
                let model: Mamba3 = config.init(&device);
                let x = input(&config, 6);
                let path = Mamba3SsdPath::default();
                let label = format!("{pattern:?} {kind:?}");

                let (whole, whole_cache) = model.forward_single_ssd(x.clone(), None, &path, None);
                let (head, mid) = model.forward_single_ssd(x.clone().narrow(1, 0, 2), None, &path, None);
                let (tail, split_cache) =
                    model.forward_single_ssd(x.narrow(1, 2, 4), Some(mid), &path, None);
                let split = Tensor::cat(vec![head, tail], 1);

                let d = max_abs_diff(whole, split);
                assert!(d < dtype_tol(1e-4), "{label}: split prefill output: {d:.3e}");
                let d = max_abs_diff(whole_cache.ssm_bhpr, split_cache.ssm_bhpr);
                assert!(d < dtype_tol(1e-4), "{label}: split prefill final state: {d:.3e}");
            }
        }
    }

    /// `forward` continued from a cache **`step` wrote** — the one check that
    /// pins both to the *same* FIFO convention (slot order, and the decay each
    /// slot carries) rather than to two self-consistent ones.
    #[test]
    fn lattice_forward_continues_a_stepped_cache() {
        let device = test_device();
        for pattern in IN_TOKEN_AND_TWO_TAP {
            for micro_steps in [2, 3] {
                let config = cfg_for(pattern, RotationKind::Complex2D, micro_steps);
                let model: Mamba3 = config.init(&device);
                let x = input(&config, 5);
                let label = format!("{pattern:?} u={micro_steps}");

                let mut cache: Option<Mamba3Cache> = None;
                for t in 0..2 {
                    let (_, c) = model.step(x.clone().narrow(1, t, 1).squeeze_dim(1), cache);
                    cache = Some(c);
                }
                let stepped = cache.expect("two tokens").single_ssd().expect("single-ssd");
                let (out_tail, cache_fwd) = model.forward_single_ssd(
                    x.clone().narrow(1, 2, 3),
                    Some(stepped),
                    &Mamba3SsdPath::default(),
                    None,
                );

                let (out_all, cache_all) = unrolled(&model, &x);
                let cache_all = cache_all.single_ssd().expect("single-ssd");
                let d = max_abs_diff(out_tail, out_all.narrow(1, 2, 3));
                assert!(d < dtype_tol(1e-4), "{label}: forward after step: {d:.3e}");
                let d = max_abs_diff(cache_fwd.ssm_bhpr, cache_all.ssm_bhpr);
                assert!(d < dtype_tol(1e-4), "{label}: final state: {d:.3e}");
                let d = max_abs_diff(
                    cache_fwd.frame.tap_fifo.expect("fifo").x_buhp,
                    cache_all.frame.tap_fifo.expect("fifo").x_buhp,
                );
                assert!(d < dtype_tol(1e-4), "{label}: tap FIFO x (decay convention): {d:.3e}");
            }
        }
    }

    /// Gradients too, through the default recompute backward: the closing tap,
    /// the second mass and the extra pass are all on the autodiff path.
    #[test]
    fn lattice_grads_match_step() {
        let device = test_device();
        for pattern in IN_TOKEN_AND_TWO_TAP {
            let config = cfg_for(pattern, RotationKind::Complex2D, 2);
            let model: Mamba3 = config.init(&device.clone().autodiff());
            let x = input(&config, 4);
            let head = Tensor::random(x.dims(), Distribution::Normal(0.0, 1.0), &device);

            let grads = |out: Tensor<3>| {
                let loss = (out * Tensor::from_inner(head.clone())).sum();
                let grads = loss.backward();
                (
                    model
                        .in_proj
                        .weight
                        .val()
                        .grad(&grads)
                        .expect("in_proj.weight"),
                    model.dt_bias_h.val().grad(&grads).expect("dt_bias_h"),
                )
            };
            let x = Tensor::from_inner(x);
            let (fwd_w, fwd_dt) = grads(model.forward(x.clone(), None, Mamba3SsdPath::default(), None).0);
            let (step_w, step_dt) = grads(unrolled(&model, &x).0);
            assert!(
                max_rel_diff(step_w, fwd_w) < dtype_tol(1e-4),
                "{pattern:?}: d in_proj.weight"
            );
            assert!(
                max_rel_diff(step_dt, fwd_dt) < dtype_tol(1e-4),
                "{pattern:?}: d dt_bias_h"
            );
        }
    }

    /// Append `width` dead columns (zero weights, constant bias) to a block's
    /// in-projection — how a wider pattern is built from a narrower one's
    /// weights, so the two differ in nothing but the appended sub-projections.
    fn append_dead_sub_projections(
        block: &Mamba3,
        into: &mut Mamba3,
        width: usize,
        bias: f64,
        device: &burn::prelude::Device,
    ) {
        use burn::module::Param;
        let w = block.in_proj.weight.val();
        let [d_model, _] = w.dims();
        into.in_proj.weight = Param::from_tensor(Tensor::cat(
            vec![w, Tensor::zeros([d_model, width], device)],
            1,
        ));
        let b = block.in_proj.bias.as_ref().expect("has_proj_bias").val();
        into.in_proj.bias = Some(Param::from_tensor(Tensor::cat(
            vec![b, Tensor::full([width], bias, device)],
            0,
        )));
    }

    /// Everything but the in-projection, shared outright — the rest of a block
    /// is identical across patterns, so only the projection has to be built.
    fn share_weights_except_in_proj(from: &Mamba3, into: &mut Mamba3) {
        into.out_proj = from.out_proj.clone();
        into.dt_bias_h = from.dt_bias_h.clone();
        into.d_h = from.d_h.clone();
        into.b_norm = from.b_norm.clone();
        into.c_norm = from.c_norm.clone();
        into.b_bias_hmr = from.b_bias_hmr.clone();
        into.c_bias_hmr = from.c_bias_hmr.clone();
    }

    /// At `u = 1` every position is a token's first micro-step, so
    /// `HorizontalInToken`'s tap is closed everywhere and its mass returns to `γ`:
    /// the pattern **is** the ablation, whatever `λ` says. Built from a `None`
    /// block plus a live-but-ignored `λ` sub-projection, so the claim is about
    /// the closed tap and not about the projection.
    #[test]
    fn horizontal_in_token_is_none_at_one_micro_step() {
        let device = test_device();
        let config = cfg()
            .with_rotation(RotationKind::Real1D)
            .with_has_proj_bias(true);
        let none = config.clone().with_tap_pattern(TapPattern::None).init(&device);
        let mut in_token = config
            .clone()
            .with_tap_pattern(TapPattern::HorizontalInToken)
            .init(&device);
        share_weights_except_in_proj(&none, &mut in_token);
        append_dead_sub_projections(&none, &mut in_token, config.nheads(), 0.0, &device);

        let x = input(&config, 6);
        let path = Mamba3SsdPath::default();
        // The double pathway runs the *same* γ pass in both, plus a β pass whose
        // coefficient is exactly zero — so this one is bit for bit.
        assert_eq!(
            max_abs_diff(
                none.forward_double_ssd(x.clone(), None, &path, None).0,
                in_token.forward_double_ssd(x.clone(), None, &path, None).0,
            ),
            0.0,
            "double: HorizontalInToken is None at u = 1"
        );
        assert_eq!(
            max_abs_diff(unrolled(&none, &x).0, unrolled(&in_token, &x).0),
            0.0,
            "step: HorizontalInToken is None at u = 1"
        );
        // The single pathway does not delegate here (the pattern *has* a tap),
        // so it reassembles the same numbers by a different route.
        let d = max_abs_diff(
            none.forward_single_ssd(x.clone(), None, &path, None).0,
            in_token.forward_single_ssd(x, None, &path, None).0,
        );
        assert!(d < dtype_tol(1e-6), "single: HorizontalInToken is None at u = 1: {d:.3e}");
    }

    /// The semantic claim: closing a tap returns its mass to `γ`,
    /// i.e. `λ = 1` there — so `HorizontalInToken` is exactly
    /// `HorizontalCarryOver` with each token's first micro-step saturated, and a
    /// *submodel* of it rather than a lossy version
    /// (`info/mamba-3/trapezoid-as-integration.md` §8).
    #[test]
    fn horizontal_in_token_is_the_carry_over_with_lambda_saturated() {
        use burn::module::Param;
        let device = test_device();
        let u = 3;
        let config = cfg_for(TapPattern::HorizontalInToken, RotationKind::Real1D, u)
            .with_has_proj_bias(true);
        let in_token = config.clone().init(&device);
        let mut carry = config
            .clone()
            .with_tap_pattern(TapPattern::HorizontalCarryOver)
            .init(&device);
        share_weights_except_in_proj(&in_token, &mut carry);
        carry.in_proj = in_token.in_proj.clone();

        let x = input(&config, 4);
        let path = Mamba3SsdPath::default();
        // Not vacuous: with the same weights the two patterns disagree.
        let d = max_abs_diff(
            in_token.forward_single_ssd(x.clone(), None, &path, None).0,
            carry.forward_single_ssd(x.clone(), None, &path, None).0,
        );
        assert!(d > dtype_tol(1e-4), "the closed tap has to do something: {d:.3e}");

        // `Real1D` and one tap ⇒ the `λ` sub-projections are the trailing
        // columns, and their first `nheads` columns are micro-step 0's.
        // Saturate exactly those.
        let nheads = config.nheads();
        let start = config.d_in_proj() - u * nheads;
        let w = in_token.in_proj.weight.val();
        let [d_model, width] = w.dims();
        carry.in_proj.weight = Param::from_tensor(Tensor::cat(
            vec![
                w.clone().narrow(1, 0, start),
                Tensor::zeros([d_model, nheads], &device),
                w.narrow(1, start + nheads, width - start - nheads),
            ],
            1,
        ));
        let b = in_token.in_proj.bias.as_ref().expect("has_proj_bias").val();
        carry.in_proj.bias = Some(Param::from_tensor(Tensor::cat(
            vec![
                b.clone().narrow(0, 0, start),
                Tensor::full([nheads], 30.0, &device),
                b.narrow(0, start + nheads, width - start - nheads),
            ],
            0,
        )));

        // σ(30) is exactly 1 in f32, so the two blocks now form the same
        // coefficients and run the same code — bit for bit, on every route.
        for (label, a, b) in [
            (
                "single",
                in_token.forward_single_ssd(x.clone(), None, &path, None).0,
                carry.forward_single_ssd(x.clone(), None, &path, None).0,
            ),
            (
                "double",
                in_token.forward_double_ssd(x.clone(), None, &path, None).0,
                carry.forward_double_ssd(x.clone(), None, &path, None).0,
            ),
            ("step", unrolled(&in_token, &x).0, unrolled(&carry, &x).0),
        ] {
            assert_eq!(
                max_abs_diff(a, b),
                0.0,
                "{label}: HorizontalInToken is HorizontalCarryOver with λ = 1 at each token's start"
            );
        }
    }

    /// Several chunks, with tokens straddling their boundaries — at the default
    /// chunk length the tests above fit in one, so nothing there exercises the
    /// cross-chunk carry the single-SSD `h'` is defined by. This is where a
    /// lag-`u` pattern is most exposed: a sample's installments can be paid in a
    /// later chunk (the scale rides the chunk state) while its correction band
    /// is computed outside the kernel from the raw tensors, so the two have to
    /// agree without ever meeting. `u = 3`, chunks of 4: every token but the
    /// first straddles one.
    #[test]
    fn lattice_multi_chunk_matches_step() {
        let device = test_device();
        for pattern in [
            TapPattern::Vertical,
            TapPattern::HorizontalInToken,
            TapPattern::VerticalPlusHorizontalInToken,
            TapPattern::VerticalPlusHorizontalCarryOver,
        ] {
            let config = cfg_for(pattern, RotationKind::Complex2D, 3);
            let model: Mamba3 = config.init(&device);
            let x = input(&config, 5);
            let (out_step, cache_step) = unrolled(&model, &x);
            let cache_step = cache_step.single_ssd().expect("single-ssd");

            for path in [
                Mamba3SsdPath::Minimal(Some(4)),
                Mamba3SsdPath::SerialRecalculated(Some(4)),
            ] {
                let label = format!("{pattern:?} {path:?}");
                let (out_single, cache_single) =
                    model.forward_single_ssd(x.clone(), None, &path, None);
                let (out_double, cache_double) =
                    model.forward_double_ssd(x.clone(), None, &path, None);
                for (pathway, out, ssm) in [
                    ("single", out_single, cache_single.ssm_bhpr),
                    ("double", out_double, cache_double.ssm_bhpr),
                ] {
                    let d = max_abs_diff(out, out_step.clone());
                    assert!(d < dtype_tol(1e-4), "{label} {pathway}: forward vs step: {d:.3e}");
                    let d = max_abs_diff(ssm, cache_step.ssm_bhpr.clone());
                    assert!(d < dtype_tol(1e-4), "{label} {pathway}: final state: {d:.3e}");
                }
            }
        }
    }

    /// Non-vacuity, which the parity tests above cannot give: a second tap that
    /// were quietly dropped would leave both pathways agreeing with each other
    /// and with `step`, and only a comparison against the *one-tap* pattern they
    /// would then be notices. At `μ = σ(0) = ½` each two-tap member must differ
    /// from [`Vertical`](TapPattern::Vertical) — and from the other, since only
    /// one of them closes its lag-1 tap at each token's first micro-step.
    #[test]
    fn two_tap_patterns_actually_spend_their_second_tap() {
        let device = test_device();
        let u = 3;
        let config = cfg_for(TapPattern::Vertical, RotationKind::Real1D, u).with_has_proj_bias(true);
        let vertical = config.clone().init(&device);
        let x = input(&config, 5);
        let path = Mamba3SsdPath::default();
        let out_vertical = vertical.forward_single_ssd(x.clone(), None, &path, None).0;

        let two_tap: Vec<Tensor<3>> = [
            TapPattern::VerticalPlusHorizontalInToken,
            TapPattern::VerticalPlusHorizontalCarryOver,
        ]
        .into_iter()
        .map(|pattern| {
            let mut block = config.clone().with_tap_pattern(pattern).init(&device);
            share_weights_except_in_proj(&vertical, &mut block);
            append_dead_sub_projections(&vertical, &mut block, u * config.nheads(), 0.0, &device);
            let out = block.forward_single_ssd(x.clone(), None, &path, None).0;
            let d = max_abs_diff(out.clone(), out_vertical.clone());
            assert!(d > dtype_tol(1e-4), "{pattern:?} vs Vertical: {d:.3e}");
            out
        })
        .collect();
        let d = max_abs_diff(two_tap[0].clone(), two_tap[1].clone());
        assert!(
            d > dtype_tol(1e-4),
            "VerticalPlusHorizontalInToken and VerticalPlusHorizontalCarryOver must differ: {d:.3e}"
        );
    }

    /// The join: `VerticalPlusHorizontalCarryOver` **contains** both implemented
    /// single-tap patterns — `μ ≡ 1` is the carry-over, `μ ≡ 0` is the vertical
    /// — so `μ` is a per-(head, micro-step) interpolation between them rather
    /// than a third model. Built by appending saturated `μ` sub-projections to
    /// their shared in-projection.
    #[test]
    fn vertical_plus_carry_over_contains_both_single_tap_patterns() {
        let device = test_device();
        let u = 3;
        let config = cfg_for(TapPattern::HorizontalCarryOver, RotationKind::Real1D, u)
            .with_has_proj_bias(true);
        let carry = config.clone().init(&device);
        let mut vertical = config
            .clone()
            .with_tap_pattern(TapPattern::Vertical)
            .init(&device);
        share_weights_except_in_proj(&carry, &mut vertical);
        vertical.in_proj = carry.in_proj.clone();

        let x = input(&config, 5);
        let path = Mamba3SsdPath::default();
        for (bias, target, label, tol) in [
            (30.0, &carry, "μ = 1 is the carry-over", 0.0),
            (-30.0, &vertical, "μ = 0 is the vertical", 1e-6),
        ] {
            let mut join = config
                .clone()
                .with_tap_pattern(TapPattern::VerticalPlusHorizontalCarryOver)
                .init(&device);
            share_weights_except_in_proj(&carry, &mut join);
            // `Real1D` ⇒ the `μ` sub-projections are the trailing columns:
            // append them dead, at the saturation that selects one of the two
            // taps.
            append_dead_sub_projections(&carry, &mut join, u * config.nheads(), bias, &device);

            for (pathway, a, b) in [
                (
                    "single",
                    target.forward_single_ssd(x.clone(), None, &path, None).0,
                    join.forward_single_ssd(x.clone(), None, &path, None).0,
                ),
                (
                    "double",
                    target.forward_double_ssd(x.clone(), None, &path, None).0,
                    join.forward_double_ssd(x.clone(), None, &path, None).0,
                ),
                ("step", unrolled(target, &x).0, unrolled(&join, &x).0),
            ] {
                let d = max_abs_diff(a, b);
                assert!(d <= dtype_tol(tol), "{pathway}: {label}: {d:.3e}");
            }
        }
    }

    /// The tap buffer is what the pattern changes in the cache: none without a
    /// tap, one slot for the lag-1 patterns, `u` for the lag-`u` ones.
    #[test]
    fn tap_slots_follow_the_lag() {
        assert_eq!(TapPattern::None.tap_slots(3), 0);
        assert_eq!(TapPattern::HorizontalInToken.tap_slots(3), 1);
        assert_eq!(TapPattern::HorizontalCarryOver.tap_slots(3), 1);
        assert_eq!(TapPattern::Vertical.tap_slots(3), 3);
        assert_eq!(TapPattern::VerticalPlusHorizontalInToken.tap_slots(3), 3);
        // At `u = 1` the lag-`u` patterns *are* the lag-1 one, and the two
        // patterns that degenerate to each other agree on the buffer too.
        assert_eq!(TapPattern::Vertical.tap_slots(1), 1);
        assert_eq!(TapPattern::VerticalPlusHorizontalInToken.tap_slots(1), 1);
    }
}
