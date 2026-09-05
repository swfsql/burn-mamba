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
//!
//! ## One rule for every member
//!
//! A pattern says which earlier samples are **admissible** at each folded
//! position; what they are *worth* is not a second thing to decide. The step's
//! mass `Δₚ` is split by two learned per-(head, micro-step) scalars, and a tap
//! that is not admissible hands its mass back one level — interior → far,
//! far → `γ`:
//!
//! ```text
//!   γ = λΔ            the right endpoint, what the read sees   (λ = σ(λ̂))
//!   ν = (1−λ)Δ        the left-endpoint mass                   (μ = σ(μ̂))
//!   νⁱⁿᵗ = ν·μ        at the lag-1 tap  — pairs micro-steps inside a token
//!   νᶠᵃʳ = ν − νⁱⁿᵗ   at the lag-`u` tap — the same micro-step of the previous token
//! ```
//!
//! so `γ + νⁱⁿᵗ + νᶠᵃʳ = Δ` at every position, whatever is gated. `λ` keeps the
//! meaning the note gives it (the splitting parameter of §4); `μ` — the one
//! channel the two-tap members add — says **which earlier sample the left
//! endpoint is**, interpolating between the two candidates. The quadrature
//! weights therefore stay non-negative and still sum to `Δ`, so §5's
//! "top-up, never a rollback" reading survives the wider tap set, and so does
//! the collapse: sample `s` carries the one scalar `γₛ + νⁱⁿᵗₛ₊₁ + νᶠᵃʳₛ₊ᵤ`.
//!
//! The fallback is what makes the gated members *submodels* rather than
//! different animals: [`HorizontalReset`](Trapezoid::HorizontalReset) is
//! [`HorizontalCarryOver`](Trapezoid::HorizontalCarryOver) with `λ = 1` at each
//! token's first micro-step (§8's own limit), and
//! [`VerticalPlusHorizontalCarryOver`](Trapezoid::VerticalPlusHorizontalCarryOver)
//! *contains* both implemented single-tap patterns at `μ ≡ 1` and `μ ≡ 0`.
//! Dropping the mass instead of handing it back would make both statements
//! false and would quietly weaken the write at `1/u` of the positions.

/// Which earlier sample(s) the trapezoid's `β` tap reads.
///
/// The choice is **structural**, not a knob on one common algorithm: it decides
/// the shift(s) the chunkwise pathways apply before chunking, the width of the
/// single-SSD γ-correction band, whether the in-projection spends `λ` and `μ`
/// channels at all, and how many `(B, x)` tap slots the cache carries
/// ([`tap_slots`](Self::tap_slots)).
///
/// The lattice is the product of the two taps' settings — the lag-1
/// (*horizontal*) one being absent, gated to within a token, or unrestricted,
/// times the lag-`u` (*vertical*) one being absent or present — and all six
/// cells exist:
///
/// ```text
///                   no vertical                 + vertical (lag u)
///   no horizontal   None                        Vertical
///   reset (j ≥ 1)   HorizontalReset             VerticalPlusHorizontalReset
///   carry (all j)   HorizontalCarryOver         VerticalPlusHorizontalCarryOver
/// ```
///
/// The left column is one algorithm read at two lags
/// ([`tap_lag`](Self::tap_lag)) — the lag-`u` cell of that column *is*
/// [`Vertical`](Self::Vertical) — and the right column adds one lag-1 tap to
/// it, mixed in by `μ` (module header). At `u = 1` the column collapses: every
/// row's two taps coincide, so [`Vertical`](Self::Vertical) *is*
/// [`HorizontalCarryOver`](Self::HorizontalCarryOver), the two-tap members are
/// it as well (they fold, and spend no `μ` channels —
/// [`has_interior_tap`](Self::has_interior_tap)), and
/// [`HorizontalReset`](Self::HorizontalReset) *is* [`None`](Self::None).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Trapezoid {
    /// **No second tap.** `λ ≡ 1`, hence `β = 0` and `γ = Δ`: the write is plain
    /// exponential-Euler, i.e. Mamba-2's (the note's §4 Lie–Trotter row).
    ///
    /// Structural in the same sense as
    /// [`RotationKind::Real1D`](crate::mamba3::rotation::RotationKind::Real1D),
    /// and nothing is paid for the absent term: the in-projection spends no `λ`
    /// channels (so Muon sees no `λ` segment), no `β` is formed, the caches'
    /// tap slots are `None`, `forward` makes **one** SSD call, and `step` drops
    /// the second outer product. The two SSD pathways coincide here — with no second
    /// pass to fuse, single-SSD's composite key scale is `γ` and its
    /// same-step correction is the whole diagonal — so
    /// [`forward_single_ssd`](crate::mamba3::mamba3::Mamba3::forward_single_ssd)
    /// runs the double-SSD code and the caches convert by field identity at
    /// *every* position, not just at boundaries.
    None,

    /// **Lag `u`, always**: the tap reads the *same* micro-step of the previous
    /// **token**, so every tap crosses a token boundary and the pattern is `u`
    /// parallel token-rate filters, one per micro-step channel. Restores the
    /// `u = 1` tap semantics at every micro-step.
    ///
    /// What is "vertical" is the **tap graph**, not the scan order: the state
    /// still runs the one flattened chain
    /// ([`crate::mamba3::product`]), which is what keeps the pattern causal and
    /// `forward` equal to an unrolled `step`. A scan that truly took every token
    /// at micro-step `0` before micro-step `1` would have to read token `t+1`
    /// before finishing token `t` — or keep `u` separate states, which is a
    /// layer stack (`burn_stack::Layers`), not a tap pattern.
    ///
    /// At token resolution the pattern has a closed form: with `Aᵗ` the token
    /// transition and `ṽᵗⱼ` micro-step `j`'s write transported to the end of its
    /// own token, the tap's transport `M₍ₜ₋₁,ⱼ₊₁₎:₍ₜ,ⱼ₎` factors through the
    /// state's own, and the whole `β` side lands **at the token boundary**:
    ///
    /// ```text
    ///   hₜ = Aₜ · ( hₜ₋₁ + Σⱼ νₜ,ⱼ ṽₜ₋₁,ⱼ ) + Σⱼ γₜ,ⱼ ṽₜ,ⱼ ,     ν = (1−λ)Δ
    /// ```
    ///
    /// i.e. **the `u = 1` trapezoid at token resolution, its left endpoint
    /// promoted from rank 1 to rank `u`** — where
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) taps only the previous
    /// token's *last* micro-step. `A` is untouched, so it is still `λ`-free
    /// (`info/trapezoid-as-integration.md` §7).
    ///
    /// Costs (§9): key scale `γₛ + (1−λₛ₊ᵤ)Δₛ₊ᵤ`; the tap is transported across
    /// its own `u`-position gap; the cache's tap buffer becomes a `u`-deep FIFO
    /// (exactly the previous token). The single-SSD same-step correction widens
    /// from the diagonal to a `u`-wide band — and that band is **exactly the
    /// token** at the only positions whose output survives (`j = u−1`), so it is
    /// one small intra-token contraction outside the chunked kernel rather than
    /// a wider mask (`crate::mamba3::single_ssd::token_band`).
    Vertical,

    /// **Lag 1, suppressed at each token's first micro-step**: taps pair
    /// micro-steps *within* a token and never cross a token boundary. Having no
    /// cross-token path at all, it cannot do the job the trapezoid was
    /// introduced for — it is a component of a pattern rather than an
    /// alternative to one.
    ///
    /// The suppressed step has no admissible earlier sample, so its whole mass
    /// returns to `γ` (`λ = 1` there, module header): each token **starts** on
    /// plain exponential-Euler and is trapezoidal inside. That is exactly the
    /// limit `info/trapezoid-as-integration.md` §8 shows is reachable by
    /// learning under [`HorizontalCarryOver`](Self::HorizontalCarryOver), which
    /// makes this member that pattern's constrained **submodel** rather than a
    /// rival to it — and is why the mass is handed back rather than dropped.
    ///
    /// Caches like [`HorizontalCarryOver`](Self::HorizontalCarryOver): the same
    /// one-slot lag-1 buffer, ignored at each token's first micro-step. (That is
    /// what makes the carried value inert across a call boundary — the slot is
    /// layout, not information.)
    HorizontalReset,

    /// **Lag 1, always** — the default: one
    /// tap per position of the folded sequence, so `1/u` of the taps cross a
    /// token boundary and `(u−1)/u` pair two projections of the same token
    /// (§8). The cache carries one `(B, x)` slot, the last micro-step of the
    /// last token.
    #[default]
    HorizontalCarryOver,

    /// [`Vertical`](Self::Vertical) **and**
    /// [`HorizontalReset`](Self::HorizontalReset) at once, the left-endpoint
    /// mass shared between them by `μ` instead of the two jobs sharing one `λ`.
    /// The first of the two members whose tap graph is not a single lag on the
    /// folded chain: on the `token × micro-step` grid, cell `(t, j)` is written
    /// by *two* taps — `(t−1, j)` above it and `(t, j−1)` beside it, the latter
    /// only for `j ≥ 1`.
    ///
    /// What is 2-D is that graph, not the algorithm. Both taps are transported
    /// across their own gap (`1` and `u` positions of the same folded chain), so
    /// §9's collapse still applies term by term: the state stays one matrix and
    /// sample `s` still carries **one** scalar, `γₛ + νⁱⁿᵗₛ₊₁ + νᶠᵃʳₛ₊ᵤ`, into
    /// every later step, hence one single-SSD pass. Costs over
    /// [`Vertical`](Self::Vertical): one more per-micro-step scalar channel
    /// (`μ`) and a third nonzero per row of the mask's banded factor — the band
    /// itself is already `u` wide, and the within-token tap never crosses a
    /// token boundary, so the cache and the boundary seed are
    /// [`Vertical`](Self::Vertical)'s unchanged. On the double-SSD pathway the
    /// two taps have different shifts and cannot share a pass, so that pathway
    /// runs **three** SSD calls; this pattern's home is the single one.
    VerticalPlusHorizontalReset,

    /// [`Vertical`](Self::Vertical) **and**
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) at once — the same two
    /// taps as [`VerticalPlusHorizontalReset`](Self::VerticalPlusHorizontalReset)
    /// with the lag-1 one left unrestricted, so at `j = 0` it reads the previous
    /// token's *last* micro-step while the lag-`u` tap reads its *first*. Both
    /// candidates for the step's left endpoint are then always available, and
    /// `μ` interpolates between them everywhere.
    ///
    /// Hence the member that **contains** the two single-tap patterns rather
    /// than sitting beside them: `μ ≡ 1` is
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) and `μ ≡ 0` is
    /// [`Vertical`](Self::Vertical), both exactly, and `μ` is per (head,
    /// micro-step) — so the choice the rest of this enum makes at configuration
    /// time is made here by descent instead.
    ///
    /// Costs over [`VerticalPlusHorizontalReset`](Self::VerticalPlusHorizontalReset):
    /// nothing in the cache (the `u`-deep FIFO's **newest** slot *is* the lag-1
    /// slot, and it carries the empty decay product), and one extra term in the
    /// single-SSD boundary seed — the lag-1 tap now crosses a call boundary,
    /// which under the reset it never does.
    VerticalPlusHorizontalCarryOver,
}

impl Trapezoid {
    /// Whether the pattern has a second (`β`) tap at all — `false` only for
    /// [`None`](Self::None). The predicate every site branches on: no tap means
    /// no `λ` channels, no `β` coefficient, no tap slots in the cache, and no
    /// previous-sample term anywhere in the recurrence.
    pub fn has_beta_tap(self) -> bool {
        self != Trapezoid::None
    }

    /// How far back the `β` tap reaches, in **folded** positions
    /// ([`crate::mamba3::product`]): `0` with no tap, `1` for the lag-1
    /// patterns, `u` for the lag-`u` ones.
    ///
    /// This is the crate's single knob for the two implemented tapping
    /// patterns — [`HorizontalCarryOver`](Self::HorizontalCarryOver) is `lag =
    /// 1` and [`Vertical`](Self::Vertical) is `lag = u`, and every site that
    /// touches the tap (the double-SSD shift, the single-SSD key scale, `step`'s
    /// FIFO depth, the cache) reads it rather than branching on the pattern. At
    /// `u = 1` they return the same `1`, which is why the two coincide there.
    ///
    /// The tap must be transported across *its own* gap, so a lag-`L` tap
    /// carries `Πᵈ⁼⁰..ᴸ⁻¹ αₚ₋ᵈ` rather than `αₚ`; that is the condition
    /// `info/trapezoid-as-integration.md` §9 shows preserves the `Δ̃` collapse,
    /// hence the single-SSD pathway.
    pub fn tap_lag(self, micro_steps: usize) -> usize {
        match self {
            Trapezoid::None => 0,
            Trapezoid::HorizontalCarryOver | Trapezoid::HorizontalReset => 1,
            Trapezoid::Vertical
            | Trapezoid::VerticalPlusHorizontalReset
            | Trapezoid::VerticalPlusHorizontalCarryOver => micro_steps,
        }
    }

    /// Whether the pattern adds a **second**, lag-1 tap beside the one at
    /// [`tap_lag`](Self::tap_lag) — the two-tap members, and only while the two
    /// lags differ.
    ///
    /// At `u = 1` they do not: both taps read the same sample, so the split is
    /// a decomposition of one coefficient and the pair **folds** back into a
    /// single tap. The fold is structural, as
    /// [`None`](Self::None)'s missing `λ` is — no `μ` channels in the
    /// in-projection, no `μ` segment for Muon, one shift, one SSD pass — which
    /// is what makes a two-tap member at `u = 1` *be*
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) rather than an
    /// expensive spelling of it.
    pub fn has_interior_tap(self, micro_steps: usize) -> bool {
        micro_steps > 1
            && matches!(
                self,
                Trapezoid::VerticalPlusHorizontalReset
                    | Trapezoid::VerticalPlusHorizontalCarryOver
            )
    }

    /// Whether the tap at [`tap_lag`](Self::tap_lag) is admissible at a token's
    /// **first** micro-step, where it is the only tap that would cross a token
    /// boundary. `false` for [`HorizontalReset`](Self::HorizontalReset) alone,
    /// whose left-endpoint mass then returns to `γ` (module header).
    pub fn far_tap_crosses_tokens(self) -> bool {
        self != Trapezoid::HorizontalReset
    }

    /// Whether the *interior* (lag-1) tap of a two-tap member is admissible at a
    /// token's first micro-step, where it would cross a token boundary — the
    /// only difference between
    /// [`VerticalPlusHorizontalCarryOver`](Self::VerticalPlusHorizontalCarryOver)
    /// (`true`) and
    /// [`VerticalPlusHorizontalReset`](Self::VerticalPlusHorizontalReset)
    /// (`false`). Where it is closed, `μ`'s share returns to the far tap.
    ///
    /// Only consulted when [`has_interior_tap`](Self::has_interior_tap).
    pub fn interior_tap_crosses_tokens(self) -> bool {
        self == Trapezoid::VerticalPlusHorizontalCarryOver
    }

    /// How many `(B, x)` tap slots a cache carries for this pattern at
    /// `micro_steps` — the depth of the FIFO the tap reads from, hence
    /// [`tap_lag`](Self::tap_lag) exactly: a lag-`L` tap needs the last `L`
    /// positions live. The SSM state is one matrix whatever this says — only the
    /// trapezoid's tap buffer changes.
    ///
    /// A slot is a *layout*, not a claim that something crosses the call
    /// boundary through it: [`HorizontalReset`](Self::HorizontalReset) shares
    /// the carry-over's slot and ignores it per token. Nor does a second tap
    /// need a second buffer — the `u`-deep FIFO's **newest** slot *is* the
    /// lag-1 slot, carrying the empty decay product, so the two-tap members read
    /// one FIFO twice.
    pub fn tap_slots(self, micro_steps: usize) -> usize {
        self.tap_lag(micro_steps)
    }
}

/// Everything the discretisation needs from the block: the tap pattern, the
/// micro-steps its gates are periodic in, and the two clamps the coefficients
/// are formed under.
///
/// Carried by [`Mamba3`](crate::mamba3::mamba3::Mamba3)
/// ([`trapezoid_spec`](crate::mamba3::mamba3::Mamba3::trapezoid_spec)) and
/// handed to `helpers::trapezoidal_coefficients`, so `forward` and `step` derive
/// the masses from **one** definition — the same reason
/// [`RotationSpec`](crate::mamba3::rotation::RotationSpec) exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrapezoidSpec {
    /// Which earlier sample(s) the `β` tap reads ([`Trapezoid`]).
    pub pattern: Trapezoid,
    /// Recurrence micro-steps per token (`u`). The gates are periodic in it,
    /// and at `u = 1` a two-tap pattern folds
    /// ([`Trapezoid::has_interior_tap`]).
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
    use crate::mamba3::mamba3::{Mamba3, Mamba3Config};
    use crate::mamba3::rotation::RotationKind;
    use crate::mamba3::ssd_path::Mamba3SsdPath;
    use burn::prelude::*;
    use burn::tensor::Distribution;
    use burn_stack::utils::test_helpers::max_abs_diff;

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
            .with_trapezoid(Trapezoid::None)
    }

    fn input(config: &Mamba3Config, tokens: usize) -> Tensor<3> {
        Tensor::random(
            [2, tokens, config.d_model],
            Distribution::Normal(0.0, 1.0),
            &Default::default(),
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
    const PATTERNS: [Trapezoid; 6] = [
        Trapezoid::None,
        Trapezoid::HorizontalReset,
        Trapezoid::HorizontalCarryOver,
        Trapezoid::Vertical,
        Trapezoid::VerticalPlusHorizontalReset,
        Trapezoid::VerticalPlusHorizontalCarryOver,
    ];

    fn cfg_for(pattern: Trapezoid, kind: RotationKind, micro_steps: usize) -> Mamba3Config {
        cfg()
            .with_rotation(kind)
            .with_micro_steps(micro_steps)
            .with_trapezoid(pattern)
    }

    /// The default must stay the pattern the crate has always run, so an
    /// untouched config keeps building.
    #[test]
    fn default_is_the_carry_over() {
        assert_eq!(Trapezoid::default(), Trapezoid::HorizontalCarryOver);
        assert_eq!(cfg().trapezoid, Trapezoid::HorizontalCarryOver);
        let device: burn::prelude::Device = Default::default();
        let _ = cfg().init(&device);
    }

    /// The lattice is closed: every member builds, and each pays for exactly the
    /// masses its taps need — `λ` unless the tap is absent, `μ` only for the
    /// two-tap members, and neither at a `u` where they would be inert.
    #[test]
    fn every_pattern_builds_and_pays_for_its_own_masses() {
        let device: burn::prelude::Device = Default::default();
        for pattern in PATTERNS {
            for micro_steps in [1, 3] {
                let config = cfg_for(pattern, RotationKind::Complex2D, micro_steps);
                let block = config.init(&device);
                let scalars = 2
                    + usize::from(pattern.has_beta_tap())
                    + usize::from(pattern.has_interior_tap(micro_steps));
                let label = format!("{pattern:?} u={micro_steps}");
                assert_eq!(
                    block.in_proj.weight.dims()[1],
                    config.d_in_proj(),
                    "{label}: in-projection width"
                );
                assert_eq!(
                    block.lambda_channels_total() + block.mu_channels_total(),
                    (scalars - 2) * micro_steps * config.nheads(),
                    "{label}: mass channels"
                );
                #[cfg(feature = "optim")]
                {
                    let names: Vec<String> = config.muon_projections()[0]
                        .segments
                        .iter()
                        .map(|s| s.name.to_string())
                        .collect();
                    assert_eq!(
                        names.iter().any(|n| n == "lambda"),
                        pattern.has_beta_tap(),
                        "{label}: λ segment"
                    );
                    assert_eq!(
                        names.iter().any(|n| n == "mu"),
                        pattern.has_interior_tap(micro_steps),
                        "{label}: μ segment"
                    );
                }
            }
        }
    }

    /// `None` is the ablation, and it is structural: no `λ` channels in the
    /// in-projection, and none of Muon's `λ` segments either.
    #[test]
    fn none_drops_the_lambda_channels() {
        let with = cfg();
        let without = cfg().with_trapezoid(Trapezoid::None);
        assert!(!Trapezoid::None.has_beta_tap());
        assert_eq!(
            with.d_in_proj() - without.d_in_proj(),
            with.micro_steps * with.nheads(),
            "one λ channel per (head, micro-step)"
        );
        let device: burn::prelude::Device = Default::default();
        let block = without.init(&device);
        assert_eq!(block.lambda_channels_total(), 0);
        assert_eq!(block.in_proj.weight.dims()[1], without.d_in_proj());
        #[cfg(feature = "optim")]
        {
            let names = |c: &Mamba3Config| {
                c.muon_projections()[0]
                    .segments
                    .iter()
                    .map(|s| s.name.to_string())
                    .collect::<Vec<_>>()
            };
            assert!(names(&with).iter().any(|n| n == "lambda"));
            assert!(!names(&without).iter().any(|n| n == "lambda"));
        }
    }

    /// `forward` and an unrolled `step` must still agree with the tap removed —
    /// on both pathways, every rotation kind, and `u > 1` (where the removed tap
    /// would have straddled micro-steps).
    #[test]
    fn none_forward_matches_step() {
        let device: burn::prelude::Device = Default::default();
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
                    model.forward_single_ssd(x.clone(), None, &Mamba3SsdPath::default());
                let (out_double, cache_double) =
                    model.forward_double_ssd(x.clone(), None, &Mamba3SsdPath::default());
                assert_eq!(
                    max_abs_diff(out_single.clone(), out_double),
                    0.0,
                    "{label}: the pathways coincide under Trapezoid::None"
                );
                assert_eq!(
                    max_abs_diff(cache_single.ssm_bhpr.clone(), cache_double.ssm_bhpr),
                    0.0,
                    "{label}: h' ≡ h under Trapezoid::None"
                );

                assert!(
                    max_abs_diff(out_single, out_step) < 1e-4,
                    "{label}: forward vs unrolled step"
                );
                let cache_step = cache_step
                    .single_ssd()
                    .expect("a missing cache defaults to the single-ssd pathway");
                assert!(
                    max_abs_diff(cache_single.ssm_bhpr, cache_step.ssm_bhpr.clone()) < 1e-4,
                    "{label}: final ssm state"
                );
                // Nothing to carry: the slots are absent, not zeroed.
                assert!(cache_step.k_state_bumhr.is_none(), "{label}");
                assert!(cache_step.v_state_buhp.is_none(), "{label}");
            }
        }
    }

    /// Gradients too: one SSD pass instead of two must still match the unrolled
    /// recurrence's, through the default recompute backward.
    #[test]
    fn none_grads_match_step() {
        let device: burn::prelude::Device = Default::default();
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
                grads(model.forward(x.clone(), None, Mamba3SsdPath::default()).0);
            let (step_w, step_dt) = grads(unrolled(&model, &x).0);
            assert!(
                max_abs_diff(fwd_w, step_w) < 1e-4,
                "{kind:?}: d in_proj.weight"
            );
            assert!(
                max_abs_diff(fwd_dt, step_dt) < 1e-4,
                "{kind:?}: d dt_bias_h"
            );
        }
    }

    /// The semantic claim: `Trapezoid::None` **is** `λ ≡ 1`.
    ///
    /// `σ(30)` rounds to exactly `1.0` in f32, so a carry-over block whose `λ`
    /// columns are zeroed and whose `λ` bias is `30` has `β = 0` and `γ = Δ`
    /// exactly. Built by appending that dead segment to a `None` block's own
    /// in-projection, so the two blocks agree on every other weight.
    #[test]
    fn none_equals_lambda_saturated_at_one() {
        use burn::module::Param;
        let device: burn::prelude::Device = Default::default();
        let config = cfg()
            .with_rotation(RotationKind::Real1D)
            .with_has_proj_bias(true);
        let none = config.clone().with_trapezoid(Trapezoid::None).init(&device);
        let mut carry = config.clone().init(&device);

        // Everything but the in-projection is shared outright.
        carry.dt_bias_h = none.dt_bias_h.clone();
        carry.d_h = none.d_h.clone();
        carry.b_norm = none.b_norm.clone();
        carry.c_norm = none.c_norm.clone();
        carry.b_bias_hmr = none.b_bias_hmr.clone();
        carry.c_bias_hmr = none.c_bias_hmr.clone();
        carry.out_proj = none.out_proj.clone();

        // `Real1D` projects no rotation, so `λ` is the trailing segment: append
        // a dead one (zero weights, +30 bias) to reach the carry-over width.
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
        let (out_none, _) = none.forward(x.clone(), None, path.clone());
        for (label, out) in [
            ("single", carry.forward_single_ssd(x.clone(), None, &path).0),
            ("double", carry.forward_double_ssd(x.clone(), None, &path).0),
        ] {
            let d = max_abs_diff(out_none.clone(), out);
            assert!(d < 1e-6, "λ≡1 carry-over ({label}) vs None: {d:.3e}");
        }
        // …and the same through the decode path.
        let d = max_abs_diff(unrolled(&none, &x).0, unrolled(&carry, &x).0);
        assert!(d < 1e-6, "λ≡1 carry-over vs None, stepped: {d:.3e}");
    }

    // ── Trapezoid::Vertical ───────────────────────────────────────────────
    //
    // The lag-`u` pattern. `forward` (both pathways) must equal an unrolled
    // `step`, the two pathways must agree with each other, and at `u = 1` the
    // pattern must *be* the carry-over.

    fn cfg_vertical(kind: RotationKind, micro_steps: usize) -> Mamba3Config {
        cfg()
            .with_rotation(kind)
            .with_micro_steps(micro_steps)
            .with_trapezoid(Trapezoid::Vertical)
    }

    /// The lag is what the pattern is, and both implemented patterns read it
    /// from the same accessor.
    #[test]
    fn vertical_is_lag_u() {
        for u in [1, 2, 5] {
            assert_eq!(Trapezoid::Vertical.tap_lag(u), u);
            assert_eq!(Trapezoid::HorizontalCarryOver.tap_lag(u), 1);
            assert_eq!(Trapezoid::None.tap_lag(u), 0);
        }
        assert!(Trapezoid::Vertical.has_beta_tap());
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
        let device: burn::prelude::Device = Default::default();
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
                    model.forward_single_ssd(x.clone(), None, &Mamba3SsdPath::default());
                let (out_double, cache_double) =
                    model.forward_double_ssd(x.clone(), None, &Mamba3SsdPath::default());

                for (pathway, out, ssm) in [
                    ("single", out_single, cache_single.ssm_bhpr),
                    ("double", out_double, cache_double.ssm_bhpr),
                ] {
                    let d = max_abs_diff(out, out_step.clone());
                    assert!(d < 1e-4, "{label} {pathway}: forward vs step: {d:.3e}");
                    let d = max_abs_diff(ssm, cache_step.ssm_bhpr.clone());
                    assert!(d < 1e-4, "{label} {pathway}: final ssm state: {d:.3e}");
                }

                // The tap FIFO is the previous token, `u` slots deep.
                let k = cache_step.k_state_bumhr.expect("a β tap keeps its slots");
                let v = cache_step.v_state_buhp.expect("a β tap keeps its slots");
                assert_eq!(k.dims()[1], micro_steps, "{label}");
                assert_eq!(v.dims()[1], micro_steps, "{label}");
            }
        }
    }

    /// A split prefill must reach the same place as one call — the case the
    /// cached slots' decay pre-scaling exists for, since a lag-`u` tap's gap
    /// then straddles the boundary.
    #[test]
    fn vertical_split_prefill_matches() {
        let device: burn::prelude::Device = Default::default();
        for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
            let config = cfg_vertical(kind, 3);
            let model: Mamba3 = config.init(&device);
            let x = input(&config, 6);
            let path = Mamba3SsdPath::default();
            let label = format!("{kind:?}");

            let (whole, whole_cache) = model.forward_single_ssd(x.clone(), None, &path);
            let (head, mid) = model.forward_single_ssd(x.clone().narrow(1, 0, 2), None, &path);
            let (tail, split_cache) =
                model.forward_single_ssd(x.narrow(1, 2, 4), Some(mid), &path);
            let split = Tensor::cat(vec![head, tail], 1);

            let d = max_abs_diff(whole, split);
            assert!(d < 1e-4, "{label}: split prefill output: {d:.3e}");
            let d = max_abs_diff(whole_cache.ssm_bhpr, split_cache.ssm_bhpr);
            assert!(d < 1e-4, "{label}: split prefill final state: {d:.3e}");
        }
    }

    /// MIMO end to end: the boundary seed fuses the `lag` slots with the
    /// `mimo_rank` ranks into one contraction, and the correction band fuses
    /// them the same way — both need `m > 1` to be more than a reshape.
    #[test]
    fn vertical_matches_step_with_mimo() {
        let device: burn::prelude::Device = Default::default();
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
                    let (o, c) = model.forward_single_ssd(x.clone(), None, &path);
                    (o, c.ssm_bhpr)
                }),
                ("double", {
                    let (o, c) = model.forward_double_ssd(x.clone(), None, &path);
                    (o, c.ssm_bhpr)
                }),
            ] {
                let d = max_abs_diff(out, out_step.clone());
                assert!(d < 1e-4, "{label} {pathway}: forward vs step: {d:.3e}");
                let d = max_abs_diff(cache_ssm, cache_step.ssm_bhpr.clone());
                assert!(d < 1e-4, "{label} {pathway}: final state: {d:.3e}");
            }
        }
    }

    /// `forward` continued from a cache **`step` wrote** must equal `step` all
    /// the way — the one check that pins the two down to the *same* FIFO
    /// convention (slot order and the decay each slot carries) rather than
    /// merely to two self-consistent ones.
    #[test]
    fn vertical_forward_continues_a_stepped_cache() {
        let device: burn::prelude::Device = Default::default();
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
                );

                let (out_all, cache_all) = unrolled(&model, &x);
                let cache_all = cache_all.single_ssd().expect("single-ssd");
                let d = max_abs_diff(out_tail, out_all.narrow(1, 2, 3));
                assert!(d < 1e-4, "{label}: forward after step: {d:.3e}");
                let d = max_abs_diff(cache_fwd.ssm_bhpr, cache_all.ssm_bhpr);
                assert!(d < 1e-4, "{label}: final state: {d:.3e}");
                let d = max_abs_diff(
                    cache_fwd.k_state_bumhr.expect("slots"),
                    cache_all.k_state_bumhr.expect("slots"),
                );
                assert!(d < 1e-4, "{label}: tap FIFO B: {d:.3e}");
                let d = max_abs_diff(
                    cache_fwd.v_state_buhp.expect("slots"),
                    cache_all.v_state_buhp.expect("slots"),
                );
                assert!(d < 1e-4, "{label}: tap FIFO x (decay convention): {d:.3e}");
            }
        }
    }

    /// Gradients too, through the default recompute backward — the band
    /// correction and the seed are on the autodiff path like everything else.
    #[test]
    fn vertical_grads_match_step() {
        let device: burn::prelude::Device = Default::default();
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
                grads(model.forward(x.clone(), None, Mamba3SsdPath::default()).0);
            let (step_w, step_dt) = grads(unrolled(&model, &x).0);
            assert!(
                max_abs_diff(fwd_w, step_w) < 1e-4,
                "{kind:?}: d in_proj.weight"
            );
            assert!(
                max_abs_diff(fwd_dt, step_dt) < 1e-4,
                "{kind:?}: d dt_bias_h"
            );
        }
    }

    /// At `u = 1` "the same micro-step of the previous token" *is* "the previous
    /// position", so every lag-`u` pattern must run identically to the carry-over
    /// — the same weights, and bit for bit. For the two-tap members that is the
    /// **fold**: with both taps on one sample there is no mix to make, so they
    /// spend no `μ` channels and the pair is one tap
    /// ([`Trapezoid::has_interior_tap`]).
    #[test]
    fn the_lag_u_patterns_are_the_carry_over_at_one_micro_step() {
        let device: burn::prelude::Device = Default::default();
        let base = cfg().with_rotation(RotationKind::Complex2D);
        let carry = base.clone().init(&device);
        let x = input(&base, 6);
        let path = Mamba3SsdPath::default();
        for pattern in [
            Trapezoid::Vertical,
            Trapezoid::VerticalPlusHorizontalReset,
            Trapezoid::VerticalPlusHorizontalCarryOver,
        ] {
            let mut folded: Mamba3 = base.clone().with_trapezoid(pattern).init(&device);
            // A tap pattern is not a parameterisation: at `u = 1` the blocks have
            // the same weight shapes, so sharing every weight compares them.
            share_weights_except_in_proj(&carry, &mut folded);
            folded.in_proj = carry.in_proj.clone();

            for (label, a, b) in [
                (
                    "single",
                    carry.forward_single_ssd(x.clone(), None, &path).0,
                    folded.forward_single_ssd(x.clone(), None, &path).0,
                ),
                (
                    "double",
                    carry.forward_double_ssd(x.clone(), None, &path).0,
                    folded.forward_double_ssd(x.clone(), None, &path).0,
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

    // ── The gated and two-tap members ─────────────────────────────────────
    //
    // `HorizontalReset` (lag 1, closed at each token's first micro-step) and the
    // two members that add a lag-1 tap beside the lag-`u` one. The invariants
    // are the same for all three — `forward` on either pathway equals an
    // unrolled `step`, from a fresh cache and from a continued one — and the
    // degeneracy tests below are what pin their *semantics*.

    const GATED_AND_TWO_TAP: [Trapezoid; 3] = [
        Trapezoid::HorizontalReset,
        Trapezoid::VerticalPlusHorizontalReset,
        Trapezoid::VerticalPlusHorizontalCarryOver,
    ];

    /// `forward` from a cache equals `step` unrolled from that same cache — both
    /// pathways, every rotation kind, both `u`s, and the tap FIFO as deep as the
    /// pattern says.
    #[test]
    fn lattice_forward_matches_step() {
        let device: burn::prelude::Device = Default::default();
        for pattern in GATED_AND_TWO_TAP {
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
                        model.forward_single_ssd(x.clone(), None, &Mamba3SsdPath::default());
                    let (out_double, cache_double) =
                        model.forward_double_ssd(x.clone(), None, &Mamba3SsdPath::default());

                    for (pathway, out, ssm) in [
                        ("single", out_single, cache_single.ssm_bhpr),
                        ("double", out_double, cache_double.ssm_bhpr),
                    ] {
                        let d = max_abs_diff(out, out_step.clone());
                        assert!(d < 1e-4, "{label} {pathway}: forward vs step: {d:.3e}");
                        let d = max_abs_diff(ssm, cache_step.ssm_bhpr.clone());
                        assert!(d < 1e-4, "{label} {pathway}: final ssm state: {d:.3e}");
                    }

                    let slots = pattern.tap_slots(micro_steps);
                    let k = cache_step.k_state_bumhr.expect("a β tap keeps its slots");
                    assert_eq!(k.dims()[1], slots, "{label}");
                }
            }
        }
    }

    /// MIMO end to end: the seed fuses `(slots, ranks)` into one contraction and
    /// the band correction fuses `(taps, ranks)` the same way, so `m > 1` is not
    /// a reshape of the SISO case.
    #[test]
    fn lattice_matches_step_with_mimo() {
        let device: burn::prelude::Device = Default::default();
        for pattern in GATED_AND_TWO_TAP {
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
                        let (o, c) = model.forward_single_ssd(x.clone(), None, &path);
                        (o, c.ssm_bhpr)
                    }),
                    ("double", {
                        let (o, c) = model.forward_double_ssd(x.clone(), None, &path);
                        (o, c.ssm_bhpr)
                    }),
                ] {
                    let d = max_abs_diff(out, out_step.clone());
                    assert!(d < 1e-4, "{label} {pathway}: forward vs step: {d:.3e}");
                    let d = max_abs_diff(ssm, cache_step.ssm_bhpr.clone());
                    assert!(d < 1e-4, "{label} {pathway}: final state: {d:.3e}");
                }
            }
        }
    }

    /// A split prefill reaches the same place as one call — the case the cached
    /// slots' decay pre-scaling and the boundary seed exist for, and the one
    /// that would notice an installment paid twice or not at all.
    #[test]
    fn lattice_split_prefill_matches() {
        let device: burn::prelude::Device = Default::default();
        for pattern in GATED_AND_TWO_TAP {
            for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
                let config = cfg_for(pattern, kind, 3);
                let model: Mamba3 = config.init(&device);
                let x = input(&config, 6);
                let path = Mamba3SsdPath::default();
                let label = format!("{pattern:?} {kind:?}");

                let (whole, whole_cache) = model.forward_single_ssd(x.clone(), None, &path);
                let (head, mid) = model.forward_single_ssd(x.clone().narrow(1, 0, 2), None, &path);
                let (tail, split_cache) =
                    model.forward_single_ssd(x.narrow(1, 2, 4), Some(mid), &path);
                let split = Tensor::cat(vec![head, tail], 1);

                let d = max_abs_diff(whole, split);
                assert!(d < 1e-4, "{label}: split prefill output: {d:.3e}");
                let d = max_abs_diff(whole_cache.ssm_bhpr, split_cache.ssm_bhpr);
                assert!(d < 1e-4, "{label}: split prefill final state: {d:.3e}");
            }
        }
    }

    /// `forward` continued from a cache **`step` wrote** — the one check that
    /// pins both to the *same* FIFO convention (slot order, and the decay each
    /// slot carries) rather than to two self-consistent ones.
    #[test]
    fn lattice_forward_continues_a_stepped_cache() {
        let device: burn::prelude::Device = Default::default();
        for pattern in GATED_AND_TWO_TAP {
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
                );

                let (out_all, cache_all) = unrolled(&model, &x);
                let cache_all = cache_all.single_ssd().expect("single-ssd");
                let d = max_abs_diff(out_tail, out_all.narrow(1, 2, 3));
                assert!(d < 1e-4, "{label}: forward after step: {d:.3e}");
                let d = max_abs_diff(cache_fwd.ssm_bhpr, cache_all.ssm_bhpr);
                assert!(d < 1e-4, "{label}: final state: {d:.3e}");
                let d = max_abs_diff(
                    cache_fwd.v_state_buhp.expect("slots"),
                    cache_all.v_state_buhp.expect("slots"),
                );
                assert!(d < 1e-4, "{label}: tap FIFO x (decay convention): {d:.3e}");
            }
        }
    }

    /// Gradients too, through the default recompute backward: the gate, the
    /// second mass and the extra pass are all on the autodiff path.
    #[test]
    fn lattice_grads_match_step() {
        let device: burn::prelude::Device = Default::default();
        for pattern in GATED_AND_TWO_TAP {
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
            let (fwd_w, fwd_dt) = grads(model.forward(x.clone(), None, Mamba3SsdPath::default()).0);
            let (step_w, step_dt) = grads(unrolled(&model, &x).0);
            assert!(
                max_abs_diff(fwd_w, step_w) < 1e-4,
                "{pattern:?}: d in_proj.weight"
            );
            assert!(
                max_abs_diff(fwd_dt, step_dt) < 1e-4,
                "{pattern:?}: d dt_bias_h"
            );
        }
    }

    /// Append `width` dead columns (zero weights, constant bias) to a block's
    /// in-projection — how a wider pattern is built from a narrower one's
    /// weights, so the two differ in nothing but the appended segment.
    fn append_dead_segment(
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
    /// `HorizontalReset`'s tap is closed everywhere and its mass returns to `γ`:
    /// the pattern **is** the ablation, whatever `λ` says. Built from a `None`
    /// block plus a live-but-ignored `λ` segment, so the claim is about the gate
    /// and not about the projection.
    #[test]
    fn horizontal_reset_is_none_at_one_micro_step() {
        let device: burn::prelude::Device = Default::default();
        let config = cfg()
            .with_rotation(RotationKind::Real1D)
            .with_has_proj_bias(true);
        let none = config.clone().with_trapezoid(Trapezoid::None).init(&device);
        let mut reset = config
            .clone()
            .with_trapezoid(Trapezoid::HorizontalReset)
            .init(&device);
        share_weights_except_in_proj(&none, &mut reset);
        append_dead_segment(&none, &mut reset, config.nheads(), 0.0, &device);

        let x = input(&config, 6);
        let path = Mamba3SsdPath::default();
        // The double pathway runs the *same* γ pass in both, plus a β pass whose
        // coefficient is exactly zero — so this one is bit for bit.
        assert_eq!(
            max_abs_diff(
                none.forward_double_ssd(x.clone(), None, &path).0,
                reset.forward_double_ssd(x.clone(), None, &path).0,
            ),
            0.0,
            "double: HorizontalReset is None at u = 1"
        );
        assert_eq!(
            max_abs_diff(unrolled(&none, &x).0, unrolled(&reset, &x).0),
            0.0,
            "step: HorizontalReset is None at u = 1"
        );
        // The single pathway does not delegate here (the pattern *has* a tap),
        // so it reassembles the same numbers by a different route.
        let d = max_abs_diff(
            none.forward_single_ssd(x.clone(), None, &path).0,
            reset.forward_single_ssd(x, None, &path).0,
        );
        assert!(d < 1e-6, "single: HorizontalReset is None at u = 1: {d:.3e}");
    }

    /// The semantic claim for the gate: closing a tap returns its mass to `γ`,
    /// i.e. `λ = 1` there — so `HorizontalReset` is exactly
    /// `HorizontalCarryOver` with each token's first micro-step saturated, and a
    /// *submodel* of it rather than a lossy version
    /// (`info/trapezoid-as-integration.md` §8).
    #[test]
    fn horizontal_reset_is_the_carry_over_with_lambda_saturated() {
        use burn::module::Param;
        let device: burn::prelude::Device = Default::default();
        let u = 3;
        let config = cfg_for(Trapezoid::HorizontalReset, RotationKind::Real1D, u)
            .with_has_proj_bias(true);
        let reset = config.clone().init(&device);
        let mut carry = config
            .clone()
            .with_trapezoid(Trapezoid::HorizontalCarryOver)
            .init(&device);
        share_weights_except_in_proj(&reset, &mut carry);
        carry.in_proj = reset.in_proj.clone();

        let x = input(&config, 4);
        let path = Mamba3SsdPath::default();
        // Not vacuous: with the same weights the two patterns disagree.
        let d = max_abs_diff(
            reset.forward_single_ssd(x.clone(), None, &path).0,
            carry.forward_single_ssd(x.clone(), None, &path).0,
        );
        assert!(d > 1e-4, "the gate has to do something: {d:.3e}");

        // `Real1D` and one tap ⇒ `λ` is the trailing segment, and its first
        // `nheads` columns are micro-step 0's. Saturate exactly those.
        let nheads = config.nheads();
        let start = config.d_in_proj() - u * nheads;
        let w = reset.in_proj.weight.val();
        let [d_model, width] = w.dims();
        carry.in_proj.weight = Param::from_tensor(Tensor::cat(
            vec![
                w.clone().narrow(1, 0, start),
                Tensor::zeros([d_model, nheads], &device),
                w.narrow(1, start + nheads, width - start - nheads),
            ],
            1,
        ));
        let b = reset.in_proj.bias.as_ref().expect("has_proj_bias").val();
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
                reset.forward_single_ssd(x.clone(), None, &path).0,
                carry.forward_single_ssd(x.clone(), None, &path).0,
            ),
            (
                "double",
                reset.forward_double_ssd(x.clone(), None, &path).0,
                carry.forward_double_ssd(x.clone(), None, &path).0,
            ),
            ("step", unrolled(&reset, &x).0, unrolled(&carry, &x).0),
        ] {
            assert_eq!(
                max_abs_diff(a, b),
                0.0,
                "{label}: the reset is the carry-over with λ = 1 at each token's start"
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
        let device: burn::prelude::Device = Default::default();
        for pattern in [
            Trapezoid::Vertical,
            Trapezoid::HorizontalReset,
            Trapezoid::VerticalPlusHorizontalReset,
            Trapezoid::VerticalPlusHorizontalCarryOver,
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
                    model.forward_single_ssd(x.clone(), None, &path);
                let (out_double, cache_double) =
                    model.forward_double_ssd(x.clone(), None, &path);
                for (pathway, out, ssm) in [
                    ("single", out_single, cache_single.ssm_bhpr),
                    ("double", out_double, cache_double.ssm_bhpr),
                ] {
                    let d = max_abs_diff(out, out_step.clone());
                    assert!(d < 1e-4, "{label} {pathway}: forward vs step: {d:.3e}");
                    let d = max_abs_diff(ssm, cache_step.ssm_bhpr.clone());
                    assert!(d < 1e-4, "{label} {pathway}: final ssm state: {d:.3e}");
                }
            }
        }
    }

    /// Non-vacuity, which the parity tests above cannot give: a second tap that
    /// were quietly dropped would leave both pathways agreeing with each other
    /// and with `step`, and only a comparison against the *one-tap* pattern they
    /// would then be notices. At `μ = σ(0) = ½` each two-tap member must differ
    /// from [`Vertical`](Trapezoid::Vertical) — and from the other, since their
    /// gates part company at each token's first micro-step.
    #[test]
    fn two_tap_patterns_actually_spend_their_second_tap() {
        let device: burn::prelude::Device = Default::default();
        let u = 3;
        let config = cfg_for(Trapezoid::Vertical, RotationKind::Real1D, u).with_has_proj_bias(true);
        let vertical = config.clone().init(&device);
        let x = input(&config, 5);
        let path = Mamba3SsdPath::default();
        let out_vertical = vertical.forward_single_ssd(x.clone(), None, &path).0;

        let two_tap: Vec<Tensor<3>> = [
            Trapezoid::VerticalPlusHorizontalReset,
            Trapezoid::VerticalPlusHorizontalCarryOver,
        ]
        .into_iter()
        .map(|pattern| {
            let mut block = config.clone().with_trapezoid(pattern).init(&device);
            share_weights_except_in_proj(&vertical, &mut block);
            append_dead_segment(&vertical, &mut block, u * config.nheads(), 0.0, &device);
            let out = block.forward_single_ssd(x.clone(), None, &path).0;
            let d = max_abs_diff(out.clone(), out_vertical.clone());
            assert!(d > 1e-4, "{pattern:?} vs Vertical: {d:.3e}");
            out
        })
        .collect();
        let d = max_abs_diff(two_tap[0].clone(), two_tap[1].clone());
        assert!(d > 1e-4, "the reset and the carry-over of one lag: {d:.3e}");
    }

    /// The join: `VerticalPlusHorizontalCarryOver` **contains** both implemented
    /// single-tap patterns — `μ ≡ 1` is the carry-over, `μ ≡ 0` is the vertical
    /// — so `μ` is a per-(head, micro-step) interpolation between them rather
    /// than a third model. Built by appending a saturated `μ` segment to their
    /// shared in-projection.
    #[test]
    fn vertical_plus_carry_over_contains_both_single_tap_patterns() {
        let device: burn::prelude::Device = Default::default();
        let u = 3;
        let config = cfg_for(Trapezoid::HorizontalCarryOver, RotationKind::Real1D, u)
            .with_has_proj_bias(true);
        let carry = config.clone().init(&device);
        let mut vertical = config
            .clone()
            .with_trapezoid(Trapezoid::Vertical)
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
                .with_trapezoid(Trapezoid::VerticalPlusHorizontalCarryOver)
                .init(&device);
            share_weights_except_in_proj(&carry, &mut join);
            // `Real1D` ⇒ `μ` is the trailing segment: append it dead, at the
            // saturation that selects one of the two taps.
            append_dead_segment(&carry, &mut join, u * config.nheads(), bias, &device);

            for (pathway, a, b) in [
                (
                    "single",
                    target.forward_single_ssd(x.clone(), None, &path).0,
                    join.forward_single_ssd(x.clone(), None, &path).0,
                ),
                (
                    "double",
                    target.forward_double_ssd(x.clone(), None, &path).0,
                    join.forward_double_ssd(x.clone(), None, &path).0,
                ),
                ("step", unrolled(target, &x).0, unrolled(&join, &x).0),
            ] {
                let d = max_abs_diff(a, b);
                assert!(d <= tol, "{pathway}: {label}: {d:.3e}");
            }
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
