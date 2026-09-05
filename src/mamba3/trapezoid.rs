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
/// Implemented: [`HorizontalCarryOver`](Self::HorizontalCarryOver) (the default,
/// and what the crate has always done), [`Vertical`](Self::Vertical) and
/// [`None`](Self::None) (the ablation). The first two are **one algorithm read
/// at two lags** ([`tap_lag`](Self::tap_lag)); the rest are named so the lattice
/// has a vocabulary and so
/// [`Mamba3Config::init`](crate::mamba3::mamba3::Mamba3Config::init) rejects
/// them loudly ([`assert_implemented`](Self::assert_implemented)) instead of
/// silently running a different recurrence than the one asked for.
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
    /// Caches like [`HorizontalCarryOver`](Self::HorizontalCarryOver): the same
    /// one-slot lag-1 buffer, reset at each token's first micro-step. (The reset
    /// is what makes the carried value inert across a call boundary — the slot
    /// is layout, not information.)
    HorizontalReset,

    /// **Lag 1, always** — the default: one
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
            Trapezoid::Vertical | Trapezoid::VerticalPlusHorizontalReset => micro_steps,
        }
    }

    /// Whether this crate implements the pattern today —
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver),
    /// [`Vertical`](Self::Vertical) and [`None`](Self::None) do.
    pub fn is_implemented(self) -> bool {
        matches!(
            self,
            Trapezoid::HorizontalCarryOver | Trapezoid::Vertical | Trapezoid::None
        )
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
             Trapezoid::HorizontalCarryOver (the default), Trapezoid::Vertical and \
             Trapezoid::None have an algorithm; see info/trapezoid-as-integration.md \
             §9 for what the others cost"
        );
    }

    /// How many `(B, x)` tap slots a cache carries for this pattern at
    /// `micro_steps` — the depth of the FIFO the tap reads from, hence
    /// [`tap_lag`](Self::tap_lag) exactly: a lag-`L` tap needs the last `L`
    /// positions live. The SSM state is one matrix whatever this says — only the
    /// trapezoid's tap buffer changes.
    ///
    /// A slot is a *layout*, not a claim that something crosses the call
    /// boundary through it: [`HorizontalReset`](Self::HorizontalReset) shares
    /// the carry-over's slot and resets it per token, and the `u` slots of the
    /// lag-`u` patterns already cover their within-token tap, whose partner is
    /// one of those same `u` positions.
    pub fn tap_slots(self, micro_steps: usize) -> usize {
        self.tap_lag(micro_steps)
    }
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

    /// `None` is the ablation, and it is structural: no `λ` channels in the
    /// in-projection, and none of Muon's `λ` segments either.
    #[test]
    fn none_drops_the_lambda_channels() {
        let with = cfg();
        let without = cfg().with_trapezoid(Trapezoid::None);
        assert!(Trapezoid::None.is_implemented());
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

    /// The unimplemented patterns are each a different algorithm *and* a
    /// different cache, so selecting one must fail at construction rather than
    /// silently run one of the ones that exist.
    #[test]
    fn unimplemented_patterns_panic_at_init() {
        let device: burn::prelude::Device = Default::default();
        for pattern in [
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
        assert!(Trapezoid::Vertical.is_implemented());
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
    /// position", so the two implemented tapping patterns must run identically —
    /// the same weights, and bit for bit.
    #[test]
    fn vertical_is_the_carry_over_at_one_micro_step() {
        let device: burn::prelude::Device = Default::default();
        let base = cfg().with_rotation(RotationKind::Complex2D);
        let carry = base.clone().init(&device);
        let mut vertical: Mamba3 = base.clone().with_trapezoid(Trapezoid::Vertical).init(&device);
        // A tap pattern is not a parameterisation: the two blocks have the same
        // weight shapes, so sharing every weight is enough to compare them.
        vertical.in_proj = carry.in_proj.clone();
        vertical.out_proj = carry.out_proj.clone();
        vertical.dt_bias_h = carry.dt_bias_h.clone();
        vertical.d_h = carry.d_h.clone();
        vertical.b_norm = carry.b_norm.clone();
        vertical.c_norm = carry.c_norm.clone();
        vertical.b_bias_hmr = carry.b_bias_hmr.clone();
        vertical.c_bias_hmr = carry.c_bias_hmr.clone();

        let x = input(&base, 6);
        let path = Mamba3SsdPath::default();
        for (label, a, b) in [
            (
                "single",
                carry.forward_single_ssd(x.clone(), None, &path).0,
                vertical.forward_single_ssd(x.clone(), None, &path).0,
            ),
            (
                "double",
                carry.forward_double_ssd(x.clone(), None, &path).0,
                vertical.forward_double_ssd(x.clone(), None, &path).0,
            ),
            ("step", unrolled(&carry, &x).0, unrolled(&vertical, &x).0),
        ] {
            assert_eq!(
                max_abs_diff(a, b),
                0.0,
                "{label}: Vertical and HorizontalCarryOver coincide at u = 1"
            );
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
