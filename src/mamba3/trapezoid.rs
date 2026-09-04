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
/// and what the crate has always done) and [`None`](Self::None) (the ablation).
/// The rest are named so the lattice has a vocabulary and so
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
    /// tap slots are `None`, `forward` makes **one** SSD call, `step` drops the
    /// second outer product, and `step_infinite`'s numerator loses its `β`
    /// rotation product. The two SSD pathways coincide here — with no second
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

    /// Whether this crate implements the pattern today —
    /// [`HorizontalCarryOver`](Self::HorizontalCarryOver) and
    /// [`None`](Self::None) do.
    pub fn is_implemented(self) -> bool {
        matches!(self, Trapezoid::HorizontalCarryOver | Trapezoid::None)
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
             Trapezoid::HorizontalCarryOver (the default) and Trapezoid::None have \
             an algorithm; see info/trapezoid-as-integration.md §9 for what the \
             others cost"
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

    /// The three unimplemented patterns are each a different algorithm *and* a
    /// different cache, so selecting one must fail at construction rather than
    /// silently run one of the two that exist.
    #[test]
    fn unimplemented_patterns_panic_at_init() {
        let device: burn::prelude::Device = Default::default();
        for pattern in [
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
                assert!(cache_step.k_state_bmhr.is_none(), "{label}");
                assert!(cache_step.v_state_bhp.is_none(), "{label}");
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

    /// `step_infinite` drops the `β` term from its numerator (and, on the
    /// quaternion kinds, the rotation product that carried it), so its closed
    /// form has to be re-checked against the recurrence it is the limit of.
    /// The decay knobs match the `step_constant` suite's.
    #[test]
    fn none_step_infinite_matches_unrolled() {
        let device: burn::prelude::Device = Default::default();
        for (kind, micro_steps, mimo_rank) in [
            (RotationKind::Real1D, 1, 1),
            (RotationKind::Real1D, 3, 1),
            (RotationKind::Complex2D, 1, 1),
            (RotationKind::Complex2D, 3, 2),
            // The non-abelian kinds have a limit only at `u = 1`.
            (RotationKind::Quaternion4D, 1, 1),
            (RotationKind::Rotor4D, 1, 2),
        ] {
            let config = cfg_none(kind, micro_steps)
                .with_mimo_rank(mimo_rank)
                .with_a_floor(1.0)
                .with_dt_limit((0.05, 5.0));
            let model: Mamba3 = config.init(&device);
            let token = Tensor::<2>::random(
                [2, config.d_model],
                Distribution::Normal(0.0, 1.0),
                &device,
            );

            let mut cache: Option<Mamba3Cache> = None;
            let mut out = None;
            for _ in 0..300 {
                let (o, c) = model.step(token.clone(), cache);
                cache = Some(c);
                out = Some(o);
            }
            let d = max_abs_diff(out.expect("300 steps"), model.step_infinite(token));
            assert!(
                d < 1e-3,
                "{kind:?} u={micro_steps} M={mimo_rank}: step_infinite vs 300 unrolled \
                 steps, max abs diff = {d:.6}"
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
