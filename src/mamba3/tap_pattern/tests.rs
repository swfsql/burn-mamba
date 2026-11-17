use super::*;
use crate::mamba3::cache::Mamba3Cache;
use crate::mamba3::in_proj_layout::GroupKey;
use crate::mamba3::mamba3::{Mamba3, Mamba3Config};
use crate::mamba3::rotation::RotationKind;
use crate::mamba3::scan_options::Mamba3ScanOptions;
use crate::ssd_path::SsdPath;
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

/// The lattice is closed: every member builds. Each member pays for exactly the
/// masses that its taps need. It has `λ` unless the tap is absent, and `μ` only
/// as a two-tap member. It has neither at a `u` where they would be inert.
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

/// `forward` and an unrolled `step` must still agree with the tap removed. The
/// test covers both pathways, every rotation kind, and `u > 1`, where the
/// removed tap would have straddled micro-steps.
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
                model.forward_single_ssd(x.clone(), None, &Mamba3ScanOptions::default(), None);
            let (out_double, cache_double) =
                model.forward_double_ssd(x.clone(), None, &Mamba3ScanOptions::default(), None);
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
            grads(model.forward(x.clone(), None, Mamba3ScanOptions::default(), None).0);
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
/// `σ(30)` rounds to exactly `1.0` in f32. So a carry-over block with zero `λ`
/// columns and a `λ` bias of `30` has `β = 0` and `γ = Δ` exactly. Built by
/// appending that dead sub-projection to a `None` block's
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
    let scan_options = Mamba3ScanOptions::default();
    let (out_none, _) = none.forward(x.clone(), None, scan_options.clone(), None);
    for (label, out) in [
        ("single", carry.forward_single_ssd(x.clone(), None, &scan_options, None).0),
        ("double", carry.forward_double_ssd(x.clone(), None, &scan_options, None).0),
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
/// unrolled from that same cache. The test covers both pathways, every
/// rotation kind, and both `u`s.
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
                model.forward_single_ssd(x.clone(), None, &Mamba3ScanOptions::default(), None);
            let (out_double, cache_double) =
                model.forward_double_ssd(x.clone(), None, &Mamba3ScanOptions::default(), None);

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

/// A split prefill must reach the same place as one call. The decay
/// pre-scaling of the cached slots exists for this case, because the gap of a
/// lag-`u` tap then straddles the boundary.
#[test]
fn vertical_split_prefill_matches() {
    let device = test_device();
    for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
        let config = cfg_vertical(kind, 3);
        let model: Mamba3 = config.init(&device);
        let x = input(&config, 6);
        let scan_options = Mamba3ScanOptions::default();
        let label = format!("{kind:?}");

        let (whole, whole_cache) = model.forward_single_ssd(x.clone(), None, &scan_options, None);
        let (head, mid) = model.forward_single_ssd(x.clone().narrow(1, 0, 2), None, &scan_options, None);
        let (tail, split_cache) =
            model.forward_single_ssd(x.narrow(1, 2, 4), Some(mid), &scan_options, None);
        let split = Tensor::cat(vec![head, tail], 1);

        let d = max_abs_diff(whole, split);
        assert!(d < dtype_tol(1e-4), "{label}: split prefill output: {d:.3e}");
        let d = max_abs_diff(whole_cache.ssm_bhpr, split_cache.ssm_bhpr);
        assert!(d < dtype_tol(1e-4), "{label}: split prefill final state: {d:.3e}");
    }
}

/// MIMO end to end: the boundary seed fuses the `lag` slots with the
/// `mimo_rank` ranks into one contraction. The correction band fuses them the
/// same way. Both need `m > 1` to be more than a reshape.
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
        let scan_options = Mamba3ScanOptions::default();

        for (pathway, (out, cache_ssm)) in [
            ("single", {
                let (o, c) = model.forward_single_ssd(x.clone(), None, &scan_options, None);
                (o, c.ssm_bhpr)
            }),
            ("double", {
                let (o, c) = model.forward_double_ssd(x.clone(), None, &scan_options, None);
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
/// the way. This is the one check that pins the two to the *same* FIFO
/// convention, not merely to two self-consistent ones. The convention is the
/// slot order and the decay that each slot carries.
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
                &Mamba3ScanOptions::default(),
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
            grads(model.forward(x.clone(), None, Mamba3ScanOptions::default(), None).0);
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
/// position". So every lag-`u` pattern must run identically to the carry-over,
/// with the same weights and bit for bit. For the two-tap members, that is the
/// **fold** ([`TapPattern::has_two_taps`]). Both taps are on one sample, so
/// there is no mix to make. Thus the two-tap members spend no `μ` channels, and
/// the pair is one tap.
#[test]
fn the_lag_u_patterns_are_the_carry_over_at_one_micro_step() {
    let device = test_device();
    let base = cfg().with_rotation(RotationKind::Complex2D);
    let carry = base.clone().init(&device);
    let x = input(&base, 6);
    let scan_options = Mamba3ScanOptions::default();
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
                carry.forward_single_ssd(x.clone(), None, &scan_options, None).0,
                folded.forward_single_ssd(x.clone(), None, &scan_options, None).0,
            ),
            (
                "double",
                carry.forward_double_ssd(x.clone(), None, &scan_options, None).0,
                folded.forward_double_ssd(x.clone(), None, &scan_options, None).0,
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
// are the same for all three. In each, `forward` on either pathway equals an
// unrolled `step`, from a fresh cache and from a continued one. The
// degeneracy tests below pin their *semantics*.

const IN_TOKEN_AND_TWO_TAP: [TapPattern; 3] = [
    TapPattern::HorizontalInToken,
    TapPattern::VerticalPlusHorizontalInToken,
    TapPattern::VerticalPlusHorizontalCarryOver,
];

/// `forward` from a cache equals `step` unrolled from that same cache. The test
/// covers both pathways, every rotation kind and both `u`s. It also checks that
/// the tap FIFO is as deep as the pattern says.
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
                    model.forward_single_ssd(x.clone(), None, &Mamba3ScanOptions::default(), None);
                let (out_double, cache_double) =
                    model.forward_double_ssd(x.clone(), None, &Mamba3ScanOptions::default(), None);

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

/// MIMO end to end: the seed fuses `(slots, ranks)` into one contraction. The
/// correction band fuses `(taps, ranks)` the same way. So `m > 1` is not a
/// reshape of the SISO case.
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
            let scan_options = Mamba3ScanOptions::default();

            for (pathway, (out, ssm)) in [
                ("single", {
                    let (o, c) = model.forward_single_ssd(x.clone(), None, &scan_options, None);
                    (o, c.ssm_bhpr)
                }),
                ("double", {
                    let (o, c) = model.forward_double_ssd(x.clone(), None, &scan_options, None);
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

/// A split prefill reaches the same place as one call. The decay pre-scaling of
/// the cached slots and the boundary seed exist for this case. This test would
/// also notice an installment paid twice or not at all.
#[test]
fn lattice_split_prefill_matches() {
    let device = test_device();
    for pattern in IN_TOKEN_AND_TWO_TAP {
        for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
            let config = cfg_for(pattern, kind, 3);
            let model: Mamba3 = config.init(&device);
            let x = input(&config, 6);
            let scan_options = Mamba3ScanOptions::default();
            let label = format!("{pattern:?} {kind:?}");

            let (whole, whole_cache) = model.forward_single_ssd(x.clone(), None, &scan_options, None);
            let (head, mid) = model.forward_single_ssd(x.clone().narrow(1, 0, 2), None, &scan_options, None);
            let (tail, split_cache) =
                model.forward_single_ssd(x.narrow(1, 2, 4), Some(mid), &scan_options, None);
            let split = Tensor::cat(vec![head, tail], 1);

            let d = max_abs_diff(whole, split);
            assert!(d < dtype_tol(1e-4), "{label}: split prefill output: {d:.3e}");
            let d = max_abs_diff(whole_cache.ssm_bhpr, split_cache.ssm_bhpr);
            assert!(d < dtype_tol(1e-4), "{label}: split prefill final state: {d:.3e}");
        }
    }
}

/// `forward` continued from a cache **`step` wrote**. This is the one check
/// that pins both to the *same* FIFO convention, not to two self-consistent
/// ones. The convention is the slot order and the decay that each slot carries.
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
                &Mamba3ScanOptions::default(),
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
        let (fwd_w, fwd_dt) = grads(model.forward(x.clone(), None, Mamba3ScanOptions::default(), None).0);
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
/// in-projection. This builds a wider pattern from the weights of a narrower
/// one. The two then differ only in the appended sub-projections.
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

/// At `u = 1` every position is a token's first micro-step. So the tap of
/// `HorizontalInToken` is closed everywhere, and its mass returns to `γ`. The
/// pattern **is** the ablation, whatever `λ` says. Built from a `None`
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
    let scan_options = Mamba3ScanOptions::default();
    // The double pathway runs the *same* γ pass in both, plus a β pass whose
    // coefficient is exactly zero. So this one is bit for bit.
    assert_eq!(
        max_abs_diff(
            none.forward_double_ssd(x.clone(), None, &scan_options, None).0,
            in_token.forward_double_ssd(x.clone(), None, &scan_options, None).0,
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
        none.forward_single_ssd(x.clone(), None, &scan_options, None).0,
        in_token.forward_single_ssd(x, None, &scan_options, None).0,
    );
    assert!(d < dtype_tol(1e-6), "single: HorizontalInToken is None at u = 1: {d:.3e}");
}

/// The semantic claim: a closed tap returns its mass to `γ`, which means
/// `λ = 1` there. So `HorizontalInToken` is exactly `HorizontalCarryOver` with
/// each token's first micro-step saturated. It is a *submodel* of that pattern,
/// not a lossy version (`info/mamba-3/trapezoid-as-integration.md` §8).
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
    let scan_options = Mamba3ScanOptions::default();
    // Not vacuous: with the same weights the two patterns disagree.
    let d = max_abs_diff(
        in_token.forward_single_ssd(x.clone(), None, &scan_options, None).0,
        carry.forward_single_ssd(x.clone(), None, &scan_options, None).0,
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
    // coefficients and run the same code. They agree bit for bit, on every
    // route.
    for (label, a, b) in [
        (
            "single",
            in_token.forward_single_ssd(x.clone(), None, &scan_options, None).0,
            carry.forward_single_ssd(x.clone(), None, &scan_options, None).0,
        ),
        (
            "double",
            in_token.forward_double_ssd(x.clone(), None, &scan_options, None).0,
            carry.forward_double_ssd(x.clone(), None, &scan_options, None).0,
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

/// Several chunks, with tokens straddling their boundaries. At the default
/// chunk length, the tests above fit in one chunk. So nothing there exercises
/// the cross-chunk carry that defines the single-SSD `h'`.
///
/// This is where a lag-`u` pattern is most exposed. A later chunk can pay the
/// installments of a sample (the scale rides the chunk state). But the
/// correction band of that sample comes from the raw tensors, outside the
/// kernel. So the two have to agree without ever meeting. `u = 3`, chunks of 4:
/// every token but the first straddles one.
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

        for scan_options in [
            Mamba3ScanOptions::new(SsdPath::Minimal).with_chunk_len(Some(4)),
            Mamba3ScanOptions::new(SsdPath::SerialRecalculated).with_chunk_len(Some(4)),
        ] {
            let label = format!("{pattern:?} {scan_options:?}");
            let (out_single, cache_single) =
                model.forward_single_ssd(x.clone(), None, &scan_options, None);
            let (out_double, cache_double) =
                model.forward_double_ssd(x.clone(), None, &scan_options, None);
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

/// Non-vacuity, which the parity tests above cannot give. A quietly dropped
/// second tap would leave both pathways agreeing with each other and with
/// `step`. Without that tap, a two-tap member would be the *one-tap* pattern.
/// Only a comparison against that pattern notices the drop. At `μ = σ(0) = ½`,
/// each two-tap member must differ from
/// [`Vertical`](TapPattern::Vertical). The two members must also differ from
/// each other: only one of them closes its lag-1 tap at each token's first
/// micro-step.
#[test]
fn two_tap_patterns_actually_spend_their_second_tap() {
    let device = test_device();
    let u = 3;
    let config = cfg_for(TapPattern::Vertical, RotationKind::Real1D, u).with_has_proj_bias(true);
    let vertical = config.clone().init(&device);
    let x = input(&config, 5);
    let scan_options = Mamba3ScanOptions::default();
    let out_vertical = vertical.forward_single_ssd(x.clone(), None, &scan_options, None).0;

    let two_tap: Vec<Tensor<3>> = [
        TapPattern::VerticalPlusHorizontalInToken,
        TapPattern::VerticalPlusHorizontalCarryOver,
    ]
    .into_iter()
    .map(|pattern| {
        let mut block = config.clone().with_tap_pattern(pattern).init(&device);
        share_weights_except_in_proj(&vertical, &mut block);
        append_dead_sub_projections(&vertical, &mut block, u * config.nheads(), 0.0, &device);
        let out = block.forward_single_ssd(x.clone(), None, &scan_options, None).0;
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
/// single-tap patterns. `μ ≡ 1` is the carry-over, and `μ ≡ 0` is the vertical.
/// So `μ` is a per-(head, micro-step) interpolation between them, not a third
/// model. Built by appending saturated `μ` sub-projections to
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
    let scan_options = Mamba3ScanOptions::default();
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
                target.forward_single_ssd(x.clone(), None, &scan_options, None).0,
                join.forward_single_ssd(x.clone(), None, &scan_options, None).0,
            ),
            (
                "double",
                target.forward_double_ssd(x.clone(), None, &scan_options, None).0,
                join.forward_double_ssd(x.clone(), None, &scan_options, None).0,
            ),
            ("step", unrolled(target, &x).0, unrolled(&join, &x).0),
        ] {
            let d = max_abs_diff(a, b);
            assert!(d <= dtype_tol(tol), "{pathway}: {label}: {d:.3e}");
        }
    }
}

/// The tap FIFO is what the pattern changes in the cache. The cache has no tap
/// FIFO without a tap, one slot for the lag-1 patterns, and `u` slots for the
/// lag-`u` ones.
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
