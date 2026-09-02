use super::*;
use crate::mamba3::cache::Mamba3Cache;
use crate::mamba3::mamba3::{Mamba3, Mamba3Config};
use crate::mamba3::rotation::RotationKind;
use crate::mamba3::ssd_path::Mamba3SsdPath;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

const MICRO: usize = 3;

fn cfg(kind: RotationKind, micro_steps: usize) -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_rotation(kind)
        .with_micro_steps(micro_steps)
}

// ---------------------------------------------------------------------------
// The folding helpers
// ---------------------------------------------------------------------------

/// `unfold_micro_bs` must read a token's micro-steps out in projection order:
/// channel `j·width + c` becomes position `j`, channel `c`.
#[test]
fn unfold_reads_micro_steps_in_order() {
    let device: Device = Default::default();
    // [1, 2, 6] = 2 tokens × (3 micro-steps × width 2), values 0..12.
    let t = Tensor::<1, burn::tensor::Int>::arange(0..12, &device).float().reshape([1, 2, 6]);
    let out = unfold_micro_bs(t, 3);
    assert_eq!([1, 6, 2], out.dims());
    let flat: Vec<f32> = out.into_data().try_to_vec::<f32>().unwrap();
    // Token 0's micro-steps are (0,1), (2,3), (4,5); token 1's follow.
    assert_eq!(
        vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        flat
    );
    // …and the single-token form agrees with it.
    let s = Tensor::<1, burn::tensor::Int>::arange(0..6, &device).float().reshape([1, 6]);
    let out = unfold_micro_b(s, 3);
    assert_eq!([1, 3, 2], out.dims());
    let flat: Vec<f32> = out.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(vec![0., 1., 2., 3., 4., 5.], flat);
}

/// `repeat_micro_bs` puts the same per-token value at every micro-step, and
/// `last_micro*` picks the last one back out — so the two are inverse on the
/// positions that survive.
#[test]
fn repeat_then_take_last_is_identity() {
    let device: Device = Default::default();
    let t = Tensor::<1, burn::tensor::Int>::arange(0..8, &device).float().reshape([1, 4, 2]);
    let wide = repeat_micro_bs(t.clone(), MICRO);
    assert_eq!([1, 12, 2], wide.dims());
    let back = last_micro4(wide.clone().reshape([1, 12, 2, 1]), MICRO);
    assert_eq!([1, 4, 2, 1], back.dims());
    let d = max_abs_diff(back.reshape([1, 4, 2]), t);
    assert_eq!(0.0, d);
    // last_micro5 picks the same positions one rank up.
    let five = wide.reshape([1, 12, 1, 2, 1]);
    let picked = last_micro5(five, MICRO);
    assert_eq!([1, 4, 1, 2, 1], picked.dims());
}

// ---------------------------------------------------------------------------
// Config arithmetic
// ---------------------------------------------------------------------------

/// `micro_steps = 1` must leave the projection width exactly where it was, and
/// `u` must widen only the per-micro-step segments (`z` and `C` stay put).
#[test]
fn in_proj_widens_only_the_per_micro_step_segments() {
    for kind in [
        RotationKind::Real1D,
        RotationKind::Complex2D,
        RotationKind::Quaternion4D,
    ] {
        let one = cfg(kind, 1);
        let stock = 2 * one.d_inner()
            + 2 * one.ngroups * one.state_rank * one.mimo_rank
            + 3 * one.nheads()
            + one.num_rotation_channels();
        assert_eq!(stock, one.d_in_proj(), "{kind:?}: u = 1 is stock Mamba-3");

        let u = cfg(kind, MICRO);
        let per_micro = one.d_inner()
            + one.ngroups * one.state_rank * one.mimo_rank
            + 3 * one.nheads()
            + one.num_rotation_channels();
        assert_eq!(
            stock + (MICRO - 1) * per_micro,
            u.d_in_proj(),
            "{kind:?}: each extra micro-step adds one x/B/Δ/A/λ/rotation set"
        );

        // The Muon plan must still tile the whole fused axis.
        let width: usize = u.muon_projections()[0]
            .segments
            .iter()
            .map(|s| s.width)
            .sum();
        assert_eq!(u.d_in_proj(), width, "{kind:?}: muon segments tile in_proj");
    }
}

// ---------------------------------------------------------------------------
// forward ≡ step, on both pathways
// ---------------------------------------------------------------------------

/// Unroll `step` token by token and compare against `forward` over the same
/// sequence — the property that pins the fold: `forward` runs the recurrence on
/// the folded sequence, `step` runs `u` explicit micro-steps, and they must
/// agree on outputs *and* on every cache field.
fn forward_matches_step_double(kind: RotationKind, micro_steps: usize) {
    let device: Device = Default::default();
    let config = cfg(kind, micro_steps);
    let model: Mamba3 = config.init(&device);

    let batch = 2;
    let tokens = 5;
    let input = Tensor::<3>::random(
        [batch, tokens, config.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (out_fwd, cache_fwd) =
        model.forward_double_ssd(input.clone(), None, &Mamba3SsdPath::default());

    let mut cache = None;
    let mut outs = Vec::new();
    for t in 0..tokens {
        let (o, c) = model.step_double_ssd(input.clone().narrow(1, t, 1).squeeze_dim(1), cache);
        outs.push(o.unsqueeze_dim::<3>(1));
        cache = Some(c);
    }
    let out_step = Tensor::cat(outs, 1);
    let cache_step = cache.unwrap();

    let label = format!("{kind:?} u={micro_steps}");
    assert!(
        max_abs_diff(out_fwd, out_step) < 1e-4,
        "{label}: forward vs unrolled step outputs"
    );
    assert!(
        max_abs_diff(cache_fwd.ssm_bhpr, cache_step.ssm_bhpr) < 1e-4,
        "{label}: final ssm state"
    );
    assert!(
        max_abs_diff(cache_fwd.k_state_bmhr, cache_step.k_state_bmhr) < 1e-4,
        "{label}: final k_state (last micro-step's B)"
    );
    assert!(
        max_abs_diff(cache_fwd.v_state_bhp, cache_step.v_state_bhp) < 1e-4,
        "{label}: final v_state (last micro-step's x)"
    );
}

#[test]
fn forward_matches_step_double_real() {
    forward_matches_step_double(RotationKind::Real1D, 2);
    forward_matches_step_double(RotationKind::Real1D, MICRO);
}

#[test]
fn forward_matches_step_double_complex() {
    forward_matches_step_double(RotationKind::Complex2D, 2);
    forward_matches_step_double(RotationKind::Complex2D, MICRO);
}

#[test]
fn forward_matches_step_double_quaternion() {
    forward_matches_step_double(RotationKind::Quaternion4D, 2);
}

#[test]
fn forward_matches_step_double_rotor() {
    forward_matches_step_double(RotationKind::Rotor4D, 2);
}

/// The same for the single-pass pathway, which reaches `step` by round-tripping
/// through the double-ssd cache — so this also pins that the two accumulators
/// still coincide at token boundaries when a token is `u` micro-steps long.
fn forward_single_matches_step(kind: RotationKind, micro_steps: usize) {
    let device: Device = Default::default();
    let config = cfg(kind, micro_steps);
    let model: Mamba3 = config.init(&device);

    let batch = 2;
    let tokens = 5;
    let input = Tensor::<3>::random(
        [batch, tokens, config.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (out_fwd, _) = model.forward_single_ssd(input.clone(), None, &Mamba3SsdPath::default());

    let mut cache: Option<Mamba3Cache> = None;
    let mut outs = Vec::new();
    for t in 0..tokens {
        let (o, c) = model.step(input.clone().narrow(1, t, 1).squeeze_dim(1), cache);
        outs.push(o.unsqueeze_dim::<3>(1));
        cache = Some(c);
    }
    assert!(
        max_abs_diff(out_fwd, Tensor::cat(outs, 1)) < 1e-4,
        "{kind:?} u={micro_steps}: forward_single_ssd vs unrolled step"
    );
}

#[test]
fn forward_single_matches_step_all_kinds() {
    for kind in [
        RotationKind::Real1D,
        RotationKind::Complex2D,
        RotationKind::Quaternion4D,
        RotationKind::Rotor4D,
    ] {
        forward_single_matches_step(kind, 2);
    }
    forward_single_matches_step(RotationKind::Complex2D, MICRO);
}

/// A prefill split at a token boundary must equal the whole one: the cache
/// carries the *last micro-step*, which is exactly what the next call's first
/// micro-step follows.
#[test]
fn split_prefill_matches_full() {
    let device: Device = Default::default();
    for kind in [RotationKind::Complex2D, RotationKind::Quaternion4D] {
        let config = cfg(kind, 2);
        let model: Mamba3 = config.init(&device);
        let (batch, tokens, split) = (2, 6, 4);
        let input = Tensor::<3>::random(
            [batch, tokens, config.d_model],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let path = Mamba3SsdPath::default();

        let (full, _) = model.forward_single_ssd(input.clone(), None, &path);
        let (head, cache) =
            model.forward_single_ssd(input.clone().narrow(1, 0, split), None, &path);
        let (tail, _) = model.forward_single_ssd(
            input.narrow(1, split, tokens - split),
            Some(cache),
            &path,
        );
        assert!(
            max_abs_diff(full, Tensor::cat(vec![head, tail], 1)) < 1e-4,
            "{kind:?}: split prefill vs full"
        );
    }
}

// ---------------------------------------------------------------------------
// Gradients reach every micro-step
// ---------------------------------------------------------------------------

/// Every micro-step's slice of the widened `in_proj` must receive gradient.
/// A fold that dropped a micro-step (or read the segments in the wrong order)
/// would leave one of these blocks dead, and every value-parity test above
/// would still pass.
#[test]
fn every_micro_step_receives_gradient() {
    let device: Device = Default::default();
    let config = cfg(RotationKind::Complex2D, MICRO);
    let model: Mamba3 = config.init(&device.clone().autodiff());

    let (batch, tokens) = (2, 4);
    let input = Tensor::<3>::from_inner(Tensor::random(
        [batch, tokens, config.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    ));
    let (out, _) = model.forward_single_ssd(input, None, &Mamba3SsdPath::default());
    let grads = out.sum().backward();
    let g = model
        .in_proj
        .weight
        .val()
        .grad(&grads)
        .expect("in_proj.weight grad");

    let d_inner = config.d_inner();
    let nheads = config.nheads();
    let bc = config.ngroups * config.state_rank * config.mimo_rank;
    let rot = config.num_rotation_channels();
    // [z | x·u | B·u | C | Δ·u | A·u | λ·u | rotation·u]
    let mut at = d_inner;
    for (name, width) in [
        ("x", d_inner),
        ("b", bc),
        ("skip-c", 0),
        ("dt", nheads),
        ("a", nheads),
        ("lambda", nheads),
        ("rotation", rot),
    ] {
        if width == 0 {
            at += bc; // the per-token C segment sits between B and Δ
            continue;
        }
        for j in 0..MICRO {
            let block = g.clone().narrow(1, at + j * width, width);
            let m = block.abs().max().into_scalar::<f32>();
            assert!(
                m > 0.0,
                "micro-step {j} of the `{name}` segment received no gradient"
            );
        }
        at += MICRO * width;
    }
    assert_eq!(config.d_in_proj(), at, "walked the whole fused axis");
}

// ---------------------------------------------------------------------------
// step_infinite
// ---------------------------------------------------------------------------

/// Force a healthy decay (`α ≤ exp(−0.05)` per *micro*-step) so a few hundred
/// unrolled steps reach the fixed point to fp32 accuracy — same knob the
/// `step_constant` suite uses.
fn decaying(cfg: Mamba3Config) -> Mamba3Config {
    cfg.with_a_floor(1.0).with_dt_limit((0.05, 5.0))
}

/// The closed-form fixed point must be the limit of the recurrence it claims to
/// be the limit of — checked by stepping the same token until the output stops
/// moving.
fn step_infinite_matches_unrolled(kind: RotationKind, micro_steps: usize, mimo_rank: usize) {
    let device: Device = Default::default();
    let config = decaying(cfg(kind, micro_steps).with_mimo_rank(mimo_rank));
    let model: Mamba3 = config.init(&device);

    let batch = 2;
    let token = Tensor::<2>::random(
        [batch, config.d_model],
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
    let limit = model.step_infinite(token);
    let d = max_abs_diff(out.unwrap(), limit);
    assert!(
        d < 1e-3,
        "{kind:?} u={micro_steps} M={mimo_rank}: step_infinite vs 300 unrolled steps, \
         max abs diff = {d:.6}"
    );
}

#[test]
fn step_infinite_matches_unrolled_abelian() {
    for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
        // `u = 1` must still be the stock closed form.
        step_infinite_matches_unrolled(kind, 1, 1);
        step_infinite_matches_unrolled(kind, 2, 1);
        step_infinite_matches_unrolled(kind, MICRO, 1);
        // MIMO exercises the per-micro-step Gram accumulation.
        step_infinite_matches_unrolled(kind, MICRO, 2);
    }
}

/// `u = 1` keeps every kind, including the non-abelian ones.
#[test]
fn step_infinite_matches_unrolled_non_abelian_at_one() {
    step_infinite_matches_unrolled(RotationKind::Quaternion4D, 1, 1);
    step_infinite_matches_unrolled(RotationKind::Rotor4D, 1, 1);
}

/// …and refuses at `u > 1`, where the rotation product and its partials stop
/// commuting and the output is almost-periodic rather than convergent.
#[test]
#[should_panic(expected = "no limit to return")]
fn step_infinite_refuses_non_abelian_product() {
    let device: Device = Default::default();
    let config = cfg(RotationKind::Quaternion4D, 2);
    let model: Mamba3 = config.init(&device);
    let token = Tensor::<2>::random([1, config.d_model], Distribution::Normal(0.0, 1.0), &device);
    let _ = model.step_infinite(token);
}
