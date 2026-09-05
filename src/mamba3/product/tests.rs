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
    // The tap slots are absent under `Trapezoid::None`, in both caches.
    assert_eq!(
        cache_fwd.k_state_bumhr.is_some(),
        cache_step.k_state_bumhr.is_some(),
        "{label}: the two paths disagree on whether a tap slot exists"
    );
    if let (Some(fwd), Some(step)) = (cache_fwd.k_state_bumhr, cache_step.k_state_bumhr) {
        assert!(
            max_abs_diff(fwd, step) < 1e-4,
            "{label}: final k_state (the tap FIFO's B)"
        );
    }
    if let (Some(fwd), Some(step)) = (cache_fwd.v_state_buhp, cache_step.v_state_buhp) {
        assert!(
            max_abs_diff(fwd, step) < 1e-4,
            "{label}: final v_state (the tap FIFO's x)"
        );
    }
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

/// The third leg of forward ≡ step, alongside the outputs and the cache:
/// backprop through the chunked `forward` and through the unrolled `step` must
/// agree. Same function ⇒ same gradients, so a fold whose backward disagreed
/// with its own recurrence — a micro-step whose gradient landed on the wrong
/// position, or a `C` broadcast whose gradient was not summed back over the
/// group — would show up here and in no value test.
fn forward_step_grad_parity(kind: RotationKind, micro_steps: usize) {
    let device: Device = Default::default();
    let config = cfg(kind, micro_steps);
    let model: Mamba3 = config.init(&device.clone().autodiff());

    let (batch, tokens) = (2, 4);
    let input = Tensor::<3>::random(
        [batch, tokens, config.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let head = Tensor::<3>::random(
        [batch, tokens, config.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // Fresh autodiff leaves per path.
    let p_fwd = burn::module::Param::from_tensor(Tensor::from_inner(input.clone()));
    let p_step = burn::module::Param::from_tensor(Tensor::from_inner(input));

    let (out_fwd, _) = model.forward(p_fwd.val(), None, Mamba3SsdPath::Minimal(None));
    let g_fwd = (out_fwd * Tensor::from_inner(head.clone())).sum().backward();

    let mut cache: Option<Mamba3Cache> = None;
    let mut outs = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let (o, c) = model.step(p_step.val().narrow(1, t, 1).squeeze_dim::<2>(1), cache);
        outs.push(o.unsqueeze_dim::<3>(1));
        cache = Some(c);
    }
    let g_step = (Tensor::cat(outs, 1) * Tensor::from_inner(head))
        .sum()
        .backward();

    let label = format!("{kind:?} u={micro_steps}");
    let d_in = max_abs_diff(
        p_fwd.val().grad(&g_fwd).expect("grad input (forward)"),
        p_step.val().grad(&g_step).expect("grad input (step)"),
    );
    let weight = model.in_proj.weight.val();
    let d_w = max_abs_diff(
        weight.clone().grad(&g_fwd).expect("grad in_proj (forward)"),
        weight.grad(&g_step).expect("grad in_proj (step)"),
    );
    assert!(d_in < 1e-2, "{label}: forward/step input-grad diff {d_in:.6}");
    assert!(d_w < 1e-2, "{label}: forward/step in_proj-grad diff {d_w:.6}");
}

#[test]
fn forward_step_grad_parity_abelian() {
    for kind in [RotationKind::Real1D, RotationKind::Complex2D] {
        forward_step_grad_parity(kind, 2);
        forward_step_grad_parity(kind, MICRO);
    }
}

#[test]
fn forward_step_grad_parity_non_abelian() {
    forward_step_grad_parity(RotationKind::Quaternion4D, 2);
    forward_step_grad_parity(RotationKind::Rotor4D, 2);
}

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
