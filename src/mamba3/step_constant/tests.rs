use super::*;
use crate::modules::LayersBuilder;
use crate::modules::network::LatentNetworkBuilder;
use crate::utils::test_helpers::max_abs_diff;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

fn small_config() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
}

fn quat_config() -> Mamba3Config {
    small_config().with_rotation(RotationKind::Quaternion4D)
}

fn rotor_config() -> Mamba3Config {
    small_config().with_rotation(RotationKind::Rotor4D)
}

// ---------------------------------------------------------------------------
// step_infinite — convergence to the unrolled fixed point
// ---------------------------------------------------------------------------

/// Force a healthy decay (`α ≤ exp(−0.05)`) so a few hundred unrolled steps
/// reach the fixed point to fp32 accuracy.
fn decaying(cfg: Mamba3Config) -> Mamba3Config {
    cfg.with_a_floor(1.0).with_dt_limit((0.05, 5.0))
}

fn run_step_infinite_matches_unroll(label: &str, cfg: Mamba3Config, steps: usize, tol: f32) {
    let device: Device = Default::default();
    let model = cfg.init(&device);
    let batch = 2;
    let x = Tensor::<2>::random(
        [batch, cfg.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let mut cache = None;
    let mut out = None;
    for _ in 0..steps {
        let (o, c) = model.step(x.clone(), cache);
        out = Some(o);
        cache = Some(c);
    }
    let y_inf = model.step_infinite(x);
    let d = max_abs_diff(out.unwrap(), y_inf);
    assert!(
        d < tol,
        "{label}: step_infinite vs {steps} unrolled steps max abs diff = {d:.6} (tol {tol})"
    );
}

#[test]
fn step_infinite_matches_unroll_complex_siso() {
    run_step_infinite_matches_unroll("complex siso", decaying(small_config()), 300, 1e-3);
}

#[test]
fn step_infinite_matches_unroll_complex_mimo() {
    run_step_infinite_matches_unroll(
        "complex mimo",
        decaying(small_config().with_mimo_rank(2)),
        300,
        1e-3,
    );
}

#[test]
fn step_infinite_matches_unroll_quat_siso() {
    run_step_infinite_matches_unroll("quat siso", decaying(quat_config()), 300, 1e-3);
}

/// The quaternion default is a *full* rotation, so the partial case needs
/// asking for: `state_rank = 8` turned by one of its two 4-blocks, with the
/// pass-through half decaying as a plain geometric series.
#[test]
fn step_infinite_matches_unroll_quat_rope_partial() {
    run_step_infinite_matches_unroll(
        "quat rope=0.5",
        decaying(quat_config().with_rope_fraction(0.5)),
        300,
        1e-3,
    );
}

/// The `SO(4)` fixed point takes a different route entirely — the two-sided
/// series leaves the commutative subalgebra the quaternion branch inverts in,
/// so it is resolved by Cayley–Hamilton over `ℝ⁴`
/// ([`rotor_resolvent_partial`]). Agreeing with the unrolled recurrence is what
/// pins that closed form.
#[test]
fn step_infinite_matches_unroll_rotor_siso() {
    run_step_infinite_matches_unroll("rotor siso", decaying(rotor_config()), 300, 3e-3);
}

#[test]
fn step_infinite_matches_unroll_rotor_mimo() {
    run_step_infinite_matches_unroll(
        "rotor mimo",
        decaying(rotor_config().with_mimo_rank(2)),
        300,
        3e-3,
    );
}

/// The pass-through half must still be a plain geometric series when only one
/// of the two 4-blocks is turned.
#[test]
fn step_infinite_matches_unroll_rotor_rope_partial() {
    run_step_infinite_matches_unroll(
        "rotor rope=0.5",
        decaying(rotor_config().with_rope_fraction(0.5)),
        300,
        3e-3,
    );
}

/// A decay close to 1 is where the resolvent's determinant approaches
/// `(1−α)⁴`: the factored form it is built from must not lose it to
/// cancellation.
#[test]
fn step_infinite_matches_unroll_rotor_slow_decay() {
    run_step_infinite_matches_unroll(
        "rotor slow decay",
        rotor_config().with_a_floor(0.05).with_dt_limit((0.05, 1.0)),
        4000,
        1e-2,
    );
}

/// `step_infinite` derives its per-step rotation from the same
/// [`RotationSpec`](crate::mamba3::rotation::RotationSpec) as `step`, so a
/// non-default
/// [`rotation_range`](crate::mamba3::mamba3::Mamba3Config::rotation_range) must
/// reach both. This is the cross-site guard: the constant-input shortcut used
/// to spell the `Δ·π·tanh(ϑ)` formula out for itself, and a knob added to one
/// site and missed at the other would leave the "fixed point" converging to
/// something the recurrence never reaches.
#[test]
fn step_infinite_matches_unroll_at_a_non_default_range() {
    for (label, cfg, tol) in [
        ("complex range 1.4", decaying(small_config()), 1e-3),
        ("quat range 1.4", decaying(quat_config()), 1e-3),
        // The reference side is the noisy one here: 300 sequential steps of a
        // two-factor rotation accumulate about twice the fp32 drift of the
        // one-factor kind (and the drift is in `step`, not in the closed form).
        ("rotor range 1.4", decaying(rotor_config()), 3e-3),
    ] {
        run_step_infinite_matches_unroll(label, cfg.with_rotation_range(1.4), 300, tol);
    }
}

// ---------------------------------------------------------------------------
// Upstream: Layers / LatentNetwork
// ---------------------------------------------------------------------------

/// Two stacked layers: the limit composes layer by layer, so `step_infinite`
/// converges to the unrolled fixed point of the whole network.
#[test]
fn network_two_layers_constant_input() {
    let device: Device = Default::default();
    let cfg = decaying(small_config());
    let net = LatentNetworkBuilder {
        input_size: 8,
        layers: LayersBuilder::new(2, cfg),
        output_size: 8,
        final_norm: false,
        class_tokens: Vec::new(),
    }
    .init(&device);
    let x = Tensor::<2>::random([2, 8], Distribution::Normal(0.0, 1.0), &device);

    // Unrolled fixed point.
    let steps = 300;
    let mut caches = None;
    let mut out = None;
    for _ in 0..steps {
        let (o, c) = net.step(x.clone(), caches, None);
        out = Some(o);
        caches = Some(c);
    }
    let out = out.unwrap();

    let y_inf = net.step_infinite(x);
    let d_inf = max_abs_diff(out, y_inf);
    assert!(
        d_inf < 1e-3,
        "two-layer step_infinite vs unroll: {d_inf:.6}"
    );
}
