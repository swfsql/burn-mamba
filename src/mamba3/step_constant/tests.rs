use super::*;
use crate::mamba3::single_ssd::prelude::Mamba3SingleSsdCache;
use crate::modules::LayersBuilder;
use crate::modules::network::LatentNetworkBuilder;
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
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

/// Step the model over `tokens` from a fresh cache. Deterministic, so calling
/// it twice with the same tokens reproduces the same cache (the cache enum is
/// not `Clone`).
fn warm(model: &Mamba3, tokens: &[Tensor<2>]) -> Option<Mamba3Cache> {
    let mut cache = None;
    for t in tokens {
        let (_o, c) = model.step(t.clone(), cache);
        cache = Some(c);
    }
    cache
}

/// Field-level view of either cache variant (the conversion is a field move).
fn cache_fields(cache: Mamba3Cache) -> Mamba3SingleSsdCache {
    match cache {
        Mamba3Cache::SingleSsd(c) => c,
        Mamba3Cache::DoubleSsd(c) => c.into(),
    }
}

/// Compare every cache field. Angles are compared through `(cos, sin)` — the
/// unrolled and jumped cumulative angles are equal only mod `2π` at the wrap
/// boundary.
fn assert_caches_match(label: &str, a: Mamba3SingleSsdCache, b: Mamba3SingleSsdCache, tol: f32) {
    let checks = [
        ("ssm", max_abs_diff(a.ssm_bhpr, b.ssm_bhpr)),
        ("k_state", max_abs_diff(a.k_state_bmhr, b.k_state_bmhr)),
        ("v_state", max_abs_diff(a.v_state_bhp, b.v_state_bhp)),
    ];
    for (name, d) in checks {
        assert!(d < tol, "{label}: {name} max abs diff = {d:.6} (tol {tol})");
    }
    match (a.rotation, b.rotation) {
        (RotationState::Angle(x), RotationState::Angle(y)) => {
            let dc = max_abs_diff(x.clone().cos(), y.clone().cos());
            let ds = max_abs_diff(x.sin(), y.sin());
            assert!(
                dc < tol && ds < tol,
                "{label}: cum angle (cos, sin) max abs diff = ({dc:.6}, {ds:.6}) (tol {tol})"
            );
        }
        (RotationState::Quaternion(x), RotationState::Quaternion(y)) => {
            let d = max_abs_diff(x, y);
            assert!(
                d < tol,
                "{label}: cum quaternion max abs diff = {d:.6} (tol {tol})"
            );
        }
        _ => panic!("{label}: rotation state kinds differ"),
    }
}

/// `step_n_approx(x, n)` must equal `n` literal `step(x)` calls — outputs and
/// every final cache field — starting from a cache warmed with *different*
/// tokens (so the first jump step consumes a non-matching previous-token
/// entry).
fn run_step_n_matches_repeated(label: &str, cfg: Mamba3Config, n: usize, tol: f32) {
    let device: Device = Default::default();
    let model = cfg.init(&device);
    let batch = 2;
    let normal = Distribution::Normal(0.0, 1.0);
    let tokens: Vec<Tensor<2>> = (0..2)
        .map(|_| Tensor::<2>::random([batch, cfg.d_model], normal, &device))
        .collect();
    let x = Tensor::<2>::random([batch, cfg.d_model], normal, &device);

    // Reference: n literal steps of the same token.
    let mut cache_ref = warm(&model, &tokens);
    let mut out_ref = None;
    for _ in 0..n {
        let (o, c) = model.step(x.clone(), cache_ref);
        out_ref = Some(o);
        cache_ref = Some(c);
    }

    // Closed-form jump from an identically warmed cache.
    let (out_jump, cache_jump) = model.step_n_approx(x, n, warm(&model, &tokens));

    let d = max_abs_diff(out_ref.unwrap(), out_jump);
    assert!(d < tol, "{label}: output max abs diff = {d:.6} (tol {tol})");
    assert_caches_match(
        label,
        cache_fields(cache_ref.unwrap()),
        cache_fields(cache_jump),
        tol,
    );
}

#[test]
fn step_n_matches_repeated_complex_siso() {
    run_step_n_matches_repeated("complex siso", small_config(), 6, 1e-4);
}

#[test]
fn step_n_matches_repeated_complex_single_jump() {
    // n = 2 ⇒ K = 1: the degenerate geometric series (one driven term).
    run_step_n_matches_repeated("complex n=2", small_config(), 2, 1e-4);
}

#[test]
fn step_n_matches_repeated_complex_mimo() {
    run_step_n_matches_repeated("complex mimo", small_config().with_mimo_rank(2), 6, 1e-4);
}

#[test]
fn step_n_matches_repeated_complex_ngroups2() {
    let cfg = Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(16)
        .with_ngroups(2);
    run_step_n_matches_repeated("complex ngroups2", cfg, 6, 1e-4);
}

#[test]
fn step_n_matches_repeated_complex_rope_full() {
    run_step_n_matches_repeated(
        "complex rope=1.0",
        small_config().with_rope_fraction(1.0),
        6,
        1e-4,
    );
}

#[test]
fn step_n_matches_repeated_complex_rope_zero() {
    run_step_n_matches_repeated(
        "complex rope=0.0",
        small_config().with_rope_fraction(0.0),
        6,
        1e-4,
    );
}

#[test]
fn step_n_matches_repeated_complex_outproj_norm_mimo() {
    let cfg = small_config()
        .with_mimo_rank(2)
        .with_has_outproj_norm(true);
    run_step_n_matches_repeated("complex outproj_norm mimo", cfg, 6, 1e-4);
}

#[test]
fn step_n_matches_repeated_quat_siso() {
    run_step_n_matches_repeated("quat siso", quat_config(), 6, 1e-4);
}

#[test]
fn step_n_matches_repeated_quat_mimo() {
    run_step_n_matches_repeated("quat mimo", quat_config().with_mimo_rank(2), 6, 1e-4);
}

#[test]
fn step_n_matches_repeated_quat_rope_full() {
    run_step_n_matches_repeated(
        "quat rope=1.0",
        quat_config().with_rope_fraction(1.0),
        6,
        1e-4,
    );
}

/// `n = 1` must be *exactly* one `step` (same code path up to the jump branch).
#[test]
fn step_n_1_equals_step() {
    let device: Device = Default::default();
    let model = small_config().init(&device);
    let x = Tensor::<2>::random([2, 32], Distribution::Normal(0.0, 1.0), &device);
    let (a, _) = model.step(x.clone(), None);
    let (b, _) = model.step_n_approx(x, 1, None);
    assert!(max_abs_diff(a, b) < 1e-6);
}

/// The returned cache keeps the supplied pathway variant (double-ssd in,
/// double-ssd out), and the jump agrees with the unrolled steps on it.
#[test]
fn step_n_preserves_double_ssd_variant() {
    let device: Device = Default::default();
    let cfg = small_config();
    let model = cfg.init(&device);
    let batch = 2;
    let normal = Distribution::Normal(0.0, 1.0);
    let t0 = Tensor::<2>::random([batch, cfg.d_model], normal, &device);
    let x = Tensor::<2>::random([batch, cfg.d_model], normal, &device);

    let as_double = |c: Mamba3Cache| Mamba3Cache::DoubleSsd(cache_fields(c).into());

    let mut cache_ref = Some(as_double(warm(&model, &[t0.clone()]).unwrap()));
    let mut out_ref = None;
    for _ in 0..5 {
        let (o, c) = model.step(x.clone(), cache_ref);
        out_ref = Some(o);
        cache_ref = Some(c);
    }
    let (out_jump, cache_jump) = model.step_n_approx(
        x,
        5,
        Some(as_double(warm(&model, &[t0.clone()]).unwrap())),
    );
    assert!(matches!(cache_jump, Mamba3Cache::DoubleSsd(_)));
    assert!(max_abs_diff(out_ref.unwrap(), out_jump) < 1e-4);
    assert_caches_match(
        "double-ssd variant",
        cache_fields(cache_ref.unwrap()),
        cache_fields(cache_jump),
        1e-4,
    );
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
    let x = Tensor::<2>::random([batch, cfg.d_model], Distribution::Normal(0.0, 1.0), &device);

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

#[test]
fn step_infinite_matches_unroll_quat_rope_full() {
    run_step_infinite_matches_unroll(
        "quat rope=1.0",
        decaying(quat_config().with_rope_fraction(1.0)),
        300,
        1e-3,
    );
}

/// `step_n_approx` for large `n` must agree with `step_infinite` (single block:
/// both exact; the αⁿ transient is below tolerance).
#[test]
fn step_n_large_matches_step_infinite() {
    let device: Device = Default::default();
    let cfg = decaying(small_config());
    let model = cfg.init(&device);
    let x = Tensor::<2>::random([2, cfg.d_model], Distribution::Normal(0.0, 1.0), &device);
    let (y_n, _cache) = model.step_n_approx(x.clone(), 400, None);
    let y_inf = model.step_infinite(x);
    let d = max_abs_diff(y_n, y_inf);
    assert!(d < 1e-3, "step_n(400) vs step_infinite: {d:.6}");
}

// ---------------------------------------------------------------------------
// Gradient parity: the jump is an algebraic reformulation of the unroll
// ---------------------------------------------------------------------------

struct StepGrads {
    out: Tensor<2>,
    d_input: Tensor<2>,
    d_in_proj_w: Tensor<2>,
    d_dt_bias: Tensor<1>,
    d_b_bias: Tensor<3>,
    d_out_proj_w: Tensor<2>,
}

struct StepHeads {
    out: Tensor<2>,
    ssm: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<3>,
}

/// Forward `f`, couple the output and the final cache fields to a scalar loss
/// via fixed random heads, and return the gradients of the input and a
/// representative parameter set.
fn run_with_grads(
    model: &Mamba3,
    input: &Param<Tensor<2>>,
    heads: &StepHeads,
    f: impl FnOnce(&Mamba3, Tensor<2>) -> (Tensor<2>, Mamba3Cache),
) -> StepGrads {
    let (out, cache) = f(model, input.val());
    let out_inner = out.clone().inner();
    let c = cache_fields(cache);
    let loss = (out * Tensor::from_inner(heads.out.clone())).sum()
        + (c.ssm_bhpr * Tensor::from_inner(heads.ssm.clone())).sum()
        + (c.k_state_bmhr * Tensor::from_inner(heads.k.clone())).sum()
        + (c.v_state_bhp * Tensor::from_inner(heads.v.clone())).sum();
    let grads = loss.backward();
    StepGrads {
        out: out_inner,
        d_input: input.val().grad(&grads).expect("grad input"),
        d_in_proj_w: model
            .in_proj
            .weight
            .val()
            .grad(&grads)
            .expect("grad in_proj.weight"),
        d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("grad dt_bias_h"),
        d_b_bias: model
            .b_bias_hmr
            .val()
            .grad(&grads)
            .expect("grad b_bias_hmr"),
        d_out_proj_w: model
            .out_proj
            .weight
            .val()
            .grad(&grads)
            .expect("grad out_proj.weight"),
    }
}

fn run_step_n_grads_match_unroll(label: &str, cfg: Mamba3Config, n: usize, grad_tol: f32) {
    let device: Device = Default::default();
    let model = cfg.init(&device.clone().autodiff());
    let batch = 2;
    let normal = Distribution::Normal(0.0, 1.0);
    let input = Tensor::<2>::random([batch, cfg.d_model], normal, &device);
    let heads = StepHeads {
        out: Tensor::<2>::random([batch, cfg.d_model], normal, &device),
        ssm: Tensor::<4>::random(
            [batch, cfg.nheads(), cfg.per_head_dim, cfg.state_rank],
            normal,
            &device,
        ),
        k: Tensor::<4>::random(
            [batch, cfg.mimo_rank, cfg.nheads(), cfg.state_rank],
            normal,
            &device,
        ),
        v: Tensor::<3>::random([batch, cfg.nheads(), cfg.per_head_dim], normal, &device),
    };
    let param = |t: &Tensor<2>| Param::from_tensor(Tensor::from_inner(t.clone()));

    let in_unroll = param(&input);
    let r_unroll = run_with_grads(&model, &in_unroll, &heads, |m, x| {
        let mut cache = None;
        let mut out = None;
        for _ in 0..n {
            let (o, c) = m.step(x.clone(), cache);
            out = Some(o);
            cache = Some(c);
        }
        (out.unwrap(), cache.unwrap())
    });

    let in_jump = param(&input);
    let r_jump = run_with_grads(&model, &in_jump, &heads, |m, x| m.step_n_approx(x, n, None));

    let d_out = max_abs_diff(r_unroll.out.clone(), r_jump.out.clone());
    assert!(d_out < 1e-4, "{label}: output max abs diff = {d_out:.6}");
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($field:ident, $name:expr) => {{
            let d = max_abs_diff(r_unroll.$field.clone(), r_jump.$field.clone());
            eprintln!("{label:>24} {:>16} | max abs diff = {d:>10.6}", $name);
            if d >= grad_tol {
                failures.push(format!(
                    "{label}: grad of {} max abs diff = {d:.6} (tol {grad_tol})",
                    $name
                ));
            }
        }};
    }
    check!(d_input, "input");
    check!(d_in_proj_w, "in_proj.weight");
    check!(d_dt_bias, "dt_bias_h");
    check!(d_b_bias, "b_bias_hmr");
    check!(d_out_proj_w, "out_proj.weight");
    assert!(
        failures.is_empty(),
        "gradient mismatches:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn step_n_grads_match_unroll_complex() {
    run_step_n_grads_match_unroll("complex grads", small_config(), 5, 1e-3);
}

#[test]
fn step_n_grads_match_unroll_quat() {
    run_step_n_grads_match_unroll("quat grads", quat_config(), 5, 1e-3);
}

// ---------------------------------------------------------------------------
// Upstream: Layers / LatentNetwork
// ---------------------------------------------------------------------------

/// A single-layer stack adds nothing on top of the block (the residual is
/// pointwise), so `Layers::step_n_approx` is exact there.
#[test]
fn layers_single_layer_step_n_is_exact() {
    let device: Device = Default::default();
    let cfg = small_config();
    let layers = LayersBuilder::new(1, cfg.clone()).init(&device);
    let x = Tensor::<2>::random([2, cfg.d_model], Distribution::Normal(0.0, 1.0), &device);

    let n = 6;
    let mut caches = None;
    let mut out = None;
    for _ in 0..n {
        let (o, c) = layers.step(x.clone(), caches, None, None);
        out = Some(o);
        caches = Some(c);
    }
    let (o_jump, _c) = layers.step_n_approx(x, n, None);
    let d = max_abs_diff(out.unwrap(), o_jump);
    assert!(d < 1e-4, "single-layer step_n_approx: {d:.6}");
}

/// Two stacked layers: `step_n_approx(1)` is exactly one `step`, the large-`n`
/// jump and `step_infinite` both converge to the unrolled fixed point (the
/// per-layer held-input error decays geometrically).
#[test]
fn network_two_layers_constant_input() {
    let device: Device = Default::default();
    let cfg = decaying(small_config());
    let net = LatentNetworkBuilder {
        input_size: 8,
        layers: LayersBuilder::new(2, cfg),
        output_size: 8,
        class_tokens: Vec::new(),
    }
    .init(&device);
    let x = Tensor::<2>::random([2, 8], Distribution::Normal(0.0, 1.0), &device);

    // n = 1 is exactly one step, even through the stack.
    let (o_step, _) = net.step(x.clone(), None, None, None, None);
    let (o_jump1, _) = net.step_n_approx(x.clone(), 1, None);
    let d1 = max_abs_diff(o_step, o_jump1);
    assert!(d1 < 1e-5, "two-layer step_n_approx(1) vs step: {d1:.6}");

    // Unrolled fixed point.
    let steps = 300;
    let mut caches = None;
    let mut out = None;
    for _ in 0..steps {
        let (o, c) = net.step(x.clone(), caches, None, None, None);
        out = Some(o);
        caches = Some(c);
    }
    let out = out.unwrap();

    let y_inf = net.step_infinite(x.clone());
    let d_inf = max_abs_diff(out.clone(), y_inf);
    assert!(d_inf < 1e-3, "two-layer step_infinite vs unroll: {d_inf:.6}");

    let (y_n, _caches) = net.step_n_approx(x, steps, None);
    let d_n = max_abs_diff(out, y_n);
    assert!(d_n < 1e-3, "two-layer step_n_approx({steps}) vs unroll: {d_n:.6}");
}
