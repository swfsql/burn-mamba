use super::*;
use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::mamba3::Mamba3Config;
use burn::backend::{Autodiff, Flex};
use burn::module::Param;
use burn::tensor::Distribution;

type InnerB = Flex;
type B = Autodiff<InnerB>;
type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

fn small_config() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
}

fn small_config_mimo() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_mimo_rank(2)
}

fn cfg_ngroups2() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(16)
        .with_ngroups(2)
}

fn cfg_mimo_ngroups2() -> Mamba3Config {
    cfg_ngroups2().with_mimo_rank(2)
}

/// Build a matched pair of initial caches for cross-algorithm parity
/// (`forward_double_ssd`/`step_double_ssd` use [`Mamba3DoubleSsdCache`];
/// `forward_single_ssd` uses [`Mamba3SingleSsdCache`]).
/// With `random = true` the SSM state and cumulative RoPE angle are random while
/// the previous-token K/V history is **zero** — so the single-ssd form's
/// boundary-β seed is zero and both forms share the exact same logical initial state.
/// With `random = false` everything is zero.
fn build_cross_caches(
    cfg: &Mamba3Config,
    batch: usize,
    random: bool,
) -> (Mamba3DoubleSsdCache<B>, Mamba3SingleSsdCache<B>) {
    let device: Device = Default::default();
    let nheads = cfg.nheads();
    let per_head_dim = cfg.per_head_dim;
    let state_rank = cfg.state_rank;
    let mimo_rank = cfg.mimo_rank;
    let num_rope_angles = cfg.num_rope_angles();
    let dist = Distribution::Normal(0.0, 1.0);
    let ssm = if random {
        Tensor::<InnerB, 4>::random([batch, nheads, per_head_dim, state_rank], dist, &device)
    } else {
        Tensor::<InnerB, 4>::zeros([batch, nheads, per_head_dim, state_rank], &device)
    };
    let angle = if random {
        Tensor::<InnerB, 3>::random([batch, nheads, num_rope_angles], dist, &device)
    } else {
        Tensor::<InnerB, 3>::zeros([batch, nheads, num_rope_angles], &device)
    };
    // Zero previous-token history so the two cache forms agree logically.
    let k = Tensor::<InnerB, 4>::zeros([batch, mimo_rank, nheads, state_rank], &device);
    let v = Tensor::<InnerB, 3>::zeros([batch, nheads, per_head_dim], &device);
    let c3 = Mamba3DoubleSsdCache {
        ssm_bhpr: Tensor::from_inner(ssm.clone()),
        k_state_bmhr: Tensor::from_inner(k.clone()),
        v_state_bhp: Tensor::from_inner(v.clone()),
        cum_angle_bha: Tensor::from_inner(angle.clone()),
    };
    let cm = Mamba3SingleSsdCache {
        ssm_bhpr: Tensor::from_inner(ssm),
        k_state_bmhr: Tensor::from_inner(k),
        v_state_bhp: Tensor::from_inner(v),
        cum_angle_bha: Tensor::from_inner(angle),
    };
    (c3, cm)
}

/// Build an initial [`Mamba3SingleSsdCache`] for the single-ssd form continuity test.
/// With `random = true` *every* field (including the previous-token K/V
/// history) is random, exercising forward_single_ssd continuation from an arbitrary
/// single-ssd form state.
fn build_single_ssd_cache(
    cfg: &Mamba3Config,
    batch: usize,
    random: bool,
) -> Mamba3SingleSsdCache<B> {
    let device: Device = Default::default();
    let nheads = cfg.nheads();
    let per_head_dim = cfg.per_head_dim;
    let state_rank = cfg.state_rank;
    let mimo_rank = cfg.mimo_rank;
    let num_rope_angles = cfg.num_rope_angles();
    let dist = Distribution::Normal(0.0, 1.0);
    let mk4 = |shape: [usize; 4]| {
        let t = if random {
            Tensor::<InnerB, 4>::random(shape, dist, &device)
        } else {
            Tensor::<InnerB, 4>::zeros(shape, &device)
        };
        Tensor::from_inner(t)
    };
    let mk3 = |shape: [usize; 3]| {
        let t = if random {
            Tensor::<InnerB, 3>::random(shape, dist, &device)
        } else {
            Tensor::<InnerB, 3>::zeros(shape, &device)
        };
        Tensor::from_inner(t)
    };
    Mamba3SingleSsdCache {
        ssm_bhpr: mk4([batch, nheads, per_head_dim, state_rank]),
        k_state_bmhr: mk4([batch, mimo_rank, nheads, state_rank]),
        v_state_bhp: mk3([batch, nheads, per_head_dim]),
        cum_angle_bha: mk3([batch, nheads, num_rope_angles]),
    }
}

/// Per-run gradient bundle (subset of params; mirrors the equivalent struct
/// in `mamba3::tests` but kept local to avoid cross-module visibility).
struct RunGrads {
    out: Tensor<InnerB, 3>,
    d_input: Tensor<InnerB, 3>,
    d_in_proj_w: Tensor<InnerB, 2>,
    d_dt_bias: Tensor<InnerB, 1>,
    d_d: Tensor<InnerB, 1>,
    d_b_norm_gamma: Tensor<InnerB, 1>,
    d_c_norm_gamma: Tensor<InnerB, 1>,
    d_b_bias: Tensor<InnerB, 3>,
    d_c_bias: Tensor<InnerB, 3>,
    d_out_proj_w: Tensor<InnerB, 2>,
}

fn run_with_grads(
    model: &Mamba3<B>,
    input: &Param<Tensor<B, 3>>,
    head: &Tensor<InnerB, 3>,
    forward: impl FnOnce(&Mamba3<B>, Tensor<B, 3>) -> Tensor<B, 3>,
) -> RunGrads {
    let out = forward(model, input.val());
    let out_inner = out.clone().inner();
    let head = Tensor::from_inner(head.clone());
    let loss = (out * head).sum();
    let grads = loss.backward();
    RunGrads {
        out: out_inner,
        d_input: input.val().grad(&grads).expect("grad input"),
        d_in_proj_w: model
            .in_proj
            .weight
            .val()
            .grad(&grads)
            .expect("in_proj.weight"),
        d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("dt_bias_h"),
        d_d: model.d_h.val().grad(&grads).expect("d_h"),
        d_b_norm_gamma: model.b_norm.gamma.val().grad(&grads).expect("b_norm.gamma"),
        d_c_norm_gamma: model.c_norm.gamma.val().grad(&grads).expect("c_norm.gamma"),
        d_b_bias: model.b_bias_hmr.val().grad(&grads).expect("b_bias_hmr"),
        d_c_bias: model.c_bias_hmr.val().grad(&grads).expect("c_bias_hmr"),
        d_out_proj_w: model
            .out_proj
            .weight
            .val()
            .grad(&grads)
            .expect("out_proj.weight"),
    }
}

fn check_grads_match(label: &str, a: &RunGrads, b: &RunGrads, grad_tol: f32) {
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($field:ident, $name:expr) => {{
            let d = (a.$field.clone() - b.$field.clone())
                .abs()
                .max()
                .into_scalar();
            eprintln!("{:>40} {:>16} | max abs diff = {:>10.6}", label, $name, d);
            if d >= grad_tol {
                failures.push(format!(
                    "{}: grad of {} max abs diff = {:.6} (tol {})",
                    label, $name, d, grad_tol
                ));
            }
        }};
    }
    check!(d_input, "input");
    check!(d_in_proj_w, "in_proj.weight");
    check!(d_dt_bias, "dt_bias_h");
    check!(d_d, "d_h");
    check!(d_b_norm_gamma, "b_norm.gamma");
    check!(d_c_norm_gamma, "c_norm.gamma");
    check!(d_b_bias, "b_bias_hmr");
    check!(d_c_bias, "c_bias_hmr");
    check!(d_out_proj_w, "out_proj.weight");
    assert!(
        failures.is_empty(),
        "gradient mismatches:\n  {}",
        failures.join("\n  ")
    );
}

fn param_input(input: &Tensor<InnerB, 3>) -> Param<Tensor<B, 3>> {
    Param::from_tensor(Tensor::from_inner(input.clone()))
}

/// Random downstream heads for the single-ssd form continuity loss (output plus
/// every single-ssd cache field).
struct Heads {
    out: Tensor<InnerB, 3>,
    ssm: Tensor<InnerB, 4>,
    k: Tensor<InnerB, 4>,
    v: Tensor<InnerB, 3>,
    angle: Tensor<InnerB, 3>,
}

/// A [`RunGrads`] plus the final single-ssd cache fields, for the continuity test.
struct SingleSsdRun {
    rg: RunGrads,
    final_ssm: Tensor<InnerB, 4>,
    final_k: Tensor<InnerB, 4>,
    final_v: Tensor<InnerB, 3>,
    final_angle: Tensor<InnerB, 3>,
}

/// Like [`run_with_grads`] but the loss couples the output with every final
/// single-ssd cache field, and the final cache is returned for comparison. Both
/// runs being compared use `forward_single_ssd`, so the single-ssd cache semantics match.
fn run_with_grads_single_ssd(
    model: &Mamba3<B>,
    input: &Param<Tensor<B, 3>>,
    heads: &Heads,
    runner: impl FnOnce(&Mamba3<B>, Tensor<B, 3>) -> (Tensor<B, 3>, Mamba3SingleSsdCache<B>),
) -> SingleSsdRun {
    let (out, cache) = runner(model, input.val());
    let out_inner = out.clone().inner();
    let ssm = cache.ssm_bhpr;
    let k = cache.k_state_bmhr;
    let v = cache.v_state_bhp;
    let angle = cache.cum_angle_bha;
    let final_ssm = ssm.clone().inner();
    let final_k = k.clone().inner();
    let final_v = v.clone().inner();
    let final_angle = angle.clone().inner();

    let out_head = Tensor::from_inner(heads.out.clone());
    let ssm_head = Tensor::from_inner(heads.ssm.clone());
    let k_head = Tensor::from_inner(heads.k.clone());
    let v_head = Tensor::from_inner(heads.v.clone());
    let angle_head = Tensor::from_inner(heads.angle.clone());
    let loss = (out * out_head).sum()
        + (ssm * ssm_head).sum()
        + (k * k_head).sum()
        + (v * v_head).sum()
        + (angle * angle_head).sum();
    let grads = loss.backward();

    let rg = RunGrads {
        out: out_inner,
        d_input: input.val().grad(&grads).expect("grad input"),
        d_in_proj_w: model
            .in_proj
            .weight
            .val()
            .grad(&grads)
            .expect("in_proj.weight"),
        d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("dt_bias_h"),
        d_d: model.d_h.val().grad(&grads).expect("d_h"),
        d_b_norm_gamma: model.b_norm.gamma.val().grad(&grads).expect("b_norm.gamma"),
        d_c_norm_gamma: model.c_norm.gamma.val().grad(&grads).expect("c_norm.gamma"),
        d_b_bias: model.b_bias_hmr.val().grad(&grads).expect("b_bias_hmr"),
        d_c_bias: model.c_bias_hmr.val().grad(&grads).expect("c_bias_hmr"),
        d_out_proj_w: model
            .out_proj
            .weight
            .val()
            .grad(&grads)
            .expect("out_proj.weight"),
    };
    SingleSsdRun {
        rg,
        final_ssm,
        final_k,
        final_v,
        final_angle,
    }
}

/// Compare output, every final single-ssd cache field, and parameter gradients.
fn check_single_ssd_match(
    label: &str,
    a: &SingleSsdRun,
    b: &SingleSsdRun,
    val_tol: f32,
    grad_tol: f32,
) {
    use crate::utils::test_helpers::max_abs_diff;
    let vals = [
        ("output", max_abs_diff(a.rg.out.clone(), b.rg.out.clone())),
        (
            "final ssm",
            max_abs_diff(a.final_ssm.clone(), b.final_ssm.clone()),
        ),
        (
            "final k_state",
            max_abs_diff(a.final_k.clone(), b.final_k.clone()),
        ),
        (
            "final v_state",
            max_abs_diff(a.final_v.clone(), b.final_v.clone()),
        ),
        (
            "final cum_angle",
            max_abs_diff(a.final_angle.clone(), b.final_angle.clone()),
        ),
    ];
    for (name, d) in vals {
        assert!(
            d < val_tol,
            "{label}: {name} max abs diff = {d:.6} (tol {val_tol})"
        );
    }
    check_grads_match(label, &a.rg, &b.rg, grad_tol);
}

/// Guard: a random initial state must actually change the forward_single_ssd output
/// (vs a *zero* single-ssd cache). Otherwise the initial state is being silently
/// ignored, which would make the parity comparisons pass trivially.
fn guard_random_init_consumed(
    random_init: bool,
    model: &Mamba3<B>,
    cfg: &Mamba3Config,
    batch: usize,
    input: &Tensor<InnerB, 3>,
    ssd_path: &Mamba3SsdPath,
    random_out: &Tensor<InnerB, 3>,
) {
    if !random_init {
        return;
    }
    use crate::utils::test_helpers::max_abs_diff;
    let (out_zero, _) = model.forward_single_ssd(
        Tensor::from_inner(input.clone()),
        Some(build_single_ssd_cache(cfg, batch, false)),
        Mamba3SingleSsdPath::from(ssd_path.clone()),
    );
    let d = max_abs_diff(random_out.clone(), out_zero.inner());
    assert!(
        d > 1e-3,
        "random initial state appears ignored: random-init vs zero-init \
         output max abs diff = {d:.6} (expected a clear difference)"
    );
}

/// forward_single_ssd ≡ forward_double_ssd on values and gradients, from the same
/// initial state. With `random_init = true` the shared logical initial state
/// is random (random SSM state + cumulative RoPE angle; zero previous-token
/// history so the single-ssd and double-ssd forms coincide). The output and all
/// parameter gradients must agree. The single-ssd cache SSM accumulator itself is
/// not compared here (different semantics from the double-form state); the
/// single-ssd cache is compared in `run_forward_single_ssd_split_matches_full`.
fn forward_match(cfg: Mamba3Config, ssd_path: Mamba3SsdPath, random_init: bool) {
    let device: Device = Default::default();
    let model = cfg.init::<B>(&device);

    let batch = 2;
    let seq_len = 5;
    let d_model = cfg.d_model;
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
    let head = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);

    let (c3, cm) = build_cross_caches(&cfg, batch, random_init);

    let input_a = param_input(&input);
    let c3c = c3;
    let path_a = Mamba3DoubleSsdPath::from(ssd_path.clone());
    let r_fwd_double_ssd = run_with_grads(&model, &input_a, &head, |m, x| {
        let (out, _) = m.forward_double_ssd(x, Some(c3c), path_a);
        out
    });

    let input_b = param_input(&input);
    let cmc = cm;
    let single_ssd_b = Mamba3SingleSsdPath::from(ssd_path.clone());
    let r_fwd_single_ssd = run_with_grads(&model, &input_b, &head, |m, x| {
        let (out, _) = m.forward_single_ssd(x, Some(cmc), single_ssd_b);
        out
    });

    let diff = (r_fwd_double_ssd.out.clone() - r_fwd_single_ssd.out.clone())
        .abs()
        .max()
        .into_scalar();
    assert!(
        diff < 1e-4,
        "forward_double_ssd vs forward_single_ssd max absolute difference = {diff:.6} (expected < 1e-4)"
    );
    check_grads_match(
        "forward_single_ssd vs forward_double_ssd",
        &r_fwd_double_ssd,
        &r_fwd_single_ssd,
        1e-3,
    );

    guard_random_init_consumed(
        random_init,
        &model,
        &cfg,
        batch,
        &input,
        &ssd_path,
        &r_fwd_single_ssd.out,
    );
}

#[test]
fn forward_match_simple() {
    forward_match(small_config(), Mamba3SsdPath::Minimal(Some(4)), false);
}

#[test]
fn forward_match_random_init() {
    forward_match(small_config(), Mamba3SsdPath::Minimal(Some(4)), true);
}

#[test]
fn forward_match_ngroups2() {
    forward_match(cfg_ngroups2(), Mamba3SsdPath::Minimal(Some(4)), false);
}

#[test]
fn forward_match_ngroups2_random_init() {
    forward_match(cfg_ngroups2(), Mamba3SsdPath::Minimal(Some(4)), true);
}

#[test]
fn forward_match_mimo() {
    forward_match(small_config_mimo(), Mamba3SsdPath::Minimal(Some(4)), false);
}

#[test]
fn forward_match_mimo_random_init() {
    forward_match(small_config_mimo(), Mamba3SsdPath::Minimal(Some(4)), true);
}

#[test]
fn forward_match_mimo_ngroups2() {
    forward_match(cfg_mimo_ngroups2(), Mamba3SsdPath::Minimal(Some(4)), false);
}

#[test]
fn forward_match_mimo_ngroups2_random_init() {
    forward_match(cfg_mimo_ngroups2(), Mamba3SsdPath::Minimal(Some(4)), true);
}

#[test]
fn forward_match_serial() {
    forward_match(small_config(), Mamba3SsdPath::Serial(Some(4)), false);
}

#[test]
fn forward_match_serial_mimo() {
    forward_match(small_config_mimo(), Mamba3SsdPath::Serial(Some(4)), false);
}

#[test]
fn forward_match_recalc() {
    forward_match(
        small_config(),
        Mamba3SsdPath::SerialRecalculated(Some(4)),
        false,
    );
}

#[test]
fn forward_match_recalc_mimo() {
    forward_match(
        small_config_mimo(),
        Mamba3SsdPath::SerialRecalculated(Some(4)),
        false,
    );
}

/// forward_single_ssd ≡ token-by-token step on values and gradients, from the same
/// initial state (random when `random_init = true`, with zero previous-token
/// history so the single-ssd and recurrent forms coincide).
fn run_forward_single_ssd_matches_step(
    cfg: Mamba3Config,
    single_ssd_path: Mamba3SingleSsdPath,
    random_init: bool,
) {
    let device: Device = Default::default();
    let model = cfg.init::<B>(&device);

    let batch = 2;
    let seq_len = 5;
    let d_model = cfg.d_model;
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
    let head = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);

    let (_c3, cm) = build_cross_caches(&cfg, batch, random_init);

    let input_a = param_input(&input);
    let cmc = cm.clone();
    let single_ssd_a = single_ssd_path.clone();
    let r_fwd_single_ssd = run_with_grads(&model, &input_a, &head, |m, x| {
        let (out, _) = m.forward_single_ssd(x, Some(cmc), single_ssd_a);
        out
    });

    let input_b = param_input(&input);
    let cmc = cm;
    let r_step = run_with_grads(&model, &input_b, &head, |m, x| {
        let mut cache: Option<Mamba3SingleSsdCache<B>> = Some(cmc);
        let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
            let (out_t, new_cache) = m.step_single_ssd(token, cache);
            cache = Some(new_cache);
            outs.push(out_t);
        }
        Tensor::stack(outs, 1)
    });

    let diff = (r_fwd_single_ssd.out.clone() - r_step.out.clone())
        .abs()
        .max()
        .into_scalar();
    assert!(
        diff < 1e-4,
        "forward_single_ssd vs step max absolute difference = {diff:.6} (expected < 1e-4)"
    );
    check_grads_match(
        "forward_single_ssd vs step",
        &r_fwd_single_ssd,
        &r_step,
        1e-3,
    );

    guard_random_init_consumed(
        random_init,
        &model,
        &cfg,
        batch,
        &input,
        &(single_ssd_path.into()),
        &r_fwd_single_ssd.out,
    );
}

#[test]
fn forward_single_ssd_matches_step() {
    run_forward_single_ssd_matches_step(
        small_config(),
        Mamba3SingleSsdPath::Minimal(Some(4)),
        false,
    );
}

#[test]
fn forward_single_ssd_matches_step_random_init() {
    run_forward_single_ssd_matches_step(
        small_config(),
        Mamba3SingleSsdPath::Minimal(Some(4)),
        true,
    );
}

#[test]
fn forward_single_ssd_matches_step_mimo() {
    run_forward_single_ssd_matches_step(
        small_config_mimo(),
        Mamba3SingleSsdPath::Minimal(Some(4)),
        false,
    );
}

#[test]
fn forward_single_ssd_matches_step_mimo_random_init() {
    run_forward_single_ssd_matches_step(
        small_config_mimo(),
        Mamba3SingleSsdPath::Minimal(Some(4)),
        true,
    );
}

#[test]
fn forward_single_ssd_matches_step_serial() {
    run_forward_single_ssd_matches_step(
        small_config(),
        Mamba3SingleSsdPath::Serial(Some(4)),
        false,
    );
}

#[test]
fn forward_single_ssd_matches_step_serial_mimo() {
    run_forward_single_ssd_matches_step(
        small_config_mimo(),
        Mamba3SingleSsdPath::Serial(Some(4)),
        false,
    );
}

#[test]
fn forward_single_ssd_matches_step_recalc() {
    run_forward_single_ssd_matches_step(
        small_config(),
        Mamba3SingleSsdPath::SerialRecalculated(Some(4)),
        false,
    );
}

#[test]
fn forward_single_ssd_matches_step_recalc_mimo() {
    run_forward_single_ssd_matches_step(
        small_config_mimo(),
        Mamba3SingleSsdPath::SerialRecalculated(Some(4)),
        false,
    );
}

/// forward_single_ssd continuation from a **random** initial single-ssd cache:
/// `forward_single_ssd(full, cache) ≡ forward_single_ssd(prefix, cache)` then
/// `forward_single_ssd(suffix, mid_cache)`. Compares outputs, the final single-ssd cache,
/// and gradients. This replaces the old zero-init split-vs-full test: a
/// random initial cache subsumes the chunked-prefill continuity guarantee
/// from an arbitrary starting state, and the guard at the end confirms the
/// initial cache is actually consumed (not silently ignored).
fn run_forward_single_ssd_split_matches_full(
    cfg: Mamba3Config,
    single_ssd_path: Mamba3SingleSsdPath,
) {
    let device: Device = Default::default();
    let model = cfg.init::<B>(&device);

    let batch = 2;
    let seq_len = 6;
    let split = 2;
    let d_model = cfg.d_model;
    let nheads = cfg.nheads();
    let per_head_dim = cfg.per_head_dim;
    let state_rank = cfg.state_rank;
    let mimo_rank = cfg.mimo_rank;
    let num_rope_angles = cfg.num_rope_angles();
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
        ssm: Tensor::<InnerB, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            normal,
            &device,
        ),
        k: Tensor::<InnerB, 4>::random([batch, mimo_rank, nheads, state_rank], normal, &device),
        v: Tensor::<InnerB, 3>::random([batch, nheads, per_head_dim], normal, &device),
        angle: Tensor::<InnerB, 3>::random([batch, nheads, num_rope_angles], normal, &device),
    };

    let init_cache = build_single_ssd_cache(&cfg, batch, true);

    let input_full = param_input(&input);
    let cache_full = init_cache.clone();
    let single_ssd_f = single_ssd_path.clone();
    let r_full = run_with_grads_single_ssd(&model, &input_full, &heads, |m, x| {
        m.forward_single_ssd(x, Some(cache_full), single_ssd_f)
    });

    let input_split = param_input(&input);
    let cache_split = init_cache;
    let single_ssd_s = single_ssd_path.clone();
    let r_split = run_with_grads_single_ssd(&model, &input_split, &heads, |m, x| {
        let prefix = x.clone().narrow(1, 0, split);
        let suffix = x.narrow(1, split, seq_len - split);
        let (out_prefix, mid) =
            m.forward_single_ssd(prefix, Some(cache_split), single_ssd_s.clone());
        let (out_suffix, last) = m.forward_single_ssd(suffix, Some(mid), single_ssd_s);
        (Tensor::cat(vec![out_prefix, out_suffix], 1), last)
    });

    check_single_ssd_match(
        "forward_single_ssd split vs full",
        &r_full,
        &r_split,
        1e-4,
        1e-3,
    );

    // Guard: the random initial single_ssd cache must change the full output.
    {
        use crate::utils::test_helpers::max_abs_diff;
        let (out_zero, _) = model.forward_single_ssd(
            Tensor::from_inner(input.clone()),
            Some(build_single_ssd_cache(&cfg, batch, false)),
            single_ssd_path.clone(),
        );
        let d = max_abs_diff(r_full.rg.out.clone(), out_zero.inner());
        assert!(
            d > 1e-3,
            "random initial state appears ignored: random-init vs zero-init \
             output max abs diff = {d:.6} (expected a clear difference)"
        );
    }
}

#[test]
fn forward_single_ssd_split_matches_full() {
    run_forward_single_ssd_split_matches_full(
        small_config(),
        Mamba3SingleSsdPath::Minimal(Some(4)),
    );
}

#[test]
fn forward_single_ssd_split_matches_full_mimo() {
    run_forward_single_ssd_split_matches_full(
        small_config_mimo(),
        Mamba3SingleSsdPath::Minimal(Some(4)),
    );
}

#[test]
fn forward_single_ssd_split_matches_full_serial() {
    run_forward_single_ssd_split_matches_full(small_config(), Mamba3SingleSsdPath::Serial(Some(4)));
}

#[test]
fn forward_single_ssd_split_matches_full_serial_mimo() {
    run_forward_single_ssd_split_matches_full(
        small_config_mimo(),
        Mamba3SingleSsdPath::Serial(Some(4)),
    );
}

#[test]
fn forward_single_ssd_split_matches_full_recalc() {
    run_forward_single_ssd_split_matches_full(
        small_config(),
        Mamba3SingleSsdPath::SerialRecalculated(Some(4)),
    );
}

#[test]
fn forward_single_ssd_split_matches_full_recalc_mimo() {
    run_forward_single_ssd_split_matches_full(
        small_config_mimo(),
        Mamba3SingleSsdPath::SerialRecalculated(Some(4)),
    );
}

// ── Cross-pathway cache conversion parity ───────────────────────────────

/// Like [`run_with_grads_single_ssd`], but the runner hands back the four
/// final-cache field tensors directly (so the concrete cache type — single
/// or double — does not matter). The loss couples the output with every
/// final-cache field; gradients of the input and representative parameters
/// are returned alongside the (inner) output and final-cache values.
#[allow(clippy::type_complexity)]
fn run_cache_fields_with_grads(
    model: &Mamba3<B>,
    input: &Param<Tensor<B, 3>>,
    heads: &Heads,
    runner: impl FnOnce(
        &Mamba3<B>,
        Tensor<B, 3>,
    ) -> (
        Tensor<B, 3>, // out
        Tensor<B, 4>, // ssm_bhpr
        Tensor<B, 4>, // k_state_bmhr
        Tensor<B, 3>, // v_state_bhp
        Tensor<B, 3>, // cum_angle_bha
    ),
) -> SingleSsdRun {
    let (out, ssm, k, v, angle) = runner(model, input.val());
    let out_inner = out.clone().inner();
    let final_ssm = ssm.clone().inner();
    let final_k = k.clone().inner();
    let final_v = v.clone().inner();
    let final_angle = angle.clone().inner();

    let out_head = Tensor::from_inner(heads.out.clone());
    let ssm_head = Tensor::from_inner(heads.ssm.clone());
    let k_head = Tensor::from_inner(heads.k.clone());
    let v_head = Tensor::from_inner(heads.v.clone());
    let angle_head = Tensor::from_inner(heads.angle.clone());
    let loss = (out * out_head).sum()
        + (ssm * ssm_head).sum()
        + (k * k_head).sum()
        + (v * v_head).sum()
        + (angle * angle_head).sum();
    let grads = loss.backward();

    let rg = RunGrads {
        out: out_inner,
        d_input: input.val().grad(&grads).expect("grad input"),
        d_in_proj_w: model
            .in_proj
            .weight
            .val()
            .grad(&grads)
            .expect("in_proj.weight"),
        d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("dt_bias_h"),
        d_d: model.d_h.val().grad(&grads).expect("d_h"),
        d_b_norm_gamma: model.b_norm.gamma.val().grad(&grads).expect("b_norm.gamma"),
        d_c_norm_gamma: model.c_norm.gamma.val().grad(&grads).expect("c_norm.gamma"),
        d_b_bias: model.b_bias_hmr.val().grad(&grads).expect("b_bias_hmr"),
        d_c_bias: model.c_bias_hmr.val().grad(&grads).expect("c_bias_hmr"),
        d_out_proj_w: model
            .out_proj
            .weight
            .val()
            .grad(&grads)
            .expect("out_proj.weight"),
    };
    SingleSsdRun {
        rg,
        final_ssm,
        final_k,
        final_v,
        final_angle,
    }
}

/// Cache-conversion parity. From one shared, fully-random initial cache,
/// two consecutive forward calls split a sequence into prefix+suffix with a
/// cross-pathway cache conversion in between:
///
/// - **A**: `forward_double_ssd(prefix)` → convert (double→single) →
///   `forward_single_ssd(suffix)`.
/// - **B**: `forward_single_ssd(prefix)` → convert (single→double) →
///   `forward_double_ssd(suffix)`.
///
/// Both directions must yield the same concatenated output, the same final
/// cache (every field — compared directly, with no further conversion), and
/// the same parameter/input gradients. This exercises the `From` impls in
/// [`crate::mamba3::cache`] inside the autodiff graph (so the conversion must
/// also be gradient-transparent), and the mid-point cache always carries a
/// non-trivial previous-token K/V history.
fn run_cache_conversion_parity(cfg: Mamba3Config, ssd_path: Mamba3SsdPath) {
    let device: Device = Default::default();
    let model = cfg.init::<B>(&device);

    let batch = 2;
    let seq_len = 6;
    let split = 2;
    let d_model = cfg.d_model;
    let nheads = cfg.nheads();
    let per_head_dim = cfg.per_head_dim;
    let state_rank = cfg.state_rank;
    let mimo_rank = cfg.mimo_rank;
    let num_rope_angles = cfg.num_rope_angles();
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
        ssm: Tensor::<InnerB, 4>::random(
            [batch, nheads, per_head_dim, state_rank],
            normal,
            &device,
        ),
        k: Tensor::<InnerB, 4>::random([batch, mimo_rank, nheads, state_rank], normal, &device),
        v: Tensor::<InnerB, 3>::random([batch, nheads, per_head_dim], normal, &device),
        angle: Tensor::<InnerB, 3>::random([batch, nheads, num_rope_angles], normal, &device),
    };

    // Shared, fully-random initial cache fields (including the previous-token
    // K/V history) — both runs start from the exact same logical state.
    let init_ssm =
        Tensor::<InnerB, 4>::random([batch, nheads, per_head_dim, state_rank], normal, &device);
    let init_k =
        Tensor::<InnerB, 4>::random([batch, mimo_rank, nheads, state_rank], normal, &device);
    let init_v = Tensor::<InnerB, 3>::random([batch, nheads, per_head_dim], normal, &device);
    let init_angle = Tensor::<InnerB, 3>::random([batch, nheads, num_rope_angles], normal, &device);

    let path_double = Mamba3DoubleSsdPath::from(ssd_path.clone());
    let path_single = Mamba3SingleSsdPath::from(ssd_path);

    // ── Run A: double → (convert) → single ───────────────────────────────
    let input_a = param_input(&input);
    let (pd_a, ps_a) = (path_double.clone(), path_single.clone());
    let (ssm_a, k_a, v_a, ang_a) = (
        init_ssm.clone(),
        init_k.clone(),
        init_v.clone(),
        init_angle.clone(),
    );
    let run_a = run_cache_fields_with_grads(&model, &input_a, &heads, move |m, x| {
        let init_double = Mamba3DoubleSsdCache {
            ssm_bhpr: Tensor::from_inner(ssm_a),
            k_state_bmhr: Tensor::from_inner(k_a),
            v_state_bhp: Tensor::from_inner(v_a),
            cum_angle_bha: Tensor::from_inner(ang_a),
        };
        let prefix = x.clone().narrow(1, 0, split);
        let suffix = x.narrow(1, split, seq_len - split);
        let (out_prefix, mid_double) = m.forward_double_ssd(prefix, Some(init_double), pd_a);
        let mid_single = Mamba3SingleSsdCache::from(mid_double);
        let (out_suffix, last) = m.forward_single_ssd(suffix, Some(mid_single), ps_a);
        let out = Tensor::cat(vec![out_prefix, out_suffix], 1);
        (
            out,
            last.ssm_bhpr,
            last.k_state_bmhr,
            last.v_state_bhp,
            last.cum_angle_bha,
        )
    });

    // ── Run B: single → (convert) → double ───────────────────────────────
    let input_b = param_input(&input);
    let (pd_b, ps_b) = (path_double, path_single);
    let (ssm_b, k_b, v_b, ang_b) = (init_ssm, init_k, init_v, init_angle);
    let run_b = run_cache_fields_with_grads(&model, &input_b, &heads, move |m, x| {
        let init_single = Mamba3SingleSsdCache {
            ssm_bhpr: Tensor::from_inner(ssm_b),
            k_state_bmhr: Tensor::from_inner(k_b),
            v_state_bhp: Tensor::from_inner(v_b),
            cum_angle_bha: Tensor::from_inner(ang_b),
        };
        let prefix = x.clone().narrow(1, 0, split);
        let suffix = x.narrow(1, split, seq_len - split);
        let (out_prefix, mid_single) = m.forward_single_ssd(prefix, Some(init_single), ps_b);
        let mid_double = Mamba3DoubleSsdCache::from(mid_single);
        let (out_suffix, last) = m.forward_double_ssd(suffix, Some(mid_double), pd_b);
        let out = Tensor::cat(vec![out_prefix, out_suffix], 1);
        (
            out,
            last.ssm_bhpr,
            last.k_state_bmhr,
            last.v_state_bhp,
            last.cum_angle_bha,
        )
    });

    check_single_ssd_match(
        "cache conversion parity (double↔single)",
        &run_a,
        &run_b,
        1e-4,
        1e-3,
    );
}

#[test]
fn cache_conversion_parity() {
    run_cache_conversion_parity(small_config(), Mamba3SsdPath::Minimal(Some(4)));
}

#[test]
fn cache_conversion_parity_mimo() {
    run_cache_conversion_parity(small_config_mimo(), Mamba3SsdPath::Minimal(Some(4)));
}

#[test]
fn cache_conversion_parity_ngroups2() {
    run_cache_conversion_parity(cfg_ngroups2(), Mamba3SsdPath::Minimal(Some(4)));
}

#[test]
fn cache_conversion_parity_mimo_ngroups2() {
    run_cache_conversion_parity(cfg_mimo_ngroups2(), Mamba3SsdPath::Minimal(Some(4)));
}

#[test]
fn cache_conversion_parity_serial() {
    run_cache_conversion_parity(small_config(), Mamba3SsdPath::Serial(Some(4)));
}

#[test]
fn cache_conversion_parity_recalc_mimo() {
    run_cache_conversion_parity(
        small_config_mimo(),
        Mamba3SsdPath::SerialRecalculated(Some(4)),
    );
}
