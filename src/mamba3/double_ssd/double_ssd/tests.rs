use super::*;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

fn small_config() -> Mamba3Config {
    Mamba3Config::new(32) // d_model = 32
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

/// A bundle of input + model-parameter gradients extracted from one
/// forward+backward run.  Each `check_grads_match` call compares these
/// across two runs that should be mathematically equivalent.
struct RunGrads {
    out: Tensor<3>,
    /// Final SSM hidden state from the returned cache.
    final_ssm: Tensor<4>,
    /// Final previous-token B state from the returned cache.
    final_k: Tensor<4>,
    /// Final previous-token x state from the returned cache.
    final_v: Tensor<3>,
    /// Final cumulative RoPE angle from the returned cache.
    final_angle: Tensor<3>,
    d_input: Tensor<3>,
    d_in_proj_w: Tensor<2>,
    d_dt_bias: Tensor<1>,
    d_d: Tensor<1>,
    d_b_norm_gamma: Tensor<1>,
    d_c_norm_gamma: Tensor<1>,
    d_b_bias: Tensor<3>,
    d_c_bias: Tensor<3>,
    d_out_proj_w: Tensor<2>,
}

/// Fixed (non-tracked) random "downstream heads" used to form a scalar loss
/// from the output **and** every final cache field, so the backward pass
/// exercises both the output and the state path.
struct Heads {
    out: Tensor<3>,
    ssm: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<3>,
    angle: Tensor<3>,
}

/// Build the initial cache passed to both `forward` and the `step`
/// unrolling. With `random = false` it is zero (the standard fresh start);
/// with `random = true` every field (SSM state, previous-token B/x, and
/// cumulative RoPE angle) holds random values, exercising parity from an
/// arbitrary initial state.
fn build_init_cache(cfg: &Mamba3Config, batch: usize, random: bool) -> Mamba3DoubleSsdCache {
    let device: Device = Default::default();
    let nheads = cfg.nheads();
    let per_head_dim = cfg.per_head_dim;
    let state_rank = cfg.state_rank;
    let mimo_rank = cfg.mimo_rank;
    let num_rope_angles = cfg.num_rope_angles();
    let dist = Distribution::Normal(0.0, 1.0);
    let mk4 = |shape: [usize; 4]| {
        let t = if random {
            Tensor::<4>::random(shape, dist, &device)
        } else {
            Tensor::<4>::zeros(shape, &device)
        };
        Tensor::from_inner(t)
    };
    let mk3 = |shape: [usize; 3]| {
        let t = if random {
            Tensor::<3>::random(shape, dist, &device)
        } else {
            Tensor::<3>::zeros(shape, &device)
        };
        Tensor::from_inner(t)
    };
    Mamba3DoubleSsdCache {
        ssm_bhpr: mk4([batch, nheads, per_head_dim, state_rank]),
        k_state_bmhr: mk4([batch, mimo_rank, nheads, state_rank]),
        v_state_bhp: mk3([batch, nheads, per_head_dim]),
        cum_angle_bha: mk3([batch, nheads, num_rope_angles]),
    }
}

/// Compare the output and every final cache field of two runs.
fn assert_outputs_match(label: &str, a: &RunGrads, b: &RunGrads, tol: f32) {
    use crate::utils::test_helpers::max_abs_diff;
    let checks = [
        ("output", max_abs_diff(a.out.clone(), b.out.clone())),
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
    for (name, d) in checks {
        assert!(d < tol, "{label}: {name} max abs diff = {d:.6} (tol {tol})");
    }
}

/// Run a closure that produces an output tensor from a model and an input
/// (wrapped as a `Param` so it has its own autodiff leaf), then derive a
/// scalar loss with a fixed (non-tracked) random "head" and return the
/// gradients of the input and a representative set of model parameters.
fn run_with_grads(
    model: &Mamba3,
    input: &Param<Tensor<3>>,
    heads: &Heads,
    forward: impl FnOnce(&Mamba3, Tensor<3>) -> (Tensor<3>, Mamba3DoubleSsdCache),
) -> RunGrads {
    let (out, cache) = forward(model, input.val());
    let out_inner = out.clone().inner();
    let ssm = cache.ssm_bhpr;
    let k = cache.k_state_bmhr;
    let v = cache.v_state_bhp;
    let angle = cache.cum_angle_bha;
    let final_ssm = ssm.clone().inner();
    let final_k = k.clone().inner();
    let final_v = v.clone().inner();
    let final_angle = angle.clone().inner();

    // Loss couples the output and every final cache field (each via its own
    // random head) so parameter gradients reflect both output and state.
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

    RunGrads {
        out: out_inner,
        final_ssm,
        final_k,
        final_v,
        final_angle,
        d_input: input.val().grad(&grads).expect("grad input"),
        d_in_proj_w: model
            .in_proj
            .weight
            .val()
            .grad(&grads)
            .expect("grad in_proj.weight"),
        d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("grad dt_bias_h"),
        d_d: model.d_h.val().grad(&grads).expect("grad d_h"),
        d_b_norm_gamma: model
            .b_norm
            .gamma
            .val()
            .grad(&grads)
            .expect("grad b_norm.gamma"),
        d_c_norm_gamma: model
            .c_norm
            .gamma
            .val()
            .grad(&grads)
            .expect("grad c_norm.gamma"),
        d_b_bias: model
            .b_bias_hmr
            .val()
            .grad(&grads)
            .expect("grad b_bias_hmr"),
        d_c_bias: model
            .c_bias_hmr
            .val()
            .grad(&grads)
            .expect("grad c_bias_hmr"),
        d_out_proj_w: model
            .out_proj
            .weight
            .val()
            .grad(&grads)
            .expect("grad out_proj.weight"),
    }
}

/// Assert that every entry in `a` and `b` agrees to within `grad_tol`,
/// printing every comparison so a failure dump shows the full picture
/// (instead of stopping at the first mismatch).
fn check_grads_match(label: &str, a: &RunGrads, b: &RunGrads, grad_tol: f32) {
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($field:ident, $name:expr) => {{
            let d = (a.$field.clone() - b.$field.clone())
                .abs()
                .max()
                .into_scalar::<f32>();
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

/// Build a fresh `Param<Tensor>` from a stable inner tensor.
/// A new Param is needed per run so that the autodiff leaf has a fresh
/// node, isolating each backward pass to its own forward graph.
fn param_input(input: &Tensor<3>) -> Param<Tensor<3>> {
    Param::from_tensor(Tensor::from_inner(input.clone()))
}

/// `forward(x)` is mathematically equivalent to repeatedly calling `step`
/// token-by-token from the **same** initial cache. Outputs, every final
/// cache field (SSM state, previous-token B/x, cumulative RoPE angle), and
/// parameter gradients must all agree up to float-summation-order noise.
///
/// With `random_init = true` the shared initial cache is random rather than
/// zero. Parity from an arbitrary initial state subsumes the chunked-prefill
/// (split-vs-full) guarantee: if `forward` from any state matches the
/// recurrent unrolling from that same state — outputs *and* final cache —
/// then feeding a `forward`-produced cache back in continues correctly.
fn run_step_matches_forward(cfg: Mamba3Config, random_init: bool) {
    let device: Device = Default::default();
    let model = cfg.init(&device.clone().autodiff());

    let batch = 2;
    let seq_len = 5;
    let d_model = cfg.d_model;
    let nheads = cfg.nheads();
    let per_head_dim = cfg.per_head_dim;
    let state_rank = cfg.state_rank;
    let mimo_rank = cfg.mimo_rank;
    let num_rope_angles = cfg.num_rope_angles();
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<3>::random([batch, seq_len, d_model], normal, &device),
        ssm: Tensor::<4>::random([batch, nheads, per_head_dim, state_rank], normal, &device),
        k: Tensor::<4>::random([batch, mimo_rank, nheads, state_rank], normal, &device),
        v: Tensor::<3>::random([batch, nheads, per_head_dim], normal, &device),
        angle: Tensor::<3>::random([batch, nheads, num_rope_angles], normal, &device),
    };

    let ssd_path = Mamba3SsdPath::Minimal(Some(4));
    let init_cache = build_init_cache(&cfg, batch, random_init);

    let input_fwd = param_input(&input);
    let cache_fwd = init_cache.clone();
    let path_fwd = ssd_path.clone();
    let r_fwd = run_with_grads(&model, &input_fwd, &heads, |m, x| {
        m.forward_double_ssd(x, Some(cache_fwd), &path_fwd)
    });

    let input_step = param_input(&input);
    let cache_step = init_cache;
    let r_step = run_with_grads(&model, &input_step, &heads, |m, x| {
        let mut cache: Option<Mamba3DoubleSsdCache> = Some(cache_step);
        let mut outs: Vec<Tensor<2>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
            let (out_t, new_cache) = m.step_double_ssd(token, cache);
            cache = Some(new_cache);
            outs.push(out_t);
        }
        (Tensor::stack(outs, 1), cache.unwrap())
    });

    assert_outputs_match("step vs forward", &r_fwd, &r_step, 1e-4);
    check_grads_match("step vs forward", &r_fwd, &r_step, 1e-3);

    // ── Guard: the random initial state must actually be consumed ─────
    // Re-run forward from a *zero* initial cache; its output must differ
    // from the random-init output. Otherwise the initial state is being
    // silently ignored and forward/step would match trivially.
    if random_init {
        use crate::utils::test_helpers::max_abs_diff;
        let (out_zero, _) = model.forward_double_ssd(
            Tensor::from_inner(input.clone()),
            Some(build_init_cache(&cfg, batch, false)),
            &ssd_path,
        );
        let d = max_abs_diff(r_fwd.out.clone(), out_zero.inner());
        assert!(
            d > 1e-3,
            "random initial state appears ignored: random-init vs zero-init \
             output max abs diff = {d:.6} (expected a clear difference)"
        );
    }
}

// Config variants exercised by the parity tests below.
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
fn cfg_rope_half() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_rope_fraction(0.5)
}
fn cfg_rope_half_mimo() -> Mamba3Config {
    cfg_rope_half().with_mimo_rank(2)
}
fn cfg_outproj_norm() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_has_outproj_norm(true)
}
fn cfg_outproj_norm_mimo() -> Mamba3Config {
    cfg_outproj_norm().with_mimo_rank(2)
}
fn cfg_rope_half_outproj_norm_mimo() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_mimo_rank(2)
        .with_rope_fraction(0.5)
        .with_has_outproj_norm(true)
}

#[test]
fn step_matches_forward() {
    run_step_matches_forward(small_config(), false);
}

#[test]
fn step_matches_forward_random_init() {
    run_step_matches_forward(small_config(), true);
}

#[test]
fn step_matches_forward_ngroups2() {
    run_step_matches_forward(cfg_ngroups2(), false);
}

#[test]
fn step_matches_forward_ngroups2_random_init() {
    run_step_matches_forward(cfg_ngroups2(), true);
}

#[test]
fn step_matches_forward_mimo() {
    run_step_matches_forward(small_config_mimo(), false);
}

#[test]
fn step_matches_forward_mimo_random_init() {
    run_step_matches_forward(small_config_mimo(), true);
}

#[test]
fn step_matches_forward_mimo_ngroups2() {
    run_step_matches_forward(cfg_mimo_ngroups2(), false);
}

#[test]
fn step_matches_forward_mimo_ngroups2_random_init() {
    run_step_matches_forward(cfg_mimo_ngroups2(), true);
}

// ── rope_fraction = 0.5 (partial RoPE) ──────────────────────────────────

#[test]
fn step_matches_forward_rope_half() {
    run_step_matches_forward(cfg_rope_half(), false);
}

#[test]
fn step_matches_forward_rope_half_random_init() {
    run_step_matches_forward(cfg_rope_half(), true);
}

#[test]
fn step_matches_forward_rope_half_mimo() {
    run_step_matches_forward(cfg_rope_half_mimo(), false);
}

#[test]
fn step_matches_forward_rope_half_mimo_random_init() {
    run_step_matches_forward(cfg_rope_half_mimo(), true);
}

// ── has_outproj_norm = true (gated RMSNorm) ─────────────────────────────

#[test]
fn step_matches_forward_outproj_norm() {
    run_step_matches_forward(cfg_outproj_norm(), false);
}

#[test]
fn step_matches_forward_outproj_norm_random_init() {
    run_step_matches_forward(cfg_outproj_norm(), true);
}

#[test]
fn step_matches_forward_outproj_norm_mimo() {
    run_step_matches_forward(cfg_outproj_norm_mimo(), false);
}

#[test]
fn step_matches_forward_outproj_norm_mimo_random_init() {
    run_step_matches_forward(cfg_outproj_norm_mimo(), true);
}

// ── Both features combined ──────────────────────────────────────────────

#[test]
fn step_matches_forward_rope_half_outproj_norm_mimo() {
    run_step_matches_forward(cfg_rope_half_outproj_norm_mimo(), false);
}

#[test]
fn step_matches_forward_rope_half_outproj_norm_mimo_random_init() {
    run_step_matches_forward(cfg_rope_half_outproj_norm_mimo(), true);
}
