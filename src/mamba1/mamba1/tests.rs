use super::*;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

fn small_config() -> Mamba1Config {
    Mamba1Config::new(32) // d_model = 32
        .with_state_rank(8)
        .with_conv_kernel(4)
        .with_expand(2)
}

/// A bundle of input + model-parameter gradients extracted from one
/// forward+backward run.  Each `check_grads_match` call compares these
/// across two runs that should be mathematically equivalent.
struct RunGrads {
    out: Tensor<3>,
    /// Final convolution window from the returned cache.
    final_conv: Tensor<3>,
    /// Final SSM state from the returned cache.
    final_ssm: Tensor<3>,
    d_input: Tensor<3>,
    d_in_proj_w: Tensor<2>,
    d_conv1d_w: Tensor<3>,
    d_x_proj_w: Tensor<2>,
    d_dt_proj_w: Tensor<2>,
    d_dt_proj_b: Tensor<1>,
    d_a_log: Tensor<2>,
    d_d: Tensor<1>,
    d_out_proj_w: Tensor<2>,
}

/// Fixed (non-tracked) random "downstream heads" used to form a scalar loss
/// from the output **and** the final cache, so the backward pass exercises
/// both the output and the state path.
struct Heads {
    out: Tensor<3>,
    conv: Tensor<3>,
    ssm: Tensor<3>,
}

/// Run a closure that produces an output tensor from a model and an input
/// (wrapped as a `Param` so it has its own autodiff leaf), then derive a
/// scalar loss with a fixed (non-tracked) random "head" and return the
/// gradients of the input and a representative set of model parameters.
fn run_with_grads(
    model: &Mamba1,
    input: &Param<Tensor<3>>,
    heads: &Heads,
    forward: impl FnOnce(&Mamba1, Tensor<3>) -> (Tensor<3>, Mamba1Cache),
) -> RunGrads {
    let (out, cache) = forward(model, input.val());
    let out_inner = out.clone().inner();
    let conv = cache.conv_bik.clone();
    let ssm = cache.ssm_bir.clone();
    let final_conv = conv.clone().inner();
    let final_ssm = ssm.clone().inner();

    // Loss couples the output and the final cache (each via its own random
    // head) so parameter gradients reflect both the output and state paths.
    let out_head = Tensor::from_inner(heads.out.clone());
    let conv_head = Tensor::from_inner(heads.conv.clone());
    let ssm_head = Tensor::from_inner(heads.ssm.clone());
    let loss = (out * out_head).sum() + (conv * conv_head).sum() + (ssm * ssm_head).sum();
    let grads = loss.backward();

    RunGrads {
        out: out_inner,
        final_conv,
        final_ssm,
        d_input: input.val().grad(&grads).expect("grad input"),
        d_in_proj_w: model
            .in_proj
            .weight
            .val()
            .grad(&grads)
            .expect("grad in_proj.weight"),
        d_conv1d_w: model
            .conv1d
            .weight
            .val()
            .grad(&grads)
            .expect("grad conv1d.weight"),
        d_x_proj_w: model
            .x_proj
            .weight
            .val()
            .grad(&grads)
            .expect("grad x_proj.weight"),
        d_dt_proj_w: model
            .dt_proj
            .weight
            .val()
            .grad(&grads)
            .expect("grad dt_proj.weight"),
        d_dt_proj_b: model
            .dt_proj
            .bias
            .as_ref()
            .expect("dt_proj has bias")
            .val()
            .grad(&grads)
            .expect("grad dt_proj.bias"),
        d_a_log: model.a_log.val().grad(&grads).expect("grad a_log"),
        d_d: model.d.val().grad(&grads).expect("grad d"),
        d_out_proj_w: model
            .out_proj
            .weight
            .val()
            .grad(&grads)
            .expect("grad out_proj.weight"),
    }
}

/// Compare the output and final cache (conv window + SSM state) of two runs.
fn assert_outputs_match(label: &str, a: &RunGrads, b: &RunGrads, tol: f32) {
    use crate::utils::test_helpers::max_abs_diff;
    let d_out = max_abs_diff(a.out.clone(), b.out.clone());
    let d_conv = max_abs_diff(a.final_conv.clone(), b.final_conv.clone());
    let d_ssm = max_abs_diff(a.final_ssm.clone(), b.final_ssm.clone());
    assert!(
        d_out < tol,
        "{label}: output max abs diff = {d_out:.6} (tol {tol})"
    );
    assert!(
        d_conv < tol,
        "{label}: final conv window max abs diff = {d_conv:.6} (tol {tol})"
    );
    assert!(
        d_ssm < tol,
        "{label}: final SSM state max abs diff = {d_ssm:.6} (tol {tol})"
    );
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
    check!(d_conv1d_w, "conv1d.weight");
    check!(d_x_proj_w, "x_proj.weight");
    check!(d_dt_proj_w, "dt_proj.weight");
    check!(d_dt_proj_b, "dt_proj.bias");
    check!(d_a_log, "a_log");
    check!(d_d, "d");
    check!(d_out_proj_w, "out_proj.weight");
    assert!(
        failures.is_empty(),
        "gradient mismatches:\n  {}",
        failures.join("\n  ")
    );
}

/// Helper that builds a fresh `Param<Tensor>` from a stable inner tensor.
/// A new Param is needed per run so that the autodiff leaf has a fresh
/// node, isolating each backward pass to its own forward graph.
fn param_input(input: &Tensor<3>) -> Param<Tensor<3>> {
    Param::from_tensor(Tensor::from_inner(input.clone()))
}

/// Build the initial cache (conv window + SSM state) passed to both
/// `forward` and the `step` unrolling. With `random = false` the cache is
/// zero (the standard fresh start); with `random = true` it holds random
/// values, exercising forward/step parity from an arbitrary initial state.
fn build_init_cache(cfg: &Mamba1Config, batch: usize, random: bool) -> Mamba1Cache {
    let device: Device = Default::default();
    let d_inner = cfg.d_inner();
    let conv_kernel = cfg.conv_kernel;
    let state_rank = cfg.state_rank;
    let (conv, ssm) = if random {
        let dist = Distribution::Normal(0.0, 1.0);
        (
            Tensor::<3>::random([batch, d_inner, conv_kernel], dist, &device),
            Tensor::<3>::random([batch, d_inner, state_rank], dist, &device),
        )
    } else {
        (
            Tensor::<3>::zeros([batch, d_inner, conv_kernel], &device),
            Tensor::<3>::zeros([batch, d_inner, state_rank], &device),
        )
    };
    Mamba1Cache {
        conv_bik: Tensor::from_inner(conv),
        ssm_bir: Tensor::from_inner(ssm),
    }
}

/// `forward(x)` is mathematically equivalent to repeatedly calling `step`
/// token-by-token from the **same** initial cache: the latter is the
/// recurrent unrolling of the former. Both the outputs, the final cache
/// (conv window + SSM state), and the parameter gradients must agree up to
/// float-summation-order noise.
///
/// With `random_init = true` the shared initial cache is random rather than
/// zero. Parity from an arbitrary initial state subsumes the chunked-prefill
/// (split-vs-full) guarantee: if `forward` from any state equals the
/// recurrent unrolling from that same state — outputs *and* final cache —
/// then feeding a `forward`-produced cache back in continues correctly.
fn run_step_matches_forward(cfg: Mamba1Config, random_init: bool) {
    let device: Device = Default::default();
    let model = cfg.init(&device.clone().autodiff());

    let batch = 2;
    // seq_len >= d_conv so the final conv window is fully determined by the
    // sequence (the initial window has been flushed out), keeping the
    // window comparison well-defined for both zero and random init.
    let seq_len = 5;
    let d_model = cfg.d_model;
    let d_inner = cfg.d_inner();
    let conv_kernel = cfg.conv_kernel;
    let state_rank = cfg.state_rank;
    assert!(seq_len >= conv_kernel);

    let normal = Distribution::Normal(0.0, 1.0);
    let input = Tensor::<3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<3>::random([batch, seq_len, d_model], normal, &device),
        conv: Tensor::<3>::random([batch, d_inner, conv_kernel], normal, &device),
        ssm: Tensor::<3>::random([batch, d_inner, state_rank], normal, &device),
    };

    let init_cache = build_init_cache(&cfg, batch, random_init);

    let input_fwd = param_input(&input);
    let cache_fwd = init_cache.clone();
    let r_fwd = run_with_grads(&model, &input_fwd, &heads, |m, x| {
        m.forward(x, Some(cache_fwd))
    });

    let input_step = param_input(&input);
    let cache_step = init_cache;
    let r_step = run_with_grads(&model, &input_step, &heads, |m, x| {
        let mut cache = cache_step;
        let mut outs: Vec<Tensor<2>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
            let (out_t, new_cache) = m.step(token, cache);
            cache = new_cache;
            outs.push(out_t);
        }
        (Tensor::stack(outs, 1), cache)
    });

    assert_outputs_match("step vs forward", &r_fwd, &r_step, 1e-4);
    // step() and forward() are different reductions of the same SSM, so
    // their per-parameter gradients should also agree, modulo float-
    // summation order noise.
    check_grads_match("step vs forward", &r_fwd, &r_step, 1e-3);

    // ── Guard: the random initial state must actually be consumed ─────
    // Re-run forward from a *zero* initial cache; its output must differ
    // from the random-init output. Otherwise the initial state is being
    // silently ignored and forward/step would match trivially.
    if random_init {
        use crate::utils::test_helpers::max_abs_diff;
        let (out_zero, _) = model.forward(
            Tensor::from_inner(input.clone()),
            Some(build_init_cache(&cfg, batch, false)),
        );
        let d = max_abs_diff(r_fwd.out.clone(), out_zero.inner());
        assert!(
            d > 1e-3,
            "random initial state appears ignored: random-init vs zero-init \
             output max abs diff = {d:.6} (expected a clear difference)"
        );
    }
}

#[test]
fn step_matches_forward() {
    run_step_matches_forward(small_config(), false);
}

#[test]
fn step_matches_forward_random_init() {
    run_step_matches_forward(small_config(), true);
}

// ── Varying d_state ─────────────────────────────────────────────────────

fn cfg_d_state_16() -> Mamba1Config {
    Mamba1Config::new(32)
        .with_state_rank(16)
        .with_conv_kernel(4)
        .with_expand(2)
}

#[test]
fn step_matches_forward_d_state_16() {
    run_step_matches_forward(cfg_d_state_16(), false);
}

#[test]
fn step_matches_forward_d_state_16_random_init() {
    run_step_matches_forward(cfg_d_state_16(), true);
}

// ── Varying d_conv (causal convolution window) ──────────────────────────

fn cfg_d_conv_2() -> Mamba1Config {
    Mamba1Config::new(32)
        .with_state_rank(8)
        .with_conv_kernel(2)
        .with_expand(2)
}

#[test]
fn step_matches_forward_d_conv_2() {
    run_step_matches_forward(cfg_d_conv_2(), false);
}

#[test]
fn step_matches_forward_d_conv_2_random_init() {
    run_step_matches_forward(cfg_d_conv_2(), true);
}

// ── Varying expand (inner width) ────────────────────────────────────────

fn cfg_expand_1() -> Mamba1Config {
    Mamba1Config::new(32)
        .with_state_rank(8)
        .with_conv_kernel(4)
        .with_expand(1)
}

#[test]
fn step_matches_forward_expand_1() {
    run_step_matches_forward(cfg_expand_1(), false);
}

#[test]
fn step_matches_forward_expand_1_random_init() {
    run_step_matches_forward(cfg_expand_1(), true);
}

// ── Custom dt_rank (Δ projection rank) ──────────────────────────────────

fn cfg_custom_dt_rank() -> Mamba1Config {
    Mamba1Config::new(32)
        .with_state_rank(8)
        .with_conv_kernel(4)
        .with_expand(2)
        .with_dt_rank(Some(8))
}

#[test]
fn step_matches_forward_custom_dt_rank() {
    run_step_matches_forward(cfg_custom_dt_rank(), false);
}

#[test]
fn step_matches_forward_custom_dt_rank_random_init() {
    run_step_matches_forward(cfg_custom_dt_rank(), true);
}

// ── Chunked-prefill continuity (split vs full) ──────────────────────────

/// One longer `forward` over the whole sequence equals two consecutive
/// `forward` calls (prefix then suffix) that thread the returned cache (conv
/// window + SSM state) between them — outputs, final cache, and parameter
/// gradients all agree. The initial cache is random, so the chaining starts
/// from a non-trivial state.
///
/// `step_matches_forward_random_init` already subsumes this (parity from an
/// arbitrary initial state implies a `forward`-produced cache continues
/// correctly); this is a direct, explicit check.
fn run_split_matches_full(cfg: Mamba1Config) {
    let device: Device = Default::default();
    let model = cfg.init(&device.clone().autodiff());

    let batch = 2;
    let seq_len = 6;
    let split = 2;
    let d_model = cfg.d_model;
    let d_inner = cfg.d_inner();
    let conv_kernel = cfg.conv_kernel;
    let state_rank = cfg.state_rank;
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<3>::random([batch, seq_len, d_model], normal, &device),
        conv: Tensor::<3>::random([batch, d_inner, conv_kernel], normal, &device),
        ssm: Tensor::<3>::random([batch, d_inner, state_rank], normal, &device),
    };

    let init_cache = build_init_cache(&cfg, batch, true);

    let input_full = param_input(&input);
    let cache_full = init_cache.clone();
    let r_full = run_with_grads(&model, &input_full, &heads, |m, x| {
        m.forward(x, Some(cache_full))
    });

    let input_split = param_input(&input);
    let cache_split = init_cache;
    let r_split = run_with_grads(&model, &input_split, &heads, |m, x| {
        let prefix = x.clone().narrow(1, 0, split);
        let suffix = x.narrow(1, split, seq_len - split);
        let (out_prefix, mid) = m.forward(prefix, Some(cache_split));
        let (out_suffix, last) = m.forward(suffix, Some(mid));
        (Tensor::cat(vec![out_prefix, out_suffix], 1), last)
    });

    assert_outputs_match("split vs full", &r_full, &r_split, 1e-4);
    check_grads_match("split vs full", &r_full, &r_split, 1e-3);
}

#[test]
fn split_matches_full() {
    run_split_matches_full(small_config());
}
