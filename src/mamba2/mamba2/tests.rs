use super::*;
use burn::backend::{Autodiff, Flex};
use burn::tensor::Distribution;

/// Inner (non-autodiff) backend used for materialising values and
/// extracted gradients.
type InnerB = Flex;
/// Autodiff-wrapped backend used to drive `.backward()`.
type B = Autodiff<InnerB>;

type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

fn small_config() -> Mamba2Config {
    Mamba2Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
}

/// A bundle of input + model-parameter gradients extracted from one
/// forward+backward run.  Each `check_grads_match` call compares these
/// across two runs that should be mathematically equivalent.
struct RunGrads {
    out: Tensor<InnerB, 3>,
    /// Final convolution window from the returned cache.
    final_conv: Tensor<InnerB, 3>,
    /// Final SSM hidden state from the returned cache.
    final_ssm: Tensor<InnerB, 4>,
    d_input: Tensor<InnerB, 3>,
    d_in_proj_w: Tensor<InnerB, 2>,
    d_conv1d_w: Tensor<InnerB, 3>,
    d_dt_bias: Tensor<InnerB, 1>,
    d_a_log: Tensor<InnerB, 1>,
    d_d: Tensor<InnerB, 1>,
    d_norm_gamma: Tensor<InnerB, 1>,
    d_out_proj_w: Tensor<InnerB, 2>,
}

/// Fixed (non-tracked) random "downstream heads" used to form a scalar loss
/// from the output **and** the final cache, so the backward pass exercises
/// both the output and the state path.
struct Heads {
    out: Tensor<InnerB, 3>,
    conv: Tensor<InnerB, 3>,
    ssm: Tensor<InnerB, 4>,
}

/// Build the initial cache passed to both `forward` and the `step`
/// unrolling. With `random = false` it is zero (the standard fresh start);
/// with `random = true` it holds random values, exercising parity from an
/// arbitrary initial state (conv window + SSM hidden state).
fn build_init_cache(cfg: &Mamba2Config, batch: usize, random: bool) -> Mamba2Cache<B> {
    let device: Device = Default::default();
    let conv_dim = cfg.conv_dim();
    let conv_kernel = cfg.conv_kernel;
    let nheads = cfg.nheads();
    let per_head_dim = cfg.per_head_dim;
    let state_rank = cfg.state_rank;
    let (conv, ssm) = if random {
        let dist = Distribution::Normal(0.0, 1.0);
        (
            Tensor::<InnerB, 3>::random([batch, conv_dim, conv_kernel], dist, &device),
            Tensor::<InnerB, 4>::random([batch, nheads, per_head_dim, state_rank], dist, &device),
        )
    } else {
        (
            Tensor::<InnerB, 3>::zeros([batch, conv_dim, conv_kernel], &device),
            Tensor::<InnerB, 4>::zeros([batch, nheads, per_head_dim, state_rank], &device),
        )
    };
    Mamba2Cache {
        conv_bvk: Tensor::from_inner(conv),
        ssm_bhpr: Tensor::from_inner(ssm),
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

/// Run a closure that produces an output tensor from a model and an input
/// (wrapped as a `Param` so it has its own autodiff leaf), then derive a
/// scalar loss with a fixed (non-tracked) random "head" and return the
/// gradients of the input and a representative set of model parameters.
fn run_with_grads(
    model: &Mamba2<B>,
    input: &Param<Tensor<B, 3>>,
    heads: &Heads,
    forward: impl FnOnce(&Mamba2<B>, Tensor<B, 3>) -> (Tensor<B, 3>, Mamba2Cache<B>),
) -> RunGrads {
    let (out, cache) = forward(model, input.val());
    let out_inner = out.clone().inner();
    let conv = cache.conv_bvk;
    let ssm = cache.ssm_bhpr;
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
        d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("grad dt_bias_h"),
        d_a_log: model.a_log_h.val().grad(&grads).expect("grad a_log_h"),
        d_d: model.d_h.val().grad(&grads).expect("grad d_h"),
        d_norm_gamma: model
            .norm
            .gamma
            .val()
            .grad(&grads)
            .expect("grad norm.gamma"),
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
    check!(d_conv1d_w, "conv1d.weight");
    check!(d_dt_bias, "dt_bias_h");
    check!(d_a_log, "a_log_h");
    check!(d_d, "d_h");
    check!(d_norm_gamma, "norm.gamma");
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
fn param_input(input: &Tensor<InnerB, 3>) -> Param<Tensor<B, 3>> {
    Param::from_tensor(Tensor::from_inner(input.clone()))
}

/// `forward(x)` is mathematically equivalent to repeatedly calling `step`
/// token-by-token from the **same** initial cache. Outputs, the final cache
/// (conv window + SSM state), and parameter gradients must all agree up to
/// float-summation-order noise.
///
/// With `random_init = true` the shared initial cache is random rather than
/// zero. Parity from an arbitrary initial state subsumes the chunked-prefill
/// (split-vs-full) guarantee: if `forward` from any state matches the
/// recurrent unrolling from that same state — outputs *and* final cache —
/// then feeding a `forward`-produced cache back in continues correctly.
fn run_step_matches_forward(cfg: Mamba2Config, ssd_path: Mamba2SsdPath, random_init: bool) {
    let device: Device = Default::default();
    let model = cfg.init::<B>(&device);

    let batch = 2;
    // seq_len >= conv_kernel so the final conv window is fully determined by
    // the sequence (the initial window is flushed out), keeping the window
    // comparison well-defined for both zero and random init.
    let seq_len = 5;
    let d_model = cfg.d_model;
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
        conv: Tensor::<InnerB, 3>::random(
            [batch, cfg.conv_dim(), cfg.conv_kernel],
            normal,
            &device,
        ),
        ssm: Tensor::<InnerB, 4>::random(
            [batch, cfg.nheads(), cfg.per_head_dim, cfg.state_rank],
            normal,
            &device,
        ),
    };

    let init_cache = build_init_cache(&cfg, batch, random_init);

    let input_fwd = param_input(&input);
    let cache_fwd = init_cache.clone();
    let path_fwd = ssd_path.clone();
    let r_fwd = run_with_grads(&model, &input_fwd, &heads, |m, x| {
        m.forward(x, Some(cache_fwd), path_fwd)
    });

    let input_step = param_input(&input);
    let cache_step = init_cache;
    let r_step = run_with_grads(&model, &input_step, &heads, |m, x| {
        let mut cache: Option<Mamba2Cache<B>> = Some(cache_step);
        let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
            let (out_t, new_cache) = m.step(token, cache);
            cache = Some(new_cache);
            outs.push(out_t);
        }
        (Tensor::stack(outs, 1), cache.unwrap())
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
            ssd_path.clone(),
        );
        let d = max_abs_diff(r_fwd.out.clone(), out_zero.inner());
        assert!(
            d > 1e-3,
            "random initial state appears ignored: random-init vs zero-init \
             output max abs diff = {d:.6} (expected a clear difference)"
        );
    }
}

fn cfg_ngroups2() -> Mamba2Config {
    Mamba2Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(16)
        .with_ngroups(2)
}

fn cfg_norm_before_gate() -> Mamba2Config {
    Mamba2Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_is_norm_before_gate(true)
}

#[test]
fn step_matches_forward() {
    run_step_matches_forward(small_config(), Mamba2SsdPath::Minimal(Some(4)), false);
}

#[test]
fn step_matches_forward_random_init() {
    run_step_matches_forward(small_config(), Mamba2SsdPath::Minimal(Some(4)), true);
}

#[test]
fn step_matches_forward_ngroups2() {
    run_step_matches_forward(cfg_ngroups2(), Mamba2SsdPath::Minimal(Some(4)), false);
}

#[test]
fn step_matches_forward_ngroups2_random_init() {
    run_step_matches_forward(cfg_ngroups2(), Mamba2SsdPath::Minimal(Some(4)), true);
}

// ── is_norm_before_gate = true ───────────────────────────────────────────

#[test]
fn step_matches_forward_norm_before_gate() {
    run_step_matches_forward(
        cfg_norm_before_gate(),
        Mamba2SsdPath::Minimal(Some(4)),
        false,
    );
}

#[test]
fn step_matches_forward_norm_before_gate_random_init() {
    run_step_matches_forward(
        cfg_norm_before_gate(),
        Mamba2SsdPath::Minimal(Some(4)),
        true,
    );
}

// ── SSD path agreement ───────────────────────────────────────────────────

/// `Minimal`, `Serial`, and `SerialRecalculated` are chunkwise reformulations
/// of the same SSD, so their block-level outputs, final caches, and gradients
/// must agree — from a zero (`random_init = false`) or random initial cache.
fn run_ssd_paths_agree(cfg: Mamba2Config, random_init: bool) {
    let device: Device = Default::default();
    let model = cfg.init::<B>(&device);

    let batch = 2;
    let seq_len = 8;
    let d_model = cfg.d_model;
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
        conv: Tensor::<InnerB, 3>::random(
            [batch, cfg.conv_dim(), cfg.conv_kernel],
            normal,
            &device,
        ),
        ssm: Tensor::<InnerB, 4>::random(
            [batch, cfg.nheads(), cfg.per_head_dim, cfg.state_rank],
            normal,
            &device,
        ),
    };

    let init_cache = build_init_cache(&cfg, batch, random_init);

    let run = |path: Mamba2SsdPath| {
        let input_p = param_input(&input);
        let cache_p = init_cache.clone();
        run_with_grads(&model, &input_p, &heads, |m, x| {
            m.forward(x, Some(cache_p), path)
        })
    };
    let r_min = run(Mamba2SsdPath::Minimal(Some(4)));
    let r_ser = run(Mamba2SsdPath::Serial(Some(4)));
    let r_rec = run(Mamba2SsdPath::SerialRecalculated(Some(4)));

    assert_outputs_match("Minimal vs Serial", &r_min, &r_ser, 1e-4);
    assert_outputs_match("Minimal vs SerialRecalculated", &r_min, &r_rec, 1e-4);
    check_grads_match("Minimal vs Serial", &r_min, &r_ser, 1e-3);
    check_grads_match("Minimal vs SerialRecalculated", &r_min, &r_rec, 1e-3);

    // ── Guard: the random initial state must actually be consumed ─────
    if random_init {
        use crate::utils::test_helpers::max_abs_diff;
        let (out_zero, _) = model.forward(
            Tensor::from_inner(input.clone()),
            Some(build_init_cache(&cfg, batch, false)),
            Mamba2SsdPath::Minimal(Some(4)),
        );
        let d = max_abs_diff(r_min.out.clone(), out_zero.inner());
        assert!(
            d > 1e-3,
            "random initial state appears ignored: random-init vs zero-init \
             output max abs diff = {d:.6} (expected a clear difference)"
        );
    }
}

#[test]
fn ssd_paths_agree() {
    run_ssd_paths_agree(small_config(), false);
}

#[test]
fn ssd_paths_agree_random_init() {
    run_ssd_paths_agree(small_config(), true);
}

#[test]
fn ssd_paths_agree_ngroups2() {
    run_ssd_paths_agree(cfg_ngroups2(), false);
}

#[test]
fn ssd_paths_agree_ngroups2_random_init() {
    run_ssd_paths_agree(cfg_ngroups2(), true);
}
