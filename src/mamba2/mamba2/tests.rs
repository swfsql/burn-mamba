use super::*;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

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
    out: Tensor<3>,
    /// Final convolution window from the returned cache.
    final_conv: Tensor<3>,
    /// Final SSM hidden state from the returned cache.
    final_ssm: Tensor<4>,
    d_input: Tensor<3>,
    d_in_proj_w: Tensor<2>,
    d_conv1d_w: Tensor<3>,
    d_dt_bias: Tensor<1>,
    d_a_log: Tensor<1>,
    d_d: Tensor<1>,
    d_norm_gamma: Tensor<1>,
    d_out_proj_w: Tensor<2>,
}

/// Fixed (non-tracked) random "downstream heads" used to form a scalar loss
/// from the output **and** the final cache, so the backward pass exercises
/// both the output and the state path.
struct Heads {
    out: Tensor<3>,
    conv: Tensor<3>,
    ssm: Tensor<4>,
}

/// Build the initial cache passed to both `forward` and the `step`
/// unrolling. With `random = false` it is zero (the standard fresh start);
/// with `random = true` it holds random values, exercising parity from an
/// arbitrary initial state (conv window + SSM hidden state).
fn build_init_cache(cfg: &Mamba2Config, batch: usize, random: bool) -> Mamba2Cache {
    let device: Device = Default::default();
    let conv_dim = cfg.conv_dim();
    let conv_kernel = cfg.conv_kernel;
    let nheads = cfg.nheads();
    let per_head_dim = cfg.per_head_dim;
    let state_rank = cfg.state_rank;
    let (conv, ssm) = if random {
        let dist = Distribution::Normal(0.0, 1.0);
        (
            Tensor::<3>::random([batch, conv_dim, conv_kernel], dist, &device),
            Tensor::<4>::random([batch, nheads, per_head_dim, state_rank], dist, &device),
        )
    } else {
        (
            Tensor::<3>::zeros([batch, conv_dim, conv_kernel], &device),
            Tensor::<4>::zeros([batch, nheads, per_head_dim, state_rank], &device),
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
    model: &Mamba2,
    input: &Param<Tensor<3>>,
    heads: &Heads,
    forward: impl FnOnce(&Mamba2, Tensor<3>) -> (Tensor<3>, Mamba2Cache),
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
fn param_input(input: &Tensor<3>) -> Param<Tensor<3>> {
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
    let model = cfg.init(&device.clone().autodiff());

    let batch = 2;
    // seq_len >= conv_kernel so the final conv window is fully determined by
    // the sequence (the initial window is flushed out), keeping the window
    // comparison well-defined for both zero and random init.
    let seq_len = 5;
    let d_model = cfg.d_model;
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<3>::random([batch, seq_len, d_model], normal, &device),
        conv: Tensor::<3>::random([batch, cfg.conv_dim(), cfg.conv_kernel], normal, &device),
        ssm: Tensor::<4>::random(
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
        let mut cache: Option<Mamba2Cache> = Some(cache_step);
        let mut outs: Vec<Tensor<2>> = Vec::with_capacity(seq_len);
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

// ── Chunked-prefill continuity (split vs full) ──────────────────────────

/// One longer `forward` over the whole sequence equals two consecutive
/// `forward` calls (prefix then suffix) that thread the returned cache (conv
/// window + SSM state) between them — outputs, final cache, and parameter
/// gradients all agree. The initial cache is random, so the chaining starts
/// from a non-trivial state.
///
/// `step_matches_forward_random_init` already subsumes this (parity from an
/// arbitrary initial state implies a `forward`-produced cache continues
/// correctly); this is a direct, explicit check. A single `Minimal` config
/// suffices: the three ssd-paths are proven equivalent among themselves in
/// `mamba2::ssd::ssd_path::tests`.
fn run_split_matches_full(cfg: Mamba2Config) {
    let device: Device = Default::default();
    let model = cfg.init(&device.clone().autodiff());

    let batch = 2;
    let seq_len = 6;
    let split = 2;
    let d_model = cfg.d_model;
    let normal = Distribution::Normal(0.0, 1.0);

    let input = Tensor::<3>::random([batch, seq_len, d_model], normal, &device);
    let heads = Heads {
        out: Tensor::<3>::random([batch, seq_len, d_model], normal, &device),
        conv: Tensor::<3>::random([batch, cfg.conv_dim(), cfg.conv_kernel], normal, &device),
        ssm: Tensor::<4>::random(
            [batch, cfg.nheads(), cfg.per_head_dim, cfg.state_rank],
            normal,
            &device,
        ),
    };

    let ssd_path = Mamba2SsdPath::Minimal(Some(4));
    let init_cache = build_init_cache(&cfg, batch, true);

    let input_full = param_input(&input);
    let cache_full = init_cache.clone();
    let path_full = ssd_path.clone();
    let r_full = run_with_grads(&model, &input_full, &heads, |m, x| {
        m.forward(x, Some(cache_full), path_full)
    });

    let input_split = param_input(&input);
    let cache_split = init_cache;
    let path_split = ssd_path;
    let r_split = run_with_grads(&model, &input_split, &heads, |m, x| {
        let prefix = x.clone().narrow(1, 0, split);
        let suffix = x.narrow(1, split, seq_len - split);
        let (out_prefix, mid) = m.forward(prefix, Some(cache_split), path_split.clone());
        let (out_suffix, last) = m.forward(suffix, Some(mid), path_split);
        (Tensor::cat(vec![out_prefix, out_suffix], 1), last)
    });

    assert_outputs_match("split vs full", &r_full, &r_split, 1e-4);
    check_grads_match("split vs full", &r_full, &r_split, 1e-3);
}

#[test]
fn split_matches_full() {
    run_split_matches_full(small_config());
}

// ── State moments (forward vs step accumulation) ────────────────────────

/// `forward_with_state_moments` returns exactly the moments a token-by-token
/// `step` loop reading `cache.ssm_bhpr` accumulates — from a random initial
/// cache and across sequence padding (`seq_len` not a multiple of the chunk
/// length) — and leaves the plain `forward` outputs and final cache untouched.
/// Streamed continuation is also checked: two chained calls whose merged
/// moments must equal the one-shot moments.
#[test]
fn forward_state_moments_match_step() {
    use crate::utils::test_helpers::max_abs_diff;
    let device: Device = Default::default();
    let cfg = small_config();
    let model = cfg.init(&device.clone().autodiff());

    let batch = 2;
    let seq_len = 7; // chunk_len 4 ⇒ padded to 8: exercises the validity mask
    let split = 3;
    let ssd_path = Mamba2SsdPath::SerialRecalculated(Some(4));
    let input = Tensor::<3>::random(
        [batch, seq_len, cfg.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let init_cache = build_init_cache(&cfg, batch, true);
    let fwd_input = || Tensor::from_inner(input.clone());

    let (out_m, cache_m, moments) =
        model.forward_with_state_moments(fwd_input(), Some(init_cache.clone()), ssd_path.clone());

    // The moments branch must not perturb the ordinary outputs.
    let (out_plain, cache_plain) =
        model.forward(fwd_input(), Some(init_cache.clone()), ssd_path.clone());
    assert!(max_abs_diff(out_m, out_plain) < 1e-6);
    assert!(max_abs_diff(cache_m.ssm_bhpr.clone(), cache_plain.ssm_bhpr) < 1e-6);

    // Reference: accumulate `hₜᵀhₜ` / `Σₚ hₜ` from the step loop's cache.
    let (nheads, state_rank) = (cfg.nheads(), cfg.state_rank);
    let mut cache = Some(init_cache.clone());
    let mut m2_bhrr = Tensor::<4>::from_inner(Tensor::zeros(
        [batch, nheads, state_rank, state_rank],
        &device,
    ));
    let mut m1_bhr = Tensor::<3>::from_inner(Tensor::zeros([batch, nheads, state_rank], &device));
    for t in 0..seq_len {
        let token = Tensor::from_inner(input.clone()).narrow(1, t, 1).squeeze_dim(1);
        let (_out_t, new_cache) = model.step(token, cache);
        let h_bhpr = new_cache.ssm_bhpr.clone();
        m2_bhrr = m2_bhrr + h_bhpr.clone().permute([0, 1, 3, 2]).matmul(h_bhpr.clone());
        m1_bhr = m1_bhr + h_bhpr.sum_dim(2).reshape([batch, nheads, state_rank]);
        cache = Some(new_cache);
    }

    assert_eq!(moments.count, seq_len * cfg.per_head_dim);
    let scale = max_abs_diff(m2_bhrr.clone(), m2_bhrr.zeros_like()).max(1.0);
    let d2 = max_abs_diff(moments.m2_bhrr.clone(), m2_bhrr.clone());
    let d1 = max_abs_diff(moments.m1_bhr.clone(), m1_bhr.clone());
    assert!(d2 < 1e-4 * scale, "m2 vs step: {d2:.6} (scale {scale:.3})");
    assert!(d1 < 1e-4 * scale, "m1 vs step: {d1:.6} (scale {scale:.3})");

    // Streamed continuation: prefix + suffix moments merge to the one-shot.
    let (_, mid_cache, m_prefix) = model.forward_with_state_moments(
        fwd_input().narrow(1, 0, split),
        Some(init_cache),
        ssd_path.clone(),
    );
    let (_, _, m_suffix) = model.forward_with_state_moments(
        fwd_input().narrow(1, split, seq_len - split),
        Some(mid_cache),
        ssd_path,
    );
    let merged = m_prefix.merge(m_suffix);
    assert_eq!(merged.count, moments.count);
    let d2 = max_abs_diff(merged.m2_bhrr, m2_bhrr);
    let d1 = max_abs_diff(merged.m1_bhr, m1_bhr);
    assert!(d2 < 1e-4 * scale, "merged m2 vs step: {d2:.6}");
    assert!(d1 < 1e-4 * scale, "merged m1 vs step: {d1:.6}");
}

/// Gradient counterpart of [`forward_state_moments_match_step`]: a scalar
/// loss over the **attached** closed-form moments
/// (`forward_with_state_moments_grad`) must produce the same input and
/// parameter gradients as the identical loss over moments accumulated from
/// the real `step()` loop's `ssm_bhpr` caches — the moments' own autodiff
/// subgraph vs the recurrent chain, through the shared upstream
/// (in-proj → conv → discretisation). Runs on `SerialRecalculated` to show
/// the moments subgraph coexists with the custom backward.
#[test]
fn forward_state_moments_grads_match_step() {
    use crate::utils::test_helpers::max_abs_diff;
    let device: Device = Default::default();
    let cfg = small_config();
    let model = cfg.init(&device.clone().autodiff());

    let batch = 2;
    let seq_len = 7; // chunk_len 4 ⇒ padding: the masked-`t` gradients covered
    let ssd_path = Mamba2SsdPath::SerialRecalculated(Some(4));
    let (nheads, state_rank) = (cfg.nheads(), cfg.state_rank);
    let normal = Distribution::Normal(0.0, 1.0);
    let input = Tensor::<3>::random([batch, seq_len, cfg.d_model], normal, &device);
    let m2_head = Tensor::<4>::random([batch, nheads, state_rank, state_rank], normal, &device);
    let m1_head = Tensor::<3>::random([batch, nheads, state_rank], normal, &device);
    let init_cache = build_init_cache(&cfg, batch, true);
    let loss_of = |m2: Tensor<4>, m1: Tensor<3>| {
        (m2 * Tensor::from_inner(m2_head.clone())).sum()
            + (m1 * Tensor::from_inner(m1_head.clone())).sum()
    };

    // Closed form: moments attached, `y` dropped — the loss is moments-only.
    let input_fwd = param_input(&input);
    let (_y, _cache, moments) =
        model.forward_with_state_moments_grad(input_fwd.val(), Some(init_cache.clone()), ssd_path);
    let grads_fwd = loss_of(moments.m2_bhrr, moments.m1_bhr).backward();

    // Step loop, accumulating the (tracked) cache states.
    let input_step = param_input(&input);
    let mut cache = Some(init_cache);
    let mut m2_bhrr = Tensor::<4>::from_inner(Tensor::zeros(
        [batch, nheads, state_rank, state_rank],
        &device,
    ));
    let mut m1_bhr = Tensor::<3>::from_inner(Tensor::zeros([batch, nheads, state_rank], &device));
    for t in 0..seq_len {
        let token = input_step.val().narrow(1, t, 1).squeeze_dim(1);
        let (_out_t, new_cache) = model.step(token, cache);
        let h_bhpr = new_cache.ssm_bhpr.clone();
        m2_bhrr = m2_bhrr + h_bhpr.clone().permute([0, 1, 3, 2]).matmul(h_bhpr.clone());
        m1_bhr = m1_bhr + h_bhpr.sum_dim(2).reshape([batch, nheads, state_rank]);
        cache = Some(new_cache);
    }
    let grads_step = loss_of(m2_bhrr, m1_bhr).backward();

    // Everything the states depend on: the input, `x`/`B`/`dt` (in-proj →
    // conv → split) and the decay (dt_bias, a_log). `d_h`/norm/out-proj feed
    // only `y`, so the moments loss must leave them untouched (checked last).
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($fwd:expr, $step:expr, $name:expr) => {{
            let g_fwd = $fwd.grad(&grads_fwd).expect("fwd grad");
            let g_step = $step.grad(&grads_step).expect("step grad");
            let scale = max_abs_diff(g_step.clone(), g_step.zeros_like()).max(1.0);
            let d = max_abs_diff(g_fwd, g_step);
            eprintln!("moments grads {:>16} | max abs diff = {d:>10.6} (scale {scale:.3})", $name);
            if d >= 1e-3 * scale {
                failures.push(format!("grad of {} max abs diff = {d:.6}", $name));
            }
        }};
    }
    check!(input_fwd.val(), input_step.val(), "input");
    check!(model.in_proj.weight.val(), model.in_proj.weight.val(), "in_proj.weight");
    check!(model.conv1d.weight.val(), model.conv1d.weight.val(), "conv1d.weight");
    check!(model.dt_bias_h.val(), model.dt_bias_h.val(), "dt_bias_h");
    check!(model.a_log_h.val(), model.a_log_h.val(), "a_log_h");
    assert!(
        failures.is_empty(),
        "gradient mismatches:\n  {}",
        failures.join("\n  ")
    );

    assert!(model.d_h.val().grad(&grads_fwd).is_none());
    assert!(model.d_h.val().grad(&grads_step).is_none());
}
