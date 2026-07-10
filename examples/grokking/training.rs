//! Full-batch training loop for the grokking task: AdamW with plain
//! (non-cautious) decoupled weight decay, cross-entropy on the final position
//! only, and train/test accuracy logged to `metrics.csv` in the artifacts
//! directory (log-spaced early via power-of-two steps, then every
//! `eval_every`). The whole train split is one batch — the grokking-literature
//! setup, and at `p = 97` it is at most 9408 two-token sequences.

pub use crate::common::cli::AppArgs;
use crate::dataset::{self, Split};
use crate::diagnostics::{self, StatePr, WeightPr};
pub use crate::diagnostics::PrPenaltyTarget;
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn_mamba::modules::loss::cross_entropy::CrossEntropyLossConfig;
use burn_mamba::prelude::*;
pub use burn_mamba::utils::scheduler::{ConstantLr, Lr};

/// Grokking hyperparameters: optimizer + task/split + schedule knobs.
///
/// Weight decay (in `optimizer`) is the grokking driver; `0.0` is the
/// memorization control arm. Keep `cautious_weight_decay` **off** — cautious
/// decay masks exactly the pressure grokking relies on.
#[derive(Config, Debug)]
pub struct GrokkingConfig {
    /// The optimizer configuration (AdamW).
    pub optimizer: AdamWConfig,
    /// The modulus `p` (vocab size and class count).
    #[config(default = 97)]
    pub p: usize,
    /// Number of summands `k` (sequence length). 2 = the literature-standard
    /// pair task; > 2 forces all pre-final-token information through the
    /// recurrent state (keep `pᵏ` full-batch-sized, e.g. `p = 11, k = 4`).
    #[config(default = 2)]
    pub k: usize,
    /// Fraction of the `pᵏ` sequences used for training.
    #[config(default = 0.5)]
    pub train_fraction: f64,
    /// Seed for the deterministic train/test pair split.
    #[config(default = 0)]
    pub split_seed: u64,
    /// Number of full-batch optimizer steps.
    #[config(default = 20_000)]
    pub num_steps: usize,
    /// Evaluate train/test accuracy every this many steps (power-of-two steps
    /// are always evaluated too, giving log-spaced early coverage).
    #[config(default = 250)]
    pub eval_every: usize,
    /// Save model/optimizer state every this many steps.
    #[config(default = 2_000)]
    pub save_every: usize,
    /// Learning-rate schedule.
    #[config(default = "Lr::Constant(ConstantLr::new().with_lr(1e-3))")]
    pub lr: Lr,
    /// RNG seed for model initialization.
    #[config(default = 0)]
    pub seed: u64,
    /// Run all forwards token-by-token via `step()` instead of the chunkwise
    /// `forward()` — mathematically identical (the library's parity contract),
    /// ~7× faster at T = 2, and exposes the per-step state caches (the
    /// capture point for the state-PR diagnostic). The chunkwise path uses
    /// the recompute backward and needs less memory — prefer it for capacity
    /// probes on bigger models.
    #[config(default = true)]
    pub stepwise: bool,
    /// Compute and log the PR diagnostics at eval points. Turn off for
    /// capacity probes (accuracy/loss logging remains); a full panel is
    /// always available post-hoc via `--inference` on any checkpoint.
    #[config(default = true)]
    pub diagnostics: bool,
    /// Also run the (costly: a stepping pass over the diagnostic set) state-PR
    /// part of the diagnostics; weight PRs are always logged when
    /// `diagnostics` is on.
    #[config(default = true)]
    pub state_diagnostics: bool,
    /// Coefficient of the differentiable weight-PR penalty
    /// (`loss += pr_lambda · Σ PR(W)` over `pr_target`); `0` disables it.
    /// Negative values *reward* rank expansion (the sign-check control).
    /// The causal Step-2 arm: pure rank pressure in place of weight decay.
    #[config(default = 0.0)]
    pub pr_lambda: f64,
    /// Which weights the PR penalty targets.
    #[config(default = "PrPenaltyTarget::All")]
    pub pr_target: PrPenaltyTarget,
    /// Period (in steps) of a sine modulation of the PR penalty: the effective
    /// coefficient becomes `pr_lambda · sin(2π·step/period)` — "breathing"
    /// that alternates compression (positive half-cycle) and expansion
    /// (negative half-cycle). `0` keeps the coefficient constant.
    #[config(default = 0)]
    pub pr_sine_period: usize,
    /// Offset added to the step numbers written to the console/csv logs, for
    /// resumed runs (the loop itself always runs `1..=num_steps`; eval/save
    /// cadence and the sine phase follow the raw loop step).
    #[config(default = 0)]
    pub step_offset: usize,
    /// Keep the PR penalty off until this (raw) step — let the model memorize
    /// unimpeded first (the penalty's compression gradient fights the fit,
    /// unlike weight decay). The sine phase also counts from here.
    #[config(default = 0)]
    pub pr_start_step: usize,
    /// Coefficient of the plain L2 (Frobenius²) loss penalty over the same
    /// `pr_target` matrices — the rank-specificity control for `pr_lambda`
    /// (norm pressure through the loss, no rank preference). `0` disables.
    #[config(default = 0.0)]
    pub l2_lambda: f64,
    /// Coefficient of the weight-independent-gradient control `Σ ⟨W, ε⟩`
    /// (fresh `ε ~ N(0,1)` per step, detached): per-element auxiliary
    /// gradient RMS = this coefficient, but carrying zero information about
    /// the weights. `0` disables.
    #[config(default = 0.0)]
    pub noise_lambda: f64,
    /// `>= 0`: replace AdamW with plain SGD at this momentum (0 = none) —
    /// the probe for Adam's role in the noise catalysis (no per-parameter
    /// moment normalization; auxiliary gradients keep their raw scale). The
    /// SGD path uses `sgd_wd` as its (coupled) decay and is always freshly
    /// initialized (no optimizer resume). `< 0` = AdamW.
    #[config(default = -1.0)]
    pub sgd_momentum: f64,
    /// Coupled weight-decay penalty for the SGD path (`--wd` sets this too;
    /// `AdamWConfig`'s own field is builder-only). Per-step shrink ≈
    /// `lr · sgd_wd`.
    #[config(default = 0.0)]
    pub sgd_wd: f32,
    /// Coefficient of the differentiable **state**-PR penalty:
    /// `loss += state_pr_lambda · Σ_{layer,head} PR(states)` (batch-pooled,
    /// uncentered), from the library's closed-form state moments on the
    /// training forward — direct rank pressure on the circuit-sized quantity
    /// the state-PR diagnostic measures. Requires the chunkwise path
    /// (`--chunked`); `0` disables; negative rewards state-rank expansion.
    #[config(default = 0.0)]
    pub state_pr_lambda: f64,
}

impl GrokkingConfig {
    /// The effective PR-penalty coefficient at `step` (constant `pr_lambda`,
    /// or the sine "breathing" when `pr_sine_period > 0`); `0` before
    /// `pr_start_step`.
    pub fn pr_lambda_at(&self, step: usize) -> f64 {
        if step < self.pr_start_step {
            return 0.0;
        }
        if self.pr_sine_period == 0 {
            self.pr_lambda
        } else {
            let gated_step = step - self.pr_start_step;
            let phase = 2.0 * std::f64::consts::PI * gated_step as f64 / self.pr_sine_period as f64;
            self.pr_lambda * phase.sin()
        }
    }
}

/// The SSD path used by chunkwise forwards: the recompute-backward serial
/// algorithm (the memory-saving custom backward) with `chunk_len = 2` matching
/// the two-token sequences (the default ≈32 chunk would zero-pad every
/// sequence 16×). Family follows the model (`--mamba3`).
pub fn ssd_path(model: &MambaVocabNet) -> MambaSsdPath {
    match model {
        MambaVocabNet::Mamba3(_) => {
            MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(Some(2)))
        }
        _ => MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(Some(2))),
    }
}

/// Final-position logits `[n, p]` for a batch of token sequences `[n, s]`,
/// either chunkwise (`forward()`) or token-by-token (`step()`; identical by
/// the library's parity contract).
/// [`final_logits`] via the chunkwise path, additionally returning each
/// (virtual) layer's **attached** state moments — the state-PR penalty input
/// (gradients flow through the moments into the model).
pub fn final_logits_with_moments(
    model: &MambaVocabNet,
    inputs_bs: &Tensor<2, Int>,
) -> (Tensor<2>, Vec<StateMoments>) {
    let [_b, s] = inputs_bs.dims();
    let (logits_bsc, _caches, moments) =
        model.forward_with_state_moments_grad(inputs_bs.clone(), None, ssd_path(model));
    (logits_bsc.narrow(1, s - 1, 1).squeeze_dim::<2>(1), moments)
}

pub fn final_logits(model: &MambaVocabNet, inputs_bs: &Tensor<2, Int>, stepwise: bool) -> Tensor<2> {
    let [_b, s] = inputs_bs.dims();
    if stepwise {
        let mut caches = None;
        let mut logits = None;
        for t in 0..s {
            let x_b = inputs_bs.clone().narrow(1, t, 1).squeeze_dim::<1>(1);
            let (logits_bc, new_caches) = model.step(x_b, caches, None, None);
            caches = Some(new_caches);
            logits = Some(logits_bc);
        }
        logits.expect("at least one token")
    } else {
        let (logits_bsc, _caches) = model.forward(inputs_bs.clone(), None, ssd_path(model));
        logits_bsc.narrow(1, s - 1, 1).squeeze_dim::<2>(1)
    }
}

/// Run the full training routine: load/init the model and optimizer, then take
/// `num_steps` full-batch steps, logging accuracies and checkpointing along
/// the way.
pub fn train(
    config: GrokkingConfig,
    model_config: MambaVocabNetConfig,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(config.seed);
    let eval_device = training_device.clone().inner();
    assert!(
        config.state_pr_lambda == 0.0 || !config.stepwise,
        "state_pr_lambda needs the training forward's state moments — run with --chunked"
    );

    let mut model: MambaVocabNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let mut optim = if config.sgd_momentum >= 0.0 {
        let mut sgd = burn::optim::SgdConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                config.sgd_wd,
            )))
            // Same clipping as the AdamW arm — unclipped full-batch SGD with
            // momentum NaNs on this task within ~2k steps.
            .with_gradient_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(
                1.0,
            )));
        if config.sgd_momentum > 0.0 {
            sgd = sgd.with_momentum(Some(
                burn::optim::momentum::MomentumConfig::new().with_momentum(config.sgd_momentum),
            ));
        }
        println!(
            "SGD probe: momentum {}, coupled wd {}",
            config.sgd_momentum, config.sgd_wd
        );
        sgd.init()
    } else {
        app_args.load_or_save_optim(&config.optimizer, &model)
    };

    let (train_split, test_split) =
        dataset::build(config.p, config.k, config.train_fraction, config.split_seed);
    println!(
        "p = {}, k = {}, train seqs: {}, test seqs: {} (fraction {})",
        config.p,
        config.k,
        train_split.len(),
        test_split.len(),
        config.train_fraction,
    );

    // Full-batch training tensors live on the autodiff device; the eval copies
    // on the plain inner device.
    let x_bs = train_split.inputs_tensor(&training_device);
    let targets_bp = train_split.targets_tensor(&training_device);
    let eval_train = (
        train_split.inputs_tensor(&eval_device),
        train_split.labels_tensor(&eval_device),
    );
    let eval_test = (
        test_split.inputs_tensor(&eval_device),
        test_split.labels_tensor(&eval_device),
    );

    // The PR diagnostic's eval set: all `pᵏ` sequences (or a deterministic
    // 10k sample when the space is larger), on the plain device.
    let diag_inputs = dataset::diagnostic_set(config.p, config.k, 10_000, config.split_seed)
        .inputs_tensor(&eval_device);

    let ce = CrossEntropyLossConfig::new().init();
    let metrics_path = app_args.artifacts_path.join("metrics.csv");
    let pr_path = app_args.artifacts_path.join("pr.csv");
    let weights_path = app_args.artifacts_path.join("weights.csv");
    println!("logging metrics to {metrics_path:?}, state PR to {pr_path:?}, weight PR to {weights_path:?}");

    println!("Starting training...");
    let started = std::time::Instant::now();
    for step in 1..=config.num_steps {
        // The state-PR penalty needs the training forward's attached moments,
        // so it forces the chunkwise path (asserted above).
        let (logits_bc, train_moments) = if config.state_pr_lambda != 0.0 {
            let (logits, moments) = final_logits_with_moments(&model, &x_bs);
            (logits, Some(moments))
        } else {
            (final_logits(&model, &x_bs, config.stepwise), None)
        };
        // `loss_value` (and the csv column) stays CE-only, comparable across
        // arms; the penalty value is printed separately at eval points.
        let ce_loss = ce.forward(logits_bc, targets_bp.clone());
        let loss_value = scalar_f32(ce_loss.clone());
        let pr_lambda = config.pr_lambda_at(step);
        let mut loss = ce_loss;
        if pr_lambda != 0.0 {
            let penalty = diagnostics::weight_pr_penalty(&model, config.pr_target);
            loss = loss + penalty.mul_scalar(pr_lambda);
        }
        if config.l2_lambda != 0.0 {
            let penalty = diagnostics::weight_l2_penalty(&model, config.pr_target);
            loss = loss + penalty.mul_scalar(config.l2_lambda);
        }
        if config.noise_lambda != 0.0 {
            let penalty = diagnostics::weight_noise_penalty(&model, config.pr_target);
            loss = loss + penalty.mul_scalar(config.noise_lambda);
        }
        if let Some(moments) = train_moments {
            let pairing = diagnostics::state_pairing_of(&model);
            let penalty = diagnostics::state_pr_penalty(&moments, &pairing);
            loss = loss + penalty.mul_scalar(config.state_pr_lambda);
        }

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        if nonfinite_grad_count(&model, &grads) > 0.0 {
            let logged_step = step + config.step_offset;
            eprintln!("[grad-nan] non-finite gradient at step {logged_step}");
            report_nonfinite_grads(&model, &x_bs, &targets_bp, &ce, &config, pr_lambda);
            panic!("non-finite gradient at step {logged_step}");
        }
        let lr = config.lr.get_lr(step);
        model = optim.step(lr, model, grads);

        let last = step == config.num_steps;
        if step.is_power_of_two() || step % config.eval_every == 0 || last {
            // Resumed runs log continued step numbers; the loop/cadence/sine
            // phase stay on the raw step.
            let logged_step = step + config.step_offset;
            let valid_model = model.valid();
            let train_acc = accuracy(&valid_model, &eval_train.0, &eval_train.1, config.stepwise);
            let test_acc = accuracy(&valid_model, &eval_test.0, &eval_test.1, config.stepwise);
            println!(
                "step {logged_step:>6}/{}, loss {loss_value:.4e}, train acc {train_acc:.4}, \
                 test acc {test_acc:.4}, lr {lr:.2e}, {:.1}s",
                config.num_steps + config.step_offset,
                started.elapsed().as_secs_f64(),
            );
            if config.pr_lambda != 0.0 {
                let penalty =
                    scalar_f32(diagnostics::weight_pr_penalty(&valid_model, config.pr_target));
                println!(
                    "        pr penalty {penalty:.3} (λ_eff {pr_lambda:.4}, {:?})",
                    config.pr_target
                );
            }
            if config.l2_lambda != 0.0 {
                let penalty =
                    scalar_f32(diagnostics::weight_l2_penalty(&valid_model, config.pr_target));
                println!(
                    "        l2 penalty {penalty:.3} (λ {}, {:?})",
                    config.l2_lambda, config.pr_target
                );
            }
            if config.diagnostics {
                // Stepwise runs read the cache after every `step`; chunked
                // runs get the same pooled PRs from the library's closed-form
                // state moments in one `forward` (exact, states never
                // materialised).
                let state_prs = if !config.state_diagnostics {
                    Vec::new()
                } else if config.stepwise {
                    diagnostics::state_pr(&valid_model, &diag_inputs)
                } else {
                    diagnostics::state_pr_forward(&valid_model, &diag_inputs, ssd_path(&valid_model))
                };
                let weight_prs = diagnostics::weight_pr(&valid_model, config.p);
                println!("        {}", format_prs(&state_prs, &weight_prs));
                // The state-PR penalty value over the diagnostic set: exactly
                // Σ pooled-uncentered over the entries just computed.
                if config.state_pr_lambda != 0.0 && !state_prs.is_empty() {
                    let total: f64 = state_prs.iter().map(|s| s.pooled_uncentered).sum();
                    println!(
                        "        state-pr penalty {total:.3} (λ {})",
                        config.state_pr_lambda
                    );
                }
                append_metrics(&metrics_path, logged_step, lr, loss_value, train_acc, test_acc, &weight_prs);
                if !state_prs.is_empty() {
                    append_pr(&pr_path, logged_step, &state_prs);
                }
                append_weight_pr(&weights_path, logged_step, &weight_prs);
            } else {
                append_metrics_bare(&metrics_path, logged_step, lr, loss_value, train_acc, test_acc);
            }
        }
        if step % config.save_every == 0 || last {
            app_args.save_model(&model);
            app_args.save_optim(&optim);
        }
    }
    println!("Training finished.");
}

/// Fraction of examples whose final-position argmax matches the label.
pub fn accuracy(
    model: &MambaVocabNet,
    inputs_bs: &Tensor<2, Int>,
    labels_b: &Tensor<1, Int>,
    stepwise: bool,
) -> f64 {
    let logits_bc = final_logits(model, inputs_bs, stepwise);
    let [b, _classes] = logits_bc.dims();
    let pred_b = logits_bc.argmax(1).reshape([b]);
    scalar_f32(pred_b.equal(labels_b.clone()).float().mean()) as f64
}

/// Convenience: evaluate both splits with a plain (non-autodiff) model.
pub fn eval_accuracies(
    model: &MambaVocabNet,
    train: &Split,
    test: &Split,
    device: &Device,
    stepwise: bool,
) -> (f64, f64) {
    let train_acc = accuracy(model, &train.inputs_tensor(device), &train.labels_tensor(device), stepwise);
    let test_acc = accuracy(model, &test.inputs_tensor(device), &test.labels_tensor(device), stepwise);
    (train_acc, test_acc)
}

/// Read a single-element float tensor back to the host.
fn scalar_f32(t: Tensor<1>) -> f32 {
    t.into_data().to_vec::<f32>().unwrap()[0]
}

/// `1` if a gradient tensor holds any NaN or Inf, else `0` — kept device-side
/// so many can be summed with a single host sync.
fn grad_bad_indicator<const D: usize>(g: Tensor<D>) -> Tensor<1> {
    g.clone().is_nan().any().int().float() + g.is_inf().any().int().float()
}

/// Visitor that sums [`grad_bad_indicator`] over every parameter whose gradient
/// is present in `grads` — one device-side scalar, so the per-step healthy path
/// costs a single sync.
struct GradBadCount<'a> {
    grads: &'a GradientsParams,
    acc: Option<Tensor<1>>,
}

impl ModuleVisitor for GradBadCount<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(g) = self.grads.get::<D>(param.id) else {
            return;
        };
        let bad = grad_bad_indicator(g);
        self.acc = Some(match self.acc.take() {
            Some(acc) => acc + bad,
            None => bad,
        });
    }
}

/// Number of parameters with a non-finite (NaN/Inf) gradient in `grads`.
fn nonfinite_grad_count(model: &MambaVocabNet, grads: &GradientsParams) -> f32 {
    let mut v = GradBadCount { grads, acc: None };
    model.visit(&mut v);
    v.acc.map(scalar_f32).unwrap_or(0.0)
}

/// Visitor that records the first parameter (in visitation order) whose gradient
/// is non-finite — its rank/shape and id, enough to locate the matrix. Only run
/// on the failing step (per-parameter host syncs).
struct FirstBadGrad<'a> {
    grads: &'a GradientsParams,
    found: Option<String>,
}

impl ModuleVisitor for FirstBadGrad<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if self.found.is_some() {
            return;
        }
        let Some(g) = self.grads.get::<D>(param.id) else {
            return;
        };
        let nan = scalar_f32(g.clone().is_nan().any().int().float()) > 0.0;
        let inf = scalar_f32(g.is_inf().any().int().float()) > 0.0;
        if nan || inf {
            self.found = Some(format!(
                "dims={:?} id={:?} (nan={nan}, inf={inf})",
                param.val().dims(),
                param.id
            ));
        }
    }
}

fn first_bad_grad(model: &MambaVocabNet, grads: &GradientsParams) -> Option<String> {
    let mut v = FirstBadGrad { grads, found: None };
    model.visit(&mut v);
    v.found
}

/// On a non-finite combined gradient, re-run each loss term's forward+backward
/// in isolation and report which term(s) produce the non-finite gradient (and
/// on which parameter) — the decisive CE / weight-PR / state-PR attribution.
/// Recomputes fresh graphs per term (this runs once, at the failing step).
fn report_nonfinite_grads(
    model: &MambaVocabNet,
    x_bs: &Tensor<2, Int>,
    targets_bp: &Tensor<2>,
    ce: &burn_mamba::modules::loss::cross_entropy::CrossEntropyLoss,
    config: &GrokkingConfig,
    pr_lambda: f64,
) {
    eprintln!("[grad-nan] combined gradient is non-finite; isolating per loss term:");

    let report = |name: &str, loss: Tensor<1>| {
        let grads = GradientsParams::from_grads(loss.backward(), model);
        let count = nonfinite_grad_count(model, &grads);
        let where_ = first_bad_grad(model, &grads).unwrap_or_else(|| "-".to_string());
        eprintln!("[grad-nan]   {name:>10}: {count} bad-grad param(s); first: {where_}");
    };

    // CE term (does not need the moments path).
    let ce_logits = final_logits(model, x_bs, config.stepwise);
    report("ce", ce.forward(ce_logits, targets_bp.clone()));

    if pr_lambda != 0.0 {
        let p = diagnostics::weight_pr_penalty(model, config.pr_target);
        report("weight-pr", p.mul_scalar(pr_lambda));
    }
    if config.l2_lambda != 0.0 {
        let p = diagnostics::weight_l2_penalty(model, config.pr_target);
        report("l2", p.mul_scalar(config.l2_lambda));
    }
    if config.state_pr_lambda != 0.0 {
        let (_logits, moments) = final_logits_with_moments(model, x_bs);
        let pairing = diagnostics::state_pairing_of(model);
        let p = diagnostics::state_pr_penalty(&moments, &pairing);
        report("state-pr", p.mul_scalar(config.state_pr_lambda));
    }
}

/// Compact console form of the diagnostics (centered PRs are the primary
/// read-outs).
pub fn format_prs(state_prs: &[StatePr], weight_prs: &WeightPr) -> String {
    let states: Vec<String> = state_prs
        .iter()
        .map(|r| {
            format!(
                "L{}H{} pooled {:.2} (m{:.1e}), final {:.2} (m{:.1e})",
                r.layer, r.head, r.pooled_centered, r.pooled_trace, r.final_centered, r.final_trace
            )
        })
        .collect();
    let blocks: Vec<String> = weight_prs
        .layers
        .iter()
        .map(|l| {
            format!(
                "L{} z {:.1}, x {:.1}, B {:.1}, C {:.1}, out {:.1}, B-alpha {:.1}",
                l.layer, l.z, l.x, l.b, l.c, l.out, l.b_alphabet
            )
        })
        .collect();
    format!(
        "state PR [{}] | weight PR emb {:.2}, head {:.2}, emb-freq {:.2} | block [{}]",
        states.join("; "),
        weight_prs.emb,
        weight_prs.lm_head,
        weight_prs.emb_freq,
        blocks.join("; "),
    )
}

/// Append one metrics row, creating the file with a header on first use.
fn append_metrics(
    path: &std::path::Path,
    step: usize,
    lr: f64,
    train_loss: f32,
    train_acc: f64,
    test_acc: f64,
    weight_prs: &WeightPr,
) {
    use std::io::Write as _;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open the metrics csv");
    if needs_header {
        writeln!(file, "step,lr,train_loss,train_acc,test_acc,emb_pr,head_pr,emb_freq_pr")
            .expect("failed csv header write");
    }
    writeln!(
        file,
        "{step},{lr},{train_loss},{train_acc},{test_acc},{},{},{}",
        weight_prs.emb, weight_prs.lm_head, weight_prs.emb_freq,
    )
    .expect("failed csv write");
}

/// [`append_metrics`] without diagnostics: the weight-PR columns are written
/// as `nan` so the file keeps one schema either way.
fn append_metrics_bare(
    path: &std::path::Path,
    step: usize,
    lr: f64,
    train_loss: f32,
    train_acc: f64,
    test_acc: f64,
) {
    let nan = WeightPr {
        emb: f64::NAN,
        lm_head: f64::NAN,
        emb_freq: f64::NAN,
        layers: Vec::new(),
    };
    append_metrics(path, step, lr, train_loss, train_acc, test_acc, &nan);
}

/// Append the per-(layer, head) state-PR rows, creating the file with a
/// header on first use.
fn append_pr(path: &std::path::Path, step: usize, state_prs: &[StatePr]) {
    use std::io::Write as _;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open the pr csv");
    if needs_header {
        writeln!(
            file,
            "step,layer,head,pooled_centered,pooled_uncentered,final_centered,final_uncentered,pooled_trace,final_trace"
        )
        .expect("failed csv header write");
    }
    for r in state_prs {
        writeln!(
            file,
            "{step},{},{},{},{},{},{},{},{}",
            r.layer, r.head, r.pooled_centered, r.pooled_uncentered, r.final_centered, r.final_uncentered, r.pooled_trace, r.final_trace,
        )
        .expect("failed csv write");
    }
}

/// Append the per-layer block-weight PR rows, creating the file with a header
/// on first use.
fn append_weight_pr(path: &std::path::Path, step: usize, weight_prs: &WeightPr) {
    use std::io::Write as _;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open the weights csv");
    if needs_header {
        writeln!(file, "step,layer,z,x,b,c,out,b_alphabet").expect("failed csv header write");
    }
    for l in &weight_prs.layers {
        writeln!(
            file,
            "{step},{},{},{},{},{},{},{}",
            l.layer, l.z, l.x, l.b, l.c, l.out, l.b_alphabet,
        )
        .expect("failed csv write");
    }
}
