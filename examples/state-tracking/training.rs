//! Full-batch training loop for the `A₅` word problem, in the **grokking
//! protocol** (the direct transposition of the grokking example's trainer):
//! AdamW with plain decoupled weight decay, cross-entropy on the **final
//! position only**, PR diagnostics and the differentiable rank penalties at
//! eval points, and everything logged to CSVs in the artifacts directory.
//! The extra, task-specific read-out is the **per-position accuracy** on the
//! held-out split — how deep the composition actually holds (trained
//! final-only, probed everywhere).

pub use crate::common::cli::AppArgs;
use crate::common::protocol::{
    self, append_metrics, append_metrics_bare, append_pr, append_weight_pr, first_bad_grad,
    nonfinite_grad_count, scalar_f32,
};
pub use crate::common::protocol::format_prs;
use crate::dataset::{self, CLASS_BASE, NUM_CLASSES, NUM_SYMBOLS, Split};
use crate::diagnostics;
pub use crate::diagnostics::PrPenaltyTarget;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn_mamba::modules::loss::cross_entropy::CrossEntropyLossConfig;
use burn_mamba::prelude::*;
pub use burn_mamba::utils::scheduler::{ConstantLr, Lr};

/// Word-problem hyperparameters: optimizer + task/split + schedule knobs.
/// The schedule/penalty knobs mirror the grokking example's `GrokkingConfig`
/// (weight decay is the grokking driver; `0.0` is the memorization control).
#[derive(Config, Debug)]
pub struct TrackingConfig {
    /// The optimizer configuration (AdamW).
    pub optimizer: AdamWConfig,
    /// Generators per word (the token sequence is one longer — the anchor).
    /// The enumerated space is `NUM_GENERATORS^seq_len`.
    #[config(default = 12)]
    pub seq_len: usize,
    /// Fraction of the enumerated words used for training.
    #[config(default = 0.5)]
    pub train_fraction: f64,
    /// Seed for the deterministic train/test word split.
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
    /// `forward()` — mathematically identical (the library's parity
    /// contract). Off by default here: at 13 tokens the chunkwise recompute
    /// path is the cheap one, and the state-PR penalty needs it anyway.
    #[config(default = false)]
    pub stepwise: bool,
    /// Compute and log the PR diagnostics at eval points.
    #[config(default = true)]
    pub diagnostics: bool,
    /// Also run the (costlier) state-PR part of the diagnostics; weight PRs
    /// are always logged when `diagnostics` is on.
    #[config(default = true)]
    pub state_diagnostics: bool,
    /// Coefficient of the differentiable weight-PR penalty
    /// (`loss += pr_lambda · Σ PR(W)` over `pr_target`); `0` disables it.
    /// Negative values *reward* rank expansion.
    #[config(default = 0.0)]
    pub pr_lambda: f64,
    /// Which weights the PR penalty targets.
    #[config(default = "PrPenaltyTarget::All")]
    pub pr_target: PrPenaltyTarget,
    /// Period (in steps) of a sine modulation of the PR penalty (`0` =
    /// constant coefficient).
    #[config(default = 0)]
    pub pr_sine_period: usize,
    /// Offset added to logged step numbers, for resumed runs (the loop,
    /// eval/save cadence, and sine phase run on the raw step).
    #[config(default = 0)]
    pub step_offset: usize,
    /// Keep the PR penalty off until this (raw) step.
    #[config(default = 0)]
    pub pr_start_step: usize,
    /// Coefficient of the plain L2 (Frobenius²) loss penalty over the same
    /// `pr_target` matrices (rank-specificity control). `0` disables.
    #[config(default = 0.0)]
    pub l2_lambda: f64,
    /// Coefficient of the weight-independent-gradient control `Σ ⟨W, ε⟩`
    /// (fresh `ε ~ N(0,1)` per step, detached). `0` disables.
    #[config(default = 0.0)]
    pub noise_lambda: f64,
    /// Coefficient of the differentiable **state**-PR penalty
    /// (`PR_ℂ(M_phys)`, batch-pooled, uncentered, Σ over layers/heads) from
    /// the library's state moments on the training forward. Requires the
    /// chunkwise path; `0` disables; negative rewards state-rank expansion.
    #[config(default = 0.0)]
    pub state_pr_lambda: f64,
}

impl TrackingConfig {
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

    /// The chunk length passed to the shared SSD path: one chunk spanning the
    /// whole (anchor-led) sequence.
    pub fn chunk_len(&self) -> Option<usize> {
        Some(self.seq_len + 1)
    }
}

/// The `NUM_CLASSES`-way slice of full-vocabulary logits (the class region).
fn class_logits<const D: usize>(logits: Tensor<D>) -> Tensor<D> {
    logits.narrow(D - 1, CLASS_BASE, NUM_CLASSES)
}

/// Fraction of examples whose final-position **class-slice** argmax matches
/// the label (labels are class indices `0..NUM_CLASSES`).
pub fn accuracy(
    model: &MambaVocabNet,
    inputs_bs: &Tensor<2, Int>,
    labels_b: &Tensor<1, Int>,
    stepwise: bool,
    chunk_len: Option<usize>,
) -> f64 {
    let logits_bc = class_logits(protocol::final_logits(model, inputs_bs, stepwise, chunk_len));
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
    chunk_len: Option<usize>,
) -> (f64, f64) {
    let train_acc = accuracy(model, &train.inputs_tensor(device), &train.labels_tensor(device), stepwise, chunk_len);
    let test_acc = accuracy(model, &test.inputs_tensor(device), &test.labels_tensor(device), stepwise, chunk_len);
    (train_acc, test_acc)
}

/// Per-position accuracy of the running-product read-out (the depth probe):
/// one chunkwise forward, class-slice argmax at **every** position against
/// the running products. Position 0 is the anchor (identity).
pub fn per_position_accuracy(
    model: &MambaVocabNet,
    inputs_bs: &Tensor<2, Int>,
    pos_targets_bs: &Tensor<2, Int>,
    chunk_len: Option<usize>,
) -> Vec<f64> {
    let [b, s] = inputs_bs.dims();
    let (logits_bsv, _caches) =
        model.forward(inputs_bs.clone(), None, protocol::ssd_path(model, chunk_len));
    let pred_bs = class_logits(logits_bsv).argmax(2).reshape([b, s]);
    let acc_1s = pred_bs.equal(pos_targets_bs.clone()).float().mean_dim(0);
    acc_1s
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(|a| a as f64)
        .collect()
}

/// Run the full training routine: load/init the model and optimizer, then take
/// `num_steps` full-batch steps, logging accuracies and checkpointing along
/// the way.
pub fn train(
    config: TrackingConfig,
    model_config: MambaVocabNetConfig,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(config.seed);
    let eval_device = training_device.clone().inner();
    assert!(
        config.state_pr_lambda == 0.0 || !config.stepwise,
        "state_pr_lambda needs the training forward's state moments — drop --stepwise"
    );
    let chunk = config.chunk_len();

    let mut model: MambaVocabNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let mut optim = app_args.load_or_save_optim(&config.optimizer, &model);

    let (train_split, test_split) =
        dataset::build(config.seq_len, config.train_fraction, config.split_seed);
    println!(
        "A₅ word problem: seq_len = {} (+1 anchor), train words: {}, test words: {} (fraction {})",
        config.seq_len,
        train_split.len(),
        test_split.len(),
        config.train_fraction,
    );

    // Full-batch training tensors live on the autodiff device; the eval copies
    // on the plain inner device.
    let x_bs = train_split.inputs_tensor(&training_device);
    let targets_bv = train_split.targets_tensor(&training_device);
    let eval_train = (
        train_split.inputs_tensor(&eval_device),
        train_split.labels_tensor(&eval_device),
    );
    let eval_test = (
        test_split.inputs_tensor(&eval_device),
        test_split.labels_tensor(&eval_device),
        test_split.pos_targets_tensor(&eval_device),
    );

    // The PR diagnostic's eval set: all words (or a deterministic 10k sample
    // when the space is larger), on the plain device.
    let diag_inputs = dataset::diagnostic_set(config.seq_len, 10_000, config.split_seed)
        .inputs_tensor(&eval_device);

    let ce = CrossEntropyLossConfig::new().init();
    let metrics_path = app_args.artifacts_path.join("metrics.csv");
    let pr_path = app_args.artifacts_path.join("pr.csv");
    let weights_path = app_args.artifacts_path.join("weights.csv");
    let positions_path = app_args.artifacts_path.join("positions.csv");
    println!(
        "logging metrics to {metrics_path:?}, state PR to {pr_path:?}, weight PR to \
         {weights_path:?}, per-position acc to {positions_path:?}"
    );

    println!("Starting training...");
    let started = std::time::Instant::now();
    for step in 1..=config.num_steps {
        // The state-PR penalty needs the training forward's attached moments,
        // so it forces the chunkwise path (asserted above).
        let (logits_bv, train_moments) = if config.state_pr_lambda != 0.0 {
            let (logits, moments) = protocol::final_logits_with_moments(&model, &x_bs, chunk);
            (logits, Some(moments))
        } else {
            (protocol::final_logits(&model, &x_bs, config.stepwise, chunk), None)
        };
        // `loss_value` (and the csv column) stays CE-only, comparable across
        // arms; the penalty value is printed separately at eval points.
        let ce_loss = ce.forward(logits_bv, targets_bv.clone());
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
            report_nonfinite_grads(&model, &x_bs, &targets_bv, &ce, &config, pr_lambda);
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
            let train_acc = accuracy(&valid_model, &eval_train.0, &eval_train.1, config.stepwise, chunk);
            let test_acc = accuracy(&valid_model, &eval_test.0, &eval_test.1, config.stepwise, chunk);
            println!(
                "step {logged_step:>6}/{}, loss {loss_value:.4e}, train acc {train_acc:.4}, \
                 test acc {test_acc:.4}, lr {lr:.2e}, {:.1}s",
                config.num_steps + config.step_offset,
                started.elapsed().as_secs_f64(),
            );
            // The depth probe: held-out per-position accuracy.
            let pos_acc = per_position_accuracy(&valid_model, &eval_test.0, &eval_test.2, chunk);
            println!("        test per-position acc: {}", format_per_position(&pos_acc));
            append_positions(&positions_path, logged_step, &pos_acc);
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
                let state_prs = if !config.state_diagnostics {
                    Vec::new()
                } else if config.stepwise {
                    diagnostics::state_pr(&valid_model, &diag_inputs)
                } else {
                    diagnostics::state_pr_forward(
                        &valid_model,
                        &diag_inputs,
                        protocol::ssd_path(&valid_model, chunk),
                    )
                };
                let weight_prs = diagnostics::weight_pr(&valid_model, NUM_SYMBOLS);
                println!("        {}", format_prs(&state_prs, &weight_prs));
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

/// On a non-finite combined gradient, re-run each loss term's forward+backward
/// in isolation and report which term(s) produce the non-finite gradient (and
/// on which parameter). Recomputes fresh graphs per term (runs once, at the
/// failing step).
fn report_nonfinite_grads(
    model: &MambaVocabNet,
    x_bs: &Tensor<2, Int>,
    targets_bv: &Tensor<2>,
    ce: &burn_mamba::modules::loss::cross_entropy::CrossEntropyLoss,
    config: &TrackingConfig,
    pr_lambda: f64,
) {
    eprintln!("[grad-nan] combined gradient is non-finite; isolating per loss term:");

    let report = |name: &str, loss: Tensor<1>| {
        let grads = GradientsParams::from_grads(loss.backward(), model);
        let count = nonfinite_grad_count(model, &grads);
        let where_ = first_bad_grad(model, &grads).unwrap_or_else(|| "-".to_string());
        eprintln!("[grad-nan]   {name:>10}: {count} bad-grad param(s); first: {where_}");
    };

    let chunk = config.chunk_len();
    let ce_logits = protocol::final_logits(model, x_bs, config.stepwise, chunk);
    report("ce", ce.forward(ce_logits, targets_bv.clone()));

    if pr_lambda != 0.0 {
        let p = diagnostics::weight_pr_penalty(model, config.pr_target);
        report("weight-pr", p.mul_scalar(pr_lambda));
    }
    if config.l2_lambda != 0.0 {
        let p = diagnostics::weight_l2_penalty(model, config.pr_target);
        report("l2", p.mul_scalar(config.l2_lambda));
    }
    if config.state_pr_lambda != 0.0 {
        let (_logits, moments) = protocol::final_logits_with_moments(model, x_bs, chunk);
        let pairing = diagnostics::state_pairing_of(model);
        let p = diagnostics::state_pr_penalty(&moments, &pairing);
        report("state-pr", p.mul_scalar(config.state_pr_lambda));
    }
}

/// Format a per-position accuracy vector as `"100% 73% …"`.
pub fn format_per_position(pos_acc: &[f64]) -> String {
    pos_acc
        .iter()
        .map(|a| format!("{:.0}%", 100.0 * a))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Append one per-position accuracy row, creating the file with a
/// (position-count-dependent) header on first use.
fn append_positions(path: &std::path::Path, step: usize, pos_acc: &[f64]) {
    use std::io::Write as _;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open the positions csv");
    if needs_header {
        let cols: Vec<String> = (0..pos_acc.len()).map(|t| format!("pos{t}")).collect();
        writeln!(file, "step,{}", cols.join(",")).expect("failed csv header write");
    }
    let row: Vec<String> = pos_acc.iter().map(|a| format!("{a}")).collect();
    writeln!(file, "{step},{}", row.join(",")).expect("failed csv write");
}
