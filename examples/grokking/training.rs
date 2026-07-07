//! Full-batch training loop for the grokking task: AdamW with plain
//! (non-cautious) decoupled weight decay, cross-entropy on the final position
//! only, and train/test accuracy logged to `metrics.csv` in the artifacts
//! directory (log-spaced early via power-of-two steps, then every
//! `eval_every`). The whole train split is one batch — the grokking-literature
//! setup, and at `p = 97` it is at most 9408 two-token sequences.

pub use crate::common::cli::AppArgs;
use crate::dataset::{self, Split};
use burn::module::AutodiffModule;
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
    /// Fraction of the `p²` pairs used for training.
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
    /// ~7× faster at these tiny sequence lengths, and exposes the per-step
    /// state caches (the capture point for the state-PR diagnostic).
    #[config(default = true)]
    pub stepwise: bool,
}

/// The SSD path used by chunkwise forwards: the recompute-backward serial
/// algorithm (the memory-saving custom backward) with `chunk_len = 2` matching
/// the two-token sequences (the default ≈32 chunk would zero-pad every
/// sequence 16×).
pub fn ssd_path() -> MambaSsdPath {
    MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(Some(2)))
}

/// Final-position logits `[n, p]` for a batch of token sequences `[n, s]`,
/// either chunkwise (`forward()`) or token-by-token (`step()`; identical by
/// the library's parity contract).
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
        let (logits_bsc, _caches) = model.forward(inputs_bs.clone(), None, ssd_path());
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

    let mut model: MambaVocabNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let mut optim = app_args.load_or_save_optim(&config.optimizer);

    let (train_split, test_split) =
        dataset::build(config.p, config.train_fraction, config.split_seed);
    println!(
        "p = {}, train pairs: {}, test pairs: {} (fraction {})",
        config.p,
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

    let ce = CrossEntropyLossConfig::new().init();
    let metrics_path = app_args.artifacts_path.join("metrics.csv");
    println!("logging metrics to {metrics_path:?}");

    println!("Starting training...");
    let started = std::time::Instant::now();
    for step in 1..=config.num_steps {
        let logits_bc = final_logits(&model, &x_bs, config.stepwise);
        let loss = ce.forward(logits_bc, targets_bp.clone());
        let loss_value = scalar_f32(loss.clone());

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let lr = config.lr.get_lr(step);
        model = optim.step(lr, model, grads);

        let last = step == config.num_steps;
        if step.is_power_of_two() || step % config.eval_every == 0 || last {
            let valid_model = model.valid();
            let train_acc = accuracy(&valid_model, &eval_train.0, &eval_train.1, config.stepwise);
            let test_acc = accuracy(&valid_model, &eval_test.0, &eval_test.1, config.stepwise);
            println!(
                "step {step:>6}/{}, loss {loss_value:.4e}, train acc {train_acc:.4}, \
                 test acc {test_acc:.4}, lr {lr:.2e}, {:.1}s",
                config.num_steps,
                started.elapsed().as_secs_f64(),
            );
            append_metrics(&metrics_path, step, lr, loss_value, train_acc, test_acc);
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

/// Append one metrics row, creating the file with a header on first use.
fn append_metrics(
    path: &std::path::Path,
    step: usize,
    lr: f64,
    train_loss: f32,
    train_acc: f64,
    test_acc: f64,
) {
    use std::io::Write as _;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open the metrics csv");
    if needs_header {
        writeln!(file, "step,lr,train_loss,train_acc,test_acc").expect("failed csv header write");
    }
    writeln!(file, "{step},{lr},{train_loss},{train_acc},{test_acc}").expect("failed csv write");
}
