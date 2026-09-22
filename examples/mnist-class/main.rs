//! # Sequential-MNIST classifier example
//!
//! Classifies MNIST digits by reading each image as a length-784 sequence of
//! single-pixel tokens with a Mamba-3 model (a classification head on the last
//! timestep), trained with cosine-annealing LR. Inference samples a few test
//! digits and shows each digit beside its 10-bin class-probability chart.
//!
//! This example carries one downstream flag, forwarded after the trailing `--`,
//! choosing a fresh training config's optimizer: `--muon` moves the block's
//! hidden weight matrices from AdamW to [Muon](burn_stack::optim) (the fused
//! `in_proj` is split per sub-projection first); `--sgd` trains every weight
//! with plain SGD instead, the one optimizer whose training step can replay from
//! a captured CUDA graph (`MNIST_GRAPH=0` steps it eagerly). The choice is
//! written into the artifacts' `training_config.json`, so resuming a run keeps
//! it.
//!
//! ```bash
//! # baseline (AdamW everywhere)
//! cargo run --release --example mnist-class --features backend-flex -- --training
//! # AdamW + Muon on the hidden matrices
//! cargo run --release --example mnist-class --features backend-flex -- --training -- --muon
//! # SGD, the training step replayed from a graph
//! cargo run --release --example mnist-class --features backend-cuda -- --training -- --sgd
//! ```

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    mnist::dataset,
    training::{CosineAnnealingLr, Lr, OptimizerConfig, TrainingConfig},
};

/// Inference: classify a few test digits and show their class distributions.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the classifier.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the classifier.
pub fn launch(app_args: &AppArgs) {
    // The only downstream argument: a *fresh* training config's optimizer. (A
    // persisted training config wins on reload — see HELP.)
    let optimizer = parse_optimizer(&app_args.extra_args);
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (honouring
    // the `BURN_DEVICE` env override); `configure_dtype` installs fp16/i32 when
    // `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    // setup training and model configs
    let batch_size = 16;
    let num_epochs = 4;
    let training_items = 60_000;
    let iterations_per_epoch = training_items / batch_size;
    let mut training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // Muon reuses AdamW's LR and weight decay (`MatchRmsAdamW` sizes its
        // update to AdamW's RMS), so only the optimizer of the planned matrices
        // changes between the two arms. SGD's raw gradient step wants a larger
        // peak rate, at the same peak-to-floor ratio.
        let (optimizer, max_lr, min_lr) = match optimizer {
            Optimizer::AdamW => (OptimizerConfig::adamw_only(dtype), 9.6e-3, 2.4e-4),
            Optimizer::Muon => (
                OptimizerConfig::adamw_only(dtype).with_muon_defaults(ADAMW_WEIGHT_DECAY),
                9.6e-3,
                2.4e-4,
            ),
            Optimizer::Sgd => (OptimizerConfig::sgd_only(dtype), 5e-2, 1.25e-3),
        };
        TrainingConfig::new(optimizer)
            .with_num_epochs(num_epochs)
            .with_batch_size(batch_size)
            .with_num_workers(2)
            .with_lr(Lr::CosineAnnealing(
                CosineAnnealingLr::new(num_epochs * iterations_per_epoch)
                    .with_max_lr(max_lr)
                    .with_min_lr(min_lr)
                    .with_warmup_steps(iterations_per_epoch * 5 / 100), // 5% of an epoch
            ))
    });
    app_args.override_training_config(&mut training_config);
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config()
    });
    // save configs
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(
            training_config,
            model_config.clone(),
            autodiff_device,
            app_args,
        );
    }

    if app_args.inference {
        inference::infer(model_config, device, app_args);
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

/// AdamW's default weight decay, mirrored into the Muon group so the two arms
/// decay the same weights by the same amount.
const ADAMW_WEIGHT_DECAY: f32 = 1e-4;

/// A fresh training config's optimizer (the forwarded `extra_args`).
enum Optimizer {
    AdamW,
    Muon,
    Sgd,
}

/// Parse `--muon` or `--sgd` (at most one) from the forwarded `extra_args`.
fn parse_optimizer(extra_args: &[std::ffi::OsString]) -> Optimizer {
    let mut optimizer = Optimizer::AdamW;
    for arg in extra_args {
        assert!(
            matches!(optimizer, Optimizer::AdamW),
            "--muon and --sgd are exclusive"
        );
        optimizer = match arg.to_str() {
            Some("--muon") => Optimizer::Muon,
            Some("--sgd") => Optimizer::Sgd,
            _ => panic!("unknown extra argument: {arg:?}"),
        };
    }
    optimizer
}

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    launch(&app_args);
}
