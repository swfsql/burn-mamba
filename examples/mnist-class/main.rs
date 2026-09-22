//! # Sequential-MNIST classifier example
//!
//! Classifies MNIST digits by reading each image as a length-784 sequence of
//! single-pixel tokens with a Mamba-3 model (a classification head on the last
//! timestep), trained with cosine-annealing LR. Inference samples a few test
//! digits and shows each digit beside its 10-bin class-probability chart.
//!
//! The optimizer is AdamW unless the shared flags say otherwise: `--muon` moves
//! the block's hidden weight matrices to [Muon](burn_stack::optim) (the fused
//! `in_proj` is split per sub-projection first); `--sgd` trains every weight
//! with plain SGD instead, the one optimizer whose training step can replay from
//! a captured CUDA graph (`--no-graph` steps it eagerly). The choice is written
//! into the artifacts' `training_config.json`, so resuming a run keeps it. It
//! takes no flags of its own after `--` ([`cli`]).
//!
//! ```bash
//! # baseline (AdamW everywhere)
//! cargo run --release --example mnist-class --features backend-flex -- --training
//! # AdamW + Muon on the hidden matrices
//! cargo run --release --example mnist-class --features backend-flex -- --training --muon
//! # SGD, the training step replayed from a graph
//! cargo run --release --example mnist-class --features backend-cuda -- --training --sgd
//! ```

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    mnist::dataset,
    training::{CosineAnnealingLr, Lr, OptimizerConfig, OptimizerKind, TrainingConfig},
};

/// The example's own flags (none).
pub mod cli;
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
    cli::Cli::parse(app_args);
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
        // Muon reuses its fallback's LR (`MatchRmsAdamW` sizes its update to
        // AdamW's RMS), so the peak rate follows the fallback: SGD's raw
        // gradient step wants a larger one, at the same peak-to-floor ratio.
        let optimizer = app_args.optimizer_or(OptimizerKind::AdamW);
        let (max_lr, min_lr) = match optimizer {
            OptimizerKind::AdamW | OptimizerKind::MuonAdamW => (9.6e-3, 2.4e-4),
            OptimizerKind::Sgd | OptimizerKind::MuonSgd => (5e-2, 1.25e-3),
        };
        TrainingConfig::new(OptimizerConfig::of(optimizer, dtype))
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
    app_args.override_training_config(&mut training_config, dtype);
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

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    launch(&app_args);
}
