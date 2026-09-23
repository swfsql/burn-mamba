//! # MNIST autoencoder example
//!
//! A symmetric, fully **bidirectional** ViT/MAE-style **patch** autoencoder over
//! MNIST (attention blocks replaced by Mamba-3).
//!
//! - The 28×28 image is cut into `patch×patch` tiles.
//! - The encoder compresses it to a small latent `z` (the configurable
//!   bottleneck).
//! - The decoder reconstructs every patch in one parallel pass that **reads only
//!   from `z`** (a learned positional query, FiLM-modulated by `z`).
//!
//! See [`model`] for the architecture and the design rationale. This example is
//! a work in progress: its sizes and hyperparameters are placeholders until a
//! parameter search.
//!
//! `-- --latents N` sets the latent width (default 16, [`cli`]). The block uses
//! the `Quaternion4D` rotation and virtual layers. The loss is a BCE
//! reconstruction loss.
//!
//! The optimizer is AdamW, unless the shared flags select a different one.
//! `--sgd` trains with plain SGD, the one optimizer whose training step can
//! replay from a captured CUDA graph (`--no-graph` runs it eagerly).
//!
//! ## Run
//!
//! ```bash
//! # quick type-check on flex (fp32)
//! cargo check --example mnist-ae --features "backend-flex"
//!
//! # train + reconstruct on CUDA (long-running), 16-latent bottleneck
//! cargo run --release --example mnist-ae --features "backend-cuda,fusion" -- --training --inference
//!
//! # SGD, the training step replayed from a graph
//! cargo run --release --example mnist-ae --features backend-cuda -- --training --sgd
//! ```

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    mnist::dataset,
    training::{CosineAnnealingLr, Lr, OptimizerConfig, OptimizerKind, TrainingConfig},
};

/// The example's own flags (`--latents`).
pub mod cli;
/// Inference: reconstruct a few test images and print them as ASCII art.
pub mod inference;
/// The example's model ([`AeModel`](model::AeModel)) and `model_config()`.
pub mod model;
/// Training entry point for the autoencoder.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the autoencoder.
pub fn launch(app_args: &AppArgs) {
    let n_latent = cli::Cli::parse(app_args).latents;
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (and it
    // honours the `BURN_DEVICE` env override). `configure_dtype` installs
    // fp16/i32 when `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    // setup training and model configs (placeholder values, see the header)
    let batch_size = 16;
    let num_epochs = 3;
    let training_items = 60_000;
    let iterations_per_epoch = training_items / batch_size;
    let mut training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // Muon uses the LR of its fallback (`MatchRmsAdamW` scales its update to
        // the RMS of AdamW), so the peak rate follows the fallback. The raw
        // gradient step of SGD needs a larger peak, at the same peak-to-floor
        // ratio.
        let optimizer = app_args.optimizer_or(OptimizerKind::AdamW);
        let (max_lr, min_lr) = match optimizer {
            OptimizerKind::AdamW | OptimizerKind::MuonAdamW => (1e-3, 5e-5),
            OptimizerKind::Sgd | OptimizerKind::MuonSgd => (5e-2, 2.5e-3),
        };
        TrainingConfig::new(OptimizerConfig::of(optimizer, dtype))
            .with_num_epochs(num_epochs)
            .with_batch_size(batch_size)
            .with_num_workers(2)
            .with_lr(Lr::CosineAnnealing(
                // The cosine decay spans the whole run. A 1-epoch period would
                // drop the LR to its floor after one epoch.
                CosineAnnealingLr::new(num_epochs * iterations_per_epoch)
                    .with_max_lr(max_lr)
                    .with_min_lr(min_lr)
                    .with_warmup_steps(iterations_per_epoch * 5 / 100), // 5% of an epoch
            ))
    });
    app_args.override_training_config(&mut training_config, dtype);
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config (n_latent={n_latent})");
        model::model_config(n_latent)
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
