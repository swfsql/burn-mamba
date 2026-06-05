//! # MNIST autoencoder example
//!
//! A symmetric, fully **bidirectional** ViT/MAE-style **patch** autoencoder over
//! MNIST (attention blocks replaced by Mamba-3). The 28×28 image is cut into
//! `patch×patch` tiles (default 7×7 ⇒ a length-16 sequence); the encoder
//! compresses it to a small latent `z` (the configurable bottleneck); the
//! decoder reconstructs every patch in one parallel pass **reading only from
//! `z`** (a learned positional query FiLM-modulated by `z`). See [`model`] for
//! the architecture and the design rationale.
//!
//! The latent width is chosen with `-- --latents N` (default 64). `Complex2D`
//! rotation, BCE reconstruction loss.
//!
//! ## Run
//!
//! ```bash
//! # quick type-check on flex (fp32)
//! cargo check --example mnist-ae --features "backend-flex"
//!
//! # train + reconstruct on CUDA (long-running); 64-latent bottleneck
//! cargo run --release --example mnist-ae --features "backend-cuda,fusion" -- --training --inference
//! ```

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    mnist::dataset,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// Inference: reconstruct a few test images and print them as ASCII art.
pub mod inference;
/// The example's model ([`AeModel`](model::AeModel)) and `model_config()`.
pub mod model;
/// Training entry point for the autoencoder.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

use std::ffi::OsString;

/// Wire up the device, configs, and the train/infer flow for the autoencoder.
pub fn launch(app_args: &AppArgs) {
    // The only downstream argument: the latent bottleneck width baked into a
    // fresh model config. (Once a model config is persisted, it wins on reload.)
    let n_latent = parse_latents(&app_args.extra_args);
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
    let training_items = 60_000;
    let iterations_per_epoch = training_items / batch_size;
    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        TrainingConfig::new(common::training::optimizer_config(dtype))
            .with_num_epochs(5)
            .with_batch_size(batch_size)
            .with_num_workers(2)
            .with_lr(Lr::CosineAnnealing(
                CosineAnnealingLr::new(1 * iterations_per_epoch) // 1 epoch
                    .with_max_lr(1e-3)
                    .with_min_lr(1e-5)
                    .with_warmup_steps(iterations_per_epoch * 5 / 100), // 5% of an epoch
            ))
    });
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

/// Parse `--latents N` from the forwarded `extra_args` (defaults to 64).
fn parse_latents(extra_args: &[OsString]) -> usize {
    extra_args
        .iter()
        .position(|a| a == "--latents")
        .and_then(|i| extra_args.get(i + 1))
        .map(|v| {
            v.to_string_lossy()
                .parse::<usize>()
                .expect("--latents must be a positive integer")
        })
        .unwrap_or(64)
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
