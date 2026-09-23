//! What the `tally-*` rungs share: the symbol-stream dataset and batcher, the
//! train/validate/infer loops, and the few scalar helpers of the hand-built
//! constructions. A rung supplies a [`Task`] (its alphabet, its labels and its
//! generators) and a model config. Everything else is here.
//!
//! Every rung reads one-hot symbols and classifies **every** position into two
//! classes. [`IGNORE`] marks a position whose target is not a function of the
//! history. Such a position reaches neither the loss nor the accuracy.

#![allow(dead_code)]

/// Dataset, batcher and the deterministic generator RNG.
pub mod data;
/// Scalar helpers for the hand-built blocks: embeddings, the affine channel
/// solve, tensor builders, accuracy.
pub mod handmade;
/// Inference: per-family accuracy on the evaluation sets.
pub mod inference;
/// Training loop.
pub mod training;

pub use data::{IGNORE, Rng, Task};

use crate::common::cli::AppArgs;
use crate::common::training::{
    CosineAnnealingLr, Lr, OptimizerConfig, OptimizerKind, TrainingConfig,
};
use burn_mamba::prelude::MambaLatentNetConfig;

/// Set up the device, the configs and the train/infer flow for one rung. The
/// `cli.rs` of the rung parses its own downstream flags, and its `main.rs` puts
/// them into `model_config` before it calls this.
pub fn launch(app_args: &AppArgs, task: &Task, model_config: MambaLatentNetConfig) {
    app_args.create_artifact_dir();

    let mut device = burn::prelude::Device::default();
    crate::common::device::configure_dtype(&mut device);
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    let (batch_size, num_epochs) = (64, 80);
    let mut training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // As on the `reset` ladder: a large step to leave the memoryless basin,
        // and a small step to reach the exact construction.
        let total_steps = num_epochs * data::NUM_TRAIN.div_ceil(batch_size);
        let optimizer = app_args.optimizer_or(OptimizerKind::AdamW);
        TrainingConfig::new(OptimizerConfig::of(optimizer, dtype))
        .with_num_epochs(num_epochs)
        .with_batch_size(batch_size)
        .with_num_workers(2)
        .with_lr(Lr::CosineAnnealing(
            CosineAnnealingLr::new(total_steps)
                .with_max_lr(3e-2)
                .with_min_lr(1e-4)
                .with_warmup_steps(100),
        ))
    });
    app_args.override_training_config(&mut training_config, dtype);
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model_config
    });
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(task, training_config, model_config.clone(), autodiff_device, app_args);
    }
    if app_args.inference {
        inference::infer(task, model_config, device, app_args);
    }
    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", crate::common::cli::HELP);
    }
}
