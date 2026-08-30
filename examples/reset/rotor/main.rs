//! # Reset-rotor — the smallest task a Mamba-3 block is *needed* for
//!
//! The corollary of `reset-majority`, one family up. Same three symbols, same
//! reset: the model reads `+` / `-` / `R` and must report, at **every**
//! position, where a three-detent rotor stands — the running turn count since
//! the last `R`, **mod 3**.
//!
//! The rung isolates the one thing a Mamba-3 block has that a Mamba-2 block does
//! not: a **complex transition**, the data-dependent rotation absorbed into
//! `B`/`C`. It adds two requirements, one per adversarial family — the state has
//! to *turn*, since the label is periodic in a count that a real state can hold
//! but never reduce ([`Family::Drift`](crate::dataset::Family::Drift)); and the
//! turn has to be data-dependent, since a fixed per-step angle is vanilla RoPE,
//! whose phase measures position rather than turns
//! ([`Family::Balanced`](crate::dataset::Family::Balanced)).
//!
//! The task, the measurements and how to run it: `examples/reset/README.md`.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The reset-rotor dataset and its adversarial families.
pub mod dataset;
/// Inference: per-family accuracy on fresh eval sets.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the reset-rotor task.
pub mod training;

/// The hand-built solution and the two ablation sweeps (see the module docs).
#[cfg(test)]
pub mod tests;

/// Shared example infrastructure (included by path).
#[path = "../../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    assert!(
        app_args.extra_args.is_empty(),
        "no extra arguments required"
    );
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (honouring
    // the `BURN_DEVICE` env override); `configure_dtype` installs fp16/i32 when
    // `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    // training needs an autodiff-enabled device; inference uses the plain one.
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    let (batch_size, num_epochs) = (64, 80);
    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // Turning the state on is a basin to find, not a slope to descend: the
        // run needs a large step to leave the memoryless solution and a small
        // one to settle the angle onto an exact detent, so this anneals.
        let total_steps = num_epochs * dataset::NUM_TRAIN.div_ceil(batch_size);
        TrainingConfig::new(common::training::OptimizerConfig::new(
            common::training::optimizer_config(dtype),
        ))
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
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config()
    });
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
