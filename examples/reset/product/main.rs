//! # spinor-product — the smallest task `micro_steps = 2` is needed for
//!
//! `reset-spinor`'s stream, read **two symbols per token**: the model sees an
//! ordered pair from `i` / `j` / `.` / `R` at every position and must report the
//! running product in the quaternion group `Q₈` after both of them.
//!
//! A Mamba-3 step applies one rotation, whose generator is an affine functional
//! of the token — so at `micro_steps = 1` the two symbols' generators can only
//! **add**, and `exp(v + w) ≠ exp(w) ⊗ exp(v)` for the non-commuting pairs.
//! `micro_steps = 2` (`MambaProduct`, `burn_mamba::mamba3::product`) makes a
//! token two recurrence steps, so its transition is the product itself. That
//! one config change is the whole example.
//!
//! Carries a downstream `--micro-steps N` flag (default 2) after the trailing
//! `--`; it selects the `u` baked into a **fresh** model config (a persisted one
//! wins on reload).
//!
//! The task, the measurements and how to run it: `examples/reset/README.md`.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The paired-symbol dataset, its `Q₈` arithmetic and its families.
pub mod dataset;
/// Inference: per-family accuracy on fresh eval sets.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the paired `Q₈` task.
pub mod training;

/// The hand-built `u = 2` solution, its `u = 1` twin, and the ceilings.
#[cfg(test)]
pub mod tests;

/// Shared example infrastructure (included by path).
#[path = "../../common/mod.rs"]
pub mod common;

use std::ffi::OsString;

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    // The only downstream argument: how many micro-steps a fresh model config
    // runs per token. (Once a model config is persisted, it wins on reload.)
    let micro_steps = parse_micro_steps(&app_args.extra_args);
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
        // As in the `reset-*` ladder: a large step to leave the order-blind
        // solutions, a small one to settle the rotation onto exact half-turns.
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
        println!("Initializing new model config (micro_steps = {micro_steps})");
        model::model_config(micro_steps)
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

/// `--micro-steps N`, defaulting to the 2 this example is about (`1` is the
/// ablation: stock Mamba-3, one rotation per token).
fn parse_micro_steps(extra_args: &[OsString]) -> usize {
    let value = extra_args
        .iter()
        .position(|a| a == "--micro-steps")
        .and_then(|i| extra_args.get(i + 1))
        .map(|v| v.to_string_lossy().into_owned());
    match value {
        None => dataset::PAIR,
        Some(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("--micro-steps takes a positive integer, got {v:?}")),
    }
}

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    launch(&app_args);
}
