//! # Reset-spinor: the smallest task that needs a *quaternion* Mamba-3 block
//!
//! The same argument as `reset-rotor`, one rung up the ladder. The stream has
//! the same shape, two kinds of turn and a reset. But the two turns **do not
//! commute**. The model reads `i` / `j` / `R`. At every position, it must report
//! the running product in the quaternion group `Q₈` since the last reset (one of
//! `±1, ±i, ±j, ±k`).
//!
//! `reset-rotor` needs a *complex* transition: a rotation, so the state can be
//! periodic. This rung needs a **non-abelian** transition. `ij = k` but
//! `ji = −k`, so only the *order* decides. An abelian rotation accumulates a
//! `cumsum` of angles, and a sum forgets order: it computes exactly the
//! abelianisation `Q₈/{±1}`. `Q₈` is the smallest group that the
//! [`RotationKind::Quaternion4D`] state of the block contains and its
//! [`RotationKind::Complex2D`] state does not.
//!
//! After the trailing `--`, the example takes a `--rotation
//! complex|quaternion|rotor` flag (default `quaternion`). It selects the
//! rotation of a **fresh** model config (on reload, the saved config wins).
//!
//! `examples/reset/README.md` gives the task, the measurements and how to run
//! it.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The example's own flags (`--rotation`).
pub mod cli;
/// The reset-spinor dataset, its `Q₈` arithmetic and its families.
pub mod dataset;
/// Inference: per-family accuracy on fresh eval sets.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the reset-spinor task.
pub mod training;

/// The hand-built solution, its abelian twin, and the order-blind ceiling.
#[cfg(test)]
pub mod tests;

/// Shared example infrastructure (included by path).
#[path = "../../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    let rotation = cli::Cli::parse(app_args).rotation;
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (and it
    // honours the `BURN_DEVICE` env override). `configure_dtype` installs
    // fp16/i32 when `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    // Training needs an autodiff-enabled device. Inference uses the plain one.
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    let (batch_size, num_epochs) = (64, 80);
    let mut training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // As in `reset-rotor`: a large step to leave the order-blind solution,
        // then a small one to settle the rotation onto exact half-turns.
        let total_steps = num_epochs * dataset::NUM_TRAIN.div_ceil(batch_size);
        let optimizer = app_args.optimizer_or(common::training::OptimizerKind::AdamW);
        TrainingConfig::new(common::training::OptimizerConfig::of(optimizer, dtype))
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
        println!("Initializing new model config ({rotation:?})");
        model::model_config(rotation)
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
