//! # Reset-majority — the smallest task a Mamba-2 block is *needed* for
//!
//! One Mamba-2 block, two scalar states, no convolution, no residual: the model
//! reads a stream of `+` / `-` / `R` symbols and must report, at **every**
//! position, the sign of the running vote **since the last `R`**.
//!
//! The bottom rung of the `reset-*` ladder, and the one requirement it adds is
//! that the decay be **data-dependent**: a reset must erase its past outright
//! while the votes after it stay unweighted. Two adversarial families in the
//! eval set pin that down from both sides — see [`dataset`](crate::dataset).
//!
//! The task, the measurements and how to run it: `examples/reset/README.md`.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The reset-majority dataset and its adversarial families.
pub mod dataset;
/// Inference: per-family accuracy on fresh eval sets.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the reset-majority task.
pub mod training;

/// The hand-built solution and the fixed-decay sweep (see the module docs).
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
        // Finding the selective solution needs a *large* step to leave the
        // memoryless basin, and a small one to settle into an exact hold once
        // there — a constant LR does one or the other, so this anneals.
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
