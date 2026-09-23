//! # spinor-product: the smallest task that needs `micro_steps = 2`
//!
//! The stream of `reset-spinor`, read **two symbols per token**. At every
//! position, the model sees an ordered pair from `i` / `j` / `k` / `.` / `R`. It
//! must report the running product in the quaternion group `Q₈` after both
//! symbols.
//!
//! A Mamba-3 step applies one rotation, and its generator is an affine
//! functional of the token. So at `micro_steps = 1`, the generators of the two
//! symbols can only **add**, and `exp(v + w) ≠ exp(w) ⊗ exp(v)` for the
//! non-commuting pairs. `micro_steps = 2` (`MambaProduct`,
//! `burn_mamba::mamba3::product`) makes a token two recurrence steps, so its
//! transition is the product itself. That one config change is the whole
//! example.
//!
//! After the trailing `--`, the example takes two flags. Both set a **fresh**
//! model config (on reload, the saved config wins):
//!
//! - `--micro-steps N` (default 2).
//! - `--layers N` (default 1). A second *layer* is the other way to put two
//!   rotations in one token, and `--layers 2 --micro-steps 1` is the contrast.
//!   A second layer turns a second state. So the layer below must still compute
//!   the product as a **feature**. That is expressible (`tests.rs`), but a
//!   learned product is only approximate, and `u = 2` is exact.
//!
//! `examples/reset/README.md` gives the task, the measurements and how to run
//! it.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The example's own flags (`--micro-steps`, `--layers`).
pub mod cli;
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

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    let cli::Cli {
        micro_steps,
        layers,
    } = cli::Cli::parse(app_args);
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
        // As in the `reset-*` ladder: a large step to leave the order-blind
        // solutions, then a small one to settle the rotation onto exact
        // half-turns.
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
        println!("Initializing new model config (micro_steps = {micro_steps}, layers = {layers})");
        model::model_config(micro_steps, layers)
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
