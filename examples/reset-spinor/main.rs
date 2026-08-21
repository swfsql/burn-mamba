//! # Reset-spinor example — the smallest task a *quaternion* Mamba-3 block is needed for
//!
//! The corollary of `reset-rotor`, one rung further up the ladder. Same shape of
//! stream — two kinds of turn and a reset — but the two turns **do not commute**:
//! the model reads `i` / `j` / `R` and must report, at every position, the
//! running product in the quaternion group `Q₈` since the last reset (one of
//! `±1, ±i, ±j, ±k`).
//!
//! Where `reset-rotor` needs the transition to be *complex* — a rotation, so the
//! state can be periodic — this needs it to be **non-abelian**. `Q₈` is the
//! smallest non-abelian group of unit quaternions, which is to say the smallest
//! group that the block's own [`RotationKind::Quaternion4D`] state contains and
//! its [`RotationKind::Complex2D`] state does not:
//!
//! - the answer is not a function of the current symbol, and Mamba-3 has no
//!   short convolution, so the recurrent state is the only memory,
//! - the label is periodic in every generator (`i⁴ = 1`), which no real state
//!   can report — the `reset-rotor` argument,
//! - **and it is not a function of how many `i`s and `j`s went by.** `ij = k`
//!   but `ji = −k`, so only the *order* decides. An abelian rotation accumulates
//!   a `cumsum` of angles, and a sum forgets order: what it computes is exactly
//!   the abelianisation `Q₈/{±1}`, missing the commutator.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example reset-spinor -- --training --inference
//!
//! # the ablation: the same model with the abelian rotation
//! cargo run --release --example reset-spinor -- --training --inference -- --rotation complex
//!
//! # the claims above, measured: a hand-built exact solution, its abelian twin,
//! # and the ceiling for everything order-blind
//! cargo test --release --example reset-spinor -- --nocapture
//! ```
//!
//! Like `state-tracking`, this example carries a downstream flag,
//! `--rotation complex|quaternion` (default `quaternion`), forwarded after the
//! trailing `--`; it selects the rotation baked into a **fresh** model config
//! (a persisted one wins on reload).

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

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
#[path = "../common/mod.rs"]
pub mod common;

use burn_mamba::prelude::RotationKind;
use std::ffi::OsString;

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    // The only downstream argument: which rotation to bake into a fresh model
    // config. (Once a model config is persisted, it wins on reload — see HELP.)
    let rotation = parse_rotation(&app_args.extra_args);
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
        // As in `reset-rotor`: a large step to leave the order-blind solution,
        // a small one to settle the rotation onto exact half-turns.
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

/// `--rotation complex|quaternion`, defaulting to the quaternion rotation this
/// example is about.
fn parse_rotation(extra_args: &[OsString]) -> RotationKind {
    let value = extra_args
        .iter()
        .position(|a| a == "--rotation")
        .and_then(|i| extra_args.get(i + 1))
        .map(|v| v.to_string_lossy().into_owned());
    match value.as_deref() {
        Some("quaternion") | Some("quat") | None => RotationKind::Quaternion4D,
        Some("complex") => RotationKind::Complex2D,
        Some(other) => panic!("--rotation must be 'complex' or 'quaternion', got {other:?}"),
    }
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
