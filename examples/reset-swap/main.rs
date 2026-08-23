//! # Reset-swap example — the smallest task an `SO(4)` Mamba-3 block is needed for
//!
//! The corollary of `reset-spinor`, one rung further up. Same shape of stream —
//! two kinds of turn and a reset — but now the turns are **swaps**: the model
//! reads `s` / `t` / `R` and must report, at every position, how the three items
//! `abc` are ordered, i.e. the running word in the symmetric group `S₃`.
//!
//! Where `reset-spinor` needs the transition to be **non-abelian**, this needs
//! it to be *the group itself rather than a double cover*:
//!
//! - the answer is not a function of the current symbol, and Mamba-3 has no
//!   short convolution, so the recurrent state is the only memory,
//! - `st ≠ ts`, so a `cumsum` of angles — everything
//!   [`RotationKind::Complex2D`] can accumulate — cannot decide it,
//! - **and every swap must square to the identity.** A transposition has order
//!   2, but in `SU(2)` the *only* element of order 2 is `−1`: a half-turn lifts
//!   to `(0, û)`, whose square is `−1`. So a left-isoclinic
//!   ([`RotationKind::Quaternion4D`]) state runs in the double cover `2D₃`, and
//!   the two lifts of one permutation are **antipodal** state vectors that carry
//!   the same label — which no linear readout can merge.
//!
//! Two-sided ([`RotationKind::Rotor4D`]) the block reaches conjugation
//! `v ↦ q v q̄`, i.e. `SO(3) ⊂ SO(4)`, where `±q` act identically, the three
//! swaps are three honest half-turns about three axes `60°` apart, and the state
//! *is* the permutation.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example reset-swap -- --training --inference
//!
//! # the ablations: the same model one and two rungs down
//! cargo run --release --example reset-swap -- --training --inference -- --rotation quaternion
//! cargo run --release --example reset-swap -- --training --inference -- --rotation complex
//!
//! # the claims above, measured: a hand-built exact solution, its left-isoclinic
//! # twin, and the ceilings
//! cargo test --release --example reset-swap -- --nocapture
//! ```
//!
//! Like the other `reset-*` examples this carries a downstream flag,
//! `--rotation complex|quaternion|rotor` (default `rotor`), forwarded after the
//! trailing `--`; it selects the rotation baked into a **fresh** model config
//! (a persisted one wins on reload).

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The reset-swap dataset, its `S₃` arithmetic and its families.
pub mod dataset;
/// Inference: per-family accuracy on fresh eval sets.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the reset-swap task.
pub mod training;

/// The hand-built solution, its left-isoclinic twin, and the ceilings.
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
        // As in `reset-spinor`: a large step to leave the order-blind solution,
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

/// `--rotation complex|quaternion|rotor`, defaulting to the full `SO(4)`
/// rotation this example is about.
fn parse_rotation(extra_args: &[OsString]) -> RotationKind {
    let value = extra_args
        .iter()
        .position(|a| a == "--rotation")
        .and_then(|i| extra_args.get(i + 1))
        .map(|v| v.to_string_lossy().into_owned());
    match value.as_deref() {
        Some("rotor") | Some("so4") | None => RotationKind::Rotor4D,
        Some("quaternion") | Some("quat") => RotationKind::Quaternion4D,
        Some("complex") => RotationKind::Complex2D,
        Some(other) => {
            panic!("--rotation must be 'complex', 'quaternion' or 'rotor', got {other:?}")
        }
    }
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
