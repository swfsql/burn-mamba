//! # Reset-quintic — `A₅` in one Mamba-3 block, `S₅` in two
//!
//! The stream of `reset-swap`, over **five** items. The model reads two turns
//! and a reset. At every position, it reports the arrangement of `abcde`: the
//! running word in `A₅` (`--group a5`: `d = (0 1)(2 3)`, `t = (0 2 4)`, sixty
//! classes) or in `S₅` (`--group s5`, the default: `s = (0 1)`,
//! `c = (0 1 2 3 4)`, a hundred and twenty classes).
//!
//! - `A₅` is the rotation group of the icosahedron: the smallest non-solvable
//!   group, the one behind the unsolvable quintic. The block of `reset-swap`
//!   holds it. A conjugating ([`RotationKind::Rotor4D`]) 4-block is `SO(3)`, and
//!   `d`/`t` are a half-turn and a third-turn about axes `20.9°` apart.
//! - `S₅` is not a rotation group of anything that the block turns. Its
//!   transpositions act on `A₅` by the automorphism that swaps the `72°` and
//!   `144°` rotations, and only a reflection does that. No single layer, at any
//!   size, of any rotation kind, tracks more of `S₅` than its sign. Two layers
//!   do: the first holds the sign, and the second holds the even part, turned by
//!   a step that reads both.
//!
//! The trained width ([`model::DEFAULT_WIDTH`]) is wider than the hand-built
//! floor ([`model::floor_width`]). Training puts the group on one head at a
//! time, and MIMO ranks give that head the readouts that the floor spreads over
//! heads.
//!
//! Flags after the trailing `--` ([`cli`]):
//!
//! - `--group a5|s5`. It is not persisted, so pass it on every run.
//! - `--rotation complex|quaternion|rotor` (default `rotor`), `--layers N`
//!   (default 1 for `a5`, 2 for `s5`), and the width `--d-model N --heads N
//!   --mimo-rank N --expand N` (default `4 2 2 1`, [`model::DEFAULT_WIDTH`]).
//!   These set a **fresh** model config (on reload, the saved config wins).
//! - `--train-length N` (default [`dataset::SEQ_LENGTH`]).
//!
//! `examples/reset/README.md` gives the task, the measurements and how to run
//! it.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The example's own flags (group, rotation, depth, width, train length).
pub mod cli;
/// The reset-quintic dataset, its `A₅`/`S₅` arithmetic and its families.
pub mod dataset;
/// Inference: per-family accuracy on fresh eval sets.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the reset-quintic task.
pub mod training;

/// The hand-built solutions, the obstruction, and the ceilings.
#[cfg(test)]
pub mod tests;

/// Shared example infrastructure (included by path).
#[path = "../../common/mod.rs"]
pub mod common;

use burn_mamba::prelude::MambaLatentNetConfig;

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    let cli::Cli {
        group,
        rotation,
        layers,
        train_length,
        d_model,
        expand,
        heads,
        mimo_rank,
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
        // The schedule of the ladder: a large step to leave the order-blind
        // solutions, then a small one to settle the rotation onto the exact
        // turns.
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
        println!(
            "Initializing new model config ({group:?}, {rotation:?}, {layers} layers, d_model {d_model}, expand {expand}, {heads} heads, mimo_rank {mimo_rank})"
        );
        model::model_config_with(group, rotation, layers, d_model, expand, heads, mimo_rank)
    });
    let MambaLatentNetConfig::Mamba3 { output_size, .. } = &model_config else {
        panic!("reset-quintic configures the Mamba-3 variant")
    };
    assert_eq!(
        *output_size,
        group.num_classes(),
        "the persisted model config was made for the other group — pass the matching --group"
    );
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(
            group,
            train_length,
            training_config,
            model_config.clone(),
            autodiff_device,
            app_args,
        );
    }

    if app_args.inference {
        inference::infer(group, model_config, device, app_args);
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
