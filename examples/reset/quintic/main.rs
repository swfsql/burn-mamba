//! # Reset-quintic — `A₅` in one Mamba-3 block, `S₅` in two
//!
//! `reset-swap`'s stream over **five** items. The model reads two turns and a
//! reset and reports, at every position, how `abcde` are arranged: the running
//! word in `A₅` (`--group a5`: `d = (0 1)(2 3)`, `t = (0 2 4)`, sixty classes) or
//! in `S₅` (`--group s5`, the default: `s = (0 1)`, `c = (0 1 2 3 4)`, a hundred
//! and twenty).
//!
//! `A₅` is the icosahedron's rotation group — the smallest non-solvable group,
//! the one behind the unsolvable quintic — and it is `reset-swap`'s block that
//! holds it: a conjugating ([`RotationKind::Rotor4D`]) 4-block is `SO(3)`, and
//! `d`/`t` are a half-turn and a third-turn about axes `20.9°` apart. `S₅` is
//! not a rotation group of anything the block turns: its transpositions act on
//! `A₅` by the automorphism that swaps the `72°` and `144°` rotations, and only a
//! reflection does that. No single layer, at any size, of any rotation kind,
//! tracks `S₅` exactly beyond its sign; two layers do — the first holds the sign,
//! the second the even part, turned by a step that reads both.
//!
//! The trained width ([`model::DEFAULT_WIDTH`]) is wider than the hand-built
//! floor ([`model::floor_width`]) because training lands the group on one head at
//! a time, and MIMO ranks give that head the readouts the floor spreads over heads.
//!
//! Downstream flags, after the trailing `--`:
//!
//! - `--group a5|s5` — not persisted, so pass it on every run;
//! - `--rotation complex|quaternion|rotor` (default `rotor`), `--layers N`
//!   (default 1 for `a5`, 2 for `s5`), and the width `--d-model N --heads N
//!   --mimo-rank N --expand N` (default `4 2 2 1`, [`model::DEFAULT_WIDTH`]) —
//!   baked into a **fresh** model config (a persisted one wins on reload);
//! - `--train-length N` (default [`dataset::SEQ_LENGTH`]).
//!
//! The task, the measurements and how to run it: `examples/reset/README.md`.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

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

use burn_mamba::prelude::{MambaLatentNetConfig, RotationKind};
use dataset::Group;
use std::ffi::OsString;

/// Wire up the device, configs, and the train/infer flow for the task.
pub fn launch(app_args: &AppArgs) {
    let group = parse_group(&app_args.extra_args);
    let rotation = parse_rotation(&app_args.extra_args);
    let layers = parse_layers(&app_args.extra_args, group);
    let train_length = parse_usize(&app_args.extra_args, "--train-length", dataset::SEQ_LENGTH);
    let (default_d_model, default_heads, default_mimo_rank) = model::DEFAULT_WIDTH;
    let d_model = parse_usize(&app_args.extra_args, "--d-model", default_d_model);
    let expand = parse_usize(&app_args.extra_args, "--expand", 1);
    let heads = parse_usize(&app_args.extra_args, "--heads", default_heads);
    let mimo_rank = parse_usize(&app_args.extra_args, "--mimo-rank", default_mimo_rank);
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
    let mut training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // The ladder's schedule: a large step to leave the order-blind solutions,
        // a small one to settle the rotation onto the exact turns.
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
    app_args.override_training_config(&mut training_config);
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

/// The value following `flag` among the downstream arguments.
fn flag_value(extra_args: &[OsString], flag: &str) -> Option<String> {
    extra_args
        .iter()
        .position(|a| a == flag)
        .and_then(|i| extra_args.get(i + 1))
        .map(|v| v.to_string_lossy().into_owned())
}

/// `--group a5|s5`, defaulting to `S₅`, the one that needs the second layer.
fn parse_group(extra_args: &[OsString]) -> Group {
    match flag_value(extra_args, "--group").as_deref() {
        Some("s5") | None => Group::Symmetric,
        Some("a5") => Group::Alternating,
        Some(other) => panic!("--group must be 'a5' or 's5', got {other:?}"),
    }
}

/// `--rotation complex|quaternion|rotor`, defaulting to the full `SO(4)`.
fn parse_rotation(extra_args: &[OsString]) -> RotationKind {
    match flag_value(extra_args, "--rotation").as_deref() {
        Some("rotor") | Some("so4") | None => RotationKind::Rotor4D,
        Some("quaternion") | Some("quat") => RotationKind::Quaternion4D,
        Some("complex") => RotationKind::Complex2D,
        Some(other) => {
            panic!("--rotation must be 'complex', 'quaternion' or 'rotor', got {other:?}")
        }
    }
}

/// `--layers N`, defaulting to the fewest that hold the group.
fn parse_layers(extra_args: &[OsString], group: Group) -> usize {
    parse_usize(extra_args, "--layers", model::default_layers(group))
}

/// One `--flag N` argument out of the downstream arguments.
fn parse_usize(extra_args: &[OsString], flag: &str, default: usize) -> usize {
    match flag_value(extra_args, flag) {
        None => default,
        Some(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("{flag} takes a positive integer, got {v:?}")),
    }
}

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    launch(&app_args);
}
