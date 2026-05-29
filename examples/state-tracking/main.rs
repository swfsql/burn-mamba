//! # State-tracking example — abelian RoPE vs. quaternion rotation
//!
//! A demo of the capability gap motivating the quaternion
//! (`RotationKind::Quaternion4D`) rotation: tracking composition in the
//! **non-solvable** group `A₅` (the alternating group on 5 letters, the rotation
//! group of the icosahedron).
//!
//! The model reads a sequence of `A₅` generators (one-hot) and must output, at
//! **every position**, the cumulative product so far (a 60-way classification).
//! By Barrington's theorem this word problem is `NC¹`-complete; a single-layer
//! linear SSM with **abelian** (`SO(2)`/complex RoPE) state transitions is
//! confined to the solvable/`TC⁰` regime and cannot track it, whereas the
//! **non-abelian** `SU(2)` quaternion rotation can represent the icosahedral
//! group `2I = SL(2,5)` (a double cover of `A₅`).
//!
//! Unlike the other examples this one carries a downstream flag,
//! `--rotation complex|quaternion` (default `complex`), forwarded after the
//! trailing `--`; it selects the rotation baked into the model config.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example state-tracking --features backend-flex -- --training --inference -- --rotation complex
//! cargo run --release --example state-tracking --features backend-flex -- --training --inference -- --rotation quaternion
//! ```
//!
//! Compare the two final per-token accuracies: chance is `1/60 ≈ 1.7%`. The
//! quaternion run climbs above the complex run, which tends to plateau. The
//! defaults are deliberately **tiny** (fibonacci-scale) so the demo runs
//! quickly on CPU; for a wider, cleaner gap, scale the model / sequence length /
//! `num_epochs` up and run on GPU (`--features backend-cuda`).

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{ConstantLr, Lr, TrainingConfig},
};

/// The `A₅` state-tracking dataset.
pub mod dataset;
/// Inference: per-token accuracy on a fresh eval set.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the state-tracking task.
pub mod training;

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

    // setup training and model configs
    let batch_size = 64;
    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        TrainingConfig::new(common::training::optimizer_config(dtype))
            .with_num_epochs(30)
            .with_batch_size(batch_size)
            .with_num_workers(2)
            .with_lr(Lr::Constant(ConstantLr::new().with_lr(3e-3)))
    });
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config (rotation={rotation:?})");
        model::model_config(rotation)
    });
    // save configs
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

/// Parse `--rotation complex|quaternion` from the forwarded `extra_args`
/// (defaults to `Complex2D` when absent).
fn parse_rotation(extra_args: &[OsString]) -> RotationKind {
    let value = extra_args
        .iter()
        .position(|a| a == "--rotation")
        .and_then(|i| extra_args.get(i + 1))
        .map(|v| v.to_string_lossy().into_owned());
    match value.as_deref() {
        Some("quaternion") | Some("quat") => RotationKind::Quaternion4D,
        Some("complex") | None => RotationKind::Complex2D,
        Some(other) => panic!("--rotation must be 'complex' or 'quaternion', got {other:?}"),
    }
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
