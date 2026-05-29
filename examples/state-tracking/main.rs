//! # State-tracking example — abelian RoPE vs. quaternion rotation
//!
//! A demo of the capability *motivating* the quaternion
//! (`RotationKind::Quaternion4D`) rotation: tracking composition in the
//! **non-solvable** group `A₅` (the alternating group on 5 letters, the rotation
//! group of the icosahedron).
//!
//! The model reads a leading reference token then a sequence of `A₅` generators
//! (one-hot) and must output, at **every position**, the cumulative product so
//! far (a 60-way classification). By Barrington's theorem this word problem is
//! `NC¹`-complete; a single-layer linear SSM with **abelian** (`SO(2)`/complex
//! RoPE) state transitions is confined to the solvable/`TC⁰` regime and cannot
//! *compose* it, whereas the **non-abelian** `SU(2)` quaternion rotation can
//! represent the icosahedral group `2I = SL(2,5)` (a double cover of `A₅`).
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
//! Watch the **per-position accuracy** printed at each validation pass (chance is
//! `1/60 ≈ 1.7%`). Both rotations solve the shallow positions by *memorising*
//! short prefixes; the abelian/non-abelian distinction only matters in the deep
//! positions, where genuine `A₅` composition is required.
//!
//! **What to expect (read the README first).** At this deliberately **tiny**
//! single-layer, fibonacci-scale model the gap is *modest*: both rotations track
//! to a similar depth, and the quaternion shows only a small edge in the deepest
//! positions, and only after extended training (resume with `--artifacts-path` to
//! train further; a constant LR makes resuming seamless). A clean, dramatic gap
//! requires more model capacity / scale than this demo uses — see the Mamba-3
//! paper. This example is a faithful, runnable *harness* for the comparison, not
//! a tuned benchmark.

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
    let num_epochs = 120;
    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        // "Grokking" recipe: leaving the memorisation basin for the compositional
        // solution needs real regularisation (weight decay) and many epochs. A
        // *constant* LR is used on purpose so resuming (`--artifacts-path`) is a
        // seamless continuation — a decaying schedule would restart from its peak
        // each run (the iteration counter resets), perturbing the slow transition.
        TrainingConfig::new(common::training::optimizer_config(dtype).with_weight_decay(0.1))
            .with_num_epochs(num_epochs)
            .with_batch_size(batch_size)
            .with_num_workers(2)
            .with_lr(Lr::Constant(ConstantLr::new().with_lr(1e-3)))
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
