//! # Sequential-MNIST classifier example
//!
//! Classifies MNIST digits by reading each image as a length-784 sequence of
//! single-pixel tokens with a Mamba-3 model (a classification head on the last
//! timestep), trained with cosine-annealing LR. Inference is currently a stub.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    mnist::dataset,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The example's `model_config()`.
pub mod model;
/// Training entry point for the classifier.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the classifier.
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
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    // setup training and model configs
    let batch_size = 16;
    let training_items = 60_000;
    let iterations_per_epoch = training_items / batch_size;
    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        TrainingConfig::new(common::training::optimizer_config(dtype))
            .with_num_epochs(5)
            .with_batch_size(batch_size)
            .with_num_workers(2)
            .with_lr(Lr::CosineAnnealing(
                CosineAnnealingLr::new(4 * iterations_per_epoch) // 4 epochs
                    .with_max_lr(1e-4)
                    .with_min_lr(1e-5)
                    .with_warmup_steps(iterations_per_epoch * 5 / 100), // 5% of an epoch
            ))
    });
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config()
    });
    // save configs
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(training_config, model_config, autodiff_device, app_args);
    }

    if app_args.inference {
        let _ = device;
        println!("inference is not yet implemented");
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
