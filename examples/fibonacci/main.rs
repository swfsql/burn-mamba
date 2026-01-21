use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
pub use common::{
    backend::{MainAutoBackend, MainBackend, MainDevice},
    cli::AppArgs,
    training::TrainingConfig,
};

pub mod dataset;
pub mod inference;
pub mod model;
pub mod training;

#[path = "../common/mod.rs"]
pub mod common;

pub fn launch<B, AutoB>(app_args: &AppArgs)
where
    B: Backend + MainDevice,
    AutoB: AutodiffBackend + MainDevice,
{
    app_args.create_artifact_dir();

    // setup training and model configs
    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        TrainingConfig::new(
            common::training::optimizer_config::<AutoB>()
                // fast training, where momentum isn't really required
                .with_beta_1(0.0)
                .with_beta_2(0.95),
        )
        .with_num_epochs(2)
        .with_batch_size(32)
        .with_num_workers(2)
        // fast training
        // note: Sgd works well with lr=1e-4
        .with_lr(3e-2)
    });
    let model_config = app_args
        .load_model_config::<AutoB, _>()
        .unwrap_or_else(|| model::model_config());
    // save configs
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        let training_device = AutoB::main_device();
        training::train::<AutoB>(
            training_config,
            model_config.clone(),
            training_device,
            app_args,
        );
    }

    if app_args.inference {
        let infer_device = B::main_device();
        let batch_size = 10;
        inference::infer::<B>(model_config, batch_size, infer_device, app_args);
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch::<MainBackend, MainAutoBackend>(&app_args);
}
