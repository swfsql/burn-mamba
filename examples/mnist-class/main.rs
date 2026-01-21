use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
pub use common::{
    backend::{MainAutoBackend, MainBackend, MainDevice},
    cli::AppArgs,
    mnist::dataset,
    training::TrainingConfig,
};

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
        TrainingConfig::new(common::training::optimizer_config::<AutoB>())
            .with_num_epochs(5)
            .with_batch_size(16)
            .with_num_workers(2)
            .with_lr(1e-4)
    });
    let model_config = app_args
        .load_model_config::<AutoB, _>()
        .unwrap_or_else(|| model::model_config());
    // save configs
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        let training_device = AutoB::main_device();
        training::train::<AutoB>(training_config, model_config, training_device, app_args);
    }

    if app_args.inference {
        let _infer_device = B::main_device();
        println!("inference is not yet implemented");
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
