#![allow(warnings)]

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn_mamba::prelude::Mamba3BackendExt;
pub use common::{
    backend::{MainAutoBackend, MainBackend, MainDevice},
    cli::AppArgs,
    mnist,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

pub mod cli;
pub mod dataset;
pub mod model;
pub mod training;

#[path = "../common/mod.rs"]
pub mod common;

// used to load the teacher configuration
#[path = "../mnist-class/model.rs"]
pub mod teacher_model;

// used for validity check
#[path = "../mnist-class/training.rs"]
pub mod teacher_training;

pub const ITEMS: usize = 60000; // any value works

pub fn launch<AutoB>(app_args: &AppArgs, teacher_args: &cli::TeacherArgs)
where
    AutoB: AutodiffBackend + MainDevice + Mamba3BackendExt,
    <AutoB as AutodiffBackend>::InnerBackend: MainDevice + Mamba3BackendExt,
{
    assert!(
        app_args.extra_args.is_empty(),
        "Extra arguments should have been taken"
    );
    app_args.create_artifact_dir();

    // setup training and model configs
    let batch_size = 16;
    let iterations_per_epoch = ITEMS / batch_size;

    let teacher_model_config = teacher_args
        .load_model_config::<AutoB::InnerBackend, _>()
        .expect("Failed to load the teacher model");

    let training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        TrainingConfig::new(common::training::optimizer_config::<AutoB>())
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
    let model_config = app_args.load_model_config::<AutoB, _>().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config()
    });
    // save configs
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        let training_device = AutoB::main_device();
        let infer_device = AutoB::InnerBackend::main_device();
        AutoB::set_dtype(&training_device);
        // not needed for B, it is already set and initialized
        // B::set_dtype(&infer_device);
        training::train::<AutoB>(
            training_config,
            model_config,
            teacher_model_config,
            infer_device,
            training_device,
            app_args,
            teacher_args,
        );
    }

    if app_args.inference {
        let infer_device = <AutoB::InnerBackend>::main_device();
        if !app_args.training {
            <AutoB::InnerBackend>::set_dtype(&infer_device);
        }
        println!("inference is not yet implemented");
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

fn main() {
    let mut app_args = AppArgs::parse().unwrap();
    let teacher_args = cli::TeacherArgs::parse(std::mem::take(&mut app_args.extra_args)).unwrap();
    launch::<MainAutoBackend>(&app_args, &teacher_args);
}
