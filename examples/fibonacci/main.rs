pub mod dataset;
pub mod inference;
pub mod model;
pub mod training;

#[path = "../common/mod.rs"]
pub mod common;

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use common::backend::{MainAutoBackend, MainBackend, MainDevice};

pub fn launch<B: Backend + MainDevice, AutoB: AutodiffBackend + MainDevice>() {
    let path = "/tmp/burn-mamba/fibonacci";

    // training
    let train_device = AutoB::main_device();
    let mamba_config = common::training::TrainingConfig::new(
        model::model_config(),
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
    .with_lr(1e-2);
    training::train::<AutoB>(path, mamba_config, train_device);

    // inference
    let infer_device = B::main_device();
    inference::infer::<B>(path, infer_device);
}

fn main() {
    launch::<MainBackend, MainAutoBackend>();
}
