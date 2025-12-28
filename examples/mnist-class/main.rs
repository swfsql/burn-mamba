use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

pub mod model;
pub mod training;

#[path = "../common/mod.rs"]
pub mod common;

use common::backend::{MainAutoBackend, MainBackend, MainDevice};
pub use common::mnist::dataset;

pub fn launch<B: Backend + MainDevice, AutoB: AutodiffBackend + MainDevice>() {
    let path = "/tmp/burn-mamba/mnist-class";

    // training
    let train_device = AutoB::main_device();
    let mamba_config = common::training::TrainingConfig::new(
        model::model_config(),
        common::training::optimizer_config::<AutoB>(),
    )
    .with_num_epochs(1)
    .with_batch_size(16)
    .with_num_workers(2)
    .with_lr(1e-4);
    training::train::<AutoB>(path, mamba_config, train_device);

    // no inference is used
    let _infer_device = B::main_device();
}

fn main() {
    launch::<MainBackend, MainAutoBackend>();
}
