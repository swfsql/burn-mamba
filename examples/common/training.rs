use crate::common::model::Mamba2NetworkConfig;
use burn::{optim::AdamWConfig, prelude::*, tensor::backend::AutodiffBackend};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: Mamba2NetworkConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 1)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 1e-4)]
    pub lr: f64,
}

// Create the directory to save the model and model config
pub fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn optimizer_config<AutoB: AutodiffBackend>() -> AdamWConfig {
    AdamWConfig::new()
        .with_epsilon(burn_mamba::utils::div_eps_f32::<AutoB>())
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(
            1.0,
        )))
        .with_cautious_weight_decay(true)
}
