use crate::common::{
    model::{Mamba2NetworkConfig, ModelConfigExt},
    optim::OptimConfigExt,
};
use burn::{
    module::AutodiffModule, optim::AdamWConfig, prelude::*, tensor::backend::AutodiffBackend,
};

pub trait TrainingConfigExt<AutoB: AutodiffBackend, AutoM: AutodiffModule<AutoB>>: Config {
    type ModelConfig: ModelConfigExt<AutoB>;
    type OptimConfig: OptimConfigExt<AutoB, AutoM>;
    fn optim(&self) -> &Self::OptimConfig;
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub optimizer: AdamWConfig,
    #[config(default = 1)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 2)]
    pub num_workers: usize,
    #[config(default = 1e-4)]
    pub lr: f64,
    #[config(default = 0)]
    pub seed: u64,
}

impl<AutoB: AutodiffBackend, AutoM: AutodiffModule<AutoB>> TrainingConfigExt<AutoB, AutoM>
    for TrainingConfig
{
    type ModelConfig = Mamba2NetworkConfig;
    type OptimConfig = AdamWConfig;
    fn optim(&self) -> &Self::OptimConfig {
        &self.optimizer
    }
}

pub fn optimizer_config<AutoB: AutodiffBackend>() -> AdamWConfig {
    AdamWConfig::new()
        .with_epsilon(burn_mamba::utils::div_eps_f32::<AutoB>())
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(
            1.0,
        )))
        .with_cautious_weight_decay(true)
}
