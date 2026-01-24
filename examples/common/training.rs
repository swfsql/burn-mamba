use crate::common::{model::ModelConfigExt, optim::OptimConfigExt};
use burn::{
    module::AutodiffModule, optim::AdamWConfig, prelude::*, tensor::backend::AutodiffBackend,
};

pub trait TrainingConfigExt<AutoB, AutoM, ModelConfig>
where
    Self: Config,
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
    ModelConfig: ModelConfigExt<AutoB, Model = AutoM>,
{
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

impl<AutoB, AutoM, ModelConfig> TrainingConfigExt<AutoB, AutoM, ModelConfig> for TrainingConfig
where
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
    ModelConfig: ModelConfigExt<AutoB, Model = AutoM>,
{
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
