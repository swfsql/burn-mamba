//! Shared training configuration for the examples.
//!
//! [`TrainingConfig`] holds the common hyperparameters (epochs, batch size, LR
//! schedule, seed); [`TrainingConfigExt`] ties a config to its optimizer config
//! so the generic loop stays model-agnostic.  [`optimizer_config`] builds the
//! AdamW defaults shared by the examples (per-dtype epsilon, grad clipping,
//! cautious weight decay).

use crate::common::{model::ModelConfigExt, optim::OptimConfigExt};
use burn::{
    module::AutodiffModule, optim::AdamWConfig, prelude::*, tensor::backend::AutodiffBackend,
};
pub use burn_mamba::utils::scheduler::{ConstantLr, CosineAnnealingLr, Lr};

/// A training config that can hand the loop its optimizer config.
pub trait TrainingConfigExt<AutoB, AutoM, ModelConfig>
where
    Self: Config,
    AutoB: AutodiffBackend,
    AutoM: AutodiffModule<AutoB>,
    ModelConfig: ModelConfigExt<AutoB, Model = AutoM>,
{
    /// The optimizer config type.
    type OptimConfig: OptimConfigExt<AutoB, AutoM>;
    /// Borrow the optimizer config.
    fn optim(&self) -> &Self::OptimConfig;
}

/// Common training hyperparameters shared by the examples.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// The optimizer configuration (AdamW).
    pub optimizer: AdamWConfig,
    /// Number of training epochs.
    #[config(default = 1)]
    pub num_epochs: usize,
    /// Mini-batch size.
    #[config(default = 32)]
    pub batch_size: usize,
    /// Number of dataloader worker threads.
    #[config(default = 2)]
    pub num_workers: usize,
    /// Learning-rate schedule.
    #[config(default = "Lr::Constant(ConstantLr::new())")]
    pub lr: Lr,
    /// RNG seed for reproducibility.
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

/// The AdamW defaults shared by the examples: per-dtype epsilon, gradient
/// clipping at 1.0, and cautious weight decay.
pub fn optimizer_config<AutoB: AutodiffBackend>() -> AdamWConfig {
    AdamWConfig::new()
        .with_epsilon(burn_mamba::utils::div_eps_f32::<AutoB>())
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(
            1.0,
        )))
        .with_cautious_weight_decay(true)
}
