//! Shared training configuration for the examples.
//!
//! [`TrainingConfig`] holds the common hyperparameters (epochs, batch size, LR
//! schedule, seed) plus the optimizer config.  [`optimizer_config`] builds the
//! AdamW defaults shared by the examples (epsilon, grad clipping, cautious
//! weight decay).

use burn::{optim::AdamWConfig, prelude::*, train::metric::NumericEntry};
pub use burn_mamba::utils::scheduler::{ConstantLr, CosineAnnealingLr, Lr};

/// Current value of a metric reading, or `NaN` when the metric has none yet
/// (`Numeric::value` / `running_value` are `None` for metrics that only produce
/// a value at epoch end).
pub fn metric_current(entry: Option<NumericEntry>) -> f64 {
    entry.map_or(f64::NAN, |entry| entry.current())
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

/// The AdamW defaults shared by the examples: per-dtype epsilon, gradient
/// clipping at 1.0, and cautious weight decay. `dtype` should be the device's
/// default float dtype (epsilon is sized to it).
pub fn optimizer_config(dtype: burn::tensor::DType) -> AdamWConfig {
    AdamWConfig::new()
        .with_epsilon(burn_mamba::utils::div_eps(dtype))
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(
            1.0,
        )))
        .with_cautious_weight_decay(true)
}
