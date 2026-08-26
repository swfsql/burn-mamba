//! Shared training configuration for the examples.
//!
//! [`TrainingConfig`] holds the common hyperparameters (epochs, batch size, LR
//! schedule, seed) plus the [`OptimizerConfig`].  [`optimizer_config`] builds the
//! AdamW defaults shared by the examples (epsilon, grad clipping, cautious
//! weight decay); [`OptimizerConfig::muon`] optionally moves the hidden weight
//! matrices to Muon (see `burn_stack::optim`).

use burn::{
    optim::{AdamWConfig, ModuleOptimizer, MuonConfig},
    prelude::*,
    train::metric::NumericEntry,
};
use burn_stack::optim::MuonPlan;
pub use burn_stack::utils::scheduler::{ConstantLr, CosineAnnealingLr, Lr};

/// Current value of a metric reading, or `NaN` when the metric has none yet
/// (`Numeric::value` / `running_value` are `None` for metrics that only produce
/// a value at epoch end).
pub fn metric_current(entry: Option<NumericEntry>) -> f64 {
    entry.map_or(f64::NAN, |entry| entry.current())
}

/// How the examples optimize: AdamW everywhere, optionally with Muon on the
/// hidden weight matrices.
///
/// `muon = None` is the plain-AdamW baseline. When set, the model config's
/// [`MuonPlan`] decides which weights move over (and where the fused projections
/// split) — everything else, 1-D and 3-D tensors included, keeps AdamW.
#[derive(Config, Debug)]
pub struct OptimizerConfig {
    /// AdamW: the fallback optimizer, and the one used for every parameter when
    /// `muon` is `None`.
    pub adamw: AdamWConfig,
    /// Muon for the planned hidden matrices. `None` ⇒ AdamW-only.
    pub muon: Option<MuonConfig>,
}

impl OptimizerConfig {
    /// AdamW-only (the baseline).
    pub fn adamw_only(dtype: burn::tensor::DType) -> Self {
        Self::new(optimizer_config(dtype))
    }

    /// AdamW + Muon on `plan`'s weights, sharing AdamW's weight decay and LR
    /// (see [`burn_stack::optim::muon_config`]).
    pub fn with_muon_defaults(self, weight_decay: f32) -> Self {
        self.with_muon(Some(burn_stack::optim::muon_config(weight_decay)))
    }

    /// Build the module optimizer for a model whose Muon plan is `plan`.
    pub fn init(&self, plan: &MuonPlan) -> ModuleOptimizer {
        match &self.muon {
            None => self.adamw.init(),
            Some(muon) => plan.build(&self.adamw, muon),
        }
    }
}

/// Common training hyperparameters shared by the examples.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// The optimizer configuration.
    pub optimizer: OptimizerConfig,
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
        .with_epsilon(burn_stack::utils::div_eps(dtype))
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Value(
            1.0,
        )))
        .with_cautious_weight_decay(true)
}
