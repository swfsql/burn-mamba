use burn::module::Module;
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};

/// Configuration to create a [`CrossEntropyLoss`] using the [`CrossEntropyLossConfig::init`].
#[derive(Config, Debug)]
pub struct CrossEntropyLossConfig {
    /// Treat the outputs as logits, applying log-softmax when computing the loss.
    ///
    /// When `false`, outputs are assumed to be probabilities (e.g. post-softmax).
    #[config(default = true)]
    pub output_logits: bool,

    /// Treat the targets as logits, applying softmax to normalize them before computing the loss.
    ///
    /// When `false`, targets are assumed to already be a valid probability distribution
    /// (e.g. one-hot or soft labels that sum to 1).
    #[config(default = false)]
    pub target_logits: bool,
}

impl CrossEntropyLossConfig {
    /// Initialize [`CrossEntropyLoss`].
    pub fn init(&self) -> CrossEntropyLoss {
        CrossEntropyLoss {
            output_logits: self.output_logits,
            target_logits: self.target_logits,
        }
    }
}

/// Calculate the cross-entropy loss from the output logits and the targets.
///
/// Unlike the full [`burn::nn::loss::CrossEntropyLoss`], this variant accepts
/// floating-point targets (e.g. one-hot, soft label distributions, or un-normalized logits)
/// rather than integer class indices, and omits padding, per-class weights, and label smoothing.
///
/// Should be created using [CrossEntropyLossConfig].
#[derive(Module, Clone, Debug)]
pub struct CrossEntropyLoss {
    /// Treat the outputs as logits.
    pub output_logits: bool,
    /// Treat the targets as logits.
    pub target_logits: bool,
}

impl CrossEntropyLoss {
    /// Compute the criterion on the output tensor.
    ///
    /// # Shapes
    ///
    /// - logits: `[batch_size, num_classes]`
    /// - targets: `[batch_size, num_classes]`
    pub fn forward<B: Backend>(&self, logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        let log_probs = if self.output_logits {
            // Numerically stable via log-softmax
            log_softmax(logits, 1)
        } else {
            // outputs are probabilities; clamp at -100.0 after log to avoid undefined values
            // for zero-probability classes (mirrors the BCE treatment of log(0))
            logits.log().clamp_min(-100.0)
        };

        let targets = if self.target_logits {
            softmax(targets, 1)
        } else {
            targets
        };

        (targets * log_probs).sum_dim(1).mean().neg()
    }
}
