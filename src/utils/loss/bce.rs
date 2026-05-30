//! Binary cross-entropy loss.
//!
//! When `logits = true` the loss is computed in a numerically stable way from
//! raw logits via [`log_sigmoid`]; otherwise the inputs are treated as
//! probabilities and the logs are clamped to avoid `−∞`.

use crate::utils::log_sigmoid::log_sigmoid;
use burn::backend::Backend;
use burn::module::Module;
use burn::prelude::*;

/// Configuration to create a [`BinaryCrossEntropyLoss`] using the [`BinaryCrossEntropyLossConfig::init`].
#[derive(Config, Debug)]
pub struct BinaryCrossEntropyLossConfig {
    /// Treat the inputs as logits, applying a sigmoid activation when computing the loss.
    #[config(default = false)]
    pub logits: bool,
}

impl BinaryCrossEntropyLossConfig {
    /// Initialize [`BinaryCrossEntropyLoss`].
    pub fn init(&self) -> BinaryCrossEntropyLoss {
        BinaryCrossEntropyLoss {
            logits: self.logits,
        }
    }
}

/// Calculate the binary cross entropy loss from the input logits and the targets.
///
/// Should be created using [BinaryCrossEntropyLossConfig]
#[derive(Module, Debug)]
pub struct BinaryCrossEntropyLoss {
    /// Treat the inputs as logits
    pub logits: bool,
}

impl BinaryCrossEntropyLoss {
    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// Binary:
    /// - logits: `[batch_size]`
    /// - targets: `[batch_size]`
    ///
    /// Multi-label:
    /// - logits: `[batch_size, num_classes]`
    /// - targets: `[batch_size, num_classes]`
    pub fn forward<const D: usize>(
        &self,
        logits: Tensor<D>,
        targets: Tensor<D>,
    ) -> Tensor<1> {
        let loss = if self.logits {
            // Numerically stable by combining `log(sigmoid(x))` with `log_sigmoid(x)`
            (targets.neg() + 1.) * logits.clone() - log_sigmoid(logits)
        } else {
            // - (target * log(input) + (1 - target) * log(1 - input))
            // https://github.com/tracel-ai/burn/issues/2739: clamp at -100.0 to avoid undefined values
            (targets.clone() - 1) * logits.clone().neg().log1p().clamp_min(-100.0)
                - targets * logits.log().clamp_min(-100.0)
        };

        loss.mean()
    }
}
