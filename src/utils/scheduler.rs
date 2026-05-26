// copied from:
// https://github.com/huy209vn/burn-jepa/blob/588d3654fbcfdcfce2ecdb7bcaf7a2e5bd5a70ea/src/train/scheduler.rs
// slight adaptions: added Config derives and a unified enum.

//! Learning rate schedulers for controlling the optimization process.
//!
//! This module provides various strategies to adjust the learning rate during training,
//! such as cosine annealing with linear warmup, to improve model convergence and performance.

use burn::prelude::*;
use std::f64::consts::PI;

/// A learning-rate schedule, dispatching to one of the concrete schedulers.
#[derive(Config, Debug)]
pub enum Lr {
    /// Cosine annealing with linear warmup. See [`CosineAnnealingLr`].
    CosineAnnealing(CosineAnnealingLr),
    /// Fixed learning rate for all steps. See [`ConstantLr`].
    Constant(ConstantLr),
}

impl Lr {
    /// Learning rate for the given (0-indexed) training `step`.
    pub fn get_lr(&self, step: usize) -> f64 {
        match self {
            Lr::CosineAnnealing(inner) => inner.get_lr(step),
            Lr::Constant(inner) => inner.get_lr(step),
        }
    }
}

/// # Cosine Annealing Learning Rate Scheduler with Linear Warmup.
///
/// This scheduler:
/// 1. Linearly increases LR from 0 to `max_lr` during warmup phase
/// 2. Applies cosine annealing from `max_lr` to `min_lr` after warmup
///
/// This is a common pattern in modern deep learning training.
#[derive(Config, Debug)]
pub struct CosineAnnealingLr {
    /// The maximum learning rate (reached after warmup)
    #[config(default = 1e-4)]
    pub max_lr: f64,
    /// The minimum learning rate (reached at end of training)
    #[config(default = 1e-6)]
    pub min_lr: f64,
    /// The total number of training steps
    pub total_steps: usize,
    /// The number of warmup steps
    #[config(default = 0)]
    pub warmup_steps: usize,
}

impl CosineAnnealingLr {
    /// Get the learning rate for the current training step.
    ///
    /// # Arguments
    /// * `step` - Current training step (0-indexed)
    ///
    /// # Returns
    /// * Learning rate for this step
    pub fn get_lr(&self, step: usize) -> f64 {
        // Warmup phase: linear increase from 0 to max_lr
        if step < self.warmup_steps {
            return self.max_lr * (step as f64) / (self.warmup_steps as f64);
        }

        // After total_steps, return min_lr
        if step >= self.total_steps {
            return self.min_lr;
        }

        // Cosine annealing phase
        let progress =
            (step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
        self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
    }
}

/// # Constant Learning Rate Scheduler.
///
/// Simply returns a fixed learning rate for all steps.
/// Useful for simple experiments or when learning rate scheduling is not needed.
#[derive(Config, Debug)]
pub struct ConstantLr {
    /// The fixed learning rate returned for every step.
    #[config(default = 1e-4)]
    pub lr: f64,
}

impl ConstantLr {
    /// Returns the fixed learning rate (independent of `step`).
    pub fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests;
