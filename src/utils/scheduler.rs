// copied from:
// https://github.com/huy209vn/burn-jepa/blob/588d3654fbcfdcfce2ecdb7bcaf7a2e5bd5a70ea/src/train/scheduler.rs
// slight adaptions: added Config derives and a unified enum.

//! Learning rate schedulers for controlling the optimization process.
//!
//! This module provides various strategies to adjust the learning rate during training,
//! such as cosine annealing with linear warmup, to improve model convergence and performance.

use burn::prelude::*;
use std::f64::consts::PI;

#[derive(Config, Debug)]
pub enum Lr {
    CosineAnnealing(CosineAnnealingLr),
    Constant(ConstantLr),
}

impl Lr {
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
    #[config(default = 1e-4)]
    pub lr: f64,
}

impl ConstantLr {
    pub fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_annealing_warmup() {
        let scheduler = CosineAnnealingLr::new(1.0, 0.01, 1000, 100);

        // At step 0, LR should be 0
        assert_eq!(scheduler.get_lr(0), 0.0);

        // At half warmup, LR should be max_lr / 2
        assert_eq!(scheduler.get_lr(50), 0.5);

        // At end of warmup, LR should be max_lr
        assert_eq!(scheduler.get_lr(100), 1.0);

        // After total steps, LR should be min_lr
        assert_eq!(scheduler.get_lr(1000), 0.01);
    }

    #[test]
    fn test_constant_lr() {
        let scheduler = ConstantLr::new(0.001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(1000), 0.001);
        assert_eq!(scheduler.get_lr(10000), 0.001);
    }
}
