//! # tally-drift — a decay that reads how much evidence the head holds
//!
//! One Mamba-3 block reads 33 values and a gap symbol, and reports at every
//! value whether it is **above the running estimate of the level** — an
//! estimate whose past must age across gaps, because the level walks while
//! nothing is observed.
//!
//! The optimal ageing is `Λ ← Λ/(1 + k·q·Λ)`: hyperbolic in the number of gap
//! tokens and saturating in the evidence held, which is exactly what a Kalman
//! gate computes and what a projected decay (`αᵏ·Λ`, a product of the two)
//! cannot be. It is the ladder's **modest** rung, and deliberately reported as
//! such: a stock block with a second head and its per-token gate gets within a
//! few points, because a *decision* only needs the ratio cross-multiplied, and
//! a geometric discount approximates a hyperbolic one over a narrow range.
//!
//! The task, the measurements and how to run it: `examples/tally/README.md`.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

/// The example's own flags (`--stock`).
pub mod cli;
/// The tally-drift stream, its filter and its adversarial families.
pub mod dataset;
/// The example's `model_config()`.
pub mod model;

/// The hand-built solution and the projected-decay sweep.
#[cfg(test)]
pub mod tests;

/// Shared example infrastructure (included by path).
#[path = "../../common/mod.rs"]
pub mod common;
/// The `tally-*` ladder's shared dataset, loops and helpers.
#[path = "../shared/mod.rs"]
pub mod shared;

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    let kalman = !cli::Cli::parse(&app_args).stock;
    if !kalman {
        println!("ablation: Gain::Projected (the computed decay removed)");
    }
    shared::launch(&app_args, &dataset::task(), model::model_config(kalman));
}
