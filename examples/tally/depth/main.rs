//! # tally-depth: the smallest task that needs a *tropical register*
//!
//! One Mamba-3 block reads `(` / `)` / `R`. At every `)`, it reports whether
//! the bracket depth is **still positive**. A `)` at depth 0 closes nothing:
//! the depth does not go negative.
//!
//! That floor is the whole rung. A linear recurrence computes the *unclamped*
//! sum, and the `floor` family misleads it at a third of the scored positions.
//! The clamp is `max(cₜ₋₁ + aₜ, 0)`. It is linear in the (max, +) semiring,
//! and one tropical register computes it exactly. The plant does nothing here.
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
/// The tally-depth stream and its adversarial families.
pub mod dataset;
/// The example's `model_config()`.
pub mod model;

/// The hand-built solution and the stock-gate sweep.
#[cfg(test)]
pub mod tests;

/// Shared example infrastructure (included by path).
#[path = "../../common/mod.rs"]
pub mod common;
/// The shared dataset, loops and helpers of the `tally-*` ladder.
#[path = "../shared/mod.rs"]
pub mod shared;

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    let tropical = !cli::Cli::parse(&app_args).stock;
    if !tropical {
        println!("ablation: Tropical::None (the register removed)");
    }
    shared::launch(&app_args, &dataset::task(), model::model_config(tropical));
}
