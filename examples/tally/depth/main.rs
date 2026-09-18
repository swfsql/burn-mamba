//! # tally-depth — the smallest task a *tropical register* is needed for
//!
//! One Mamba-3 block reads `(` / `)` / `R` and reports, at every `)`, whether
//! the bracket depth is **still positive** — where a `)` at depth 0 closes
//! nothing rather than going negative.
//!
//! That floor is the whole rung. A linear recurrence computes the *unclamped*
//! sum, which the `floor` family misleads at a third of the scored positions;
//! the clamp is `max(cₜ₋₁ + aₜ, 0)`, linear in the (max, +) semiring and exact
//! for one tropical register, whose plant here does nothing at all.
//!
//! The task, the measurements and how to run it: `examples/tally/README.md`.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

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
/// The `tally-*` ladder's shared dataset, loops and helpers.
#[path = "../shared/mod.rs"]
pub mod shared;

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    // The only downstream argument: `--stock` removes the tropical register
    // from a *fresh* model config (a persisted one wins on reload), which is the
    // ablation this rung is about.
    let tropical = !app_args.extra_args.iter().any(|a| a == "--stock");
    if !tropical {
        println!("ablation: Tropical::None (the register removed)");
    }
    shared::launch(&app_args, &dataset::task(), model::model_config(tropical));
}
