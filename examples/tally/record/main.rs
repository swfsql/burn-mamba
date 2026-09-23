//! # tally-record: a running maximum, and the range that makes it a rung
//!
//! One Mamba-3 block reads twelve values and a reset. At every value, it
//! reports whether the value is a **new maximum since the last reset**.
//!
//! A maximum is `max(cₜ₋₁, vₜ)`, so one tropical register holds it exactly, and
//! the state of the plant is unused. It is also a **sum in the exponential
//! domain**: `Σ exp(S·vₛ)` at decay one, and the record test is linear there.
//! So a linear state computes it too, and at a small alphabet, a trained stock
//! block does (~100% at six values and 32 tokens).
//!
//! That route costs **range**. It needs `e^{S·(v_max − v_min)}` inside one
//! in-projection channel. Every channel is an affine read of the same
//! embedding, so more heads do not give more digits. At twelve values and 64
//! tokens, the span is `e^{45.8} ≈ 8·10¹⁹`, against the seven digits of f32,
//! and the rung separates the arms. A trained stock arm gets 90.1%, against
//! 95.4% for the register at equal budget, and 100% for the hand-built
//! register. The `edge` family (every value within one step of the maximum)
//! makes an approximately right answer stop paying.
//!
//! The *other* advantage of the register, growth (`a > 0`, a decay above one,
//! which `α = exp(Δ·A) ≤ 1` cannot be), is the subject of `tally-depth`.
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
/// The tally-record stream and its adversarial families.
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
