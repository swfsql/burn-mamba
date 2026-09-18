//! # tally-record — a running maximum, and the range that makes it a rung
//!
//! One Mamba-3 block reads twelve values and a reset, and reports at every value
//! whether it is a **new maximum since the last reset**.
//!
//! A maximum is `max(cₜ₋₁, vₜ)`, so one tropical register holds it exactly with
//! the plant's state unused. It is also a **sum in the exponential domain** —
//! `Σ exp(S·vₛ)` at decay one, with the record test linear there — so a linear
//! state computes it too, and at a small alphabet a trained stock block does
//! (~100% at six values and 32 tokens, the first version of this rung).
//!
//! What that route costs is **range**: it needs `e^{S·(v_max − v_min)}` inside
//! one in-projection channel, and every channel is an affine read of the same
//! embedding, so more heads do not buy more digits. At twelve values and 64
//! tokens the span is `e^{45.8} ≈ 8·10¹⁹` against f32's seven digits, and the
//! rung separates: 90.1% for a trained stock arm against 95.4% for the register
//! at equal budget, and 100% for the register hand-built. The `edge` family —
//! every value within one step of the maximum — is what makes being
//! approximately right stop paying.
//!
//! The register's *other* advantage, growth (`a > 0`, a decay above one, which
//! `α = exp(Δ·A) ≤ 1` cannot be), is `tally-depth`.
//!
//! The task, the measurements and how to run it: `examples/tally/README.md`.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

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
/// The `tally-*` ladder's shared dataset, loops and helpers.
#[path = "../shared/mod.rs"]
pub mod shared;

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    let tropical = !app_args.extra_args.iter().any(|a| a == "--stock");
    if !tropical {
        println!("ablation: Tropical::None (the register removed)");
    }
    shared::launch(&app_args, &dataset::task(), model::model_config(tropical));
}
