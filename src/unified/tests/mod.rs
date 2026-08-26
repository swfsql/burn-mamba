//! Tests for the unified enums, and for the block-generic `burn_stack`
//! containers as exercised through the Mamba families.
//!
//! The containers themselves live in `burn-stack` and are tested there against
//! its reference block; these suites pin the same contracts against real
//! blocks — forward/step parity, cache threading, gradient reachability across
//! a `grad_horizon` cut, class-marker placement, and the Muon allowlist.

mod bidi;
mod class;
mod layer;
#[cfg(feature = "mamba3")]
mod layers;
mod multi_gate;
#[cfg(feature = "optim")]
mod optim;
