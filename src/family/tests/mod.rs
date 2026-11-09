//! Tests for the family enum, and for the block-generic `burn_stack`
//! containers as exercised through the Mamba families.
//!
//! The containers themselves live in `burn-stack` and are tested there against
//! its reference block; these suites pin the same contracts against real
//! blocks — forward/step parity, cache threading, gradient reachability across
//! a `grad_horizon` cut, class-marker placement, and the Muon allowlist.

#[cfg(feature = "mamba2")]
mod bidi;
mod block;
mod capture;
mod class;
mod config;
#[cfg(feature = "mamba3")]
mod hnet;
#[cfg(feature = "mamba3")]
mod layer;
#[cfg(feature = "mamba3")]
mod layers;
mod multi_gate;
#[cfg(feature = "optim")]
mod optim;
#[cfg(any(feature = "mamba2", feature = "mamba3"))]
mod packed;
mod reset_align;
mod untied;
