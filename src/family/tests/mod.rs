//! Tests for the family enum, and for the block-generic `burn_stack`
//! containers as exercised through the Mamba families.
//!
//! The containers are in `burn-stack`, and the tests of burn-stack run them on
//! its reference block. These files check the same properties on real blocks:
//! forward/step parity, cache threading, class-marker placement and the Muon
//! plan.
//!
//! `block_contract` runs the block contracts of burn-stack on each block
//! setting of this crate.

mod block;
mod block_contract;
mod capture;
mod class;
mod config;
#[cfg(feature = "mamba3")]
mod hnet;
#[cfg(feature = "optim")]
mod optim;
#[cfg(any(feature = "mamba2", feature = "mamba3"))]
mod packed;
mod reset_align;
mod untied;
