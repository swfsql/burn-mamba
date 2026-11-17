//! Tests of the Mamba blocks through the `Block` trait of burn-stack, and of
//! what only Mamba needs.
//!
//! burn-stack tests its containers with its reference block. Here,
//! `block_contract` runs the block contracts of burn-stack on each block
//! setting of this crate. The other files test what only Mamba needs:
//!
//! - `block.rs`: the family enum [`MambaBlock`](crate::family::MambaBlock),
//! - `config.rs`: the `{ shape, block }` configs,
//! - `reset_align.rs`: the reset alignment of each family,
//! - `capture.rs`: the cache write-back of Mamba-3,
//! - `untied.rs`: the untied `in_proj` tail of Mamba-3, and the untie checks
//!   of its Kalman gain and of its tropical register,
//! - `optim.rs`: the Muon specs of each family.

use crate::prelude::*;

mod block;
mod block_contract;
#[cfg(feature = "mamba3")]
mod capture;
mod config;
#[cfg(feature = "optim")]
mod optim;
mod reset_align;
#[cfg(feature = "mamba3")]
mod untied;

/// The width of the shared configs below.
const D_MODEL: usize = 16;

/// The base setting of Mamba-1.
#[cfg(feature = "mamba1")]
fn m1() -> Mamba1Config {
    Mamba1Config::new(D_MODEL).with_state_rank(8)
}

/// The base setting of Mamba-2, at the width `d_model`, without its scan
/// options.
#[cfg(feature = "mamba2")]
fn m2_of_width(d_model: usize) -> Mamba2Config {
    Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4)
}

/// The base setting of Mamba-2, at the width [`D_MODEL`].
#[cfg(feature = "mamba2")]
fn m2() -> Mamba2Config {
    m2_of_width(D_MODEL)
}

/// The base setting of Mamba-3, without its scan options.
#[cfg(feature = "mamba3")]
fn m3() -> Mamba3Config {
    Mamba3Config::new(D_MODEL)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rotation_fraction(0.5)
}
