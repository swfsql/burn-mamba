#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

#[cfg(feature = "mamba1")]
pub mod mamba1;
#[cfg(feature = "mamba2")]
pub mod mamba2;
pub mod schedule;

pub mod prelude {
    #[cfg(feature = "mamba1")]
    pub use crate::mamba1::*;

    #[cfg(feature = "mamba2")]
    pub use crate::mamba2::*;
}

pub mod utils;
