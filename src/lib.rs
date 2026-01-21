#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

#[cfg(feature = "mamba1")]
pub mod mamba1;
#[cfg(feature = "mamba1")]
pub mod mamba1_block;
#[cfg(feature = "mamba2")]
pub mod mamba2;
#[cfg(feature = "mamba2")]
pub mod mamba2_block;
pub mod schedule;

pub mod prelude {
    #[cfg(feature = "mamba1")]
    mod mamba1_export {
        pub use crate::mamba1::*;
        pub use crate::mamba1_block::*;
    }
    #[cfg(feature = "mamba1")]
    pub use mamba1_export::*;

    #[cfg(feature = "mamba2")]
    mod mamba2_export {
        pub use crate::mamba2::*;
        pub use crate::mamba2_block::*;
    }
    #[cfg(feature = "mamba2")]
    pub use mamba2_export::*;
}

pub mod utils;
