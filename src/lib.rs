#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

#[cfg(feature = "mamba1")]
pub mod mamba1;
#[cfg(feature = "mamba2")]
pub mod mamba2;
#[cfg(feature = "mamba3")]
pub mod mamba3;
pub mod schedule;

pub mod prelude {
    #[cfg(feature = "mamba1")]
    pub use crate::mamba1::{self, prelude::*};

    #[cfg(feature = "mamba2")]
    pub use crate::mamba2::{self, prelude::*};

    // Mamba-3 types are intentionally not glob-re-exported here to avoid
    // name collisions (e.g. `SsdPath`) with the Mamba-2 prelude.
    // Import them explicitly: `use burn_mamba::mamba3::prelude::*;`
    #[cfg(feature = "mamba3")]
    pub use crate::mamba3;
}

pub mod utils;

pub const DENY_NAN: bool = false;
pub const DENY_INF: bool = false;
