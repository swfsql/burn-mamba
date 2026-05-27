//! # Mamba-3
//!
//! Mamba-3 extends Mamba-2 with three independent additions (each works alone
//! or combined): **trapezoidal discretisation**, **data-dependent RoPE** on the
//! B/C projections, and **MIMO** (multiple-input multiple-output) rank
//! expansion.  See [`mamba3`] for the full combined math.
//!
//! ## Two SSD pathways
//!
//! The trapezoidal recurrence is realised by two interchangeable algorithms,
//! selected at runtime by which **cache variant** is supplied:
//!
//! - [`double_ssd`] — splits the trapezoid into two standard SSD calls
//!   (simple, easy to verify; ~2× the intra-chunk memory).
//! - [`single_ssd`] — one SSD call in the official-kernel form
//!   (≈ half the training memory; the cache's SSM accumulator has different
//!   mid-sequence semantics).
//!
//! [`cache`] holds the enum that dispatches between them; [`ssd_path`] selects
//! the pathway-agnostic *algorithm* (Minimal / Serial / SerialRecalculated).

pub mod double_ssd;
pub mod single_ssd;

pub mod bidi;
pub mod cache;
pub(crate) mod helpers;
pub mod layer;
pub mod mamba3;
pub mod network;
pub mod ssd_path;

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::single_ssd::prelude::*;
use burn::backend::Backend;

/// Backend capability required to run Mamba-3.
///
/// Aggregates the per-pathway extension traits ([`Mamba3DoubleSsdBackendExt`]
/// and [`Mamba3SingleSsdBackendExt`]); every plain Burn backend satisfies it
/// via the default implementations, and `Autodiff<B>` additionally gets the
/// custom memory-efficient backward.
pub trait Mamba3BackendExt:
    Backend + Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt
{
}

crate::decl_ssd_autodiff_backend_ext!(
    Mamba3AutodiffBackendExt,
    Mamba3BackendExt,
    Mamba3DoubleSsdAutodiffBackendExt,
    Mamba3SingleSsdAutodiffBackendExt
);
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3BackendExt);

/// Blanket [`Mamba3BackendExt`] implementation for autodiff backends.
#[cfg(feature = "autodiff")]
pub mod backwards {
    use super::*;
    use burn::backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};

    impl<B: Backend + Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt, C: CheckpointStrategy>
        Mamba3BackendExt for Autodiff<B, C>
    {
    }
}

/// Public re-exports for Mamba-3.
pub mod prelude {
    #[cfg(feature = "autodiff")]
    pub use super::Mamba3AutodiffBackendExt;
    pub use super::Mamba3BackendExt;
    use super::*;

    pub use cache::{Mamba3Cache, Mamba3Caches};
    pub use layer::{Mamba3Layer, Mamba3LayerConfig, Mamba3Layers, Mamba3LayersConfig};
    pub use mamba3::{Mamba3, Mamba3Config};
    pub use network::{Mamba3Network, Mamba3NetworkConfig};
    pub use ssd_path::Mamba3SsdPath;
}
