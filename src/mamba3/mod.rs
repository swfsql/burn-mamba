//! # Mamba-3
//!
//! Mamba-3 extends Mamba-2 with four independent additions. Each one works
//! alone or with the others:
//!
//! 1. **Trapezoidal discretisation** of the recurrence.
//! 2. A **complex-valued state transition**, applied as **data-dependent RoPE**
//!    on the B/C projections ([`rotation`](crate::mamba3::rotation)). This is a
//!    re-factoring of the transition, *not* a positional encoding.
//! 3. **MIMO** (multiple-input multiple-output) rank expansion.
//! 4. **MambaProduct** ([`product`](crate::mamba3::product)):
//!    [`micro_steps`](crate::mamba3::mamba3::Mamba3Config::micro_steps)
//!    recurrence steps per token, so the transition of one token is a
//!    **product** of `u` steps. The micro-steps fold into the sequence axis, so
//!    MambaProduct needs no new kernel.
//!
//! See [`mamba3`](crate::mamba3::mamba3) for the full combined math.
//!
//! ## Two SSD pathways
//!
//! Two interchangeable algorithms compute the trapezoidal recurrence. The
//! **cache variant** that the caller supplies selects one at runtime:
//!
//! - [`double_ssd`](crate::mamba3::double_ssd) — one standard SSD call per
//!   trapezoid term: one for the current sample, plus one per `β` tap
//!   ([`trapezoid`](crate::mamba3::trapezoid)). That is **two** calls at the
//!   default and three under a two-tap pattern. It is simple and easy to
//!   verify, but it uses ~2× the intra-chunk memory (~3× at two taps).
//! - [`single_ssd`](crate::mamba3::single_ssd) — **one** SSD call in the
//!   official-kernel form, for every tap pattern. It uses ≈ half the training
//!   memory of the double pathway at the default, and less when there are more
//!   taps. Its SSM accumulator has different semantics mid-sequence.
//!
//! [`cache`](crate::mamba3::cache) holds the enum that selects the pathway.
//! [`ssd_path`](crate::mamba3::ssd_path) selects the pathway-agnostic
//! *algorithm* (Minimal / Serial / SerialRecalculated).
//!
//! ## The tap pattern of the trapezoid
//!
//! [`trapezoid`](crate::mamba3::trapezoid) names the earlier sample(s) that the
//! second tap of the write reads. This choice exists only at `u > 1`, and it
//! changes the algorithm *and* the cache. Four of the six members have one lag
//! each ([`tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag)):
//!
//! - the default
//!   [`HorizontalCarryOver`](crate::mamba3::trapezoid::Trapezoid::HorizontalCarryOver)
//!   (lag 1),
//! - [`Vertical`](crate::mamba3::trapezoid::Trapezoid::Vertical) (lag `u`),
//!   which is equal to the default at `u = 1`,
//! - the gated
//!   [`HorizontalReset`](crate::mamba3::trapezoid::Trapezoid::HorizontalReset),
//! - the tapless [`None`](crate::mamba3::trapezoid::Trapezoid::None).
//!
//! The two other members have both lags. A second per-head mass mixes them.
//!
//! ## Positive systems beside the plant
//!
//! [`positive`](crate::mamba3::positive) adds a per-head scalar system. It
//! reads only the inputs and sets the coefficients of the plant:
//!
//! - a Kalman gate ([`Gain`](crate::mamba3::positive::Gain)) *computes* the
//!   decay from an accumulated precision,
//! - a soft max-plus register ([`Tropical`](crate::mamba3::positive::Tropical))
//!   adds to the readout.
//!
//! Both are nonnegative 2×2 matrix recurrences, scanned in log coordinates.
//! Each has one cache slot, and neither changes a kernel.

pub mod double_ssd;
pub mod single_ssd;

pub mod cache;
pub(crate) mod helpers;
pub mod mamba3;
pub mod positive;
pub mod product;
pub mod quat_scan;
pub mod rotation;
pub mod ssd_path;
pub mod trapezoid;

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::single_ssd::prelude::*;
use burn::backend::Backend;

/// Backend capability required to run Mamba-3.
///
/// Aggregates the per-pathway extension traits ([`Mamba3DoubleSsdBackendExt`]
/// and [`Mamba3SingleSsdBackendExt`]). Every plain Burn backend satisfies it
/// through the default implementations. `Autodiff<B>` also gets the custom
/// memory-efficient backward.
pub trait Mamba3BackendExt:
    Backend + Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt
{
}

burn_stack::decl_autodiff_backend_ext!(
    Mamba3AutodiffBackendExt,
    Mamba3BackendExt,
    Mamba3DoubleSsdAutodiffBackendExt,
    Mamba3SingleSsdAutodiffBackendExt
);
burn_stack::impl_backend_ext_for_burn_backends!(Mamba3BackendExt);

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
    pub use mamba3::{Mamba3, Mamba3Config, Mamba3Untied};
    pub use positive::{Gain, Tropical};
    pub use quat_scan::Mamba3QuatScanBackendExt;
    pub use rotation::{RotationKind, RotationState};
    pub use ssd_path::Mamba3SsdPath;
    pub use trapezoid::Trapezoid;
}
