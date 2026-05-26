//! # burn-mamba — Mamba-1/2/3 selective state space models on Burn
//!
//! A minimal, readable reference implementation of the
//! [Mamba-1](https://arxiv.org/abs/2312.00752),
//! [Mamba-2](https://arxiv.org/abs/2405.21060), and
//! [Mamba-3](https://arxiv.org/abs/2603.15569) SSM architectures on top of the
//! [Burn](https://github.com/tracel-ai/burn/) deep learning framework.
//!
//! The goal is clarity: the official CUDA/Triton kernels are ported down to
//! standard, portable Burn tensor operations, so the same code runs on every
//! backend (CPU, WGPU, CUDA, Metal, LibTorch, …).  There are **no custom
//! kernels**.
//!
//! ## Module families
//!
//! Each family lives in its own module and follows the same composition
//! (`Network` → `Layers` → `Layer` → `Block`):
//!
//! - [`mamba1`] — the original selective SSM (conv1d + sequential selective
//!   scan).
//! - [`mamba2`] — Structured State Space Duality (SSD): the recurrence is recast
//!   as a chunkwise, GEMM-friendly algorithm.
//! - [`mamba3`] — SSD extended with trapezoidal discretisation, data-dependent
//!   RoPE on B/C, and MIMO rank expansion.
//!
//! Shared infrastructure lives in [`schedule`] (virtual-layer scheduling) and
//! [`utils`] (activations, norms, losses, and the custom-backward plumbing).
//!
//! ## Two execution modes
//!
//! Every block, layer, and network exposes both a parallel `forward()` (used
//! for training and prompt prefill) and a recurrent `step()` (used for
//! token-by-token decoding).  The two are mathematically equivalent: a
//! `forward()` over a sequence equals unrolling `step()` token by token from the
//! same initial cache — a parity property the test suites assert on outputs,
//! final cache, and gradients.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

/// Mamba-1: the original selective state space model.
#[cfg(feature = "mamba1")]
pub mod mamba1;
/// Mamba-2: Structured State Space Duality (SSD).
#[cfg(feature = "mamba2")]
pub mod mamba2;
/// Mamba-3: trapezoidal SSD with data-dependent RoPE and MIMO.
#[cfg(feature = "mamba3")]
pub mod mamba3;
/// Virtual-layer → real-weight index scheduling shared by all families.
pub mod schedule;

/// Convenience re-exports: `use burn_mamba::prelude::*;` brings the enabled
/// model families and their public types into scope.
pub mod prelude {
    #[cfg(feature = "mamba1")]
    pub use crate::mamba1::{self, prelude::*};

    #[cfg(feature = "mamba2")]
    pub use crate::mamba2::{self, prelude::*};

    #[cfg(feature = "mamba3")]
    pub use crate::mamba3::{self, prelude::*};
}

/// Shared activations, norms, losses, and custom-backward plumbing.
pub mod utils;

/// When `true`, [`utils::sanity::sanity`] panics if it observes a `NaN`.
///
/// Compiled-in guard (off by default) for debugging numerical issues; leaving
/// it `false` removes the check entirely.
pub const DENY_NAN: bool = false;

/// When `true`, [`utils::sanity::sanity`] panics if it observes an `Inf`.
///
/// Compiled-in guard (off by default), companion to [`DENY_NAN`].
pub const DENY_INF: bool = false;
