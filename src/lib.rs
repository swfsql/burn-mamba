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
//! - [`mamba3`] — SSD extended with trapezoidal discretisation, a complex-valued
//!   state transition (data-dependent RoPE on B/C), and MIMO rank expansion.
//!
//! Everything *around* the block — the Pre-LN [`Layer`](burn_stack::modules::Layer),
//! the (virtual-)layer [`Layers`](burn_stack::modules::Layers) stack,
//! bidirectional pairs, latent/vocab networks, multi-gate residuals, class
//! tokens, LR/virtual-layer scheduling and the Muon parameter groups — lives in
//! the block-agnostic [`burn_stack`] crate. This crate supplies the three
//! [`Block`](burn_stack::modules::Block) implementations and, in [`unified`],
//! the runtime-selectable enums that pick a family at run time.
//!
//! ## Two execution modes
//!
//! Every block, layer, and network exposes both a parallel `forward()` (used
//! for training and prompt prefill) and a recurrent `step()` (used for
//! token-by-token decoding).  The two are mathematically equivalent: a
//! `forward()` over a sequence equals unrolling `step()` token by token from the
//! same initial cache — a parity property the test suites assert on outputs,
//! final cache, and gradients.

#![warn(missing_docs)]
#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

/// Mamba-1: the original selective state space model.
#[cfg(feature = "mamba1")]
pub mod mamba1;
/// Mamba-2: Structured State Space Duality (SSD).
#[cfg(feature = "mamba2")]
pub mod mamba2;
/// Mamba-3: trapezoidal SSD with a complex-valued state transition
/// (data-dependent RoPE) and MIMO.
#[cfg(feature = "mamba3")]
pub mod mamba3;

/// Convenience re-exports: `use burn_mamba::prelude::*;` brings the enabled
/// model families and their public types into scope.
pub mod prelude {
    #[cfg(feature = "mamba1")]
    pub use crate::mamba1::{self, prelude::*};

    #[cfg(feature = "mamba2")]
    pub use crate::mamba2::{self, prelude::*};

    #[cfg(feature = "mamba3")]
    pub use crate::mamba3::{self, prelude::*};

    // The runtime-selectable unified API (this crate).
    #[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
    pub use crate::unified::{
        MambaBidiLayers, MambaBidiLayersConfig, MambaLatentNet, MambaLatentNetConfig,
        MambaVocabNet, MambaVocabNetConfig,
    };
    pub use crate::unified::{MambaCaches, MambaSsdPath};

    // The block-generic composition layer (`burn-stack`).
    pub use burn_stack::prelude::*;
}

pub mod unified;

/// Re-export of the block-generic composition crate this one builds on, so a
/// dependent can reach `Layer`/`Layers`/networks without naming it separately.
pub use burn_stack;
