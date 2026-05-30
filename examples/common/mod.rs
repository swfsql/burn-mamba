//! Shared infrastructure for the burn-mamba examples.
//!
//! Each concrete example (`fibonacci`, `mnist-class`) wires together the pieces
//! here: runtime [`device`] selection, CLI + artifact handling ([`cli`]), the
//! example [`model`] networks, the generic [`training`] config, and the
//! sequential-[`mnist`] dataset.
//!
//! With the new Dispatch-based architecture, no module here carries a backend
//! type generic — `Tensor`/`Device`/`Module` are pinned to the global
//! `Dispatch` backend, and the device chooses the concrete runtime backend.

#![allow(dead_code)]

/// CLI parsing, artifact directory management, and the train/infer flow.
pub mod cli;
/// Runtime [`Device`] selection + optional dtype configuration.
pub mod device;
/// Sequential-MNIST dataset loading and batching.
pub mod mnist;
/// The [`ModelConfigExt`](model::ModelConfigExt) seam bridging example configs to
/// the library's unified network types.
pub mod model;
/// The shared training configuration.
pub mod training;
