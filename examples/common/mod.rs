//! Shared infrastructure for the burn-mamba examples.
//!
//! Each concrete example (`fibonacci`, `mnist-class`) wires together the pieces
//! here: compile-time [`backend`]/dtype selection, CLI + artifact handling
//! ([`cli`]), the generic [`training`] loop and [`optim`]izer wrapper, the
//! example [`model`] networks, and the sequential-[`mnist`] dataset.

#![allow(dead_code)]

/// Compile-time backend + float/int dtype selection from feature flags.
pub mod backend;
/// Argument parsing, artifact directory management, and the train/infer flow.
pub mod cli;
/// Sequential-MNIST dataset loading and batching.
pub mod mnist;
/// The example networks (`MyMamba2Network` / `MyMamba3Network`) and builders.
pub mod model;
/// Optimizer configuration wrapper.
pub mod optim;
/// The generic training loop and its configuration.
pub mod training;
