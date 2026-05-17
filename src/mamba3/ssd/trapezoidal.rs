//! Trapezoidal two-SSD wrapper.
//!
//! The trapezoidal discretization is implemented directly in [`crate::mamba3::mamba3::Mamba3::forward`]
//! by calling the standard SSD twice (γ-term and β-term) and summing the outputs.
//! This module is reserved for future utilities or higher-level abstractions.
