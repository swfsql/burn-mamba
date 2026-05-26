//! Bidirectional Mamba-2 wrappers for non-autoregressive tasks.

/// Naive bidirectional implementation (the block itself is unchanged; a
/// forward and a reversed pass are merged).
pub mod naive;
