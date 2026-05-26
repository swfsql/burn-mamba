//! Shared helpers used by both [`Mamba3::forward`](super::mamba3::Mamba3::forward)
//! and [`Mamba3::step`](super::mamba3::Mamba3::step). They isolate three blocks
//! that previously appeared in both methods at different ranks:
//!
//! 1. Trapezoidal discretisation: `dt`, `α`, `β`, `γ`, `da`.
//! 2. QK-norm + GQA expansion + per-(head, mimo-rank) bias on B / C.
//! 3. MIMO `V` construction: broadcast-multiply `x` by `mimo_x_hmp`.
//!
//! Each helper is generic over the rank `D` of the data tensors so a single
//! definition serves both the sequence-aware (`forward`) and single-token
//! (`step`) code paths.

use crate::utils::gqa::gqa_expand_to_heads;
use crate::utils::rms_norm::RmsNorm;
use crate::utils::softplus::softplus;
use burn::prelude::*;

/// Output of [`trapezoidal_coefficients`].
///
/// All tensors share the rank `D` of the inputs.
pub struct TrapezoidCoeffs<B: Backend, const D: usize> {
    /// `Δₜ = softplus(dd_dt + dt_bias)`, clamped.
    pub dt: Tensor<B, D>,
    /// `Δₜ · Aₜ` (negative; the log-decay).
    pub da: Tensor<B, D>,
    /// `αₜ = exp(Δₜ · Aₜ) ∈ (0, 1]` — decay.
    pub alpha: Tensor<B, D>,
    /// `βₜ = (1 − λₜ) · Δₜ · αₜ` — left-endpoint weight.
    pub beta: Tensor<B, D>,
    /// `γₜ = λₜ · Δₜ` — right-endpoint weight.
    pub gamma: Tensor<B, D>,
}

/// Compute the trapezoidal discretisation coefficients from the raw
/// (data-dependent) projections. See the top-of-`mamba3.rs` docs for the
/// formulas.
///
/// All four data tensors share rank `D` and have `nheads` as the last dim.
/// `dt_bias_h` is broadcast to match.
pub fn trapezoidal_coefficients<B: Backend, const D: usize>(
    dd_dt: Tensor<B, D>,
    dd_a_raw: Tensor<B, D>,
    lambda_raw: Tensor<B, D>,
    dt_bias_h: Tensor<B, 1>,
    dt_limit: (f64, f64),
    a_floor: f64,
) -> TrapezoidCoeffs<B, D> {
    // Broadcast dt_bias_h [nheads] → [1, ..., 1, nheads] so the addition aligns
    // on the last dim regardless of leading shape.
    let dt_bias_broadcast = dt_bias_h.unsqueeze::<D>();
    let dt = softplus(dd_dt + dt_bias_broadcast).clamp(dt_limit.0, dt_limit.1);
    let a = -softplus(dd_a_raw).clamp(f64::NEG_INFINITY, -a_floor);
    let da = dt.clone() * a;
    let lambda = burn::tensor::activation::sigmoid(lambda_raw);
    let alpha = da.clone().exp();
    let beta = (-lambda.clone() + 1.0) * dt.clone() * alpha.clone();
    let gamma = lambda * dt.clone();
    TrapezoidCoeffs {
        dt,
        da,
        alpha,
        beta,
        gamma,
    }
}

/// QK-Norm → GQA-expand groups→heads → add per-(head, mimo-rank) bias.
///
/// The input is the raw B/C projection already reshaped to expose the group
/// dim, with last dim = `state_rank`. The output replaces the group dim with
/// the head dim, leaving the last dim untouched.
///
/// `DP1 = D + 1` (required by [`gqa_expand_to_heads`]'s intermediate rank).
pub fn qk_norm_expand_bias<B: Backend, const D: usize, const DP1: usize>(
    raw_mgr: Tensor<B, D>,
    norm: &RmsNorm<B>,
    bias_hmr: Tensor<B, 3>,
    group_dim: usize,
    nheads: usize,
) -> Tensor<B, D> {
    // RmsNorm operates on the last dim only, so the leading shape passes through.
    let normed = norm.forward(raw_mgr);
    let expanded = gqa_expand_to_heads::<_, D, DP1>(normed, group_dim, nheads);
    // Broadcast bias [nheads, mimo_rank, state_rank] → [1, ..., 1, mimo_rank, nheads, state_rank].
    let bias = bias_hmr.permute([1, 0, 2]).unsqueeze::<D>();
    expanded + bias
}

/// Build the MIMO value tensor `v = x ⊙ mimo_x` with broadcasting.
///
/// Inserts a `mimo_rank` axis at `insert_dim`. When `mimo_x_hmp` is `None`
/// (SISO), the inserted axis has size 1 and `x` is passed through; otherwise
/// broadcasting fills the inserted axis to size `mimo_rank`.
///
/// `DP1 = D + 1`.
pub fn build_v_with_mimo<B: Backend, const D: usize, const DP1: usize>(
    x: Tensor<B, D>,
    mimo_x_hmp: Option<&Tensor<B, 3>>,
    insert_dim: usize,
) -> Tensor<B, DP1> {
    let x_with_rank_axis = x.unsqueeze_dim::<DP1>(insert_dim);
    match mimo_x_hmp {
        None => x_with_rank_axis,
        Some(mimo_x_hmp) => {
            // mimo_x_hmp [nheads, mimo_rank, per_head_dim] → permute to
            // [mimo_rank, nheads, per_head_dim] → unsqueeze leading 1s. The
            // result broadcasts against `x_with_rank_axis` over (batch, seq, …).
            let mimo_x_broadcast = mimo_x_hmp.clone().permute([1, 0, 2]).unsqueeze::<DP1>();
            x_with_rank_axis * mimo_x_broadcast
        }
    }
}
