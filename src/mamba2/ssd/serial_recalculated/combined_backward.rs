//! # Recompute-based gradient math for the Mamba-2 SSD
//!
//! The analytic backward of the `SerialRecalculated` path. The
//! [standard SSD scan](crate::standard_ssd) holds the gradient of the scan.
//! This file holds what Mamba-2 adds to it. It forms the inputs `v = Δ·x` and
//! `da = Δ·A`, and it calls the backward of the scan. Then it applies the chain
//! rule back to its own inputs, with the backward of the `D` skip:
//!
//! ```text
//! d_x = d_v·Δ + d_y·D
//! d_Δ = Σ_p d_v·x + d_da·A
//! d_A = Σ_{b,n,l} d_da·Δ
//! d_D = Σ_{b,n,l,p} d_y·x
//! ```
//!
//! The scan returns `d_b`, `d_c`, `d_initial_state` and `d_origin` directly.
//! The length-1 `m` axis of the scan exists only at the call. ADR-0010 records
//! why this path does not follow the reference line by line.
//!
//! All the math is on backend **primitives**, through the rank-tagged `F`
//! wrapper. The [recompute node](burn_stack::utils::recompute_node) runs with a
//! generic backend `B`, where the high-level `Tensor` (pinned to the global
//! `Dispatch` backend) is not available.

use super::serial_recalculated::scan_inputs;
use crate::standard_ssd;
use burn_stack::utils::fprim::{F, san};
use burn::backend::Backend;

/// Per-input gradients produced by [`combined_backward`] (one field per
/// differentiable forward input).
#[non_exhaustive]
pub struct CombinedGrads<B: Backend> {
    /// Gradient of the input `x`.
    pub d_x_bnlhp: F<B, 5>,
    /// Gradient of the step size `Δ` (`dt`).
    pub d_dt_discretized_bhnl: F<B, 4>,
    /// Gradient of the input projection `B`.
    pub d_b_bnlhr: F<B, 5>,
    /// Gradient of the output projection `C`.
    pub d_c_bnlhr: F<B, 5>,
    /// Gradient of the per-head skip term `D`.
    pub d_d_h: F<B, 1>,
    /// Gradient of the initial state.
    pub d_initial_state_bhpr: F<B, 4>,
    /// Gradient of the origin state of the resets (`None` without resets).
    pub d_origin_bhpr: Option<F<B, 4>>,
    /// Gradient of the per-head decay rate `A` (as `a_decay_h`).
    pub d_a_decay_h: F<B, 1>,
}

/// Memory-efficient backward for the Mamba-2 chunkwise SSD.
///
/// # Arguments
/// - `d_y_bnlhp` — upstream gradient of the SSD output
/// - `d_final_bhpr` — upstream gradient of the final state
/// - `x_bnlhp`, `dt_discretized_bhnl`, `b_bnlhr`, `c_bnlhr`, `d_h`,
///   `initial_state_bhpr`, `a_decay_h` — the seven saved forward inputs
/// - `restarts` — `(keep_bn, origin_bhpr)`, or `None`. `keep_bn` is the
///   constant reset factor of the forward (`0` at a chunk that starts a new
///   segment). `origin_bhpr` is the state that those chunks start from.
///
/// # Returns
/// One [`CombinedGrads`] struct containing gradients for all 7 inputs, and
/// for the origin state.
#[allow(clippy::too_many_arguments)]
pub fn combined_backward<B: Backend>(
    d_y_bnlhp: F<B, 5>,
    d_final_bhpr: F<B, 4>,
    //
    x_bnlhp: F<B, 5>,
    dt_discretized_bhnl: F<B, 4>,
    b_bnlhr: F<B, 5>,
    c_bnlhr: F<B, 5>,
    d_h: F<B, 1>,
    initial_state_bhpr: F<B, 4>,
    a_decay_h: F<B, 1>,
    restarts: Option<(F<B, 2>, F<B, 4>)>,
) -> CombinedGrads<B> {
    let [batch, nchunks, chunk_len, nheads, per_head_dim] = x_bnlhp.dims();

    san(&x_bnlhp);
    san(&dt_discretized_bhnl);
    san(&d_h);
    san(&a_decay_h);

    let dt_bnlh = dt_discretized_bhnl.permute([0, 2, 3, 1]);
    let a_decay_111h = a_decay_h.unsqueeze_dims::<4>(&[0, 1, 2]);
    let (v_bnlhp, da_bnlh) = scan_inputs(x_bnlhp.clone(), dt_bnlh.clone(), a_decay_111h.clone());

    let grads = standard_ssd::combined_backward::<B>(
        d_y_bnlhp.clone().unsqueeze_dim::<6>(3),
        d_final_bhpr,
        v_bnlhp.unsqueeze_dim::<6>(3),
        da_bnlh,
        b_bnlhr.unsqueeze_dim::<6>(3),
        c_bnlhr.unsqueeze_dim::<6>(3),
        initial_state_bhpr,
        1,
        restarts,
    );
    let d_v_bnlhp = grads.d_v_bnlmhp.squeeze_dim::<5>(3);
    let d_da_bnlh = grads.d_da_bnlh;

    // ── SKIP backward: y += D · x ─────────────────────────────────────────
    let d_d_h = (d_y_bnlhp.clone() * x_bnlhp.clone())
        .permute([3, 0, 1, 2, 4]) // _hbnlp
        .reshape([nheads, batch * nchunks * chunk_len * per_head_dim]) // _hBNLP
        .sum_dim(1) // _h1
        .squeeze_dim::<1>(1); // _h
    san(&d_d_h);
    let d_x_skip_bnlhp = d_y_bnlhp * d_h.unsqueeze_dims::<5>(&[0, 1, 2, 4]);

    // ── The chain rule ────────────────────────────────────────────────────
    let d_x_bnlhp = d_v_bnlhp.clone() * dt_bnlh.clone().unsqueeze_dim::<5>(4) + d_x_skip_bnlhp;
    san(&d_x_bnlhp);

    let d_dt_discretized_bhnl = ((d_v_bnlhp * x_bnlhp).sum_dim(4).squeeze_dim::<4>(4)
        + d_da_bnlh.clone() * a_decay_111h)
        .permute([0, 3, 1, 2]);
    san(&d_dt_discretized_bhnl);

    let d_a_decay_h = (d_da_bnlh * dt_bnlh)
        .permute([3, 0, 1, 2]) // _hbnl
        .reshape([nheads, batch * nchunks * chunk_len]) // _hBNL
        .sum_dim(1) // _h1
        .squeeze_dim::<1>(1); // _h
    san(&d_a_decay_h);

    CombinedGrads {
        d_x_bnlhp,
        d_dt_discretized_bhnl,
        d_b_bnlhr: grads.d_b_bnlmhr.squeeze_dim::<5>(3),
        d_c_bnlhr: grads.d_c_bntmhr.squeeze_dim::<5>(3),
        d_d_h,
        d_initial_state_bhpr: grads.d_initial_state_bhpr,
        d_origin_bhpr: grads.d_origin_bhpr,
        d_a_decay_h,
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
