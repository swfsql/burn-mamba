//! Shared helpers used by both [`Mamba3::forward`](super::mamba3::Mamba3::forward)
//! and [`Mamba3::step`](super::mamba3::Mamba3::step). They isolate three blocks
//! that previously appeared in both methods at different ranks:
//!
//! 1. Trapezoidal discretisation: `dt`, `α`, `β`, `γ`, `da`.
//! 2. QK-norm + GQA expansion + per-(head, mimo-rank) bias on B / C.
//! 3. MIMO `V` construction: broadcast-multiply `x` by `mimo_x_hmp`.
//! 4. The rank-summed outer product `Σₘ v[m] ⊗ k[m]` feeding the SSM state
//!    (SISO-branched).
//! 5. Peeling the rotation channels off the in-projection.
//!
//! Most helpers are generic over the rank `D` of the data tensors so a single
//! definition serves both the sequence-aware (`forward`) and single-token
//! (`step`) code paths.

use burn_stack::modules::RmsNorm;
use burn_stack::modules::gqa_expand_to_heads;
use burn_stack::modules::softplus;
use burn::prelude::*;

/// Peel a **trailing** in-projection segment of `width` channels off `proj`,
/// yielding `None` for it when the block projects none.
///
/// Used for the two segments a block may omit entirely — the rotation channels
/// ([`RotationKind::Real1D`](crate::mamba3::rotation::RotationKind::Real1D)
/// projects none) and then `λ`
/// ([`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None) projects
/// none) — which is why the in-projection lays them out last, in that order.
///
/// An optional slice cannot simply be one more entry in the main `split_into`:
/// Burn has no zero-width tensors, and `split_with_sizes` *drops* a zero-length
/// segment rather than returning an empty one, so the destructuring would come
/// up one part short.
///
/// # Shapes
/// - `proj` : `[..., w]` along `dim`
/// - out    : `[..., w − width]` and, if any, `[..., width]`
pub fn split_trailing<const D: usize>(
    proj: Tensor<D>,
    width: usize,
    dim: usize,
) -> (Tensor<D>, Option<Tensor<D>>) {
    if width == 0 {
        return (proj, None);
    }
    let rest = proj.dims()[dim] - width;
    let tail = proj.clone().narrow(dim, rest, width);
    (proj.narrow(dim, 0, rest), Some(tail))
}

/// Output of [`trapezoidal_coefficients`].
///
/// All tensors share the rank `D` of the inputs.
pub struct TrapezoidCoeffs<const D: usize> {
    /// `Δₜ = softplus(dd_dt + dt_bias)`, clamped.
    pub dt: Tensor<D>,
    /// `Δₜ · Aₜ` (negative; the log-decay).
    pub da: Tensor<D>,
    /// `αₜ = exp(Δₜ · Aₜ) ∈ (0, 1]` — decay.
    pub alpha: Tensor<D>,
    /// `βₜ = (1 − λₜ) · Δₜ · αₜ` — left-endpoint weight. `None` under
    /// [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None), which has
    /// no left endpoint: the tensor is not formed rather than formed as zeros.
    pub beta: Option<Tensor<D>>,
    /// `γₜ = λₜ · Δₜ` — right-endpoint weight, and **`Δₜ` itself** when there is
    /// no `λ` (the whole step is paid at the right endpoint).
    pub gamma: Tensor<D>,
}

/// Compute the trapezoidal discretisation coefficients from the raw
/// (data-dependent) projections. See the top-of-`mamba3.rs` docs for the
/// formulas.
///
/// All data tensors share rank `D` and have `nheads` as the last dim.
/// `dt_bias_h` is broadcast to match.
///
/// `lambda_raw` is `None` under
/// [`Trapezoid::None`](crate::mamba3::trapezoid::Trapezoid::None) — the block
/// projects no `λ` channels — and the coefficients then take their `λ ≡ 1`
/// values *by construction*: `β` is absent and `γ` **is** `Δ`, sharing its
/// tensor rather than recomputing `1 · Δ`.
pub fn trapezoidal_coefficients<const D: usize>(
    dd_dt: Tensor<D>,
    dd_a_raw: Tensor<D>,
    lambda_raw: Option<Tensor<D>>,
    dt_bias_h: Tensor<1>,
    dt_limit: (f64, f64),
    a_floor: f64,
) -> TrapezoidCoeffs<D> {
    // Broadcast dt_bias_h [nheads] → [1, ..., 1, nheads] so the addition aligns
    // on the last dim regardless of leading shape.
    let dt_bias_broadcast = dt_bias_h.unsqueeze::<D>();
    let dt = softplus(dd_dt + dt_bias_broadcast).clamp(dt_limit.0, dt_limit.1);
    // `A = −max(softplus(·), a_floor) ∈ (−∞, −a_floor]`. The floor must be
    // applied to the (positive) softplus *before* negating: a method call
    // binds tighter than unary minus, so `-softplus(x).clamp(NEG_INFINITY,
    // -a_floor)` would collapse the positive softplus to the constant
    // `-a_floor` and yield `A ≡ +a_floor` — a *growing* state (`α > 1`) with a
    // dead `dd_A` projection.
    let a = -softplus(dd_a_raw).clamp(a_floor, f64::INFINITY);
    let da = dt.clone() * a;
    let alpha = da.clone().exp();
    let (beta, gamma) = match lambda_raw {
        Some(lambda_raw) => {
            let lambda = burn::tensor::activation::sigmoid(lambda_raw);
            let beta = (-lambda.clone() + 1.0) * dt.clone() * alpha.clone();
            (Some(beta), lambda * dt.clone())
        }
        // λ ≡ 1: the left endpoint is unweighted and the right one takes Δ whole.
        None => (None, dt.clone()),
    };
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
pub fn qk_norm_expand_bias<const D: usize, const DP1: usize>(
    raw_mgr: Tensor<D>,
    norm: &RmsNorm,
    bias_hmr: Tensor<3>,
    group_dim: usize,
    nheads: usize,
) -> Tensor<D> {
    // RmsNorm operates on the last dim only, so the leading shape passes through.
    let normed = norm.forward(raw_mgr);
    let expanded = gqa_expand_to_heads::<D, DP1>(normed, group_dim, nheads);
    // Broadcast bias [nheads, mimo_rank, state_rank] → [1, ..., 1, mimo_rank, nheads, state_rank].
    let bias = bias_hmr.swap_dims(0, 1).unsqueeze::<D>();
    expanded + bias
}

/// Rank-summed outer product `state[b, h, p, r] = Σₘ v[b, m, h, p] · k[b, m, h, r]`
/// (`einsum('bmhp,bmhr->bhpr')`).
///
/// This is the per-token state contribution: each MIMO rank contributes an
/// outer product `v[m] ⊗ k[m]` and the shared state accumulates their sum.
///
/// At `mimo_rank == 1` the contracted dimension is 1, so the GEMM is rank-1 and
/// the sum is a single outer product — [`mimo_outer_sum_siso`] writes it as a
/// broadcast multiply instead. Which form wins is backend-dependent, and by a
/// wide margin at these (tiny, decode-sized) shapes: the broadcast is the
/// faster one on CUDA and *much* slower on the portable CPU backends, whose
/// broadcast-elementwise path trails their matmul by more than an order of
/// magnitude here. The choice is therefore
/// [`Mamba3Config::siso_specialization_decode`](crate::mamba3::mamba3::Mamba3Config::siso_specialization_decode)'s
/// — the per-token flag, *not* the chunkwise one, whose verdict is the opposite;
/// both branches compute the same values and gradients.
pub fn mimo_outer_sum(
    v_bmhp: Tensor<4>,
    k_bmhr: Tensor<4>,
    siso_specialization: bool,
) -> Tensor<4> {
    let [_batch, mimo_rank, _nheads, _per_head_dim] = v_bmhp.dims();
    if mimo_rank == 1 && siso_specialization {
        mimo_outer_sum_siso(v_bmhp, k_bmhr)
    } else {
        mimo_outer_sum_mimo(v_bmhp, k_bmhr)
    }
}

/// SISO (`mimo_rank == 1`) rank-summed outer product: the single `v ⊗ k` as a
/// broadcast multiply of `[b, h, p, 1]` by `[b, h, 1, r]`.
pub fn mimo_outer_sum_siso(v_bmhp: Tensor<4>, k_bmhr: Tensor<4>) -> Tensor<4> {
    let v_bhp1: Tensor<4> = v_bmhp.squeeze_dim::<3>(1).unsqueeze_dim(3);
    let k_bh1r: Tensor<4> = k_bmhr.squeeze_dim::<3>(1).unsqueeze_dim(2);
    v_bhp1 * k_bh1r
}

/// General MIMO rank-summed outer product: a matmul contracting `mimo_rank`.
pub fn mimo_outer_sum_mimo(v_bmhp: Tensor<4>, k_bmhr: Tensor<4>) -> Tensor<4> {
    // v_bmhp [b, m, h, p] -> v_bhpm [b, h, p, m]; k_bmhr [b, m, h, r] -> k_bhmr.
    let v_bhpm = v_bmhp.permute([0, 2, 3, 1]);
    let k_bhmr = k_bmhr.swap_dims(1, 2);
    v_bhpm.matmul(k_bhmr)
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;

/// Build the MIMO value tensor `v = x ⊙ mimo_x` with broadcasting.
///
/// Inserts a `mimo_rank` axis at `insert_dim`. When `mimo_x_hmp` is `None`
/// (SISO), the inserted axis has size 1 and `x` is passed through; otherwise
/// broadcasting fills the inserted axis to size `mimo_rank`.
///
/// `DP1 = D + 1`.
pub fn build_v_with_mimo<const D: usize, const DP1: usize>(
    x: Tensor<D>,
    mimo_x_hmp: Option<&Tensor<3>>,
    insert_dim: usize,
) -> Tensor<DP1> {
    let x_with_rank_axis = x.unsqueeze_dim::<DP1>(insert_dim);
    match mimo_x_hmp {
        None => x_with_rank_axis,
        Some(mimo_x_hmp) => {
            // mimo_x_hmp [nheads, mimo_rank, per_head_dim] → swap_dims to
            // [mimo_rank, nheads, per_head_dim] → unsqueeze leading 1s. The
            // result broadcasts against `x_with_rank_axis` over (batch, seq, …).
            let mimo_x_broadcast = mimo_x_hmp.clone().swap_dims(0, 1).unsqueeze::<DP1>();
            x_with_rank_axis * mimo_x_broadcast
        }
    }
}
