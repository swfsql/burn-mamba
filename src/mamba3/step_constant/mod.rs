//! # Constant-input stepping shortcut — `step_infinite`
//!
//! When the **same token** is fed to [`Mamba3::step`] over and over, every
//! data-dependent quantity is the same at each step: the trapezoid
//! coefficients `α`, `β`, `γ`, the QK-normed `b`/`c`, and the per-step
//! rotation increment (angle `θ̂` for `Complex2D`, unit quaternion `q` for
//! `Quaternion4D`). Only the *cumulative* rotation moves — by that same
//! increment each step — so with `P` the per-step rotation operator and `R₁`
//! the cumulative rotation at the first constant step,
//!
//! ```text
//!   Bₜ = R₁ Pᵗ⁻¹ b ,   Cₜ = R₁ Pᵗ⁻¹ c ,
//!   hₜ = α hₜ₋₁ + x ⊗ (β Bₜ₋₁ + γ Bₜ)            (t ≥ 2)
//! ```
//!
//! and the unrolled steps collapse to a **matrix geometric series** in
//! `α P⁻¹` (spectral radius `α < 1`, since `a_floor > 0` and `Δ > 0`).
//!
//! `P` is block-diagonal: per RoPE pair it is the complex scalar `e^{iθ̂}`, per
//! quaternion block left-multiplication by `w = q*`. In both cases all the
//! factors live in one **abelian** subalgebra (powers of a single rotation
//! commute — for quaternions, `span{1, û} ≅ ℂ`), so the series is a handful of
//! scalar complex / quaternion ops per (head, pair/block): `n` steps cost O(1).
//!
//! In the readout `y_n = Cₙᵀ h_n + D x` the unbounded phase cancels
//! (`⟨R c, R v⟩ = ⟨c, v⟩`), leaving only the *relative* rotation `P⁻¹`. As
//! `n → ∞` the initial-state term and the partial-sum correction decay like
//! `αⁿ`, so the **output converges** even though the state `h` orbits forever:
//!
//! ```text
//!   y_∞ = xᵀ · cᵀ (γ + β P⁻¹)(I − α P⁻¹)⁻¹ b  +  D x
//! ```
//!
//! — the block's stationary fixed point, independent of any starting cache.
//! [`Mamba3::step_infinite`] evaluates exactly this (and therefore takes and
//! returns no cache).
//!
//! Numerics: the rotation angle is reduced mod `2π` with the value-exact
//! [`wrap_angle`]; the geometric denominators satisfy
//! `|1 − α e^{−iθ̂}|² ≥ (1 − α)²` and are floored by
//! [`div_eps`](crate::utils::div_eps). When `α → 1` *and* `θ̂ → 0` the series
//! value `(β+γ)/(1−α)` stays finite but loses fp32 precision once `1 − α` nears
//! the epsilon floor — the same regime where the unrolled recurrence itself
//! accumulates near-undamped terms.

use crate::mamba3::double_ssd::double_ssd::StepProjection;
use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::{
    quat_conj, quat_from_scaled_axis, quat_mul, rotate_state_rank_blocks,
};
use crate::modules::sanity as san;
use crate::modules::wrap_angle;
use crate::utils::div_eps;
use burn::prelude::*;
use core::f32::consts::PI;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl Mamba3 {
    /// Stationary **fixed point** of the block under a constant token: the
    /// limit of `step(input, …)` outputs as the same `input_bd` is stepped
    /// forever.
    ///
    /// Closed form, O(1) in the horizon (see the [module header](self) for the
    /// derivation). The limit forgets any starting state (`αⁿ → 0`) and the
    /// SSM state itself never converges (it keeps rotating; only the output
    /// does), so this takes **no cache and returns none**.
    ///
    /// Both rotation kinds and both SSD pathways are covered (the recurrence
    /// is pathway-agnostic at boundaries). Differentiable; gradients are the
    /// limit of the unrolled gradients.
    ///
    /// # Shapes
    /// - `input_bd` : `[batch, d_model]`
    /// - output     : `[batch, d_model]`
    pub fn step_infinite(&self, input_bd: Tensor<2>) -> Tensor<2> {
        let StepProjection {
            z_bi,
            x_bhp,
            b_bmhr,
            c_bmhr,
            rot_ba,
            dt_bh,
            alpha_bh,
            beta_bh,
            gamma_bh,
        } = self.step_project(input_bd);
        let [batch, mimo_rank, nheads, _state_rank] = b_bmhr.dims();
        let eps = div_eps(alpha_bh.dtype());

        // Rotation-free channels: (β + γ) / (1 − α).
        let tail_bh =
            (beta_bh.clone() + gamma_bh.clone()) / (-alpha_bh.clone() + 1.0).clamp_min(eps);

        // Per-pair/block readout factor  m = (γ + β P⁻¹)(1 − α P⁻¹)⁻¹  applied
        // to `b`; the cumulative rotation cancels against `Cₙ` (orthogonality).
        let b_eff_bmhr = match self.rotation_kind() {
            RotationKind::Complex2D => {
                let theta_bha = per_step_angle(rot_ba, dt_bh); // θ̂
                let (cos, sin) = cos_sin(theta_bha);
                let a_bh1 = alpha_bh.unsqueeze_dim::<3>(2);
                let beta_bh1 = beta_bh.unsqueeze_dim::<3>(2);
                let gamma_bh1 = gamma_bh.unsqueeze_dim::<3>(2);
                // (γ + β e^{−iθ̂}) / (1 − α e^{−iθ̂})
                let num_re = gamma_bh1 + beta_bh1.clone() * cos.clone();
                let num_im = -beta_bh1 * sin.clone();
                let den_re = -a_bh1.clone() * cos + 1.0;
                let den_im = a_bh1 * sin;
                let (m_re, m_im) = complex_div(num_re, num_im, den_re, den_im);
                mul_complex_partial(b_bmhr, m_re, m_im, tail_bh, self.rope_dim, mimo_rank == 1)
            }
            RotationKind::Quaternion4D => {
                let g_bhj3 = per_step_generator(rot_ba, dt_bh, self.num_quat_blocks);
                let q_bhj4 = quat_from_scaled_axis::<4>(g_bhj3); // P⁻¹ ↔ q
                let alpha_bh11 = alpha_bh.unsqueeze_dims::<4>(&[2, 3]);
                let beta_bh11 = beta_bh.unsqueeze_dims::<4>(&[2, 3]);
                let gamma_bh11 = gamma_bh.unsqueeze_dims::<4>(&[2, 3]);
                // (γ + β q) ⊗ (1 − α q)⁻¹  — all in the abelian subalgebra of q.
                let num = quat_scalar_affine(gamma_bh11, beta_bh11, q_bhj4.clone());
                let den_inv = quat_inv(quat_one_minus(q_bhj4 * alpha_bh11));
                let f_bhj4 = quat_mul(num, den_inv);
                mul_quat_partial(b_bmhr, f_bhj4, tail_bh, self.num_quat_blocks * 4)
            }
        };
        san(&b_eff_bmhr);

        // y_∞[m] = Σ_m' ⟨c[m], m·b[m']⟩ · x_vals[m']   (then D-skip/gate/out-proj).
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let x_vals_bmhp = helpers::build_v_with_mimo::<3, 4>(x_bhp, mimo_x_hmp.as_ref(), 1);
        let out_m_bmhp = if self.use_siso_decode_kernels() {
            // SISO: the Gram matrix is the scalar ⟨c, m·b⟩ per (batch, nheads),
            // so both matmuls degenerate — contract `state_rank` with a
            // reduction and broadcast the result over `per_head_dim`.
            let gram_bmh1: Tensor<4> = (c_bmhr * b_eff_bmhr).sum_dim(3);
            x_vals_bmhp.clone() * gram_bmh1
        } else {
            let gram_bhmm = {
                let c_bhmr = c_bmhr.swap_dims(1, 2);
                let b_bhrm = b_eff_bmhr.permute([0, 2, 3, 1]);
                c_bhmr.matmul(b_bhrm) // [batch, nheads, mimo_rank, mimo_rank']
            };
            let x_bhmp = x_vals_bmhp.clone().swap_dims(1, 2);
            gram_bhmm.matmul(x_bhmp).swap_dims(1, 2)
        };
        assert_eq!(
            [batch, mimo_rank, nheads, self.per_head_dim()],
            out_m_bmhp.dims()
        );
        san(&out_m_bmhp);

        self.step_finish(out_m_bmhp, x_vals_bmhp, z_bi)
    }
}

// ---------------------------------------------------------------------------
// Per-step rotation increments (mirroring `rotate_bc_step`)
// ---------------------------------------------------------------------------

/// `θ̂ = Δ · π·tanh(rot)` — the constant per-step RoPE angle increment.
fn per_step_angle(rot_ba: Tensor<2>, dt_bh: Tensor<2>) -> Tensor<3> {
    dt_bh.unsqueeze_dim::<3>(2) * (rot_ba.tanh() * PI).unsqueeze_dim::<3>(1)
}

/// `g = Δ · π·tanh(rot)` per quaternion block — the constant per-step rotation
/// generator (`q = exp(g/2)`).
fn per_step_generator(rot_ba: Tensor<2>, dt_bh: Tensor<2>, blocks: usize) -> Tensor<4> {
    let [batch, _a] = rot_ba.dims();
    (rot_ba.tanh() * PI)
        .reshape([batch, blocks, 3])
        .unsqueeze_dim::<4>(1)
        * dt_bh.unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3)
}

// ---------------------------------------------------------------------------
// Complex helpers (per RoPE pair, `[batch, nheads, num_rope_angles]`)
// ---------------------------------------------------------------------------

/// `(cos, sin)` of an angle tensor, reduced mod `2π` first (value-exact).
fn cos_sin(angle_bha: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
    let a = wrap_angle(angle_bha);
    (a.clone().cos(), a.sin())
}

/// Component-wise complex quotient; `|den|²` floored by `div_eps`.
fn complex_div(
    nr: Tensor<3>,
    ni: Tensor<3>,
    dr: Tensor<3>,
    di: Tensor<3>,
) -> (Tensor<3>, Tensor<3>) {
    let eps = div_eps(dr.dtype());
    let d2 = (dr.clone() * dr.clone() + di.clone() * di.clone()).clamp_min(eps);
    (
        (nr.clone() * dr.clone() + ni.clone() * di.clone()) / d2.clone(),
        (ni * dr - nr * di) / d2,
    )
}

/// Multiply the rotation-active entries of `x` by the complex scalar
/// `(re, im)` per pair — same pairing conventions as [`apply_rope_partial`] —
/// and the pass-through entries by the real scalar `tail`.
///
/// `re`/`im` are `[batch, nheads, num_rope_angles]` (broadcast over the
/// `mimo_rank` axis); `tail` is `[batch, nheads]`.
fn mul_complex_partial(
    x_bmhr: Tensor<4>,
    re_bha: Tensor<3>,
    im_bha: Tensor<3>,
    tail_bh: Tensor<2>,
    rope_dim: usize,
    rotate_pairwise: bool,
) -> Tensor<4> {
    let [batch, mimo_rank, nheads, state_rank] = x_bmhr.dims();
    let tail_b1h1 = tail_bh.unsqueeze_dims::<4>(&[1, 3]);
    if rope_dim == 0 {
        // RoPE disabled: every channel is a plain geometric series.
        return x_bmhr * tail_b1h1;
    }
    let re_b1ha = re_bha.unsqueeze_dim::<4>(1);
    let im_b1ha = im_bha.unsqueeze_dim::<4>(1);

    if rotate_pairwise {
        // Interleaved (SISO/NeoX): pairs are local, so the first `rope_dim`
        // entries can be handled standalone.
        let n2 = rope_dim / 2;
        let head = x_bmhr.clone().narrow(3, 0, rope_dim);
        let head_pairs = head.reshape([batch, mimo_rank, nheads, n2, 2]);
        let x0 = head_pairs.clone().narrow(4, 0, 1).squeeze_dim::<4>(4);
        let x1 = head_pairs.narrow(4, 1, 1).squeeze_dim::<4>(4);
        let x0m = re_b1ha.clone() * x0.clone() - im_b1ha.clone() * x1.clone();
        let x1m = im_b1ha * x0 + re_b1ha * x1;
        let head = Tensor::cat(
            vec![x0m.unsqueeze_dim::<5>(4), x1m.unsqueeze_dim::<5>(4)],
            4,
        )
        .reshape([batch, mimo_rank, nheads, rope_dim]);
        if rope_dim == state_rank {
            head
        } else {
            let tail = x_bmhr.narrow(3, rope_dim, state_rank - rope_dim) * tail_b1h1;
            Tensor::cat(vec![head, tail], 3)
        }
    } else {
        // Half-and-half (MIMO/GPT-J): entry `i` pairs with `i + state_rank/2`;
        // only the first `rope_dim / 2` pairs are active.
        let half = state_rank / 2;
        let active = rope_dim / 2;
        let x_h1 = x_bmhr.clone().narrow(3, 0, half);
        let x_h2 = x_bmhr.narrow(3, half, half);
        let x_h1_rope = x_h1.clone().narrow(3, 0, active);
        let x_h2_rope = x_h2.clone().narrow(3, 0, active);
        let h1_m = re_b1ha.clone() * x_h1_rope.clone() - im_b1ha.clone() * x_h2_rope.clone();
        let h2_m = im_b1ha * x_h1_rope + re_b1ha * x_h2_rope;
        if active == half {
            Tensor::cat(vec![h1_m, h2_m], 3)
        } else {
            let h1_pass = x_h1.narrow(3, active, half - active) * tail_b1h1.clone();
            let h2_pass = x_h2.narrow(3, active, half - active) * tail_b1h1;
            Tensor::cat(vec![h1_m, h1_pass, h2_m, h2_pass], 3)
        }
    }
}

// ---------------------------------------------------------------------------
// Quaternion helpers (per block, `[batch, nheads, blocks, 4]`)
// ---------------------------------------------------------------------------

/// `1 − sq` for a (scalar-scaled) quaternion `sq`: negate, add 1 to the real
/// part.
fn quat_one_minus(sq_bhj4: Tensor<4>) -> Tensor<4> {
    let w = -sq_bhj4.clone().narrow(3, 0, 1) + 1.0;
    let xyz = -sq_bhj4.narrow(3, 1, 3);
    Tensor::cat(vec![w, xyz], 3)
}

/// `a + b·q` for per-head scalars `a`, `b` (`[batch, nheads, 1, 1]`) and a
/// quaternion tensor `q`.
fn quat_scalar_affine(a_bh11: Tensor<4>, b_bh11: Tensor<4>, q_bhj4: Tensor<4>) -> Tensor<4> {
    let w = a_bh11 + b_bh11.clone() * q_bhj4.clone().narrow(3, 0, 1);
    let xyz = b_bh11 * q_bhj4.narrow(3, 1, 3);
    Tensor::cat(vec![w, xyz], 3)
}

/// Quaternion inverse `q⁻¹ = q* / ‖q‖²`, with `‖q‖²` floored by `div_eps`.
/// (Used on `1 − α q`, whose norm is bounded below by `1 − α > 0`.)
fn quat_inv(q_bhj4: Tensor<4>) -> Tensor<4> {
    let eps = div_eps(q_bhj4.dtype());
    let n2 = (q_bhj4.clone() * q_bhj4.clone()).sum_dim(3).clamp_min(eps);
    quat_conj(q_bhj4) / n2
}

/// Left-multiply the first `rope_width` state-rank entries of `x` by the (not
/// necessarily unit) quaternion `f` per block, and scale the pass-through
/// entries by the real scalar `tail` — the quaternion analogue of
/// [`mul_complex_partial`].
fn mul_quat_partial(
    x_bmhr: Tensor<4>,
    f_bhj4: Tensor<4>,
    tail_bh: Tensor<2>,
    rope_width: usize,
) -> Tensor<4> {
    let [batch, mimo_rank, nheads, state_rank] = x_bmhr.dims();
    let blocks = f_bhj4.dims()[2];
    let tail_b1h1 = tail_bh.unsqueeze_dims::<4>(&[1, 3]);
    if rope_width == 0 {
        return x_bmhr * tail_b1h1;
    }
    let f_bmhj4 = f_bhj4
        .unsqueeze_dim::<5>(1)
        .expand([batch, mimo_rank, nheads, blocks, 4]);
    if rope_width == state_rank {
        rotate_state_rank_blocks::<4, 5>(x_bmhr, f_bmhj4)
    } else {
        let head =
            rotate_state_rank_blocks::<4, 5>(x_bmhr.clone().narrow(3, 0, rope_width), f_bmhj4);
        let tail = x_bmhr.narrow(3, rope_width, state_rank - rope_width) * tail_b1h1;
        Tensor::cat(vec![head, tail], 3)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
