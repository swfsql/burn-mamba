//! # Constant-input stepping shortcuts — `step_n_approx` / `step_infinite`
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
//! and `K = n − 1` unrolled steps collapse to a **matrix geometric series** in
//! `α P⁻¹` (spectral radius `α < 1`, since `a_floor > 0` and `Δ > 0`):
//!
//! ```text
//!   h_n = α^K h₁  +  x ⊗ R₁ P^{K−1} (I − (αP⁻¹)^K)(I − αP⁻¹)⁻¹ (β + γP) b
//! ```
//!
//! `P` is block-diagonal: per RoPE pair it is the complex scalar `e^{iθ̂}`, per
//! quaternion block left-multiplication by `w = q*`. In both cases all the
//! factors live in one **abelian** subalgebra (powers of a single rotation
//! commute — for quaternions, `span{1, û} ≅ ℂ`), so the series is a handful of
//! scalar complex / quaternion ops per (head, pair/block): `n` steps cost O(1).
//!
//! In the readout `y_n = Cₙᵀ h_n + D x` the unbounded phase cancels
//! (`⟨R c, R v⟩ = ⟨c, v⟩`), leaving only the *relative* rotation `P⁻¹`. As
//! `n → ∞` the `h₁` term and the partial-sum correction decay like `αⁿ`, so the
//! **output converges** even though the state `h` orbits forever:
//!
//! ```text
//!   y_∞ = xᵀ · cᵀ (γ + β P⁻¹)(I − α P⁻¹)⁻¹ b  +  D x
//! ```
//!
//! — the block's stationary fixed point, independent of any starting cache.
//! [`Mamba3::step_infinite`] evaluates exactly this (and therefore takes and
//! returns no cache); [`Mamba3::step_n_approx`] evaluates the finite-`n` form,
//! returning the exact step-`n` output *and* cache.
//!
//! Numerics: `α^K` is `exp(K·ΔA)`; every accumulated rotation power has its
//! (half-)angle reduced mod `2π` with the value-exact [`wrap_angle`]; the
//! geometric denominators satisfy `|1 − α e^{−iθ̂}|² ≥ (1 − α)²` and are floored
//! by [`div_eps`](crate::utils::div_eps). When `α → 1` *and* `θ̂ → 0` the series
//! value `(β+γ)(1−α^K)/(1−α) → K·(β+γ)`-ish stays finite, but its fp32
//! evaluation loses precision once `1 − α` nears the epsilon floor — the same
//! regime where the unrolled recurrence itself accumulates `K` near-undamped
//! terms.

use crate::mamba3::double_ssd::double_ssd::StepProjection;
use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::{
    RotationState, quat_conj, quat_from_scaled_axis, quat_mul, quat_normalize,
    rotate_state_rank_blocks,
};
use crate::modules::sanity as san;
use crate::modules::{apply_rope_partial, wrap_angle};
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
    /// does), so this takes **no cache and returns none**. Use
    /// [`Self::step_n_approx`] when the post-horizon cache is needed.
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
            da_bh: _,
            alpha_bh,
            beta_bh,
            gamma_bh,
        } = self.step_project(input_bd);
        let [batch, mimo_rank, nheads, _state_rank] = b_bmhr.dims();
        let eps = div_eps(alpha_bh.dtype());

        // Rotation-free channels: (β + γ) / (1 − α).
        let tail_bh = (beta_bh.clone() + gamma_bh.clone())
            / (-alpha_bh.clone() + 1.0).clamp_min(eps);

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
                mul_complex_partial(
                    b_bmhr,
                    m_re,
                    m_im,
                    tail_bh,
                    self.rope_dim,
                    mimo_rank == 1,
                )
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
        let gram_bhmm = {
            let c_bhmr = c_bmhr.permute([0, 2, 1, 3]);
            let b_bhrm = b_eff_bmhr.permute([0, 2, 3, 1]);
            c_bhmr.matmul(b_bhrm) // [batch, nheads, mimo_rank, mimo_rank']
        };
        let out_m_bmhp = {
            let x_bhmp = x_vals_bmhp.clone().permute([0, 2, 1, 3]);
            gram_bhmm.matmul(x_bhmp).permute([0, 2, 1, 3])
        };
        assert_eq!(
            [batch, mimo_rank, nheads, self.per_head_dim()],
            out_m_bmhp.dims()
        );
        san(&out_m_bmhp);

        self.step_finish(out_m_bmhp, x_vals_bmhp, z_bi)
    }

    /// Net effect of `n` consecutive [`Mamba3::step`] calls on the **same**
    /// token, in O(1): the step-`n` output and the step-`n` cache.
    ///
    /// For a single block this is *exact* (an algebraic reformulation of the
    /// unroll — same values, same gradients, up to float associativity); the
    /// `_approx` suffix refers to the stacked
    /// [`Layers`](crate::modules::Layers)-level composition, where inputs to
    /// deeper layers are held constant at their final value (error decays
    /// geometrically in `n`). `n = 1` is exactly one `step`.
    ///
    /// The first step runs as an ordinary [`Mamba3::step`] (consuming the
    /// cache's previous-token trapezoid contribution, which may belong to a
    /// *different* token); the remaining `n − 1` are the closed-form jump. The
    /// returned cache keeps the supplied pathway variant (`None` defaults to
    /// single-SSD, as in `step`).
    ///
    /// # Shapes
    /// - `input_bd` : `[batch, d_model]`
    /// - output     : `[batch, d_model]` (the **last** step's output)
    pub fn step_n_approx(
        &self,
        input_bd: Tensor<2>,
        n: usize,
        cache: Option<Mamba3Cache>,
    ) -> (Tensor<2>, Mamba3Cache) {
        assert!(n >= 1, "step_n_approx requires at least one step");
        let keep_single = !matches!(cache, Some(Mamba3Cache::DoubleSsd(_)));
        let cache = cache.map(|c| match c {
            Mamba3Cache::DoubleSsd(c) => c,
            Mamba3Cache::SingleSsd(c) => c.into(),
        });

        // Step 1: an ordinary recurrent step (boundary semantics are the
        // double-ssd ones — identical for both pathway caches at boundaries).
        let (out, cache) = self.step_double_ssd(input_bd.clone(), cache);
        let (out, cache) = if n == 1 {
            (out, cache)
        } else {
            self.jump_constant(self.step_project(input_bd), n - 1, cache)
        };

        let cache = if keep_single {
            Mamba3Cache::SingleSsd(cache.into())
        } else {
            Mamba3Cache::DoubleSsd(cache)
        };
        (out, cache)
    }

    /// The closed-form jump: advance `cache` (the state *after* the first
    /// constant step) by `k ≥ 1` further steps of the same token, whose
    /// projection is `p`.
    fn jump_constant(
        &self,
        p: StepProjection,
        k: usize,
        cache: Mamba3DoubleSsdCache,
    ) -> (Tensor<2>, Mamba3DoubleSsdCache) {
        let StepProjection {
            z_bi,
            x_bhp,
            b_bmhr,
            c_bmhr,
            rot_ba,
            dt_bh,
            da_bh,
            alpha_bh,
            beta_bh,
            gamma_bh,
        } = p;
        let [batch, mimo_rank, nheads, _state_rank] = b_bmhr.dims();
        let kf = k as f32;
        let eps = div_eps(alpha_bh.dtype());

        // α^K = exp(K·ΔA) — exact in log space.
        let alpha_k_bh = (da_bh * kf).exp();
        // Rotation-free channels: (β + γ)(1 − α^K)/(1 − α).
        let tail_bh = (beta_bh.clone() + gamma_bh.clone())
            * (-alpha_k_bh.clone() + 1.0)
            / (-alpha_bh.clone() + 1.0).clamp_min(eps);

        // Driven term Σₜ α^{n−t}(β Bₜ₋₁ + γ Bₜ) = R₁ P^{K−1} (1−(αP⁻¹)^K)(1−αP⁻¹)⁻¹ (β+γP) b,
        // plus the rotated Bₙ/Cₙ and the cumulative rotation at step n.
        let (b_drv_bmhr, b_n_bmhr, c_n_bmhr, rotation_n) = match self.rotation_kind() {
            RotationKind::Complex2D => {
                let angle1_bha = cache.rotation.clone().angle(); // Θ₁
                let theta_bha = per_step_angle(rot_ba, dt_bh); // θ̂
                let num_rope_angles = theta_bha.dims()[2];
                let rotate_pairwise = mimo_rank == 1;

                // Θₙ = Θ₁ + K·θ̂ (value-exact wrap; gradient w.r.t. θ̂ stays K).
                let angle_n_bha = wrap_angle(angle1_bha.clone() + theta_bha.clone() * kf);
                let expand_m = |a_bha: Tensor<3>| {
                    a_bha.unsqueeze_dim::<4>(1).expand([
                        batch,
                        mimo_rank,
                        nheads,
                        num_rope_angles,
                    ])
                };
                let b_n = apply_rope_partial::<4>(
                    b_bmhr.clone(),
                    expand_m(angle_n_bha.clone()),
                    self.rope_dim,
                    rotate_pairwise,
                );
                let c_n = apply_rope_partial::<4>(
                    c_bmhr,
                    expand_m(angle_n_bha.clone()),
                    self.rope_dim,
                    rotate_pairwise,
                );

                // φ = e^{i(Θ₁+(K−1)θ̂)} · (1 − α^K e^{−iKθ̂})/(1 − α e^{−iθ̂}) · (β + γ e^{iθ̂})
                let (cos1, sin1) = cos_sin(theta_bha.clone());
                let (cos_k, sin_k) = cos_sin(theta_bha.clone() * kf);
                let (lead_re, lead_im) = cos_sin(angle1_bha + theta_bha * (kf - 1.0));
                let a_bh1 = alpha_bh.clone().unsqueeze_dim::<3>(2);
                let ak_bh1 = alpha_k_bh.clone().unsqueeze_dim::<3>(2);
                let beta_bh1 = beta_bh.clone().unsqueeze_dim::<3>(2);
                let gamma_bh1 = gamma_bh.clone().unsqueeze_dim::<3>(2);
                let num_re = -ak_bh1.clone() * cos_k + 1.0;
                let num_im = ak_bh1 * sin_k;
                let den_re = -a_bh1.clone() * cos1.clone() + 1.0;
                let den_im = a_bh1 * sin1.clone();
                let (rat_re, rat_im) = complex_div(num_re, num_im, den_re, den_im);
                let bg_re = beta_bh1 + gamma_bh1.clone() * cos1;
                let bg_im = gamma_bh1 * sin1;
                let (t_re, t_im) = complex_mul(rat_re, rat_im, bg_re, bg_im);
                let (phi_re, phi_im) = complex_mul(t_re, t_im, lead_re, lead_im);

                let b_drv = mul_complex_partial(
                    b_bmhr,
                    phi_re,
                    phi_im,
                    tail_bh,
                    self.rope_dim,
                    rotate_pairwise,
                );
                (b_drv, b_n, c_n, RotationState::Angle(angle_n_bha))
            }
            RotationKind::Quaternion4D => {
                let q1_bhj4 = cache.rotation.clone().quaternion(); // Q₁
                let blocks = self.num_quat_blocks;
                let rope_width = blocks * 4;
                let g_bhj3 = per_step_generator(rot_ba, dt_bh, blocks);

                // Qₙ = q^K ⊗ Q₁ ;  q^K = exp(K·g/2) (half-angle wrapped).
                let q_pow_k = quat_pow(g_bhj3.clone(), kf);
                let q_n_bhj4 = quat_normalize(quat_mul(q_pow_k.clone(), q1_bhj4.clone()));
                let conj_qn_bmhj4 = quat_conj(q_n_bhj4.clone())
                    .unsqueeze_dim::<5>(1)
                    .expand([batch, mimo_rank, nheads, blocks, 4]);
                let b_n = crate::mamba3::rotation::rotate_blocks_partial::<4, 5>(
                    b_bmhr.clone(),
                    conj_qn_bmhj4.clone(),
                    rope_width,
                );
                let c_n = crate::mamba3::rotation::rotate_blocks_partial::<4, 5>(
                    c_bmhr,
                    conj_qn_bmhj4,
                    rope_width,
                );

                // f = Q₁* ⊗ w^{K−1} ⊗ (1 − α^K q^K) ⊗ (1 − α q)⁻¹ ⊗ (β + γ w),  w = q*.
                // Everything right of Q₁* commutes (powers of one quaternion).
                let q_bhj4 = quat_from_scaled_axis::<4>(g_bhj3.clone());
                let w_bhj4 = quat_conj(q_bhj4.clone());
                let w_pow_km1 = quat_pow(-g_bhj3, kf - 1.0);
                let alpha_bh11 = alpha_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
                let alpha_k_bh11 = alpha_k_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
                let beta_bh11 = beta_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
                let gamma_bh11 = gamma_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
                let num = quat_one_minus(q_pow_k * alpha_k_bh11);
                let den_inv = quat_inv(quat_one_minus(q_bhj4 * alpha_bh11));
                let bg = quat_scalar_affine(beta_bh11, gamma_bh11, w_bhj4);
                let f_bhj4 = quat_mul(
                    quat_conj(q1_bhj4),
                    quat_mul(w_pow_km1, quat_mul(num, quat_mul(den_inv, bg))),
                );

                let b_drv = mul_quat_partial(b_bmhr, f_bhj4, tail_bh, rope_width);
                (b_drv, b_n, c_n, RotationState::Quaternion(q_n_bhj4))
            }
        };
        san(&b_drv_bmhr);
        san(&b_n_bmhr);
        san(&c_n_bmhr);
        rotation_n.sanity();

        // hₙ = α^K h₁ + Σₘ x_vals[m] ⊗ b_drv[m]   (einsum('bmhp,bmhr->bhpr')).
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let x_vals_bmhp =
            helpers::build_v_with_mimo::<3, 4>(x_bhp.clone(), mimo_x_hmp.as_ref(), 1);
        let xb_bhpr = {
            let b_bhmr = b_drv_bmhr.permute([0, 2, 1, 3]);
            let x_bhpm = x_vals_bmhp.clone().permute([0, 2, 3, 1]);
            x_bhpm.matmul(b_bhmr)
        };
        let alpha_k_bh11 = alpha_k_bh.unsqueeze_dims::<4>(&[2, 3]);
        let state_n_bhpr = alpha_k_bh11 * cache.ssm_bhpr + xb_bhpr;
        san(&state_n_bhpr);

        // yₙ readout + block tail, exactly as in `step`.
        let out_m_bmhp = Self::step_readout(state_n_bhpr.clone(), c_n_bmhr);
        san(&out_m_bmhp);
        let out_bm = self.step_finish(out_m_bmhp, x_vals_bmhp, z_bi);

        let cache = Mamba3DoubleSsdCache {
            ssm_bhpr: state_n_bhpr,
            k_state_bmhr: b_n_bmhr,
            v_state_bhp: x_bhp,
            rotation: rotation_n,
        };
        (out_bm, cache)
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

/// Component-wise complex product `(ar + i·ai)(br + i·bi)`.
fn complex_mul(
    ar: Tensor<3>,
    ai: Tensor<3>,
    br: Tensor<3>,
    bi: Tensor<3>,
) -> (Tensor<3>, Tensor<3>) {
    (
        ar.clone() * br.clone() - ai.clone() * bi.clone(),
        ai * br + ar * bi,
    )
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

/// `exp(k·g/2)` — the per-step rotation quaternion raised to the real power
/// `k`, built like [`quat_from_scaled_axis`] but with the half-angle reduced
/// mod `2π` so large `k` stays fp-accurate (the reduction is value-exact and
/// the wrap offset is detached, so `d/dg` matches the unrolled product).
fn quat_pow(g_bhj3: Tensor<4>, k: f32) -> Tensor<4> {
    let eps = div_eps(g_bhj3.dtype());
    let angle = (g_bhj3.clone() * g_bhj3.clone())
        .sum_dim(3)
        .clamp_min(eps)
        .sqrt(); // [b, h, j, 1]
    let half = wrap_angle(angle.clone() * (0.5 * k));
    let w = half.clone().cos();
    let v = g_bhj3 * (half.sin() / angle);
    quat_normalize(Tensor::cat(vec![w, v], 3))
}

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
        let head = rotate_state_rank_blocks::<4, 5>(
            x_bmhr.clone().narrow(3, 0, rope_width),
            f_bmhj4,
        );
        let tail = x_bmhr.narrow(3, rope_width, state_rank - rope_width) * tail_b1h1;
        Tensor::cat(vec![head, tail], 3)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
