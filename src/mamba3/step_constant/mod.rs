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
//! ## MambaProduct (`micro_steps = u > 1`)
//!
//! A constant token still drives `u` *different* micro-steps, so the per-token
//! map is a product `A·P = (∏ⱼ αⱼ)·P_{u−1}⋯P₀` and the token writes `u` pairs
//! `(bⱼ, xⱼ)` at `u` different points along it. Collecting the trapezoid's two
//! taps by which pair they write gives one weight per pair,
//!
//! ```text
//!   aⱼ = ∏_{j'>j} α_{j'} ,   cⱼ = aⱼγⱼ + a_{j+1}β_{j+1}   (j < u−1)
//! ```
//!
//! and the last pair keeps its taps apart, because its β partner is the *next*
//! token's micro-step 0 and therefore sits one turn further back in the same
//! geometric series (`γ_{u−1} + a₀β₀ P⁻¹`). The limit is then a sum of `u` terms
//! of exactly the shape above:
//!
//! ```text
//!   y_∞ = Σⱼ xⱼᵀ · cᵀ Wⱼ bⱼ + D x_{u−1} ,
//!   Wⱼ = cⱼ · Qⱼ P⁻¹ (I − A P⁻¹)⁻¹  (j < u−1) ,   Q_j = Pⱼ⋯P₀
//!   W_{u−1} = (γ_{u−1} + a₀β₀ P⁻¹)(I − A P⁻¹)⁻¹
//! ```
//!
//! `Qⱼ P⁻¹` is the rotation still *to come* after micro-step `j`, and at `u = 1`
//! the whole thing collapses back to the single term above.
//!
//! **This exists only for the abelian kinds.** The readout at token `t` sees the
//! pair written at token `t−n` through `P⁻ᵗ Qⱼ Pᵗ⁻ⁿ⁻¹`, which is a function of
//! `n` alone iff `Qⱼ` commutes with `P`. For
//! [`Quaternion4D`](crate::mamba3::rotation::RotationKind::Quaternion4D) and
//! [`Rotor4D`](crate::mamba3::rotation::RotationKind::Rotor4D) it does not: the
//! conjugation keeps turning with `t` and the output is almost-periodic, not
//! convergent. [`Mamba3::step_infinite`] asserts against that combination —
//! there is no limit to return, which is a property of the recurrence and not a
//! gap here.
//!
//! Numerics: the rotation angle is reduced mod `2π` with the value-exact
//! [`wrap_angle`]; the geometric denominators satisfy
//! `|1 − α e^{−iθ̂}|² ≥ (1 − α)²` and are floored by
//! [`div_eps`](burn_stack::utils::div_eps). When `α → 1` *and* `θ̂ → 0` the series
//! value `(β+γ)/(1−α)` stays finite but loses fp32 precision once `1 − α` nears
//! the epsilon floor — the same regime where the unrolled recurrence itself
//! accumulates near-undamped terms.

use crate::mamba3::double_ssd::double_ssd::MicroProjection;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::{
    angle_increment, generator_increment, quat_conj, quat_from_scaled_axis, quat_mul,
    rotate_state_rank_blocks, safe_norm, split_rotor,
};
use burn_stack::modules::sanity as san;
use crate::mamba3::rotation::rope::wrap_angle;
use burn_stack::utils::div_eps;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl Mamba3 {
    /// Stationary **fixed point** of the block under a constant token: the
    /// limit of `step(input, …)` outputs as the same `input_bd` is stepped
    /// forever.
    ///
    /// Closed form, O(1) in the horizon (see the module header for the
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
        let proj = self.step_project(input_bd);
        let u = proj.micro_steps;
        let spec = self.rotation_spec();
        let [batch, mimo_rank, nheads, _state_rank] = proj.c_bmhr.dims();
        assert!(
            u == 1 || matches!(spec.kind, RotationKind::Real1D | RotationKind::Complex2D),
            "step_infinite has no limit to return at micro_steps = {u} with {:?}: the \
             per-token rotation product `P` and its partial products `Qⱼ` do not commute, so \
             the read-to-write relative rotation `P⁻ᵗ Qⱼ Pᵗ⁻ⁿ⁻¹` keeps turning with `t` and \
             the output is almost-periodic rather than convergent. Use micro_steps = 1, or an \
             abelian RotationKind (Real1D / Complex2D). See `crate::mamba3::product`.",
            spec.kind
        );

        let micros: Vec<MicroProjection> = (0..u).map(|j| proj.micro(j)).collect();
        let eps = div_eps(micros[0].alpha_bh.dtype());

        // Suffix decay `aⱼ = ∏_{j'>j} α_{j'}` — how much a write at micro-step
        // `j` has decayed by the token's end — and `A = ∏ⱼ αⱼ`, the token's own
        // decay. `a_{u−1} = 1`: the last micro-step's write is not decayed again
        // before the readout.
        let mut suffix_bh: Vec<Tensor<2>> =
            vec![Tensor::ones([batch, nheads], &micros[0].alpha_bh.device()); u];
        for j in (0..u.saturating_sub(1)).rev() {
            suffix_bh[j] = suffix_bh[j + 1].clone() * micros[j + 1].alpha_bh.clone();
        }
        let total_decay_bh = suffix_bh[0].clone() * micros[0].alpha_bh.clone();
        let one_minus_a_bh = (-total_decay_bh.clone() + 1.0).clamp_min(eps);

        // How much of the token's write lands on micro-step `j`'s `(b, x)` pair.
        // Two taps write it: `j`'s own γ tap, and `j+1`'s β tap (the trapezoid's
        // left endpoint). The last pair is special — its β partner is the *next
        // token's* micro-step 0, which contributes one extra factor of `P⁻¹`
        // (its write sits one token later in the same geometric series), so it
        // keeps the two taps apart.
        // Under [`Trapezoid::None`] there is no left endpoint: every pair keeps
        // only its own γ, and the last one loses the wrapped term entirely.
        let pair_weight_bh = |j: usize| {
            let own = suffix_bh[j].clone() * micros[j].gamma_bh.clone();
            match micros.get(j + 1).and_then(|m| m.beta_bh.clone()) {
                Some(beta_bh) => own + suffix_bh[j + 1].clone() * beta_bh,
                None => own,
            }
        };
        // The wrapped β weight attached to the last pair, `a₀·β₀`.
        let wrap_bh = micros[0]
            .beta_bh
            .clone()
            .map(|beta_bh| suffix_bh[0].clone() * beta_bh);

        let rot = |r: Option<Tensor<2>>| r.expect("a rotating kind projects rotation channels");
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let siso = self.use_siso_decode_kernels();

        // Per-pair/block readout factor. For the last pair it is the familiar
        // `(γ + β P⁻¹)(I − A P⁻¹)⁻¹`; for the others the weight is a scalar and
        // the rotation is the *remaining* one, `Qⱼ P⁻¹ = (P_{u−1}⋯P_{j+1})⁻¹`.
        // The cumulative rotation cancels against `Cₙ` (orthogonality), so only
        // these relative turns survive.
        let b_eff = |j: usize| -> Tensor<4> {
            let b_bmhr = micros[j].b_bmhr.clone();
            let last = j + 1 == u;
            // Rotation-free channels: the plain real geometric series.
            let tail_num_bh = match (last, wrap_bh.clone()) {
                (true, Some(wrap_bh)) => pair_weight_bh(j) + wrap_bh,
                _ => pair_weight_bh(j),
            };
            let tail_bh = tail_num_bh / one_minus_a_bh.clone();
            match spec.kind {
                // A real transition: every channel is `tail`.
                RotationKind::Real1D => b_bmhr * tail_bh.unsqueeze_dims::<4>(&[1, 3]),
                RotationKind::Complex2D => {
                    // The same per-step increments `forward`/`step` use — one
                    // definition, so the fixed point cannot drift from the
                    // recurrence it is the limit of. `Θ` is the token's total
                    // turn, `Φⱼ` the part of it still to come after `j`.
                    let theta = |k: usize| {
                        angle_increment::<2, 3>(
                            rot(micros[k].rot_ba.clone()),
                            micros[k].dt_bh.clone(),
                            spec.range,
                        )
                    };
                    let mut total_bha = theta(0);
                    for k in 1..u {
                        total_bha = total_bha + theta(k);
                    }
                    let (cos_t, sin_t) = cos_sin(total_bha.clone());
                    let a_bh1 = total_decay_bh.clone().unsqueeze_dim::<3>(2);
                    let den_re = -a_bh1.clone() * cos_t.clone() + 1.0;
                    let den_im = a_bh1 * sin_t.clone();
                    let (num_re, num_im) = if last {
                        // γ + (a₀β₀) e^{−iΘ}; with no β tap the numerator is the
                        // bare real γ and no zero imaginary part is formed.
                        let g_bh1 = pair_weight_bh(j).unsqueeze_dim::<3>(2);
                        match wrap_bh.clone() {
                            Some(wrap_bh) => {
                                let w_bh1 = wrap_bh.unsqueeze_dim::<3>(2);
                                (g_bh1 + w_bh1.clone() * cos_t, Some(-w_bh1 * sin_t))
                            }
                            None => (g_bh1, None),
                        }
                    } else {
                        // cⱼ e^{−iΦⱼ},  Φⱼ = Σ_{j'>j} θ̂_{j'}
                        let mut phi_bha = theta(j + 1);
                        for k in j + 2..u {
                            phi_bha = phi_bha + theta(k);
                        }
                        let (cos_p, sin_p) = cos_sin(phi_bha);
                        let c_bh1 = pair_weight_bh(j).unsqueeze_dim::<3>(2);
                        (c_bh1.clone() * cos_p, Some(-c_bh1 * sin_p))
                    };
                    let (m_re, m_im) = complex_div(num_re, num_im, den_re, den_im);
                    mul_complex_partial(b_bmhr, m_re, m_im, tail_bh, self.rope_dim, mimo_rank == 1)
                }
                // The non-abelian kinds are reachable only at `u == 1` (asserted
                // above), where `j` is the one and only pair and the weights
                // collapse to the plain `(γ, β, α)`.
                RotationKind::Rotor4D => {
                    let m = &micros[0];
                    let g_bhk3 = generator_increment::<2, 3, 4>(
                        rot(m.rot_ba.clone()),
                        m.dt_bh.clone(),
                        self.num_rotation_blocks(),
                        spec.range,
                    );
                    let qp_bhk4 = quat_from_scaled_axis::<4>(g_bhk3);
                    let (q_bhj4, p_bhj4) = split_rotor(qp_bhk4); // M(v) = q ⊗ v ⊗ p̄
                    rotor_resolvent_partial(
                        b_bmhr,
                        q_bhj4,
                        p_bhj4,
                        m.alpha_bh.clone(),
                        m.beta_bh.clone(),
                        m.gamma_bh.clone(),
                        tail_bh,
                        spec.rope_dim,
                    )
                }
                RotationKind::Quaternion4D => {
                    let m = &micros[0];
                    let g_bhj3 = generator_increment::<2, 3, 4>(
                        rot(m.rot_ba.clone()),
                        m.dt_bh.clone(),
                        self.num_quat_blocks,
                        spec.range,
                    );
                    let q_bhj4 = quat_from_scaled_axis::<4>(g_bhj3); // P⁻¹ ↔ q
                    let alpha_bh11 = m.alpha_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
                    let beta_bh11 = m
                        .beta_bh
                        .clone()
                        .map(|beta_bh| beta_bh.unsqueeze_dims::<4>(&[2, 3]));
                    let gamma_bh11 = m.gamma_bh.clone().unsqueeze_dims::<4>(&[2, 3]);
                    // (γ + β q) ⊗ (1 − α q)⁻¹  — all in the abelian subalgebra of q.
                    let den_inv = quat_inv(quat_one_minus(q_bhj4.clone() * alpha_bh11));
                    let f_bhj4 = match beta_bh11 {
                        Some(beta_bh11) => {
                            let num = quat_scalar_affine(gamma_bh11, beta_bh11, q_bhj4);
                            quat_mul(num, den_inv)
                        }
                        // A *real* numerator commutes with everything, so the
                        // quaternion product degenerates to a scalar multiply.
                        None => den_inv * gamma_bh11,
                    };
                    // As in `rotate_bc_step`, the rotated width comes from
                    // `rope_dim` (a partial rotation turns whole 4-blocks).
                    mul_quat_partial(b_bmhr, f_bhj4, tail_bh, spec.rope_dim)
                }
            }
        };

        // y_∞[m] = Σⱼ Σ_m' ⟨c[m], Wⱼ·bⱼ[m']⟩ · x_vals,ⱼ[m']  — one term per
        // micro-step, then the shared D-skip/gate/out-projection on the *last*
        // micro-step's values (the one the readout is contemporaneous with).
        let mut out_m_bmhp: Option<Tensor<4>> = None;
        let mut last_x_vals_bmhp: Option<Tensor<4>> = None;
        for j in 0..u {
            let b_eff_bmhr = b_eff(j);
            san(&b_eff_bmhr);
            let x_vals_bmhp = helpers::build_v_with_mimo::<3, 4>(
                micros[j].x_bhp.clone(),
                mimo_x_hmp.as_ref(),
                1,
            );
            let term_bmhp = if siso {
                // SISO: the Gram matrix is the scalar ⟨c, W·b⟩ per (batch,
                // nheads), so both matmuls degenerate — contract `state_rank`
                // with a reduction and broadcast over `per_head_dim`.
                let gram_bmh1: Tensor<4> = (proj.c_bmhr.clone() * b_eff_bmhr).sum_dim(3);
                x_vals_bmhp.clone() * gram_bmh1
            } else {
                let gram_bhmm = {
                    let c_bhmr = proj.c_bmhr.clone().swap_dims(1, 2);
                    let b_bhrm = b_eff_bmhr.permute([0, 2, 3, 1]);
                    c_bhmr.matmul(b_bhrm) // [batch, nheads, mimo_rank, mimo_rank']
                };
                let x_bhmp = x_vals_bmhp.clone().swap_dims(1, 2);
                gram_bhmm.matmul(x_bhmp).swap_dims(1, 2)
            };
            out_m_bmhp = Some(match out_m_bmhp {
                Some(acc) => acc + term_bmhp,
                None => term_bmhp,
            });
            last_x_vals_bmhp = Some(x_vals_bmhp);
        }
        let out_m_bmhp = out_m_bmhp.expect("micro_steps ≥ 1");
        assert_eq!(
            [batch, mimo_rank, nheads, self.per_head_dim()],
            out_m_bmhp.dims()
        );
        san(&out_m_bmhp);

        self.step_finish(
            out_m_bmhp,
            last_x_vals_bmhp.expect("micro_steps ≥ 1"),
            proj.z_bi,
        )
    }
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
/// A `None` numerator imaginary part means a **real** numerator (no `β` tap),
/// and drops the two products it would contribute rather than multiplying by a
/// materialised zero.
fn complex_div(
    nr: Tensor<3>,
    ni: Option<Tensor<3>>,
    dr: Tensor<3>,
    di: Tensor<3>,
) -> (Tensor<3>, Tensor<3>) {
    let eps = div_eps(dr.dtype());
    let d2 = (dr.clone() * dr.clone() + di.clone() * di.clone()).clamp_min(eps);
    match ni {
        Some(ni) => (
            (nr.clone() * dr.clone() + ni.clone() * di.clone()) / d2.clone(),
            (ni * dr - nr * di) / d2,
        ),
        None => {
            let scaled = nr / d2;
            (scaled.clone() * dr, -scaled * di)
        }
    }
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
    debug_assert!(rope_dim > 0, "a rotating kind turns at least one pair");
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
    debug_assert!(rope_width > 0, "a rotating kind turns at least one block");
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
// Rotor (`SO(4)`) helpers — the two-sided resolvent, per block
// ---------------------------------------------------------------------------

/// Apply `(γ + βM)(I − αM)⁻¹` to the rotation-active entries of `x`, where
/// `M(v) = q ⊗ v ⊗ p̄` is the per-step `SO(4)` rotation of
/// [`RotationKind::Rotor4D`], and scale the pass-through entries by the real
/// scalar `tail` — the two-sided analogue of [`mul_quat_partial`].
///
/// Two-sided, the geometric series no longer lives in a commutative subalgebra
/// (that shortcut is what [`quat_inv`] exploits for `Quaternion4D`), so the
/// resolvent is taken over `ℝ⁴` by **Cayley–Hamilton**. `M` is orthogonal with
/// characteristic polynomial `λ⁴ − c₁λ³ + c₂λ² − c₁λ + 1`, and its two plane
/// angles are `a∓b` for the half-angles `a`, `b` of `q` and `p` (the standard
/// `SO(4) ≅ (SU(2)×SU(2))/±1` decomposition — the axes fix the invariant
/// *planes*, only the half-angles fix the *angles*), so with
/// `k₁ = cos(a−b)`, `k₂ = cos(a+b)`:
///
/// ```text
///   c₁ = 2(k₁ + k₂) = tr M ,   c₂ = 2 + 4k₁k₂ ,
///   det(I − αM) = (1 − 2αk₁ + α²)(1 − 2αk₂ + α²) ,
///   (I − αM)⁻¹  = (e₀ + e₁M + e₂M² + e₃M³) / det(I − αM)
/// ```
///
/// with `e₃ = α³`, `e₂ = α² − c₁α³`, `e₁ = α − c₁α² + c₂α³`,
/// `e₀ = 1 − c₁α + c₂α² − c₁α³` (the divided-difference form
/// `q(λ) = (χ(μ)−χ(λ))/(μ−λ)` at `μ = 1/α`, rescaled). The cubic is evaluated
/// by Horner **on the vector**, so no `4×4` matrix is ever materialised: each
/// application of `M` is two quaternion products. `k₁ = k₂ = 1` (no rotation)
/// collapses the whole thing to `(β+γ)/(1−α)`, i.e. `tail`.
///
/// Numerics: the determinant is formed factor-wise as `(1−α)² + 2α(1−kᵢ)`,
/// which is manifestly `≥ (1−α)²` and free of the cancellation the expanded
/// quartic would suffer; the numerator carries the same `α → 1` precision
/// caveat as the abelian branch (see the module header).
#[allow(clippy::too_many_arguments)]
fn rotor_resolvent_partial(
    x_bmhr: Tensor<4>,
    q_bhj4: Tensor<4>,
    p_bhj4: Tensor<4>,
    alpha_bh: Tensor<2>,
    beta_bh: Option<Tensor<2>>,
    gamma_bh: Tensor<2>,
    tail_bh: Tensor<2>,
    rope_width: usize,
) -> Tensor<4> {
    let [batch, mimo_rank, nheads, state_rank] = x_bmhr.dims();
    let blocks = q_bhj4.dims()[2];
    let tail_b1h1 = tail_bh.unsqueeze_dims::<4>(&[1, 3]);
    debug_assert!(rope_width > 0, "a rotating kind turns at least one block");
    assert_eq!(
        rope_width,
        blocks * 4,
        "the rotated width is a whole number of quaternion blocks"
    );
    let eps = div_eps(alpha_bh.dtype());

    // Per-block plane cosines. `Re q = cos a`, `‖Im q‖ = |sin a|`; a sign flip
    // of `sin a` only swaps k₁ ↔ k₂, and every quantity below is symmetric in
    // the two, so the absolute value costs nothing.
    let per_block = |t: Tensor<4>| t.unsqueeze_dim::<5>(1); // [b,h,J,1] → [b,1,h,J,1]
    let wc = per_block(q_bhj4.clone().narrow(3, 0, 1) * p_bhj4.clone().narrow(3, 0, 1));
    let ss = per_block(
        safe_norm(q_bhj4.clone().narrow(3, 1, 3)) * safe_norm(p_bhj4.clone().narrow(3, 1, 3)),
    );
    let k1 = wc.clone() + ss.clone();
    let k2 = wc - ss;

    // Scalars, broadcast over (mimo, block, component).
    let a = alpha_bh.unsqueeze_dims::<5>(&[1, 3, 4]);
    let beta = beta_bh.map(|beta_bh| beta_bh.unsqueeze_dims::<5>(&[1, 3, 4]));
    let gamma = gamma_bh.unsqueeze_dims::<5>(&[1, 3, 4]);
    let a2 = a.clone() * a.clone();
    let a3 = a2.clone() * a.clone();
    let c1 = (k1.clone() + k2.clone()) * 2.0;
    let c2 = k1.clone() * k2.clone() * 4.0 + 2.0;
    let one_minus_a2 = (-a.clone() + 1.0) * (-a.clone() + 1.0);
    let det = (one_minus_a2.clone() + (-k1 + 1.0) * a.clone() * 2.0)
        * (one_minus_a2 + (-k2 + 1.0) * a.clone() * 2.0);
    let det = det.clamp_min(eps);
    let e3 = a3.clone();
    let e2 = a2.clone() - c1.clone() * a3.clone();
    let e1 = a.clone() - c1.clone() * a2.clone() + c2.clone() * a3.clone();
    let e0 = -c1.clone() * a + c2 * a2 - c1 * a3 + 1.0;

    // M(v) = q ⊗ v ⊗ p̄, on the block-split view of the rotated entries.
    let expand = |q: Tensor<4>| {
        q.unsqueeze_dim::<5>(1)
            .expand([batch, mimo_rank, nheads, blocks, 4])
    };
    let ql = expand(q_bhj4);
    let qr = expand(quat_conj(p_bhj4));
    let m = |v: Tensor<5>| quat_mul(quat_mul(ql.clone(), v), qr.clone());

    let head = x_bmhr
        .clone()
        .narrow(3, 0, rope_width)
        .reshape([batch, mimo_rank, nheads, blocks, 4]);
    // Horner on the vector: u = (e₀ + e₁M + e₂M² + e₃M³) x.
    let u = head.clone() * e3;
    let u = m(u) + head.clone() * e2;
    let u = m(u) + head.clone() * e1;
    let u = m(u) + head * e0;
    // With no β tap the second term — and the extra pair of quaternion products
    // `m(u)` it needs — does not exist.
    let out = match beta {
        Some(beta) => (u.clone() * gamma + m(u) * beta) / det,
        None => u * gamma / det,
    };
    let out = out.reshape([batch, mimo_rank, nheads, rope_width]);

    if rope_width == state_rank {
        out
    } else {
        let tail = x_bmhr.narrow(3, rope_width, state_rank - rope_width) * tail_b1h1;
        Tensor::cat(vec![out, tail], 3)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
