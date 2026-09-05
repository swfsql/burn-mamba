//! # Mamba-3 — Single-Pass SSD Forward
//!
//! This module provides the `forward_single_ssd` method on [`Mamba3`](crate::mamba3::mamba3::Mamba3):
//! The burn-mamba implementation of the **official Mamba-3 algorithm**
//! as shipped in Triton (SISO) and Tilelang (MIMO):
//!
//! ```text
//!   scaleₜ = γₜ + (1 − λₜ₊ₗₐ₉) · Δₜ₊ₗₐ₉
//!
//!   forward_single_ssd:    h' = SSD(V_raw, K_scaled = scaleₜ B) with:
//!                               * strict lower-triangular intra-chunk mask
//!                               * additive γ-weighted same-step correction
//!                               * boundary β seed Σⱼ (1−λⱼ) Δⱼ Kⱼ ⊗ xⱼ over the
//!                                 cache's `lag` tap slots
//!                               * at lag > 1, the rest of the correction band
//!                                 ([`crate::mamba3::single_ssd::token_band`])
//! ```
//!
//! `lag` is [`Trapezoid::tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag):
//! `1` for the default tap pattern, `u` for
//! [`Trapezoid::Vertical`](crate::mamba3::trapezoid::Trapezoid::Vertical).
//!
//! References:
//! - [`mamba3_siso_fwd.py`](https://github.com/state-spaces/mamba/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py),
//! - [`mamba3_mimo_fwd.py`](https://github.com/state-spaces/mamba/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py).
//!
//! See also: [`crate::mamba3::mamba3`] and [`crate::mamba3::double_ssd::double_ssd`].

use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCache;
use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::rotate_bc_forward;
use crate::mamba3::single_ssd::prelude::*;
use burn_stack::modules::Silu;
use burn_stack::modules::sanity as san;
use burn::prelude::*;

impl Mamba3 {
    /// Process a full input sequence using the **single-ssd form (single-pass)**
    /// trapezoidal algorithm.
    ///
    /// Functionally equivalent to [`Self::forward`] but uses approximately half
    /// the SSD memory during training. Cache is a separate type
    /// ([`Mamba3SingleSsdCache`]) because the stored hidden state has different
    /// semantics than the original-form cache used by [`Self::forward`].
    ///
    /// "Equivalent" is over **everything a caller can observe** — the returned
    /// output and every field of the returned cache — which is the whole of what
    /// this method is. It is not a claim about the intermediate `y` the SSD core
    /// produces at each *folded* position: under
    /// [`Trapezoid::Vertical`] the
    /// correction band is only applied at each token's last micro-step, the one
    /// the readout happens at, so the `u−1` per-token partial sums this pathway
    /// discards do not match the double-SSD ones. Nothing reads them, and the
    /// *state* is exact at every position in both pathways — see
    /// [`crate::mamba3::single_ssd::token_band`], which is also why that band
    /// never has to enter the kernel.
    ///
    /// Except under [`Trapezoid::None`], where there is no second pass to fuse:
    /// the composite key scale is `γ` and the same-step correction is the whole
    /// diagonal, so this *is* the double-SSD form — `h' ≡ h` at every position,
    /// not merely at boundaries — and the call delegates rather than run a
    /// strict-mask kernel plus a correction that reassembles what it masked.
    ///
    /// # Shapes
    /// - `input_bsm`: `[batch, sequence, d_model]`
    /// - output: `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward_single_ssd(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3SingleSsdCache>,
        ssd_path: &Mamba3SsdPath,
    ) -> (Tensor<3>, Mamba3SingleSsdCache) {
        if !self.trapezoid.has_beta_tap() {
            let (out_bsm, cache) =
                self.forward_double_ssd(input_bsm, cache.map(Into::into), ssd_path);
            return (out_bsm, cache.into());
        }
        let [batch, tokens, _d_model] = input_bsm.dims();
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let micro_steps = self.micro_steps;
        let device = input_bsm.device();

        // MambaProduct: from the split below down to the readout, one sequence
        // position *is* one micro-step — the `s` in every shape suffix counts
        // micro-steps, and `tokens` is the only name still at token resolution.
        // See [`crate::mamba3::product`].
        let sequence = tokens * micro_steps;

        assert!(tokens > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        let mut cache = cache.unwrap_or_else(|| {
            let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
            // Reached only with a β tap (`Trapezoid::None` delegated above).
            let (k_state_bumhr, v_state_buhp) = self.zero_tap_slots(batch, &device);
            let rotation = self.zero_rotation_state(batch, &device);
            Mamba3SingleSsdCache {
                ssm_bhpr,
                k_state_bumhr,
                v_state_buhp,
                rotation,
            }
        });

        // ── Step 1: In-projection ─────────────────────────────────────────────
        let proj_bsd = self.in_proj.forward(input_bsm);
        let bc_size = ngroups * state_rank * mimo_rank;

        // `u` = micro_steps widens every per-micro-step segment, and `unfold`
        // reinterprets each of them as `u` consecutive sequence positions. `z`
        // (per-token gate) and `C` (per-token read) do not widen — `C` is
        // instead broadcast across the group so its *last* copy carries the
        // right cumulative rotation. See [`crate::mamba3::product`].
        // The optional segments come off the tail first, in layout order: the
        // rotation (`Real1D` projects none) and then `λ` — which is always
        // present here, `Trapezoid::None` having delegated above — since a
        // zero-width segment would silently vanish from the split below.
        let u = micro_steps;
        let (proj_bsd, rot_btA) =
            helpers::split_trailing(proj_bsd, self.rotation_channels_total(), 2);
        let (proj_bsd, lambda_raw_btH) =
            helpers::split_trailing(proj_bsd, self.lambda_channels_total(), 2);
        let lambda_raw_btH = lambda_raw_btH.expect("a β tap projects λ");
        #[rustfmt::skip]
        let [
                z_bsi, x_btI,
                b_raw_btMGRU, c_raw_btMGR,
                dd_dt_btH, dd_A_raw_btH,
        ] = burn_stack::modules::split_into(
            proj_bsd,
            [
                d_inner, u * d_inner,
                u * bc_size, bc_size,
                u * nheads, u * nheads,
            ],
            2,
        );

        use crate::mamba3::product::{repeat_micro_bs, unfold_micro_bs};
        let x_bsi = unfold_micro_bs(x_btI, u);
        let b_raw_bsMGR = unfold_micro_bs(b_raw_btMGRU, u);
        let c_raw_bsMGR = repeat_micro_bs(c_raw_btMGR, u);
        let dd_dt_bsh = unfold_micro_bs(dd_dt_btH, u);
        let dd_A_raw_bsh = unfold_micro_bs(dd_A_raw_btH, u);
        let lambda_raw_bsh = unfold_micro_bs(lambda_raw_btH, u);
        let rot_bsa = rot_btA.map(|t| unfold_micro_bs(t, u));

        san(&z_bsi);
        san(&x_bsi);
        san(&dd_dt_bsh);

        // ── Step 2: Discretisation + trapezoidal coefficients ─────────────────
        let helpers::TrapezoidCoeffs {
            dt: dt_bsh,
            da: da_bsh,
            alpha: _alpha_bsh,
            beta: _beta_bsh,
            gamma: gamma_bsh,
        } = helpers::trapezoidal_coefficients(
            dd_dt_bsh,
            dd_A_raw_bsh,
            Some(lambda_raw_bsh.clone()),
            self.dt_bias_h.val(),
            self.dt_limit,
            self.a_floor,
        );
        san(&dt_bsh);
        san(&da_bsh);
        san(&gamma_bsh);

        // ── Compute scaleₜ = γₜ + (1 − λₜ₊ₗₐ₉) · Δₜ₊ₗₐ₉ ──────────────────────
        //
        // The shifted term is zero for the last `lag` sequence positions (the
        // taps that pay them belong to the *next* call, out of the tap slots) —
        // which is also what makes `h'` coincide with the double-SSD state at a
        // cache boundary, hence the field-identity `From` impls.
        //
        // `t+lag` is a later *folded* position: lag 1 is
        // [`Trapezoid::HorizontalCarryOver`], lag `u` is [`Trapezoid::Vertical`].
        // This is the `Δ̃` collapse (`info/trapezoid-as-integration.md` §5), and
        // §9's collapse theorem is why it survives the wider lag unchanged —
        // only the same-step correction widens from the diagonal to a `lag`-wide
        // band (see [`crate::mamba3::single_ssd::token_band`]).
        let lag = self.tap_lag();
        let lambda_bsh = burn::tensor::activation::sigmoid(lambda_raw_bsh);
        // νₜ = (1 − λₜ)·Δₜ, the tap's own coefficient before any transport.
        let nu_bsh = dt_bsh.clone() * (-lambda_bsh + 1.0);
        let shifted_gamma_bsh = {
            let zero_bLh = Tensor::zeros([batch, lag, nheads], &device);
            if sequence == lag {
                zero_bLh
            } else {
                Tensor::cat(
                    vec![nu_bsh.clone().narrow(1, lag, sequence - lag), zero_bLh],
                    1,
                )
            }
        };
        let scale_bsh = gamma_bsh.clone() + shifted_gamma_bsh.clone();
        san(&scale_bsh);

        // ── Step 3: Reshape x ─────────────────────────────────────────────────
        let x_bshp = x_bsi.reshape([batch, sequence, nheads, per_head_dim]);

        // ── Step 4: QK-Norm on B and C ────────────────────────────────────────
        let b_bsmhr = helpers::qk_norm_expand_bias::<5, 6>(
            b_raw_bsMGR.reshape([batch, sequence, mimo_rank, ngroups, state_rank]),
            &self.b_norm,
            self.b_bias_hmr.val(),
            3,
            nheads,
        );
        let c_bsmhr = helpers::qk_norm_expand_bias::<5, 6>(
            c_raw_bsMGR.reshape([batch, sequence, mimo_rank, ngroups, state_rank]),
            &self.c_norm,
            self.c_bias_hmr.val(),
            3,
            nheads,
        );

        // ── Step 5: Data-dependent transition rotation of B and C ─────────────
        // Complex2D: abelian RoPE (cumulative angle). Quaternion4D: cumulative
        // unit quaternion. Shared with the double-ssd pathway via
        // [`rotate_bc_forward`]; the single-pass SSD core below is
        // rotation-agnostic — it only ever consumes the rotated B̄/C̄ (the RoPE
        // factoring `C̄ₜᵀB̄ᵢ = Cₜᵀ·Rel(t,i)·Bᵢ` holds for either algebra).
        let (b_bsmhr, c_bsmhr, new_rotation) = rotate_bc_forward(
            rot_bsa,
            dt_bsh.clone(),
            cache.rotation.clone(),
            b_bsmhr,
            c_bsmhr,
            self.rotation_spec(),
        );
        san(&b_bsmhr);
        san(&c_bsmhr);

        // ── Save the last `lag` positions' B and x (raw, no MIMO_V) ───────────
        // The positions whose second installment the next call pays; at
        // `lag = u` that window is precisely the last token.
        let (b_last_bumhr, x_last_buhp) =
            self.save_tap_slots(&b_bsmhr, &x_bshp, &da_bsh, lag);

        // ── Boundary β seed for initial state ─────────────────────────────────
        // Add Σⱼ νⱼ · Σₘ K_prev[j, m] ⊗ (x_prev[j] ⊙ mimo_xₘ) to the carried
        // single-ssd SSM state: cache slot `j` (oldest first) is the position
        // whose tap is paid by this call's position `j`, so it takes `νⱼ` from
        // the current call's first `lag` positions. `x_prev` already carries the
        // decay from its own position to the boundary, and `K_prev` its own
        // rotation, so the pair *is* the transported write.
        //
        // γₜ = λₜ·Δₜ, so νⱼ = (1−λⱼ)·Δⱼ = Δⱼ − γⱼ.
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let nu_head_buh = nu_bsh.clone().narrow(1, 0, lag);
        let v_prev_mimo_bumhp = helpers::build_v_with_mimo::<4, 5>(
            cache
                .v_state_buhp
                .clone()
                .expect("a β tap keeps its (B, x) cache slots"),
            mimo_x_hmp.as_ref(),
            2,
        ); // [batch, lag, mimo_rank, nheads, per_head_dim]
        let v_prev_mimo_bumhp = v_prev_mimo_bumhp * nu_head_buh.unsqueeze_dims::<5>(&[2, 4]);
        // The seed contracts over the slots *and* the ranks — both are just
        // outer products sharing one state — so fusing them makes it the one
        // `mimo_outer_sum` the per-token write already uses.
        let k_prev_bumhr = cache
            .k_state_bumhr
            .clone()
            .expect("a β tap keeps its (B, x) cache slots");
        let boundary_seed_bhpr = helpers::mimo_outer_sum(
            v_prev_mimo_bumhp.reshape([batch, lag * mimo_rank, nheads, per_head_dim]),
            k_prev_bumhr.reshape([batch, lag * mimo_rank, nheads, state_rank]),
            self.use_siso_decode_kernels(),
        );
        let initial_state_bhpr = cache.ssm_bhpr.clone() + boundary_seed_bhpr;
        san(&initial_state_bhpr);

        // ── Step 6: Pad sequence to multiple of chunk_len ─────────────────────
        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        // V passed to SSD is raw x with MIMO_V applied (not γ-scaled).
        let v_bshmp = helpers::build_v_with_mimo::<4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);
        // v_bshmp has axis order [b, s, m, h, p] (insert_dim=2 onto [b,s,h,p]).

        // ── The lag-`u` correction band ───────────────────────────────────────
        // At lag 1 the key scale is wrong only on the diagonal and `ssd/diag.rs`
        // fixes it inside the kernel. At lag `u` the exception is `u` wide — and
        // is exactly the token at the positions the readout keeps, so it is one
        // contraction here rather than a wider mask. See
        // [`crate::mamba3::single_ssd::token_band`].
        let band_correction_btmhp = (lag > 1)
            .then(|| {
                crate::mamba3::single_ssd::token_band::token_band_correction(
                    v_bshmp.clone(),
                    b_bsmhr.clone(),
                    c_bsmhr.clone(),
                    shifted_gamma_bsh,
                    da_bsh.clone(),
                    micro_steps,
                )
            })
            .flatten();

        #[rustfmt::skip]
        let (v_bShmp, da_bSh, gamma_bSh, scale_bSh, b_bSmhr, c_bSmhr) = if pad == 0 {
            (v_bshmp, da_bsh, gamma_bsh, scale_bsh, b_bsmhr, c_bsmhr)
        } else {
            let pad_bShmp = Tensor::zeros([batch, pad, mimo_rank, nheads, per_head_dim], &device);
            let pad_bSh = Tensor::zeros([batch, pad, nheads], &device);
            let pad_bSmhr = Tensor::zeros([batch, pad, mimo_rank, nheads, state_rank], &device);
            (
                Tensor::cat(vec![v_bshmp, pad_bShmp], 1),
                Tensor::cat(vec![da_bsh, pad_bSh.clone()], 1),
                Tensor::cat(vec![gamma_bsh, pad_bSh.clone()], 1),
                Tensor::cat(vec![scale_bsh, pad_bSh], 1),
                Tensor::cat(vec![b_bsmhr, pad_bSmhr.clone()], 1),
                Tensor::cat(vec![c_bsmhr, pad_bSmhr], 1),
            )
        };

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let v_bnlmhp =
            v_bShmp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let gamma_bnlh = gamma_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let scale_bnlh = scale_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bnlmhr = c_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

        // ── Step 7: Run single-pass form SSD ───────────────────────────────────────
        let ssd_input = Mamba3SingleSsdInput {
            v_bnlmhp,
            b_bnlmhr,
            c_bnlmhr,
            da_bnlh,
            gamma_bnlh,
            scale_bnlh,
            initial_state_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
            siso_specialization: self.siso_specialization,
        };
        let (y_bnlmhp, final_state_bhpr) = ssd_input.run(ssd_path);

        san(&y_bnlmhp);
        san(&final_state_bhpr);
        cache.ssm_bhpr = final_state_bhpr;

        // ── Step 8: Unpad ─────────────────────────────────────────────────────
        let y_bSmhp = y_bnlmhp.reshape([batch, sequence_padded, mimo_rank, nheads, per_head_dim]);
        let y_bsmhp = if pad == 0 {
            y_bSmhp
        } else {
            y_bSmhp.narrow(1, 0, sequence)
        };

        // ── Step 8b: back to token resolution (MambaProduct) ──────────────────
        // The readout happens after all `u` writes, so only each token's last
        // micro-step survives; the intervening positions computed a `y` from a
        // repeated `C` and it is dropped here.
        let y_bsmhp = crate::mamba3::product::last_micro5(y_bsmhp, micro_steps);
        let y_bsmhp = match band_correction_btmhp {
            Some(correction_btmhp) => y_bsmhp - correction_btmhp,
            None => y_bsmhp,
        };
        let x_bthp = crate::mamba3::product::last_micro4(x_bshp.clone(), micro_steps);
        let sequence = tokens;

        // ── Step 9: D skip + gate + MIMO_O down-projection ────────────────────
        // D skip uses raw x ⊙ mimo_x (not γ-scaled, matching forward), at the
        // micro-step the readout is contemporaneous with.
        let v_raw_bsmhp = helpers::build_v_with_mimo::<4, 5>(x_bthp, mimo_x_hmp.as_ref(), 2);
        let d_111h1 = self.d_h.val().unsqueeze_dims::<5>(&[0, 1, 2, 4]);
        let y_bsmhp = y_bsmhp + d_111h1 * v_raw_bsmhp;

        let y_bsi = if mimo_rank > 1 {
            let mimo_z_hmp = self.mimo_z_hmp.as_ref().map(|p| p.val()).unwrap();
            let mimo_o_hmp = self.mimo_o_hmp.as_ref().map(|p| p.val()).unwrap();

            let z_bshp = z_bsi
                .clone()
                .reshape([batch, sequence, nheads, per_head_dim]);
            let z_bsmhp = {
                let z_bsmhp = z_bshp.unsqueeze_dim::<5>(2).expand([
                    batch,
                    sequence,
                    mimo_rank,
                    nheads,
                    per_head_dim,
                ]);
                let mimo_z_bsmhp = mimo_z_hmp
                    .swap_dims(0, 1)
                    .unsqueeze_dims::<5>(&[0, 1])
                    .expand([batch, sequence, mimo_rank, nheads, per_head_dim]);
                z_bsmhp * mimo_z_bsmhp
            };

            let y_combined_bsmhp = match &self.out_norm {
                Some(norm) => norm.forward(y_bsmhp, z_bsmhp),
                None => y_bsmhp * Silu::new().forward(z_bsmhp),
            };

            let mimo_o_bsmhp = mimo_o_hmp
                .swap_dims(0, 1)
                .unsqueeze_dims::<5>(&[0, 1])
                .expand([batch, sequence, mimo_rank, nheads, per_head_dim]);
            let y_bshp: Tensor<4> = (y_combined_bsmhp * mimo_o_bsmhp).sum_dim(2).squeeze_dim(2);
            y_bshp.reshape([batch, sequence, d_inner])
        } else {
            let y_bshp: Tensor<4> = y_bsmhp.squeeze_dim(2);
            let z_bshp = z_bsi.reshape([batch, sequence, nheads, per_head_dim]);
            let y_combined_bshp = match &self.out_norm {
                Some(norm) => norm.forward(y_bshp, z_bshp),
                None => y_bshp * Silu::new().forward(z_bshp),
            };
            y_combined_bshp.reshape([batch, sequence, d_inner])
        };
        san(&y_bsi);

        // ── Out-projection ────────────────────────────────────────────────────
        let out_bsm = self.out_proj.forward(y_bsi);
        san(&out_bsm);

        // ── Update remaining cache fields ─────────────────────────────────────
        cache.k_state_bumhr = b_last_bumhr;
        cache.v_state_buhp = x_last_buhp;
        // The new cumulative rotation (Complex2D: angle wrapped to [−π, π];
        // Quaternion4D: the cumulative quaternion), from [`rotate_bc_forward`] —
        // matches the double-ssd cache convention so the two inter-convert.
        cache.rotation = new_rotation;

        (out_bsm, cache)
    }
}

// ---------------------------------------------------------------------------
// Mamba3::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    impl Mamba3 {
        /// Process a **single token** using the pure recurrent form.
        ///
        /// For SISO (mimo_rank=1):
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ
        ///   yₜ = Cₜᵀ hₜ + D xₜ
        /// ```
        ///
        /// For MIMO (mimo_rank>1):
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + Σₘ βₜ Bₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_x_hmp[m]) + Σₘ γₜ Bₜ[m] ⊗ (xₜ ⊙ mimo_x_hmp[m])
        ///   yₜ[r] = Cₜ[r]ᵀ hₜ + D xₜ ⊙ mimo_x_hmp[r]
        ///   outₜ = Σₘ mimo_o_hmp[m] ⊙ silu(zₜ ⊙ mimo_z_hmp[m]) ⊙ yₜ[m]
        /// ```
        ///
        /// # Shapes
        /// - `input_bd` : `[batch, d_model]`
        /// - output     : `[batch, d_model]`
        #[allow(non_snake_case)]
        pub fn step_single_ssd(
            &self,
            input_bd: Tensor<2>,
            cache: Option<Mamba3SingleSsdCache>,
        ) -> (Tensor<2>, Mamba3SingleSsdCache) {
            // Token-by-token decoding always uses the recurrent (double-ssd)
            // form. A single-ssd cache holds the trapezoid state at a sequence
            // boundary, where the single- and double-ssd accumulators coincide
            // (see the `From` impls in `crate::mamba3::cache`), so converting in
            // and back out is lossless. The single recurrence step is itself a
            // boundary-to-boundary transition, so the round-trip stays exact.
            let cache = cache.map(Mamba3DoubleSsdCache::from);
            let (out_bd, cache) = self.step_double_ssd(input_bd, cache);
            (out_bd, cache.into())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — forward_single_ssd parity with forward_double_ssd, step, and split-prefill
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
