//! # Mamba-3 — Single-Pass SSD Forward
//!
//! `forward_single_ssd` and `step_single_ssd` on
//! [`Mamba3`](crate::mamba3::mamba3::Mamba3): the **official Mamba-3
//! algorithm**, as in its Triton (SISO) and Tilelang (MIMO) kernels:
//!
//! ```text
//!   scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)
//!
//!   forward_single_ssd:    h' = SSD(V_raw, K_scaled = scaleₜ B) with:
//!                               * strict lower-triangular intra-chunk mask
//!                               * additive γ-weighted same-step correction
//!                               * boundary β seed Σⱼ νⱼ Kⱼ ⊗ xⱼ over the
//!                                 cache's `lag` tap slots
//!                               * at lag > 1, the rest of the correction band
//!                                 ([`crate::mamba3::single_ssd::token_band`])
//! ```
//!
//! `lag` is [`Trapezoid::tap_lag`](crate::mamba3::trapezoid::Trapezoid::tap_lag):
//! `1` for the default tap pattern, `u` for
//! [`Trapezoid::Vertical`](crate::mamba3::trapezoid::Trapezoid::Vertical). A
//! two-tap pattern adds the term in parentheses: a second shift of the same
//! shape, because the collapse of §9 leaves one scalar per sample for any
//! number of taps. That is all it costs here: one pass either way.
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
    /// Process a full input sequence with the **single-ssd (single-pass)**
    /// trapezoidal algorithm.
    ///
    /// Functionally equivalent to [`Self::forward_double_ssd`], with about half
    /// the SSD memory in training. The cache is a separate type
    /// ([`Mamba3SingleSsdCache`]), because its hidden state has different
    /// semantics mid-sequence.
    ///
    /// "Equivalent" covers **everything that a caller can observe**: the
    /// returned output and every field of the returned cache. It does not cover
    /// the intermediate `y` of the SSD core at each *folded* position. Under a
    /// lag-`u` pattern such as [`Trapezoid::Vertical`], the correction band
    /// applies only at the last micro-step of each token (the readout). So the
    /// `u−1` per-token partial sums that this pathway discards do not match the
    /// double-SSD ones. Nothing reads them, and the *state* is exact at every
    /// position in both pathways. See [`crate::mamba3::single_ssd::token_band`],
    /// which also tells why that band never has to enter the kernel.
    ///
    /// Under [`Trapezoid::None`] there is no second pass to fuse. The composite
    /// key scale is `γ`, and the same-step correction is the whole diagonal. So
    /// this *is* the double-SSD form (`h' ≡ h` at every position, not only at
    /// boundaries), and the call delegates to `forward_double_ssd`. It does not
    /// run a strict-mask kernel plus a correction that puts back what the mask
    /// removed.
    ///
    /// `pad_bt` marks a right-padded batch of tokens, as in [`Self::forward`].
    /// The last tap installment of a real position, which a padded position
    /// would pay, then stays in the tap slots, exactly as at a call boundary.
    /// So `h'` holds the double-SSD state from the end of each slot on.
    ///
    /// # Shapes
    /// - `input_bsm`: `[batch, sequence, d_model]`
    /// - `pad_bt`: `[batch, sequence]`
    /// - output: `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward_single_ssd(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3SingleSsdCache>,
        ssd_path: &Mamba3SsdPath,
        pad_bt: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Mamba3SingleSsdCache) {
        if !self.trapezoid.has_beta_tap() {
            let (out_bsm, cache) =
                self.forward_double_ssd(input_bsm, cache.map(Into::into), ssd_path, pad_bt);
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
        // position *is* one micro-step. The `s` in every shape suffix counts
        // micro-steps, and `tokens` (`t`) is at token resolution. See
        // [`crate::mamba3::product`].
        let sequence = tokens * micro_steps;

        assert!(tokens > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // A padded token pads all of its micro-steps. `end_b` is the real
        // length of each slot on the folded axis, where its cache fields are
        // read.
        let pad_bs = pad_bt.map(|pad_bt| crate::padding::repeat_rows(pad_bt, micro_steps));
        let end_b = pad_bs.as_ref().map(crate::padding::real_len_b);

        // ── Initialise cache if not provided ──────────────────────────────────
        let mut cache = cache.unwrap_or_else(|| {
            let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
            // Reached only with a β tap (`Trapezoid::None` delegated above).
            let (k_state_bumhr, v_state_buhp) = self.zero_tap_slots(batch, &device);
            let rotation = self.zero_rotation_state(batch, &device);
            let (log_precision_bh, tropical_bh) = self.zero_positive_state(batch, &device);
            Mamba3SingleSsdCache {
                ssm_bhpr,
                k_state_bumhr,
                v_state_buhp,
                rotation,
                log_precision_bh,
                tropical_bh,
            }
        });

        // ── Step 1: In-projection ─────────────────────────────────────────────
        let proj_bsd = self.project_in(input_bsm);
        // The segments of the positive systems are the outermost tail
        // ([`crate::mamba3::positive`]). The rest is the stock layout.
        let (proj_bsd, noise_btH, tropical_btH) = self.split_positive(proj_bsd, 2);
        let bc_size = ngroups * state_rank * mimo_rank;

        // `u` = micro_steps widens every per-micro-step segment, and `unfold`
        // reads each of them as `u` consecutive sequence positions. `z`
        // (per-token gate) and `C` (per-token read) do not widen. See
        // [`crate::mamba3::product`].
        // The optional segments come off the tail first, in reverse layout
        // order: the rotation (`Real1D` projects none), then `μ` (only a
        // two-tap pattern projects it), then `λ`. `λ` is always present here,
        // because `Trapezoid::None` delegated above. A zero-width segment would
        // silently vanish from the split below.
        let u = micro_steps;
        let (proj_bsd, rot_btA) =
            helpers::split_trailing(proj_bsd, self.rotation_channels_total(), 2);
        let (proj_bsd, mu_raw_btH) = helpers::split_trailing(proj_bsd, self.mu_channels_total(), 2);
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

        use crate::mamba3::product::unfold_micro_bs;
        let x_bsi = unfold_micro_bs(x_btI, u);
        let b_raw_bsMGR = unfold_micro_bs(b_raw_btMGRU, u);
        // `C` stays at token resolution into the SSD. The readout is once per
        // token, at its last micro-step, and that is the only position where
        // the kernel computes a `y`. See [`Mamba3SingleSsdInput::c_bntmhr`].
        let dd_dt_bsh = unfold_micro_bs(dd_dt_btH, u);
        let dd_A_raw_bsh = unfold_micro_bs(dd_A_raw_btH, u);
        let lambda_raw_bsh = unfold_micro_bs(lambda_raw_btH, u);
        let mu_raw_bsh = mu_raw_btH.map(|t| unfold_micro_bs(t, u));
        let rot_bsa = rot_btA.map(|t| unfold_micro_bs(t, u));
        let noise_bsh = noise_btH.map(|t| unfold_micro_bs(t, u));

        san(&z_bsi);
        san(&x_bsi);
        san(&dd_dt_bsh);

        // ── Step 2: Discretisation + trapezoidal coefficients ─────────────────
        // Under a Kalman gain, this call computes the decay before the key
        // scale, the band and the tap slots below read it.
        let helpers::TrapezoidCoeffs {
            dt: dt_bsh,
            da: da_bsh,
            alpha: _alpha_bsh,
            nu: nu_bsh,
            nu_interior: nu_interior_bsh,
            gamma: gamma_bsh,
            log_precision: log_precision_bsh,
        } = helpers::trapezoidal_coefficients(
            dd_dt_bsh,
            dd_A_raw_bsh,
            Some(lambda_raw_bsh),
            mu_raw_bsh,
            self.dt_bias_h.val(),
            self.trapezoid_spec(),
            self.gain_input(noise_bsh, cache.log_precision_bh.clone()),
        )
        .padded(pad_bs.as_ref());
        // The tropical register's inputs, on the same folded axis.
        let tropical_ab_bsh = tropical_btH.map(|(a_btH, b_btH)| {
            (unfold_micro_bs(a_btH, u), unfold_micro_bs(b_btH, u))
        });
        let nu_bsh = nu_bsh.expect("a β tap has a mass");
        san(&dt_bsh);
        san(&da_bsh);
        san(&gamma_bsh);

        // ── Compute scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁) ─────────────────────────
        //
        // Each shifted term is zero for the last positions that its own lag
        // reaches past: the taps that pay them belong to the *next* call, from
        // the tap slots. This also makes `h'` equal to the double-SSD state at
        // a cache boundary, hence the field-identity `From` impls.
        //
        // `t+lag` is a later *folded* position: lag 1 is
        // [`Trapezoid::HorizontalCarryOver`], lag `u` is [`Trapezoid::Vertical`].
        // This is the `Δ̃` collapse (`info/mamba-3/trapezoid-as-integration.md`
        // §5). The collapse theorem of §9 is why it holds unchanged for the
        // wider lag and for a second tap: still one scalar per sample, so one
        // pass. Only the same-step correction widens from the diagonal to a
        // `lag`-wide band (see [`crate::mamba3::single_ssd::token_band`]).
        let lag = self.tap_lag();
        // νₜ₊ₗ, the mass that a later position pays to this one. Zero past the
        // end of the call, where the tap belongs to the next call.
        let pay_forward = |nu_bsh: Tensor<3>, l: usize| {
            let zero_bLh = Tensor::zeros([batch, l, nheads], &device);
            if sequence == l {
                zero_bLh
            } else {
                Tensor::cat(vec![nu_bsh.narrow(1, l, sequence - l), zero_bLh], 1)
            }
        };
        let far_shifted_bsh = pay_forward(nu_bsh.clone(), lag);
        // A two-tap pattern also collapses to one scalar (§9), with a second
        // shift in it: `scaleₜ = γₜ + νⁱⁿᵗₜ₊₁ + νᶠᵃʳₜ₊ₗₐ₉`.
        let interior_shifted_bsh = nu_interior_bsh
            .clone()
            .map(|nu_bsh| pay_forward(nu_bsh, 1));
        let scale_bsh = gamma_bsh.clone() + far_shifted_bsh.clone();
        let scale_bsh = match &interior_shifted_bsh {
            Some(interior_bsh) => scale_bsh + interior_bsh.clone(),
            None => scale_bsh,
        };
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
        let c_btmhr = helpers::qk_norm_expand_bias::<5, 6>(
            c_raw_btMGR.reshape([batch, tokens, mimo_rank, ngroups, state_rank]),
            &self.c_norm,
            self.c_bias_hmr.val(),
            3,
            nheads,
        );

        // ── Step 5: Data-dependent transition rotation of B and C ─────────────
        // Shared with the double-ssd pathway through [`rotate_bc_forward`]. The
        // single-pass SSD core below does not depend on the rotation: it only
        // uses the rotated B̄/C̄ (the RoPE factoring `C̄ₜᵀB̄ᵢ = Cₜᵀ·Rel(t,i)·Bᵢ`
        // holds for every kind). `C` turns at token resolution: it needs the
        // cumulative rotation of the micro-step where it is read, a stride slice
        // of the rotation of `B`.
        let (b_bsmhr, c_btmhr, new_rotation) = rotate_bc_forward(
            rot_bsa,
            dt_bsh.clone(),
            cache.rotation.clone(),
            b_bsmhr,
            c_btmhr,
            u,
            self.rotation_spec(),
        );
        san(&b_bsmhr);
        san(&c_btmhr);

        // ── Save the B and x of the last `lag` positions (raw, no MIMO_V) ─────
        // The next call pays their second installment. At `lag = u` that
        // window is exactly the last token.
        let (b_last_bumhr, x_last_buhp) = self.save_tap_slots(
            &b_bsmhr,
            &x_bshp,
            &da_bsh,
            lag,
            end_b.clone().map(|end_b| {
                (end_b, cache.k_state_bumhr.clone(), cache.v_state_buhp.clone())
            }),
        );

        // ── Boundary β seed for initial state ─────────────────────────────────
        // Add Σⱼ νⱼ · Σₘ K_prev[j, m] ⊗ (x_prev[j] ⊙ mimo_xₘ) to the carried
        // single-ssd SSM state. Cache slot `j` (oldest first) is the position
        // whose tap position `j` of this call pays, so it takes `νⱼ` from the
        // first `lag` positions of this call. `x_prev` already carries the
        // decay from its own position to the boundary, and `K_prev` its own
        // rotation, so the pair *is* the transported write.
        //
        // Under [`Trapezoid::VerticalPlusHorizontalCarryOver`], one more
        // installment crosses. The newest slot is also the lag-1 tap of the
        // *first* position of this call, with the empty decay product. So it is
        // the same term, with `νⁱⁿᵗ₀` added to the own weight of that slot. For
        // every other pattern, the interior tap is closed exactly there.
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let nu_head_buh = nu_bsh.clone().narrow(1, 0, lag);
        let nu_head_buh = match nu_interior_bsh
            .filter(|_| self.trapezoid.interior_tap_crosses_tokens())
        {
            // `lag ≥ 2` here: a two-tap pattern folds at `u = 1`, so an interior
            // tap and a one-slot FIFO never coexist.
            Some(nu_interior_bsh) => Tensor::cat(
                vec![
                    nu_head_buh.clone().narrow(1, 0, lag - 1),
                    nu_head_buh.narrow(1, lag - 1, 1) + nu_interior_bsh.narrow(1, 0, 1),
                ],
                1,
            ),
            None => nu_head_buh,
        };
        let v_prev_mimo_bumhp = helpers::build_v_with_mimo::<4, 5>(
            cache
                .v_state_buhp
                .clone()
                .expect("a β tap keeps its (B, x) cache slots"),
            mimo_x_hmp.as_ref(),
            2,
        ); // [batch, lag, mimo_rank, nheads, per_head_dim]
        let v_prev_mimo_bumhp = v_prev_mimo_bumhp * nu_head_buh.unsqueeze_dims::<5>(&[2, 4]);
        // The seed contracts over the slots *and* the ranks (both are outer
        // products into one state). So, fused, it is the same one
        // `mimo_outer_sum` that the per-token write uses.
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
        let chunk_len = ssd_path.chunk_len_or_optimal(self);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        // V passed to SSD is raw x with MIMO_V applied (not γ-scaled).
        let v_bshmp = helpers::build_v_with_mimo::<4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);
        // v_bshmp has axis order [b, s, m, h, p] (insert_dim=2 onto [b,s,h,p]).

        // ── The lag-`u` correction band ───────────────────────────────────────
        // At lag 1 the key scale is wrong only on the diagonal, and
        // `ssd/diag.rs` corrects it inside the kernel. At lag `u` the exception
        // is `u` wide, and at the positions that the readout keeps it is
        // exactly the token. So it is one contraction here, not a wider mask.
        // See [`crate::mamba3::single_ssd::token_band`].
        //
        // The band is the **far** installment only, also for a two-tap
        // pattern. At the surviving read `t = τu + u−1`, the interior (lag-1)
        // installment of a sample has already landed for every `j < u−1`. At
        // `j = u−1`, the kernel replaces the whole weight with `γ` anyway. Only
        // the lag-`u` installment is still unpaid there.
        let band_correction_btmhp = (lag > 1)
            .then(|| {
                crate::mamba3::single_ssd::token_band::token_band_correction(
                    v_bshmp.clone(),
                    b_bsmhr.clone(),
                    c_btmhr.clone(),
                    far_shifted_bsh,
                    da_bsh.clone(),
                    micro_steps,
                )
            })
            .flatten();

        // `C` and `γ` are on the read axis of the chunk, so they pad by whole
        // tokens. `chunk_len` is a multiple of `u`, so `pad` is too.
        let gamma_bth = helpers::read_rows::<3, 4>(gamma_bsh, 1, u);
        let tokens_padded = sequence_padded / u;
        let pad_tokens = tokens_padded - tokens;

        #[rustfmt::skip]
        let (v_bShmp, da_bSh, gamma_bTh, scale_bSh, b_bSmhr, c_bTmhr) = if pad == 0 {
            (v_bshmp, da_bsh, gamma_bth, scale_bsh, b_bsmhr, c_btmhr)
        } else {
            let pad_bShmp = Tensor::zeros([batch, pad, mimo_rank, nheads, per_head_dim], &device);
            let pad_bSh = Tensor::zeros([batch, pad, nheads], &device);
            let pad_bTh = Tensor::zeros([batch, pad_tokens, nheads], &device);
            let pad_bSmhr = Tensor::zeros([batch, pad, mimo_rank, nheads, state_rank], &device);
            let pad_bTmhr =
                Tensor::zeros([batch, pad_tokens, mimo_rank, nheads, state_rank], &device);
            (
                Tensor::cat(vec![v_bshmp, pad_bShmp], 1),
                Tensor::cat(vec![da_bsh, pad_bSh.clone()], 1),
                Tensor::cat(vec![gamma_bth, pad_bTh], 1),
                Tensor::cat(vec![scale_bsh, pad_bSh], 1),
                Tensor::cat(vec![b_bsmhr, pad_bSmhr], 1),
                Tensor::cat(vec![c_btmhr, pad_bTmhr], 1),
            )
        };

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let chunk_tokens = Mamba3SsdPath::chunk_tokens(chunk_len, u);
        let v_bnlmhp =
            v_bShmp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let gamma_bnth = gamma_bTh.reshape([batch, nchunks, chunk_tokens, nheads]);
        let scale_bnlh = scale_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bntmhr =
            c_bTmhr.reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]);

        // ── Step 7: Run single-pass form SSD ───────────────────────────────────────
        let ssd_input = Mamba3SingleSsdInput {
            v_bnlmhp,
            b_bnlmhr,
            c_bntmhr,
            da_bnlh,
            gamma_bnth,
            scale_bnlh,
            initial_state_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
            read_stride: u,
            siso_specialization: self.siso_specialization,
        };
        let (y_bntmhp, final_state_bhpr) = ssd_input.run(ssd_path);

        san(&y_bntmhp);
        san(&final_state_bhpr);
        cache.ssm_bhpr = final_state_bhpr;

        // ── Step 8: Unpad ─────────────────────────────────────────────────────
        // The SSD returns token resolution: the readout is after all `u`
        // writes, and the kernel computed only that row.
        let y_bTmhp = y_bntmhp.reshape([batch, tokens_padded, mimo_rank, nheads, per_head_dim]);
        let y_bsmhp = if pad == 0 {
            y_bTmhp
        } else {
            y_bTmhp.narrow(1, 0, tokens)
        };

        let y_bsmhp = match band_correction_btmhp {
            Some(correction_btmhp) => y_bsmhp - correction_btmhp,
            None => y_bsmhp,
        };
        let (y_bsmhp, log_precision_bh, tropical_bh) = self.positive_tail(
            y_bsmhp,
            log_precision_bsh,
            tropical_ab_bsh,
            cache.tropical_bh.clone(),
            end_b.map(|end_b| (end_b, cache.log_precision_bh.clone())),
        );
        cache.log_precision_bh = log_precision_bh;
        cache.tropical_bh = tropical_bh;
        let x_bthp = crate::mamba3::product::last_micro4(x_bshp.clone(), micro_steps);
        let sequence = tokens;

        // ── Step 9: D skip + gate + MIMO_O down-projection ────────────────────
        // The D skip uses the raw x ⊙ mimo_x (not γ-scaled, as in the double
        // pathway), at the micro-step of the readout.
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
        // The new cumulative rotation from [`rotate_bc_forward`]. It has the
        // convention of the double-ssd cache, so the two caches convert.
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
        /// Process a **single token** with the pure recurrent form (the
        /// formulas are in [`Mamba3::step`]). It converts the cache to the
        /// double-ssd form, runs [`Mamba3::step_double_ssd`], and converts back.
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
            // form. A single-ssd cache holds the trapezoid state at a call
            // boundary, where the single- and double-ssd accumulators are equal
            // (see the `From` impls in `crate::mamba3::cache`). So the
            // conversion in and back out is lossless. One recurrence step is
            // itself a boundary-to-boundary transition, so the round trip stays
            // exact.
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
