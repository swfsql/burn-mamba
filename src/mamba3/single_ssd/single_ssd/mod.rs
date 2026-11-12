//! # Mamba-3 — Single-Pass SSD Forward
//!
//! `forward_single_ssd` and `step_single_ssd` on
//! [`Mamba3`](crate::mamba3::mamba3::Mamba3): the pathway of the **official
//! Mamba-3 kernels**, Triton (SISO) and Tilelang (MIMO):
//!
//! ```text
//!   scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁)
//!
//!   forward_single_ssd:    h' = SSD(V_raw, K_scaled = scaleₜ B) with:
//!                               * strict lower-triangular intra-chunk mask
//!                               * additive γ-weighted diagonal correction
//!                               * boundary β seed Σⱼ νⱼ Kⱼ ⊗ xⱼ over the
//!                                 `lag` slots of the tap FIFO of the cache
//!                               * at lag > 1, the rest of the correction band
//!                                 ([`crate::mamba3::single_ssd::token_band`])
//! ```
//!
//! `lag` is [`TapPattern::tap_lag`](crate::mamba3::tap_pattern::TapPattern::tap_lag):
//! `1` for the default tap pattern, `u` for
//! [`TapPattern::Vertical`](crate::mamba3::tap_pattern::TapPattern::Vertical). A
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
use crate::mamba3::plant_frame::{PlantEntry, PlantResets, PlantTail};
use crate::mamba3::prelude::*;
use crate::mamba3::single_ssd::prelude::*;
use crate::standard_ssd;
use burn_stack::modules::sanity as san;
use burn::prelude::*;

impl Mamba3 {
    /// Process a full input sequence with the **single-ssd (single-pass)**
    /// pathway.
    ///
    /// Functionally equivalent to [`Self::forward_double_ssd`], with about half
    /// the SSD memory in training. The cache is a separate type
    /// ([`Mamba3SingleSsdCache`]), because this pathway carries the
    /// accumulator, which differs from the state mid-sequence.
    ///
    /// "Equivalent" covers **everything that a caller can observe**: the
    /// returned output and every field of the returned cache. It does not cover
    /// the intermediate `y` of the SSD scan at each *write* position. Under a
    /// lag-`u` pattern such as [`TapPattern::Vertical`], the correction band
    /// applies only at the last micro-step of each token (the readout). So the
    /// `u−1` per-token partial sums that this pathway discards do not match the
    /// double-SSD ones. Nothing reads them, and the *state* is exact at every
    /// position in both pathways. See [`crate::mamba3::single_ssd::token_band`],
    /// which also tells why that band never has to enter the kernel.
    ///
    /// Under [`TapPattern::None`] there is no second pass to fuse. The composite
    /// key scale is `γ`, and the γ-correction is the whole diagonal. So
    /// this *is* the double-SSD form (`h' ≡ h` at every position, not only at
    /// boundaries), and the call delegates to `forward_double_ssd`. It does not
    /// run a strict-mask kernel plus a correction that puts back what the mask
    /// removed.
    ///
    /// `pad_bt` marks a right-padded batch of tokens, as in [`Self::forward`].
    /// The last tap installment of a real position, which a padded position
    /// would pay, then stays in the tap FIFO, exactly as at a call boundary.
    /// So the accumulator `h'` equals the state from the end of each row on.
    ///
    /// # Shapes
    /// - `input_bsm`: `[batch, sequence, d_model]`
    /// - `pad_bt`: `[batch, sequence]`
    /// - output: `[batch, sequence, d_model]`
    pub fn forward_single_ssd(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3SingleSsdCache>,
        ssd_path: &Mamba3SsdPath,
        pad_bt: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Mamba3SingleSsdCache) {
        self.forward_single_ssd_packed(input_bsm, cache, None, ssd_path, pad_bt, None)
    }

    /// [`Self::forward_single_ssd`] over a packed batch: `reset_bt` marks the
    /// first token of each segment, which starts from `origin`, as in
    /// [`Self::forward_packed`].
    ///
    /// A reset is a call boundary inside the row: the positions before it pay
    /// no `ν` forward (the taps of the segment start read the slots of the
    /// origin), and the chunk restarts from the `h'` of the origin plus its
    /// own boundary seed, as a call from the origin does.
    #[allow(non_snake_case)]
    pub fn forward_single_ssd_packed(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3SingleSsdCache>,
        origin: Option<Mamba3SingleSsdCache>,
        ssd_path: &Mamba3SsdPath,
        pad_bt: Option<Tensor<2, Bool>>,
        reset_bt: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Mamba3SingleSsdCache) {
        if !self.tap_pattern.has_beta_tap() {
            let (out_bsm, cache) = self.forward_double_ssd_packed(
                input_bsm,
                cache.map(Into::into),
                origin.map(Into::into),
                ssd_path,
                pad_bt,
                reset_bt,
            );
            return (out_bsm, cache.into());
        }
        let [batch, tokens, _d_model] = input_bsm.dims();
        let nheads = self.nheads();
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let u = self.micro_steps;
        let device = input_bsm.device();
        let sequence = tokens * u;

        // ── The entry of the plant frame ──────────────────────────────────────
        // At a call boundary, the accumulator `h'` equals the state, so the
        // caches convert field by field.
        let chunk_len = ssd_path.chunk_len_or_optimal(self);
        let PlantEntry {
            mut cache,
            packed,
            row_end,
            z_bti,
            x_bshp,
            b_bsmhr,
            c_btmhr,
            da_bsh,
            alpha_bsh: _,
            nu_bsh,
            nu_interior_bsh,
            gamma_bsh,
            log_precision_bsh,
            tropical_ab_bsh,
            new_rotation,
            new_tap_fifo,
        } = self.plant_entry(
            input_bsm,
            cache.map(Into::into),
            pad_bt,
            reset_bt.map(|reset_bt| PlantResets {
                reset_bt,
                origin: origin.map(Into::into),
                chunk_len,
            }),
        );
        let nu_bsh = nu_bsh.expect("a β tap has a mass");

        // ── Compute scaleₜ = γₜ + νₜ₊ₗₐ₉ (+ νⁱⁿᵗₜ₊₁) ─────────────────────────
        //
        // Each shifted term is zero for the last positions that its own lag
        // reaches past: the taps that pay them belong to the *next* call, from
        // the tap FIFO. This also makes the accumulator `h'` equal to the
        // state at a cache boundary, hence the field-identity `From` impls.
        //
        // `t+lag` is a later *write* position: lag 1 is
        // [`TapPattern::HorizontalCarryOver`], lag `u` is [`TapPattern::Vertical`].
        // This is the `Δ̃` collapse (`info/mamba-3/trapezoid-as-integration.md`
        // §5). The collapse theorem of §9 is why it holds unchanged for the
        // wider lag and for a second tap: still one scalar per sample, so one
        // pass. Only the γ-correction widens from the diagonal to a
        // `lag`-wide band (see [`crate::mamba3::single_ssd::token_band`]).
        let lag = self.tap_lag();
        // νₜ₊ₗ, the mass that a later position pays to this one. Zero past the
        // end of the call, where the tap belongs to the next call. Zero also
        // across a reset of a packed row: the first positions of a segment pay
        // the cache slots, through the seed of their chunk (below). (No output
        // would see that mass: the restart drops the state it enters, and the
        // band correction takes it out of the reads of its token. The zero
        // keeps `h'` at a reset equal to that at a call boundary.)
        let pay_forward = |nu_bsh: Tensor<3>, l: usize| {
            let nu_bsh = match &packed {
                Some(p) => crate::padding::fill_padded(nu_bsh, &p.segments.heads(l), 0.0),
                None => nu_bsh,
            };
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

        // ── Boundary β seed for the initial accumulator ─────────────────────
        // Add Σⱼ νⱼ · Σₘ K_prev[j, m] ⊗ (x_prev[j] ⊙ mimo_xₘ) to the carried
        // accumulator. Cache slot `j` (oldest first) is the position
        // whose tap position `j` of this call pays, so it takes `νⱼ` from the
        // first `lag` positions of this call. `x_prev` already carries the
        // decay from its own position to the boundary, and `K_prev` its own
        // rotation, so the pair *is* the transported write.
        //
        // Under [`TapPattern::VerticalPlusHorizontalCarryOver`], one more
        // installment crosses. The newest slot is also the lag-1 tap of the
        // *first* position of this call, with the empty decay product. So it is
        // the same term, with `νⁱⁿᵗ₀` added to the own weight of that slot.
        // Every other pattern has no such term. Either it has no lag-1 tap
        // beside the tap at `tap_lag`, or that lag-1 tap is closed exactly
        // there.
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        // The masses that the first `width` positions of each chunk pay,
        // `[batch, nchunks, width, nheads]`. Only chunk 0 is the start of the
        // call. In a packed row, every chunk that starts a segment is the
        // start of one from the origin.
        let nchunks = sequence.div_ceil(chunk_len);
        let heads_of = |t_bsh: Tensor<3>, width: usize| {
            let pad = nchunks * chunk_len - sequence;
            let t_bSh = if pad == 0 {
                t_bsh
            } else {
                Tensor::cat(vec![t_bsh, Tensor::zeros([batch, pad, nheads], &device)], 1)
            };
            t_bSh
                .reshape([batch, nchunks, chunk_len, nheads])
                .narrow(2, 0, width)
        };
        let nu_heads_bnuh = heads_of(nu_bsh.clone(), lag);
        let nu_heads_bnuh = match nu_interior_bsh
            .filter(|_| self.tap_pattern.two_tap_lag1_crosses_tokens())
        {
            // `lag ≥ 2` here: a two-tap pattern folds at `u = 1`, so the lag-1
            // tap of a two-tap pattern and a one-slot FIFO never coexist.
            Some(nu_interior_bsh) => Tensor::cat(
                vec![
                    nu_heads_bnuh.clone().narrow(2, 0, lag - 1),
                    nu_heads_bnuh.narrow(2, lag - 1, 1) + heads_of(nu_interior_bsh, 1),
                ],
                2,
            ),
            None => nu_heads_bnuh,
        };
        // The slots of a FIFO, `v` with MIMO_V applied:
        // `([batch, lag, mimo_rank, nheads, per_head_dim], [batch, lag, mimo_rank, nheads, state_rank])`.
        let prev_fifo = |fifo: &Option<TapFifo>| {
            let TapFifo { b_bumhr, x_buhp } = fifo.clone().expect("a β tap keeps its tap FIFO");
            let v_prev_mimo_bumhp =
                helpers::build_v_with_mimo::<4, 5>(x_buhp, mimo_x_hmp.as_ref(), 2);
            (v_prev_mimo_bumhp, b_bumhr)
        };
        // The seed of the call, from the tap FIFO of the cache and the masses
        // of chunk 0.
        let (v_prev_mimo_bumhp, k_prev_bumhr) = prev_fifo(&cache.frame.tap_fifo);
        let nu_head_buh = nu_heads_bnuh.clone().narrow(1, 0, 1).squeeze_dim::<3>(1);
        let v_prev_mimo_bumhp = v_prev_mimo_bumhp * nu_head_buh.unsqueeze_dims::<5>(&[2, 4]);
        // The seed contracts over the slots *and* the ranks (both are outer
        // products into one accumulator). So, fused, it is the same one
        // `mimo_outer_sum` that the per-token write uses.
        let boundary_seed_bhpr = helpers::mimo_outer_sum(
            v_prev_mimo_bumhp.reshape([batch, lag * mimo_rank, nheads, per_head_dim]),
            k_prev_bumhr.reshape([batch, lag * mimo_rank, nheads, state_rank]),
            self.use_siso_decode_kernels(),
        );
        let initial_state_bhpr = cache.ssm_bhpr.clone() + boundary_seed_bhpr;
        san(&initial_state_bhpr);
        // A packed row: each chunk gets a seed from the slots of the origin,
        // in case it restarts. Each slot writes one outer product (over the
        // ranks). Each chunk weighs the slots by its own head masses.
        let seed_bnhpr = packed.as_ref().map(|p| {
            let (v_origin_mimo_bumhp, k_origin_bumhr) = prev_fifo(&p.origin.frame.tap_fifo);
            let slot_writes_bhuP = helpers::mimo_outer_sum(
                v_origin_mimo_bumhp.reshape([batch * lag, mimo_rank, nheads, per_head_dim]),
                k_origin_bumhr.reshape([batch * lag, mimo_rank, nheads, state_rank]),
                self.use_siso_decode_kernels(),
            )
            .reshape([batch, lag, nheads, per_head_dim * state_rank])
            .swap_dims(1, 2);
            nu_heads_bnuh
                .permute([0, 3, 1, 2]) // _bhnu
                .matmul(slot_writes_bhuP) // _bhnP
                .reshape([batch, nheads, nchunks, per_head_dim, state_rank])
                .swap_dims(1, 2)
        });

        // ── Pad sequence to multiple of chunk_len ─────────────────────────────
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
        // The band is the **lag-`u`** installment only, also for a two-tap
        // pattern. At the surviving read `t = τu + u−1`, the lag-1 installment
        // of a sample has already landed for every `j < u−1`. At
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
                    u,
                )
            })
            .flatten();

        // `C` and `γ` are on the read axis of the chunk, so they pad by whole
        // tokens. `chunk_len` is a multiple of `u`, so `pad` is too.
        let gamma_bth = helpers::read_positions::<3, 4>(gamma_bsh, 1, u);
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
        let chunk_tokens = standard_ssd::chunk_tokens(chunk_len, u);
        let v_bnlmhp =
            v_bShmp.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let gamma_bnth = gamma_bTh.reshape([batch, nchunks, chunk_tokens, nheads]);
        let scale_bnlh = scale_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bntmhr =
            c_bTmhr.reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]);

        // ── Run the single-pass form of the SSD ───────────────────────────────
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
            restarts: packed.as_ref().map(|p| crate::packing::Restarts {
                reset_bn: p.segments.reset_bn.clone(),
                origin_bhpr: p.origin.ssm_bhpr.clone(),
            }),
            seed_bnhpr,
        };
        let (y_bntmhp, final_state_bhpr) = ssd_input.run(ssd_path);

        san(&y_bntmhp);
        san(&final_state_bhpr);
        cache.ssm_bhpr = final_state_bhpr;

        // ── Unpad ─────────────────────────────────────────────────────────────
        // The SSD returns token resolution: the readout is after all `u`
        // writes, and the kernel computed only that position.
        let y_bTmhp = y_bntmhp.reshape([batch, tokens_padded, mimo_rank, nheads, per_head_dim]);
        let y_btmhp = if pad == 0 {
            y_bTmhp
        } else {
            y_bTmhp.narrow(1, 0, tokens)
        };

        let y_btmhp = match band_correction_btmhp {
            Some(correction_btmhp) => y_btmhp - correction_btmhp,
            None => y_btmhp,
        };

        // ── The tail of the plant frame ───────────────────────────────────────
        let (out_btd, cache) = self.plant_tail(
            PlantTail {
                y_btmhp,
                z_bti,
                x_bshp,
                log_precision_bsh,
                tropical_ab_bsh,
                row_end,
                packed,
                new_rotation,
                new_tap_fifo,
            },
            cache,
        );
        (out_btd, cache.into())
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
            // form. A single-ssd cache holds the accumulator at a call
            // boundary, where the accumulator equals the state (see the `From`
            // impls in `crate::mamba3::cache`). So the
            // conversion in and back out is lossless. One `step` call is
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
