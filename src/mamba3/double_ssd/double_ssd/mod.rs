//! # Mamba-3 — Double-Pass SSD Forward
//!
//! [`Mamba3::forward_double_ssd`](crate::mamba3::mamba3::Mamba3::forward_double_ssd)
//! and [`Mamba3::step_double_ssd`](crate::mamba3::mamba3::Mamba3::step_double_ssd):
//! the decomposition of
//! [`VikramLex/mamba3-minimal`](https://github.com/VikramLex/mamba3-minimal).
//!
//! ```text
//!   hₚ = αₚ hₚ₋₁ + βₚ Bₚ₋ₗₐ₉ ⊗ xₚ₋ₗₐ₉ + γₚ Bₚ ⊗ xₚ   (original double-ssd trapezoidal)
//!
//!   forward:    h = SSD(γ-scaled V, B)   +   SSD(β-scaled V_shifted, B_shifted)
//! ```
//!
//! The shift is [`TapPattern::tap_lag`](crate::mamba3::tap_pattern::TapPattern::tap_lag)
//! write positions (`1` by default, `u` for
//! [`TapPattern::Vertical`](crate::mamba3::tap_pattern::TapPattern::Vertical)), and
//! `β = ν·α` carries the transport across that whole gap.
//!
//! A two-tap pattern adds a second `β` side at lag 1. The two shifts are
//! different, so the passes cannot fuse, and this pathway runs **three** SSD
//! calls. The single-SSD pathway collapses them into its key scale and stays at
//! one call, so it suits such a pattern better.
//!
//! This is simple to derive and to verify (everything uses the standard SSD),
//! but it increases the intra-chunk and chunk-state memory during training.
//!
//! See also: [`crate::mamba3::mamba3`] and [`crate::mamba3::single_ssd::single_ssd`].

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::helpers;
use crate::mamba3::plant_frame::{PlantEntry, PlantResets, PlantTail};
use crate::mamba3::prelude::*;
use burn_stack::modules::sanity as san;
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Mamba3::forward  (chunkwise double-SSD — training / prefill)
// ---------------------------------------------------------------------------

impl Mamba3 {
    /// Process a full input sequence with the double-ssd pathway.
    ///
    /// For MIMO (mimo_rank>1), B/C have mimo_rank parallel rank channels. All
    /// ranks share the state, and each rank writes to it independently.
    ///
    /// `pad_bt` marks a right-padded batch of tokens, as in [`Self::forward`].
    ///
    /// # Shapes
    /// - `input_bsm` : `[batch, sequence, d_model]`
    /// - `pad_bt`    : `[batch, sequence]`
    /// - output      : `[batch, sequence, d_model]`
    pub fn forward_double_ssd(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3DoubleSsdCache>,
        ssd_path: &Mamba3SsdPath,
        pad_bt: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Mamba3DoubleSsdCache) {
        self.forward_double_ssd_packed(input_bsm, cache, None, ssd_path, pad_bt, None)
    }

    /// [`Self::forward_double_ssd`] over a packed batch: `reset_bt` marks the
    /// first token of each segment, which starts from `origin`, as in
    /// [`Self::forward_packed`].
    #[allow(non_snake_case)]
    pub fn forward_double_ssd_packed(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3DoubleSsdCache>,
        origin: Option<Mamba3DoubleSsdCache>,
        ssd_path: &Mamba3SsdPath,
        pad_bt: Option<Tensor<2, Bool>>,
        reset_bt: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Mamba3DoubleSsdCache) {
        let [batch, tokens, _d_model] = input_bsm.dims();
        let nheads = self.nheads();
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let u = self.micro_steps;
        let device = input_bsm.device();
        let sequence = tokens * u;

        // ── The entry of the plant frame ──────────────────────────────────────
        let chunk_len = ssd_path.chunk_len_or_optimal(self);
        let PlantEntry {
            mut cache,
            packed,
            end_b,
            z_bti,
            x_bshp,
            b_bsmhr,
            c_btmhr,
            da_bsh,
            alpha_bsh,
            nu_bsh,
            nu_interior_bsh,
            gamma_bsh,
            log_precision_bsh,
            tropical_ab_bsh,
            new_rotation,
            b_last_bumhr,
            x_last_buhp,
        } = self.plant_entry(
            input_bsm,
            cache,
            pad_bt,
            reset_bt.map(|reset_bt| PlantResets { reset_bt, origin, chunk_len }),
        );

        // ── The β term's shifted, β-scaled inputs ─────────────────────────────
        //
        // "Shift-Before-Chunking": prepend the cached xₜ₋₁ / Bₜ₋₁ at the
        // sequence level (before SSD chunking), so the β term at t=0 sees the
        // prior token of a continued cache. For a fresh (zero) cache this is
        // the same as zero-padding.
        //
        // The shift is [`TapPattern::tap_lag`] *write* positions: 1 for the
        // default [`TapPattern::HorizontalCarryOver`], `u` for
        // [`TapPattern::Vertical`]. That is exactly the FIFO depth of `step`.
        //
        // A lag-`L` tap must be transported across *its own* gap (§9): by
        // `Πᵈ⁼⁰..ᴸ⁻¹ αₚ₋ᵈ`, not `αₚ`. `β = ν·α` holds the `d = 0` factor, and
        // `interior_gap_decay` the rest. For the first `L` positions, the
        // missing factors are those that the `v` slots of the cache already
        // carry.
        //
        // A two-tap pattern adds a second, lag-1 side. The two shifts are
        // different, so they cannot share a pass, and this pathway runs three
        // SSD calls. The single-SSD pathway fuses them into its key scale and
        // stays at one. The prefix of the lag-1 tap of a two-tap pattern is the
        // newest slot of the FIFO, which *is* the lag-1 slot, undecayed. It has no effect
        // under [`TapPattern::VerticalPlusHorizontalReset`], whose `νⁱⁿᵗ` is
        // zero at every position that would read it.
        //
        // Under [`TapPattern::None`] there is no left endpoint: no shift, no
        // β-scaled copy of `x`, and (below) no second SSD call. `forward` is
        // then one standard SSD pass, with keys scaled by `γ = Δ`.
        let lag = self.tap_lag();
        // The prefix of a lag-`L` tap is the newest `L` slots of a FIFO: all
        // of it for the tap at `tap_lag`, only the newest slot for the lag-1
        // tap of a two-tap pattern.
        let heads = |fifo: &Mamba3DoubleSsdCache, lag: usize| -> (Tensor<4>, Tensor<5>) {
            let slots = "a β tap keeps its (B, x) cache slots";
            let slots_buhp = fifo.v_state_buhp.clone().expect(slots);
            let slots_bumhr = fifo.k_state_bumhr.clone().expect(slots);
            let newest = slots_buhp.dims()[1] - lag;
            (slots_buhp.narrow(1, newest, lag), slots_bumhr.narrow(1, newest, lag))
        };
        let beta_side = |nu_bsh: Tensor<3>, lag: usize| -> (Tensor<4>, Tensor<5>) {
            let (x_head_buhp, b_head_bumhr) = heads(&cache, lag);
            let x_prev_bshp = helpers::shift_stream(x_bshp.clone(), x_head_buhp, lag);
            let b_prev_bsmhr = helpers::shift_stream(b_bsmhr.clone(), b_head_bumhr, lag);
            // A packed row: each segment starts on the prefix of the origin,
            // as a call from it.
            let (x_prev_bshp, b_prev_bsmhr) = match &packed {
                Some(p) => {
                    let (x_head_buhp, b_head_bumhr) = heads(&p.origin, lag);
                    (
                        p.segments.restart_heads(x_prev_bshp, 1, x_head_buhp),
                        p.segments.restart_heads(b_prev_bsmhr, 1, b_head_bumhr),
                    )
                }
                None => (x_prev_bshp, b_prev_bsmhr),
            };
            let beta_bsh = nu_bsh * alpha_bsh.clone();
            let segments = packed.as_ref().map(|p| &p.segments);
            let beta_bsh = match helpers::interior_gap_decay(da_bsh.clone(), lag, segments) {
                Some(gap_bsh) => beta_bsh * gap_bsh,
                None => beta_bsh,
            };
            // β is a per-head scalar, broadcast over mimo_rank and per_head_dim.
            let beta_bsh1 = beta_bsh.unsqueeze_dim::<4>(3);
            (x_prev_bshp * beta_bsh1, b_prev_bsmhr) // βₚ · xₚ₋ₗₐ₉
        };
        let beta_sides: Vec<(Tensor<4>, Tensor<5>)> = nu_bsh
            .map(|nu_bsh| beta_side(nu_bsh, lag))
            .into_iter()
            .chain(nu_interior_bsh.map(|nu_bsh| beta_side(nu_bsh, 1)))
            .collect();

        // ── Scale the current-token input by γ ────────────────────────────────
        let gamma_bsh1 = gamma_bsh.unsqueeze_dim::<4>(3);
        let x_gamma_bshp = x_bshp.clone() * gamma_bsh1; // γₜ · xₜ

        // ── Pad sequence to multiple of chunk_len ─────────────────────────────
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        // Each zero block is built at most once, and the streams that need it
        // share it (a `Tensor` clone is a handle, not a copy).
        let pads = (pad > 0).then(|| {
            (
                Tensor::<4>::zeros([batch, pad, nheads, per_head_dim], &device),
                Tensor::<3>::zeros([batch, pad, nheads], &device),
                Tensor::<5>::zeros([batch, pad, mimo_rank, nheads, state_rank], &device),
            )
        });
        let pad_hp = |t: Tensor<4>| match &pads {
            Some((p, _, _)) => Tensor::cat(vec![t, p.clone()], 1),
            None => t,
        };
        let pad_h = |t: Tensor<3>| match &pads {
            Some((_, p, _)) => Tensor::cat(vec![t, p.clone()], 1),
            None => t,
        };
        let pad_mhr = |t: Tensor<5>| match &pads {
            Some((_, _, p)) => Tensor::cat(vec![t, p.clone()], 1),
            None => t,
        };

        // `C` is on the read axis of the chunk, so it pads by whole tokens.
        // `chunk_len` is a multiple of `u`, so `pad` is too.
        let tokens_padded = sequence_padded / u;
        let pad_tmhr = |t: Tensor<5>| match pad {
            0 => t,
            _ => Tensor::cat(
                vec![
                    t,
                    Tensor::zeros(
                        [batch, tokens_padded - tokens, mimo_rank, nheads, state_rank],
                        &device,
                    ),
                ],
                1,
            ),
        };

        let x_gamma_bShp = pad_hp(x_gamma_bshp);
        let da_bSh = pad_h(da_bsh);
        let b_bSmhr = pad_mhr(b_bsmhr);
        let c_bTmhr = pad_tmhr(c_btmhr);
        let beta_sides: Vec<_> = beta_sides
            .into_iter()
            .map(|(x_beta, b_prev)| (pad_hp(x_beta), pad_mhr(b_prev)))
            .collect();

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let chunk_tokens = Mamba3SsdPath::chunk_tokens(chunk_len, u);
        let x_gamma_bnlhp = x_gamma_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bntmhr =
            c_bTmhr.reshape([batch, nchunks, chunk_tokens, mimo_rank, nheads, state_rank]);

        // ── The MIMO-SSD call(s) ──────────────────────────────────────────────
        // Build the V tensors: insert the mimo_rank axis at position 3 of
        // `_bnlhp`. With a β tap, these are the two standard SSD passes that
        // give the pathway its name (γ-SSM + β-SSM, summed). Under
        // [`TapPattern::None`] the second pass does not exist, and a
        // "double"-SSD forward is one pass.
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let v_gamma_bnlmhp =
            helpers::build_v_with_mimo::<5, 6>(x_gamma_bnlhp.clone(), mimo_x_hmp.as_ref(), 3);
        // Every pass restarts its carry at the same chunks: the passes are
        // the terms of one state. The γ pass holds the state of the origin,
        // and a β pass restarts from zero, as it starts a call.
        let reset_bn = packed.as_ref().map(|p| p.segments.reset_bn.clone());

        let input_gamma = Mamba3DoubleSsdInput {
            v_bnlmhp: v_gamma_bnlmhp,
            da_bnlh: da_bnlh.clone(),
            b_bnlmhr: b_bnlmhr.clone(),
            c_bntmhr: c_bntmhr.clone(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
            read_stride: u,
            restarts: packed.as_ref().map(|p| crate::packing::Restarts {
                reset_bn: p.segments.reset_bn.clone(),
                origin_bhpr: p.origin.ssm_bhpr.clone(),
            }),
        };
        let (y_bntmhp, final_state_bhpr) = input_gamma.run(ssd_path);

        let (y_bntmhp, final_state_bhpr) = beta_sides.into_iter().fold(
            (y_bntmhp, final_state_bhpr),
            |(y_bntmhp, final_state_bhpr), (x_beta_bShp, b_prev_bSmhr)| {
                let x_beta_bnlhp =
                    x_beta_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
                let b_prev_bnlmhr =
                    b_prev_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
                let v_beta_bnlmhp =
                    helpers::build_v_with_mimo::<5, 6>(x_beta_bnlhp, mimo_x_hmp.as_ref(), 3);
                let zero_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
                let input_beta = Mamba3DoubleSsdInput {
                    v_bnlmhp: v_beta_bnlmhp,
                    da_bnlh: da_bnlh.clone(),
                    b_bnlmhr: b_prev_bnlmhr,
                    c_bntmhr: c_bntmhr.clone(),
                    initial_state_bhpr: zero_bhpr.clone(),
                    init_state_hpr: None,
                    read_stride: u,
                    restarts: reset_bn.clone().map(|reset_bn| crate::packing::Restarts {
                        reset_bn,
                        origin_bhpr: zero_bhpr,
                    }),
                };
                let (y_beta_bntmhp, final_state_beta_bhpr) = input_beta.run(ssd_path);
                (
                    y_bntmhp + y_beta_bntmhp,
                    final_state_bhpr + final_state_beta_bhpr,
                )
            },
        );

        san(&y_bntmhp);
        san(&final_state_bhpr);

        cache.ssm_bhpr = final_state_bhpr;

        // ── Unpad ─────────────────────────────────────────────────────────────
        // The SSD returns token resolution: the readout is after all `u`
        // writes, and the kernel computed only that position. From here, `t`
        // (`tokens`) is the time axis again: the read axis.
        let y_bTmhp = y_bntmhp.reshape([batch, tokens_padded, mimo_rank, nheads, per_head_dim]);
        let y_btmhp = if pad == 0 {
            y_bTmhp
        } else {
            y_bTmhp.narrow(1, 0, tokens)
        };

        // ── The tail of the plant frame ───────────────────────────────────────
        self.plant_tail(
            PlantTail {
                y_btmhp,
                z_bti,
                x_bshp,
                log_precision_bsh,
                tropical_ab_bsh,
                end_b,
                packed,
                new_rotation,
                b_last_bumhr,
                x_last_buhp,
            },
            cache,
        )
    }
}

// ---------------------------------------------------------------------------
// Mamba3::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    impl Mamba3 {
        /// State→output contraction:
        /// `out[b, m, h, p] = Σᵣ C[b, m, h, r] · state[b, h, p, r]`
        /// (`einsum('bhpr,bmhr->bmhp', state, C)`).
        ///
        /// At `mimo_rank == 1` the output axis of the GEMM is 1, so the matmul
        /// is a matrix–vector product. [`step_readout_siso`](Mamba3::step_readout_siso)
        /// writes it as a broadcast multiply plus a `state_rank` reduction.
        /// [`Mamba3Config::siso_specialization_decode`](crate::mamba3::mamba3::Mamba3Config::siso_specialization_decode)
        /// selects the branch. Both branches compute the same values and
        /// gradients.
        pub(crate) fn step_readout(
            state_bhpr: Tensor<4>,
            c_bmhr: Tensor<4>,
            siso_specialization: bool,
        ) -> Tensor<4> {
            let [_batch, mimo_rank, _nheads, _state_rank] = c_bmhr.dims();
            if mimo_rank == 1 && siso_specialization {
                Self::step_readout_siso(state_bhpr, c_bmhr)
            } else {
                Self::step_readout_mimo(state_bhpr, c_bmhr)
            }
        }

        /// SISO (`mimo_rank == 1`) state→output contraction: broadcast `C` over
        /// `per_head_dim` and reduce `state_rank`.
        pub(crate) fn step_readout_siso(state_bhpr: Tensor<4>, c_bmhr: Tensor<4>) -> Tensor<4> {
            let c_bh1r: Tensor<4> = c_bmhr.squeeze_dim::<3>(1).unsqueeze_dim(2);
            let out_bhp1: Tensor<4> = (state_bhpr * c_bh1r).sum_dim(3);
            out_bhp1.squeeze_dim::<3>(3).unsqueeze_dim(1) // out_b1hp
        }

        /// General MIMO state→output contraction: one matmul over `state_rank`.
        pub(crate) fn step_readout_mimo(state_bhpr: Tensor<4>, c_bmhr: Tensor<4>) -> Tensor<4> {
            let c_bhrm = c_bmhr.permute([0, 2, 3, 1]);
            let out_bhpm = state_bhpr.matmul(c_bhrm);
            out_bhpm.permute([0, 3, 1, 2])
        }

        /// Process a **single token** with the pure recurrent form (the
        /// formulas are in [`Mamba3::step`]). Both cache variants decode here:
        /// `step_single_ssd` converts its cache and calls this function.
        ///
        /// At `micro_steps = u > 1` the token is `u` of those micro-steps
        /// ([`crate::mamba3::product`]). They are evaluated **together**, not one
        /// at a time (see the body).
        ///
        /// # Shapes
        /// - `input_bd` : `[batch, d_model]`
        /// - output     : `[batch, d_model]`
        pub fn step_double_ssd(
            &self,
            input_bd: Tensor<2>,
            cache: Option<Mamba3DoubleSsdCache>,
        ) -> (Tensor<2>, Mamba3DoubleSsdCache) {
            let [batch, _d_model] = input_bd.dims();
            let nheads = self.nheads();
            let per_head_dim = self.per_head_dim();
            let state_rank = self.state_rank;
            let mimo_rank = self.mimo_rank;
            let u = self.micro_steps;

            // ── The entry of the plant frame, at `tokens = 1` ─────────────────
            // `step` adds a read axis of length 1 to its input, and removes it
            // from the output of the tail. The write axis of the streams is
            // then the `u` micro-steps of the token. A Kalman gain computes its
            // decay in the entry, from the cached `ln Λ`, over those `u`
            // micro-steps: the scan of `forward` at `len = u`.
            let PlantEntry {
                mut cache,
                packed: _,
                end_b: _,
                z_bti: z_b1i,
                x_bshp: x_buhp,
                b_bsmhr: b_bumhr,
                c_btmhr: c_b1mhr,
                da_bsh: da_buh,
                alpha_bsh: alpha_buh,
                nu_bsh: nu_buh,
                nu_interior_bsh: nu_interior_buh,
                gamma_bsh: gamma_buh,
                log_precision_bsh: log_precision_buh,
                tropical_ab_bsh: tropical_ab_buh,
                new_rotation,
                b_last_bumhr,
                x_last_buhp,
            } = self.plant_entry(input_bd.unsqueeze_dim::<3>(1), cache, None, None);
            let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
            let siso = self.use_siso_decode_kernels();

            // ── The `u` micro-steps of the token at once, then one readout ──────
            // MambaProduct evaluates a token as `u` consecutive positions of the
            // ordinary recurrence. So a `step` call is a **write-axis block** of
            // length `u`, and one call is `forward` on a one-token sequence
            // (the parity tests assert this). The block is not walked one
            // position at a time, because that is not necessary. After the RoPE
            // trick, the transition inside the block is the *scalar* `α`,
            // so the block has a closed form:
            //
            //     h = (∏ⱼ αⱼ)·h₋₁ + Σⱼ wⱼ · writeⱼ ,    wⱼ = ∏_{r>j} αᵣ
            //
            // Its transport `wⱼ` (the decay of a write to the end of the block)
            // is the same quantity that [`crate::mamba3::helpers::tail_decay`]
            // computes for the tap slots, over the whole block instead of its
            // last `lag`.
            //
            // Every `writeⱼ` is an outer product into one shared state. So with
            // the transport, and with `(u, mimo_rank)` fused into one
            // contracted axis, each **side** of the recurrence is one
            // [`helpers::mimo_outer_sum`] (the same collapse as the single-SSD
            // boundary seed over its slots). Thus the op count does not depend
            // on `u`. At `u = 1` the fused axis is `mimo_rank`, and this is the
            // unchanged op graph of stock Mamba-3.
            //
            // The trapezoid is unchanged by all of this. A tap at
            // [`TapPattern::tap_lag`] reads write position `p − lag`: the
            // [`helpers::shift_stream`] of `forward` over this block, with the
            // FIFO of the cache as the prefix. `β = ν·α` times
            // [`helpers::interior_gap_decay`] is the same gap transport (§9). At
            // lag 1 that gap has no interior, and the FIFO is one slot deep. At
            // lag `u` the block is exactly as long as the lag, so every tap
            // reads the FIFO, and none reads within the block.
            let lag = self.tap_lag();

            // ── The two transports of the block ───────────────────────────────
            // `w` carries a write to the end of the block (`None` at `u = 1`,
            // where that product is empty). `α_total` is the whole transition
            // of the token, the one factor on the carried state.
            let w_buh = helpers::tail_decay(da_buh.clone(), u);
            let alpha_total_bh11 = da_buh
                .clone()
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .exp()
                .unsqueeze_dims::<4>(&[2, 3]);

            // One side of the recurrence, transported and fused:
            //
            //   write[b, h, p, r] = Σ_{j,m} massⱼ·wⱼ · v[j, m, h, p] · K[j, m, h, r]
            //
            // that is `einsum('bfhp,bfhr->bhpr', mass·w·v, K)` over the fused
            // `f = u · mimo_rank` axis: one matmul, for any `u`.
            let write = |mass_buh: Tensor<3>, x_buhp: Tensor<4>, k_bumhr: Tensor<5>| {
                let mass_buh = match &w_buh {
                    Some(w_buh) => mass_buh * w_buh.clone(),
                    None => mass_buh,
                };
                let v_bumhp = helpers::build_v_with_mimo::<4, 5>(x_buhp, mimo_x_hmp.as_ref(), 2)
                    * mass_buh.unsqueeze_dims::<5>(&[2, 4]);
                san(&v_bumhp);
                let write_bhpr = helpers::mimo_outer_sum(
                    v_bumhp.reshape([batch, u * mimo_rank, nheads, per_head_dim]),
                    k_bumhr.reshape([batch, u * mimo_rank, nheads, state_rank]),
                    siso,
                );
                san(&write_bhpr);
                write_bhpr
            };

            // A tapped side at its own lag: the `beta_side` of `forward`, over
            // this block. The prefix is the newest `lag` slots of the FIFO: all
            // of it for the tap at `tap_lag`, only the newest slot for the
            // lag-1 tap of a two-tap pattern (by the FIFO convention, that slot
            // has the empty decay product). Under [`TapPattern::None`] there is
            // no tapped side: no shifted stream, no second outer product, one
            // term less in the state update.
            let tap_write = |nu_buh: Tensor<3>, lag: usize| {
                let slots_buhp = cache
                    .v_state_buhp
                    .clone()
                    .expect("a β tap keeps its (B, x) cache slots");
                let slots_bumhr = cache
                    .k_state_bumhr
                    .clone()
                    .expect("a β tap keeps its (B, x) cache slots");
                let newest = slots_buhp.dims()[1] - lag;
                let x_prev_buhp = helpers::shift_stream(
                    x_buhp.clone(),
                    slots_buhp.narrow(1, newest, lag),
                    lag,
                );
                let b_prev_bumhr =
                    helpers::shift_stream(b_bumhr.clone(), slots_bumhr.narrow(1, newest, lag), lag);
                let beta_buh = nu_buh * alpha_buh.clone();
                let beta_buh = match helpers::interior_gap_decay(da_buh.clone(), lag, None) {
                    Some(gap_buh) => beta_buh * gap_buh,
                    None => beta_buh,
                };
                write(beta_buh, x_prev_buhp, b_prev_bumhr)
            };

            // ── State update ──────────────────────────────────────────────────
            let state_bhpr = alpha_total_bh11 * cache.ssm_bhpr.clone()
                + write(gamma_buh, x_buhp.clone(), b_bumhr.clone());
            let state_bhpr = [
                nu_buh.map(|nu_buh| tap_write(nu_buh, lag)),
                nu_interior_buh.map(|nu_buh| tap_write(nu_buh, 1)),
            ]
            .into_iter()
            .flatten()
            .fold(state_bhpr, |state, write| state + write);
            san(&state_bhpr);

            // ── Readout ───────────────────────────────────────────────────────
            // yₘ[b, m, h, p] = sumᵣ C[b, m, h, r] * state[b, h, p, r], with `C`
            // at the one read position of the block.
            let c_bmhr = c_b1mhr.squeeze_dim::<4>(1);
            let y_bmhp = Self::step_readout(state_bhpr.clone(), c_bmhr, siso);
            san(&y_bmhp);
            cache.ssm_bhpr = state_bhpr;

            // ── The tail of the plant frame, on a read axis of length 1 ───────
            // The last micro-step of the block is both the read position and
            // the next carry of the positive systems.
            let (out_b1d, cache) = self.plant_tail(
                PlantTail {
                    y_btmhp: y_bmhp.unsqueeze_dim::<5>(1),
                    z_bti: z_b1i,
                    x_bshp: x_buhp,
                    log_precision_bsh: log_precision_buh,
                    tropical_ab_bsh: tropical_ab_buh,
                    end_b: None,
                    packed: None,
                    new_rotation,
                    b_last_bumhr,
                    x_last_buhp,
                },
                cache,
            );
            (out_b1d.squeeze_dim::<2>(1), cache)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
