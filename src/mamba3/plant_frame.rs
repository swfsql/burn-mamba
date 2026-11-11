//! # Mamba-3 — Plant Frame
//!
//! The plant frame is the work of a Mamba-3 block around the plant. Both
//! forward pathways and `step` call it. Each one keeps only its middle: its
//! scan and the inputs that only this scan needs. `step` enters at
//! `tokens = 1`. It adds that read axis to its input and removes it from its
//! output.
//!
//! ```text
//!   input  ── plant_entry ──▶ rotated B and C, masses, tap FIFO
//!                                        │
//!                              the scan of the pathway
//!                                        │
//!   output ◀── plant_tail ───          readout
//! ```
//!
//! - [`Mamba3::plant_entry`] holds the padding, the segments, the fresh cache,
//!   the origin, the in-projection, the split, the trapezoid coefficients,
//!   QK-Norm, the rotation and the tap FIFO.
//! - [`Mamba3::plant_tail`] holds the positive systems, the `D` skip, the gate,
//!   the MIMO merge and `out_proj`. The band correction of single-SSD runs
//!   before the tail and stays in that pathway.
//!
//! The frame has one cache type, `Mamba3DoubleSsdCache`. A forward starts and
//! ends at a call boundary, where the accumulator of single-SSD equals the
//! state. So the `From` impls convert the cache of that pathway (ADR-0003).
//! The frame writes the frame cache ([`Mamba3FrameCache`]) of the cache. Only
//! the scan of each pathway writes the state.
//!
//! A bug in the frame fails both pathways alike, and a parity test cannot show
//! it. So the tests of this module compare the split and the tail with their
//! definitions (ADR-0007).

use crate::mamba3::cache::{Mamba3FrameCache, TapFifo};
use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCache;
use crate::mamba3::helpers;
use crate::mamba3::in_proj_layout::GroupKey;
use crate::mamba3::mamba3::{Mamba3, Packed};
use crate::mamba3::product::last_micro4;
use crate::mamba3::rotation::{RotationState, rotate_bc_forward};
use crate::standard_ssd;
use burn::prelude::*;
use burn_stack::modules::{Silu, sanity as san};

/// The channel groups of the in-projection, in the order that
/// [`Mamba3::in_proj`] documents. A per-micro-step group is on the write axis.
/// The gate `z` and the read `C` stay at token resolution.
#[allow(non_snake_case)]
pub(crate) struct InProjStreams {
    pub z_bti: Tensor<3>,
    pub x_bsi: Tensor<3>,
    pub b_raw_bsMGR: Tensor<3>,
    pub c_raw_btMGR: Tensor<3>,
    pub dd_dt_bsh: Tensor<3>,
    pub dd_a_raw_bsh: Tensor<3>,
    /// `None` under `TapPattern::None`.
    pub lambda_raw_bsh: Option<Tensor<3>>,
    /// `None` unless the pattern has two taps.
    pub mu_raw_bsh: Option<Tensor<3>>,
    /// `None` under `Real1D`.
    pub rot_bsa: Option<Tensor<3>>,
    /// `None` unless the gain projects its noise.
    pub noise_bsh: Option<Tensor<3>>,
    /// `(a, b)`. `None` without the tropical register.
    pub tropical_ab_bsh: Option<(Tensor<3>, Tensor<3>)>,
}

/// The resets of a packed call. The chunk length comes with them, because only
/// the resets need it: `step` has no SSD path.
pub(crate) struct PlantResets {
    /// The first token of each segment, `[batch, tokens]`.
    pub reset_bt: Tensor<2, Bool>,
    /// The cache that each segment starts from. `None` uses a fresh cache.
    pub origin: Option<Mamba3DoubleSsdCache>,
    /// The chunk length on the write axis.
    pub chunk_len: usize,
}

/// The result of [`Mamba3::plant_entry`], one field for each stream.
///
/// A pathway moves each field out at once. It does not hold this struct across
/// its scan, so each tensor drops at its last use.
pub(crate) struct PlantEntry {
    /// The cache of the call: the one that the caller gave, or a fresh one.
    pub cache: Mamba3DoubleSsdCache,
    /// The resets and the origin of a packed call.
    pub packed: Option<Packed>,
    /// One past the last real write position of each row, for a padded call.
    pub end_b: Option<Tensor<1, Int>>,
    /// The gate.
    pub z_bti: Tensor<3>,
    /// The value stream.
    pub x_bshp: Tensor<4>,
    /// The rotated `B`.
    pub b_bsmhr: Tensor<5>,
    /// The rotated `C`.
    pub c_btmhr: Tensor<5>,
    /// The log-decay.
    pub da_bsh: Tensor<3>,
    /// `α = exp(da)`.
    pub alpha_bsh: Tensor<3>,
    /// The tap mass `ν`. `None` under `TapPattern::None`.
    pub nu_bsh: Option<Tensor<3>>,
    /// The mass of the lag-1 tap of a two-tap pattern.
    pub nu_interior_bsh: Option<Tensor<3>>,
    /// `γ = λ·Δ`.
    pub gamma_bsh: Tensor<3>,
    /// `ln Λ` at each write position, under a Kalman gain.
    pub log_precision_bsh: Option<Tensor<3>>,
    /// The `(a, b)` of the tropical register, on the write axis.
    pub tropical_ab_bsh: Option<(Tensor<3>, Tensor<3>)>,
    /// The cumulative rotation for the next call.
    pub new_rotation: RotationState,
    /// The tap FIFO for the next call.
    pub new_tap_fifo: Option<TapFifo>,
}

/// The input of [`Mamba3::plant_tail`]. The readout comes from the middle of a
/// pathway. Each other field is the field of [`PlantEntry`] with the same name.
pub(crate) struct PlantTail {
    /// The readout of the scan.
    pub y_btmhp: Tensor<5>,
    pub z_bti: Tensor<3>,
    pub x_bshp: Tensor<4>,
    pub log_precision_bsh: Option<Tensor<3>>,
    pub tropical_ab_bsh: Option<(Tensor<3>, Tensor<3>)>,
    pub end_b: Option<Tensor<1, Int>>,
    pub packed: Option<Packed>,
    pub new_rotation: RotationState,
    pub new_tap_fifo: Option<TapFifo>,
}

impl Mamba3 {
    /// The entry of the plant frame: from the input to the rotated `B` and `C`,
    /// the masses and the tap FIFO.
    ///
    /// `cache` and the origin of `resets` start fresh when they are `None`.
    ///
    /// # Shapes
    /// - `input_btd` : `[batch, tokens, d_model]`
    /// - `pad_bt`    : `[batch, tokens]`
    #[allow(non_snake_case)]
    pub(crate) fn plant_entry(
        &self,
        input_btd: Tensor<3>,
        cache: Option<Mamba3DoubleSsdCache>,
        pad_bt: Option<Tensor<2, Bool>>,
        resets: Option<PlantResets>,
    ) -> PlantEntry {
        let [batch, tokens, _d_model] = input_btd.dims();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let u = self.micro_steps;
        let device = input_btd.device();

        // MambaProduct: from the split down to the readout, one sequence
        // position *is* one micro-step. The `s` in each shape suffix counts
        // micro-steps, and `t` counts tokens. See [`crate::mamba3::product`].
        let sequence = tokens * u;

        assert!(tokens > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_btd);

        // A padded token pads all of its micro-steps. `end_b` is one past the
        // last real position of each row on the write axis, where its cache
        // fields are read.
        let pad_bs = pad_bt.map(|pad_bt| crate::padding::repeat_positions(pad_bt, u));
        let end_b = pad_bs.as_ref().map(crate::padding::real_end_b);

        // ── Initialise the caches that the caller did not give ────────────────
        let fresh = || Mamba3DoubleSsdCache::from(self.zero_cache(batch, &device));
        let cache = cache.unwrap_or_else(fresh);
        // A reset is at the first token of a chunk, which is its first
        // micro-step on the write axis. See [`crate::packing`].
        let packed = resets.map(|PlantResets { reset_bt, origin, chunk_len }| {
            let chunk_tokens = standard_ssd::chunk_tokens(chunk_len, u);
            let reset_bn = crate::packing::chunk_resets(reset_bt, chunk_tokens);
            Packed {
                segments: crate::packing::Segments::new(reset_bn, chunk_len, sequence),
                origin: origin.unwrap_or_else(fresh),
            }
        });

        // ── In-projection ─────────────────────────────────────────────────────
        let proj_btd = self.project_in(input_btd);
        san(&proj_btd);
        let InProjStreams {
            z_bti,
            x_bsi,
            b_raw_bsMGR,
            c_raw_btMGR,
            dd_dt_bsh,
            dd_a_raw_bsh,
            lambda_raw_bsh,
            mu_raw_bsh,
            rot_bsa,
            noise_bsh,
            tropical_ab_bsh,
        } = self.split_in_proj(proj_btd);
        san(&z_bti);
        san(&x_bsi);
        san(&dd_dt_bsh);

        // ── Discretisation + trapezoidal coefficients ─────────────────────────
        // Under a Kalman gain, this call computes the decay before anything
        // below reads it.
        let helpers::TrapezoidCoeffs {
            dt: dt_bsh,
            da: da_bsh,
            alpha: alpha_bsh,
            nu: nu_bsh,
            nu_interior: nu_interior_bsh,
            gamma: gamma_bsh,
            log_precision: log_precision_bsh,
        } = helpers::trapezoidal_coefficients(
            dd_dt_bsh,
            dd_a_raw_bsh,
            lambda_raw_bsh,
            mu_raw_bsh,
            self.dt_bias_h.val(),
            self.tap_pattern_spec(),
            self.gain_input(noise_bsh, cache.frame.log_precision_bh.clone(), packed.as_ref()),
        )
        .padded(pad_bs.as_ref());
        san(&dt_bsh);
        san(&da_bsh);
        san(&alpha_bsh);
        for nu_bsh in [&nu_bsh, &nu_interior_bsh].into_iter().flatten() {
            san(nu_bsh);
        }
        san(&gamma_bsh);

        let x_bshp = x_bsi.reshape([batch, sequence, nheads, per_head_dim]);

        // ── QK-Norm on B and C ────────────────────────────────────────────────
        // QK-Norm over state_rank, then expand ngroups→nheads, then add a
        // per-(head, mimo-rank) bias. The group dim is axis 3 of `_btmgr`.
        // `C` stays at token resolution: the readout is once per token, at its
        // last micro-step.
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
        assert_eq!([batch, sequence, mimo_rank, nheads, state_rank], b_bsmhr.dims());
        assert_eq!([batch, tokens, mimo_rank, nheads, state_rank], c_btmhr.dims());

        // ── Data-dependent transition rotation of B and C ─────────────────────
        // Per `RotationKind`: nothing (Real1D), the cumulative angle
        // (Complex2D), or the cumulative quaternion(s) (Quaternion4D, Rotor4D).
        // `C` turns at token resolution: it needs the cumulative rotation of
        // the micro-step where it is read, a stride slice of the rotation of
        // `B`.
        let (b_bsmhr, c_btmhr, new_rotation) = rotate_bc_forward(
            rot_bsa,
            dt_bsh,
            cache.frame.rotation.clone(),
            b_bsmhr,
            c_btmhr,
            u,
            self.rotation_spec(),
            packed.as_ref().map(|p| (&p.segments.start_bs, p.origin.frame.rotation.clone())),
        );
        san(&b_bsmhr);
        san(&c_btmhr);
        new_rotation.sanity();

        // ── Save the B and x of the last `lag` positions for the cache ────────
        // These are exactly the positions whose taps the next call must pay.
        // So the trapezoid continues unchanged across a split prefill (at
        // `lag = u`, that window is exactly the last token). With no β tap
        // there is nothing to continue, and the FIFO stays empty.
        let new_tap_fifo = self.save_tap_slots(
            &b_bsmhr,
            &x_bshp,
            &da_bsh,
            self.tap_lag(),
            end_b.clone().map(|end_b| (end_b, cache.frame.tap_fifo.clone())),
            packed.as_ref(),
        );

        PlantEntry {
            cache,
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
            new_tap_fifo,
        }
    }

    /// The tail of the plant frame: from the readout of a scan to `out_proj`.
    ///
    /// It runs the positive systems, the `D` skip, the gate (or the gated
    /// RMSNorm), the MIMO merge and `out_proj`. It then writes the frame cache
    /// of the next call. The caller has already written the state.
    ///
    /// # Shapes
    /// - output : `[batch, tokens, d_model]`
    pub(crate) fn plant_tail(
        &self,
        tail: PlantTail,
        mut cache: Mamba3DoubleSsdCache,
    ) -> (Tensor<3>, Mamba3DoubleSsdCache) {
        let PlantTail {
            y_btmhp,
            z_bti,
            x_bshp,
            log_precision_bsh,
            tropical_ab_bsh,
            end_b,
            packed,
            new_rotation,
            new_tap_fifo,
        } = tail;
        let [batch, tokens, mimo_rank, nheads, per_head_dim] = y_btmhp.dims();
        let btmhp = [batch, tokens, mimo_rank, nheads, per_head_dim];
        let d_inner = self.d_inner();

        let (y_btmhp, log_precision_bh, tropical_bh) = self.positive_tail(
            y_btmhp,
            log_precision_bsh,
            tropical_ab_bsh,
            cache.frame.tropical_bh.clone(),
            end_b.map(|end_b| (end_b, cache.frame.log_precision_bh.clone())),
            packed.as_ref(),
        );

        // ── D skip ────────────────────────────────────────────────────────────
        // It reads the raw `x ⊙ mimo_x` (not scaled by `γ`), at the micro-step
        // of the readout.
        let x_bthp = last_micro4(x_bshp, self.micro_steps);
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let v_raw_btmhp = helpers::build_v_with_mimo::<4, 5>(x_bthp, mimo_x_hmp.as_ref(), 2);
        let d_111h1 = self.d_h.val().unsqueeze_dims::<5>(&[0, 1, 2, 4]);
        let y_btmhp = y_btmhp + d_111h1 * v_raw_btmhp;
        san(&y_btmhp);

        // ── Gate (or gated norm) and rank aggregation ─────────────────────────
        // When `out_norm` is set, a per-head gated RMSNorm replaces the SiLU
        // gate: `RmsNormGated(y, z) = norm(y) * silu(z)`, with the norm over
        // `per_head_dim`.
        let z_bthp = z_bti.reshape([batch, tokens, nheads, per_head_dim]);
        let y_bti = if mimo_rank > 1 {
            let mimo_z_hmp = self.mimo_z_hmp.as_ref().map(|p| p.val()).unwrap();
            let mimo_o_hmp = self.mimo_o_hmp.as_ref().map(|p| p.val()).unwrap();

            // zₘ = z ⊙ mimo_z[m]
            let z_btmhp = z_bthp
                .unsqueeze_dim::<5>(2) // z_bt1hp
                .expand(btmhp);
            let mimo_z_btmhp = mimo_z_hmp
                .swap_dims(0, 1) // mimo_z_mhp
                .unsqueeze_dims::<5>(&[0, 1]) // mimo_z_11mhp
                .expand(btmhp);
            let z_btmhp = z_btmhp * mimo_z_btmhp;
            san(&z_btmhp);

            let combined_btmhp = match &self.out_norm {
                Some(norm) => norm.forward(y_btmhp, z_btmhp),
                None => y_btmhp * Silu::new().forward(z_btmhp),
            };
            san(&combined_btmhp);

            // out = sumₘ mimo_o[m] ⊙ combined[m]
            let mimo_o_btmhp = mimo_o_hmp
                .swap_dims(0, 1) // mimo_o_mhp
                .unsqueeze_dims::<5>(&[0, 1]) // mimo_o_11mhp
                .expand(btmhp);
            let y_bthp: Tensor<4> = (combined_btmhp * mimo_o_btmhp)
                .sum_dim(2) // y_bt1hp
                .squeeze_dim(2);
            y_bthp.reshape([batch, tokens, d_inner])
        } else {
            // SISO: one rank, so the gate alone.
            let y_bthp: Tensor<4> = y_btmhp.squeeze_dim(2);
            let combined_bthp = match &self.out_norm {
                Some(norm) => norm.forward(y_bthp, z_bthp),
                None => y_bthp * Silu::new().forward(z_bthp),
            };
            combined_bthp.reshape([batch, tokens, d_inner])
        };
        san(&y_bti);

        // ── Out-projection ────────────────────────────────────────────────────
        let out_btd = self.out_proj.forward(y_bti);
        san(&out_btd);

        cache.frame = Mamba3FrameCache {
            tap_fifo: new_tap_fifo,
            rotation: new_rotation,
            log_precision_bh,
            tropical_bh,
        };

        (out_btd, cache)
    }

    /// Split the output of the in-projection into its channel groups, in the
    /// layout of the table in [`crate::mamba3::in_proj_layout`].
    ///
    /// # Shapes
    /// - `proj_btd` : `[batch, tokens, d_in_proj]`
    pub(crate) fn split_in_proj(&self, proj_btd: Tensor<3>) -> InProjStreams {
        use GroupKey::*;
        let mut parts = self.in_proj_layout().split(proj_btd);
        InProjStreams {
            z_bti: parts.take_fixed(Z),
            x_bsi: parts.take_fixed(X),
            b_raw_bsMGR: parts.take_fixed(BRaw),
            c_raw_btMGR: parts.take_fixed(CRaw),
            dd_dt_bsh: parts.take_fixed(DdDt),
            dd_a_raw_bsh: parts.take_fixed(DdARaw),
            lambda_raw_bsh: parts.take(LambdaRaw),
            mu_raw_bsh: parts.take(MuRaw),
            rot_bsa: parts.take(Rotation),
            noise_bsh: parts.take(KalmanNoise),
            tropical_ab_bsh: parts.take(TropicalA).zip(parts.take(TropicalB)),
        }
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
