//! ## Physical-frame state moments for Mamba-3 (serial chunkwise, exact)
//!
//! The Mamba-3 state is **complex** (paper Props. *complex-to-real* /
//! *rope-trick*): the real cache state `h̃ₜ` (`ssm_bhpr`) carries the rotations
//! absorbed into B̃/C̃, and the **physical** state — what raw, un-rotated C
//! reads — is the per-token de-rotation `cₜ = Dₜ†h̃ₜ` (see
//! [`RotationSeq::derotate_states`]). The shipped observable/penalty is the
//! Hermitian PR of `M_phys = Σₜ cₜᴴcₜ`
//! ([`StateMoments::pr_complex`](crate::modules::StateMoments::pr_complex)).
//!
//! Unlike the Mamba-2 moments (`mamba2/ssd/moments.rs`), **no closed form
//! exists**: in `M_phys[a,b]` the token index couples to the matrix entry
//! through the relative phase `e^{i(φ_{t,b} − φ_{t,a})}`, so the per-token
//! phases do not factor out of the Gram kernel — every exact factorisation
//! reproduces the per-token state as an intermediate. Exact `M_phys` therefore
//! costs materialised states, affordable **chunk-locally** in the
//! `SerialRecalculated` discipline. Per chunk (serial over `n`, one small
//! `(m2, m1)` accumulator pair carried):
//!
//! 1. Chunk-local decay `dₜ = exp(a_cum[t])` and 1-semiseparable mask
//!    `L[t,j] = exp(a_cum[t] − a_cum[j])` (as SSD Step 1).
//! 2. One chunk of cache-frame states, folding the combined-injection channel
//!    axis `M` into the write index:
//!    `h̃[t] = dₜ·h₋ + Σ_{j≤t,m} L[t,j] · x̂[j,m] ⊗ b̂[j,m]` — a broadcast
//!    product plus one batched GEMM; the transients are
//!    `[batch, nheads, l, l·M, per_head_dim]` and one chunk of states
//!    `[batch, nheads, l, per_head_dim, state_rank]`.
//! 3. De-rotate the states' `r` axis to the physical frame
//!    ([`RotationSeq::derotate_states`] — θ-differentiable, which is what lets
//!    a PR penalty shape the rotation itself).
//! 4. Accumulate the plain real `m2`/`m1` sums, masked to `valid_len`.
//!
//! The **combined injections** realise the trapezoid as one scalar-decay
//! system with a `2·mimo_rank` channel axis: `x̂ = concat_m(v_γ, v_β)`,
//! `b̂ = concat_m(B̃, B̃_prev)` (the β stream shift-before-chunking, its first
//! element from the cache's `k_state`/`v_state`), shared log-decay `da`. The
//! initial state is the cache's `ssm_bhpr` (+ optional learnable `h₀`),
//! counted exactly **once** — never the single-ssd kernel's seed-augmented
//! initial state, whose boundary-β term the β stream already carries.
//!
//! The chunk carry `h₋` is read from the **unmasked** last position (zero pads:
//! `Δ = 0` ⇒ identity decay, zero write, so the state is carried through
//! unchanged); the `valid_len` mask alone excludes pad garbage from the sums
//! (pad rotation values are irrelevant — masked before accumulation).
//!
//! Plain autodiff over the serial loop retains every chunk's states (the full
//! per-token trajectory) — the study-scale mode. The at-scale execution model
//! is a custom recompute backward in the `SerialRecalculated` pattern (module
//! `backward`, milestone 3).

#[cfg(feature = "autodiff")]
mod backward;
mod recalculated;

pub use recalculated::Mamba3MomentsBackendExt;

use crate::mamba3::helpers;
use crate::mamba3::prelude::Mamba3;
use crate::mamba3::rotation::RotationSeq;
use crate::modules::{StateMoments, sanity as san, segsum};
use burn::backend::Dispatch;
use burn::prelude::*;

/// Inputs of the physical-frame state-moments computation — the combined
/// (γ + β) injections of one chunked Mamba-3 `forward`, plus the per-token
/// cumulative rotation. Built at either pathway's pre-SSD seam; `C`/`D` are
/// never read. See the module header for the math.
#[allow(non_snake_case)]
pub struct Mamba3MomentsInput {
    /// Combined value injections, already γ/β-scaled — `M = 2·mimo_rank`
    /// channels (γ channels first, then the shifted β channels). Zero at pads.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, M, nheads, per_head_dim]`
    pub xhat_bnlMhp: Tensor<6>,
    /// Combined key injections: the **rotated** B̃ per γ channel and the
    /// shifted B̃ₜ₋₁ per β channel. Zero at pads.
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, M, nheads, state_rank]`
    pub bhat_bnlMhr: Tensor<6>,
    /// Log-decay `Δₜ·Aₜ` (negative; zero at pads ⇒ identity decay).
    ///
    /// # Shape
    /// - `[batch, nchunks, chunk_len, nheads]`
    pub da_bnlh: Tensor<4>,
    /// Per-token cumulative rotation over the **padded** sequence
    /// (`sequence = nchunks·chunk_len`; pad values are arbitrary — masked).
    pub rotation: RotationSeq,
    /// The SSM state entering this call (the cache's `ssm_bhpr` — **not** the
    /// single-ssd kernel's seed-augmented initial state; see module header).
    ///
    /// # Shape
    /// - `[batch, nheads, per_head_dim, state_rank]`
    pub initial_state_bhpr: Tensor<4>,
    /// Optional learnable initial state `h₀`, added to `initial_state_bhpr`
    /// exactly once.
    ///
    /// # Shape
    /// - `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Tensor<3>>,
}

impl Mamba3MomentsInput {
    /// A value-identical copy detached from any autodiff graph (used by the
    /// diagnostic wrapper so the moments branch records no backward nodes).
    pub fn detached(&self) -> Self {
        Self {
            xhat_bnlMhp: self.xhat_bnlMhp.clone().detach(),
            bhat_bnlMhr: self.bhat_bnlMhr.clone().detach(),
            da_bnlh: self.da_bnlh.clone().detach(),
            rotation: self.rotation.detached(),
            initial_state_bhpr: self.initial_state_bhpr.clone().detach(),
            init_state_hpr: self.init_state_hpr.clone().map(Tensor::detach),
        }
    }

    /// Exact pooled moments of every per-token **physical-frame** SSM state
    /// (see the module header). `valid_len` is the unpadded token count —
    /// states at zero-pad positions are excluded.
    ///
    /// Returns raw sums; `count = valid_len · per_head_dim` samples per
    /// `(batch, head)` slice (MIMO ranks share the state, so the count is
    /// unchanged from Mamba-2).
    #[allow(non_snake_case)]
    pub fn state_moments_phys(&self, valid_len: usize) -> StateMoments {
        let [batch, nchunks, chunk_len, chan, nheads, per_head_dim] = self.xhat_bnlMhp.dims();
        let [b2, n2, l2, c2, h2, state_rank] = self.bhat_bnlMhr.dims();
        assert_eq!([batch, nchunks, chunk_len, chan, nheads], [b2, n2, l2, c2, h2]);
        assert_eq!([batch, nchunks, chunk_len, nheads], self.da_bnlh.dims());
        assert!(
            (1..=nchunks * chunk_len).contains(&valid_len),
            "valid_len must be within the (padded) sequence"
        );
        let device = self.xhat_bnlMhp.device();

        // Combined initial state (cache + optional learnable h₀, once).
        let mut h_bhpr = self.initial_state_bhpr.clone();
        if let Some(init_hpr) = &self.init_state_hpr {
            h_bhpr = h_bhpr
                + init_hpr.clone().unsqueeze_dim::<4>(0).expand([
                    batch,
                    nheads,
                    per_head_dim,
                    state_rank,
                ]);
        }

        let mut m2_bhrr = Tensor::zeros([batch, nheads, state_rank, state_rank], &device);
        let mut m1_bhr = Tensor::zeros([batch, nheads, state_rank], &device);

        for n in 0..nchunks {
            let start = n * chunk_len;
            if start >= valid_len {
                // Every remaining token is a pad; nothing left to accumulate.
                break;
            }

            // ── Chunk-local decay dₜ and mask L (SSD Step 1) ──────────────────
            let da_blh = self.da_bnlh.clone().narrow(1, n, 1).squeeze_dim::<3>(1);
            let a_bhl = da_blh.permute([0, 2, 1]);
            let d_bhl = a_bhl.clone().cumsum(2).exp();
            let l_bhll = segsum::<3, 4>(a_bhl).exp();
            san(&d_bhl);
            san(&l_bhll);

            // ── One chunk of cache-frame states ───────────────────────────────
            // xw[b,h,t,j,m,p] = L[t,j] · x̂[j,m,p] (the chunk transient), then
            // one GEMM contracts the folded (j, m) write index against b̂.
            let xhat_blMhp = self
                .xhat_bnlMhp
                .clone()
                .narrow(1, n, 1)
                .squeeze_dim::<5>(1);
            let bhat_blMhr = self
                .bhat_bnlMhr
                .clone()
                .narrow(1, n, 1)
                .squeeze_dim::<5>(1);

            let xhat_bh1jMp = xhat_blMhp
                .permute([0, 3, 1, 2, 4]) // [b, h, j, M, p]
                .unsqueeze_dim::<6>(2); // [b, h, 1, j, M, p]
            let l_bhtj11 = l_bhll.unsqueeze_dims::<6>(&[4, 5]);
            let xw_bhtJMp =
                (l_bhtj11 * xhat_bh1jMp).reshape([batch, nheads, chunk_len, chunk_len * chan, per_head_dim]);

            let bhat_bh1JMr = bhat_blMhr
                .permute([0, 3, 1, 2, 4]) // [b, h, j, M, r]
                .reshape([batch, nheads, 1, chunk_len * chan, state_rank]);

            // intra[b,h,t,p,r] = Σ_{jm} xw[t, jm, p] · b̂[jm, r]
            let intra_bhtpr = xw_bhtJMp.permute([0, 1, 2, 4, 3]).matmul(bhat_bh1JMr);

            let states_bhtpr = d_bhl.unsqueeze_dims::<5>(&[3, 4])
                * h_bhpr.clone().unsqueeze_dim::<5>(2)
                + intra_bhtpr;
            san(&states_bhtpr);

            // Carry (unmasked: pads are identity steps, so the last position
            // already holds the state after the last real token).
            h_bhpr = states_bhtpr
                .clone()
                .narrow(2, chunk_len - 1, 1)
                .squeeze_dim::<4>(2);

            // ── De-rotate to the physical frame, mask, accumulate ─────────────
            let phys_blhpr = self
                .rotation
                .derotate_states(states_bhtpr.permute([0, 2, 1, 3, 4]), start);
            let mask_1l111 = Tensor::<1, Int>::arange(0..chunk_len as i64, &device)
                .lower_elem((valid_len - start) as i64)
                .float()
                .reshape([1, chunk_len, 1, 1, 1]);
            let masked_blhpr = phys_blhpr * mask_1l111;

            let masked_bhLPr = masked_blhpr
                .permute([0, 2, 1, 3, 4])
                .reshape([batch, nheads, chunk_len * per_head_dim, state_rank]);
            m2_bhrr = m2_bhrr
                + masked_bhLPr
                    .clone()
                    .permute([0, 1, 3, 2])
                    .matmul(masked_bhLPr.clone());
            m1_bhr = m1_bhr + masked_bhLPr.sum_dim(2).squeeze_dim::<3>(2);
        }
        san(&m2_bhrr);
        san(&m1_bhr);

        StateMoments {
            m2_bhrr,
            m1_bhr,
            count: valid_len * per_head_dim,
        }
    }

    /// [`Self::state_moments_phys`] with a custom **recompute backward** — the
    /// at-scale execution model. Mathematically identical (values and
    /// gradients, asserted by the tests); only the backward's memory profile
    /// differs: plain autodiff retains every chunk's states (the full
    /// per-token trajectory), while this node saves only the leaf inputs and
    /// re-materialises one chunk at a time during backprop (see
    /// `moments/backward.rs`).
    ///
    /// The optional learnable `init_state_hpr` is folded into the initial
    /// state *outside* the node (a plain autodiff add), so its gradient flows
    /// through `d_initial`.
    pub fn state_moments_phys_recalculated(&self, valid_len: usize) -> StateMoments {
        let [batch, nchunks, chunk_len, chan, nheads, per_head_dim] = self.xhat_bnlMhp.dims();
        let [b2, n2, l2, c2, h2, state_rank] = self.bhat_bnlMhr.dims();
        assert_eq!([batch, nchunks, chunk_len, chan, nheads], [b2, n2, l2, c2, h2]);
        assert_eq!([batch, nchunks, chunk_len, nheads], self.da_bnlh.dims());
        assert!(
            (1..=nchunks * chunk_len).contains(&valid_len),
            "valid_len must be within the (padded) sequence"
        );

        let mut initial_bhpr = self.initial_state_bhpr.clone();
        if let Some(init_hpr) = &self.init_state_hpr {
            initial_bhpr = initial_bhpr
                + init_hpr.clone().unsqueeze_dim::<4>(0).expand([
                    batch,
                    nheads,
                    per_head_dim,
                    state_rank,
                ]);
        }

        let (rot, quaternion, rope_dim, rotate_pairwise) = match &self.rotation {
            RotationSeq::Angle {
                cum_bsha,
                rope_dim,
                rotate_pairwise,
            } => {
                assert_eq!(cum_bsha.dims()[1], nchunks * chunk_len);
                (
                    cum_bsha.clone().into_dispatch(),
                    false,
                    *rope_dim,
                    *rotate_pairwise,
                )
            }
            RotationSeq::Quaternion { cum_bshj4 } => {
                assert_eq!(cum_bshj4.dims()[1], nchunks * chunk_len);
                (cum_bshj4.clone().into_dispatch(), true, 0, false)
            }
        };

        let (m2, m1) = <Dispatch as Mamba3MomentsBackendExt>::mamba3_state_moments_phys(
            self.xhat_bnlMhp.clone().into_dispatch(),
            self.bhat_bnlMhr.clone().into_dispatch(),
            self.da_bnlh.clone().into_dispatch(),
            rot,
            initial_bhpr.into_dispatch(),
            valid_len,
            quaternion,
            rope_dim,
            rotate_pairwise,
        );
        let m2_bhrr = Tensor::<4>::from_dispatch(m2);
        let m1_bhr = Tensor::<3>::from_dispatch(m1);
        san(&m2_bhrr);
        san(&m1_bhr);

        StateMoments {
            m2_bhrr,
            m1_bhr,
            count: valid_len * per_head_dim,
        }
    }
}

impl Mamba3 {
    /// Build the [`Mamba3MomentsInput`] from one `forward`'s **pre-SSD,
    /// sequence-level** tensors — the pathway-agnostic seam (both the
    /// double-ssd and single-ssd forwards call this with the same pieces, so
    /// the moments are structurally identical across pathways).
    ///
    /// Performs the β stream's shift-before-chunking (`prev_*` = the cache's
    /// `v_state`/`k_state`, read **before** they are overwritten), the γ/β
    /// scaling, the MIMO value expansion, and the zero-pad + chunk reshape.
    ///
    /// # Shapes
    /// - `x_bshp`: raw values `[batch, sequence, nheads, per_head_dim]`
    /// - `b_rot_bsmhr`: **rotated** B̃ `[batch, sequence, mimo, nheads, r]`
    /// - `gamma_bsh` / `beta_bsh` / `da_bsh`: trapezoid coefficients
    /// - `prev_x_bhp` / `prev_b_bmhr`: the cache's previous-token entries
    /// - `ssm_bhpr`: the cache's SSM state entering this call (**not** the
    ///   single-ssd kernel's seed-augmented initial state)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_moments_input(
        &self,
        x_bshp: Tensor<4>,
        b_rot_bsmhr: Tensor<5>,
        gamma_bsh: Tensor<3>,
        beta_bsh: Tensor<3>,
        da_bsh: Tensor<3>,
        prev_x_bhp: Tensor<3>,
        prev_b_bmhr: Tensor<4>,
        ssm_bhpr: Tensor<4>,
        rotation_seq: RotationSeq,
        chunk_len: usize,
    ) -> Mamba3MomentsInput {
        let [batch, sequence, nheads, per_head_dim] = x_bshp.dims();
        let [.., mimo_rank, _nheads, state_rank] = b_rot_bsmhr.dims();
        let device = x_bshp.device();

        // ── β stream: shift-before-chunking ───────────────────────────────────
        let prev_x_b1hp = prev_x_bhp.unsqueeze_dim::<4>(1);
        let x_prev_bshp = if sequence == 1 {
            prev_x_b1hp
        } else {
            Tensor::cat(
                vec![prev_x_b1hp, x_bshp.clone().narrow(1, 0, sequence - 1)],
                1,
            )
        };
        let prev_b_b1mhr = prev_b_bmhr.unsqueeze_dim::<5>(1);
        let b_prev_bsmhr = if sequence == 1 {
            prev_b_b1mhr
        } else {
            Tensor::cat(
                vec![
                    prev_b_b1mhr,
                    b_rot_bsmhr.clone().narrow(1, 0, sequence - 1),
                ],
                1,
            )
        };

        // ── γ/β scaling ───────────────────────────────────────────────────────
        let x_gamma_bshp = x_bshp * gamma_bsh.unsqueeze_dim::<4>(3);
        let x_beta_bshp = x_prev_bshp * beta_bsh.unsqueeze_dim::<4>(3);

        // ── Zero-pad to a chunk_len multiple ──────────────────────────────────
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;
        #[rustfmt::skip]
        let (x_gamma_bShp, x_beta_bShp, da_bSh, b_bSmhr, b_prev_bSmhr) = if pad == 0 {
            (x_gamma_bshp, x_beta_bshp, da_bsh, b_rot_bsmhr, b_prev_bsmhr)
        } else {
            let pad_bShp = Tensor::zeros([batch, pad, nheads, per_head_dim], &device);
            let pad_bSh = Tensor::zeros([batch, pad, nheads], &device);
            let pad_bSmhr = Tensor::zeros([batch, pad, mimo_rank, nheads, state_rank], &device);
            (
                Tensor::cat(vec![x_gamma_bshp, pad_bShp.clone()], 1),
                Tensor::cat(vec![x_beta_bshp, pad_bShp], 1),
                Tensor::cat(vec![da_bsh, pad_bSh], 1),
                Tensor::cat(vec![b_rot_bsmhr, pad_bSmhr.clone()], 1),
                Tensor::cat(vec![b_prev_bsmhr, pad_bSmhr], 1),
            )
        };

        // ── Chunk + MIMO value expansion + combined channels ──────────────────
        let nchunks = sequence_padded / chunk_len;
        let x_gamma_bnlhp =
            x_gamma_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let x_beta_bnlhp = x_beta_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr =
            b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let b_prev_bnlmhr =
            b_prev_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let v_gamma_bnlmhp =
            helpers::build_v_with_mimo::<5, 6>(x_gamma_bnlhp, mimo_x_hmp.as_ref(), 3);
        let v_beta_bnlmhp =
            helpers::build_v_with_mimo::<5, 6>(x_beta_bnlhp, mimo_x_hmp.as_ref(), 3);

        Mamba3MomentsInput {
            xhat_bnlMhp: Tensor::cat(vec![v_gamma_bnlmhp, v_beta_bnlmhp], 3),
            bhat_bnlMhr: Tensor::cat(vec![b_bnlmhr, b_prev_bnlmhr], 3),
            da_bnlh,
            rotation: rotation_seq.pad_to(sequence_padded),
            initial_state_bhpr: ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        }
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
