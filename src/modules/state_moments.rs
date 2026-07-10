//! # Pooled SSM-state moments (for state participation ratios)
//!
//! [`StateMoments`] carries the **exact** first and second moments of every
//! per-token SSM state `hₜ ∈ ℝ^{per_head_dim × state_rank}` of a `forward`
//! pass, treating each state **row** (one `(token, per_head_dim)` pair) as a
//! sample in `ℝ^{state_rank}` — the same sample convention a token-by-token
//! `step` loop reading the cache would produce.
//!
//! The moments are all a participation ratio (PR, a differentiable effective
//! rank) needs: with `Σ` the sample covariance, `PR = (tr Σ)² / tr(Σ²)` and
//! both traces derive from `Σ hhᵀ`, `Σ h`, and the sample count. Storing raw
//! **sums** (not averages) makes moments *composable*: [`StateMoments::merge`]
//! pools across forward calls (streaming chunks, eval batches) and the PR of
//! the merged moments is the exact pooled PR.

use burn::prelude::*;

/// How the `state_rank` axis of a (realified) SSM state groups into complex /
/// quaternionic coordinates — the realification layout the Mamba-3 rotation
/// applies to B/C (and hence to the state). Consumed by
/// [`StateMoments::pr_complex`].
///
/// Construct it with `Mamba3::state_pairing()` (the single source of truth,
/// mirroring exactly what `rotate_bc_forward` applies) — never re-derive the
/// layout by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatePairing {
    /// No rotation pairing: every coordinate is real (`rope_fraction = 0`);
    /// [`StateMoments::pr_complex`] then equals [`StateMoments::pr`].
    Real,
    /// Complex pairs, interleaved (NeoX-style; Mamba-3 SISO): coordinate `2a`
    /// is the real and `2a + 1` the imaginary part of complex coordinate `a`,
    /// for `a < num_pairs`; the tail `[2·num_pairs, state_rank)` stays real.
    ComplexInterleaved {
        /// Number of rotated complex pairs (`rope_dim / 2`).
        num_pairs: usize,
    },
    /// Complex pairs, half-and-half (GPT-J-style; Mamba-3 MIMO): coordinate
    /// `a` is the real and `state_rank/2 + a` the imaginary part, for
    /// `a < num_pairs`; the remainder of **both** halves stays real.
    ComplexHalfHalf {
        /// Number of rotated complex pairs (`rope_dim / 2`).
        num_pairs: usize,
    },
    /// Quaternion blocks (Mamba-3 `Quaternion4D`): coordinates `[4j, 4j + 4)`
    /// form quaternion coordinate `j` (components ordered `w, x, y, z`), for
    /// `j < num_blocks`; the tail stays real.
    QuaternionBlocks {
        /// Number of rotated quaternion blocks (`rope_width / 4`).
        num_blocks: usize,
    },
}

/// Raw (un-normalised) first/second moments of the per-token SSM states of
/// one block's `forward` pass, pooled over tokens and `per_head_dim` rows.
///
/// Produced by `forward_with_state_moments` — closed-form for Mamba-2
/// (`Mamba2SsdInput::state_moments`, no state materialisation) and serial
/// chunkwise for Mamba-3 (`Mamba3MomentsInput::state_moments_phys`, the
/// per-token **physical-frame** states of the complex SSM).
#[derive(Debug, Clone)]
pub struct StateMoments {
    /// Second-moment (Gram) sum `Σₜ hₜᵀ hₜ` — the `ᵀ` contraction pools the
    /// `per_head_dim` rows, the `Σₜ` the (unpadded) tokens.
    ///
    /// # Shape
    /// - `[batch, nheads, state_rank, state_rank]`
    pub m2_bhrr: Tensor<4>,
    /// First-moment sum `Σₜ Σₚ hₜ[p, :]`.
    ///
    /// # Shape
    /// - `[batch, nheads, state_rank]`
    pub m1_bhr: Tensor<3>,
    /// Samples pooled into each `(batch, head)` slice:
    /// `valid_tokens · per_head_dim` (grows additively under [`Self::merge`]).
    pub count: usize,
}

impl StateMoments {
    /// Pool two moment sets (e.g. consecutive streamed `forward` calls, or
    /// separate eval batches). PR of the merged moments is the exact PR of
    /// the union of samples.
    pub fn merge(self, other: Self) -> Self {
        assert_eq!(
            self.m2_bhrr.dims(),
            other.m2_bhrr.dims(),
            "merged state moments must share [batch, nheads, state_rank]"
        );
        Self {
            m2_bhrr: self.m2_bhrr + other.m2_bhrr,
            m1_bhr: self.m1_bhr + other.m1_bhr,
            count: self.count + other.count,
        }
    }

    /// Fold the batch dimension into the samples (batch-pooled moments with
    /// `batch = 1`), matching diagnostics that treat every
    /// `(token, batch, per_head_dim)` triple as one sample.
    pub fn pool_batch(self) -> Self {
        let [batch, _h, _r, _r2] = self.m2_bhrr.dims();
        Self {
            m2_bhrr: self.m2_bhrr.sum_dim(0),
            m1_bhr: self.m1_bhr.sum_dim(0),
            count: self.count * batch,
        }
    }

    /// Participation ratio `(tr Σ)² / tr(Σ²)` of the sample covariance, per
    /// `(batch, head)` slice; `center` subtracts the sample mean (`Σ` becomes
    /// the centered covariance instead of the raw second moment).
    ///
    /// Differentiable (two traces, no eigendecomposition).
    ///
    /// # Shape
    /// - output: `[batch, nheads]`
    pub fn pr(&self, center: bool) -> Tensor<2> {
        let [batch, nheads, state_rank, _] = self.m2_bhrr.dims();
        assert!(self.count > 0, "state moments hold no samples");
        let device = self.m2_bhrr.device();
        let samples = self.count as f32;

        let sigma_bhrr = {
            let m2_bhrr = self.m2_bhrr.clone() / samples;
            if center {
                let mu_bhr = self.m1_bhr.clone() / samples;
                let outer_bhrr =
                    mu_bhr.clone().unsqueeze_dim::<4>(3) * mu_bhr.unsqueeze_dim::<4>(2);
                m2_bhrr - outer_bhrr
            } else {
                m2_bhrr
            }
        };

        // tr Σ via an identity mask; tr(Σ²) = ‖Σ‖²_F (Σ is symmetric).
        let eye_11rr = Tensor::<2>::eye(state_rank, &device).unsqueeze::<4>();
        let tr_bh = (sigma_bhrr.clone() * eye_11rr.clone())
            .sum_dim(3)
            .sum_dim(2)
            .reshape([batch, nheads]);
        // `PR = (tr Σ)² / tr(Σ²)`. Computed via the trace-normalised
        // `Σ̂ = Σ / tr(Σ).detach()`, keeping **both** traces of `Σ̂`:
        //
        //     PR = tr(Σ̂)² / tr(Σ̂²).
        //
        // With `c = tr(Σ).detach()` a frozen scalar, the `c²` cancels between
        // numerator and denominator, so this equals `tr(Σ)²/tr(Σ²)` **as a
        // function of Σ** — identical value *and* exact gradient — while every
        // differentiated intermediate stays O(1). Two subtleties that a naive
        // rewrite gets wrong:
        //   - Keep the numerator `tr(Σ̂)²`: it is numerically 1, but with the
        //     normaliser detached its *gradient* w.r.t. Σ is not zero — it
        //     carries PR's rank-reducing (trace-tangential) component.
        //     Collapsing to `1/tr(Σ̂²)` drops it, leaving only the radial
        //     (magnitude) direction, orthogonal to ∇PR — a penalty that no
        //     longer reduces rank (see `pr_gradient_matches_direct_formula`).
        //   - Differentiating *through* `tr(Σ)` instead (no detach) is
        //     value/grad-correct but puts `tr(Σ²)²` in the backward, which
        //     underflows fp32 to 0 (→ NaN gradient) once the state magnitude
        //     `tr(Σ) ≲ 1e-11` — which weight decay drives it toward. The
        //     detached O(1) form is finite at every representable magnitude
        //     (see `pr_gradient_finite_as_magnitude_shrinks`).
        //
        // Two floors, for two quantities at very different scales. The
        // normaliser `tr Σ` is a *magnitude* that weight decay drives down to —
        // and below — `div_eps` (the crate's O(1)-calibrated negligibility
        // threshold): flooring it there would corrupt PR across the live
        // operating range, so it is floored only at the dtype's smallest
        // positive normal (`finfo().min_positive`), firing solely for an
        // all-zero state (`Σ ≡ 0`, e.g. a dead head — then `Σ̂ = 0/ε = 0`, no
        // `0/0`). `tr(Σ̂²)` is scale-normalised (`∈ [1/r, 1]` for any nonzero Σ)
        // and nears zero only for `Σ ≡ 0`, so `div_eps(dtype)` is the correct
        // dtype-aware guard there (cf. `MseLoss`'s fp16 path).
        let dtype = self.m2_bhrr.dtype();
        let min_positive = dtype.finfo().expect("state moments are a float dtype").min_positive;
        let scale_bh = tr_bh.clamp_min(min_positive).detach();
        let sigma_hat = sigma_bhrr / scale_bh.reshape([batch, nheads, 1, 1]);
        let tr1_hat_bh = (sigma_hat.clone() * eye_11rr)
            .sum_dim(3)
            .sum_dim(2)
            .reshape([batch, nheads]);
        let tr2_hat_bh = sigma_hat
            .powf_scalar(2.0)
            .sum_dim(3)
            .sum_dim(2)
            .reshape([batch, nheads])
            .clamp_min(crate::utils::div_eps(dtype));
        tr1_hat_bh.clone() * tr1_hat_bh / tr2_hat_bh
    }

    /// Participation ratio of the **Hermitian** sample covariance, treating the
    /// `state_rank` axis as realified complex (or quaternionic) coordinates per
    /// `pairing` — the Mamba-3 counterpart of [`Self::pr`].
    ///
    /// With the pairing's complex view `c = x + iy`, the Hermitian moment is
    /// `M = A + iS` with `A = Σ(xxᵀ + yyᵀ)` and `S = Σ(xyᵀ − yxᵀ)` — both linear
    /// recombinations of `m2_bhrr` sub-blocks, so centering `Σ` centers `M`
    /// identically. `PR_ℂ = (tr M)² / tr(M²)`; the trace is real and equals the
    /// full real trace (frame-invariant), while `tr(M²) = Σ|M_ab|²`
    /// (`= ‖A‖²_F + ‖S‖²_F` for the fully-rotated complex case). One complex
    /// (or quaternionic) direction counts as **one** — the ×2 (×4) realified
    /// count is a representation artifact — so a rank-1 rotating conveyor reads
    /// `PR_ℂ ≡ 1` where [`Self::pr`] reads up to the block size.
    ///
    /// Un-rotated coordinates (partial `rope_fraction`) stay a real block `U`
    /// of the mixed Hermitian `[[M, X], [Xᴴ, U]]`; its trace and `Σ|·|²` join
    /// the sums (`tr(·²)` gains `2‖X‖²_F + ‖U‖²_F`). Quaternionic pairing uses
    /// the same formulas with `M_jk = Σ q̄ⱼqₖ` (diagonal real, 4-component
    /// norms).
    ///
    /// Differentiable; numerics mirror [`Self::pr`] (detached trace
    /// normalisation, `min_positive` / `div_eps` floors — see the comments
    /// there for why).
    ///
    /// # Shape
    /// - output: `[batch, nheads]`
    pub fn pr_complex(&self, pairing: &StatePairing, center: bool) -> Tensor<2> {
        let [batch, nheads, state_rank, _] = self.m2_bhrr.dims();
        assert!(self.count > 0, "state moments hold no samples");
        let device = self.m2_bhrr.device();
        let samples = self.count as f32;

        // Reorder the `r` axis into the canonical `[x-pairs | y-pairs | real]`
        // layout (complex pairings only — quaternion blocks are already
        // consecutive with a trailing real tail), so every block below is one
        // contiguous `narrow`. `rotated` is the realified width of the
        // rotated prefix in that canonical order.
        enum Blocks {
            Complex { num_pairs: usize },
            Quaternion { num_blocks: usize },
        }
        let (reorder, blocks, rotated) = match *pairing {
            StatePairing::Real => return self.pr(center),
            StatePairing::ComplexInterleaved { num_pairs } => {
                assert!(
                    num_pairs > 0 && 2 * num_pairs <= state_rank,
                    "interleaved pairing exceeds state_rank"
                );
                let mut idx: Vec<i64> = (0..num_pairs).map(|a| 2 * a as i64).collect();
                idx.extend((0..num_pairs).map(|a| 2 * a as i64 + 1));
                idx.extend((2 * num_pairs..state_rank).map(|k| k as i64));
                (Some(idx), Blocks::Complex { num_pairs }, 2 * num_pairs)
            }
            StatePairing::ComplexHalfHalf { num_pairs } => {
                let half = state_rank / 2;
                assert!(
                    num_pairs > 0 && num_pairs <= half,
                    "half-and-half pairing exceeds state_rank / 2"
                );
                let mut idx: Vec<i64> = (0..num_pairs).map(|a| a as i64).collect();
                idx.extend((0..num_pairs).map(|a| (half + a) as i64));
                idx.extend((num_pairs..half).map(|k| k as i64));
                idx.extend((half + num_pairs..state_rank).map(|k| k as i64));
                (Some(idx), Blocks::Complex { num_pairs }, 2 * num_pairs)
            }
            StatePairing::QuaternionBlocks { num_blocks } => {
                assert!(
                    num_blocks > 0 && 4 * num_blocks <= state_rank,
                    "quaternion pairing exceeds state_rank"
                );
                (None, Blocks::Quaternion { num_blocks }, 4 * num_blocks)
            }
        };

        let sigma_bhrr = {
            let m2_bhrr = self.m2_bhrr.clone() / samples;
            let sigma = if center {
                let mu_bhr = self.m1_bhr.clone() / samples;
                let outer_bhrr =
                    mu_bhr.clone().unsqueeze_dim::<4>(3) * mu_bhr.unsqueeze_dim::<4>(2);
                m2_bhrr - outer_bhrr
            } else {
                m2_bhrr
            };
            match reorder {
                Some(idx) => {
                    let idx = Tensor::<1, Int>::from_ints(idx.as_slice(), &device);
                    sigma.select(2, idx.clone()).select(3, idx)
                }
                None => sigma,
            }
        };

        // Frobenius norm² of a `[batch, nheads, ·, ·]` block, per (batch, head).
        let fro2_bh = |t: Tensor<4>| -> Tensor<2> {
            t.powf_scalar(2.0)
                .sum_dim(3)
                .sum_dim(2)
                .reshape([batch, nheads])
        };

        // The Hermitian trace equals the full real trace (the phases cancel on
        // the diagonal), so the normaliser is exactly [`Self::pr`]'s.
        let eye_11rr = Tensor::<2>::eye(state_rank, &device).unsqueeze::<4>();
        let tr_bh = (sigma_bhrr.clone() * eye_11rr.clone())
            .sum_dim(3)
            .sum_dim(2)
            .reshape([batch, nheads]);
        let dtype = self.m2_bhrr.dtype();
        let min_positive = dtype
            .finfo()
            .expect("state moments are a float dtype")
            .min_positive;
        let scale_bh = tr_bh.clamp_min(min_positive).detach();
        let sigma_hat = sigma_bhrr / scale_bh.reshape([batch, nheads, 1, 1]);
        let tr1_hat_bh = (sigma_hat.clone() * eye_11rr)
            .sum_dim(3)
            .sum_dim(2)
            .reshape([batch, nheads]);

        // tr(Σ̂²) of the mixed Hermitian [[M, X], [Xᴴ, U]] = Σ|M_ab|² + 2‖X‖² + ‖U‖².
        let mut tr2_hat_bh = match blocks {
            Blocks::Complex { num_pairs } => {
                let np = num_pairs;
                let xx = sigma_hat.clone().narrow(2, 0, np).narrow(3, 0, np);
                let yy = sigma_hat.clone().narrow(2, np, np).narrow(3, np, np);
                let xy = sigma_hat.clone().narrow(2, 0, np).narrow(3, np, np);
                // A = Σ(xxᵀ + yyᵀ), S = Σ(xyᵀ − yxᵀ); Σ symmetric ⇒ (yx)_ab = (xy)_ba.
                let a = xx + yy;
                let s = xy.clone() - xy.transpose();
                fro2_bh(a) + fro2_bh(s)
            }
            Blocks::Quaternion { num_blocks } => {
                let j = num_blocks;
                let q_bhjaja = sigma_hat
                    .clone()
                    .narrow(2, 0, 4 * j)
                    .narrow(3, 0, 4 * j)
                    .reshape([batch, nheads, j, 4, j, 4]);
                // Component (α, β) sub-block Σ[(4j+α), (4k+β)] as [b, h, J, J].
                let c = |alpha: usize, beta: usize| -> Tensor<4> {
                    q_bhjaja
                        .clone()
                        .narrow(3, alpha, 1)
                        .narrow(5, beta, 1)
                        .reshape([batch, nheads, j, j])
                };
                // Components of M_jk = Σ q̄ⱼqₖ (Hamilton product, conjugate left).
                let w = c(0, 0) + c(1, 1) + c(2, 2) + c(3, 3);
                let x = c(0, 1) - c(1, 0) - c(2, 3) + c(3, 2);
                let y = c(0, 2) + c(1, 3) - c(2, 0) - c(3, 1);
                let z = c(0, 3) - c(1, 2) + c(2, 1) - c(3, 0);
                fro2_bh(w) + fro2_bh(x) + fro2_bh(y) + fro2_bh(z)
            }
        };
        if rotated < state_rank {
            let tail = state_rank - rotated;
            let cross = sigma_hat.clone().narrow(2, 0, rotated).narrow(3, rotated, tail);
            let u = sigma_hat.narrow(2, rotated, tail).narrow(3, rotated, tail);
            tr2_hat_bh = tr2_hat_bh + fro2_bh(cross) * 2.0 + fro2_bh(u);
        }
        let tr2_hat_bh = tr2_hat_bh.clamp_min(crate::utils::div_eps(dtype));
        tr1_hat_bh.clone() * tr1_hat_bh / tr2_hat_bh
    }

    /// Raw uncentered state magnitude `tr Σ = trace(m2)/count` per
    /// `(batch, head)` — the mean squared state magnitude `⟨‖h‖²⟩`, which is
    /// [`Self::pr`]'s numerator scale. Reported alongside PR to tell a genuine
    /// rank-1 state (`PR → 1`, magnitude healthy) apart from a state
    /// collapsing toward zero (where `pr`'s `1e-12` denominator clamp drags
    /// the ratio below its true floor of 1).
    ///
    /// # Shape
    /// - output: `[batch, nheads]`
    pub fn trace(&self) -> Tensor<2> {
        let [batch, nheads, state_rank, _] = self.m2_bhrr.dims();
        assert!(self.count > 0, "state moments hold no samples");
        let eye_11rr = Tensor::<2>::eye(state_rank, &self.m2_bhrr.device()).unsqueeze::<4>();
        (self.m2_bhrr.clone() * eye_11rr)
            .sum_dim(3)
            .sum_dim(2)
            .reshape([batch, nheads])
            / self.count as f32
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
