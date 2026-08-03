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

/// Raw (un-normalised) first/second moments of the per-token SSM states of
/// one block's `forward` pass, pooled over tokens and `per_head_dim` rows.
///
/// Produced by `forward_with_state_moments` (currently Mamba-2 only) without
/// ever materialising the per-token states — see
/// `Mamba2SsdInput::state_moments`.
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
