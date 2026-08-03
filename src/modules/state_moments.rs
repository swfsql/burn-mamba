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
        let tr_bh = (sigma_bhrr.clone() * eye_11rr)
            .sum_dim(3)
            .sum_dim(2)
            .reshape([batch, nheads]);
        let tr2_bh = sigma_bhrr
            .powf_scalar(2.0)
            .sum_dim(3)
            .sum_dim(2)
            .reshape([batch, nheads])
            .clamp_min(1e-12);
        tr_bh.clone() * tr_bh / tr2_bh
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
