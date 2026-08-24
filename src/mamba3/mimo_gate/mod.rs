//! Data-dependent soft mixing of Mamba-3's MIMO ranks — the alternative to the
//! paper's uniform sum.
//!
//! # What is being mixed
//!
//! A rank-`M` MIMO SSM is `M²` SISO SSMs sharing one recurrence (paper
//! eqs. 42–44): `M` of them *write* into a shared state, and each of the `M`
//! *readouts* sees the whole of it. Both directions are plain sums:
//!
//! ```text
//!   hₜ    = αₜhₜ₋₁ + Σⱼ B̄ₜ[j] ⊗ vₜ[j]                          (write sum)
//!   yₜ[i] = C̃ₜ[i]ᵀ hₜ                                          (per-rank readout)
//!   outₜ  = Σᵢ mimo_o[i] ⊙ silu(zₜ ⊙ mimo_z[i]) ⊙ yₜ[i]        (read merge)
//! ```
//!
//! The write sum is *uniform* — nothing weights rank `j` per token — and the
//! read merge's cross-rank profile comes from data-independent scales
//! (`mimo_o`, `mimo_z`) warping one shared gate stream `z`. This module makes
//! both weightings **data-dependent**, in the shape of
//! [`MultiGateResidual`](crate::modules::MultiGateResidual)'s two halves:
//!
//! | Multi-Gate Residuals (depth-stream axis) | here (MIMO rank axis) |
//! |---|---|
//! | mixer `βᵢ = σ(⟨w⁽ᵝ⁾, RMSNorm(sᵢ)⟩/√d + bᵢ)` — how much of `F_l` enters stream `i` | **write gate** `wₜ[m]` — how much of token `t` enters rank `m` |
//! | aggregator `αᵢ = softmax_i ⟨w⁽ᵅ⁾, RMSNorm(sᵢ)⟩/√d` | **read pool** `αₜ[m]` over the readouts `yₜ[m]` |
//!
//! Both arms are one [`MimoGateArm`]: a learnable per-head query scored against
//! the RMS-normalised content ([`normed_score`], shared with MGR), plus a
//! per-`(head, rank)` bias, squashed by [`GateKind`].
//!
//! # What is *not* mixed: the dynamics themselves
//!
//! `α`, `Δ`, `λ` and the rotation are projected **per head only** — never per
//! rank. That is what lets the `M` inner SSMs collapse into a single `[N×P]`
//! state (`Σⱼ αhⱼ = α Σⱼ hⱼ`). Give rank `m` its own decay or its own rotation
//! and the states no longer telescope: state bytes go `N·P → M·N·P` per head
//! and decode traffic multiplies by `M`, inverting the arithmetic-intensity
//! argument that motivates MIMO in the first place. So this module mixes what
//! goes *into* and comes *out of* one shared recurrence; per-rank dynamics
//! would be a different architecture, not a config flag.
//!
//! # Why it is free
//!
//! Both arms are pointwise in `t` and cost `O(b·s·m·h)` scalars.
//!
//! * The **write gate** is folded into `B` right after QK-norm. Since
//!   `Σₘ (wₘBₘ) ⊗ vₘ ≡ Σₘ Bₘ ⊗ (wₘvₘ)`, gating the key is gating the write —
//!   and `B` is the factor that already carries the rank axis *in the cache*
//!   (`k_state_bmhr`), where `v_state_bhp` does not. So the shifted β-term, the
//!   single-ssd boundary seed and the decode recurrence all pick up the
//!   previous token's gate with **no cache-schema change and no kernel
//!   change**, and both SSD pathways stay exactly equivalent. The `D` skip,
//!   which bypasses the state, is untouched — correctly, since this gate is
//!   about writing *to* the state.
//! * The **read pool** lives in the block tail
//!   ([`Mamba3::mimo_merge`](crate::mamba3::mamba3::Mamba3)), after the SSD.
//!
//! Nothing is added to the state, so decode wall-clock — MIMO's whole point —
//! is unchanged. The four parameters total `nheads·(N + P + 2M)`, negligible
//! beside the `d_model·N·M` the B projection already spends, which keeps the
//! paper's "no multiplicative parameter growth" property.
//!
//! # Identity at initialisation
//!
//! Queries and biases start at zero, and both squashings are **mean-one**
//! there: `2σ(0) = 1` and `M·softmax(0) = 1`. A gated block is therefore
//! bit-exact with [`MimoMix::Sum`] at init (asserted by
//! `tests::gated_equals_sum_at_init`), so enabling it perturbs no
//! initialisation and no loaded checkpoint's forward — only its trainability.
//!
//! # Scoring the *pre-rotation* `B`
//!
//! The write arm scores `B` **before** the transition rotation is applied.
//! Rotating first would make `⟨w, R̄ₜBₜ⟩` swing with the *cumulative* rotation —
//! an absolute-frame quantity, i.e. exactly the position-like reading of the
//! rotation that [`crate::mamba3::rotation`] exists to deny. Pre-rotation, the
//! score depends on the token alone. The gate itself is a scalar and commutes
//! with the rotation, so *where* it is applied is a free choice; where it is
//! *scored* is not.

use crate::modules::{normed_score, score_scale};
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};

/// How one arm's scores become mixing weights over the `mimo_rank` axis.
///
/// Both are **mean-one at zero logits**, which is what makes a fresh gated
/// block identical to [`MimoMix::Sum`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GateKind {
    /// `wₘ = 2σ(scoreₘ) ∈ (0, 2)` — each rank decides independently, as in
    /// MGR's mixer (the only gate that module implements). Ranks do not
    /// compete: all of them may open at once.
    #[default]
    Independent,
    /// `wₘ = M · softmax_m(score) ∈ (0, M)`, `Σₘ wₘ = M` — the ranks compete
    /// for a fixed budget, as in MGR's aggregator. On the write arm this is a
    /// router: the token's total write energy is conserved and it must choose
    /// *which* rank receives it. Carries the usual routing risk of collapsing
    /// onto a few ranks; the zero-init (uniform) start and the `1/√width`
    /// temperature are the only mitigations here — there is no load-balancing
    /// loss.
    Competitive,
}

/// How a Mamba-3 block combines its `mimo_rank` inner SSMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MimoMix {
    /// The paper form: a uniform write sum and the fixed `mimo_o` read merge.
    #[default]
    Sum,
    /// Data-dependent mixing on either or both directions (module header).
    /// `None` leaves that direction exactly as [`MimoMix::Sum`]; at least one
    /// must be `Some`, and the block must be MIMO (`mimo_rank > 1`).
    Gated {
        /// Weights the per-rank **write** into the shared state.
        write: Option<GateKind>,
        /// Pools the per-rank **readouts** into the block output.
        read: Option<GateKind>,
    },
}

/// One direction's gate: a per-head query scored against RMS-normalised
/// content, plus a per-`(head, rank)` bias.
///
/// The scored content is `[‥, mimo_rank, nheads, width]` — `state_rank` wide
/// for the write arm (it scores `B`), `per_head_dim` for the read arm (it
/// scores the readouts `y`). Rank-generic: `D = 5` over a sequence, `D = 4` for
/// a single token, with the rank axis always at `D-3`.
#[derive(Module, Debug)]
pub struct MimoGateArm {
    /// Query `w ∈ ℝ^{nheads × width}`, `[nheads, width]`; initialised to zero.
    ///
    /// **Per head**, not shared: `B` is GQA-expanded, so heads inside one group
    /// carry identical content up to `b_bias_hmr`, and a single shared query
    /// would hand them identical scores — the same symmetry MGR avoids by
    /// seeding its streams distinctly.
    pub query_hf: Param<Tensor<2>>,
    /// Per-`(head, rank)` score bias, `[nheads, mimo_rank]`; initialised to
    /// zero. A learnable prior over the ranks, MGR's `b⁽ᵝ⁾` per head.
    pub bias_hm: Param<Tensor<2>>,
    /// How scores become weights.
    #[module(skip)]
    pub kind: GateKind,
    /// MIMO rank `M` — the width of the mixed axis.
    #[module(skip)]
    pub mimo_rank: usize,
}

impl MimoGateArm {
    /// Mixing weights `[‥, mimo_rank, nheads, 1]` for content
    /// `[‥, mimo_rank, nheads, width]`, ready to broadcast back onto it.
    pub fn weights<const D: usize>(&self, content_mhf: Tensor<D>) -> Tensor<D> {
        let [rank_axis, head_axis, feat_axis] = [D - 3, D - 2, D - 1];
        let [nheads, width] = self.query_hf.dims();
        assert_eq!(
            content_mhf.dims()[rank_axis], self.mimo_rank,
            "gated content must carry the full mimo_rank axis"
        );
        assert_eq!(content_mhf.dims()[head_axis], nheads);
        assert_eq!(content_mhf.dims()[feat_axis], width);

        // Query → `[1, ‥, 1, nheads, width]`: broadcast over rank (and batch /
        // sequence), full width on the feature axis.
        let query = self.query_hf.val().unsqueeze::<D>();
        // Bias → `[1, ‥, 1, mimo_rank, nheads, 1]`, matching the score's shape.
        let bias = self
            .bias_hm
            .val()
            .swap_dims(0, 1) // [mimo_rank, nheads]
            .unsqueeze_dim::<3>(2) // [mimo_rank, nheads, 1]
            .unsqueeze::<D>();
        let score = normed_score(content_mhf, query, score_scale(width)) + bias;

        match self.kind {
            GateKind::Independent => sigmoid(score) * 2.0,
            GateKind::Competitive => softmax(score, rank_axis) * self.mimo_rank as f64,
        }
    }

    /// Score `content` and scale it by its own weights — the write arm's whole
    /// job.
    pub fn apply<const D: usize>(&self, content_mhf: Tensor<D>) -> Tensor<D> {
        content_mhf.clone() * self.weights(content_mhf)
    }
}

/// The per-block gate parameters selected by [`MimoMix::Gated`]: up to one
/// [`MimoGateArm`] per direction.
#[derive(Module, Debug)]
pub struct MimoGate {
    /// Weights the per-rank write into the shared state (scores `B`).
    pub write: Option<MimoGateArm>,
    /// Pools the per-rank readouts into the block output (scores `y`).
    pub read: Option<MimoGateArm>,
}

/// Allocates one [`MimoGateArm`] — zero query, zero bias (the mean-one,
/// `Sum`-identical start).
#[derive(Config, Debug)]
pub struct MimoGateArmConfig {
    /// Number of heads.
    pub nheads: usize,
    /// MIMO rank `M`.
    pub mimo_rank: usize,
    /// Width of the scored content: `state_rank` (write) or `per_head_dim` (read).
    pub width: usize,
    /// How scores become weights.
    pub kind: GateKind,
}

impl MimoGateArmConfig {
    /// Allocate the arm's parameters on `device`.
    pub fn init(&self, device: &Device) -> MimoGateArm {
        MimoGateArm {
            query_hf: Initializer::Zeros.init::<2, _>([self.nheads, self.width], device),
            bias_hm: Initializer::Zeros.init::<2, _>([self.nheads, self.mimo_rank], device),
            kind: self.kind,
            mimo_rank: self.mimo_rank,
        }
    }
}

impl MimoMix {
    /// Build the block's [`MimoGate`], or `None` for [`MimoMix::Sum`].
    ///
    /// Panics for a gated SISO block: at `mimo_rank == 1` there is nothing to
    /// mix (a softmax over one rank is identically `1`), so the request is a
    /// configuration error rather than a silent no-op.
    pub fn init(
        &self,
        nheads: usize,
        mimo_rank: usize,
        state_rank: usize,
        per_head_dim: usize,
        device: &Device,
    ) -> Option<MimoGate> {
        let (write, read) = match self {
            MimoMix::Sum => return None,
            MimoMix::Gated { write, read } => (*write, *read),
        };
        assert!(
            mimo_rank > 1,
            "MimoMix::Gated needs mimo_rank > 1 — a SISO block has a single rank to mix"
        );
        assert!(
            write.is_some() || read.is_some(),
            "MimoMix::Gated with neither direction gated is MimoMix::Sum"
        );
        let arm = |kind, width| {
            MimoGateArmConfig::new(nheads, mimo_rank, width, kind).init(device)
        };
        Some(MimoGate {
            write: write.map(|k| arm(k, state_rank)),
            read: read.map(|k| arm(k, per_head_dim)),
        })
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
