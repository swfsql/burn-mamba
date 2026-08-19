//! Muon support: which weights the optimizer may touch, and how the fused
//! projections are split before it sees them.
//!
//! [Muon](burn::optim::Muon) replaces a 2-D weight's momentum update with the
//! nearest orthogonal matrix (Newton–Schulz). That only makes sense for a
//! parameter that *is* one linear map, which rules out three things in this
//! crate:
//!
//! 1. **Rank ≠ 2.** Burn's `Muon::step` asserts `D == 2`, so biases, norm gains,
//!    `dt_bias`/`D`, the Mamba-1/2 depthwise conv weight, and every MIMO/bias
//!    3-D tensor must stay on the fallback optimizer. Passing one to Muon
//!    panics — which is why the plan built here is an **allowlist**: only the
//!    weights named by a [`ProjSpec`] are moved off AdamW.
//! 2. **Embedding-like matrices.** The token embedding, the LM head, the
//!    `LatentNetwork` input/output projections and the class-token/latent tables
//!    are rank-2 but are lookup/readout tables at the model boundary; the usual
//!    Muon recipe keeps them on AdamW. They are simply never listed.
//! 3. **Fused projections.** Every family concatenates several independent maps
//!    into one `Linear` (`in_proj` most of all). Orthogonalising that
//!    concatenation ties together maps that share nothing but an allocation, so
//!    the plan carries the column seams and [`Segmented`] applies Muon
//!    **per block**. The model is untouched — the forward pass keeps its single
//!    fused GEMM.
//!
//! Within a fused projection one more distinction applies: blocks that emit
//! *per-head scalars* (Δ, `A`, `λ`) are gains rather than feature maps and stay
//! on AdamW ([`ProjSegment::adamw`]).
//!
//! ## Why the 3-D tensors are not "stacked matrices"
//!
//! MIMO's *math* is R independent SSMs summed together, so a `[nheads,
//! mimo_rank, per_head_dim]` tensor looks like a stack of matrices that a
//! stack-aware Muon could take a slice at a time. It is not: the paper
//! deliberately avoids instantiating the R maps, because that would multiply
//! every per-head projection's parameter count by R (`DP → DPR` *per head*).
//! Instead it keeps the SISO projection and element-wise scales its output to
//! size R with a learnable vector (`DP + PR` per head) — in the appendix's
//! contraction notation `X = contract(PR, P → PR)(W_X, X')`, where `P` appears
//! in both inputs *and* the output and so is never contracted. `mimo_x`/`mimo_z`
//! /`mimo_o` are therefore **diagonals**, and their matrix shape is a layout
//! coincidence; orthogonalising one would constrain a set of gains. The part of
//! MIMO that really is an R-fold matrix expansion is B/C (`DN → DNR`), which
//! lives in `in_proj` and is already Muon's. The remaining 3-D tensors are a
//! bias (`b_bias_hmr`/`c_bias_hmr`), an initial condition (`init_state_hpr`), or
//! a depthwise filter (the Mamba-1/2 conv) — diagonal or embedding-like again.
//!
//! The seams a [`ProjSpec`] carries are therefore **per named sub-projection**,
//! not per head/group/MIMO rank — the same boundaries the forward's
//! `split_into` uses, and the usual Muon convention (a transformer's `W_q` goes
//! to Muon with all heads fused). A caller who wants to test a finer split can
//! just hand [`ProjSpec::block`] more segments.
//!
//! ## Usage
//!
//! ```ignore
//! let plan = model_config.muon_plan();
//! let mut optim = plan.build(&adamw_config, &muon_config);
//! // ... then the usual `optim.step(lr, model, grads)`.
//! ```
//!
//! ## Learning rate
//!
//! Muon and AdamW share the one learning rate `optim.step` is called with.
//! [`AdjustLrFn::MatchRmsAdamW`](burn::optim::AdjustLrFn::MatchRmsAdamW) rescales
//! Muon's update so its per-element RMS is `0.2·lr` — AdamW's own ballpark — so
//! an LR schedule tuned for AdamW can be reused as-is. That is what
//! [`muon_config`] returns; the default
//! [`Original`](burn::optim::AdjustLrFn::Original) instead expects a Muon-specific
//! (typically much larger) LR.

/// [`MuonPlan::describe`]: the per-parameter optimizer assignment, for checking
/// a plan against a real model.
pub mod report;
/// [`Segmented`]: a different optimizer per column block of a fused weight.
pub mod segmented;
/// The column layout of the fused projection weights.
pub mod spec;

#[cfg(test)]
mod tests;

pub use segmented::{BlockState, Segmented, SegmentedState};
pub use spec::{BLOCK_CONTAINERS, ProjScope, ProjSegment, ProjSpec};

use burn::grad_clipping::GradientClipping;
use burn::optim::{AdamWConfig, AdjustLrFn, ModuleOptimizer, MuonConfig};

/// The Muon defaults this crate recommends: `MatchRmsAdamW` LR adjustment, so
/// Muon and AdamW can share one learning rate and one weight decay.
///
/// `weight_decay` should mirror the AdamW config's (Muon applies it after
/// orthogonalisation, with the *unadjusted* LR).
pub fn muon_config(weight_decay: f32) -> MuonConfig {
    MuonConfig::new()
        .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(weight_decay)))
}

/// Which weights Muon owns in a model, and where their fused columns split.
///
/// Built from a *block* config (via [`MambaBlockConfig::muon_projections`]), so
/// it is independent of the network topology: the specs are matched as path
/// substrings, and therefore cover a plain stack, a virtual-layer stack, and a
/// bidirectional stack alike.
///
/// [`MambaBlockConfig::muon_projections`]: crate::modules::MambaBlockConfig::muon_projections
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MuonPlan {
    /// The fused (or plain) weights Muon may touch.
    pub specs: Vec<ProjSpec>,
}

impl MuonPlan {
    /// An empty plan — everything stays on the fallback optimizer.
    pub fn empty() -> Self {
        Self::default()
    }

    /// A plan over the given specs.
    pub fn new(specs: Vec<ProjSpec>) -> Self {
        Self { specs }
    }

    /// Append another plan's specs.
    pub fn extend(mut self, other: Self) -> Self {
        self.specs.extend(other.specs);
        self
    }

    /// Add the [`GatedMlp`](crate::modules::GatedMlp) weights of a layer's
    /// optional feed-forward sub-block.
    pub fn with_mlp(self, mlp: Option<&crate::modules::GatedMlpConfig>) -> Self {
        match mlp {
            None => self,
            Some(mlp) => {
                let hidden = mlp.hidden();
                self.extend(Self::new(vec![
                    // fc1 fuses the SwiGLU value and gate halves.
                    ProjSpec::path(
                        "mlp.fc1.weight",
                        vec![
                            ProjSegment::muon("value", hidden),
                            ProjSegment::muon("gate", hidden),
                        ],
                    ),
                    ProjSpec::path_whole("mlp.fc2.weight", mlp.d_model),
                ]))
            }
        }
    }

    /// Drop every segment named `name` from Muon's ownership (it falls back to
    /// AdamW). Lets a caller opt a sub-projection out without rebuilding the plan.
    pub fn without_segment(mut self, name: &str) -> Self {
        for spec in &mut self.specs {
            for segment in &mut spec.segments {
                if segment.name == name {
                    segment.muon = false;
                }
            }
        }
        self
    }

    /// Assemble the [`ModuleOptimizer`]: AdamW everywhere, Muon on the planned
    /// weights.
    ///
    /// The AdamW group is the fallback (it must match everything), so any
    /// parameter the plan does not name — every 1-D and 3-D tensor included —
    /// keeps AdamW. `adamw`'s gradient clipping is applied to every group.
    pub fn build(&self, adamw: &AdamWConfig, muon: &MuonConfig) -> ModuleOptimizer {
        let mut optim = adamw.init();
        let clipping: Option<GradientClipping> = optim.grad_clipping().cloned();

        for spec in &self.specs {
            if !spec.has_muon() {
                continue;
            }
            let group = spec.param_group();
            optim = if spec.is_whole_muon() {
                // Unfused: stock Muon, stock state.
                optim.with_group(group, muon.build(), clipping.clone())
            } else {
                optim.with_group(
                    group,
                    Segmented::new(spec, muon.build(), adamw.build(), 1),
                    clipping.clone(),
                )
            };
        }

        optim
    }
}
