//! # Mamba-3 SSM Block — Exponential-Trapezoidal SSD with Data-Dependent RoPE
//!
//! This module implements the core **Mamba-3 layer** from the paper
//! *"Mamba-3: Improved Sequence Modeling using State Space Principles"*.
//!
//! Mamba-3 adds three independent extensions to the Mamba-2 SSD recurrence:
//!
//! 1. trapezoidal discretisation,
//! 2. a **complex-valued state transition**, implemented as data-dependent
//!    rotary embeddings on B and C (the "RoPE trick"),
//! 3. MIMO (multiple-input multiple-output) projection.
//!
//! Sections 1–3 show each one alone, and section 4 combines them. Section 5
//! adds this crate's fourth dial, MambaProduct.
//!
//! ## 1. Trapezoidal recurrence (SISO, no RoPE, no MIMO — Proposition 1)
//!
//! ```text
//!   hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ xₜ₋₁ᵀ + γₜ Bₜ xₜᵀ   (state update)
//!   yₜ = Cₜᵀ hₜ + D xₜ                          (output)
//! ```
//!
//! where the trapezoidal coefficients are
//!
//! ```text
//!   αₜ = exp(Δₜ Aₜ)                — decay (Aₜ < 0, data-dependent)
//!   βₜ = (1 − λₜ) Δₜ αₜ            — left-endpoint weight (Bₜ₋₁ xₜ₋₁ contribution)
//!   γₜ = λₜ Δₜ                      — right-endpoint weight (Bₜ xₜ contribution)
//! ```
//!
//! with `λₜ = σ(λ̂ₜ) ∈ (0, 1)`, the left/right split of the trapezoid.
//! `λ ≡ 1` gives the Mamba-2 (Euler / right-endpoint) form.
//!
//! [`Mamba3Config::trapezoid`] selects *which* earlier sample the `βₜ` tap
//! reads. At `micro_steps = 1` (below), `t−1` is the only choice. At `u > 1`:
//!
//! - the default [`Trapezoid::HorizontalCarryOver`] reads the recurrence step
//!   just before,
//! - [`Trapezoid::Vertical`] reads the same micro-step of the previous
//!   **token**, with one lag at every tap site,
//! - two members read **both**. A second projected scalar `μₜ = σ(μ̂ₜ)` splits
//!   `(1 − λₜ)Δₜ` between them. One of the two gates its lag-1 tap to within a
//!   token.
//!
//! The tap and its mass are one choice. A closed tap gives its share back (to
//! the other tap, or to `γ`), so the mass of the step is `Δₜ` under every
//! member. See the header of [`Trapezoid`].
//!
//! ## 2. Complex transition, a.k.a. "data-dependent RoPE" (no trapezoid, no MIMO
//! — paper section *Complex-Valued SSMs*)
//!
//! Despite the name, this is **not a positional encoding**. It is the
//! *imaginary part of the state transition*. The `A` of Mamba-3 is complex
//! (`A + iϑ`). After discretisation, a complex SSM of state `N/2` is exactly a
//! real SSM of state `N` whose transition is the scalar decay `α` times a
//! block-diagonal of `2×2` rotations (paper Prop. *Complex-to-Real SSM
//! Equivalence*):
//!
//! ```text
//!   ρₜ = R(Δₜ · ϱ · π · tanh(ϑₜ)) ∈ SO(2)^{N/2}  — per-step rotation (data-dependent)
//!   hₜ = αₜ ρₜ hₜ₋₁ + Δₜ Bₜ xₜᵀ                   — rotational state update
//! ```
//!
//! `ϱ` is [`Mamba3Config::rotation_range`], the per-step bound in half-turns
//! per unit `Δ`. It is `2` here (a whole turn of the group per unit `Δ`) and
//! `1` in the reference. It is a gradient budget, not a reach limit (see its
//! docs).
//!
//! The **exponential** discretisation is necessary here, not only more
//! accurate than forward Euler. `|exp(iΔϑ)| = 1` exactly, so the transition is
//! orthogonal, and a tracked rotation does not decay or grow. With Euler,
//! `|1 + iΔϑ| > 1` spirals outward (`info/mamba-3/rotation-as-optimization.md`
//! §9.5).
//!
//! `αₜ` is a scalar (so it commutes with `ρₜ`), and each `ρₜ` is orthogonal.
//! So the *cumulative* rotation telescopes out of the recurrence, and B/C can
//! absorb it: the **"RoPE trick"** (paper Prop. *Complex SSM, Data-Dependent
//! RoPE Equivalence*). This module implements that form. The rotation never
//! touches the state (the cached `h` is in the rotated frame), and the SSD core
//! stays the plain scalar-decay kernel.
//!
//! ```text
//!   θₜ = θₜ₋₁ + Δₜ · ϱ · π · tanh(ϑₜ)    — cumulative angles (per-pair)
//!   Rₜ = R(θₜ) = ρₜ ⋯ ρ₁ ∈ SO(2)^{N/2}   — block-diagonal cumulative rotation
//!   B̃ₜ = Rₜ Bₜ,   C̃ₜ = Rₜ Cₜ              — rotated state-space projections
//! ```
//!
//! The standard recurrence then runs with `B̃ₜ`/`C̃ₜ` in place of `Bₜ`/`Cₜ`:
//!
//! ```text
//!   hₜ = Āₜ hₜ₋₁ + B̄̃ₜ xₜᵀ                (scalar-decay state update)
//!   yₜ = C̃ₜᵀ hₜ + D xₜ                   (output)
//! ```
//!
//! Because of orthogonality, the readout-vs-input similarity depends only on
//! the rotation *accumulated between* the two steps:
//!
//! ```text
//!   C̃ᵢᵀ B̃ⱼ = (Rᵢ Cᵢ)ᵀ (Rⱼ Bⱼ) = Cᵢᵀ R(θⱼ − θᵢ) Bⱼ
//! ```
//!
//! `θⱼ − θᵢ` is not a position. It is `Σ Δ·ϱ·π·tanh(ϑ)` over the steps between:
//! how far the transition itself rotated, driven by the inputs. It is the
//! relative position of vanilla RoPE only in the degenerate case of an
//! input-independent, constant per-step angle (`θⱼ − θᵢ = (j−i)·θ`). The data
//! dependence is the point: it gives the layer rotational state dynamics, and
//! thus state-tracking (parity, mod-k), which a real SSM with non-negative
//! eigenvalues provably cannot express. See [`crate::mamba3::rotation`]. Its
//! `Quaternion4D` applies the same argument to a non-abelian rotation group.
//!
//! ## 3. MIMO Extension (no trapezoid coefficients, no RoPE — `mimo_rank = M > 1`)
//!
//! With MIMO, B/C carry M parallel rank channels, and the state update is a
//! sum of M outer products. The readout gives M outputs, which the gate
//! combines back:
//!
//! ```text
//!   hₜ = Āₜ hₜ₋₁ + Σₘ B̄ₜ[m] ⊗ (xₜ ⊙ mimo_x[m])                  (state update)
//!   yₜ[m] = Cₜ[m]ᵀ hₜ + D · (xₜ ⊙ mimo_x[m])                    (per-rank output)
//!   outₜ  = Σₘ mimo_o[m] ⊙ silu(zₜ ⊙ mimo_z[m]) ⊙ yₜ[m]         (rank merge)
//! ```
//!
//! All ranks share the hidden state hₜ. Each rank writes to it independently,
//! and each rank reads the full shared state for its output.
//!
//! ## 4. Combined formulation (everything together)
//!
//! Trapezoid + RoPE + MIMO in one expression. `B̃ₜ[m] = Rₜ Bₜ[m]` and
//! `C̃ₜ[m] = Rₜ Cₜ[m]` are the RoPE-rotated MIMO projections. The `β` tap is at
//! the lag 1 of the default member (§1: a member with lag `l` reads `t−l`, and
//! a two-tap member adds a second such term):
//!
//! ```text
//!   hₜ = αₜ hₜ₋₁
//!      + βₜ Σₘ B̃ₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_x[m])
//!      + γₜ Σₘ B̃ₜ[m]   ⊗ (xₜ   ⊙ mimo_x[m])
//!
//!   yₜ[m] = C̃ₜ[m]ᵀ hₜ + D · (xₜ ⊙ mimo_x[m])
//!   outₜ  = Σₘ mimo_o[m] ⊙ silu(zₜ ⊙ mimo_z[m]) ⊙ yₜ[m]
//! ```
//!
//! ## 5. MambaProduct (`micro_steps = u > 1`)
//!
//! A fourth, independent dial: the dial of DeltaProduct, not its mechanism
//! (see [`crate::mamba3::product`] and
//! `info/mamba-3/rotation-as-optimization.md`). There are `u` recurrence
//! micro-steps per token. Each one is a full step of the above, with its own
//! projected `x`, `B`, `Δ`, `A`, `λ` and rotation. The transition of one
//! *token* is then the **product**
//!
//! ```text
//!   Mₜ = (∏ⱼ αₜ,ⱼ) · Rₜ,ᵤ ⋯ Rₜ,₁
//! ```
//!
//! and its write is a sum of `u` outer products staggered along it. The read
//! `C`, the gate `z` and the `D` skip stay per token. The block folds the
//! micro-steps into the sequence axis, so nothing below this line changes. See
//! [`crate::mamba3::product`] for what `u` buys per [`RotationKind`]. The
//! answer is very different for the abelian and non-abelian kinds.
//!
//! Implementation notes:
//!
//! - The double-ssd pathway splits the trapezoidal recurrence into a γ-SSD
//!   (the current sample) and one β-SSD per tap, at the lag of that tap (see
//!   [`crate::mamba3::double_ssd::ssd::ssd_path`]). The single-ssd pathway
//!   does the same work in one call.
//! - The rotation applies to B and C before the SSD calls, through the one
//!   entry point [`rotate_bc_forward`](crate::mamba3::rotation::rotate_bc_forward)
//!   ([`apply_rope`](crate::mamba3::rotation::rope::apply_rope) for the
//!   abelian kind, a quaternion scan for the others).
//! - The MIMO expansion multiplies the V tensor by the per-rank `mimo_x`
//!   projection.
//!
//! See also: [`crate::mamba3::double_ssd::double_ssd`] and [`crate::mamba3::single_ssd::single_ssd`].
//!
//! ## Notation / Dimension Keys
//!
//! In all Mamba-3 files, a tensor name has a suffix that gives its shape. The
//! letters are different from those of the paper and of the Python
//! implementation. The "Paper" column gives the symbol in the Mamba-3 paper.
//! The "Python" column gives the name in the reference implementation
//! (`mamba_ssm/modules/mamba3.py` in `state-spaces/mamba`).
//!
//! | Letter | Dimension | Paper | Python | Typical value |
//! |--------|-----------|-------|--------|---------------|
//! | `b`    | `batch` | — | `batch` | varies |
//! | `s`    | `sequence` length, **folded** = `tokens · u` | `T` | `seqlen` | varies |
//! | `t`    | `tokens` = `s`/`u` — the [read axis](crate::mamba3::product) | `T` | `seqlen` | varies |
//! | `u`    | `micro_steps` (MambaProduct) | — | — | 1 (stock) |
//! | `d`    | `d_model` | `D` | `d_model` | 768, 1024 |
//! | `i`    | `d_inner` = `expand`·`d_model` | `E·D` | `d_inner` | 2·`d_model` |
//! | `h`    | `nheads` | `H` | `nheads` | `d_inner` / `per_head_dim` |
//! | `p`    | `per_head_dim` | `P` | `headdim` | 64, 128 |
//! | `r`    | `state_rank` | `N` | `d_state` | 64, 256 |
//! | `m`    | `mimo_rank` | `M` | `mimo_rank` | 1, .., 8 |
//! | `n`    | `nchunks` = `sequence`/`chunk_len` | — | `nchunks` | varies |
//! | `g`    | `ngroups` | `G` | `num_bc_heads` | 1, .., `nheads` |
//! | `l`    | `chunk_len` | `Q` | `chunk_size` | 64, .., 256 |
//! | `a`    | `num_rope_angles` = `state_rank` / 2 (or `rope_dim` / 2) | — | `num_rope_angles` | varies |
//!
//! An uppercase letter is a relation (offset, multiple, concat, stack) of
//! lowercase letters. For example, `X` can be `x+1`, `x-1` or `x*2`, and `XY`
//! can be `x+y` or `x*y`.

use crate::mamba3::positive::{Gain, Tropical};
use crate::mamba3::prelude::*;
use crate::mamba3::rotation::RotationKind;
use crate::mamba3::trapezoid::Trapezoid;
use burn_stack::modules::sanity as san;
use burn_stack::modules::{RmsNorm, RmsNormConfig, RmsNormGated, RmsNormGatedConfig};
use burn::prelude::*;
use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
};
use burn_stack::utils::{UntiedParam, untied};

// ---------------------------------------------------------------------------
// Mamba3  (the SSM block)
// ---------------------------------------------------------------------------

/// The Mamba-3 SSM block.
///
/// The full Mamba-3 layer with exponential-trapezoidal discretization and
/// data-dependent RoPE, for SISO (mimo_rank=1) and MIMO (mimo_rank>1). It has
/// two execution modes:
///
/// - [`Self::forward`] — chunkwise SSD for training / prefill (the cache
///   variant selects the double- or single-SSD pathway)
/// - [`Self::step`]    — recurrent form for token-by-token decoding
#[derive(Module, Debug)]
pub struct Mamba3 {
    /// Input projection; width [`Mamba3Config::d_in_proj`].
    ///
    /// Output splits, with `u` = [`Self::micro_steps`]:
    /// `[z | x·u | B_raw·u | C_raw | dd_dt·u | dd_A·u | lambda_raw·u | mu_raw·u
    /// | rotation·u | noise·u | tropical_a·u | tropical_b·u]`
    ///
    /// The segments from `lambda_raw` on are optional, so they are at the end:
    ///
    /// - [`Trapezoid::None`] projects no `lambda_raw`,
    /// - a one-tap pattern projects no `mu_raw`,
    /// - [`RotationKind::Real1D`] projects no `rotation`,
    /// - every [`Gain`] except [`Gain::KalmanProjectedNoise`] projects no `noise`,
    /// - [`Tropical::None`] projects no `tropical_*`.
    ///
    /// The block projects each **per-micro-step** stream `u` times, and
    /// [`crate::mamba3::product`] folds them into the sequence. The gate `z`
    /// and the read `C` are per token. At the default `u = 1`, this is the
    /// stock `d_model → 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads
    /// + num_rotation_channels`.
    ///
    /// Under [`Mamba3Untied::InProjTail`], `in_proj` stops at `C_raw`, and the
    /// rest is in [`Self::in_proj_tail`]. [`Self::project_in`] reads the two
    /// as one.
    pub in_proj: Linear,

    /// The trailing per-micro-step segments of `in_proj`,
    /// `[dd_dt·u | dd_A·u | lambda_raw·u | mu_raw·u | rotation·u | noise·u |
    /// tropical_a·u | tropical_b·u]`, when [`Mamba3Untied::InProjTail`] unties
    /// them: one copy per application, along the output axis. `None` ⇒
    /// `in_proj` holds them.
    pub in_proj_tail: Option<Linear>,

    /// Per-head bias for the discretisation step size Δ.
    /// Shape: `[nheads]`
    pub dt_bias_h: Param<Tensor<1>>,

    /// Hard clamp applied to Δ after softplus.
    pub dt_limit: (f64, f64),

    /// Minimum absolute value of A: `A ∈ (−∞, −a_floor]`.
    pub a_floor: f64,

    /// Per-head skip (D) coefficient.
    /// Shape: `[nheads]`; initialised to ones.
    pub d_h: Param<Tensor<1>>,

    /// RMSNorm applied to the B projection (QK-Norm, no gating).
    /// Normalises over the `state_rank` dimension.
    pub b_norm: RmsNorm,

    /// RMSNorm applied to the C projection (QK-Norm, no gating).
    /// Normalises over the `state_rank` dimension.
    pub c_norm: RmsNorm,

    /// Learnable per-head, per-rank bias for B, added after QK-norm.
    /// Shape: `[nheads, mimo_rank, state_rank]`; initialised to ones.
    pub b_bias_hmr: Param<Tensor<3>>,

    /// Learnable per-head, per-rank bias for C, added after QK-norm.
    /// Shape: `[nheads, mimo_rank, state_rank]`; initialised to ones.
    pub c_bias_hmr: Param<Tensor<3>>,

    /// MIMO up-projection for x (values).
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`.  When SISO, this is `None`.
    pub mimo_x_hmp: Option<Param<Tensor<3>>>,

    /// MIMO up-projection for z (gate).
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`. When SISO, this is `None`.
    pub mimo_z_hmp: Option<Param<Tensor<3>>>,

    /// MIMO down-projection for the output.
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`. When SISO, this is `None`.
    pub mimo_o_hmp: Option<Param<Tensor<3>>>,

    /// Optional gated RMSNorm applied before the output projection.
    ///
    /// When `Some`, `RmsNormGated(y, z)` replaces the SiLU gate at the block
    /// tail. It normalises `y` over `per_head_dim` and gates with `SiLU(z)`.
    /// Present when `has_outproj_norm = true`.
    pub out_norm: Option<RmsNormGated>,

    /// Output projection: maps `d_inner → d_model`.
    pub out_proj: Linear,

    /// Optional learnable initial hidden state `h₀`.
    /// Shape: `[nheads, per_head_dim, state_rank]`
    ///
    /// Each `forward` adds it to the incoming cache state. Only the `Minimal`
    /// SSD path supports it: the two serial paths panic. `step` does not read
    /// it.
    pub init_state_hpr: Option<Param<Tensor<3>>>,

    /// `ln κₕ`, the per-head noise scale of the Kalman gate (`qₜ = κₕ·Δₜ`, see
    /// [`crate::mamba3::positive`]). `−∞` is the stock block.
    /// Shape: `[nheads]`. `None` under [`Gain::Projected`].
    ///
    /// The block holds `κ` in logs, not as `softplus` of a raw parameter. `κ`
    /// scales `q` over many decades (the doubt of a gap against that of a
    /// value), and an AdamW step in `ln κ` is a relative change at each of
    /// them. The cost: the stock block is at `ln κ = −∞`, a limit that training
    /// approaches but never reaches, and `∂/∂ln κ ∝ κ` fades on the way. The
    /// join is live at the init ([`Mamba3Config::kalman_kappa_init`]). It is
    /// exact only by assignment, which is how the tests reach it.
    pub kalman_log_kappa_h: Option<Param<Tensor<1>>>,

    /// `ωₕ`, the read exponent: the block scales the SSD readout by
    /// `(Λₜ + ε)^(−ωₕ)`. So `ω = 1` reads the estimate `η/Λ`, and `ω = 0` (the
    /// init) reads the stock information `η`. Shape: `[nheads]`. `None` under
    /// [`Gain::Projected`].
    ///
    /// It stays under `has_outproj_norm` too. The scale applies to the SSD
    /// readout before the `D` skip and the `c·e` of the register are added, so
    /// the per-head norm cannot divide it out: `ω` sets the share of the SSD
    /// against theirs.
    pub kalman_read_h: Option<Param<Tensor<1>>>,

    /// `eₕ`, the tropical register's readout: `yₜ,ₕ += cₜ,ₕ·eₕ`.
    /// Shape: `[nheads, per_head_dim]`; initialised to zeros. `None` under
    /// [`Tropical::None`].
    pub tropical_readout_hp: Option<Param<Tensor<2>>>,

    /// State rank — the latent dimension of the SSM hidden state.
    ///
    /// Paper: `N`. Python: `d_state`.
    pub state_rank: usize,

    /// Number of B/C groups. Must divide `nheads`.
    ///
    /// Paper: `G`. Python: `num_bc_heads`.
    pub ngroups: usize,

    /// Number of RoPE angle pairs (`rope_dim / 2`).
    ///
    /// Python: `num_rope_angles`.
    pub num_rope_angles: usize,

    /// Effective RoPE dimension (= `2 · num_rope_angles`). Always even and
    /// `≤ state_rank`. The rotation turns only the first `rope_dim` entries of
    /// B/C. `0` for [`RotationKind::Real1D`], which rotates nothing (like every
    /// other rotation count here).
    pub rope_dim: usize,

    /// MIMO rank. 1 = SISO (standard Mamba-3).
    ///
    /// Paper: `M`. Python: `mimo_rank`.
    pub mimo_rank: usize,

    /// Recurrence micro-steps per token — DeltaProduct's `u`
    /// (see [`Mamba3Config::micro_steps`] and [`crate::mamba3::product`]).
    /// `1` is stock Mamba-3.
    pub micro_steps: usize,

    /// Which transition rotation the block applies to `B`/`C` ([`RotationKind`]).
    /// A non-parameter constant: `#[module(skip)]` keeps it out of the record
    /// and keeps it unchanged through `load_record`/`to_device`/….
    #[module(skip)]
    pub rotation: RotationKind,

    /// Which earlier sample(s) the trapezoid's `β` tap reads ([`Trapezoid`]).
    /// A non-parameter constant, like [`Self::rotation`].
    #[module(skip)]
    pub trapezoid: Trapezoid,

    /// How the decay is formed ([`Gain`]). A non-parameter constant, like
    /// [`Self::rotation`].
    #[module(skip)]
    pub gain: Gain,

    /// Whether each head carries a tropical register ([`Tropical`]). A
    /// non-parameter constant, like [`Self::rotation`].
    #[module(skip)]
    pub tropical: Tropical,

    /// How far one step may rotate, in half-turns per unit `Δ`
    /// (see [`Mamba3Config::rotation_range`]).
    pub rotation_range: f64,

    /// Number of in-projection channels for the rotation parameters, per
    /// micro-step: `num_rope_angles` for `Complex2D`,
    /// `nheads·3·num_rotation_blocks` for the quaternion kinds, `0` for
    /// `Real1D`. At `0`, `in_proj` has no rotation segment.
    pub num_rotation_channels: usize,

    /// Number of quaternion blocks (`rope_dim / 4`). Only
    /// [`RotationKind::Quaternion4D`] / [`RotationKind::Rotor4D`] use it.
    pub num_quat_blocks: usize,

    /// Whether the `mimo_rank == 1` specialized *chunkwise* kernel is enabled
    /// (see [`Mamba3Config::siso_specialization`]).
    ///
    /// A performance knob, not a semantic one: both branches compute the same
    /// values and gradients. A non-parameter constant (`#[module(skip)]` keeps
    /// it out of the record).
    #[module(skip)]
    pub siso_specialization: bool,

    /// Whether the `mimo_rank == 1` specialized *per-token* kernels are enabled
    /// (see [`Mamba3Config::siso_specialization_decode`]). Same performance-only
    /// nature, opposite backend preference.
    #[module(skip)]
    pub siso_specialization_decode: bool,

    /// The parameters held once per application instead of tied
    /// ([`Mamba3Config::untied`]). A non-parameter constant, like
    /// [`Self::rotation`].
    #[module(skip)]
    pub untied: Vec<Mamba3Untied>,
}

impl Mamba3 {
    /// `d_inner = expand · d_model`.
    pub fn d_inner(&self) -> usize {
        // Inferred from `out_proj`
        let [d_inner, _d_model] = self.out_proj.weight.dims();
        d_inner
    }

    /// `nheads = d_inner / per_head_dim`.
    pub fn nheads(&self) -> usize {
        //  Inferred from `d_h`
        let [nheads] = self.d_h.dims();
        nheads
    }

    /// `per_head_dim = d_inner / nheads`.
    pub fn per_head_dim(&self) -> usize {
        self.d_inner() / self.nheads()
    }

    /// The block's rotation kind ([`RotationKind`]).
    pub fn rotation_kind(&self) -> RotationKind {
        self.rotation
    }

    /// Number of quaternion blocks of the rotation projection and of its
    /// cumulative scan: `num_quat_blocks · quat_factors`. For
    /// [`RotationKind::Rotor4D`] this is twice the 4-block count of the state,
    /// because its left and right factors share one stacked block axis.
    pub fn num_rotation_blocks(&self) -> usize {
        self.num_quat_blocks * self.rotation.quat_factors()
    }

    /// Width of the trailing rotation segment of the in-projection: the
    /// per-step [`Self::num_rotation_channels`] once per micro-step, because
    /// each micro-step turns the state by its own rotation.
    pub fn rotation_channels_total(&self) -> usize {
        self.micro_steps * self.num_rotation_channels
    }

    /// Width of the `λ` segment of the in-projection: one channel per (head,
    /// micro-step), or **`0`** under [`Trapezoid::None`], which has no `λ`.
    /// `helpers::split_trailing` removes it from the tail, just inside the
    /// rotation segment.
    pub fn lambda_channels_total(&self) -> usize {
        if self.trapezoid.has_beta_tap() {
            self.micro_steps * self.nheads()
        } else {
            0
        }
    }

    /// Width of the `μ` segment of the in-projection (the mix of the second
    /// tap): one channel per (head, micro-step) for the two-tap patterns.
    /// **`0`** for every other pattern, and for a two-tap one at `u = 1`,
    /// where the taps fold ([`Trapezoid::has_interior_tap`]).
    /// `helpers::split_trailing` removes it from the tail, just inside the `λ`
    /// segment.
    pub fn mu_channels_total(&self) -> usize {
        if self.trapezoid.has_interior_tap(self.micro_steps) {
            self.micro_steps * self.nheads()
        } else {
            0
        }
    }

    /// Width of the Kalman noise segment `r` of the in-projection: one channel
    /// per (head, micro-step) under [`Gain::KalmanProjectedNoise`], **`0`** for
    /// every other gain. `split_positive` removes it from the tail,
    /// just inside the tropical segments.
    pub fn noise_channels_total(&self) -> usize {
        if self.gain.projects_noise() {
            self.micro_steps * self.nheads()
        } else {
            0
        }
    }

    /// Width of the tropical segments `(a, b)` of the in-projection: two
    /// channels per (head, micro-step) under [`Tropical::MaxPlus`], **`0`**
    /// otherwise. It is the outermost trailing segment, so `split_positive`
    /// removes it first.
    pub fn tropical_channels_total(&self) -> usize {
        if self.tropical.is_on() {
            2 * self.micro_steps * self.nheads()
        } else {
            0
        }
    }

    /// Remove the trailing segments of the positive systems from an
    /// in-projection whose channel axis is `dim`. Returns
    /// `(rest, noise, tropical (a, b))`, each `None` when the block does not
    /// project it. The two tropical halves are still `u`-wide, `a` first.
    #[allow(clippy::type_complexity)]
    pub(crate) fn split_positive<const D: usize>(
        &self,
        proj: Tensor<D>,
        dim: usize,
    ) -> (Tensor<D>, Option<Tensor<D>>, Option<(Tensor<D>, Tensor<D>)>) {
        let (proj, tropical) = crate::mamba3::helpers::split_trailing(
            proj,
            self.tropical_channels_total(),
            dim,
        );
        let (proj, noise) =
            crate::mamba3::helpers::split_trailing(proj, self.noise_channels_total(), dim);
        let tropical = tropical.map(|ab| {
            let half = ab.dims()[dim] / 2;
            (ab.clone().narrow(dim, 0, half), ab.narrow(dim, half, half))
        });
        (proj, noise, tropical)
    }

    /// What the Kalman gate reads in addition to the discretisation. `None`
    /// under [`Gain::Projected`]. `carry_bh` is the `ln Λ` of the cache.
    ///
    /// # Shapes
    /// - `noise_bsh` : `[batch, len, nheads]`, present iff
    ///   [`Gain::KalmanProjectedNoise`]
    /// - `carry_bh`  : `[batch, nheads]`
    pub(crate) fn gain_input(
        &self,
        noise_bsh: Option<Tensor<3>>,
        carry_bh: Option<Tensor<2>>,
    ) -> Option<crate::mamba3::positive::kalman::GainInput> {
        let log_kappa_h = self.kalman_log_kappa_h.as_ref()?.val();
        Some(crate::mamba3::positive::kalman::GainInput {
            log_kappa_h,
            noise_bsh,
            carry_bh: carry_bh.expect("a Kalman gain keeps its ln Λ cache slot"),
        })
    }

    /// The fresh-sequence positive-system slots: `(ln Λ, c)`, each
    /// `[batch, nheads]` at [`LOG_ZERO`](crate::mamba3::positive::LOG_ZERO)
    /// (no evidence; the max of nothing), or `None` when the block has no such
    /// system.
    pub fn zero_positive_state(
        &self,
        batch: usize,
        device: &Device,
    ) -> (Option<Tensor<2>>, Option<Tensor<2>>) {
        crate::mamba3::positive::fresh_slots(self.gain, self.tropical, batch, self.nheads(), device)
    }

    /// The tail of the positive systems. The `forward` of both pathways and
    /// `step` (whose folded run is the `u` micro-steps of one token) share it.
    /// It computes:
    ///
    /// - the scan of the register,
    /// - the carry of each system for the next call (the last position of the
    ///   run),
    /// - the two readout ports at the read rows ([`Self::positive_read`]).
    ///
    /// Returns `(y, ln Λ carry, c carry)`.
    ///
    /// `end` is `(real folded positions per slot, the incoming ln Λ carry)`
    /// for a right-padded call. Each carry then comes from the last real
    /// position of its slot, or is the incoming one for a slot with none.
    ///
    /// # Shapes
    /// - `y_btmhp`           : `[batch, tokens, mimo_rank, nheads, per_head_dim]`
    /// - `log_precision_bsh` : `[batch, tokens·u, nheads]`, from the discretisation
    /// - `tropical_ab_bsh`   : the register's `(a, b)`, each `[batch, tokens·u, nheads]`
    /// - `tropical_carry_bh` : the cache's `c`, `[batch, nheads]`
    /// - `end`               : `[batch]`, and the cache's `ln Λ`, `[batch, nheads]`
    #[allow(clippy::type_complexity)]
    pub(crate) fn positive_tail(
        &self,
        y_btmhp: Tensor<5>,
        log_precision_bsh: Option<Tensor<3>>,
        tropical_ab_bsh: Option<(Tensor<3>, Tensor<3>)>,
        tropical_carry_bh: Option<Tensor<2>>,
        end: Option<(Tensor<1, Int>, Option<Tensor<2>>)>,
    ) -> (Tensor<5>, Option<Tensor<2>>, Option<Tensor<2>>) {
        let u = self.micro_steps;
        let tropical_bsh = tropical_ab_bsh.map(|(a_bsh, b_bsh)| {
            let carry_bh = tropical_carry_bh
                .clone()
                .expect("a tropical register keeps its cache slot");
            crate::mamba3::positive::tropical::register(a_bsh, b_bsh, carry_bh)
        });
        let (end_b, log_precision_carry_bh) = end.unzip();
        let last_bh = |t_bsh: &Tensor<3>, carry_bh: Option<Tensor<2>>| match (&end_b, carry_bh) {
            (Some(end_b), Some(carry_bh)) => crate::padding::window(
                Tensor::cat(vec![carry_bh.unsqueeze_dim::<3>(1), t_bsh.clone()], 1),
                1,
                end_b.clone(),
                1,
            )
            .squeeze_dim::<2>(1),
            _ => {
                let len = t_bsh.dims()[1];
                t_bsh.clone().narrow(1, len - 1, 1).squeeze_dim::<2>(1)
            }
        };
        let log_precision_bh = log_precision_bsh
            .as_ref()
            .map(|t_bsh| last_bh(t_bsh, log_precision_carry_bh.flatten()));
        let tropical_bh = tropical_bsh
            .as_ref()
            .map(|t_bsh| last_bh(t_bsh, tropical_carry_bh));
        let read_rows = |t_bsh: Tensor<3>| crate::mamba3::helpers::read_rows::<3, 4>(t_bsh, 1, u);
        let y_btmhp = self.positive_read(
            y_btmhp,
            log_precision_bsh.map(read_rows),
            tropical_bsh.map(read_rows),
        );
        (y_btmhp, log_precision_bh, tropical_bh)
    }

    /// The two readout ports of the positive systems, on the SSD readout,
    /// before the `D` skip, the gate and the rank merge:
    ///
    /// - `C`: `y ← y·(Λ + ε)^(−ω)` when the block has [`Self::kalman_read_h`],
    /// - `D`: `y ← y + c·e` when it has [`Self::tropical_readout_hp`].
    ///
    /// Both broadcast over the mimo ranks, which share the state.
    ///
    /// # Shapes
    /// - `y_btmhp`            : `[batch, tokens, mimo_rank, nheads, per_head_dim]`
    /// - `log_precision_bth`  : `[batch, tokens, nheads]`, `ln Λ` at the read rows
    /// - `tropical_bth`       : `[batch, tokens, nheads]`, `c` at the read rows
    pub(crate) fn positive_read(
        &self,
        y_btmhp: Tensor<5>,
        log_precision_bth: Option<Tensor<3>>,
        tropical_bth: Option<Tensor<3>>,
    ) -> Tensor<5> {
        let y_btmhp = match (&self.kalman_read_h, log_precision_bth) {
            (Some(omega_h), Some(log_precision_bth)) => {
                const LN_EPS: f32 = -13.815511; // ln 1e-6
                let ln_lambda_bth = crate::mamba3::positive::scan::lse(
                    log_precision_bth.clone(),
                    log_precision_bth.full_like(LN_EPS),
                );
                let scale_bth = (-(ln_lambda_bth * omega_h.val().unsqueeze::<3>())).exp();
                y_btmhp * scale_bth.unsqueeze_dims::<5>(&[2, 4])
            }
            _ => y_btmhp,
        };
        match (&self.tropical_readout_hp, tropical_bth) {
            (Some(e_hp), Some(c_bth)) => {
                y_btmhp + c_bth.unsqueeze_dims::<5>(&[2, 4]) * e_hp.val().unsqueeze::<5>()
            }
            _ => y_btmhp,
        }
    }

    /// Everything the discretisation needs, in one place: the tap pattern, the
    /// micro-steps (the period of its gates), and the two clamps. Every site
    /// that computes the masses of the trapezoid ([`forward`](Self::forward),
    /// [`step`](Self::step)) uses this.
    pub fn trapezoid_spec(&self) -> crate::mamba3::trapezoid::TrapezoidSpec {
        crate::mamba3::trapezoid::TrapezoidSpec {
            pattern: self.trapezoid,
            micro_steps: self.micro_steps,
            dt_limit: self.dt_limit,
            a_floor: self.a_floor,
        }
    }

    /// Everything the rotation needs, in one place: the algebra, the rotated
    /// width, and the per-step bound. Every site that computes a per-step
    /// rotation ([`forward`](Self::forward), [`step`](Self::step)) uses this.
    pub fn rotation_spec(&self) -> crate::mamba3::rotation::RotationSpec {
        crate::mamba3::rotation::RotationSpec {
            kind: self.rotation,
            rope_dim: self.rope_dim,
            range: self.rotation_range,
        }
    }

    /// Whether the specialized `mimo_rank == 1` **per-token** kernels run: the
    /// block is SISO **and** [`Mamba3Config::siso_specialization_decode`] is
    /// enabled.
    ///
    /// The chunkwise counterpart is [`Mamba3Config::siso_specialization`]. It
    /// goes to its kernel as a field of the SSD input bundle, not through a
    /// method.
    ///
    /// Only a performance choice (see
    /// [`Mamba3Config::siso_specialization_decode`]).
    pub fn use_siso_decode_kernels(&self) -> bool {
        self.mimo_rank == 1 && self.siso_specialization_decode
    }

    /// The output of the in-projection, in the layout that [`Self::in_proj`]
    /// documents (`[z | … | tropical_b·u]`): `in_proj` alone, or `in_proj`
    /// then [`Self::in_proj_tail`] under [`Mamba3Untied::InProjTail`].
    pub fn project_in<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match &self.in_proj_tail {
            None => self.in_proj.forward(x),
            Some(tail) => Tensor::cat(vec![self.in_proj.forward(x.clone()), tail.forward(x)], D - 1),
        }
    }

    /// The parameters held once per application ([`Self::untied`]), each with
    /// the axis its copies lie along. See [`burn_stack::utils::untied`].
    pub fn untied_params(&self) -> Vec<UntiedParam> {
        let axis0 = |p: &Param<Tensor<1>>| UntiedParam::new(p, 0);
        let axis0_hmx = |p: &Param<Tensor<3>>| UntiedParam::new(p, 0);
        self.untied
            .iter()
            .flat_map(|part| -> Vec<UntiedParam> {
                match part {
                    Mamba3Untied::InProjTail => self
                        .in_proj_tail
                        .iter()
                        .flat_map(|tail| {
                            std::iter::once(UntiedParam::new(&tail.weight, 1))
                                .chain(tail.bias.as_ref().map(axis0))
                        })
                        .collect(),
                    Mamba3Untied::DtBias => vec![axis0(&self.dt_bias_h)],
                    Mamba3Untied::D => vec![axis0(&self.d_h)],
                    Mamba3Untied::BNorm => vec![axis0(&self.b_norm.gamma)],
                    Mamba3Untied::CNorm => vec![axis0(&self.c_norm.gamma)],
                    Mamba3Untied::BBias => vec![axis0_hmx(&self.b_bias_hmr)],
                    Mamba3Untied::CBias => vec![axis0_hmx(&self.c_bias_hmr)],
                    Mamba3Untied::MimoX => self.mimo_x_hmp.iter().map(axis0_hmx).collect(),
                    Mamba3Untied::MimoZ => self.mimo_z_hmp.iter().map(axis0_hmx).collect(),
                    Mamba3Untied::MimoO => self.mimo_o_hmp.iter().map(axis0_hmx).collect(),
                    Mamba3Untied::OutNorm => self.out_norm.iter().map(|n| axis0(&n.gamma)).collect(),
                    Mamba3Untied::InitState => self.init_state_hpr.iter().map(axis0_hmx).collect(),
                    Mamba3Untied::KalmanKappa => self.kalman_log_kappa_h.iter().map(axis0).collect(),
                    Mamba3Untied::KalmanRead => self.kalman_read_h.iter().map(axis0).collect(),
                    Mamba3Untied::TropicalReadout => self
                        .tropical_readout_hp
                        .iter()
                        .map(|p| UntiedParam::new(p, 0))
                        .collect(),
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Mamba3Config  (hyperparameters and factory)
// ---------------------------------------------------------------------------

/// A [`Mamba3`] parameter that the block can hold once per application of its
/// real layer instead of tied across them (see [`burn_stack::utils::untied`]).
/// The big maps, the `z|x|B|C` head of `in_proj` and `out_proj`, are always
/// tied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Mamba3Untied {
    /// The trailing per-micro-step scalar and rotation segments of `in_proj`,
    /// `[Δ·u | A·u | λ·u | μ·u | rotation·u | noise·u | tropical·2u]`, moved
    /// into [`Mamba3::in_proj_tail`] (a second, small GEMM).
    InProjTail,
    /// The Δ bias [`Mamba3::dt_bias_h`].
    DtBias,
    /// The skip [`Mamba3::d_h`].
    D,
    /// The QK-norm gain of `B` ([`Mamba3::b_norm`]).
    BNorm,
    /// The QK-norm gain of `C` ([`Mamba3::c_norm`]).
    CNorm,
    /// The bias of `B`, [`Mamba3::b_bias_hmr`].
    BBias,
    /// The bias of `C`, [`Mamba3::c_bias_hmr`].
    CBias,
    /// The MIMO value up-projection [`Mamba3::mimo_x_hmp`] (`mimo_rank > 1`).
    MimoX,
    /// The MIMO gate up-projection [`Mamba3::mimo_z_hmp`] (`mimo_rank > 1`).
    MimoZ,
    /// The MIMO output down-projection [`Mamba3::mimo_o_hmp`] (`mimo_rank > 1`).
    MimoO,
    /// The gain of the output norm ([`Mamba3::out_norm`], `has_outproj_norm`).
    OutNorm,
    /// The learnable initial state [`Mamba3::init_state_hpr`]
    /// (`has_learnable_init_state`).
    InitState,
    /// The `ln κ` of the Kalman gate, [`Mamba3::kalman_log_kappa_h`] (a
    /// Kalman [`Gain`]).
    KalmanKappa,
    /// The Kalman read exponent [`Mamba3::kalman_read_h`] (a Kalman [`Gain`]).
    KalmanRead,
    /// The readout of the tropical register, [`Mamba3::tropical_readout_hp`]
    /// ([`Tropical::MaxPlus`]).
    TropicalReadout,
}

/// Hyperparameters for the Mamba-3 SSM block.
#[derive(Config, Debug)]
pub struct Mamba3Config {
    /// Model (hidden) dimension.
    ///
    /// Paper: `D`. Python: `d_model`.
    pub d_model: usize,

    /// State rank — the latent dimension of the SSM hidden state.
    /// **Must be even** for every rotating [`RotationKind`] (RoPE pairing).
    /// [`RotationKind::Real1D`] makes no pairs, so it also accepts an odd rank
    /// (down to the scalar state `1`).
    ///
    /// Paper: `N`. Python: `d_state`.
    #[config(default = 128)]
    pub state_rank: usize,

    /// Expansion factor for `d_inner = expand · d_model`.
    ///
    /// Paper: `E`. Python: `expand`.
    #[config(default = 2)]
    pub expand: usize,

    /// Head dimension. `per_head_dim = d_inner / nheads`.
    ///
    /// Paper: `P`. Python: `headdim`.
    #[config(default = 64)]
    pub per_head_dim: usize,

    /// Number of B/C groups. Must divide `nheads`.
    ///
    /// Paper: `G`. Python: `num_bc_heads`.
    #[config(default = 1)]
    pub ngroups: usize,

    /// MIMO rank. `1` = standard SISO Mamba-3.
    ///
    /// When `mimo_rank > 1`, the B/C projections have `mimo_rank` parallel rank
    /// channels. Three extra weights (`mimo_x_hmp`, `mimo_z_hmp`, `mimo_o_hmp`)
    /// give element-wise up/down projections in head-space across ranks. At
    /// init, a MIMO block is its SISO block (`info/mamba-3/mimo-as-batch.md`).
    ///
    /// Paper: `M`. Python: `mimo_rank`.
    #[config(default = 1)]
    pub mimo_rank: usize,

    /// **MambaProduct**: recurrence micro-steps per token — DeltaProduct's `u`.
    /// `1` (the default) is stock Mamba-3, byte for byte.
    ///
    /// Each micro-step is a full Mamba-3 step with its own projected `x`, `B`,
    /// `Δ`, `A`, `λ` and rotation. So the transition of one *token* is the
    /// **product** `(∏ⱼ αⱼ)·Rᵤ⋯R₁`, and its write is `u` staggered outer
    /// products. The read `C`, the gate `z`, the `D` skip and the output are
    /// per token. The cost is `u`× the recurrence. The state does not grow.
    ///
    /// What it buys depends on [`Self::rotation`]:
    ///
    /// - [`Real1D`](crate::mamba3::rotation::RotationKind::Real1D): the factors
    ///   are scalars and commute, so `u` widens only the write (the sequential
    ///   reading of what [`Self::mimo_rank`] does jointly).
    /// - [`Complex2D`](crate::mamba3::rotation::RotationKind::Complex2D): it
    ///   also multiplies the per-token angle reach by `u`, *without* putting
    ///   any single factor on the flat region of `tanh` (see
    ///   [`Self::rotation_range`]).
    /// - The non-abelian kinds: it composes generators that no single step can
    ///   express.
    ///
    /// See [`crate::mamba3::product`] for the full argument and the folding.
    #[config(default = 1)]
    pub micro_steps: usize,

    /// Which earlier sample(s) the `β` tap of the trapezoid reads: the **tap
    /// pattern** ([`Trapezoid`]).
    ///
    /// This choice exists only at [`Self::micro_steps`] `> 1`, where "the
    /// previous step" can mean the previous micro-step or the previous token.
    /// At `u = 1`, every pattern is equal to the default or switches the
    /// trapezoid off. It selects an *algorithm*, a *cache layout* and the
    /// number of per-head masses in the in-projection, not the value of a
    /// coefficient. See [`Trapezoid`] and
    /// `info/mamba-3/trapezoid-as-integration.md` §9.
    ///
    /// Defaults to [`Trapezoid::HorizontalCarryOver`] (one lag-1 tap).
    #[config(default = "crate::mamba3::trapezoid::Trapezoid::HorizontalCarryOver")]
    pub trapezoid: Trapezoid,

    /// How the decay of each head is formed ([`Gain`]):
    ///
    /// - projected from the token (stock), or
    /// - **computed** by a per-head Kalman filter. Its precision `Λ` adds up
    ///   the evidence that the head wrote. So the decay reads the history of
    ///   the head through a scalar that reads only the inputs, and the
    ///   chunkwise pass still works.
    ///
    /// See [`crate::mamba3::positive`]. Structural: a Kalman member allocates
    /// `κ` and `ω` per head and one cache slot. Under
    /// [`Gain::KalmanProjectedNoise`], it also adds one in-projection channel
    /// per (head, micro-step). Defaults to [`Gain::Projected`], the stock
    /// block.
    #[config(default = "crate::mamba3::positive::Gain::Projected")]
    pub gain: Gain,

    /// The initial `κ` of a Kalman [`Gain`] (`qₜ = κ·Δₜ`). It is small, so the
    /// block starts near stock with a live gradient. Stock itself is `κ = 0`,
    /// which the log parameterisation of `κ` cannot reach (see
    /// [`Mamba3::kalman_log_kappa_h`]). Ignored under [`Gain::Projected`].
    #[config(default = 1e-2)]
    pub kalman_kappa_init: f64,

    /// Whether each head has a tropical register that adds to its readout
    /// ([`Tropical`]): `cₜ = lse(cₜ₋₁ + aₜ, bₜ)` with `(a, b)` projected. This
    /// is a soft `max(cₜ₋₁ + aₜ, bₜ)`: counters clamped at a floor, running
    /// maxima and resets. A linear recurrence computes none of these. See
    /// [`crate::mamba3::positive`]. Defaults to [`Tropical::None`].
    #[config(default = "crate::mamba3::positive::Tropical::None")]
    pub tropical: Tropical,

    /// Minimum absolute value of A after clamping.
    #[config(default = "1e-4")]
    pub a_floor: f64,

    /// Minimum value of the initial Δ distribution.
    #[config(default = 1e-3)]
    pub dt_min: f64,

    /// Maximum value of the initial Δ distribution.
    #[config(default = 0.1)]
    pub dt_max: f64,

    /// Floor clamped onto sampled initial Δ values.
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,

    /// Hard clamp limits for Δ at runtime.
    #[config(default = "(0., 6.5504e+4)")]
    pub dt_limit: (f64, f64),

    /// Whether to add a bias term to the `in_proj` and `out_proj`.
    #[config(default = false)]
    pub has_proj_bias: bool,

    /// Whether to allocate a learnable initial SSM state `h₀`. Only the
    /// `Minimal` SSD path supports it (see [`Mamba3::init_state_hpr`]).
    #[config(default = false)]
    pub has_learnable_init_state: bool,

    /// Fraction of `state_rank` that the transition rotation turns (must be
    /// `0.5` or `1.0`).
    ///
    /// To disable the rotation, select the *kind* [`RotationKind::Real1D`], not
    /// a fraction of zero. A real transition projects no rotation channels and
    /// caches no accumulator. `Real1D` ignores this field.
    ///
    /// - `0.5`: partial rotation. Only `state_rank / 2` dimensions turn, and
    ///   the rest are unchanged. This is the reference value in `mamba3.py`.
    ///   Set it explicitly to reproduce that model.
    /// - `1.0` (default): full rotation. Every B/C dimension turns.
    ///
    /// The default turns everything, for all rotation kinds. A partial
    /// rotation is a capacity trade (it keeps "content" channels out of the
    /// turn), so a config must ask for it explicitly. For
    /// [`Quaternion4D`](RotationKind::Quaternion4D) and
    /// [`Rotor4D`](RotationKind::Rotor4D), the default also keeps the smallest
    /// legal `state_rank` (4, one quaternion block) usable, because a partial
    /// quaternion rotation must cover whole 4-blocks.
    #[config(default = 1.0)]
    pub rope_fraction: f64,

    /// Whether to apply a gated RMSNorm before the output projection.
    ///
    /// When `true`, a per-head [`RmsNormGated`] (group size = `per_head_dim`)
    /// replaces the SiLU gate at the end of the block. It normalises `y` and
    /// gates it with `SiLU(z)`. Same as the `is_outproj_norm` argument of the
    /// reference `mamba3.py`.
    #[config(default = false)]
    pub has_outproj_norm: bool,

    /// Which rotational-state algebra to use for the data-dependent rotation
    /// carried by the state transition and absorbed into `B`/`C` (see
    /// [`RotationKind`]).
    ///
    /// Defaults to the abelian [`Complex2D`](RotationKind::Complex2D) — the
    /// current Mamba-3 RoPE — for which the block is byte-for-byte unchanged.
    /// [`Quaternion4D`](RotationKind::Quaternion4D) selects the non-abelian
    /// quaternion rotation; its in-projection devotes
    /// `nheads · 3 · num_quat_blocks` channels to per-head quaternion
    /// generators in place of the shared `num_rope_angles` angle channels (see
    /// [`Self::num_rotation_channels`]). [`Rotor4D`](RotationKind::Rotor4D)
    /// selects the full `SO(4)` rotation of each 4-block and doubles that
    /// channel count (a left and a right generator per head and block).
    /// [`Real1D`](RotationKind::Real1D) removes the rotation altogether — no
    /// channels, no accumulator, a real transition.
    #[config(default = "crate::mamba3::rotation::RotationKind::Complex2D")]
    pub rotation: RotationKind,

    /// How far a single step may rotate the state, in **half-turns per unit
    /// `Δ`**: the per-step angle is bounded by `rotation_range · π · Δ`.
    ///
    /// The default `2.0` means "one unit of `Δ` can go once around the whole
    /// rotation group". This means the same for every kind, although their
    /// periods are different:
    ///
    /// - [`Complex2D`](RotationKind::Complex2D): `2π` is a full turn of
    ///   `SO(2)`.
    /// - [`Quaternion4D`](RotationKind::Quaternion4D): `2π` reaches every
    ///   element of `SU(2)`, whose period is `4π`. `q` and `−q` are the two
    ///   lifts of one `SO(3)` rotation, and they turn the state differently.
    /// - [`Rotor4D`](RotationKind::Rotor4D): the bound applies to **each
    ///   factor**. So at the default, each of `q` and `p` independently covers
    ///   all of `SU(2)`, and the pair reaches every element of
    ///   `SO(4) ≅ (SU(2)×SU(2))/±1`. A bound on the composite (half the
    ///   per-factor angle) would lose reach and gain nothing: the two plane
    ///   angles `a∓b` are periodic in `2π` anyway.
    ///
    /// The bound buys **gradients**, not reach. A rotation of exactly
    /// `range·π` is at the asymptote of `tanh`, where the gradient is
    /// **exactly zero** in f32 (`tanh(10) == 1.0`), not only small.
    /// State-tracking wants half-turns. At `range = 1`, where `π` *is* the
    /// bound, the optimiser can never reach the most useful rotation. At `2.0`
    /// it is at `tanh = 1/2`, in the interior.
    ///
    /// `1.0` is the abelian bound of the reference implementation
    /// (`Δ·π·tanh(ϑ)`). Set it explicitly to reproduce that model.
    /// [`Real1D`](RotationKind::Real1D) ignores it, because it never turns.
    #[config(default = 2.0)]
    pub rotation_range: f64,

    /// Whether to use the specialized `mimo_rank == 1` (SISO) **chunkwise**
    /// kernel: the same-step γ-correction
    /// ([`y_diag_correction`](crate::mamba3::single_ssd::ssd::diag::y_diag_correction)),
    /// forward *and* its analytic backward.
    ///
    /// At `mimo_rank == 1` the `m × m` Gram is a scalar. The general form then
    /// issues a `1×r×1` and a `1×1×p` GEMM per `(batch, nchunks, chunk_len,
    /// nheads)`: thousands of degenerate products. The SISO branch replaces all
    /// of them with one elementwise multiply and one `state_rank` reduction over
    /// the whole tensor. Removing the tiny GEMMs helps on every backend in
    /// `bench.md` (from ≈ neutral to ~9 % faster). Leave it on.
    ///
    /// **Values and gradients are identical either way.** This only selects the
    /// op mix. Ignored when `mimo_rank > 1`, where only the general branch
    /// applies. See also [`Self::siso_specialization_decode`], the per-token
    /// counterpart, which has a *different* trade-off.
    #[config(default = true)]
    pub siso_specialization: bool,

    /// Whether to use the specialized `mimo_rank == 1` (SISO) **per-token**
    /// kernels: the decode state update and readout
    /// (`helpers::mimo_outer_sum`, `Mamba3::step_readout`) and the single-SSD
    /// boundary-β seed.
    ///
    /// **Set this to `false` on the CPU backends.** Unlike the chunkwise flag,
    /// this one replaces *one, already efficient* batched matmul with a
    /// broadcast multiply, so the result depends on the backend. At the default
    /// size of `benches/layer.rs`, the `step` group is ~12 % faster on CUDA and
    /// about **3× slower on flex**. At these shapes, the broadcast-elementwise
    /// path of flex is more than 10× slower than its matmul. The default
    /// (`true`) suits the GPU backends. A CPU deployment that decodes should
    /// turn it off, and it can keep [`Self::siso_specialization`] on. That is
    /// why there are two flags.
    ///
    /// **Values and gradients are identical either way.** Ignored when
    /// `mimo_rank > 1`.
    #[config(default = true)]
    pub siso_specialization_decode: bool,

    /// The parameters held once per application of the block's real layer
    /// instead of tied across them ([`Mamba3Untied`]). Only virtual layers
    /// apply a real layer more than once. Each listed parameter must exist in
    /// this config. The layout depends only on this list:
    /// [`Mamba3Untied::InProjTail`] splits `in_proj` also for one application.
    /// So every real layer of a stack has the same Muon plan. See
    /// [`burn_stack::utils::untied`].
    #[config(default = "Vec::new()")]
    pub untied: Vec<Mamba3Untied>,
}

impl Mamba3Config {
    /// Inner (expanded) channel width: `expand · d_model`.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }
    /// Number of SSM heads: `d_inner / per_head_dim`.
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.per_head_dim
    }

    /// Effective RoPE dimension: the number of B/C channels that turn.
    /// `state_rank` for full RoPE (`rope_fraction = 1.0`), `state_rank / 2` for
    /// `rope_fraction = 0.5`, and `0` for [`RotationKind::Real1D`], which
    /// rotates nothing.
    pub fn rope_dim(&self) -> usize {
        if self.rotation == RotationKind::Real1D {
            return 0;
        }
        let mut d = (self.state_rank as f64 * self.rope_fraction) as usize;
        if !d.is_multiple_of(2) {
            d -= 1;
        }
        d
    }

    /// Number of RoPE rotation angles projected per head: `rope_dim / 2` (`0`
    /// for [`RotationKind::Real1D`]). `init` asserts that every rotating kind
    /// turns at least one pair, so this is `> 0` wherever the block reads it.
    pub fn num_rope_angles(&self) -> usize {
        self.rope_dim() / 2
    }

    /// Number of **state** quaternion blocks for the quaternion kinds:
    /// `rope_dim / 4`. For them, `init` asserts that it is a whole non-zero
    /// number. `0` for [`RotationKind::Real1D`]. Meaningless (and not read) for
    /// [`RotationKind::Complex2D`].
    pub fn num_quat_blocks(&self) -> usize {
        self.rope_dim() / 4
    }

    /// Number of in-projection channels for the rotation parameters, per
    /// [`RotationKind`]:
    ///
    /// - [`Real1D`](RotationKind::Real1D): none. The trailing `in_proj`
    ///   segment is absent, not zero-width (Burn has no zero-width tensors).
    /// - [`Complex2D`](RotationKind::Complex2D): `num_rope_angles` angle
    ///   channels, shared across heads and scaled per head by `Δ`, as in the
    ///   reference.
    /// - [`Quaternion4D`](RotationKind::Quaternion4D) /
    ///   [`Rotor4D`](RotationKind::Rotor4D):
    ///   `nheads · 3 · num_rotation_blocks` quaternion-generator channels. This
    ///   is one axis·angle generator **per head** and block (and, for
    ///   `Rotor4D`, per *factor*: one generator for the left quaternion and one
    ///   for the right), sent through
    ///   [`quat_from_scaled_axis`](crate::mamba3::rotation::quat_from_scaled_axis).
    ///   For a non-abelian transition, the axis holds the expressiveness. Heads
    ///   that share one axis and differ only in `Δ` track one word at different
    ///   speeds, not different words. See
    ///   [`generator_increment`](crate::mamba3::rotation::generator_increment).
    pub fn num_rotation_channels(&self) -> usize {
        match self.rotation {
            // A real transition has no rotation to parameterise.
            RotationKind::Real1D => 0,
            RotationKind::Complex2D => self.num_rope_angles(),
            // Three generator channels per (head, block) and **per factor**:
            // one factor for Quaternion4D, two for Rotor4D.
            RotationKind::Quaternion4D | RotationKind::Rotor4D => {
                self.nheads() * 3 * self.num_rotation_blocks()
            }
        }
    }

    /// Number of quaternion blocks of the *rotation projection and scan*:
    /// [`Self::num_quat_blocks`] times [`RotationKind::quat_factors`]. That is
    /// `2 · blocks` for [`RotationKind::Rotor4D`], whose left and right factors
    /// share one stacked block axis.
    pub fn num_rotation_blocks(&self) -> usize {
        self.num_quat_blocks() * self.rotation.quat_factors()
    }

    /// Total input projection output size.
    ///
    /// ```text
    ///   [ z | x·u | B·u | C | Δ·u | A·u | λ·u | μ·u | rotation·u | r·u | a·u | b·u ]
    /// ```
    ///
    /// That is `d_inner + u·d_inner + u·bc + bc + (2 … 7)·u·nheads +
    /// u·num_rotation_channels`, with `bc = ngroups·state_rank·mimo_rank` and
    /// `u` = [`Self::micro_steps`]. Only the **per-micro-step** segments widen.
    /// The gate `z` and the read `C` are per token (see
    /// [`crate::mamba3::product`]). At `u = 1`, the stock block (the default
    /// trapezoid, no positive system) has
    /// `2·d_inner + 2·bc + 3·nheads + num_rotation_channels`.
    ///
    /// A block can omit every segment from `λ` on (see [`Mamba3::in_proj`] for
    /// which setting omits which). `helpers::split_trailing` removes them in
    /// the reverse order.
    pub fn d_in_proj(&self) -> usize {
        let u = self.micro_steps;
        let bc = self.ngroups * self.state_rank * self.mimo_rank;
        self.d_inner() + u * self.d_inner() + u * bc + bc + self.d_in_proj_tail()
    }

    /// Width of the trailing per-micro-step segments of the in-projection,
    /// `[Δ·u | A·u | λ·u | μ·u | rotation·u | noise·u | tropical·2u]`:
    /// `(2 … 7)·u·nheads + u·num_rotation_channels`.
    /// [`Mamba3Untied::InProjTail`] moves them into `in_proj_tail`.
    pub fn d_in_proj_tail(&self) -> usize {
        let u = self.micro_steps;
        let scalars = 2
            + usize::from(self.trapezoid.has_beta_tap())
            + usize::from(self.trapezoid.has_interior_tap(u))
            + usize::from(self.gain.projects_noise())
            + 2 * usize::from(self.tropical.is_on());
        scalars * u * self.nheads() + u * self.num_rotation_channels()
    }

    /// The block's 2-D weights Muon may own, and how their fused columns split.
    ///
    /// The segments of `in_proj` mirror the `split_into` in the SSD pathways
    /// (`[z | x·u | B·u | C | Δ·u | A·u | λ·u | μ·u | rotation·u | r·u | a·u |
    /// b·u]`). The per-head *scalar* channels stay on the fallback optimizer.
    /// See [`burn_stack::optim`].
    ///
    /// Each micro-step gets its **own** segment, not a share of one `u`-wide
    /// slab: they are `u` independent `d_model → width` maps, and Muon must
    /// orthogonalise each one alone. They share a name, so
    /// [`MuonPlan::without_segment`](burn_stack::optim::MuonPlan::without_segment)
    /// still removes all the micro-steps of a stream at once.
    ///
    /// Under [`Mamba3Untied::InProjTail`], the segments from `Δ` on belong to
    /// `in_proj_tail`, one copy per application ([`ProjSpec::tiled`]).
    ///
    /// [`ProjSpec::tiled`]: burn_stack::optim::ProjSpec::tiled
    #[cfg(feature = "optim")]
    pub fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
        use burn_stack::optim::{ProjSegment as Seg, ProjSpec};
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let bc = self.ngroups * self.state_rank * self.mimo_rank;
        let rot = self.num_rotation_channels();
        // One copy of `seg` per micro-step.
        let per_micro = |seg: Seg| std::iter::repeat_n(seg, self.micro_steps);
        let head: Vec<Seg> = std::iter::once(Seg::muon("z", d_inner))
            .chain(per_micro(Seg::muon("x", d_inner)))
            .chain(per_micro(Seg::muon("b", bc)))
            .chain(std::iter::once(Seg::muon("c", bc)))
            .collect();
        let tail: Vec<Seg> = per_micro(Seg::adamw("dt", nheads))
            .chain(per_micro(Seg::adamw("a", nheads)))
            // `Trapezoid::None` projects no `λ` (as `Real1D` projects no
            // rotation). Only a two-tap pattern projects `μ`, the mix of the
            // second mass.
            .chain(
                self.trapezoid
                    .has_beta_tap()
                    .then(|| per_micro(Seg::adamw("lambda", nheads)))
                    .into_iter()
                    .flatten(),
            )
            .chain(
                self.trapezoid
                    .has_interior_tap(self.micro_steps)
                    .then(|| per_micro(Seg::adamw("mu", nheads)))
                    .into_iter()
                    .flatten(),
            )
            // `Real1D` has no rotation columns, and a segment cannot have
            // zero width (see `num_rotation_channels`).
            .chain(
                (rot > 0)
                    .then(|| per_micro(Seg::muon("rotation", rot)))
                    .into_iter()
                    .flatten(),
            )
            // The channels of the positive systems are per-head scalars, like `Δ`.
            .chain(
                self.gain
                    .projects_noise()
                    .then(|| per_micro(Seg::adamw("kalman_noise", nheads)))
                    .into_iter()
                    .flatten(),
            )
            .chain(
                self.tropical
                    .is_on()
                    .then(|| {
                        per_micro(Seg::adamw("tropical_a", nheads))
                            .chain(per_micro(Seg::adamw("tropical_b", nheads)))
                    })
                    .into_iter()
                    .flatten(),
            )
            .collect();
        let in_proj = match self.untied.contains(&Mamba3Untied::InProjTail) {
            true => vec![
                ProjSpec::block("in_proj.weight", head),
                ProjSpec::block("in_proj_tail.weight", tail).tiled(),
            ],
            false => vec![ProjSpec::block("in_proj.weight", [head, tail].concat())],
        };
        in_proj
            .into_iter()
            .chain([ProjSpec::block_whole("out_proj.weight", self.d_model)])
            .collect()
    }

    /// Allocate and initialise all Mamba-3 block parameters on `device`, for a
    /// single application.
    pub fn init(&self, device: &Device) -> Mamba3 {
        self.init_applications(1, device)
    }

    /// Allocate and initialise all Mamba-3 block parameters on `device`, for a
    /// real layer applied `n_applications` times: every [`Self::untied`]
    /// parameter holds that many copies of one initialisation.
    pub fn init_applications(&self, n_applications: usize, device: &Device) -> Mamba3 {
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let num_rope_angles = self.num_rope_angles();

        assert!(state_rank > 0, "state_rank must be positive");
        // RoPE pairing needs an even rank. `Real1D` makes no pairs, so it is
        // the one kind that accepts an odd (for example, scalar) state.
        assert!(
            self.rotation == RotationKind::Real1D || state_rank.is_multiple_of(2),
            "state_rank must be even for RoPE pairing"
        );
        assert!(self.per_head_dim > 0, "per_head_dim must be positive");
        assert_eq!(
            nheads * self.per_head_dim,
            d_inner,
            "d_inner must be divisible by per_head_dim"
        );
        assert_ne!(ngroups, 0, "ngroups must be at least 1");
        assert_eq!(nheads % ngroups, 0, "nheads must be divisible by ngroups");
        assert!(self.a_floor > 0.0, "a_floor must be positive");
        assert!(mimo_rank >= 1, "mimo_rank must be at least 1");
        assert!(self.micro_steps >= 1, "micro_steps must be at least 1");
        assert!(
            [0.5, 1.0].contains(&self.rope_fraction),
            "rope_fraction must be 0.5 or 1.0 (for no rotation use RotationKind::Real1D)"
        );
        if self.rotation != RotationKind::Real1D {
            // Every rotating kind must turn something. The no-rotation
            // ablation is its own kind (`Real1D`), not a fraction of zero.
            assert!(
                num_rope_angles > 0,
                "{:?} rotates nothing at state_rank = {} and rope_fraction = {} — use RotationKind::Real1D",
                self.rotation,
                self.state_rank,
                self.rope_fraction
            );
        }
        if matches!(
            self.rotation,
            RotationKind::Quaternion4D | RotationKind::Rotor4D
        ) {
            assert!(
                self.state_rank.is_multiple_of(4),
                "{:?} requires state_rank to be a multiple of 4",
                self.rotation
            );
            // The rotated width is a whole number of quaternion blocks, so a
            // partial rotation must cover a multiple of 4. Without this check,
            // the block would silently round `num_quat_blocks` up to its floor
            // of 1 and rotate *everything* when asked for half.
            assert!(
                self.rope_dim().is_multiple_of(4),
                "{:?} rotates whole 4-blocks: rope_fraction·state_rank = {} is not a multiple of 4",
                self.rotation,
                self.rope_dim()
            );
        }
        assert!(self.rotation_range > 0.0, "rotation_range must be positive");
        let unties = |part| self.untied.contains(&part);
        assert!(
            mimo_rank > 1
                || ![Mamba3Untied::MimoX, Mamba3Untied::MimoZ, Mamba3Untied::MimoO]
                    .into_iter()
                    .any(unties),
            "Mamba3Untied::Mimo* unties a MIMO projection, and mimo_rank = 1 has none"
        );
        assert!(
            self.has_outproj_norm || !unties(Mamba3Untied::OutNorm),
            "Mamba3Untied::OutNorm unties the output norm, and has_outproj_norm is off"
        );
        assert!(
            self.has_learnable_init_state || !unties(Mamba3Untied::InitState),
            "Mamba3Untied::InitState unties the initial state, and has_learnable_init_state is off"
        );
        assert!(
            self.gain.is_kalman() || !unties(Mamba3Untied::KalmanKappa),
            "Mamba3Untied::KalmanKappa unties the Kalman gate's κ, and the gain is Gain::Projected"
        );
        assert!(
            self.gain.is_kalman() || !unties(Mamba3Untied::KalmanRead),
            "Mamba3Untied::KalmanRead unties the Kalman read exponent, and the gain is Gain::Projected"
        );
        assert!(
            self.tropical.is_on() || !unties(Mamba3Untied::TropicalReadout),
            "Mamba3Untied::TropicalReadout unties the tropical readout, and tropical is Tropical::None"
        );
        assert!(
            !self.gain.is_kalman() || self.kalman_kappa_init > 0.0,
            "kalman_kappa_init must be positive (κ = 0 is Gain::Projected)"
        );
        // How many copies `part` holds: one per application if untied.
        let copies = |part| if unties(part) { n_applications } else { 1 };

        let uniform_init = |fan_in: usize| {
            let bound = 1.0 / (fan_in as f64).sqrt();
            Initializer::Uniform {
                min: -bound,
                max: bound,
            }
        };

        let d_in_proj_tail = unties(Mamba3Untied::InProjTail).then(|| self.d_in_proj_tail());
        let in_proj = LinearConfig::new(self.d_model, self.d_in_proj() - d_in_proj_tail.unwrap_or(0))
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform_init(self.d_model))
            .init(device);
        let in_proj_tail = d_in_proj_tail.map(|width| {
            let Linear { weight, bias } = LinearConfig::new(self.d_model, width)
                .with_bias(self.has_proj_bias)
                .with_initializer(uniform_init(self.d_model))
                .init(device);
            Linear {
                weight: untied::tile(weight, 1, n_applications),
                bias: bias.map(|b| untied::tile(b, 0, n_applications)),
            }
        });

        // dt_bias: inverse-softplus initialisation
        let expm1 = |t: Tensor<1>| t.exp() - 1.;
        let dt_h = Tensor::random(
            [nheads],
            burn::tensor::Distribution::Uniform(self.dt_min.ln(), self.dt_max.ln()),
            device,
        )
        .exp();
        let dt_h = dt_h.clamp(self.dt_init_floor, f64::INFINITY);
        let inv_dt_h = dt_h.clone() + (-expm1(-dt_h)).log();
        let dt_bias_h = untied::tile(
            Param::from_tensor(inv_dt_h),
            0,
            copies(Mamba3Untied::DtBias),
        );

        let d_h = Initializer::Ones.init::<1, _>([nheads], device);
        let d_h = untied::tile(d_h, 0, copies(Mamba3Untied::D));

        let mut b_norm = RmsNormConfig::new(state_rank).init(device);
        b_norm.gamma = untied::tile(b_norm.gamma, 0, copies(Mamba3Untied::BNorm));
        let mut c_norm = RmsNormConfig::new(state_rank).init(device);
        c_norm.gamma = untied::tile(c_norm.gamma, 0, copies(Mamba3Untied::CNorm));

        // B/C biases: [nheads, mimo_rank, state_rank], init to ones
        let b_bias_hmr = Initializer::Ones.init::<3, _>([nheads, mimo_rank, state_rank], device);
        let b_bias_hmr = untied::tile(b_bias_hmr, 0, copies(Mamba3Untied::BBias));
        let c_bias_hmr = Initializer::Ones.init::<3, _>([nheads, mimo_rank, state_rank], device);
        let c_bias_hmr = untied::tile(c_bias_hmr, 0, copies(Mamba3Untied::CBias));

        // MIMO projections (only for mimo_rank > 1)
        let (mimo_x_hmp, mimo_z_hmp, mimo_o_hmp) = if mimo_rank > 1 {
            let per_head_dim = self.per_head_dim;
            // Init: mimo_x_hmp and mimo_o_hmp to 1/mimo_rank, mimo_z_hmp to 1
            let mx = Param::from_tensor(Tensor::full(
                [nheads, mimo_rank, per_head_dim],
                1.0 / mimo_rank as f64,
                device,
            ));
            let mz = Param::from_tensor(Tensor::ones([nheads, mimo_rank, per_head_dim], device));
            let mo = Param::from_tensor(Tensor::full(
                [nheads, mimo_rank, per_head_dim],
                1.0 / mimo_rank as f64,
                device,
            ));
            (
                Some(untied::tile(mx, 0, copies(Mamba3Untied::MimoX))),
                Some(untied::tile(mz, 0, copies(Mamba3Untied::MimoZ))),
                Some(untied::tile(mo, 0, copies(Mamba3Untied::MimoO))),
            )
        } else {
            (None, None, None)
        };

        // Gated RMSNorm applied per-head (group size = per_head_dim).
        let out_norm = self.has_outproj_norm.then(|| {
            let mut norm = RmsNormGatedConfig::new(self.per_head_dim)
                .with_norm_before_gate(true)
                .init(device);
            norm.gamma = untied::tile(norm.gamma, 0, copies(Mamba3Untied::OutNorm));
            norm
        });

        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform_init(d_inner))
            .init(device);

        let init_state_hpr = self.has_learnable_init_state.then(|| {
            let init = Initializer::Zeros.init::<3, _>([nheads, self.per_head_dim, state_rank], device);
            untied::tile(init, 0, copies(Mamba3Untied::InitState))
        });

        let kalman_log_kappa_h = self.gain.is_kalman().then(|| {
            let init = Param::from_tensor(Tensor::full([nheads], self.kalman_kappa_init.ln(), device));
            untied::tile(init, 0, copies(Mamba3Untied::KalmanKappa))
        });
        let kalman_read_h = self.gain.is_kalman().then(|| {
            let init = Initializer::Zeros.init::<1, _>([nheads], device);
            untied::tile(init, 0, copies(Mamba3Untied::KalmanRead))
        });
        let tropical_readout_hp = self.tropical.is_on().then(|| {
            let init = Initializer::Zeros.init::<2, _>([nheads, self.per_head_dim], device);
            untied::tile(init, 0, copies(Mamba3Untied::TropicalReadout))
        });

        Mamba3 {
            in_proj,
            in_proj_tail,
            dt_bias_h,
            dt_limit: self.dt_limit,
            a_floor: self.a_floor,
            d_h,
            b_norm,
            c_norm,
            b_bias_hmr,
            c_bias_hmr,
            mimo_x_hmp,
            mimo_z_hmp,
            mimo_o_hmp,
            out_norm,
            out_proj,
            init_state_hpr,
            kalman_log_kappa_h,
            kalman_read_h,
            tropical_readout_hp,
            state_rank,
            ngroups,
            rope_dim: self.rope_dim(),
            num_rope_angles,
            mimo_rank,
            micro_steps: self.micro_steps,
            rotation: self.rotation,
            trapezoid: self.trapezoid,
            gain: self.gain,
            tropical: self.tropical,
            rotation_range: self.rotation_range,
            num_rotation_channels: self.num_rotation_channels(),
            num_quat_blocks: self.num_quat_blocks(),
            siso_specialization: self.siso_specialization,
            siso_specialization_decode: self.siso_specialization_decode,
            untied: self.untied.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Mamba3::forward  (chunkwise SSD — training / prefill)
// ---------------------------------------------------------------------------

impl Mamba3 {
    /// Process a full input sequence with the chunkwise trapezoidal SSD.
    ///
    /// The cache variant selects the pathway: [`Mamba3Cache::DoubleSsd`] runs
    /// `forward_double_ssd`, and [`Mamba3Cache::SingleSsd`] (also the choice
    /// for a missing cache) runs `forward_single_ssd`. Both give the same
    /// output and final cache.
    ///
    /// For MIMO (mimo_rank>1), B/C have mimo_rank parallel rank channels. All
    /// ranks share the hidden state, and each rank writes to it independently.
    ///
    /// `pad_bs` (`true` at padding, `None` ⇒ no padding) marks a right-padded
    /// batch of **tokens**. A padded token is absent:
    ///
    /// - all `u` of its micro-steps are the identity step (no decay, no mass:
    ///   `TrapezoidCoeffs::padded`),
    /// - each slot reads every "last samples" cache field at its own end (see
    ///   [`burn_stack::modules::Block::block_forward`]).
    ///
    /// # Shapes
    /// - `input_bsm` : `[batch, sequence, d_model]`
    /// - `pad_bs`    : `[batch, sequence]`
    /// - output      : `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3Cache>,
        ssd_path: Mamba3SsdPath,
        pad_bs: Option<Tensor<2, Bool>>,
    ) -> (Tensor<3>, Mamba3Cache) {
        let [batch, sequence, _d_model] = input_bsm.dims();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let device = input_bsm.device();

        assert!(sequence > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        // A missing cache selects the single-ssd pathway (every rotation kind
        // runs on it).
        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &device));

        // ── SSD Pathway Selection ─────────────────────────────────────────────
        match cache {
            Mamba3Cache::DoubleSsd(cache) => {
                let (out_bsm, cache) =
                    self.forward_double_ssd(input_bsm, Some(cache), &ssd_path, pad_bs);
                (out_bsm, cache.into())
            }
            Mamba3Cache::SingleSsd(cache) => {
                let (out_bsm, cache) =
                    self.forward_single_ssd(input_bsm, Some(cache), &ssd_path, pad_bs);
                (out_bsm, cache.into())
            }
        }
    }

    /// Build the default per-call cache (single-ssd pathway, for every rotation
    /// kind). The rotation accumulator is the matching [`RotationState`] variant.
    fn zero_cache(&self, batch: usize, device: &Device) -> Mamba3Cache {
        let nheads = self.nheads();
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], device);
        let (k_state_bumhr, v_state_buhp) = self.zero_tap_slots(batch, device);
        let rotation = self.zero_rotation_state(batch, device);
        let (log_precision_bh, tropical_bh) = self.zero_positive_state(batch, device);
        crate::mamba3::single_ssd::cache::Mamba3SingleSsdCache {
            ssm_bhpr,
            k_state_bumhr,
            v_state_buhp,
            rotation,
            log_precision_bh,
            tropical_bh,
        }
        .into()
    }

    /// How far back the `β` tap of this block reaches, in folded positions:
    /// [`Trapezoid::tap_lag`] at the `micro_steps` of this block. `0` when the
    /// pattern has no tap.
    pub fn tap_lag(&self) -> usize {
        self.trapezoid.tap_lag(self.micro_steps)
    }

    /// The tap FIFO of the trapezoid, filled with zeros: [`Self::tap_lag`]
    /// slots of `(B, x)`, oldest first. `None` (no allocation) under a pattern
    /// with no `β` tap.
    ///
    /// # Shapes
    /// - `.0` : `[batch, tap_slots, mimo_rank, nheads, state_rank]`
    /// - `.1` : `[batch, tap_slots, nheads, per_head_dim]`
    pub fn zero_tap_slots(
        &self,
        batch: usize,
        device: &Device,
    ) -> (Option<Tensor<5>>, Option<Tensor<4>>) {
        let slots = self.tap_lag();
        if slots == 0 {
            return (None, None);
        }
        let nheads = self.nheads();
        (
            Some(Tensor::zeros(
                [batch, slots, self.mimo_rank, nheads, self.state_rank],
                device,
            )),
            Some(Tensor::zeros(
                [batch, slots, nheads, self.per_head_dim()],
                device,
            )),
        )
    }

    /// The tap FIFO for the next call: the `(B, x)` of the last `lag` folded
    /// positions, oldest first. `(None, None)` when the pattern has no `β` tap.
    ///
    /// `x` is pre-scaled by the decay accumulated since its own position
    /// ([`crate::mamba3::helpers::tail_decay`]). This lets the gap transport of
    /// a lag-`u` tap cross the call boundary. At `lag = 1` that product is
    /// empty, and the slot is the plain last `x`. Both pathways use this
    /// function, so their caches stay field-identical.
    ///
    /// `end` is `(real folded positions per slot, the incoming FIFO)` for a
    /// right-padded call. The slots then hold the last `lag` real positions of
    /// each slot, read from the incoming FIFO followed by this call. For a slot
    /// with no real position, that is the incoming FIFO itself.
    ///
    /// # Shapes
    /// - `b_bsmhr` : `[batch, sequence, mimo_rank, nheads, state_rank]`
    /// - `x_bshp`  : `[batch, sequence, nheads, per_head_dim]`
    /// - `da_bsh`  : `[batch, sequence, nheads]`
    /// - `end`     : `[batch]`, and the incoming `(B, x)` slots
    pub(crate) fn save_tap_slots(
        &self,
        b_bsmhr: &Tensor<5>,
        x_bshp: &Tensor<4>,
        da_bsh: &Tensor<3>,
        lag: usize,
        end: Option<(Tensor<1, Int>, Option<Tensor<5>>, Option<Tensor<4>>)>,
    ) -> (Option<Tensor<5>>, Option<Tensor<4>>) {
        if lag == 0 {
            return (None, None);
        }
        let end = end.map(|(end_b, prev_b, prev_x)| {
            let slots = "a β tap keeps its (B, x) cache slots";
            (end_b, prev_b.expect(slots), prev_x.expect(slots))
        });
        let sequence = x_bshp.dims()[1];
        let (b_last_bumhr, x_last_buhp, da_bsh) = match end {
            None => (
                b_bsmhr.clone().narrow(1, sequence - lag, lag),
                x_bshp.clone().narrow(1, sequence - lag, lag),
                da_bsh.clone(),
            ),
            Some((end_b, prev_bumhr, prev_buhp)) => {
                use crate::padding::window;
                // The incoming slots already carry their decay to the old
                // boundary, and a slot's padding adds none.
                let [batch, _, nheads] = da_bsh.dims();
                let prev_buh = Tensor::zeros([batch, lag, nheads], &da_bsh.device());
                let b_all = Tensor::cat(vec![prev_bumhr, b_bsmhr.clone()], 1);
                let x_all = Tensor::cat(vec![prev_buhp, x_bshp.clone()], 1);
                let da_all = Tensor::cat(vec![prev_buh, da_bsh.clone()], 1);
                (
                    window(b_all, 1, end_b.clone(), lag),
                    window(x_all, 1, end_b.clone(), lag),
                    window(da_all, 1, end_b, lag),
                )
            }
        };
        let x_last_buhp = match crate::mamba3::helpers::tail_decay(da_bsh, lag) {
            Some(tail_buh) => x_last_buhp * tail_buh.unsqueeze_dim::<4>(3),
            None => x_last_buhp,
        };
        (Some(b_last_bumhr), Some(x_last_buhp))
    }

    /// The fresh-sequence rotation accumulator for the [`RotationKind`] of this
    /// block: the identity rotation, in the matching [`RotationState`] variant.
    pub fn zero_rotation_state(&self, batch: usize, device: &Device) -> RotationState {
        RotationState::identity(
            self.rotation,
            batch,
            self.nheads(),
            self.num_rope_angles,
            self.num_quat_blocks,
            device,
        )
    }
}

// ---------------------------------------------------------------------------
// Mamba3::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    impl Mamba3 {
        /// Process a **single token** with the pure recurrent form.
        ///
        /// For SISO (mimo_rank=1), at `u = 1` and the default lag-1 tap:
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ
        ///   yₜ = Cₜᵀ hₜ + D xₜ
        /// ```
        ///
        /// For MIMO (mimo_rank>1):
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + Σₘ βₜ Bₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_x_hmp[m]) + Σₘ γₜ Bₜ[m] ⊗ (xₜ ⊙ mimo_x_hmp[m])
        ///   yₜ[m] = Cₜ[m]ᵀ hₜ + D xₜ ⊙ mimo_x_hmp[m]
        ///   outₜ = Σₘ mimo_o_hmp[m] ⊙ silu(zₜ ⊙ mimo_z_hmp[m]) ⊙ yₜ[m]
        /// ```
        ///
        /// At `u > 1`, one call solves the `u` micro-steps of the token in
        /// closed form (see [`crate::mamba3::product`]). Both cache variants
        /// decode through the double-ssd recurrence. A missing cache starts a
        /// single-ssd one.
        ///
        /// # Shapes
        /// - `input_bd` : `[batch, d_model]`
        /// - output     : `[batch, d_model]`
        #[allow(non_snake_case)]
        pub fn step(
            &self,
            input_bd: Tensor<2>,
            cache: Option<Mamba3Cache>,
        ) -> (Tensor<2>, Mamba3Cache) {
            let [batch, _d_model] = input_bd.dims();
            let nheads = self.nheads();
            let ngroups = self.ngroups;
            let device = input_bd.device();

            assert_eq!(nheads % ngroups, 0);
            san(&input_bd);

            // ── Initialise cache if not provided ──────────────────────────────────
            // A missing cache selects the single-ssd pathway.
            let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &device));

            // ── SSD Pathway Selection ─────────────────────────────────────────────
            match cache {
                Mamba3Cache::DoubleSsd(cache) => {
                    let (out_bsm, cache) = self.step_double_ssd(input_bd, Some(cache));
                    (out_bsm, cache.into())
                }
                Mamba3Cache::SingleSsd(cache) => {
                    let (out_bsm, cache) = self.step_single_ssd(input_bd, Some(cache));
                    (out_bsm, cache.into())
                }
            }
        }
    }
}
