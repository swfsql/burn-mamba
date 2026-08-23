//! # Mamba-3 SSM Block — Exponential-Trapezoidal SSD with Data-Dependent RoPE
//!
//! This module implements the core **Mamba-3 layer** from the paper
//! *"Mamba-3: Improved Sequence Modeling using State Space Principles"*.
//!
//! Mamba-3 adds three independent extensions to the Mamba-2 SSD recurrence:
//! (1) trapezoidal discretisation, (2) a **complex-valued state transition**,
//! implemented as data-dependent rotary embeddings on B and C (the "RoPE
//! trick"), and (3) MIMO (multiple-input multiple-output) projection.
//! Each is shown in isolation below, then combined.
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
//! with `λₜ = σ(λ̂ₜ) ∈ (0, 1)` controlling the left/right split of the trapezoid.
//! Setting `λ ≡ 1` collapses this to the Mamba-2 (Euler / right-endpoint) form.
//!
//! ## 2. Complex transition, a.k.a. "data-dependent RoPE" (no trapezoid, no MIMO
//! — paper section *Complex-Valued SSMs*)
//!
//! Despite the name this is **not a positional encoding**: it is the *imaginary
//! part of the state transition*. Mamba-3's `A` is complex (`A + iϑ`), and under
//! discretisation a complex SSM of state `N/2` is exactly a real SSM of state `N`
//! whose transition is the scalar decay `α` times a block-diagonal of `2×2`
//! rotations (paper Prop. *Complex-to-Real SSM Equivalence*):
//!
//! ```text
//!   ρₜ = R(Δₜ · π · tanh(ϑₜ)) ∈ SO(2)^{N/2}   — per-step rotation (data-dependent)
//!   hₜ = αₜ ρₜ hₜ₋₁ + Δₜ Bₜ xₜᵀ                — rotational state update
//! ```
//!
//! `αₜ` is a scalar (so it commutes with `ρₜ`) and each `ρₜ` is orthogonal, so
//! the *cumulative* rotation telescopes out of the recurrence and can be absorbed
//! into B/C instead — the **"RoPE trick"** (paper Prop. *Complex SSM,
//! Data-Dependent RoPE Equivalence*). That is the form implemented here: the
//! rotation never touches the state (the cached `h` is the rotated-frame one) and
//! the SSD core stays the plain scalar-decay kernel.
//!
//! ```text
//!   θₜ = θₜ₋₁ + Δₜ · π · tanh(ϑₜ)        — cumulative angles (per-pair)
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
//! Orthogonality makes the readout-vs-input similarity depend only on the
//! rotation *accumulated between* the two steps:
//!
//! ```text
//!   C̃ᵢᵀ B̃ⱼ = (Rᵢ Cᵢ)ᵀ (Rⱼ Bⱼ) = Cᵢᵀ R(θⱼ − θᵢ) Bⱼ
//! ```
//!
//! `θⱼ − θᵢ` is not a position: it is `Σ Δ·π·tanh(ϑ)` over the intervening
//! steps — how far the transition itself rotated, driven by the inputs. It would
//! reduce to vanilla RoPE's relative position only in the degenerate case of an
//! input-independent, constant per-step angle (`θⱼ − θᵢ = (j−i)·θ`). Data
//! dependence is the whole point: it gives the layer rotational state dynamics,
//! hence state-tracking (parity, mod-k) that a real, non-negative-eigenvalue SSM
//! provably cannot express. See [`crate::mamba3::rotation`], whose `Quaternion4D`
//! carries the same argument to a non-abelian rotation group.
//!
//! ## 3. MIMO Extension (no trapezoid coefficients, no RoPE — `mimo_rank = M > 1`)
//!
//! With MIMO, B/C carry M parallel rank channels and the state update is a sum
//! of M outer-product contributions; the readout produces M outputs which are
//! gated and combined back:
//!
//! ```text
//!   hₜ = Āₜ hₜ₋₁ + Σₘ B̄ₜ[m] ⊗ (xₜ ⊙ mimo_x[m])                  (state update)
//!   yₜ[m] = Cₜ[m]ᵀ hₜ + D · (xₜ ⊙ mimo_x[m])                    (per-rank output)
//!   outₜ  = Σₘ mimo_o[m] ⊙ silu(zₜ ⊙ mimo_z[m]) ⊙ yₜ[m]         (rank merge)
//! ```
//!
//! The hidden state hₜ is shared across ranks; each rank contributes to it
//! independently but reads the full shared state when producing its output.
//!
//! ## 4. Combined formulation (everything together)
//!
//! Putting trapezoid + RoPE + MIMO into a single expression — `B̃ₜ[m] = Rₜ Bₜ[m]`
//! and `C̃ₜ[m] = Rₜ Cₜ[m]` denote the RoPE-rotated MIMO projections:
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
//! Implementation note: the trapezoidal recurrence (in the double-ssd pathway)
//! is computed by splitting it into a γ-SSD (current-token contributions) and
//! a β-SSD (previous-token contributions); see [`crate::mamba3::double_ssd::ssd::ssd_path`].
//! RoPE is applied to B and C before the SSD calls
//! (see [`crate::modules::misc::rope::apply_rope`]),
//! and MIMO expansion happens by augmenting the V tensor with the per-rank
//! `mimo_x` projection.
//!
//! See also: [`crate::mamba3::double_ssd::double_ssd`] and [`crate::mamba3::single_ssd::single_ssd`].
//!
//! ## Notation / Dimension Keys
//!
//! Throughout all Mamba-3 files, tensor names carry a suffix representing their shape.
//! The letters used differ from the reference paper and the python implementation.
//! The "Paper" column gives the symbol from the Mamba-3 paper; the "Python" column
//! gives the field/variable name in the reference implementation
//! (`refs/state-spaces/mamba/mamba_ssm/modules/mamba3.py`).
//!
//! | Letter | Dimension | Paper | Python | Typical value |
//! |--------|-----------|-------|--------|---------------|
//! | `b`    | `batch` | — | `batch` | varies |
//! | `s`    | `sequence` length | `T` | `seqlen` | varies |
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
//! Uppercase letters represent a relation (e.g. offset, multiple, concat, stacking)
//! of the lowercase letters. e.g. `X` may represent `x+1`, `x-1`, `x*2`, etc.
//! `XY` may also represent `x+y`, `x*y`, etc.

use crate::mamba3::prelude::*;
use crate::mamba3::rotation::RotationKind;
use crate::modules::sanity as san;
use crate::modules::{RmsNorm, RmsNormConfig, RmsNormGated, RmsNormGatedConfig};
use burn::prelude::*;
use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
};

// ---------------------------------------------------------------------------
// Mamba3  (the SSM block)
// ---------------------------------------------------------------------------

/// The Mamba-3 SSM block.
///
/// Implements the full Mamba-3 layer with exponential-trapezoidal discretization
/// and data-dependent RoPE.  Supports SISO (mimo_rank=1) and MIMO (mimo_rank>1).
/// Supports two execution modes:
///
/// - [`Self::forward`] — chunkwise double-SSD algorithm for training / prefill
/// - [`Self::step`]    — recurrent form for token-by-token decoding
#[derive(Module, Debug)]
pub struct Mamba3 {
    /// Input projection.
    ///
    /// For SISO (R=1), maps:
    /// `d_model → 2·d_inner + 2·ngroups·state_rank + 3·nheads + num_rope_angles`
    /// For MIMO (R>1), maps:
    /// `d_model → 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads + num_rope_angles`
    ///
    /// Output splits: `[z | x | B_raw | C_raw | dd_dt | dd_A | lambda_raw | theta_raw]`
    pub in_proj: Linear,

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
    /// When `Some`, the SiLU gate at the block tail is replaced by
    /// `RmsNormGated(y, z)` which normalises `y` over `per_head_dim` and
    /// gates with `SiLU(z)`. Created when `has_outproj_norm = true`.
    pub out_norm: Option<RmsNormGated>,

    /// Output projection: maps `d_inner → d_model`.
    pub out_proj: Linear,

    /// Optional learnable initial hidden state `h₀`.
    /// Shape: `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Param<Tensor<3>>>,

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
    /// `≤ state_rank`. Only the first `rope_dim` entries of B/C are rotated.
    pub rope_dim: usize,

    /// MIMO rank. 1 = SISO (standard Mamba-3).
    ///
    /// Paper: `M`. Python: `mimo_rank`.
    pub mimo_rank: usize,

    /// Which transition rotation the block applies to `B`/`C` ([`RotationKind`]).
    /// A non-parameter constant — `#[module(skip)]` keeps it out of the record and
    /// carries it through `load_record`/`to_device`/… unchanged.
    #[module(skip)]
    pub rotation: RotationKind,

    /// How far one step may rotate, in half-turns per unit `Δ`
    /// (see [`Mamba3Config::rotation_range`]).
    pub rotation_range: f64,

    /// Number of in-projection channels devoted to the rotation parameters
    /// (`num_rope_angles` for `Complex2D`, `nheads·3·num_quat_blocks` for
    /// `Quaternion4D`); the size of the last `in_proj` split segment.
    pub num_rotation_channels: usize,

    /// Number of quaternion blocks (`rope_dim / 4`); only used for
    /// [`RotationKind::Quaternion4D`].
    pub num_quat_blocks: usize,

    /// Whether the `mimo_rank == 1` specialized *chunkwise* kernel is enabled
    /// (see [`Mamba3Config::siso_specialization`]).
    ///
    /// A performance knob, not a semantic one: both branches compute the same
    /// values and gradients. A non-parameter constant — `#[module(skip)]` keeps
    /// it out of the record.
    #[module(skip)]
    pub siso_specialization: bool,

    /// Whether the `mimo_rank == 1` specialized *per-token* kernels are enabled
    /// (see [`Mamba3Config::siso_specialization_decode`]). Same performance-only
    /// nature, opposite backend preference.
    #[module(skip)]
    pub siso_specialization_decode: bool,
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

    /// Number of quaternion blocks the rotation projection and its cumulative
    /// scan run over: `num_quat_blocks · quat_factors` — twice the state's
    /// 4-block count for [`RotationKind::Rotor4D`], whose left and right
    /// factors share one stacked block axis.
    pub fn num_rotation_blocks(&self) -> usize {
        self.num_quat_blocks * self.rotation.quat_factors()
    }

    /// Everything the rotation needs, in one place — the algebra, the rotated
    /// width, and the per-step bound. Every site that materialises a per-step
    /// rotation ([`forward`](Self::forward), [`step`](Self::step),
    /// [`step_infinite`](Self::step_infinite)) goes through this.
    pub fn rotation_spec(&self) -> crate::mamba3::rotation::RotationSpec {
        crate::mamba3::rotation::RotationSpec {
            kind: self.rotation,
            rope_dim: self.rope_dim,
            range: self.rotation_range,
        }
    }

    /// Whether the specialized `mimo_rank == 1` **per-token** kernels should run:
    /// the block is SISO **and** [`Mamba3Config::siso_specialization_decode`] is
    /// enabled.
    ///
    /// The chunkwise counterpart is [`Mamba3Config::siso_specialization`], which
    /// travels to its kernel as a field of the SSD input bundle rather than
    /// through a method.
    ///
    /// Purely a performance choice — see [`Mamba3Config::siso_specialization_decode`].
    pub fn use_siso_decode_kernels(&self) -> bool {
        self.mimo_rank == 1 && self.siso_specialization_decode
    }
}

// ---------------------------------------------------------------------------
// Mamba3Config  (hyperparameters and factory)
// ---------------------------------------------------------------------------

/// Hyperparameters for the Mamba-3 SSM block.
#[derive(Config, Debug)]
pub struct Mamba3Config {
    /// Model (hidden) dimension.
    ///
    /// Paper: `D`. Python: `d_model`.
    pub d_model: usize,

    /// State rank — the latent dimension of the SSM hidden state.
    /// **Must be even** (required for RoPE pairing).
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
    /// When `mimo_rank > 1`, the B/C projections have `mimo_rank` parallel rank channels.
    /// Three extra weight matrices (`mimo_x_hmp`, `mimo_z_hmp`, `mimo_o_hmp`) provide
    /// element-wise up/down projections in head-space across ranks.
    ///
    /// Paper: `M`. Python: `mimo_rank`.
    #[config(default = 1)]
    pub mimo_rank: usize,

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

    /// Whether to allocate a learnable initial SSM state `h₀`.
    #[config(default = false)]
    pub has_learnable_init_state: bool,

    /// Fraction of `state_rank` the transition rotation turns (must be `0.0`,
    /// `0.5`, or `1.0`).
    ///
    /// - `0.0`: rotation disabled — no B/C dimension is turned (both kinds
    ///   short-circuit to the identity). Intended for ablations only. The
    ///   rotation projection and its cumulative data flow are kept intact (with
    ///   a single dummy channel group, see [`Self::num_rope_angles`] /
    ///   [`Self::num_quat_blocks`]) so the rest of the block is structurally
    ///   unchanged.
    /// - `0.5`: partial rotation — only `state_rank / 2` dimensions are turned;
    ///   the rest pass through unchanged. This is the reference's value in
    ///   `mamba3.py`; set it explicitly to reproduce that model.
    /// - `1.0` (default): full rotation — every B/C dimension is turned.
    ///
    /// The default turns everything, for both rotation kinds: a partial
    /// rotation is a capacity trade (keeping "content" channels out of the
    /// turn), which is a deliberate choice a config should have to ask for
    /// rather than get by omission. For
    /// [`Quaternion4D`](RotationKind::Quaternion4D) and
    /// [`Rotor4D`](RotationKind::Rotor4D) it is also what keeps the smallest
    /// legal `state_rank` (4 — a single quaternion block) usable, since a
    /// partial quaternion rotation must land on whole 4-blocks.
    #[config(default = 1.0)]
    pub rope_fraction: f64,

    /// Whether to apply a gated RMSNorm before the output projection.
    ///
    /// When `true`, the SiLU gate at the end of the block is replaced by a
    /// per-head [`RmsNormGated`] (group size = `per_head_dim`) which both
    /// normalises `y` and gates it with `SiLU(z)`. Matches the reference's
    /// `is_outproj_norm` argument in `mamba3.py`.
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
    #[config(default = "crate::mamba3::rotation::RotationKind::Complex2D")]
    pub rotation: RotationKind,

    /// How far a single step may rotate the state, in **half-turns per unit
    /// `Δ`**: the per-step angle is bounded by `rotation_range · π · Δ`.
    ///
    /// The default of `2.0` is "one unit of `Δ` may traverse the whole rotation
    /// group once", which means the same thing for both kinds even though their
    /// periods differ:
    ///
    /// - [`Complex2D`](RotationKind::Complex2D): `2π` is a full turn of
    ///   `SO(2)`.
    /// - [`Quaternion4D`](RotationKind::Quaternion4D): `2π` reaches every
    ///   element of `SU(2)`, whose period is `4π` — `q` and `−q` are the two
    ///   lifts of one `SO(3)` rotation and turn the state differently.
    /// - [`Rotor4D`](RotationKind::Rotor4D): the bound applies to **each
    ///   factor**, so at the default each of `q` and `p` independently sweeps
    ///   all of `SU(2)` and the pair reaches every element of
    ///   `SO(4) ≅ (SU(2)×SU(2))/±1`. Bounding the composite instead (halving
    ///   the per-factor angle) would cost reach for nothing: the two plane
    ///   angles `a∓b` are periodic in `2π` anyway.
    ///
    /// What the bound buys is not reach but **gradients**. A rotation of exactly
    /// `range·π` sits at `tanh`'s asymptote, where the gradient is not merely
    /// small but **exactly zero** in f32 (`tanh(10) == 1.0`). Half-turns are
    /// what state-tracking wants, so at `range = 1` — where `π` *is* the
    /// bound — the most useful rotation is the one the optimiser can never
    /// reach. At `2.0` it sits at `tanh = 1/2`, in the interior.
    ///
    /// `1.0` is the reference implementation's abelian bound (`Δ·π·tanh(ϑ)`);
    /// set it explicitly to reproduce that model.
    #[config(default = 2.0)]
    pub rotation_range: f64,

    /// Whether to use the specialized `mimo_rank == 1` (SISO) **chunkwise**
    /// kernel: the same-step γ-correction
    /// ([`y_diag_correction`](crate::mamba3::single_ssd::ssd::diag::y_diag_correction)),
    /// forward *and* its analytic backward.
    ///
    /// At `mimo_rank == 1` the `m × m` Gram is a scalar, so the general form
    /// issues a `1×r×1` and a `1×1×p` GEMM per `(batch, nchunks, chunk_len,
    /// nheads)` — thousands of degenerate products. The SISO branch replaces all
    /// of them with one elementwise multiply and a `state_rank` reduction over
    /// the whole tensor, which is why deleting the tiny GEMMs helps everywhere:
    /// ≈ neutral to ~9 % faster across the backends in `bench.md`. Leave it on.
    ///
    /// **Values and gradients are identical either way** — this only selects the
    /// op mix. Ignored when `mimo_rank > 1`, where only the general branch
    /// applies. See also [`Self::siso_specialization_decode`], the per-token
    /// counterpart, whose trade-off is *not* the same.
    #[config(default = true)]
    pub siso_specialization: bool,

    /// Whether to use the specialized `mimo_rank == 1` (SISO) **per-token**
    /// kernels: the decode state update and readout
    /// (`helpers::mimo_outer_sum`, `Mamba3::step_readout`), the single-SSD
    /// boundary-β seed, and [`step_infinite`](Mamba3::step_infinite)'s Gram.
    ///
    /// **Set this to `false` on the CPU backends.** Unlike the chunkwise flag,
    /// this one replaces a *single, already efficient* batched matmul with a
    /// broadcast multiply, so the verdict flips with the backend: at
    /// `benches/layer.rs`'s default size the `step` group is ~12 % faster on CUDA
    /// and about **3× slower on flex**, whose broadcast-elementwise path trails
    /// its matmul by more than an order of magnitude at these shapes. The default
    /// (`true`) suits the GPU backends; a CPU deployment that decodes should turn
    /// it off — and can do so while keeping [`Self::siso_specialization`] on,
    /// which is the point of having two flags.
    ///
    /// **Values and gradients are identical either way.** Ignored when
    /// `mimo_rank > 1`.
    #[config(default = true)]
    pub siso_specialization_decode: bool,
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

    /// Effective RoPE dimension: the number of B/C channels actually rotated.
    /// `state_rank` for full RoPE (`rope_fraction = 1.0`), `state_rank / 2` for
    /// `rope_fraction = 0.5`, and `0` when RoPE is disabled
    /// (`rope_fraction = 0`), in which case the rotation is the identity.
    pub fn rope_dim(&self) -> usize {
        let mut d = (self.state_rank as f64 * self.rope_fraction) as usize;
        if !d.is_multiple_of(2) {
            d -= 1;
        }
        d
    }

    /// Number of RoPE rotation angles projected per head: `rope_dim / 2`, but
    /// **at least 1**.
    ///
    /// When RoPE is disabled (`rope_fraction = 0` ⇒ `rope_dim = 0`) the floor of
    /// 1 keeps a single angle channel alive: Burn has no zero-width tensors, and
    /// retaining one channel lets the angle projection / cumulative-angle data
    /// flow stay intact while the rotation itself short-circuits to the identity.
    pub fn num_rope_angles(&self) -> usize {
        (self.rope_dim() / 2).max(1)
    }

    /// Number of quaternion blocks for [`RotationKind::Quaternion4D`]: the
    /// rotated width (`state_rank · rope_fraction`) rounded **down to a multiple
    /// of 4**, divided by 4, and floored at 1 (Burn has no zero-width tensors —
    /// mirrors [`Self::num_rope_angles`]).
    pub fn num_quat_blocks(&self) -> usize {
        let mut d = (self.state_rank as f64 * self.rope_fraction) as usize;
        d -= d % 4;
        (d / 4).max(1)
    }

    /// Number of in-projection channels devoted to the rotation parameters,
    /// per the configured [`RotationKind`]:
    ///
    /// - [`Complex2D`](RotationKind::Complex2D): `num_rope_angles` angle
    ///   channels, shared across heads and scaled per-head by `Δ`, as in the
    ///   reference.
    /// - [`Quaternion4D`](RotationKind::Quaternion4D) /
    ///   [`Rotor4D`](RotationKind::Rotor4D):
    ///   `nheads · 3 · num_rotation_blocks` quaternion-generator channels — an
    ///   axis·angle generator **per head** and block (and, for `Rotor4D`, per
    ///   *factor*: one generator for the left quaternion and one for the
    ///   right), fed through
    ///   [`quat_from_scaled_axis`](crate::mamba3::rotation::quat_from_scaled_axis).
    ///   For a non-abelian transition the axis is where the expressiveness
    ///   lives: heads sharing one axis and differing only in `Δ` track one word
    ///   at different speeds instead of different words. See
    ///   [`generator_increment`](crate::mamba3::rotation::generator_increment).
    pub fn num_rotation_channels(&self) -> usize {
        match self.rotation {
            RotationKind::Complex2D => self.num_rope_angles(),
            // Three generator channels per (head, block) and **per factor**:
            // one factor for Quaternion4D, two for Rotor4D.
            RotationKind::Quaternion4D | RotationKind::Rotor4D => {
                self.nheads() * 3 * self.num_rotation_blocks()
            }
        }
    }

    /// Number of quaternion blocks the *rotation projection and scan* run over:
    /// [`Self::num_quat_blocks`] times [`RotationKind::quat_factors`] — i.e.
    /// `2 · blocks` for [`RotationKind::Rotor4D`], whose left and right factors
    /// share one stacked block axis.
    pub fn num_rotation_blocks(&self) -> usize {
        self.num_quat_blocks() * self.rotation.quat_factors()
    }

    /// Total input projection output size.
    ///
    /// `d_in_proj = 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads + num_rotation_channels`
    /// where the last term is [`Self::num_rotation_channels`] (`num_rope_angles`
    /// for the default `Complex2D`, so the size is unchanged from before).
    pub fn d_in_proj(&self) -> usize {
        2 * self.d_inner()
            + 2 * self.ngroups * self.state_rank * self.mimo_rank
            + 3 * self.nheads()
            + self.num_rotation_channels()
    }

    /// The block's 2-D weights Muon may own, and how their fused columns split.
    ///
    /// `in_proj`'s segments mirror the `split_into` in the SSD pathways
    /// (`[z | x | B | C | Δ | A | λ | rotation]`); the three per-head *scalar*
    /// channels stay on AdamW. See [`crate::optim`].
    #[cfg(feature = "optim")]
    pub fn muon_projections(&self) -> Vec<crate::optim::ProjSpec> {
        use crate::optim::{ProjSegment as Seg, ProjSpec};
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let bc = self.ngroups * self.state_rank * self.mimo_rank;
        vec![
            ProjSpec::block(
                "in_proj.weight",
                vec![
                    Seg::muon("z", d_inner),
                    Seg::muon("x", d_inner),
                    Seg::muon("b", bc),
                    Seg::muon("c", bc),
                    Seg::adamw("dt", nheads),
                    Seg::adamw("a", nheads),
                    Seg::adamw("lambda", nheads),
                    Seg::muon("rotation", self.num_rotation_channels()),
                ],
            ),
            ProjSpec::block_whole("out_proj.weight", self.d_model),
        ]
    }

    /// Allocate and initialise all Mamba-3 block parameters on `device`.
    pub fn init(&self, device: &Device) -> Mamba3 {
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let num_rope_angles = self.num_rope_angles();

        assert!(
            state_rank.is_multiple_of(2),
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
        assert!(
            [0.0, 0.5, 1.0].contains(&self.rope_fraction),
            "rope_fraction must be 0.0, 0.5 or 1.0"
        );
        assert!(num_rope_angles > 0, "num_rope_angles must be at least 1");
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
            // partial rotation has to land on a multiple of 4. Without this the
            // block would quietly round `num_quat_blocks` up to its floor of 1
            // and rotate *everything* when asked for half.
            assert!(
                self.rope_dim().is_multiple_of(4),
                "{:?} rotates whole 4-blocks: rope_fraction·state_rank = {} is not a multiple of 4",
                self.rotation,
                self.rope_dim()
            );
        }
        assert!(self.rotation_range > 0.0, "rotation_range must be positive");

        let uniform_init = |fan_in: usize| {
            let bound = 1.0 / (fan_in as f64).sqrt();
            Initializer::Uniform {
                min: -bound,
                max: bound,
            }
        };

        let in_proj = LinearConfig::new(self.d_model, self.d_in_proj())
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform_init(self.d_model))
            .init(device);

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
        let dt_bias_h = Param::from_tensor(inv_dt_h);

        let d_h = Initializer::Ones.init::<1, _>([nheads], device);

        let b_norm = RmsNormConfig::new(state_rank).init(device);
        let c_norm = RmsNormConfig::new(state_rank).init(device);

        // B/C biases: [nheads, mimo_rank, state_rank], init to ones
        let b_bias_hmr = Initializer::Ones.init::<3, _>([nheads, mimo_rank, state_rank], device);
        let c_bias_hmr = Initializer::Ones.init::<3, _>([nheads, mimo_rank, state_rank], device);

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
            (Some(mx), Some(mz), Some(mo))
        } else {
            (None, None, None)
        };

        // Gated RMSNorm applied per-head (group size = per_head_dim).
        let out_norm = self.has_outproj_norm.then(|| {
            RmsNormGatedConfig::new(self.per_head_dim)
                .with_norm_before_gate(true)
                .init(device)
        });

        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform_init(d_inner))
            .init(device);

        let init_state_hpr = self.has_learnable_init_state.then(|| {
            Initializer::Zeros.init::<3, _>([nheads, self.per_head_dim, state_rank], device)
        });

        Mamba3 {
            in_proj,
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
            state_rank,
            ngroups,
            rope_dim: self.rope_dim(),
            num_rope_angles,
            mimo_rank,
            rotation: self.rotation,
            rotation_range: self.rotation_range,
            num_rotation_channels: self.num_rotation_channels(),
            num_quat_blocks: self.num_quat_blocks(),
            siso_specialization: self.siso_specialization,
            siso_specialization_decode: self.siso_specialization_decode,
        }
    }
}

// ---------------------------------------------------------------------------
// Mamba3::forward  (chunkwise double-SSD — training / prefill)
// ---------------------------------------------------------------------------

impl Mamba3 {
    /// Process a full input sequence using the trapezoidal double-SSD algorithm.
    ///
    /// For SISO (mimo_rank=1), this is the standard double-SSD decomposition.
    /// For MIMO (mimo_rank>1), B/C have mimo_rank parallel rank channels.
    /// The hidden state is shared across mimo ranks; each mimo rank contributes independently.
    ///
    /// # Shapes
    /// - `input_bsm` : `[batch, sequence, d_model]`
    /// - output      : `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_bsm: Tensor<3>,
        cache: Option<Mamba3Cache>,
        ssd_path: Mamba3SsdPath,
    ) -> (Tensor<3>, Mamba3Cache) {
        let [batch, sequence, _d_model] = input_bsm.dims();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let _per_head_dim = self.per_head_dim();
        let _state_rank = self.state_rank;
        let _num_rope_angles = self.num_rope_angles;
        let _mimo_rank = self.mimo_rank;
        let device = input_bsm.device();

        assert!(sequence > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        // A missing cache implies the single-ssd pathway (both rotation kinds are
        // supported there; see [`forward_single_ssd`]).
        let cache = cache.unwrap_or_else(|| self.zero_cache(batch, &device));

        // ── SSD Pathway Selection ─────────────────────────────────────────────
        match cache {
            Mamba3Cache::DoubleSsd(cache) => {
                let (out_bsm, cache) = self.forward_double_ssd(input_bsm, Some(cache), &ssd_path);
                (out_bsm, cache.into())
            }
            Mamba3Cache::SingleSsd(cache) => {
                let (out_bsm, cache) = self.forward_single_ssd(input_bsm, Some(cache), &ssd_path);
                (out_bsm, cache.into())
            }
        }
    }

    /// Build the default per-call cache (single-ssd pathway, for either rotation
    /// kind). The rotation accumulator is the matching [`RotationState`] variant.
    fn zero_cache(&self, batch: usize, device: &Device) -> Mamba3Cache {
        let nheads = self.nheads();
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], device);
        let k_state_bmhr = Tensor::zeros([batch, mimo_rank, nheads, state_rank], device);
        let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], device);
        let rotation = self.zero_rotation_state(batch, device);
        crate::mamba3::single_ssd::cache::Mamba3SingleSsdCache {
            ssm_bhpr,
            k_state_bmhr,
            v_state_bhp,
            rotation,
        }
        .into()
    }

    /// The fresh-sequence rotation accumulator for this block's
    /// [`RotationKind`] — the identity rotation, in the matching
    /// [`RotationState`] variant.
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
        /// Process a **single token** using the pure recurrent form.
        ///
        /// For SISO (mimo_rank=1):
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + βₜ Bₜ₋₁ ⊗ xₜ₋₁ + γₜ Bₜ ⊗ xₜ
        ///   yₜ = Cₜᵀ hₜ + D xₜ
        /// ```
        ///
        /// For MIMO (mimo_rank>1):
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + Σₘ βₜ Bₜ₋₁[m] ⊗ (xₜ₋₁ ⊙ mimo_x_hmp[m]) + Σₘ γₜ Bₜ[m] ⊗ (xₜ ⊙ mimo_x_hmp[m])
        ///   yₜ[r] = Cₜ[r]ᵀ hₜ + D xₜ ⊙ mimo_x_hmp[r]
        ///   outₜ = Σₘ mimo_o_hmp[m] ⊙ silu(zₜ ⊙ mimo_z_hmp[m]) ⊙ yₜ[m]
        /// ```
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
            let _per_head_dim = self.per_head_dim();
            let _state_rank = self.state_rank;
            let _num_rope_angles = self.num_rope_angles;
            let _mimo_rank = self.mimo_rank;
            let device = input_bd.device();

            assert_eq!(nheads % ngroups, 0);
            san(&input_bd);

            // ── Initialise cache if not provided ──────────────────────────────────
            // Implies single-ssd pathway if missing (double-ssd for Quaternion4D).
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
