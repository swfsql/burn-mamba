//! # Mamba-3 SSM Block — Exponential-Trapezoidal SSD with Data-Dependent RoPE
//!
//! This module implements the core **Mamba-3 layer** from the paper
//! *"The Mamba-3 Framework: Structured State Spaces with Trapezoidal
//! Discretization and Data-Dependent Rotary Embeddings"*.
//!
//! Mamba-3 adds three independent extensions to the Mamba-2 SSD recurrence:
//! (1) trapezoidal discretisation, (2) data-dependent rotary position embeddings
//! (RoPE) on B and C, and (3) MIMO (multiple-input multiple-output) projection.
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
//! ## 2. Data-dependent RoPE (no trapezoid, no MIMO — Section "Data-Dependent RoPE")
//!
//! Each Bₜ, Cₜ ∈ ℝᴺ (N = `state_rank`, even) is treated as N/2 complex pairs and
//! rotated by a cumulative, data-dependent angle θₜ ∈ ℝ^{N/2}:
//!
//! ```text
//!   θₜ = θₜ₋₁ + Δₜ · π · tanh(ϑₜ)        — cumulative angles (per-pair)
//!   Rₜ = R(θₜ) ∈ SO(2)^{N/2}             — block-diagonal pairwise rotation
//!   B̃ₜ = Rₜ Bₜ,   C̃ₜ = Rₜ Cₜ              — rotated state-space projections
//! ```
//!
//! The standard recurrence then runs with `B̃ₜ`/`C̃ₜ` in place of `Bₜ`/`Cₜ`:
//!
//! ```text
//!   hₜ = Āₜ hₜ₋₁ + B̄̃ₜ xₜᵀ                (state update with RoPE)
//!   yₜ = C̃ₜᵀ hₜ + D xₜ                   (output with RoPE)
//! ```
//!
//! Because each rotation `Rₜ` is orthogonal, the readout-vs-input similarity
//! folds into the relative rotation between the two steps:
//!
//! ```text
//!   C̃ᵢᵀ B̃ⱼ = (Rᵢ Cᵢ)ᵀ (Rⱼ Bⱼ) = Cᵢᵀ R(θⱼ − θᵢ) Bⱼ
//! ```
//!
//! so the cumulative-angle difference encodes the (data-dependent) relative
//! position between query step i and key step j.
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
//! Implementation note: the trapezoidal recurrence is computed by splitting it
//! into a γ-SSD (current-token contributions) and a β-SSD (previous-token
//! contributions); see [`crate::mamba3::ssd::ssd_path`]. RoPE is applied to B and C before the
//! SSD calls (see [`apply_rope`]), and MIMO expansion happens by augmenting the
//! V tensor with the per-rank `mimo_x` projection.
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

use crate::mamba3::helpers;
use crate::mamba3::prelude::*;
use crate::utils::sanity::sanity as san;
use crate::utils::{
    rms_norm::{RmsNorm, RmsNormConfig},
    rms_norm_gated::{RmsNormGated, RmsNormGatedConfig},
    silu::Silu,
};
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
/// - [`Self::forward`] — chunkwise two-SSD algorithm for training / prefill
/// - [`Self::step`]    — recurrent form for token-by-token decoding
#[derive(Module, Debug)]
pub struct Mamba3<B: Backend> {
    /// Input projection.
    ///
    /// For SISO (R=1), maps:
    /// `d_model → 2·d_inner + 2·ngroups·state_rank + 3·nheads + num_rope_angles`
    /// For MIMO (R>1), maps:
    /// `d_model → 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads + num_rope_angles`
    ///
    /// Output splits: `[z | x | B_raw | C_raw | dd_dt | dd_A | lambda_raw | theta_raw]`
    pub in_proj: Linear<B>,

    /// Per-head bias for the discretisation step size Δ.
    /// Shape: `[nheads]`
    pub dt_bias_h: Param<Tensor<B, 1>>,

    /// Hard clamp applied to Δ after softplus.
    pub dt_limit: (f64, f64),

    /// Minimum absolute value of A: `A ∈ (−∞, −a_floor]`.
    pub a_floor: f64,

    /// Per-head skip (D) coefficient.
    /// Shape: `[nheads]`; initialised to ones.
    pub d_h: Param<Tensor<B, 1>>,

    /// RMSNorm applied to the B projection (QK-Norm, no gating).
    /// Normalises over the `state_rank` dimension.
    pub b_norm: RmsNorm<B>,

    /// RMSNorm applied to the C projection (QK-Norm, no gating).
    /// Normalises over the `state_rank` dimension.
    pub c_norm: RmsNorm<B>,

    /// Learnable per-head, per-rank bias for B, added after QK-norm.
    /// Shape: `[nheads, mimo_rank, state_rank]`; initialised to ones.
    pub b_bias_hmr: Param<Tensor<B, 3>>,

    /// Learnable per-head, per-rank bias for C, added after QK-norm.
    /// Shape: `[nheads, mimo_rank, state_rank]`; initialised to ones.
    pub c_bias_hmr: Param<Tensor<B, 3>>,

    /// MIMO up-projection for x (values).
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`.  When SISO, this is `None`.
    pub mimo_x_hmp: Option<Param<Tensor<B, 3>>>,

    /// MIMO up-projection for z (gate).
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`. When SISO, this is `None`.
    pub mimo_z_hmp: Option<Param<Tensor<B, 3>>>,

    /// MIMO down-projection for the output.
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`. When SISO, this is `None`.
    pub mimo_o_hmp: Option<Param<Tensor<B, 3>>>,

    /// Optional gated RMSNorm applied before the output projection.
    ///
    /// When `Some`, the SiLU gate at the block tail is replaced by
    /// `RmsNormGated(y, z)` which normalises `y` over `per_head_dim` and
    /// gates with `SiLU(z)`. Created when `has_outproj_norm = true`.
    pub out_norm: Option<RmsNormGated<B>>,

    /// Output projection: maps `d_inner → d_model`.
    pub out_proj: Linear<B>,

    /// Optional learnable initial hidden state `h₀`.
    /// Shape: `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Param<Tensor<B, 3>>>,

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
}

impl<B: Backend> Mamba3<B> {
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

    /// Fraction of `state_rank` to which RoPE is applied (must be `0.5` or `1.0`).
    ///
    /// - `0.5` (default): partial RoPE — only `state_rank / 2` dimensions are
    ///   rotated; the rest pass through unchanged.
    /// - `1.0`: full RoPE — every B/C dimension is rotated.
    ///
    /// Default matches the reference's `rope_fraction` argument in `mamba3.py`.
    #[config(default = 0.5)]
    pub rope_fraction: f64,

    /// Whether to apply a gated RMSNorm before the output projection.
    ///
    /// When `true`, the SiLU gate at the end of the block is replaced by a
    /// per-head [`RmsNormGated`] (group size = `per_head_dim`) which both
    /// normalises `y` and gates it with `SiLU(z)`. Matches the reference's
    /// `is_outproj_norm` argument in `mamba3.py`.
    #[config(default = false)]
    pub has_outproj_norm: bool,
}

impl Mamba3Config {
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.per_head_dim
    }

    /// Effective RoPE dimension: `2 · num_rope_angles`. Equals `state_rank`
    /// for full RoPE, and `state_rank / 2` for `rope_fraction = 0.5`.
    pub fn rope_dim(&self) -> usize {
        let mut d = (self.state_rank as f64 * self.rope_fraction) as usize;
        if d % 2 != 0 {
            d -= 1;
        }
        d
    }

    pub fn num_rope_angles(&self) -> usize {
        self.rope_dim() / 2
    }

    /// Total input projection output size.
    ///
    /// `d_in_proj = 2·d_inner + 2·ngroups·state_rank·mimo_rank + 3·nheads + num_rope_angles`
    pub fn d_in_proj(&self) -> usize {
        2 * self.d_inner()
            + 2 * self.ngroups * self.state_rank * self.mimo_rank
            + 3 * self.nheads()
            + self.num_rope_angles()
    }

    /// Allocate and initialise all Mamba-3 block parameters on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba3<B> {
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let state_rank = self.state_rank;
        let mimo_rank = self.mimo_rank;
        let num_rope_angles = self.num_rope_angles();

        assert!(
            state_rank % 2 == 0,
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
            self.rope_fraction == 0.5 || self.rope_fraction == 1.0,
            "rope_fraction must be 0.5 or 1.0"
        );
        assert!(num_rope_angles > 0, "num_rope_angles must be at least 1");

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
            .init::<B>(device);

        // dt_bias: inverse-softplus initialisation
        let expm1 = |t: Tensor<B, 1>| t.exp() - 1.;
        let dt_h = Tensor::random(
            [nheads],
            burn::tensor::Distribution::Uniform(self.dt_min.ln(), self.dt_max.ln()),
            device,
        )
        .exp();
        let dt_h = dt_h.clamp(self.dt_init_floor, f64::INFINITY);
        let inv_dt_h = dt_h.clone() + (-expm1(-dt_h)).log();
        let dt_bias_h = Param::from_tensor(inv_dt_h);

        let d_h = Initializer::Ones.init::<B, 1, _>([nheads], device);

        let b_norm = RmsNormConfig::new(state_rank).init(device);
        let c_norm = RmsNormConfig::new(state_rank).init(device);

        // B/C biases: [nheads, mimo_rank, state_rank], init to ones
        let b_bias_hmr = Initializer::Ones.init::<B, 3, _>([nheads, mimo_rank, state_rank], device);
        let c_bias_hmr = Initializer::Ones.init::<B, 3, _>([nheads, mimo_rank, state_rank], device);

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
            Initializer::Zeros.init::<B, 3, _>([nheads, self.per_head_dim, state_rank], device)
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
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE utility
// ---------------------------------------------------------------------------

/// Apply rotary position embeddings to `x` along its last dimension.
///
/// Two pairing conventions are supported, selected by `rotate_pairwise`:
///
/// - `rotate_pairwise = true` — **interleaved** (NeoX / Triton style): adjacent
///   pairs `(0,1)`, `(2,3)`, … are rotated together. Used by the SISO Triton
///   kernel (`mamba3_siso_*.py`).
/// - `rotate_pairwise = false` — **half-and-half** (GPT-J style): position `n`
///   is paired with `n + state_rank/2`. Used by the MIMO Tilelang kernel
///   (`mamba3_mimo_fwd.py`).
///
/// Reference: `mamba3.py:335` sets `rotate_pairwise = not self.is_mimo`.
///
/// # Shapes
/// - `x`:      `[..., state_rank]` where `state_rank` is even
/// - `angles`: `[..., state_rank / 2]`  (one angle per pair)
/// - output:   same shape as `x`
pub fn apply_rope<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    angles: Tensor<B, D>,
    rotate_pairwise: bool,
) -> Tensor<B, D> {
    let dims = x.dims();
    let n = dims[D - 1];
    let n2 = n / 2;
    let leading: usize = dims[..D - 1].iter().product();

    let angles_flat = angles.reshape([leading, n2]);
    let cos = angles_flat.clone().cos();
    let sin = angles_flat.sin();

    if rotate_pairwise {
        // Interleaved: reshape to [leading, n2, 2], pairs along last axis.
        let x_pairs = x.reshape([leading, n2, 2]);
        let x0 = x_pairs.clone().narrow(2, 0, 1).squeeze_dim(2);
        let x1 = x_pairs.narrow(2, 1, 1).squeeze_dim(2);

        let x0r = cos.clone() * x0.clone() - sin.clone() * x1.clone();
        let x1r = sin * x0 + cos * x1;

        Tensor::cat(
            vec![x0r.unsqueeze_dim::<3>(2), x1r.unsqueeze_dim::<3>(2)],
            2,
        )
        .reshape(dims)
    } else {
        // Half-and-half: reshape to [leading, 2, n2], halves along middle axis.
        let x_halves = x.reshape([leading, 2, n2]);
        let x0 = x_halves.clone().narrow(1, 0, 1).squeeze_dim(1);
        let x1 = x_halves.narrow(1, 1, 1).squeeze_dim(1);

        let x0r = cos.clone() * x0.clone() - sin.clone() * x1.clone();
        let x1r = sin * x0 + cos * x1;

        Tensor::cat(
            vec![x0r.unsqueeze_dim::<3>(1), x1r.unsqueeze_dim::<3>(1)],
            1,
        )
        .reshape(dims)
    }
}

/// Apply RoPE to only the rotation-active entries of the last dimension; the
/// remainder passes through unchanged. Falls back to [`apply_rope`] when
/// `rope_dim == state_rank` (full RoPE).
///
/// Pairing scheme (must match the reference kernels — see Section
/// "Data-Dependent RoPE" in the paper, and `mamba3_siso_fwd.py` /
/// `mamba3_mimo_fwd.py`):
///
/// - `rotate_pairwise = true` (SISO, interleaved/NeoX): pairs `(0,1), (2,3), …`.
///   Only pairs `0..num_rope_angles` are rotated; pairs beyond are passed
///   through. Equivalent to slicing the first `rope_dim` entries and rotating
///   them.
/// - `rotate_pairwise = false` (MIMO, half-and-half/GPT-J): pair distance is
///   always `state_rank/2`, i.e. element `n` is paired with element
///   `state_rank/2 + n`. With partial RoPE only the first `num_rope_angles`
///   pairs are rotated; the remaining elements in both halves pass through.
pub(crate) fn apply_rope_partial<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    angles: Tensor<B, D>,
    rope_dim: usize,
    rotate_pairwise: bool,
) -> Tensor<B, D> {
    let state_rank = x.dims()[D - 1];
    if rope_dim == state_rank {
        return apply_rope::<B, D>(x, angles, rotate_pairwise);
    }

    if rotate_pairwise {
        // Pairs are local — slicing the first rope_dim entries gives the same
        // result as the reference (which rotates the whole headdim but with
        // identity cos/sin for the tail pairs).
        let x_rope = x.clone().narrow(D - 1, 0, rope_dim);
        let x_rest = x.narrow(D - 1, rope_dim, state_rank - rope_dim);
        let x_rope_rotated = apply_rope::<B, D>(x_rope, angles, true);
        return Tensor::cat(vec![x_rope_rotated, x_rest], D - 1);
    }

    // Half-and-half partial RoPE: pair distance must be `state_rank/2`, not
    // `rope_dim/2`. Slicing the first `rope_dim` entries and calling
    // `apply_rope` would pair within the slice and produce the wrong rotation.
    let half = state_rank / 2;
    let num_rope_angles = rope_dim / 2;
    debug_assert!(
        num_rope_angles < half,
        "partial RoPE requires rope_dim < state_rank here"
    );

    // Split x into the two halves, then within each half separate the
    // rotation-active prefix from the pass-through suffix.
    let x_h1 = x.clone().narrow(D - 1, 0, half);
    let x_h2 = x.narrow(D - 1, half, half);
    let x_h1_rope = x_h1.clone().narrow(D - 1, 0, num_rope_angles);
    let x_h1_pass = x_h1.narrow(D - 1, num_rope_angles, half - num_rope_angles);
    let x_h2_rope = x_h2.clone().narrow(D - 1, 0, num_rope_angles);
    let x_h2_pass = x_h2.narrow(D - 1, num_rope_angles, half - num_rope_angles);

    // angles: [..., num_rope_angles] — broadcasts element-wise against the rope-active slices.
    let cos = angles.clone().cos();
    let sin = angles.sin();
    let x_h1_rot = cos.clone() * x_h1_rope.clone() - sin.clone() * x_h2_rope.clone();
    let x_h2_rot = sin * x_h1_rope + cos * x_h2_rope;

    // Reassemble: [ first-half-rotated | first-half-passthrough | second-half-rotated | second-half-passthrough ]
    let x_h1_out = Tensor::cat(vec![x_h1_rot, x_h1_pass], D - 1);
    let x_h2_out = Tensor::cat(vec![x_h2_rot, x_h2_pass], D - 1);
    Tensor::cat(vec![x_h1_out, x_h2_out], D - 1)
}

// ---------------------------------------------------------------------------
// Mamba3::forward  (chunkwise two-SSD — training / prefill)
// ---------------------------------------------------------------------------

impl<B: Backend + Mamba3BackendExt> Mamba3<B> {
    /// Process a full input sequence using the trapezoidal two-SSD algorithm.
    ///
    /// For SISO (mimo_rank=1), this is the standard two-SSD decomposition.
    /// For MIMO (mimo_rank>1), B/C have mimo_rank parallel rank channels.
    /// The hidden state is shared across mimo ranks; each mimo rank contributes independently.
    ///
    /// # Shapes
    /// - `input_bsm` : `[batch, sequence, d_model]`
    /// - output      : `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_bsm: Tensor<B, 3>,
        cache: Option<Mamba3Cache<B>>,
        ssd_path: Mamba3SsdPath,
    ) -> (Tensor<B, 3>, Mamba3Cache<B>) {
        let [batch, sequence, _d_model] = input_bsm.dims();
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let num_rope_angles = self.num_rope_angles;
        let mimo_rank = self.mimo_rank;
        let device = input_bsm.device();

        assert!(sequence > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        let mut cache = cache.unwrap_or_else(|| {
            let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
            let k_state_bmhr = Tensor::zeros([batch, mimo_rank, nheads, state_rank], &device);
            let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], &device);
            let cum_angle_bha = Tensor::zeros([batch, nheads, num_rope_angles], &device);
            Mamba3Cache {
                ssm_bhpr,
                k_state_bmhr,
                v_state_bhp,
                cum_angle_bha,
            }
        });

        // ── Step 1: In-projection ─────────────────────────────────────────────
        let proj_bsd = self.in_proj.forward(input_bsm);
        let bc_size = ngroups * state_rank * mimo_rank;

        // [batch, sequence, *] split along channel dim.
        // b_raw_bsMGR / c_raw_bsMGR have channel size `mimo_rank * ngroups * state_rank`.
        #[rustfmt::skip]
        let [
                z_bsi, x_bsi,
                b_raw_bsMGR, c_raw_bsMGR,
                dd_dt_bsh, dd_A_raw_bsh, lambda_raw_bsh,
                theta_bsa
        ] = crate::utils::split::split_into(
            proj_bsd,
            [
                d_inner, d_inner,
                bc_size, bc_size,
                nheads, nheads, nheads,
                num_rope_angles,
            ],
            2,
        );

        san(&z_bsi);
        san(&x_bsi);
        san(&dd_dt_bsh);

        // ── Step 2: Discretisation + trapezoidal coefficients ─────────────────
        let helpers::TrapCoeffs {
            dt: dt_bsh,
            da: da_bsh,
            alpha: _alpha_bsh,
            beta: beta_bsh,
            gamma: gamma_bsh,
        } = helpers::trapezoidal_coefficients(
            dd_dt_bsh,
            dd_A_raw_bsh,
            lambda_raw_bsh,
            self.dt_bias_h.val(),
            self.dt_limit,
            self.a_floor,
        );

        san(&dt_bsh);
        san(&da_bsh);
        san(&beta_bsh);
        san(&gamma_bsh);

        // ── Step 3: Reshape x ─────────────────────────────────────────────────
        let x_bshp = x_bsi.reshape([batch, sequence, nheads, per_head_dim]);

        // ── Step 4: QK-Norm on B and C  ───────────────────────────────────────
        // QK-Norm over state_rank, then expand ngroups→nheads, then add per-(head,
        // mimo-rank) bias [nheads, mimo_rank, state_rank]. Group dim is axis 3 of
        // `_bsmgr` (D = 5).
        let b_bsmhr = helpers::qk_norm_expand_bias::<_, 5, 6>(
            b_raw_bsMGR.reshape([batch, sequence, mimo_rank, ngroups, state_rank]),
            &self.b_norm,
            self.b_bias_hmr.val(),
            3,
            nheads,
        );
        let c_bsmhr = helpers::qk_norm_expand_bias::<_, 5, 6>(
            c_raw_bsMGR.reshape([batch, sequence, mimo_rank, ngroups, state_rank]),
            &self.c_norm,
            self.c_bias_hmr.val(),
            3,
            nheads,
        );
        assert_eq!(
            [batch, sequence, mimo_rank, nheads, state_rank],
            b_bsmhr.dims()
        );
        assert_eq!(
            [batch, sequence, mimo_rank, nheads, state_rank],
            c_bsmhr.dims()
        );

        // ── Step 5: Data-dependent cumulative RoPE angles ─────────────────────
        let theta_scaled_bsa = theta_bsa.tanh() * std::f32::consts::PI;
        let raw_angles_bsha =
            dt_bsh.clone().unsqueeze_dim::<4>(3) // dt_bsh1
            *
            theta_scaled_bsa.unsqueeze_dim::<4>(2) // theta_scaled_bs1a
            ;

        let cumsum_bsha = raw_angles_bsha.cumsum(1);
        let cum_angles_bsha = cache.cum_angle_bha.clone()
            .unsqueeze_dim::<4>(1) // cum_angle_b1ha
            + cumsum_bsha;
        assert_eq!(
            [batch, sequence, nheads, num_rope_angles],
            cum_angles_bsha.dims()
        );
        san(&cum_angles_bsha);

        // Apply RoPE to B and C: angles broadcast over the mimo_rank dim.
        let cum_angles_bsmha = cum_angles_bsha
            .clone()
            .unsqueeze_dim::<5>(2) // cum_angles_bs1ha
            .expand([batch, sequence, mimo_rank, nheads, num_rope_angles]); // cum_angles_bsmha
        // SISO uses interleaved (pairwise) pairing; MIMO uses half-and-half.
        // Partial RoPE: rotate only the first `rope_dim` entries of B/C.
        let rotate_pairwise = mimo_rank == 1;
        let rope_dim = self.rope_dim;
        let b_bsmhr = apply_rope_partial::<B, 5>(
            b_bsmhr,
            cum_angles_bsmha.clone(),
            rope_dim,
            rotate_pairwise,
        );
        let c_bsmhr =
            apply_rope_partial::<B, 5>(c_bsmhr, cum_angles_bsmha, rope_dim, rotate_pairwise);
        san(&b_bsmhr);
        san(&c_bsmhr);

        // ── Step 6: Build shifted inputs for β term ───────────────────────────
        //
        // "Shift-Before-Chunking": prepend the cached xₜ₋₁ / Bₜ₋₁ at the
        // sequence level (before SSD chunking) so the β term at t=0 sees the
        // prior token from a continued cache. For a fresh (zero) cache this is
        // equivalent to zero-padding.
        let x_prev_first_b1hp = cache.v_state_bhp.clone().unsqueeze_dim::<4>(1);
        let x_prev_bshp = if sequence == 1 {
            x_prev_first_b1hp
        } else {
            Tensor::cat(
                vec![x_prev_first_b1hp, x_bshp.clone().narrow(1, 0, sequence - 1)],
                1,
            )
        };
        let b_prev_first_b1mhr = cache.k_state_bmhr.clone().unsqueeze_dim::<5>(1);
        let b_prev_bsmhr = if sequence == 1 {
            b_prev_first_b1mhr
        } else {
            Tensor::cat(
                vec![
                    b_prev_first_b1mhr,
                    b_bsmhr.clone().narrow(1, 0, sequence - 1),
                ],
                1,
            )
        };

        // ── Step 7: Scale inputs by trapezoidal coefficients ──────────────────
        // gamma and beta are per-head scalars, broadcast over mimo_rank and per_head_dim:
        let gamma_bsh1 = gamma_bsh.unsqueeze_dim::<4>(3);
        let beta_bsh1 = beta_bsh.unsqueeze_dim::<4>(3);
        let x_gamma_bshp = x_bshp.clone() * gamma_bsh1; // γₜ · xₜ
        let x_beta_bshp = x_prev_bshp * beta_bsh1; // βₜ · xₜ₋₁

        // ── Save last-token B for cache ───────────────────────────────────────
        let b_last_bmhr = b_bsmhr
            .clone()
            .narrow(1, sequence - 1, 1)
            .reshape([batch, mimo_rank, nheads, state_rank]);

        // ── Step 8: Pad sequence to multiple of chunk_len ─────────────────────
        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        #[rustfmt::skip]
        let (x_gamma_bShp, x_beta_bShp, da_bSh, b_bSmhr, b_prev_bSmhr, c_bSmhr) = if pad == 0 {
            (x_gamma_bshp, x_beta_bshp, da_bsh, b_bsmhr, b_prev_bsmhr, c_bsmhr)
        } else {
            let pad_bShp = Tensor::zeros([batch, pad, nheads, per_head_dim], &device);
            let pad_bSh = Tensor::zeros([batch, pad, nheads], &device);
            let pad_bSmhr = Tensor::zeros([batch, pad, mimo_rank, nheads, state_rank], &device);
            (
                Tensor::cat(vec![x_gamma_bshp, pad_bShp.clone()], 1),
                Tensor::cat(vec![x_beta_bshp, pad_bShp], 1),
                Tensor::cat(vec![da_bsh, pad_bSh], 1),
                Tensor::cat(vec![b_bsmhr, pad_bSmhr.clone()], 1),
                Tensor::cat(vec![b_prev_bsmhr, pad_bSmhr.clone()], 1),
                Tensor::cat(vec![c_bsmhr, pad_bSmhr], 1),
            )
        };

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let x_gamma_bnlhp = x_gamma_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let x_beta_bnlhp = x_beta_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlmhr = b_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let b_prev_bnlmhr =
            b_prev_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bnlmhr = c_bSmhr.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

        // ── Step 9: Two MIMO-SSD calls ────────────────────────────────────────
        // Build V tensors — insert the mimo_rank axis at position 3 of `_bnlhp`.
        let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
        let v_gamma_bnlmhp =
            helpers::build_v_with_mimo::<_, 5, 6>(x_gamma_bnlhp.clone(), mimo_x_hmp.as_ref(), 3);
        let v_beta_bnlmhp =
            helpers::build_v_with_mimo::<_, 5, 6>(x_beta_bnlhp, mimo_x_hmp.as_ref(), 3);

        let input_gamma = Mamba3SsdInput {
            v_bnlmhp: v_gamma_bnlmhp,
            da_bnlh: da_bnlh.clone(),
            b_bnlmhr: b_bnlmhr.clone(),
            c_bnlmhr: c_bnlmhr.clone(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        };
        let (y_gamma_bnlmhp, final_state_gamma_bhpr) = ssd_path.clone().run(input_gamma);

        let input_beta = Mamba3SsdInput {
            v_bnlmhp: v_beta_bnlmhp,
            da_bnlh,
            b_bnlmhr: b_prev_bnlmhr,
            c_bnlmhr,
            initial_state_bhpr: Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device),
            init_state_hpr: None,
        };
        let (y_beta_bnlmhp, final_state_beta_bhpr) = ssd_path.run(input_beta);

        let y_bnlmhp = y_gamma_bnlmhp + y_beta_bnlmhp;
        let final_state_bhpr = final_state_gamma_bhpr + final_state_beta_bhpr;

        san(&y_bnlmhp);
        san(&final_state_bhpr);

        cache.ssm_bhpr = final_state_bhpr;

        // ── Step 10: Unpad ────────────────────────────────────────────────────
        let y_bSmhp = y_bnlmhp.reshape([batch, sequence_padded, mimo_rank, nheads, per_head_dim]);
        let y_bsmhp = if pad == 0 {
            y_bSmhp
        } else {
            y_bSmhp.narrow(1, 0, sequence)
        };

        // ── Step 11: D skip + gate + aggregate ranks ──────────────────────────
        // D skip uses raw x * mimo_x_hmp (not gamma-scaled)
        // Insert the mimo_rank axis at position 2 of `_bshp`.
        let v_raw_bsmhp =
            helpers::build_v_with_mimo::<_, 4, 5>(x_bshp.clone(), mimo_x_hmp.as_ref(), 2);

        let d_111h1 = self.d_h.val().unsqueeze_dims::<5>(&[0, 1, 2, 4]);
        let y_bsmhp = y_bsmhp + d_111h1 * v_raw_bsmhp.clone();

        // ── Gate (or gated norm) and rank aggregation ─────────────────────────
        // When `out_norm` is set, the SiLU gate is replaced by a per-head
        // gated RMSNorm: `RmsNormGated(y, z) = norm(y) * silu(z)`.
        let y_bsi = if mimo_rank > 1 {
            let mimo_z_hmp = self.mimo_z_hmp.as_ref().map(|p| p.val()).unwrap();
            let mimo_o_hmp = self.mimo_o_hmp.as_ref().map(|p| p.val()).unwrap();

            let z_bshp = z_bsi
                .clone()
                .reshape([batch, sequence, nheads, per_head_dim]);
            let z_bsmhp = {
                let z_bsmhp = z_bshp
                    .unsqueeze_dim::<5>(2) // z_bs1hp
                    .expand([batch, sequence, mimo_rank, nheads, per_head_dim]); // z_bsmhp
                let mimo_z_bsmhp = mimo_z_hmp
                    .permute([1, 0, 2]) // mimo_z_mhp
                    .unsqueeze_dims::<5>(&[0, 1]) // mimo_z_11mhp
                    .expand([batch, sequence, mimo_rank, nheads, per_head_dim]); // mimo_z_bsmhp
                z_bsmhp * mimo_z_bsmhp
            };

            // gate or gated norm:
            //   without out_norm: y_r * silu(z_r)
            //   with    out_norm: norm(y_r) * silu(z_r)  (norm over per_head_dim)
            let y_combined_bsmhp = match &self.out_norm {
                Some(norm) => norm.forward(y_bsmhp, z_bsmhp),
                None => y_bsmhp * Silu::new().forward(z_bsmhp),
            };

            // Down-project with mimoₒ_hmp: out = sumₘ mimoₒ_hmp[h, r, p] * yᵣ
            let mimo_o_bsmhp = mimo_o_hmp
                .permute([1, 0, 2]) // mimo_o_mhp
                .unsqueeze_dims::<5>(&[0, 1]) // mimo_o_11mhp
                .expand([batch, sequence, mimo_rank, nheads, per_head_dim]); // mimo_o_bsmhp
            // sum over mimo rank dim
            let y_bshp: Tensor<B, 4> = (y_combined_bsmhp * mimo_o_bsmhp)
                .sum_dim(2) // y_bs1hp
                .squeeze_dim(2); // y_bshp
            y_bshp.reshape([batch, sequence, d_inner])
        } else {
            // SISO: squeeze rank dim, apply gate (or gated norm) over per_head_dim.
            let y_bshp: Tensor<B, 4> = y_bsmhp.squeeze_dim(2); // mimo_rank == 1
            let z_bshp = z_bsi.reshape([batch, sequence, nheads, per_head_dim]);
            let y_combined_bshp = match &self.out_norm {
                Some(norm) => norm.forward(y_bshp, z_bshp),
                None => y_bshp * Silu::new().forward(z_bshp),
            };
            y_combined_bshp.reshape([batch, sequence, d_inner])
        };
        san(&y_bsi);

        // ── Out-projection ────────────────────────────────────────────────────
        let out_bsm = self.out_proj.forward(y_bsi);
        san(&out_bsm);

        // ── Update remaining cache fields ─────────────────────────────────────
        // k_state = B at last token
        cache.k_state_bmhr = b_last_bmhr;

        // v_state = x at last token
        cache.v_state_bhp = x_bshp
            .narrow(1, sequence - 1, 1) // x_b1hp
            .squeeze_dim::<3>(1); // x_bhp

        // cum_angle at last token
        cache.cum_angle_bha = cum_angles_bsha
            .narrow(1, sequence - 1, 1) // cum_angles_b1ha
            .squeeze_dim::<3>(1); // cum_angles_bha

        (out_bsm, cache)
    }
}

// ---------------------------------------------------------------------------
// Mamba3::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    impl<B: Backend> Mamba3<B> {
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
            input_bd: Tensor<B, 2>,
            cache: Option<Mamba3Cache<B>>,
        ) -> (Tensor<B, 2>, Mamba3Cache<B>) {
            let [batch, d_model] = input_bd.dims();
            let d_inner = self.d_inner();
            let nheads = self.nheads();
            let ngroups = self.ngroups;
            let per_head_dim = self.per_head_dim();
            let state_rank = self.state_rank;
            let num_rope_angles = self.num_rope_angles;
            let mimo_rank = self.mimo_rank;
            let device = &input_bd.device();
            let ssm_shape = [batch, nheads, per_head_dim, state_rank];

            assert_eq!(nheads % ngroups, 0);

            let mut cache = cache.unwrap_or_else(|| {
                let ssm_bhpr = Tensor::zeros(ssm_shape, device);
                let k_state_bmhr = Tensor::zeros([batch, mimo_rank, nheads, state_rank], device);
                let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], device);
                let cum_angle_bha = Tensor::zeros([batch, nheads, num_rope_angles], device);
                Mamba3Cache {
                    ssm_bhpr,
                    k_state_bmhr,
                    v_state_bhp,
                    cum_angle_bha,
                }
            });

            // ── In-projection ─────────────────────────────────────────────────
            let proj_bd = self.in_proj.forward(input_bd);
            let bc_size = ngroups * state_rank * mimo_rank;
            // [batch, *] split along channel dim.
            // b_raw_bMGR / c_raw_bMGR have channel size `mimo_rank * ngroups * state_rank`.
            #[rustfmt::skip]
            let [
                    z_bi, x_bi,
                    b_raw_bMGR, c_raw_bMGR,
                    dd_dt_bh, dd_a_raw_bh, lambda_raw_bh,
                    theta_ba,
            ] = crate::utils::split::split_into(
                proj_bd,
                [
                    d_inner, d_inner,
                    bc_size, bc_size,
                    nheads, nheads, nheads,
                    num_rope_angles,
                ],
                1,
            );

            // ── Reshape x ─────────────────────────────────────────────────────
            let x_bhp = x_bi.reshape([batch, nheads, per_head_dim]);

            // ── Discretisation + trapezoidal coefficients ─────────────────────
            let helpers::TrapCoeffs {
                dt: dt_bh,
                da: _da_bh,
                alpha: alpha_bh,
                beta: beta_bh,
                gamma: gamma_bh,
            } = helpers::trapezoidal_coefficients(
                dd_dt_bh,
                dd_a_raw_bh,
                lambda_raw_bh,
                self.dt_bias_h.val(),
                self.dt_limit,
                self.a_floor,
            );

            // ── QK-Norm on B and C ────────────────────────────────────────────
            // Group dim is axis 2 of `_bmgr` (D = 4).
            let b_bmhr = helpers::qk_norm_expand_bias::<_, 4, 5>(
                b_raw_bMGR.reshape([batch, mimo_rank, ngroups, state_rank]),
                &self.b_norm,
                self.b_bias_hmr.val(),
                2,
                nheads,
            );
            let c_bmhr = helpers::qk_norm_expand_bias::<_, 4, 5>(
                c_raw_bMGR.reshape([batch, mimo_rank, ngroups, state_rank]),
                &self.c_norm,
                self.c_bias_hmr.val(),
                2,
                nheads,
            );
            assert_eq!([batch, mimo_rank, nheads, state_rank], b_bmhr.dims());

            // ── RoPE: update cumulative angle, rotate B and C ──────────────────
            let theta_scaled_ba = theta_ba.tanh() * std::f32::consts::PI;
            let raw_angle_bha = dt_bh.unsqueeze_dim::<3>(2) * theta_scaled_ba.unsqueeze_dim::<3>(1);
            let new_cum_angle_bha = cache.cum_angle_bha.clone() + raw_angle_bha;

            // Broadcast angles over mimo_rank
            let new_cum_angle_bmha = new_cum_angle_bha
                .clone()
                .unsqueeze_dim::<4>(1) // new_cum_angle_b1ha
                .expand([batch, mimo_rank, nheads, num_rope_angles]); // new_cum_angle_bmha
            // SISO uses interleaved (pairwise) pairing; MIMO uses half-and-half.
            // Partial RoPE: rotate only the first `rope_dim` entries of B/C.
            let rotate_pairwise = mimo_rank == 1;
            let rope_dim = self.rope_dim;
            let b_bmhr = apply_rope_partial::<B, 4>(
                b_bmhr,
                new_cum_angle_bmha.clone(),
                rope_dim,
                rotate_pairwise,
            );
            let c_bmhr =
                apply_rope_partial::<B, 4>(c_bmhr, new_cum_angle_bmha, rope_dim, rotate_pairwise);

            // ── Build MIMO value tensors ───────────────────────────────────────
            // Insert the mimo_rank axis at position 1 of `_bhp`.
            let mimo_x_hmp = self.mimo_x_hmp.as_ref().map(|p| p.val());
            let x_vals_bmhp =
                helpers::build_v_with_mimo::<_, 3, 4>(x_bhp.clone(), mimo_x_hmp.as_ref(), 1);
            let xs_vals_bmhp = helpers::build_v_with_mimo::<_, 3, 4>(
                cache.v_state_bhp.clone(),
                mimo_x_hmp.as_ref(),
                1,
            );

            // ── SSM state update ───────────────────────────────────────────────
            // new_state[b, h, p, r] = alpha * state
            //   + sumₘ gamma * x_vals[m] ⊗ B_cur[m]
            //   + sumₘ beta  * xs_vals[m] ⊗ B_state[m]
            //
            // For the outer product sum:
            //   xBt[b, h, p, r] = sumₘ coeff[m, h, p] * B[m, h, n]
            //   = einsum('bmhp,bmhr->bhpr', coeff*x_vals, B)
            //   = matmul over m: [b, h, p, m] @ [b, h, m, r]
            // x_vals_bmhp * gamma_b1h1
            // Need gamma as [b, 1, h, 1] to broadcast over m and p:
            let gamma_b1h1 = gamma_bh.clone().unsqueeze_dims::<4>(&[1, 3]);
            let beta_b1h1 = beta_bh.clone().unsqueeze_dims::<4>(&[1, 3]);

            let x_gamma_bmhp = x_vals_bmhp.clone() * gamma_b1h1;
            let x_beta_bmhp = xs_vals_bmhp * beta_b1h1;

            // einsum('bmhp,bmhr->bhpr', x_gamma, B_cur):
            let xbt_state_bhpr = {
                let b_bhmr = b_bmhr.clone().permute([0, 2, 1, 3]);
                let xg_bhpm = x_gamma_bmhp.permute([0, 2, 3, 1]);
                xg_bhpm.matmul(b_bhmr)
            };
            let xbt_prev_bhpr = {
                let b_state_bhmr = cache.k_state_bmhr.clone().permute([0, 2, 1, 3]);
                let xb_bhpm = x_beta_bmhp.permute([0, 2, 3, 1]);
                xb_bhpm.matmul(b_state_bhmr)
            };

            let alpha_bh11 = alpha_bh.unsqueeze_dims::<4>(&[2, 3]);
            let new_state_bhpr =
                alpha_bh11 * cache.ssm_bhpr.clone() + xbt_state_bhpr + xbt_prev_bhpr;

            // ── Output ────────────────────────────────────────────────────────
            // outₘ[b, m, h, p] = sumᵣ C[b, m, h, r] * state[b, h, p, r] + D * x_vals[b, m, h, p]
            // = einsum('bhpr,bmhr->bmhp', state, C)
            let out_m_bmhp = {
                let c_bhrm = c_bmhr.permute([0, 2, 3, 1]);
                let out_bhpm = new_state_bhpr.clone().matmul(c_bhrm);
                out_bhpm.permute([0, 3, 1, 2])
            };

            // D skip
            let d_bmhp = self
                .d_h
                .val()
                .unsqueeze_dims::<4>(&[0, 1, 3]) // d_11h1
                .expand([batch, mimo_rank, nheads, per_head_dim]); // d_bmhp
            let out_m_bmhp = out_m_bmhp + d_bmhp * x_vals_bmhp;

            // ── Gate (or gated norm) and rank aggregation ─────────────────────
            // When `out_norm` is set, the SiLU gate is replaced by a per-head
            // gated RMSNorm: `RmsNormGated(y, z) = norm(y) * silu(z)`.
            let z_bhp = z_bi.reshape([batch, nheads, per_head_dim]);
            let y_bi = if mimo_rank > 1 {
                let mimo_z_hmp = self.mimo_z_hmp.as_ref().map(|p| p.val()).unwrap();
                let mimo_o_hmp = self.mimo_o_hmp.as_ref().map(|p| p.val()).unwrap();

                // zₘ = z * mimo_z_hmp[m]
                let z_bmhp = z_bhp
                    .unsqueeze_dim::<4>(1) // z_b1hp
                    .expand([batch, mimo_rank, nheads, per_head_dim]); // z_bmhp
                // mimo_z_hmp
                let mimo_z_bmhp = mimo_z_hmp
                    .permute([1, 0, 2]) // mimo_z_mhp
                    .unsqueeze_dim::<4>(0) // mimo_z_1mhp
                    .expand([batch, mimo_rank, nheads, per_head_dim]); // mimo_z_bmhp
                let z_bmhp = z_bmhp * mimo_z_bmhp;

                // Per-rank gate or gated norm.
                let combined_bmhp = match &self.out_norm {
                    Some(norm) => norm.forward(out_m_bmhp, z_bmhp),
                    None => out_m_bmhp * Silu::new().forward(z_bmhp),
                };

                // Project down: out = sumₘ mimo_o_hmp[m] * combined_bmhp[m]
                let mimo_o_bmhp = mimo_o_hmp
                    .permute([1, 0, 2]) // mimo_o_mhp
                    .unsqueeze_dim::<4>(0) // mimo_o_1mhp
                    .expand([batch, mimo_rank, nheads, per_head_dim]); // mimo_o_bmhp
                let out_bhp: Tensor<B, 3> = (combined_bmhp * mimo_o_bmhp)
                    .sum_dim(1) // out_b1hp
                    .squeeze_dim(1); // out_bhp
                out_bhp.reshape([batch, d_inner]) // y_bi
            } else {
                // SISO: squeeze rank dim, gate (or gated norm) over per_head_dim.
                let y_bhp: Tensor<B, 3> = out_m_bmhp.squeeze_dim(1);
                let combined = match &self.out_norm {
                    Some(norm) => norm.forward(y_bhp, z_bhp),
                    None => y_bhp * Silu::new().forward(z_bhp),
                };
                combined.reshape([batch, d_inner])
            };

            // ── Out-projection ────────────────────────────────────────────────
            let out_bm = self.out_proj.forward(y_bi);
            assert_eq!([batch, d_model], out_bm.dims());

            // ── Update cache ──────────────────────────────────────────────────
            cache.ssm_bhpr = new_state_bhpr;
            cache.k_state_bmhr = b_bmhr;
            cache.v_state_bhp = x_bhp;
            cache.cum_angle_bha = new_cum_angle_bha;

            (out_bm, cache)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Flex};
    use burn::tensor::Distribution;

    /// Inner (non-autodiff) backend used for materialising values and
    /// extracted gradients.
    type InnerB = Flex;
    /// Autodiff-wrapped backend used to drive `.backward()`.
    type B = Autodiff<InnerB>;

    type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

    fn small_config() -> Mamba3Config {
        Mamba3Config::new(32) // d_model = 32
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
    }

    fn small_config_mimo() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_mimo_rank(2)
    }

    /// A bundle of input + model-parameter gradients extracted from one
    /// forward+backward run.  Each `check_grads_match` call compares these
    /// across two runs that should be mathematically equivalent.
    struct RunGrads {
        out: Tensor<InnerB, 3>,
        /// Final SSM hidden state from the returned cache.
        final_ssm: Tensor<InnerB, 4>,
        /// Final previous-token B state from the returned cache.
        final_k: Tensor<InnerB, 4>,
        /// Final previous-token x state from the returned cache.
        final_v: Tensor<InnerB, 3>,
        /// Final cumulative RoPE angle from the returned cache.
        final_angle: Tensor<InnerB, 3>,
        d_input: Tensor<InnerB, 3>,
        d_in_proj_w: Tensor<InnerB, 2>,
        d_dt_bias: Tensor<InnerB, 1>,
        d_d: Tensor<InnerB, 1>,
        d_b_norm_gamma: Tensor<InnerB, 1>,
        d_c_norm_gamma: Tensor<InnerB, 1>,
        d_b_bias: Tensor<InnerB, 3>,
        d_c_bias: Tensor<InnerB, 3>,
        d_out_proj_w: Tensor<InnerB, 2>,
    }

    /// Fixed (non-tracked) random "downstream heads" used to form a scalar loss
    /// from the output **and** every final cache field, so the backward pass
    /// exercises both the output and the state path.
    struct Heads {
        out: Tensor<InnerB, 3>,
        ssm: Tensor<InnerB, 4>,
        k: Tensor<InnerB, 4>,
        v: Tensor<InnerB, 3>,
        angle: Tensor<InnerB, 3>,
    }

    /// Build the initial cache passed to both `forward` and the `step`
    /// unrolling. With `random = false` it is zero (the standard fresh start);
    /// with `random = true` every field (SSM state, previous-token B/x, and
    /// cumulative RoPE angle) holds random values, exercising parity from an
    /// arbitrary initial state.
    fn build_init_cache(cfg: &Mamba3Config, batch: usize, random: bool) -> Mamba3Cache<B> {
        let device: Device = Default::default();
        let nheads = cfg.nheads();
        let per_head_dim = cfg.per_head_dim;
        let state_rank = cfg.state_rank;
        let mimo_rank = cfg.mimo_rank;
        let num_rope_angles = cfg.num_rope_angles();
        let dist = Distribution::Normal(0.0, 1.0);
        let mk4 = |shape: [usize; 4]| {
            let t = if random {
                Tensor::<InnerB, 4>::random(shape, dist, &device)
            } else {
                Tensor::<InnerB, 4>::zeros(shape, &device)
            };
            Tensor::from_inner(t)
        };
        let mk3 = |shape: [usize; 3]| {
            let t = if random {
                Tensor::<InnerB, 3>::random(shape, dist, &device)
            } else {
                Tensor::<InnerB, 3>::zeros(shape, &device)
            };
            Tensor::from_inner(t)
        };
        Mamba3Cache {
            ssm_bhpr: mk4([batch, nheads, per_head_dim, state_rank]),
            k_state_bmhr: mk4([batch, mimo_rank, nheads, state_rank]),
            v_state_bhp: mk3([batch, nheads, per_head_dim]),
            cum_angle_bha: mk3([batch, nheads, num_rope_angles]),
        }
    }

    /// Compare the output and every final cache field of two runs.
    fn assert_outputs_match(label: &str, a: &RunGrads, b: &RunGrads, tol: f32) {
        use crate::utils::test_helpers::max_abs_diff;
        let checks = [
            ("output", max_abs_diff(a.out.clone(), b.out.clone())),
            ("final ssm", max_abs_diff(a.final_ssm.clone(), b.final_ssm.clone())),
            ("final k_state", max_abs_diff(a.final_k.clone(), b.final_k.clone())),
            ("final v_state", max_abs_diff(a.final_v.clone(), b.final_v.clone())),
            ("final cum_angle", max_abs_diff(a.final_angle.clone(), b.final_angle.clone())),
        ];
        for (name, d) in checks {
            assert!(d < tol, "{label}: {name} max abs diff = {d:.6} (tol {tol})");
        }
    }

    /// Run a closure that produces an output tensor from a model and an input
    /// (wrapped as a `Param` so it has its own autodiff leaf), then derive a
    /// scalar loss with a fixed (non-tracked) random "head" and return the
    /// gradients of the input and a representative set of model parameters.
    fn run_with_grads(
        model: &Mamba3<B>,
        input: &Param<Tensor<B, 3>>,
        heads: &Heads,
        forward: impl FnOnce(&Mamba3<B>, Tensor<B, 3>) -> (Tensor<B, 3>, Mamba3Cache<B>),
    ) -> RunGrads {
        let (out, cache) = forward(model, input.val());
        let out_inner = out.clone().inner();
        let ssm = cache.ssm_bhpr;
        let k = cache.k_state_bmhr;
        let v = cache.v_state_bhp;
        let angle = cache.cum_angle_bha;
        let final_ssm = ssm.clone().inner();
        let final_k = k.clone().inner();
        let final_v = v.clone().inner();
        let final_angle = angle.clone().inner();

        // Loss couples the output and every final cache field (each via its own
        // random head) so parameter gradients reflect both output and state.
        let out_head = Tensor::from_inner(heads.out.clone());
        let ssm_head = Tensor::from_inner(heads.ssm.clone());
        let k_head = Tensor::from_inner(heads.k.clone());
        let v_head = Tensor::from_inner(heads.v.clone());
        let angle_head = Tensor::from_inner(heads.angle.clone());
        let loss = (out * out_head).sum()
            + (ssm * ssm_head).sum()
            + (k * k_head).sum()
            + (v * v_head).sum()
            + (angle * angle_head).sum();
        let grads = loss.backward();

        RunGrads {
            out: out_inner,
            final_ssm,
            final_k,
            final_v,
            final_angle,
            d_input: input.val().grad(&grads).expect("grad input"),
            d_in_proj_w: model
                .in_proj
                .weight
                .val()
                .grad(&grads)
                .expect("grad in_proj.weight"),
            d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("grad dt_bias_h"),
            d_d: model.d_h.val().grad(&grads).expect("grad d_h"),
            d_b_norm_gamma: model
                .b_norm
                .gamma
                .val()
                .grad(&grads)
                .expect("grad b_norm.gamma"),
            d_c_norm_gamma: model
                .c_norm
                .gamma
                .val()
                .grad(&grads)
                .expect("grad c_norm.gamma"),
            d_b_bias: model
                .b_bias_hmr
                .val()
                .grad(&grads)
                .expect("grad b_bias_hmr"),
            d_c_bias: model
                .c_bias_hmr
                .val()
                .grad(&grads)
                .expect("grad c_bias_hmr"),
            d_out_proj_w: model
                .out_proj
                .weight
                .val()
                .grad(&grads)
                .expect("grad out_proj.weight"),
        }
    }

    /// Assert that every entry in `a` and `b` agrees to within `grad_tol`,
    /// printing every comparison so a failure dump shows the full picture
    /// (instead of stopping at the first mismatch).
    fn check_grads_match(label: &str, a: &RunGrads, b: &RunGrads, grad_tol: f32) {
        let mut failures: Vec<String> = Vec::new();
        macro_rules! check {
            ($field:ident, $name:expr) => {{
                let d = (a.$field.clone() - b.$field.clone())
                    .abs()
                    .max()
                    .into_scalar();
                eprintln!("{:>40} {:>16} | max abs diff = {:>10.6}", label, $name, d);
                if d >= grad_tol {
                    failures.push(format!(
                        "{}: grad of {} max abs diff = {:.6} (tol {})",
                        label, $name, d, grad_tol
                    ));
                }
            }};
        }
        check!(d_input, "input");
        check!(d_in_proj_w, "in_proj.weight");
        check!(d_dt_bias, "dt_bias_h");
        check!(d_d, "d_h");
        check!(d_b_norm_gamma, "b_norm.gamma");
        check!(d_c_norm_gamma, "c_norm.gamma");
        check!(d_b_bias, "b_bias_hmr");
        check!(d_c_bias, "c_bias_hmr");
        check!(d_out_proj_w, "out_proj.weight");
        assert!(
            failures.is_empty(),
            "gradient mismatches:\n  {}",
            failures.join("\n  ")
        );
    }

    /// Build a fresh `Param<Tensor>` from a stable inner tensor.
    /// A new Param is needed per run so that the autodiff leaf has a fresh
    /// node, isolating each backward pass to its own forward graph.
    fn param_input(input: &Tensor<InnerB, 3>) -> Param<Tensor<B, 3>> {
        Param::from_tensor(Tensor::from_inner(input.clone()))
    }

    /// `forward(x)` is mathematically equivalent to repeatedly calling `step`
    /// token-by-token from the **same** initial cache. Outputs, every final
    /// cache field (SSM state, previous-token B/x, cumulative RoPE angle), and
    /// parameter gradients must all agree up to float-summation-order noise.
    ///
    /// With `random_init = true` the shared initial cache is random rather than
    /// zero. Parity from an arbitrary initial state subsumes the chunked-prefill
    /// (split-vs-full) guarantee: if `forward` from any state matches the
    /// recurrent unrolling from that same state — outputs *and* final cache —
    /// then feeding a `forward`-produced cache back in continues correctly.
    fn run_step_matches_forward(cfg: Mamba3Config, random_init: bool) {
        let device: Device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 5;
        let d_model = cfg.d_model;
        let nheads = cfg.nheads();
        let per_head_dim = cfg.per_head_dim;
        let state_rank = cfg.state_rank;
        let mimo_rank = cfg.mimo_rank;
        let num_rope_angles = cfg.num_rope_angles();
        let normal = Distribution::Normal(0.0, 1.0);

        let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
        let heads = Heads {
            out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
            ssm: Tensor::<InnerB, 4>::random(
                [batch, nheads, per_head_dim, state_rank],
                normal,
                &device,
            ),
            k: Tensor::<InnerB, 4>::random(
                [batch, mimo_rank, nheads, state_rank],
                normal,
                &device,
            ),
            v: Tensor::<InnerB, 3>::random([batch, nheads, per_head_dim], normal, &device),
            angle: Tensor::<InnerB, 3>::random([batch, nheads, num_rope_angles], normal, &device),
        };

        let ssd_path = Mamba3SsdPath::Minimal(Some(4));
        let init_cache = build_init_cache(&cfg, batch, random_init);

        let input_fwd = param_input(&input);
        let cache_fwd = init_cache.clone();
        let path_fwd = ssd_path.clone();
        let r_fwd = run_with_grads(&model, &input_fwd, &heads, |m, x| {
            m.forward(x, Some(cache_fwd), path_fwd)
        });

        let input_step = param_input(&input);
        let cache_step = init_cache;
        let r_step = run_with_grads(&model, &input_step, &heads, |m, x| {
            let mut cache: Option<Mamba3Cache<B>> = Some(cache_step);
            let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
                let (out_t, new_cache) = m.step(token, cache);
                cache = Some(new_cache);
                outs.push(out_t);
            }
            (Tensor::stack(outs, 1), cache.unwrap())
        });

        assert_outputs_match("step vs forward", &r_fwd, &r_step, 1e-4);
        check_grads_match("step vs forward", &r_fwd, &r_step, 1e-3);
    }

    // Config variants exercised by the parity tests below.
    fn cfg_ngroups2() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_ngroups(2)
    }
    fn cfg_mimo_ngroups2() -> Mamba3Config {
        cfg_ngroups2().with_mimo_rank(2)
    }
    fn cfg_rope_half() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_rope_fraction(0.5)
    }
    fn cfg_rope_half_mimo() -> Mamba3Config {
        cfg_rope_half().with_mimo_rank(2)
    }
    fn cfg_outproj_norm() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_has_outproj_norm(true)
    }
    fn cfg_outproj_norm_mimo() -> Mamba3Config {
        cfg_outproj_norm().with_mimo_rank(2)
    }
    fn cfg_rope_half_outproj_norm_mimo() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_mimo_rank(2)
            .with_rope_fraction(0.5)
            .with_has_outproj_norm(true)
    }

    #[test]
    fn step_matches_forward() {
        run_step_matches_forward(small_config(), false);
    }

    #[test]
    fn step_matches_forward_random_init() {
        run_step_matches_forward(small_config(), true);
    }

    #[test]
    fn step_matches_forward_ngroups2() {
        run_step_matches_forward(cfg_ngroups2(), false);
    }

    #[test]
    fn step_matches_forward_ngroups2_random_init() {
        run_step_matches_forward(cfg_ngroups2(), true);
    }

    #[test]
    fn step_matches_forward_mimo() {
        run_step_matches_forward(small_config_mimo(), false);
    }

    #[test]
    fn step_matches_forward_mimo_random_init() {
        run_step_matches_forward(small_config_mimo(), true);
    }

    #[test]
    fn step_matches_forward_mimo_ngroups2() {
        run_step_matches_forward(cfg_mimo_ngroups2(), false);
    }

    #[test]
    fn step_matches_forward_mimo_ngroups2_random_init() {
        run_step_matches_forward(cfg_mimo_ngroups2(), true);
    }

    // ── rope_fraction = 0.5 (partial RoPE) ──────────────────────────────────

    #[test]
    fn step_matches_forward_rope_half() {
        run_step_matches_forward(cfg_rope_half(), false);
    }

    #[test]
    fn step_matches_forward_rope_half_random_init() {
        run_step_matches_forward(cfg_rope_half(), true);
    }

    #[test]
    fn step_matches_forward_rope_half_mimo() {
        run_step_matches_forward(cfg_rope_half_mimo(), false);
    }

    #[test]
    fn step_matches_forward_rope_half_mimo_random_init() {
        run_step_matches_forward(cfg_rope_half_mimo(), true);
    }

    // ── has_outproj_norm = true (gated RMSNorm) ─────────────────────────────

    #[test]
    fn step_matches_forward_outproj_norm() {
        run_step_matches_forward(cfg_outproj_norm(), false);
    }

    #[test]
    fn step_matches_forward_outproj_norm_random_init() {
        run_step_matches_forward(cfg_outproj_norm(), true);
    }

    #[test]
    fn step_matches_forward_outproj_norm_mimo() {
        run_step_matches_forward(cfg_outproj_norm_mimo(), false);
    }

    #[test]
    fn step_matches_forward_outproj_norm_mimo_random_init() {
        run_step_matches_forward(cfg_outproj_norm_mimo(), true);
    }

    // ── Both features combined ──────────────────────────────────────────────

    #[test]
    fn step_matches_forward_rope_half_outproj_norm_mimo() {
        run_step_matches_forward(cfg_rope_half_outproj_norm_mimo(), false);
    }

    #[test]
    fn step_matches_forward_rope_half_outproj_norm_mimo_random_init() {
        run_step_matches_forward(cfg_rope_half_outproj_norm_mimo(), true);
    }
}
