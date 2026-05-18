//! # Mamba-3 SSM Block — Exponential-Trapezoidal SSD with Data-Dependent RoPE
//!
//! This module implements the core **Mamba-3 layer** from the paper
//! *"The Mamba-3 Framework: Structured State Spaces with Trapezoidal
//! Discretization and Data-Dependent Rotary Embeddings"*.
//!
//! ## The Mamba-3 Recurrence (SISO, Proposition 1)
//!
//! ```text
//!   hₜ = αₜ hₜ₋₁ + βₜ B_{t-1} x_{t-1}ᵀ + γₜ Bₜ xₜᵀ   (state update)
//!   yₜ = Cₜᵀ hₜ + D xₜ                                  (output)
//! ```
//!
//! ## MIMO Extension (mimo_rank = R > 1)
//!
//! With MIMO, the state update becomes a sum of R outer-product contributions:
//!
//! ```text
//!   hₜ = αₜ hₜ₋₁ + βₜ Σ_r B_{t-1}[r] ⊗ (x_{t-1} ⊙ mimo_x[r])
//!                   + γₜ Σ_r Bₜ[r] ⊗ (xₜ ⊙ mimo_x[r])
//!   yₜ[r] = Cₜ[r]ᵀ hₜ + D xₜ ⊙ mimo_x[r]
//!   outₜ = Σ_r mimo_o[r] ⊙ silu(zₜ ⊙ mimo_z[r]) ⊙ yₜ[r]
//! ```
//!
//! The hidden state hₜ is shared across ranks; each rank contributes to it
//! independently but reads the full shared state when producing its output.
//!
//! ## Notation / Dimension Keys
//!
//! | Letter | Dimension | Typical value |
//! |--------|-----------|---------------|
//! | `b`    | batch     | varies        |
//! | `s`    | sequence length T | varies |
//! | `m`    | d_model   | 768, 1024 … |
//! | `i`    | d_inner = expand·d_model | 2·d_model |
//! | `h`    | nheads H  | d_inner / P  |
//! | `p`    | per_head_dim P | 64, 128 |
//! | `r`    | state_rank N   | 64–256  |
//! | `R`    | mimo_rank      | 1–8     |
//! | `n`    | nchunks = T/Q  | varies  |
//! | `l`    | chunk_len Q    | 64–256  |
//! | `a`    | num_rope_angles = state_rank / 2 | varies |

use crate::mamba3::prelude::*;
use crate::utils::sanity::sanity as san;
use crate::utils::{
    rms_norm::{RmsNorm, RmsNormConfig},
    rms_norm_gated::{RmsNormGated, RmsNormGatedConfig},
    silu::Silu,
    softplus::softplus,
};
use burn::prelude::*;
use burn::{
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
};

/// Element-wise sigmoid: σ(x) = 1 / (1 + exp(-x)).
fn sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    ((-x).exp() + 1.).recip()
}

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
    /// For SISO (R=1): maps `d_model → 2·d_inner + 2·ngroups·state_rank + 3·nheads + num_rope_angles`.
    /// For MIMO (R>1): maps `d_model → 2·d_inner + 2·ngroups·state_rank·R + 3·nheads + num_rope_angles`.
    ///
    /// Output splits: `[z | x | B_raw | C_raw | dd_dt | dd_A | lam_raw | theta_raw]`
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
    ///
    /// For SISO (mimo_rank=1) this has shape `[nheads, 1, state_rank]`.
    pub b_bias_hrn: Param<Tensor<B, 3>>,

    /// Learnable per-head, per-rank bias for C, added after QK-norm.
    /// Shape: `[nheads, mimo_rank, state_rank]`; initialised to ones.
    pub c_bias_hrn: Param<Tensor<B, 3>>,

    /// MIMO up-projection for x (values).
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`.  When SISO, this is `None`.
    pub mimo_x: Option<Param<Tensor<B, 3>>>,

    /// MIMO up-projection for z (gate).
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`.
    pub mimo_z: Option<Param<Tensor<B, 3>>>,

    /// MIMO down-projection for the output.
    /// Shape: `[nheads, mimo_rank, per_head_dim]`.
    /// Only present when `mimo_rank > 1`.
    pub mimo_o: Option<Param<Tensor<B, 3>>>,

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

    /// State rank N.
    pub state_rank: usize,

    /// Number of B/C groups G.  Must divide `nheads`.
    pub ngroups: usize,

    /// Number of RoPE angle pairs (`rope_dim / 2`).
    pub num_rope_angles: usize,

    /// Effective RoPE dimension (= `2 · num_rope_angles`). Always even and
    /// `≤ state_rank`. Only the first `rope_dim` entries of B/C are rotated.
    pub rope_dim: usize,

    /// MIMO rank R.  1 = SISO (standard Mamba-3).
    pub mimo_rank: usize,
}

impl<B: Backend> Mamba3<B> {
    /// `d_inner = expand · d_model`.  Inferred from `out_proj`.
    pub fn d_inner(&self) -> usize {
        let [d_inner, _d_model] = self.out_proj.weight.dims();
        d_inner
    }

    /// `nheads = d_inner / per_head_dim`.  Inferred from `d_h`.
    pub fn nheads(&self) -> usize {
        let [nheads] = self.d_h.dims();
        nheads
    }

    /// `per_head_dim P = d_inner / nheads`.
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
    /// Model (hidden) dimension D.
    pub d_model: usize,

    /// State rank N — the latent dimension of the SSM hidden state.
    /// **Must be even** (required for RoPE pairing).
    #[config(default = 128)]
    pub state_rank: usize,

    /// Expansion factor for `d_inner = expand · d_model`.
    #[config(default = 2)]
    pub expand: usize,

    /// Head dimension P.  `nheads = d_inner / P`.
    #[config(default = 64)]
    pub per_head_dim: usize,

    /// Number of B/C groups G.  Must divide `nheads`.
    #[config(default = 1)]
    pub ngroups: usize,

    /// MIMO rank R.  `1` = standard SISO Mamba-3.
    ///
    /// When `R > 1`, the B/C projections have `R` parallel rank channels.
    /// Three extra weight matrices (`mimo_x`, `mimo_z`, `mimo_o`) provide
    /// element-wise up/down projections in head-space across ranks.
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
    /// Default matches the reference's `rope_fraction` argument in `mamba3.py:36`.
    #[config(default = 0.5)]
    pub rope_fraction: f64,

    /// Whether to apply a gated RMSNorm before the output projection.
    ///
    /// When `true`, the SiLU gate at the end of the block is replaced by a
    /// per-head [`RmsNormGated`] (group size = `per_head_dim`) which both
    /// normalises `y` and gates it with `SiLU(z)`. Matches the reference's
    /// `is_outproj_norm` argument in `mamba3.py:41`.
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
    /// for full RoPE, `state_rank / 2` for `rope_fraction = 0.5`.
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
        let b_bias_hrn = Initializer::Ones.init::<B, 3, _>([nheads, mimo_rank, state_rank], device);
        let c_bias_hrn = Initializer::Ones.init::<B, 3, _>([nheads, mimo_rank, state_rank], device);

        // MIMO projections (only for R > 1)
        let (mimo_x, mimo_z, mimo_o) = if mimo_rank > 1 {
            let per_head_dim = self.per_head_dim;
            // Init: mimo_x and mimo_o to 1/R, mimo_z to 1
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
            b_bias_hrn,
            c_bias_hrn,
            mimo_x,
            mimo_z,
            mimo_o,
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
///   is paired with `n + N/2`. Used by the MIMO Tilelang kernel
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
fn apply_rope_partial<B: Backend, const D: usize>(
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
// MIMO helpers
// ---------------------------------------------------------------------------

/// Build the V (value) tensor for MIMO by expanding x over ranks.
///
/// # Shapes
/// - `x_bShp`:    `[batch, S, nheads, per_head_dim]`
/// - `mimo_x_hrp`: `[nheads, mimo_rank, per_head_dim]`
/// - output:       `[batch, S, mimo_rank, nheads, per_head_dim]`
///
/// When `mimo_x_hrp` is `None` (SISO), wraps `x` in a rank-1 dim.
fn build_v_mimo<B: Backend>(
    x_bshp: Tensor<B, 4>,
    mimo_x_hrp: Option<&Tensor<B, 3>>,
) -> Tensor<B, 5> {
    let [batch, seq, nheads, per_head_dim] = x_bshp.dims();
    match mimo_x_hrp {
        None => {
            // SISO: just add a rank dimension of size 1
            x_bshp.unsqueeze_dim::<5>(2) // [b, s, 1, h, p]
        }
        Some(mimo_x) => {
            let [_, mimo_rank, _] = mimo_x.dims();
            // x_bshp:  [b, s, h, p] → [b, s, 1, h, p]
            let x_exp =
                x_bshp
                    .unsqueeze_dim::<5>(2)
                    .expand([batch, seq, mimo_rank, nheads, per_head_dim]);
            // mimo_x: [h, r, p] → [1, 1, r, h, p]
            let mx_exp = mimo_x
                .clone()
                .permute([1, 0, 2]) // [r, h, p]
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(0)
                .expand([batch, seq, mimo_rank, nheads, per_head_dim]);
            x_exp * mx_exp // [b, s, r, h, p]
        }
    }
}

// ---------------------------------------------------------------------------
// Mamba3::forward  (chunkwise two-SSD — training / prefill)
// ---------------------------------------------------------------------------

impl<B: Backend + Mamba3BackendExt> Mamba3<B> {
    /// Process a full input sequence using the trapezoidal two-SSD algorithm.
    ///
    /// For SISO (mimo_rank=1), this is the standard two-SSD decomposition.
    /// For MIMO (mimo_rank=R>1), B/C have R parallel rank channels. The hidden
    /// state is shared across ranks; each rank contributes independently.
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
        let heads_per_group = nheads / ngroups;
        let mimo_rank = self.mimo_rank;
        let device = input_bsm.device();

        assert!(sequence > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        let mut cache = cache.unwrap_or_else(|| {
            let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
            let k_state_brhn = Tensor::zeros([batch, mimo_rank, nheads, state_rank], &device);
            let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], &device);
            let cum_angle_bhr = Tensor::zeros([batch, nheads, num_rope_angles], &device);
            Mamba3Cache {
                ssm_bhpr,
                k_state_brhn,
                v_state_bhp,
                cum_angle_bhr,
            }
        });

        // ── Step 1: In-projection ─────────────────────────────────────────────
        let proj_bsd = self.in_proj.forward(input_bsm);
        let bc_size = ngroups * state_rank * mimo_rank;

        let mut parts = proj_bsd
            .split_with_sizes(
                vec![
                    d_inner,
                    d_inner,
                    bc_size,
                    bc_size,
                    nheads,
                    nheads,
                    nheads,
                    num_rope_angles,
                ],
                2,
            )
            .into_iter();
        let z_bsi = parts.next().unwrap(); // [B, T, d_inner]
        let x_bsi = parts.next().unwrap(); // [B, T, d_inner]
        let b_raw_bsd = parts.next().unwrap(); // [B, T, ngroups*state_rank*mimo_rank]
        let c_raw_bsd = parts.next().unwrap(); // [B, T, ngroups*state_rank*mimo_rank]
        let dd_dt_bsh = parts.next().unwrap(); // [B, T, nheads]
        let dd_A_raw_bsh = parts.next().unwrap(); // [B, T, nheads]
        let lam_raw_bsh = parts.next().unwrap(); // [B, T, nheads]
        let theta_bsa = parts.next().unwrap(); // [B, T, num_rope_angles]

        san(&z_bsi);
        san(&x_bsi);
        san(&dd_dt_bsh);

        // ── Step 2: Discretisation + trapezoidal coefficients ─────────────────
        let dt_bias_11h = self.dt_bias_h.val().unsqueeze_dims(&[0, 1]);
        let dt_bsh = softplus(dd_dt_bsh + dt_bias_11h).clamp(self.dt_limit.0, self.dt_limit.1);

        let a_bsh = -softplus(dd_A_raw_bsh).clamp(f64::NEG_INFINITY, -self.a_floor);
        let da_bsh = dt_bsh.clone() * a_bsh;

        let alpha_bsh = da_bsh.clone().exp();
        let lam_bsh = sigmoid(lam_raw_bsh);
        let gamma_bsh = lam_bsh.clone() * dt_bsh.clone();
        let beta_bsh = (-lam_bsh.clone() + 1.0) * dt_bsh.clone() * alpha_bsh.clone();

        san(&dt_bsh);
        san(&da_bsh);
        san(&gamma_bsh);
        san(&beta_bsh);

        // ── Step 3: Reshape x ─────────────────────────────────────────────────
        let x_bshp = x_bsi.reshape([batch, sequence, nheads, per_head_dim]);

        // ── Step 4: QK-Norm on B and C → [b, T, R, H, N] ─────────────────────
        // Reshape: [b, T, R*G*N] → [b, T, R, G, N]
        // QK-Norm over N, then expand G→H, then add per-head+rank bias [H, R, N].
        let b_bsrhr = {
            let b_bsrgr = b_raw_bsd.reshape([batch, sequence, mimo_rank, ngroups, state_rank]);
            // Norm over last dim (state_rank) for each (b, s, r, g) slice:
            // Flatten leading dims so RmsNorm operates on last dim only.
            let b_norm = self
                .b_norm
                .forward(b_bsrgr.reshape([batch * sequence * mimo_rank, ngroups, state_rank]))
                .reshape([batch, sequence, mimo_rank, ngroups, state_rank]);
            // Expand groups → heads: [b, T, R, G, N] → [b, T, R, G, H/G, N] → [b, T, R, H, N]
            let b_exp = b_norm
                .unsqueeze_dim::<6>(4) // [b, T, R, G, 1, N]
                .expand([
                    batch,
                    sequence,
                    mimo_rank,
                    ngroups,
                    heads_per_group,
                    state_rank,
                ])
                .reshape([batch, sequence, mimo_rank, nheads, state_rank]);
            // Add bias [H, R, N] → broadcast as [1, 1, R, H, N]
            // b_bias_hrn: [H, R, N] → permute to [R, H, N] → unsqueeze → [1, 1, R, H, N]
            let bias = self
                .b_bias_hrn
                .val()
                .permute([1, 0, 2]) // [R, H, N]
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(0); // [1, 1, R, H, N]
            b_exp + bias
        };
        let c_bsrhr = {
            let c_bsrgr = c_raw_bsd.reshape([batch, sequence, mimo_rank, ngroups, state_rank]);
            let c_norm = self
                .c_norm
                .forward(c_bsrgr.reshape([batch * sequence * mimo_rank, ngroups, state_rank]))
                .reshape([batch, sequence, mimo_rank, ngroups, state_rank]);
            let c_exp = c_norm
                .unsqueeze_dim::<6>(4)
                .expand([
                    batch,
                    sequence,
                    mimo_rank,
                    ngroups,
                    heads_per_group,
                    state_rank,
                ])
                .reshape([batch, sequence, mimo_rank, nheads, state_rank]);
            let bias = self
                .c_bias_hrn
                .val()
                .permute([1, 0, 2])
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(0);
            c_exp + bias
        };
        // b_bsrhr: [b, T, R, H, N]
        assert_eq!(
            [batch, sequence, mimo_rank, nheads, state_rank],
            b_bsrhr.dims()
        );
        assert_eq!(
            [batch, sequence, mimo_rank, nheads, state_rank],
            c_bsrhr.dims()
        );

        // ── Step 5: Data-dependent cumulative RoPE angles ─────────────────────
        let theta_scaled_bsa = theta_bsa.tanh() * std::f32::consts::PI;
        let raw_angles_bsha =
            dt_bsh.clone().unsqueeze_dim::<4>(3) * theta_scaled_bsa.unsqueeze_dim::<4>(2);

        let cumsum_bsha = raw_angles_bsha.cumsum(1);
        let cum_angles_bsha = cache.cum_angle_bhr.clone().unsqueeze_dim::<4>(1) + cumsum_bsha;
        assert_eq!(
            [batch, sequence, nheads, num_rope_angles],
            cum_angles_bsha.dims()
        );
        san(&cum_angles_bsha);

        // Apply RoPE to B and C: angles broadcast over the R dim.
        // b_bsrhr: [b, T, R, H, N], angles: [b, T, H, A] → [b, T, 1, H, A]
        let angles_exp_bsrha = cum_angles_bsha.clone().unsqueeze_dim::<5>(2).expand([
            batch,
            sequence,
            mimo_rank,
            nheads,
            num_rope_angles,
        ]);
        // SISO uses interleaved (pairwise) pairing; MIMO uses half-and-half.
        // Partial RoPE: rotate only the first `rope_dim` entries of B/C.
        let rotate_pairwise = mimo_rank == 1;
        let rope_dim = self.rope_dim;
        let b_bsrhn = apply_rope_partial::<B, 5>(
            b_bsrhr,
            angles_exp_bsrha.clone(),
            rope_dim,
            rotate_pairwise,
        );
        let c_bsrhn =
            apply_rope_partial::<B, 5>(c_bsrhr, angles_exp_bsrha, rope_dim, rotate_pairwise);
        san(&b_bsrhn);
        san(&c_bsrhn);

        // ── Step 6: Build shifted inputs for β term ───────────────────────────
        //
        // "Shift-Before-Chunking": prepend the cached x_{-1} / B_{-1} at the
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
        // b_prev: [b, T, R, H, N]
        let b_prev_first_b1rhn = cache.k_state_brhn.clone().unsqueeze_dim::<5>(1);
        let b_prev_bsrhn = if sequence == 1 {
            b_prev_first_b1rhn
        } else {
            Tensor::cat(
                vec![
                    b_prev_first_b1rhn,
                    b_bsrhn.clone().narrow(1, 0, sequence - 1),
                ],
                1,
            )
        };

        // ── Step 7: Scale inputs by trapezoidal coefficients ──────────────────
        // gamma and beta are per-head scalars, broadcast over R and P:
        let gamma_bsh1 = gamma_bsh.unsqueeze_dim::<4>(3);
        let beta_bsh1 = beta_bsh.unsqueeze_dim::<4>(3);
        let x_gamma_bshp = x_bshp.clone() * gamma_bsh1; // γ_t · x_t
        let x_beta_bshp = x_prev_bshp * beta_bsh1; // β_t · x_{t-1}

        // ── Save last-token B for cache ───────────────────────────────────────
        let b_last_brhn = b_bsrhn
            .clone()
            .narrow(1, sequence - 1, 1)
            .reshape([batch, mimo_rank, nheads, state_rank]);

        // ── Step 8: Pad sequence to multiple of chunk_len ─────────────────────
        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        let (x_gamma_bShp, x_beta_bShp, da_bSh, b_bSrhn, b_prev_bSrhn, c_bSrhn) = if pad == 0 {
            (
                x_gamma_bshp,
                x_beta_bshp,
                da_bsh,
                b_bsrhn,
                b_prev_bsrhn,
                c_bsrhn,
            )
        } else {
            let pad_hp = Tensor::zeros([batch, pad, nheads, per_head_dim], &device);
            let pad_h = Tensor::zeros([batch, pad, nheads], &device);
            let pad_rhn = Tensor::zeros([batch, pad, mimo_rank, nheads, state_rank], &device);
            (
                Tensor::cat(vec![x_gamma_bshp, pad_hp.clone()], 1),
                Tensor::cat(vec![x_beta_bshp, pad_hp], 1),
                Tensor::cat(vec![da_bsh, pad_h], 1),
                Tensor::cat(vec![b_bsrhn, pad_rhn.clone()], 1),
                Tensor::cat(vec![b_prev_bsrhn, pad_rhn.clone()], 1),
                Tensor::cat(vec![c_bsrhn, pad_rhn], 1),
            )
        };

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let x_gamma_bnlhp = x_gamma_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let x_beta_bnlhp = x_beta_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let da_bnlh = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        // [b, S, R, H, N] → [b, n, l, R, H, N]
        let b_bnlrhn = b_bSrhn.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let b_prev_bnlrhn =
            b_prev_bSrhn.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);
        let c_bnlrhn = c_bSrhn.reshape([batch, nchunks, chunk_len, mimo_rank, nheads, state_rank]);

        // ── Step 9: Two MIMO-SSD calls ────────────────────────────────────────
        // Build V tensors: [b, n, l, R, H, P]
        let mimo_x_val = self.mimo_x.as_ref().map(|p| p.val());
        let v_gamma_bnlrhp = build_v_mimo_chunked(
            x_gamma_bnlhp.clone(),
            mimo_x_val.as_ref(),
            batch,
            nchunks,
            chunk_len,
            mimo_rank,
            nheads,
            per_head_dim,
        );
        let v_beta_bnlrhp = build_v_mimo_chunked(
            x_beta_bnlhp,
            mimo_x_val.as_ref(),
            batch,
            nchunks,
            chunk_len,
            mimo_rank,
            nheads,
            per_head_dim,
        );

        let input_gamma = Mamba3SsdInput {
            v_bnlrhp: v_gamma_bnlrhp,
            da_bnlh: da_bnlh.clone(),
            b_bnlrhn: b_bnlrhn.clone(),
            c_bnlrhn: c_bnlrhn.clone(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        };
        let (y_gamma_bnlrhp, final_state_gamma) = ssd_path.clone().run(input_gamma);

        let input_beta = Mamba3SsdInput {
            v_bnlrhp: v_beta_bnlrhp,
            da_bnlh,
            b_bnlrhn: b_prev_bnlrhn,
            c_bnlrhn,
            initial_state_bhpr: Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device),
            init_state_hpr: None,
        };
        let (y_beta_bnlrhp, final_state_beta) = ssd_path.run(input_beta);

        // y_bnlrhp: [b, n, l, R, H, P]
        let y_bnlrhp = y_gamma_bnlrhp + y_beta_bnlrhp;
        let final_state_bhpr = final_state_gamma + final_state_beta;

        san(&y_bnlrhp);
        san(&final_state_bhpr);

        cache.ssm_bhpr = final_state_bhpr;

        // ── Step 10: Unpad ────────────────────────────────────────────────────
        let y_bSrhp = y_bnlrhp.reshape([batch, sequence_padded, mimo_rank, nheads, per_head_dim]);
        let y_bsrhp = if pad == 0 {
            y_bSrhp
        } else {
            y_bSrhp.narrow(1, 0, sequence)
        };

        // ── Step 11: D skip + gate + aggregate ranks ──────────────────────────
        // D skip uses raw x * mimo_x (not gamma-scaled)
        let v_raw_bsrhp = build_v_mimo::<B>(x_bshp.clone(), mimo_x_val.as_ref());

        let d_11_h1 = self.d_h.val().unsqueeze_dims::<5>(&[0, 1, 2, 4]); // [1, 1, 1, H, 1]
        let y_bsrhp = y_bsrhp + d_11_h1 * v_raw_bsrhp.clone();

        // ── Gate (or gated norm) and rank aggregation ─────────────────────────
        // When `out_norm` is set, the SiLU gate is replaced by a per-head
        // gated RMSNorm: `RmsNormGated(y, z) = norm(y) * silu(z)`.
        let y_bsi = if mimo_rank > 1 {
            let mimo_z_val = self.mimo_z.as_ref().map(|p| p.val()).unwrap();
            let mimo_o_val = self.mimo_o.as_ref().map(|p| p.val()).unwrap();

            // z_r = z * mimo_z[r]: [b, s, h, p] * [h, r, p] → [b, s, r, h, p]
            let z_bshp = z_bsi
                .clone()
                .reshape([batch, sequence, nheads, per_head_dim]);
            let z_bsrhp = {
                // z: [b, s, h, p] → [b, s, 1, h, p]
                // mimo_z: [h, r, p] → [r, h, p] → [1, 1, r, h, p]
                let z_exp = z_bshp.unsqueeze_dim::<5>(2).expand([
                    batch,
                    sequence,
                    mimo_rank,
                    nheads,
                    per_head_dim,
                ]);
                let mz = mimo_z_val
                    .permute([1, 0, 2]) // [r, h, p]
                    .unsqueeze_dim::<4>(0)
                    .unsqueeze_dim::<5>(0)
                    .expand([batch, sequence, mimo_rank, nheads, per_head_dim]);
                z_exp * mz
            };

            // Per-rank gate or gated norm:
            //   without out_norm: y_r * silu(z_r)
            //   with    out_norm: norm(y_r) * silu(z_r)  (norm over per_head_dim)
            let y_combined_bsrhp = match &self.out_norm {
                Some(norm) => norm.forward(y_bsrhp, z_bsrhp),
                None => y_bsrhp * Silu::new().forward(z_bsrhp),
            };

            // Down-project with mimo_o: out = sum_r mimo_o[h, r, p] * y_r
            // mimo_o: [h, r, p] → [r, h, p] → [1, 1, r, h, p]
            let mo = mimo_o_val
                .permute([1, 0, 2]) // [r, h, p]
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(0)
                .expand([batch, sequence, mimo_rank, nheads, per_head_dim]);
            // sum over rank dim (dim=2): [b, s, r, h, p] → [b, s, h, p]
            let y_bhp: Tensor<B, 4> = (y_combined_bsrhp * mo).sum_dim(2).squeeze_dim(2);
            y_bhp.reshape([batch, sequence, d_inner])
        } else {
            // SISO: squeeze rank dim, apply gate (or gated norm) over per_head_dim.
            let y_bshp: Tensor<B, 4> = y_bsrhp.squeeze_dim(2); // [b, s, h, p]
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
        // k_state = B at last token: [b, R, H, N]
        cache.k_state_brhn = b_last_brhn;

        // v_state = x at last token: [b, H, P]
        cache.v_state_bhp =
            x_bshp
                .narrow(1, sequence - 1, 1)
                .reshape([batch, nheads, per_head_dim]);

        // cum_angle at last token
        cache.cum_angle_bhr =
            cum_angles_bsha
                .narrow(1, sequence - 1, 1)
                .reshape([batch, nheads, num_rope_angles]);

        (out_bsm, cache)
    }
}

fn build_v_mimo_chunked<B: Backend>(
    x_bnlhp: Tensor<B, 5>,
    mimo_x: Option<&Tensor<B, 3>>,
    batch: usize,
    nchunks: usize,
    chunk_len: usize,
    mimo_rank: usize,
    nheads: usize,
    per_head_dim: usize,
) -> Tensor<B, 6> {
    match mimo_x {
        None => x_bnlhp.unsqueeze_dim::<6>(3),
        Some(mx) => {
            let x_exp = x_bnlhp.unsqueeze_dim::<6>(3).expand([
                batch,
                nchunks,
                chunk_len,
                mimo_rank,
                nheads,
                per_head_dim,
            ]);
            let mx_exp = mx
                .clone()
                .permute([1, 0, 2])
                .unsqueeze_dim::<4>(0)
                .unsqueeze_dim::<5>(0)
                .unsqueeze_dim::<6>(0)
                .expand([batch, nchunks, chunk_len, mimo_rank, nheads, per_head_dim]);
            x_exp * mx_exp
        }
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
        ///   hₜ = αₜ hₜ₋₁ + βₜ B_{t-1} ⊗ x_{t-1} + γₜ Bₜ ⊗ xₜ
        ///   yₜ = Cₜᵀ hₜ + D xₜ
        /// ```
        ///
        /// For MIMO (mimo_rank=R>1):
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + Σ_r βₜ B_{t-1}[r] ⊗ (x_{t-1} ⊙ mimo_x[r])
        ///                  + Σ_r γₜ Bₜ[r] ⊗ (xₜ ⊙ mimo_x[r])
        ///   yₜ[r] = Cₜ[r]ᵀ hₜ + D xₜ ⊙ mimo_x[r]
        ///   outₜ = Σ_r mimo_o[r] ⊙ silu(zₜ ⊙ mimo_z[r]) ⊙ yₜ[r]
        /// ```
        ///
        /// # Shapes
        /// - `input_bm` : `[batch, d_model]`
        /// - output     : `[batch, d_model]`
        #[allow(non_snake_case)]
        pub fn step(
            &self,
            input_bm: Tensor<B, 2>,
            cache: Option<Mamba3Cache<B>>,
        ) -> (Tensor<B, 2>, Mamba3Cache<B>) {
            let [batch, d_model] = input_bm.dims();
            let d_inner = self.d_inner();
            let nheads = self.nheads();
            let ngroups = self.ngroups;
            let per_head_dim = self.per_head_dim();
            let state_rank = self.state_rank;
            let num_rope_angles = self.num_rope_angles;
            let heads_per_group = nheads / ngroups;
            let mimo_rank = self.mimo_rank;
            let device = &input_bm.device();
            let ssm_shape = [batch, nheads, per_head_dim, state_rank];

            assert_eq!(nheads % ngroups, 0);

            let mut cache = cache.unwrap_or_else(|| {
                let ssm_bhpr = Tensor::zeros(ssm_shape, device);
                let k_state_brhn = Tensor::zeros([batch, mimo_rank, nheads, state_rank], device);
                let v_state_bhp = Tensor::zeros([batch, nheads, per_head_dim], device);
                let cum_angle_bhr = Tensor::zeros([batch, nheads, num_rope_angles], device);
                Mamba3Cache {
                    ssm_bhpr,
                    k_state_brhn,
                    v_state_bhp,
                    cum_angle_bhr,
                }
            });

            // ── In-projection ─────────────────────────────────────────────────
            let proj_bd = self.in_proj.forward(input_bm);
            let bc_size = ngroups * state_rank * mimo_rank;
            let mut parts = proj_bd
                .split_with_sizes(
                    vec![
                        d_inner,
                        d_inner,
                        bc_size,
                        bc_size,
                        nheads,
                        nheads,
                        nheads,
                        num_rope_angles,
                    ],
                    1,
                )
                .into_iter();
            let z_bi = parts.next().unwrap(); // [B, d_inner]
            let x_bi = parts.next().unwrap(); // [B, d_inner]
            let b_raw_bd = parts.next().unwrap(); // [B, ngroups*state_rank*mimo_rank]
            let c_raw_bd = parts.next().unwrap();
            let dd_dt_bh = parts.next().unwrap(); // [B, nheads]
            let dd_A_raw_bh = parts.next().unwrap();
            let lam_raw_bh = parts.next().unwrap();
            let theta_ba = parts.next().unwrap(); // [B, num_rope_angles]

            // ── Reshape x ─────────────────────────────────────────────────────
            let x_bhp = x_bi.reshape([batch, nheads, per_head_dim]);

            // ── Discretisation ─────────────────────────────────────────────────
            let dt_bias_1h = self.dt_bias_h.val().unsqueeze_dim(0);
            let dt_bh = softplus(dd_dt_bh + dt_bias_1h).clamp(self.dt_limit.0, self.dt_limit.1);
            let a_bh = -softplus(dd_A_raw_bh).clamp(f64::NEG_INFINITY, -self.a_floor);
            let da_bh = dt_bh.clone() * a_bh;
            let alpha_bh = da_bh.exp();
            let lam_bh = sigmoid(lam_raw_bh);
            let gamma_bh = lam_bh.clone() * dt_bh.clone();
            let beta_bh = (-lam_bh.clone() + 1.0) * dt_bh.clone() * alpha_bh.clone();

            // ── QK-Norm on B and C → [B, R, H, N] ────────────────────────────
            // b_raw: [B, R*G*N] → [B, R, G, N] → norm → expand → [B, R, H, N] → add bias
            let b_brhn = {
                let b_brgn = b_raw_bd.reshape([batch, mimo_rank, ngroups, state_rank]);
                let b_norm = self
                    .b_norm
                    .forward(b_brgn.reshape([batch * mimo_rank, ngroups, state_rank]))
                    .reshape([batch, mimo_rank, ngroups, state_rank]);
                let b_exp = b_norm
                    .unsqueeze_dim::<5>(3) // [B, R, G, 1, N]
                    .expand([batch, mimo_rank, ngroups, heads_per_group, state_rank])
                    .reshape([batch, mimo_rank, nheads, state_rank]);
                // bias: [H, R, N] → [R, H, N] → [1, R, H, N]
                let bias = self
                    .b_bias_hrn
                    .val()
                    .permute([1, 0, 2])
                    .unsqueeze_dim::<4>(0);
                b_exp + bias
            };
            let c_brhn = {
                let c_brgn = c_raw_bd.reshape([batch, mimo_rank, ngroups, state_rank]);
                let c_norm = self
                    .c_norm
                    .forward(c_brgn.reshape([batch * mimo_rank, ngroups, state_rank]))
                    .reshape([batch, mimo_rank, ngroups, state_rank]);
                let c_exp = c_norm
                    .unsqueeze_dim::<5>(3)
                    .expand([batch, mimo_rank, ngroups, heads_per_group, state_rank])
                    .reshape([batch, mimo_rank, nheads, state_rank]);
                let bias = self
                    .c_bias_hrn
                    .val()
                    .permute([1, 0, 2])
                    .unsqueeze_dim::<4>(0);
                c_exp + bias
            };
            assert_eq!([batch, mimo_rank, nheads, state_rank], b_brhn.dims());

            // ── RoPE: update cumulative angle, rotate B and C ──────────────────
            let theta_scaled_ba = theta_ba.tanh() * std::f32::consts::PI;
            let raw_angle_bha = dt_bh.unsqueeze_dim::<3>(2) * theta_scaled_ba.unsqueeze_dim::<3>(1);
            let new_cum_angle_bha = cache.cum_angle_bhr.clone() + raw_angle_bha;

            // Broadcast angles over R: [b, H, A] → [b, 1, H, A] → [b, R, H, A]
            let angles_brha = new_cum_angle_bha.clone().unsqueeze_dim::<4>(1).expand([
                batch,
                mimo_rank,
                nheads,
                num_rope_angles,
            ]);
            // SISO uses interleaved (pairwise) pairing; MIMO uses half-and-half.
            // Partial RoPE: rotate only the first `rope_dim` entries of B/C.
            let rotate_pairwise = mimo_rank == 1;
            let rope_dim = self.rope_dim;
            let b_brhn =
                apply_rope_partial::<B, 4>(b_brhn, angles_brha.clone(), rope_dim, rotate_pairwise);
            let c_brhn = apply_rope_partial::<B, 4>(c_brhn, angles_brha, rope_dim, rotate_pairwise);

            // ── Build MIMO value tensors ───────────────────────────────────────
            // x_vals[b, r, h, p] = x[b, h, p] * mimo_x[h, r, p]
            // xs_vals[b, r, h, p] = x_state[b, h, p] * mimo_x[h, r, p]
            let mimo_x_val = self.mimo_x.as_ref().map(|p| p.val());
            let (x_vals_brhp, xs_vals_brhp) = build_mimo_vals(
                x_bhp.clone(),
                cache.v_state_bhp.clone(),
                mimo_x_val.as_ref(),
                batch,
                mimo_rank,
                nheads,
                per_head_dim,
                device,
            );

            // ── SSM state update ───────────────────────────────────────────────
            // new_state[b, h, p, n] = alpha * state
            //   + sum_r gamma * x_vals[r] ⊗ B_cur[r]
            //   + sum_r beta  * xs_vals[r] ⊗ B_state[r]
            //
            // For the outer product sum:
            //   xBt[b, h, p, n] = sum_r coeff[r, h, p] * B[r, h, n]
            //   = einsum('brhp,brhn->bhpn', coeff*x_vals, B)
            //   = matmul over r: [b, h, p, r] @ [b, h, r, n]
            // x_vals_brhp * gamma_b1h1 (broadcast: [b, r, h, p] * [b, 1, h, 1]):
            // Need gamma as [b, 1, h, 1] to broadcast over r and p:
            let gamma_b1h1 = gamma_bh.clone().unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(3); // [b, 1, h, 1]
            let beta_b1h1 = beta_bh.clone().unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(3);

            let x_gamma_brhp = x_vals_brhp.clone() * gamma_b1h1; // [b, r, h, p]
            let x_beta_brhp = xs_vals_brhp * beta_b1h1; // [b, r, h, p]

            // einsum('brhp,brhn->bhpn', x_gamma, B_cur):
            // [b, r, h, p] → permute to [b, h, p, r]
            // [b, r, h, n] → permute to [b, h, r, n]
            // matmul: [b, h, p, r] @ [b, h, r, n] = [b, h, p, n]
            let xBt_state = {
                let b_bhrn = b_brhn.clone().permute([0, 2, 1, 3]); // [b, h, r, n]
                let xg_bhpr = x_gamma_brhp.permute([0, 2, 3, 1]); // [b, h, p, r]
                xg_bhpr.matmul(b_bhrn) // [b, h, p, n]
            };
            let xBt_prev = {
                let b_state_bhrn = cache.k_state_brhn.clone().permute([0, 2, 1, 3]); // [b, h, r, n]
                let xb_bhpr = x_beta_brhp.permute([0, 2, 3, 1]); // [b, h, p, r]
                xb_bhpr.matmul(b_state_bhrn) // [b, h, p, n]
            };

            let alpha_bh11 = alpha_bh.unsqueeze_dims::<4>(&[2, 3]);
            let new_state_bhpn = alpha_bh11 * cache.ssm_bhpr.clone() + xBt_state + xBt_prev;

            // ── Output ────────────────────────────────────────────────────────
            // out_r[b, r, h, p] = sum_n C[b, r, h, n] * state[b, h, p, n] + D * x_vals[b, r, h, p]
            // = einsum('bhpn,brhn->brhp', state, C)
            let out_r_brhp = {
                // state: [b, h, p, n], C: [b, r, h, n]
                // For each (b, h): [p, n] @ [r, n]^T = [p, r]
                // state: [b, h, p, n] → [b, h, p, n]
                // C_bhrn: [b, h, r, n] = b_brhn permuted = c_brhn permuted
                let c_bhrn = c_brhn.permute([0, 2, 1, 3]); // [b, h, r, n]
                // [b, h, p, n] @ [b, h, n, r] = [b, h, p, r]
                let c_bhnr = c_bhrn.permute([0, 1, 3, 2]); // [b, h, n, r]
                let out_bhpr = new_state_bhpn.clone().matmul(c_bhnr); // [b, h, p, r]
                out_bhpr.permute([0, 3, 1, 2]) // [b, r, h, p]
            };

            // D skip: D[h] * x_vals[b, R, H, P], broadcast [1, 1, H, 1] over [b, R, H, P]
            let d_skip = self
                .d_h
                .val()
                .unsqueeze_dims::<4>(&[0, 1, 3]) // [1, 1, h, 1]
                .expand([batch, mimo_rank, nheads, per_head_dim]);
            let out_r_brhp = out_r_brhp + d_skip * x_vals_brhp;

            // ── Gate (or gated norm) and rank aggregation ─────────────────────
            // When `out_norm` is set, the SiLU gate is replaced by a per-head
            // gated RMSNorm: `RmsNormGated(y, z) = norm(y) * silu(z)`.
            let z_bhp = z_bi.reshape([batch, nheads, per_head_dim]);
            let y_bi = if mimo_rank > 1 {
                let mimo_z_val = self.mimo_z.as_ref().map(|p| p.val()).unwrap();
                let mimo_o_val = self.mimo_o.as_ref().map(|p| p.val()).unwrap();

                // z_r = z * mimo_z[r]: z[b, h, p] * mimo_z[h, r, p] → [b, r, h, p]
                let z_exp =
                    z_bhp
                        .unsqueeze_dim::<4>(1)
                        .expand([batch, mimo_rank, nheads, per_head_dim]);
                // mimo_z: [h, r, p] → [r, h, p] → [1, r, h, p]
                let mz = mimo_z_val.permute([1, 0, 2]).unsqueeze_dim::<4>(0).expand([
                    batch,
                    mimo_rank,
                    nheads,
                    per_head_dim,
                ]);
                let z_r = z_exp * mz;

                // Per-rank gate or gated norm.
                let combined = match &self.out_norm {
                    Some(norm) => norm.forward(out_r_brhp, z_r),
                    None => out_r_brhp * Silu::new().forward(z_r),
                };

                // Project down: out = sum_r mimo_o[r] * combined[r]
                // mimo_o: [h, r, p] → [r, h, p] → [1, r, h, p]
                let mo = mimo_o_val.permute([1, 0, 2]).unsqueeze_dim::<4>(0).expand([
                    batch,
                    mimo_rank,
                    nheads,
                    per_head_dim,
                ]);
                let out_bhp: Tensor<B, 3> = (combined * mo).sum_dim(1).squeeze_dim(1);
                out_bhp.reshape([batch, d_inner])
            } else {
                // SISO: squeeze rank dim, gate (or gated norm) over per_head_dim.
                let y_bhp: Tensor<B, 3> = out_r_brhp.squeeze_dim(1); // [b, h, p]
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
            cache.ssm_bhpr = new_state_bhpn;
            // k_state: B at current token [b, R, H, N]
            cache.k_state_brhn = b_brhn; // already [b, r, h, n]
            cache.v_state_bhp = x_bhp;
            cache.cum_angle_bhr = new_cum_angle_bha;

            (out_bm, cache)
        }
    }

    /// Build MIMO value tensors for x_current and x_state.
    ///
    /// Returns `(x_vals_brhp, xs_vals_brhp)` both of shape `[batch, mimo_rank, nheads, per_head_dim]`.
    fn build_mimo_vals<B: Backend>(
        x_bhp: Tensor<B, 3>,
        x_state_bhp: Tensor<B, 3>,
        mimo_x: Option<&Tensor<B, 3>>,
        batch: usize,
        mimo_rank: usize,
        nheads: usize,
        per_head_dim: usize,
        _device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        match mimo_x {
            None => {
                // SISO: add rank dim of 1
                (
                    x_bhp.unsqueeze_dim::<4>(1),
                    x_state_bhp.unsqueeze_dim::<4>(1),
                )
            }
            Some(mx) => {
                // x: [b, h, p] → [b, 1, h, p] → [b, R, h, p]
                let x_exp =
                    x_bhp
                        .unsqueeze_dim::<4>(1)
                        .expand([batch, mimo_rank, nheads, per_head_dim]);
                let xs_exp = x_state_bhp.unsqueeze_dim::<4>(1).expand([
                    batch,
                    mimo_rank,
                    nheads,
                    per_head_dim,
                ]);
                // mimo_x: [h, r, p] → [r, h, p] → [1, r, h, p]
                let mx_exp = mx.clone().permute([1, 0, 2]).unsqueeze_dim::<4>(0).expand([
                    batch,
                    mimo_rank,
                    nheads,
                    per_head_dim,
                ]);
                (x_exp * mx_exp.clone(), xs_exp * mx_exp)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
    use burn::backend::Flex;

    type B = Flex;

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

    fn run_step_matches_forward(cfg: Mamba3Config) {
        let device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 5;
        let d_model = cfg.d_model;

        let input = Tensor::<B, 3>::random(
            [batch, seq_len, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let ssd_path = Mamba3SsdPath::Minimal(Some(4));
        let (out_fwd, _) = model.forward(input.clone(), None, ssd_path);

        let mut cache: Option<Mamba3Cache<B>> = None;
        let mut step_outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let token = input.clone().narrow(1, t, 1).squeeze_dim(1);
            let (out_t, new_cache) = model.step(token, cache);
            cache = Some(new_cache);
            step_outputs.push(out_t);
        }

        let out_step = Tensor::stack(step_outputs, 1);

        let diff = (out_fwd - out_step).abs().max().into_scalar();
        assert!(
            diff < 1e-4,
            "step() vs forward() max absolute difference = {diff:.6} (expected < 1e-4)"
        );
    }

    #[test]
    fn step_matches_forward() {
        run_step_matches_forward(small_config());
    }

    #[test]
    fn step_matches_forward_ngroups2() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_ngroups(2);
        run_step_matches_forward(cfg);
    }

    #[test]
    fn step_matches_forward_mimo() {
        run_step_matches_forward(small_config_mimo());
    }

    #[test]
    fn step_matches_forward_mimo_ngroups2() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_ngroups(2)
            .with_mimo_rank(2);
        run_step_matches_forward(cfg);
    }

    /// forward(full) ≡ forward(prefix) then forward(suffix, cache_from_prefix).
    ///
    /// Verifies stateful chunked-prefill: the β term at the start of the second
    /// chunk must see `x_{-1}` and `B_{-1}` from the cache, not zeros.
    fn run_split_matches_full(cfg: Mamba3Config) {
        let device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 6;
        let split = 2;
        let d_model = cfg.d_model;

        let input = Tensor::<B, 3>::random(
            [batch, seq_len, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let ssd_path = Mamba3SsdPath::Minimal(Some(4));
        let (out_full, _) = model.forward(input.clone(), None, ssd_path.clone());

        let prefix = input.clone().narrow(1, 0, split);
        let suffix = input.narrow(1, split, seq_len - split);
        let (out_prefix, cache) = model.forward(prefix, None, ssd_path.clone());
        let (out_suffix, _) = model.forward(suffix, Some(cache), ssd_path);
        let out_split = Tensor::cat(vec![out_prefix, out_suffix], 1);

        let diff = (out_full - out_split).abs().max().into_scalar();
        assert!(
            diff < 1e-4,
            "split forward vs full forward max absolute difference = {diff:.6} (expected < 1e-4)"
        );
    }

    #[test]
    fn split_matches_full() {
        run_split_matches_full(small_config());
    }

    #[test]
    fn split_matches_full_ngroups2() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_ngroups(2);
        run_split_matches_full(cfg);
    }

    #[test]
    fn split_matches_full_mimo() {
        run_split_matches_full(small_config_mimo());
    }

    #[test]
    fn split_matches_full_mimo_ngroups2() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_ngroups(2)
            .with_mimo_rank(2);
        run_split_matches_full(cfg);
    }

    // ── rope_fraction = 0.5 (partial RoPE) ──────────────────────────────────

    #[test]
    fn step_matches_forward_rope_half() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_rope_fraction(0.5);
        run_step_matches_forward(cfg);
    }

    #[test]
    fn step_matches_forward_rope_half_mimo() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_mimo_rank(2)
            .with_rope_fraction(0.5);
        run_step_matches_forward(cfg);
    }

    #[test]
    fn split_matches_full_rope_half() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_rope_fraction(0.5);
        run_split_matches_full(cfg);
    }

    // ── has_outproj_norm = true (gated RMSNorm) ─────────────────────────────

    #[test]
    fn step_matches_forward_outproj_norm() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_has_outproj_norm(true);
        run_step_matches_forward(cfg);
    }

    #[test]
    fn step_matches_forward_outproj_norm_mimo() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_mimo_rank(2)
            .with_has_outproj_norm(true);
        run_step_matches_forward(cfg);
    }

    #[test]
    fn split_matches_full_outproj_norm() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_has_outproj_norm(true);
        run_split_matches_full(cfg);
    }

    // ── Both features combined ──────────────────────────────────────────────

    #[test]
    fn step_matches_forward_rope_half_outproj_norm_mimo() {
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_mimo_rank(2)
            .with_rope_fraction(0.5)
            .with_has_outproj_norm(true);
        run_step_matches_forward(cfg);
    }
}
