//! # Mamba-3 SSM Block — Exponential-Trapezoidal SSD with Data-Dependent RoPE
//!
//! This module implements the core **Mamba-3 layer** from the paper
//! *"The Mamba-3 Framework: Structured State Spaces with Trapezoidal
//! Discretization and Data-Dependent Rotary Embeddings"*.
//!
//! ## The Mamba-3 Recurrence (SISO, Proposition 1)
//!
//! ```text
//!   hₜ = αₜ hₜ₋₁ + βₜ B_{t-1} x_{t-1} + γₜ Bₜ xₜ   (state update)
//!   yₜ = Cₜᵀ hₜ + D xₜ                                (output)
//! ```
//!
//! where:
//! - `αₜ = exp(Δₜ A)`                       — decay (same as Mamba-2)
//! - `βₜ = (1 − λₜ) · Δₜ · exp(Δₜ A)`      — left-endpoint coefficient
//! - `γₜ = λₜ · Δₜ`                         — right-endpoint coefficient
//! - `λₜ = σ(u_λ,t)`                        — data-dependent interpolation
//!
//! ## Key differences from Mamba-2
//!
//! | Aspect | Mamba-2 | Mamba-3 |
//! |--------|---------|---------|
//! | Recurrence | 2-term | 3-term (trapezoidal) |
//! | λ parameter | absent | per-head, data-dependent |
//! | Short conv | present | **removed** |
//! | B/C norm | post-SSD gated RMSNorm | QK-Norm before SSD |
//! | B/C bias | none | learnable, init=1 |
//! | Data-dep RoPE | none | per-head, angles from input |
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
//! | `n`    | nchunks = T/Q  | varies  |
//! | `l`    | chunk_len Q    | 64–256  |
//! | `a`    | num_rope_angles = state_rank / 2 | varies |

use crate::mamba3::prelude::*;
use crate::mamba3::ssd::SsdInput;
use crate::utils::sanity::sanity as san;
use crate::utils::{
    rms_norm::{RmsNorm, RmsNormConfig},
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
/// and data-dependent RoPE.  Supports two execution modes:
///
/// - [`Self::forward`] — chunkwise two-SSD algorithm for training / prefill
/// - [`Self::step`]    — recurrent form for token-by-token decoding
#[derive(Module, Debug)]
pub struct Mamba3<B: Backend> {
    /// Input projection: maps `d_model → 2·d_inner + 2·ngroups·state_rank + 3·nheads + num_rope_angles`.
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

    /// Learnable per-head bias for B, added after QK-norm.
    /// Shape: `[nheads, state_rank]`; initialised to ones.
    ///
    /// This bias (combined with the trapezoidal discretization) provides
    /// convolution-like behavior, removing the need for an explicit Conv1d.
    pub b_bias_hr: Param<Tensor<B, 2>>,

    /// Learnable per-head bias for C, added after QK-norm.
    /// Shape: `[nheads, state_rank]`; initialised to ones.
    pub c_bias_hr: Param<Tensor<B, 2>>,

    /// Output projection: maps `d_inner → d_model`.
    pub out_proj: Linear<B>,

    /// Optional learnable initial hidden state `h₀`.
    /// Shape: `[nheads, per_head_dim, state_rank]`
    pub init_state_hpr: Option<Param<Tensor<B, 3>>>,

    /// State rank N.
    pub state_rank: usize,

    /// Number of B/C groups G.  Must divide `nheads`.
    ///
    /// Setting G < nheads reduces the B and C projection sizes (analogous to
    /// GQA in attention), saving memory without a large accuracy cost.
    pub ngroups: usize,

    /// Number of RoPE angle pairs = state_rank / 2.
    pub num_rope_angles: usize,
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
    ///
    /// Setting G < nheads reduces the B and C projection sizes (analogous to
    /// GQA in attention), saving memory without a large accuracy cost.
    #[config(default = 1)]
    pub ngroups: usize,

    /// Minimum absolute value of A after clamping: `A ∈ (−∞, −a_floor]`.
    ///
    /// Prevents A from collapsing to zero (which would make exp(ΔA) = 1 and
    /// kill the SSM decay).  Defaults to `1e-6`, matching the reference.
    #[config(default = "1e-6")]
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

    /// Hard clamp limits for Δ at runtime: `Δ ∈ [dt_limit.0, dt_limit.1]`.
    ///
    /// Defaults to `(0, f16::MAX ≈ 65504)`, effectively only clamping at 0.
    #[config(default = "(0., 6.5504e+4)")]
    pub dt_limit: (f64, f64),

    /// Whether to add a bias term to the `in_proj` and `out_proj`.
    #[config(default = false)]
    pub has_proj_bias: bool,

    /// Whether to allocate a learnable initial SSM state `h₀`.
    ///
    /// When `false` (default), the hidden state starts at zero for every
    /// sequence.  When `true`, `init_state_hpr` is allocated as a trainable
    /// parameter of shape `[nheads, per_head_dim, state_rank]`.
    #[config(default = false)]
    pub has_learnable_init_state: bool,
}

impl Mamba3Config {
    // -----------------------------------------------------------------------
    // Computed dimensions
    // -----------------------------------------------------------------------

    /// `d_inner = expand · d_model`.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// `nheads H = d_inner / per_head_dim`.
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.per_head_dim
    }

    /// Number of RoPE angle pairs = state_rank / 2.
    pub fn num_rope_angles(&self) -> usize {
        self.state_rank / 2
    }

    /// Total input projection output size.
    ///
    /// `d_in_proj = 2·d_inner + 2·ngroups·state_rank + 3·nheads + num_rope_angles`
    ///
    /// Splits: `[z | x | B | C | dd_dt | dd_A | λ | θ]`
    pub fn d_in_proj(&self) -> usize {
        2 * self.d_inner() + 2 * self.ngroups * self.state_rank + 3 * self.nheads() + self.num_rope_angles()
    }

    // -----------------------------------------------------------------------
    // Initialisation
    // -----------------------------------------------------------------------

    /// Allocate and initialise all Mamba-3 block parameters on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba3<B> {
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let state_rank = self.state_rank;
        let num_rope_angles = self.num_rope_angles();

        assert!(state_rank % 2 == 0, "state_rank must be even for RoPE pairing");
        assert!(self.per_head_dim > 0, "per_head_dim must be positive");
        assert_eq!(
            nheads * self.per_head_dim,
            d_inner,
            "d_inner must be divisible by per_head_dim"
        );
        assert_ne!(ngroups, 0, "ngroups must be at least 1");
        assert_eq!(nheads % ngroups, 0, "nheads must be divisible by ngroups");
        assert!(self.a_floor > 0.0, "a_floor must be positive");

        // Uniform initialiser matching PyTorch's default: U(-1/√fan_in, 1/√fan_in).
        let uniform_init = |fan_in: usize| {
            let bound = 1.0 / (fan_in as f64).sqrt();
            Initializer::Uniform {
                min: -bound,
                max: bound,
            }
        };

        // ── in_proj ──────────────────────────────────────────────────────────
        // Projects d_model → (z, xbc, dt_raw).
        let in_proj = LinearConfig::new(self.d_model, self.d_in_proj())
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform_init(self.d_model))
            .init::<B>(device);

        // ── dt_bias ──────────────────────────────────────────────────────────
        // We want the initial Δ values (after softplus) to be log-uniformly
        // distributed in [dt_min, dt_max].  The inverse softplus (inverse of
        // softplus(x) = ln(1 + exp(x))) is used to back-solve for the bias:
        //   dt_bias = softplus⁻¹(dt) = dt + ln(1 - exp(-dt)) ≈ dt + ln(dt)
        // which simplifies to `dt + log(exp(dt) - 1)` in the formula below.
        let expm1 = |t: Tensor<B, 1>| t.exp() - 1.;
        let dt_h = Tensor::random(
            [nheads],
            burn::tensor::Distribution::Uniform(self.dt_min.ln(), self.dt_max.ln()),
            device,
        )
        .exp();
        let dt_h = dt_h.clamp(self.dt_init_floor, f64::INFINITY);
        // Inverse softplus: softplus⁻¹(y) = y + log(1 - e^{-y}) = y + log(e^y - 1) - y = log(e^y - 1)
        let inv_dt_h = dt_h.clone() + (-expm1(-dt_h)).log();
        let dt_bias_h = Param::from_tensor(inv_dt_h);

        // ── D (skip connection) ───────────────────────────────────────────────
        let d_h = Initializer::Ones.init::<B, 1, _>([nheads], device);

        // ── B/C QK-Norms ──────────────────────────────────────────────────────
        let b_norm = RmsNormConfig::new(state_rank).init(device);
        let c_norm = RmsNormConfig::new(state_rank).init(device);

        // ── B/C biases (initialised to ones) ─────────────────────────────────
        let b_bias_hr = Initializer::Ones.init::<B, 2, _>([nheads, state_rank], device);
        let c_bias_hr = Initializer::Ones.init::<B, 2, _>([nheads, state_rank], device);

        // ── out_proj ──────────────────────────────────────────────────────────
        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform_init(d_inner))
            .init(device);

        // ── learnable initial state (optional) ────────────────────────────────
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
            b_bias_hr,
            c_bias_hr,
            out_proj,
            init_state_hpr,
            state_rank,
            ngroups,
            num_rope_angles,
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE utility
// ---------------------------------------------------------------------------

/// Apply rotary position embeddings to `x` along its last dimension.
///
/// Uses **interleaved pairing** (NeoX / Triton style): adjacent pairs `(0,1)`, `(2,3)`, …
/// are rotated together.  This matches the official Mamba-3 Triton kernel convention.
///
/// # Shapes
/// - `x`:      `[..., state_rank]` where `state_rank` is even
/// - `angles`: `[..., state_rank / 2]`  (one angle per adjacent pair)
/// - output:   same shape as `x`
pub fn apply_rope<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    angles: Tensor<B, D>,
) -> Tensor<B, D> {
    let dims = x.dims();
    let n = dims[D - 1];
    let n2 = n / 2;
    let leading: usize = dims[..D - 1].iter().product();

    // Flatten to [leading, N] then view as [leading, N/2, 2] so each row is one pair.
    let x_pairs = x.reshape([leading, n]).reshape([leading, n2, 2]);
    let angles_flat = angles.reshape([leading, n2]);

    // Extract even (x[2i]) and odd (x[2i+1]) elements.
    let x0 = x_pairs.clone().narrow(2, 0, 1).squeeze_dim(2); // [leading, N/2]
    let x1 = x_pairs.narrow(2, 1, 1).squeeze_dim(2);          // [leading, N/2]

    let cos = angles_flat.clone().cos();
    let sin = angles_flat.sin();

    // Rotate: [x[2i], x[2i+1]] → [x[2i]·cos − x[2i+1]·sin, x[2i]·sin + x[2i+1]·cos]
    let x0r = cos.clone() * x0.clone() - sin.clone() * x1.clone();
    let x1r = sin * x0 + cos * x1;

    // Interleave back: cat along a new last dim → [leading, N/2, 2], then reshape.
    Tensor::cat(vec![x0r.unsqueeze_dim::<3>(2), x1r.unsqueeze_dim::<3>(2)], 2)
        .reshape(dims)
}

// ---------------------------------------------------------------------------
// Mamba3::forward  (chunkwise two-SSD — training / prefill)
// ---------------------------------------------------------------------------

impl<B: Backend + Mamba3BackendExt> Mamba3<B> {
    /// Process a full input sequence using the trapezoidal two-SSD algorithm.
    ///
    /// The trapezoidal recurrence is decomposed into two independent SSMs
    /// sharing the same decay α:
    ///
    /// ```text
    ///   h_t^γ = α_t h_{t-1}^γ + γ_t B_t x_t        (γ-SSM: current token)
    ///   h_t^β = α_t h_{t-1}^β + β_t B_{t-1} x_{t-1}  (β-SSM: previous token)
    ///   h_t   = h_t^γ + h_t^β
    /// ```
    ///
    /// # Shapes
    /// - `input_bsm` : `[batch, sequence, d_model]`
    /// - output      : `[batch, sequence, d_model]`
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_bsm: Tensor<B, 3>,
        cache: Option<Mamba3Cache<B>>,
        ssd_path: SsdPath,
    ) -> (Tensor<B, 3>, Mamba3Cache<B>) {
        let [batch, sequence, _d_model] = input_bsm.dims();
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let ngroups = self.ngroups;
        let per_head_dim = self.per_head_dim();
        let state_rank = self.state_rank;
        let num_rope_angles = self.num_rope_angles;
        let heads_per_group = nheads / ngroups;
        let device = input_bsm.device();

        assert!(sequence > 0, "sequence length must be at least 1");
        assert_eq!(nheads % ngroups, 0);
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        let mut cache = cache.unwrap_or_else(|| {
            let ssm_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
            let prev_bx_bhpr = Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device);
            let cum_angle_bhr = Tensor::zeros([batch, nheads, num_rope_angles], &device);
            Mamba3Cache { ssm_bhpr, prev_bx_bhpr, cum_angle_bhr }
        });

        // ── Step 1: In-projection ─────────────────────────────────────────────
        // Output layout: [z | x | B_raw | C_raw | dd_dt | dd_A | lam_raw | theta_raw]
        let proj_bsd = self.in_proj.forward(input_bsm);

        let mut parts = proj_bsd
            .split_with_sizes(
                vec![d_inner, d_inner, ngroups * state_rank, ngroups * state_rank, nheads, nheads, nheads, num_rope_angles],
                2,
            )
            .into_iter();
        let z_bsi        = parts.next().unwrap(); // [B, T, d_inner]
        let x_bsi        = parts.next().unwrap(); // [B, T, d_inner]
        let b_bsgr       = parts.next().unwrap(); // [B, T, ngroups·state_rank]
        let c_bsgr       = parts.next().unwrap(); // [B, T, ngroups·state_rank]
        let dd_dt_bsh    = parts.next().unwrap(); // [B, T, nheads]
        let dd_A_raw_bsh = parts.next().unwrap(); // [B, T, nheads]
        let lam_raw_bsh  = parts.next().unwrap(); // [B, T, nheads]
        let theta_bsa    = parts.next().unwrap(); // [B, T, num_rope_angles]

        san(&z_bsi);
        san(&x_bsi);
        san(&b_bsgr);
        san(&dd_dt_bsh);

        // ── Step 2: Discretisation + trapezoidal coefficients ─────────────────
        let dt_bias_11h = self.dt_bias_h.val().unsqueeze_dims(&[0, 1]);
        let dt_bsh = softplus(dd_dt_bsh + dt_bias_11h).clamp(self.dt_limit.0, self.dt_limit.1);

        // Data-dependent A: A_t = -softplus(dd_A_t), clamped to ≤ -a_floor.
        let a_bsh = -softplus(dd_A_raw_bsh).clamp(f64::NEG_INFINITY, -self.a_floor);
        let da_bsh = dt_bsh.clone() * a_bsh; // Δ·A, [B, T, H]

        let alpha_bsh = da_bsh.clone().exp();                                   // α = exp(ΔA)
        let lam_bsh = sigmoid(lam_raw_bsh);                                     // λ ∈ (0,1)
        let gamma_bsh = lam_bsh.clone() * dt_bsh.clone();                      // γ = λΔ
        let beta_bsh = (-lam_bsh.clone() + 1.0) * dt_bsh.clone() * alpha_bsh.clone(); // β = (1-λ)Δα

        san(&dt_bsh);
        san(&da_bsh);
        san(&gamma_bsh);
        san(&beta_bsh);

        // ── Step 3: Reshape x ─────────────────────────────────────────────────
        let x_bshp = x_bsi.reshape([batch, sequence, nheads, per_head_dim]);

        // ── Step 4: QK-Norm on B and C, then expand groups → heads, add bias ────
        // Reshape to [B, T, G, N], normalise over N (last dim), expand to [B, T, H, N].
        let b_bsgr = self.b_norm.forward(b_bsgr.reshape([batch, sequence, ngroups, state_rank]));
        let c_bsgr = self.c_norm.forward(c_bsgr.reshape([batch, sequence, ngroups, state_rank]));

        // Expand groups to heads, then add per-head bias [H, N].
        // Follows the same pattern as Mamba-2's group→head expand in step():
        //   [B, T, G, N] → [B, T, G, H/G, N] → [B, T, H, N]
        let b_bshr = b_bsgr
            .unsqueeze_dim::<5>(3) // [B, T, G, 1, N]
            .expand([batch, sequence, ngroups, heads_per_group, state_rank])
            .reshape([batch, sequence, nheads, state_rank])
            + self.b_bias_hr.val().unsqueeze_dims::<4>(&[0, 1]);
        let c_bshr = c_bsgr
            .unsqueeze_dim::<5>(3)
            .expand([batch, sequence, ngroups, heads_per_group, state_rank])
            .reshape([batch, sequence, nheads, state_rank])
            + self.c_bias_hr.val().unsqueeze_dims::<4>(&[0, 1]);
        assert_eq!([batch, sequence, nheads, state_rank], b_bshr.dims());
        assert_eq!([batch, sequence, nheads, state_rank], c_bshr.dims());

        // ── Step 5: Data-dependent cumulative RoPE angles ─────────────────────
        // Matches the reference angle_dt_fwd kernel:
        //   scaled[t,j]   = tanh(θ_{t,j}) · π          (squash to ±π)
        //   raw[t,h,j]    = scaled_j · Δ_{t,h}          (per-head scale)
        //   cum[t,h,j]    = init[h,j] + Σ_{i=0}^{t} raw[i,h,j]   (forward, additive)
        let theta_scaled_bsa = (theta_bsa * std::f32::consts::PI).tanh(); // [B, T, A]
        let raw_angles_bsha = dt_bsh.clone().unsqueeze_dim::<4>(3) // [B, T, H, 1]
            * theta_scaled_bsa.unsqueeze_dim::<4>(2); // [B, T, 1, A]
        // → [B, T, H, A]

        let cumsum_bsha = raw_angles_bsha.cumsum(1);
        let cum_angles_bsha = cache.cum_angle_bhr.clone().unsqueeze_dim::<4>(1)
            + cumsum_bsha;
        assert_eq!([batch, sequence, nheads, num_rope_angles], cum_angles_bsha.dims());
        san(&cum_angles_bsha);

        // Apply RoPE to B and C.
        let b_bshr = apply_rope::<B, 4>(b_bshr, cum_angles_bsha.clone());
        let c_bshr = apply_rope::<B, 4>(c_bshr, cum_angles_bsha.clone());
        san(&b_bshr);
        san(&c_bshr);

        // ── Step 6: Build shifted inputs for β term ───────────────────────────
        // x_prev[t] = x[t-1], x_prev[0] = 0   (zero boundary condition)
        // B_prev[t] = B[t-1], B_prev[0] = 0
        let x_prev_bshp = if sequence > 1 {
            Tensor::cat(
                vec![
                    Tensor::zeros([batch, 1, nheads, per_head_dim], &device),
                    x_bshp.clone().narrow(1, 0, sequence - 1),
                ],
                1,
            )
        } else {
            Tensor::zeros([batch, sequence, nheads, per_head_dim], &device)
        };
        let b_prev_bshr = if sequence > 1 {
            Tensor::cat(
                vec![
                    Tensor::zeros([batch, 1, nheads, state_rank], &device),
                    b_bshr.clone().narrow(1, 0, sequence - 1),
                ],
                1,
            )
        } else {
            Tensor::zeros([batch, sequence, nheads, state_rank], &device)
        };

        // ── Step 7: Scale inputs by trapezoidal coefficients ──────────────────
        let x_gamma_bshp = x_bshp.clone() * gamma_bsh.unsqueeze_dim::<4>(3); // γ_t · x_t
        let x_beta_bshp = x_prev_bshp * beta_bsh.unsqueeze_dim::<4>(3);       // β_t · x_{t-1}

        // ── Save last-token state for cache before b_bshr is moved ──────────────
        let b_last_bhr = b_bshr
            .clone()
            .narrow(1, sequence - 1, 1)
            .reshape([batch, nheads, state_rank]);

        // ── Step 8: Pad sequence to multiple of chunk_len ─────────────────────
        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;

        let (x_gamma_bShp, x_beta_bShp, da_bSh, b_bShr, b_prev_bShr, c_bShr) = if pad == 0 {
            (x_gamma_bshp, x_beta_bshp, da_bsh, b_bshr, b_prev_bshr, c_bshr)
        } else {
            let pad_hp = Tensor::zeros([batch, pad, nheads, per_head_dim], &device);
            let pad_h  = Tensor::zeros([batch, pad, nheads], &device);
            let pad_hr = Tensor::zeros([batch, pad, nheads, state_rank], &device);
            (
                Tensor::cat(vec![x_gamma_bshp, pad_hp.clone()], 1),
                Tensor::cat(vec![x_beta_bshp,  pad_hp], 1),
                Tensor::cat(vec![da_bsh,        pad_h], 1),
                Tensor::cat(vec![b_bshr,        pad_hr.clone()], 1),
                Tensor::cat(vec![b_prev_bshr,   pad_hr.clone()], 1),
                Tensor::cat(vec![c_bshr,        pad_hr], 1),
            )
        };

        // ── Reshape into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let x_gamma_bnlhp = x_gamma_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let x_beta_bnlhp  = x_beta_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        // da_bsh holds Δ·A (the pre-exponentiated decay).
        // The SSD functions expect a_decay_h as the continuous-time A (scalar per head),
        // and dt_bnlh separately.  We pass da as dt and a_decay=1 so that
        // the effective discretization inside SSD yields exp(da) = exp(Δ·A) = α.
        let da_bnlh       = da_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        // After bias expansion ngroups_eff = nheads; pass as the g dimension.
        let b_bnlhr       = b_bShr.reshape([batch, nchunks, chunk_len, nheads, state_rank]);
        let b_prev_bnlhr  = b_prev_bShr.reshape([batch, nchunks, chunk_len, nheads, state_rank]);
        let c_bnlhr       = c_bShr.reshape([batch, nchunks, chunk_len, nheads, state_rank]);

        // ── Step 9: Two SSD calls ─────────────────────────────────────────────
        // D (skip) is zeroed out here; we add it manually after summing.
        // a_decay_h = ones so SSD computes exp(da * 1) = exp(da) = α.
        let a_ones_h = Tensor::ones([nheads], &device);
        let zeros_h  = Tensor::zeros([nheads], &device);

        let ssd_gamma = SsdInput {
            x_bnlhp: x_gamma_bnlhp,
            dt_bnlh: da_bnlh.clone(),
            a_decay_h: a_ones_h.clone(),
            b_bnlgr: b_bnlhr,
            c_bnlgr: c_bnlhr.clone(),
            d_h: zeros_h.clone(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        };

        // β-SSM: zero initial state (no history before t=0)
        let ssd_beta = SsdInput {
            x_bnlhp: x_beta_bnlhp,
            dt_bnlh: da_bnlh,
            a_decay_h: a_ones_h,
            b_bnlgr: b_prev_bnlhr,
            c_bnlgr: c_bnlhr,
            d_h: zeros_h,
            initial_state_bhpr: Tensor::zeros([batch, nheads, per_head_dim, state_rank], &device),
            init_state_hpr: None,
        };

        ssd_gamma.sanity();
        ssd_beta.sanity();

        let (y_gamma_bnlhp, final_state_gamma) = match ssd_path {
            SsdPath::Minimal(_) => Self::ssd_minimal(ssd_gamma),
            SsdPath::Serial(_) => Self::ssd_serial(ssd_gamma),
            SsdPath::SerialRecalculated(_) => Self::ssd_serial_recalculated(ssd_gamma),
        };
        let (y_beta_bnlhp, final_state_beta) = match ssd_path {
            SsdPath::Minimal(_) => Self::ssd_minimal(ssd_beta),
            SsdPath::Serial(_) => Self::ssd_serial(ssd_beta),
            SsdPath::SerialRecalculated(_) => Self::ssd_serial_recalculated(ssd_beta),
        };

        let y_bnlhp = y_gamma_bnlhp + y_beta_bnlhp;
        let final_state_bhpr = final_state_gamma + final_state_beta;

        san(&y_bnlhp);
        san(&final_state_bhpr);

        cache.ssm_bhpr = final_state_bhpr;

        // ── Step 10: Unpad, D skip, gate, out-projection ─────────────────────
        let y_bShp = y_bnlhp.reshape([batch, sequence_padded, nheads, per_head_dim]);
        let y_bshp = if pad == 0 {
            y_bShp
        } else {
            y_bShp.narrow(1, 0, sequence)
        };

        // D skip: y = y + D * x   (D is per-head scalar)
        let d_11h1 = self.d_h.val().unsqueeze_dims(&[0, 1, 3]);
        let y_bshp = y_bshp + d_11h1 * x_bshp.clone();

        // Flatten heads → [B, T, d_inner]
        let y_bsi = y_bshp.reshape([batch, sequence, d_inner]);

        // Gate: y = y * silu(z)
        let y_bsi = y_bsi * Silu::new().forward(z_bsi);
        san(&y_bsi);

        // Out-projection
        let out_bsm = self.out_proj.forward(y_bsi);
        san(&out_bsm);

        // ── Update remaining cache fields ─────────────────────────────────────
        // prev_bx = outer product of last token's (B, x): [B, H, P, N]
        // b_last_bhr was saved before b_bshr was moved into the padding step.
        let x_last_bhp = x_bshp.narrow(1, sequence - 1, 1).reshape([batch, nheads, per_head_dim]);
        cache.prev_bx_bhpr =
            x_last_bhp.unsqueeze_dim::<4>(3).expand([batch, nheads, per_head_dim, state_rank])
            * b_last_bhr.unsqueeze_dim::<4>(2).expand([batch, nheads, per_head_dim, state_rank]);

        // cum_angle at last token position
        cache.cum_angle_bhr = cum_angles_bsha
            .narrow(1, sequence - 1, 1)
            .reshape([batch, nheads, num_rope_angles]);

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
        /// Runs one tick of the trapezoidal Mamba-3 recurrence:
        ///
        /// ```text
        ///   hₜ = αₜ hₜ₋₁ + βₜ prev_Bx + γₜ Bₜ xₜᵀ
        ///   yₜ = Cₜᵀ hₜ + D xₜ
        /// ```
        ///
        /// where `prev_Bx = B_{t-1} xₜ₋₁ᵀ` is stored in the cache.
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
            let device = &input_bm.device();
            let ssm_shape = [batch, nheads, per_head_dim, state_rank];

            assert_eq!(nheads % ngroups, 0);

            let mut cache = cache.unwrap_or_else(|| {
                let ssm_bhpr = Tensor::zeros(ssm_shape, device);
                let prev_bx_bhpr = Tensor::zeros(ssm_shape, device);
                let cum_angle_bhr = Tensor::zeros([batch, nheads, num_rope_angles], device);
                Mamba3Cache { ssm_bhpr, prev_bx_bhpr, cum_angle_bhr }
            });

            // ── In-projection ─────────────────────────────────────────────────
            let proj_bd = self.in_proj.forward(input_bm);
            let mut parts = proj_bd
                .split_with_sizes(
                    vec![d_inner, d_inner, ngroups * state_rank, ngroups * state_rank, nheads, nheads, nheads, num_rope_angles],
                    1,
                )
                .into_iter();
            let z_bi         = parts.next().unwrap(); // [B, d_inner]
            let x_bi         = parts.next().unwrap(); // [B, d_inner]
            let b_bgr        = parts.next().unwrap(); // [B, ngroups·state_rank]
            let c_bgr        = parts.next().unwrap(); // [B, ngroups·state_rank]
            let dd_dt_bh     = parts.next().unwrap(); // [B, nheads]
            let dd_A_raw_bh  = parts.next().unwrap(); // [B, nheads]
            let lam_raw_bh   = parts.next().unwrap(); // [B, nheads]
            let theta_ba     = parts.next().unwrap(); // [B, num_rope_angles]

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
            let beta_bh  = (-lam_bh.clone() + 1.0) * dt_bh.clone() * alpha_bh.clone();

            // ── QK-Norm on B and C, then expand groups → heads, add per-head bias
            // Reshape to [B, G, N], normalise over N, expand to [B, H, N].
            // Mirrors the forward() group→head expansion and Mamba-2's step() pattern.
            let b_bgr = self.b_norm.forward(b_bgr.reshape([batch, ngroups, state_rank]));
            let c_bgr = self.c_norm.forward(c_bgr.reshape([batch, ngroups, state_rank]));
            // [B, G, N] → [B, G, H/G, N] → [B, H, N]
            let b_bhr = b_bgr
                .unsqueeze_dim::<4>(2) // [B, G, 1, N]
                .expand([batch, ngroups, heads_per_group, state_rank])
                .reshape([batch, nheads, state_rank])
                + self.b_bias_hr.val().unsqueeze_dim::<3>(0); // + [1, H, N]
            let c_bhr = c_bgr
                .unsqueeze_dim::<4>(2)
                .expand([batch, ngroups, heads_per_group, state_rank])
                .reshape([batch, nheads, state_rank])
                + self.c_bias_hr.val().unsqueeze_dim::<3>(0);
            assert_eq!([batch, nheads, state_rank], b_bhr.dims());

            // ── RoPE: update cumulative angle and rotate B and C ───────────────
            // Matches angle_dt_fwd: tanh(θ)*π then scale by Δ, then add to state.
            let theta_scaled_ba = (theta_ba * std::f32::consts::PI).tanh();
            let raw_angle_bha =
                dt_bh.unsqueeze_dim::<3>(2) * theta_scaled_ba.unsqueeze_dim::<3>(1);
            let new_cum_angle_bha = cache.cum_angle_bhr.clone() + raw_angle_bha;

            let b_bhr = apply_rope::<B, 3>(b_bhr, new_cum_angle_bha.clone());
            let c_bhr = apply_rope::<B, 3>(c_bhr, new_cum_angle_bha.clone());

            // ── Compute Bx = B_t ⊗ x_t → [B, H, P, N] ───────────────────────
            let bx_bhpr =
                x_bhp.clone().unsqueeze_dim::<4>(3).expand(ssm_shape)
                * b_bhr.unsqueeze_dim::<4>(2).expand(ssm_shape);

            // ── SSM recurrence ─────────────────────────────────────────────────
            // h_t = α * h_{t-1} + β * prev_Bx + γ * Bx_current
            let alpha_bhpr = alpha_bh.unsqueeze_dims::<4>(&[2, 3]).expand(ssm_shape);
            let beta_bhpr  = beta_bh.unsqueeze_dims::<4>(&[2, 3]).expand(ssm_shape);
            let gamma_bhpr = gamma_bh.unsqueeze_dims::<4>(&[2, 3]).expand(ssm_shape);

            let h_bhpr = alpha_bhpr * cache.ssm_bhpr.clone()
                + beta_bhpr  * cache.prev_bx_bhpr.clone()
                + gamma_bhpr * bx_bhpr.clone();

            // ── Output: y = C^T h + D x ───────────────────────────────────────
            let y_bi = {
                let c_bhpr = c_bhr.unsqueeze_dim::<4>(2).expand(ssm_shape);
                let ch_bhp = (c_bhpr * h_bhpr.clone()).sum_dim(3).squeeze_dim(3); // [B, H, P]

                let d_1h1 = self.d_h.val().unsqueeze_dims::<3>(&[0, 2]);
                let skip_bhp = d_1h1.expand([batch, nheads, per_head_dim]) * x_bhp.clone();

                let y_bhp = ch_bhp + skip_bhp;
                let y_bi = y_bhp.reshape([batch, d_inner]);
                y_bi * Silu::new().forward(z_bi)
            };

            // ── Out-projection ────────────────────────────────────────────────
            let out_bm = self.out_proj.forward(y_bi);
            assert_eq!([batch, d_model], out_bm.dims());

            // ── Update cache ──────────────────────────────────────────────────
            cache.ssm_bhpr = h_bhpr;
            cache.prev_bx_bhpr = bx_bhpr;
            cache.cum_angle_bhr = new_cum_angle_bha;

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
    use burn::backend::Flex;

    type B = Flex;

    fn small_config() -> Mamba3Config {
        Mamba3Config::new(32) // d_model = 32
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
    }

    /// step() token-by-token must produce the same outputs as forward() on the full sequence.
    #[test]
    fn step_matches_forward() {
        let device = Default::default();
        let cfg = small_config();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 5;
        let d_model = cfg.d_model;

        let input = Tensor::<B, 3>::random(
            [batch, seq_len, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // ── Forward pass (whole sequence) ─────────────────────────────────────
        let ssd_path = SsdPath::Minimal(Some(4));
        let (out_fwd, _) = model.forward(input.clone(), None, ssd_path);
        // out_fwd: [batch, seq_len, d_model]

        // ── Step-by-step pass ─────────────────────────────────────────────────
        let mut cache: Option<Mamba3Cache<B>> = None;
        let mut step_outputs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // narrow to [batch, 1, d_model], then squeeze the length dim → [batch, d_model]
            let token = input.clone().narrow(1, t, 1).squeeze_dim(1);
            let (out_t, new_cache) = model.step(token, cache);
            cache = Some(new_cache);
            step_outputs.push(out_t);
        }

        // Stack step outputs → [batch, seq_len, d_model]
        let out_step = Tensor::stack(step_outputs, 1);

        // ── Compare ───────────────────────────────────────────────────────────
        let diff = (out_fwd - out_step).abs().max().into_scalar();
        assert!(
            diff < 1e-4,
            "step() vs forward() max absolute difference = {diff:.6} (expected < 1e-4)"
        );
    }

    /// Same consistency check with ngroups=2 (grouped B/C, 2 groups for 4 heads).
    #[test]
    fn step_matches_forward_ngroups2() {
        let device = Default::default();
        // d_model=32, expand=2 → d_inner=64, per_head_dim=16 → nheads=4, ngroups=2
        let cfg = Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_ngroups(2);
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 5;
        let d_model = cfg.d_model;

        let input = Tensor::<B, 3>::random(
            [batch, seq_len, d_model],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let ssd_path = SsdPath::Minimal(Some(4));
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
            "ngroups=2: step() vs forward() max absolute difference = {diff:.6} (expected < 1e-4)"
        );
    }
}
