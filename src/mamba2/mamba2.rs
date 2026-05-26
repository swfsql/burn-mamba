//! # Mamba-2 SSM Block — Structured State Space Duality (SSD)
//!
//! This module implements the core **SSD layer** from the paper
//! *"Transformers are SSMs: Generalized Models and Efficient Algorithms
//! through Structured State Space Duality"* (Dao & Gu, 2024).
//!
//! ## The SSD Model
//!
//! The SSD layer is a multi-head selective SSM.  Each head processes a
//! sequence of `per_head_dim`-dimensional inputs `X ∈ ℝ^{sequence×per_head_dim}` through the recurrence
//! (Eq. 1–2 of the paper):
//!
//! ```text
//!   hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ          (state update)
//!   yₜ = Cₜᵀ hₜ                  (output readout)
//! ```
//!
//! where:
//! - `hₜ ∈ ℝ^{`state_rank`×per_head_dim}` is the hidden state
//! - `Āₜ = exp(Δₜ A) ∈ ℝ` is a scalar decay (the key SSD constraint:
//!   Āₜ = αₜ · I, i.e. scalar times identity, rather than a diagonal matrix)
//! - `B̄ₜ = Δₜ Bₜ ∈ ℝᴺ`  is the (discretised) input projection
//! - `Cₜ ∈ ℝᴺ`           is the output projection
//! - `Δₜ > 0`             is the (input-dependent) discretisation step size
//! - `A < 0`              is a learnable scalar decay rate per head
//!
//! ## State Space Duality
//!
//! Unrolling the recurrence yields an equivalent **attention-like** form
//! (Eq. 6–7):
//!
//! ```text
//!   M = L ∘ (C Bᵀ)      ∈ ℝ^{sequence×sequence}
//!   Y = M · X
//! ```
//!
//! where `L` is the **1-semiseparable mask** (Eq. 4–5):
//!
//! ```text
//!   Lᵢⱼ = āᵢ · āᵢ₋₁ · ... · āⱼ₊₁    (i ≥ j)
//!   Lᵢⱼ = 0                     (i < j)
//! ```
//!
//! This makes the layer equivalent to causal linear attention
//! `Y = (L ∘ QKᵀ) V` under the renaming `(C, B, X) ↦ (Q, K, V)`.
//!
//! ## The Chunkwise SSD Algorithm
//!
//! See [`ssd::minimal`](crate::mamba2::ssd::minimal) for more information.
//!
//! ## Notation / Dimension Keys
//!
//! Throughout all Mamba-2 files, tensor names carry a suffix representing their shape.
//! The letters used differ from the reference paper and the python implementation.
//! The "Paper" column gives the symbol from the Mamba-2 paper (Dao & Gu, 2024); the
//! "Python" column gives the field/variable name in the reference implementation
//! (`refs/state-spaces/mamba/mamba_ssm/modules/mamba2.py`).
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
//! | `v`    | `conv_dim` | — | `conv_dim` | `d_inner` + 2·`ngroups`·`state_rank` |
//! | `k`    | `conv_kernel` | — | `d_conv` | 4 |
//! | `g`    | `ngroups` | `G` | `ngroups` | 1, .., `nheads` |
//! | `n`    | `nchunks` = `sequence`/`chunk_len` | — | `nchunks` | varies |
//! | `l`    | `chunk_len` | `Q` | `chunk_size` | 64, .., 256 |
//!
//! Uppercase letters represent a relation (e.g. offset, multiple, concat, stacking)
//! of the lowercase letters. e.g. `X` may represent `x+1`, `x-1`, `x*2`, etc.
//! `XY` may also represent `x+y`, `x*y`, etc.

use crate::mamba2::prelude::*;
use crate::utils::sanity::sanity as san;
use crate::utils::{
    rms_norm_gated::{RmsNormGated, RmsNormGatedConfig},
    silu::Silu,
    softplus::softplus,
};
use burn::prelude::*;
use burn::{
    module::{Module, Param},
    nn::conv::{Conv1d, Conv1dConfig},
    nn::{Initializer, Linear, LinearConfig},
};

// ---------------------------------------------------------------------------
// Mamba2  (the SSM block)
// ---------------------------------------------------------------------------

/// The Mamba-2 SSM block.
///
/// Implements the full SSD layer as described in §5 of the paper.  Supports
/// two execution modes:
///
/// - [`Self::forward`] — chunkwise SSD for training / prefill
///   (exploits tensor cores; linear in sequence length)
/// - [`Self::step`]    — pure recurrent form for token-by-token decoding
///   (O(`nheads`·`per_head_dim`·`state_rank`) per step)
#[derive(Module, Debug)]
pub struct Mamba2<B: Backend> {
    /// Input projection: maps `d_model → d_inner + conv_dim + nheads`.
    ///
    /// The output is split into three parts:
    /// - `z      [batch, sequence, d_inner]`  — multiplicative gate for the output norm
    /// - `xbc    [batch, sequence, conv_dim]` — input to the causal convolution, which
    ///   is then split into (x, B, C) after activation
    /// - `dt_raw [batch, sequence, nheads]`   — raw (pre-softplus) discretisation step Δ
    pub in_proj: Linear<B>,

    /// Causal depthwise Conv1d applied to the `xbc` projection.
    ///
    /// - Input/output channels: `conv_dim`
    /// - Kernel size: `conv_kernel` (typically 4)
    /// - Groups: `conv_dim` (fully depthwise — each channel is independent)
    /// - Padding: **none** (left-padding is applied manually so the convolution
    ///   is strictly causal)
    ///
    /// The convolution provides a local `conv_kernel`-token context window
    /// before the SSM, which helps the model capture short-range dependencies
    /// that the SSM's recurrent form handles less efficiently.
    pub conv1d: Conv1d<B>,

    /// Per-head bias for the discretisation step size Δ.
    ///
    /// Shape: `[nheads]`
    ///
    /// At inference time, `Δₜ = softplus(dt_rawₜ + dt_bias)`.
    /// Initialised such that the corresponding initial `Δ` values are
    /// log-uniformly distributed in `[dt_min, dt_max]`.
    pub dt_bias_h: Param<Tensor<B, 1>>,

    /// Hard clamp applied to Δ after softplus:  `Δ ∈ [dt_limit.0, dt_limit.1]`.
    ///
    /// Prevents degenerate discretisations (e.g. Δ → 0 causes Ā → 1, meaning
    /// the state never decays; Δ → ∞ causes Ā → 0, meaning the state is
    /// immediately wiped each step).
    pub dt_limit: (f64, f64),

    /// Per-head log-magnitude of the continuous-time decay parameter A.
    ///
    /// Shape: `[nheads]`
    ///
    /// The actual (negative) decay rate is `A = -exp(a_log)`.  The discrete
    /// decay is `Āₜ = exp(Δₜ · A) = exp(-Δₜ · exp(a_log)) ∈ (0, 1)`.
    ///
    /// Storing the *log* of the magnitude and negating ensures A < 0
    /// (decaying system) unconditionally and avoids any sign-constraint
    /// during gradient descent.
    pub a_log_h: Param<Tensor<B, 1>>,

    /// Per-head skip (D) coefficient.
    ///
    /// Shape: `[nheads]`
    ///
    /// Adds a direct path from the (post-convolution, pre-SSM) input to the
    /// output:  `yₜ += D · xₜ`.  Initialised to ones.
    pub d_h: Param<Tensor<B, 1>>,

    /// Gated RMSNorm applied to the SSM output, conditioned on the gate `z`.
    ///
    /// Input channel dimension: `d_inner`.
    ///
    /// This combines the multiplicative gate (from `z`) and a normalisation
    /// step into a single fused operation, matching the architecture in §5.2
    /// of the paper.
    pub norm: RmsNormGated<B>,

    /// Output projection: maps `d_inner → d_model`.
    pub out_proj: Linear<B>,

    /// Optional learnable initial hidden state `h₀`.
    ///
    /// Shape: `[nheads, per_head_dim, state_rank]`.
    ///
    /// When `None`, the initial state is zero (the standard default).
    /// When `Some`, the stored tensor is used as the initial condition for
    /// *every* forward call (not per-batch; it is broadcast over the batch
    /// dimension).
    pub init_state_hpr: Option<Param<Tensor<B, 3>>>,

    /// `state_rank` — the number of latent dimensions in the SSM hidden
    /// state `h ∈ ℝ^{state_rank×per_head_dim}` per head.
    ///
    /// Paper: `N`. Python: `d_state`.
    pub state_rank: usize,

    /// Number of B/C groups `ngroups` for grouped SSM heads (analogous to
    /// grouped-query attention). `ngroups` divides `nheads`; all `nheads/ngroups` heads
    /// within a group share the same B and C projections while having
    /// independent X, A, and Z projections.
    ///
    /// Paper: `G`. Python: `ngroups`.
    pub ngroups: usize,
}

impl<B: Backend> Mamba2<B> {
    /// `d_inner = expand · d_model`.  Inferred from the norm's weight shape.
    pub fn d_inner(&self) -> usize {
        let [d_inner] = self.norm.gamma.dims();
        d_inner
    }

    /// `nheads = d_inner / per_head_dim`.
    pub fn nheads(&self) -> usize {
        // Inferred from `a_log_h`
        let [nheads] = self.a_log_h.dims();
        nheads
    }

    /// `per_head_dim = d_inner / nheads`.
    pub fn per_head_dim(&self) -> usize {
        self.d_inner() / self.nheads()
    }

    /// `conv_dim = d_inner + 2 · ngroups · state_rank`.
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.state_rank
    }
}

// ---------------------------------------------------------------------------
// Mamba2Config  (hyperparameters and factory)
// ---------------------------------------------------------------------------

/// Hyperparameters for the Mamba-2 SSM block.
///
/// All computed quantities (e.g. `nheads`, `d_inner`, `conv_dim`) are derived
/// from the stored fields; see the helper methods on [`Mamba2Config`].
#[derive(Config, Debug)]
pub struct Mamba2Config {
    /// Model (hidden) dimension. Every token is represented as a
    /// `d_model`-dimensional vector at the input and output of the block.
    ///
    /// Paper: `D`. Python: `d_model`.
    pub d_model: usize,

    /// State rank — the latent dimension of the SSM hidden state.
    ///
    /// Larger `state_rank` gives a more expressive state but increases memory and compute
    /// per step. The paper uses N ∈ {64, 128, 256} for most experiments.
    ///
    /// Paper: `N`. Python: `d_state`.
    #[config(default = 128)]
    pub state_rank: usize,

    /// Causal convolution window length. Typically 4.
    ///
    /// Python: `d_conv`.
    #[config(default = 4)]
    pub conv_kernel: usize,

    /// Expansion factor for `d_inner = expand · d_model`.
    ///
    /// An expansion of 2 doubles the internal width, keeping the parameter
    /// count of the SSM block comparable to a standard attention layer.
    ///
    /// Paper: `E`. Python: `expand`.
    #[config(default = 2)]
    pub expand: usize,

    /// Head dimension. The total `d_inner` is split into
    /// `nheads = d_inner / per_head_dim` independent SSM heads.
    ///
    /// Typical values: 64 or 128.
    ///
    /// Paper: `P`. Python: `headdim`.
    #[config(default = 64)]
    pub per_head_dim: usize,

    /// Number of B/C groups. Must divide `nheads`.
    ///
    /// Setting `ngroups < nheads` reduces the B and C projection sizes (analogous to
    /// GQA in attention), saving memory without a large accuracy cost.
    ///
    /// Paper: `G`. Python: `ngroups`.
    #[config(default = 1)]
    pub ngroups: usize,

    /// Range `[lo, hi]` for the uniform initialisation of the magnitude of A.
    ///
    /// `A = -Uniform(lo, hi)`, stored as `a_log = log(Uniform(lo, hi))`.
    /// The paper uses `[1, 16]` by default.
    #[config(default = "(1., 16.)")]
    pub a_init_range: (f64, f64),

    /// Gated RMSNorm mode: when `true` the norm is applied *before* the gate;
    /// when `false` (default) the gate is applied first (SiLU-gated norm).
    #[config(default = false)]
    pub is_norm_before_gate: bool,

    /// Minimum value of the initial Δ distribution.  Used to set `dt_bias`.
    #[config(default = 1e-3)]
    pub dt_min: f64,

    /// Maximum value of the initial Δ distribution.  Used to set `dt_bias`.
    #[config(default = 0.1)]
    pub dt_max: f64,

    /// Floor clamped onto the sampled initial Δ values before inverting to
    /// obtain `dt_bias`.  Prevents numerical issues with very small Δ.
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,

    /// Hard clamp limits for Δ at runtime: `Δ ∈ [dt_limit.0, dt_limit.1]`.
    ///
    /// Defaults to `(0, f16::MAX ≈ 65504)`, effectively only clamping at 0.
    #[config(default = "(0., 6.5504e+4)")]
    pub dt_limit: (f64, f64),

    /// Whether to add a bias term to the `in_proj` and `out_proj` projections.
    #[config(default = false)]
    pub has_proj_bias: bool,

    /// Whether to add a bias term to the depthwise convolution.
    #[config(default = true)]
    pub has_conv_bias: bool,

    /// Whether to allocate a learnable initial SSM state `h₀`.
    ///
    /// When `false` (default), the hidden state starts at zero for every
    /// sequence.  When `true`, `init_state_hpr` is allocated as a trainable
    /// parameter of shape `[nheads, per_head_dim, state_rank]`.
    #[config(default = false)]
    pub has_learnable_init_state: bool,
}

impl Mamba2Config {
    // -----------------------------------------------------------------------
    // Computed dimensions
    // -----------------------------------------------------------------------

    /// `d_inner = expand · d_model`.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// `nheads = d_inner / per_head_dim`.
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.per_head_dim
    }

    /// `conv_dim = d_inner + 2 · ngroups · state_rank`.
    ///
    /// The depthwise convolution processes `x`, `B`, and `C` concatenated:
    /// x contributes `d_inner` channels, B and C each contribute
    /// `ngroups · state_rank` channels.
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.state_rank
    }

    // -----------------------------------------------------------------------
    // Initialisation
    // -----------------------------------------------------------------------

    /// Allocate and initialise all Mamba-2 block parameters on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2<B> {
        let d_inner = self.d_inner();
        let nheads = self.nheads();
        let conv_dim = self.conv_dim();

        assert!(self.per_head_dim > 0, "per_head_dim must be positive");
        assert_eq!(
            nheads * self.per_head_dim,
            d_inner,
            "d_inner must be divisible by per_head_dim"
        );
        assert_eq!(
            nheads % self.ngroups,
            0,
            "nheads must be divisible by ngroups"
        );

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
        // Size:  d_inner  +  conv_dim  +  nheads
        let d_in_proj_out = d_inner + conv_dim + nheads;
        let in_proj = LinearConfig::new(self.d_model, d_in_proj_out)
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform_init(self.d_model))
            .init::<B>(device);

        // ── conv1d ───────────────────────────────────────────────────────────
        // Causal depthwise convolution.  Left-padding is applied manually in
        // `forward` and `step`, so we request "Valid" (no automatic padding).
        // The initialiser fan_in is `in_channels / groups * kernel_size = 1 * conv_kernel`.
        let conv1d = Conv1dConfig::new(conv_dim, conv_dim, self.conv_kernel)
            .with_padding(burn::nn::PaddingConfig1d::Valid)
            .with_groups(conv_dim)
            .with_bias(self.has_conv_bias)
            .with_initializer(uniform_init(self.conv_kernel))
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

        // ── a_log ─────────────────────────────────────────────────────────────
        // A is constrained to be negative (decaying system).
        // We store a_log = log(|A|) and recover A = -exp(a_log) at runtime.
        // This parameterisation ensures A < 0 unconditionally.
        assert!(
            self.a_init_range.0 > 0.0,
            "a_init_range lower bound must be > 0"
        );
        assert!(
            self.a_init_range.0 < self.a_init_range.1,
            "a_init_range must satisfy lo < hi"
        );
        let a_h = Tensor::random(
            [nheads],
            burn::tensor::Distribution::Uniform(self.a_init_range.0, self.a_init_range.1),
            device,
        );
        let a_log_h = Param::from_tensor(a_h.log());

        // ── D (skip connection) ───────────────────────────────────────────────
        // Initialised to ones, adding a direct residual path from input to output.
        let d_h = Initializer::Ones.init::<B, 1, _>([nheads], device);

        // ── norm (gated RMSNorm) and out_proj ─────────────────────────────────
        let norm = RmsNormGatedConfig::new(d_inner)
            .with_norm_before_gate(self.is_norm_before_gate)
            .init(device);
        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(self.has_proj_bias)
            .with_initializer(uniform_init(d_inner))
            .init(device);

        // ── learnable initial state (optional) ────────────────────────────────
        let init_state_hpr = self.has_learnable_init_state.then(|| {
            Initializer::Zeros.init::<B, 3, _>([nheads, self.per_head_dim, self.state_rank], device)
        });

        Mamba2 {
            in_proj,
            conv1d,
            dt_bias_h,
            dt_limit: self.dt_limit,
            a_log_h,
            d_h,
            norm,
            out_proj,
            init_state_hpr,
            state_rank: self.state_rank,
            ngroups: self.ngroups,
        }
    }
}

// ---------------------------------------------------------------------------
// Mamba2::forward  (chunkwise SSD — training / prefill)
// ---------------------------------------------------------------------------

impl<B: Backend + Mamba2BackendExt> Mamba2<B> {
    /// Process a full input sequence using the chunkwise SSD algorithm.
    ///
    /// This is the primary training and prefill path.  The computation is
    /// **linear in sequence** but uses batched matrix multiplications (GEMMs) that
    /// can exploit GPU tensor cores — unlike the naive sequential recurrence,
    /// which requires O(sequence) serial steps.
    ///
    /// ## Full dataflow
    ///
    /// 1. **In-projection**: `u → (z, xbc, dt_raw)` via a single linear layer.
    /// 2. **Causal Conv1d + SiLU**: local context mixing over `xbc`.
    /// 3. **Split**: `xbc → (x, B, C)`.
    /// 4. **Discretise**: `Δ = softplus(dt_raw + dt_bias)`;
    ///    `Ā = exp(Δ · A)`;  `B̄ = Δ · B`.
    /// 5. **Padding**: sequence padding.
    /// 6. **SSD Algorithm**: chunkwise selective scan algorithm selection.
    ///    See [`Mamba2SsdPath`] for more info.
    /// 7. **Gated RMSNorm**: `y = RMSNorm(y) · σ(z)`.
    /// 8. **Out-projection**: `y → output`.
    ///
    /// ## Sequence padding
    ///
    /// If `sequence_unpadded % chunk_len ≠ 0`, the sequence is zero-padded
    /// to the next multiple of chunk_len.  Zero-padding is equivalent to inserting
    /// identity steps (`Δ = 0  ⇒  Ā = exp(0) = 1,  B̄ = 0`), so the SSM
    /// state is carried forward unchanged through the pad — making it safe to
    /// read the final state of the padded last chunk as the true final state.
    ///
    /// ## Shapes
    /// - `input_bsm` : `[batch, sequence, d_model]`
    /// - output      : `[batch, sequence, d_model]`
    /// - cache (out) : updated convolution window and SSM state
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_bsm: Tensor<B, 3>,
        cache: Option<Mamba2Cache<B>>,
        ssd_path: Mamba2SsdPath,
    ) -> (Tensor<B, 3>, Mamba2Cache<B>) {
        let [batch, sequence, _d_model] = input_bsm.dims();
        let d_inner = self.d_inner();
        let ngroups = self.ngroups;
        let nheads = self.nheads();
        let per_head_dim = self.per_head_dim();
        let conv_dim = self.conv_dim();
        let state_rank = self.state_rank;
        let [_conv_dim, _, conv_kernel] = self.conv1d.weight.dims();
        let [_d_model, d_in_proj_out] = self.in_proj.weight.dims();
        let device = input_bsm.device();
        assert_eq!(conv_dim, _conv_dim);
        assert_ne!(ngroups, 0);
        assert_eq!(conv_dim, _conv_dim);
        assert_eq!(nheads % ngroups, 0);
        assert!(sequence > 0, "sequence length must be at least 1");
        san(&input_bsm);

        // ── Initialise cache if not provided ──────────────────────────────────
        let mut cache = cache.unwrap_or_else(|| {
            let conv_bvk = Tensor::zeros(Shape::new([batch, conv_dim, conv_kernel]), &device);
            let ssm_bhpr = Tensor::zeros(
                Shape::new([batch, nheads, per_head_dim, state_rank]),
                &device,
            );
            Mamba2Cache { conv_bvk, ssm_bhpr }
        });
        cache.sanity();

        // ── Step 1: In-projection ─────────────────────────────────────────────
        // One linear layer projects the input to all SSM parameters at once.
        // This "parallel projection" structure (vs. Mamba-1's sequential
        // projections) enables tensor parallelism with only 1 all-reduce per
        // layer instead of 2.
        //
        // Projection output:  [z | xbc | dt_raw]
        //   `z      [batch, sequence, d_inner]`  — gate for the output RMSNorm
        //   `xbc    [batch, sequence, conv_dim]` — will become (x, B, C) after conv + split
        //   `dt_raw [batch, sequence, nheads]`   — raw discretisation step (pre-softplus)
        let (z_gate_bsi, xbc_bsv, dt_raw_bsh) = {
            let z_xbc_dt_bsd = self.in_proj.forward(input_bsm);
            assert_eq!([batch, sequence, d_in_proj_out], z_xbc_dt_bsd.dims());
            assert_eq!(
                [batch, sequence, d_inner + conv_dim + nheads],
                z_xbc_dt_bsd.dims(),
            );

            let [z_gate_bsi, xbc_bsv, dt_raw_bsh] =
                crate::utils::split::split_into(z_xbc_dt_bsd, [d_inner, conv_dim, nheads], 2);
            (z_gate_bsi, xbc_bsv, dt_raw_bsh)
        };
        assert_eq!([batch, sequence, d_inner], z_gate_bsi.dims());
        assert_eq!([batch, sequence, conv_dim], xbc_bsv.dims());
        assert_eq!([batch, sequence, nheads], dt_raw_bsh.dims());
        san(&z_gate_bsi);
        san(&xbc_bsv);
        san(&dt_raw_bsh);

        // ── Step 2: Causal depthwise Conv1d ───────────────────────────────────
        // Apply the causal 1-dimensional depthwise convolution to `xbc`.  To maintain
        // strict causality, the input is left-padded with the last
        // `(conv_kernel - 1)` columns from the cache (the tail of the previous
        // chunk), giving a padded input of length `(conv_kernel-1) + sequence`.
        // After the convolution the output has length `sequence` (Valid padding).
        //
        // The right-most `conv_kernel` columns of the padded input become the
        // new convolution cache for the next call.
        let xbc_bvs = xbc_bsv.permute([0, 2, 1]);
        assert_eq!([batch, conv_dim, sequence], xbc_bvs.dims());

        // Build the causally-padded input: [cached tail | new input]
        let xbc_padded_bvS = if conv_kernel >= 2 {
            // Drop the oldest (leftmost) element of the cache, keeping the
            // last (conv_kernel - 1) columns.
            let tail_bvK = cache.conv_bvk.slice(s![.., .., 1..]);
            assert_eq!([batch, conv_dim, conv_kernel - 1], tail_bvK.dims());
            Tensor::cat(vec![tail_bvK, xbc_bvs], 2)
        } else {
            // conv_kernel == 1: no causal padding needed.
            xbc_bvs
        };
        assert_eq!(
            [batch, conv_dim, (conv_kernel - 1) + sequence],
            xbc_padded_bvS.dims()
        );
        san(&xbc_padded_bvS);

        // Update the cache: save the last `conv_kernel` columns of the padded
        // input (i.e. starting at position `sequence - 1` from the new input).
        cache.conv_bvk = xbc_padded_bvS.clone().slice(s![.., .., (sequence - 1)..]);
        assert_eq!([batch, conv_dim, conv_kernel], cache.conv_bvk.dims());

        // Apply the depthwise convolution and transpose back to [batch, sequence, conv_dim].
        let xbc_bvs = self.conv1d.forward(xbc_padded_bvS);
        assert_eq!([batch, conv_dim, sequence], xbc_bvs.dims());
        san(&xbc_bvs);

        let xbc_bsv = xbc_bvs.permute([0, 2, 1]);
        assert_eq!([batch, sequence, conv_dim], xbc_bsv.dims());

        // SiLU activation (element-wise).
        let xbc_bsv = Silu::new().forward(xbc_bsv);
        assert_eq!([batch, sequence, conv_dim], xbc_bsv.dims());
        san(&xbc_bsv);

        // ── Step 3: Split xbc into (x, B, C) ──────────────────────────────────
        // After activation, xbc is partitioned along the channel dimension:
        //   x_bsi          → reshaped to x_bshp   (input)
        //   b_bsGR        → reshaped to b_bsgr   (state input proj)
        //   c_bsGR        → reshaped to c_bsgr   (state output proj)
        //
        // Note: in the SSM/attention duality, C ↔ Q, B ↔ K, x ↔ V.
        let (x_bshp, b_bsgr, c_bsgr) = {
            let [x_bsi, b_bsGR, c_bsGR] = crate::utils::split::split_into(
                xbc_bsv,
                [d_inner, ngroups * state_rank, ngroups * state_rank],
                2,
            );
            (
                x_bsi.reshape([batch, sequence, nheads, per_head_dim]), // x_bshp
                b_bsGR.reshape([batch, sequence, ngroups, state_rank]), // b_bsgr
                c_bsGR.reshape([batch, sequence, ngroups, state_rank]), // c_bsgr
            )
        };
        // No shape assertions on reshapes (shapes are algebraically guaranteed).

        // ── Step 4: Discretisation ────────────────────────────────────────────
        // Compute the discrete-time parameters from the continuous-time A
        // and input-dependent step size Δ (ZOH discretisation, §4.5):
        //
        //   Δₜ = softplus(dt_rawₜ + dt_bias)     ∈ (0, ∞)
        //   Āₜ = exp(Δₜ · A)                       ∈ (0, 1)   [scalar per head]
        //   B̄ₜ = Δₜ · Bₜ                           ∈ ℝᴺ       [Euler approx]
        //
        // The Euler approximation B̄ ≈ ΔB (instead of the exact ZOH formula)
        // is standard in Mamba-1 and Mamba-2 (see §4.5 of the reference).
        //
        // `a_head_decay_h` = A = -exp(a_log) < 0 (negative, one scalar per head).
        // Note: we pass this negative value to the ssd algo; inside
        // that function it is multiplied by Δ > 0, giving a negative exponent
        // which produces Āₜ = exp(Δₜ·A) ∈ (0,1) as required.
        let dt_bias_11h = self.dt_bias_h.val().unsqueeze_dims(&[0, 1]);
        assert_eq!([1, 1, nheads], dt_bias_11h.dims());

        let dt_bsh = softplus(dt_raw_bsh + dt_bias_11h).clamp(self.dt_limit.0, self.dt_limit.1);
        assert_eq!([batch, sequence, nheads], dt_bsh.dims());
        san(&dt_bsh);

        let a_head_decay_h = -self.a_log_h.val().exp(); // A = -exp(a_log) < 0
        assert_eq!([nheads], a_head_decay_h.dims());
        san(&a_head_decay_h);

        // ── Step 5: Pad sequence to a multiple of chunk_len ───────────────────
        // Zeros are the correct pad: Δ=0  ⇒  Ā=exp(0·A)=1, B̄=0·B=0.
        // The state is thus carried through unchanged, so the final state of
        // the padded last chunk equals the state after the last real token.
        let chunk_len = ssd_path.chunk_len_or_optimal(state_rank, per_head_dim);
        assert!(chunk_len > 0);
        let sequence_padded = sequence.next_multiple_of(chunk_len);
        let pad = sequence_padded - sequence;
        let (x_bShp, dt_bSh, b_bSgr, c_bSgr) = if pad == 0 {
            (x_bshp.clone(), dt_bsh, b_bsgr, c_bsgr)
        } else {
            let pad_bShp = Tensor::zeros(Shape::new([batch, pad, nheads, per_head_dim]), &device);
            let pad_bSh = Tensor::zeros(Shape::new([batch, pad, nheads]), &device);
            let pad_bSgr = Tensor::zeros(Shape::new([batch, pad, ngroups, state_rank]), &device);
            let x_bshp = Tensor::cat(vec![x_bshp.clone(), pad_bShp], 1);
            let dt_bsh = Tensor::cat(vec![dt_bsh, pad_bSh], 1);
            let b_bsgr = Tensor::cat(vec![b_bsgr, pad_bSgr.clone()], 1);
            let c_bsgr = Tensor::cat(vec![c_bsgr, pad_bSgr], 1);
            (x_bshp.clone(), dt_bsh, b_bsgr, c_bsgr)
        };

        // ── Reshapes into chunks ───────────────────────────────────────────────
        let nchunks = sequence_padded / chunk_len;
        let x_bnlhp = x_bShp.reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let dt_bnlh = dt_bSh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlgr = b_bSgr.reshape([batch, nchunks, chunk_len, ngroups, state_rank]);
        let c_bnlgr = c_bSgr.reshape([batch, nchunks, chunk_len, ngroups, state_rank]);

        // GQA expansion: B and C are produced per-group, but the SSD algorithms expect
        // them per-head. Group dim is at axis 3 (`[b, n, l, g, r]`).
        let b_bnlhr = crate::utils::gqa::gqa_expand_to_heads::<_, 5, 6>(b_bnlgr, 3, nheads);
        let c_bnlhr = crate::utils::gqa::gqa_expand_to_heads::<_, 5, 6>(c_bnlgr, 3, nheads);

        // ── Step 6: Selective Scan ────────────────────────────────────────────
        let ssd_input = crate::mamba2::ssd::Mamba2SsdInput {
            x_bnlhp,
            dt_bnlh,
            a_decay_h: a_head_decay_h,
            b_bnlhr,
            c_bnlhr,
            d_h: self.d_h.val(),
            initial_state_bhpr: cache.ssm_bhpr,
            init_state_hpr: self.init_state_hpr.as_ref().map(|s| s.val()),
        };
        ssd_input.sanity();
        let (y_bnlhp, final_state_bhpr) = ssd_path.run(ssd_input);
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_bnlhp.dims()
        );
        san(&y_bnlhp);
        san(&final_state_bhpr);

        // Update the SSM state in the cache for the next call.
        cache.ssm_bhpr = final_state_bhpr;
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            cache.ssm_bhpr.dims()
        );

        // Remove zero-pad columns that were added at Step 5.
        let y_bShp = y_bnlhp.reshape([batch, sequence_padded, nheads, per_head_dim]);
        let y_bshp = if pad == 0 {
            y_bShp
        } else {
            y_bShp.slice(s![.., 0..sequence, .., ..])
        };

        // Reshape into sequence.
        let y_bsi = y_bshp.reshape([batch, sequence, d_inner]);
        assert_eq!([batch, sequence, d_inner], y_bsi.dims());

        // ── Step 7: Gated RMSNorm ─────────────────────────────────────────────
        let y_bsi = self.norm.forward(y_bsi, z_gate_bsi);
        assert_eq!([batch, sequence, d_inner], y_bsi.dims());
        san(&y_bsi);

        // ── Step 8: Out-projection ────────────────────────────────────────────
        let out_bsm = self.out_proj.forward(y_bsi);
        assert_eq!([batch, sequence, _d_model], out_bsm.dims());
        san(&out_bsm);

        (out_bsm, cache)
    }
}

// ---------------------------------------------------------------------------
// Mamba2::step  (recurrent SSM — token-by-token decoding)
// ---------------------------------------------------------------------------

mod step {
    use super::*;

    impl<B: Backend> Mamba2<B> {
        /// Process a **single token** using the pure recurrent SSM form.
        ///
        /// This is the O(nheads·per_head_dim·state_rank)-per-token decoding path.  It runs one tick of
        /// the discretised Mamba-2 recurrence:
        ///
        /// ```text
        ///   Āₜ  = exp(Δₜ · A)           scalar per head, ∈ (0, 1)
        ///   B̄ₜ  = Δₜ · Bₜ               ∈ ℝᴺ   (Euler discretisation)
        ///   hₜ  = Āₜ · hₜ₋₁ + B̄ₜ · xₜᵀ  ∈ ℝ^{per_head_dim×state_rank}   (outer product update)
        ///   yₜ  = Cₜᵀ · hₜ + D · xₜ     ∈ ℝᴾ   (output)
        /// ```
        ///
        /// The convolution is handled by manually sliding the cache window:
        /// the oldest input column is dropped and the new token's projection
        /// is appended.
        ///
        /// The SSM hidden state `cache.ssm_bhpr` is updated in-place via
        /// the recurrence above.
        ///
        /// # Shapes
        /// - `input_bm` : `[batch, d_model]`
        /// - output     : `[batch, d_model]`
        #[allow(non_snake_case)]
        pub fn step(
            &self,
            input_bm: Tensor<B, 2>,
            cache: Option<Mamba2Cache<B>>,
        ) -> (Tensor<B, 2>, Mamba2Cache<B>) {
            let [batch, d_model] = input_bm.dims();
            let d_inner = self.d_inner();
            let ngroups = self.ngroups;
            let nheads = self.nheads();
            let per_head_dim = self.per_head_dim();
            let conv_dim = self.conv_dim();
            let state_rank = self.state_rank;
            let [_conv_dim, _, conv_kernel] = self.conv1d.weight.dims();
            let [_d_model, d_in_proj_out] = self.in_proj.weight.dims();

            assert_eq!(conv_dim, _conv_dim);
            assert_eq!(nheads % ngroups, 0);

            let mut cache = cache.unwrap_or_else(|| {
                let device = &input_bm.device();
                let conv_bvk = Tensor::zeros(Shape::new([batch, conv_dim, conv_kernel]), device);
                let ssm_bhpr = Tensor::zeros(
                    Shape::new([batch, nheads, per_head_dim, state_rank]),
                    device,
                );
                Mamba2Cache { conv_bvk, ssm_bhpr }
            });

            // ── In-projection ─────────────────────────────────────────────────
            let (z_gate_bi, xbc_bv, dt_raw_bh) = {
                let z_xbc_dt_bd = self.in_proj.forward(input_bm);
                assert_eq!([batch, d_in_proj_out], z_xbc_dt_bd.dims());
                assert_eq!([batch, d_inner + conv_dim + nheads], z_xbc_dt_bd.dims());

                let [z_gate_bi, xbc_bv, dt_raw_bh] =
                    crate::utils::split::split_into(z_xbc_dt_bd, [d_inner, conv_dim, nheads], 1);
                (z_gate_bi, xbc_bv, dt_raw_bh)
            };
            assert_eq!([batch, d_inner], z_gate_bi.dims());
            assert_eq!([batch, conv_dim], xbc_bv.dims());
            assert_eq!([batch, nheads], dt_raw_bh.dims());

            // ── Causal convolution (single step) ──────────────────────────────
            // Slide the cache window left by one position, then insert the new
            // token's projection `xbc_bv` as the rightmost column.
            cache.conv_bvk = {
                let conv_bvk = cache.conv_bvk;
                assert_eq!([batch, conv_dim, conv_kernel], conv_bvk.dims());

                // Drop the oldest (leftmost) column.
                let tail_bvK = conv_bvk.slice(s![.., .., 1..]);
                assert_eq!([batch, conv_dim, conv_kernel - 1], tail_bvK.dims());

                // Append the new token as the rightmost column.
                let updated_bvk = Tensor::cat([tail_bvK, xbc_bv.unsqueeze_dim(2)].to_vec(), 2);
                assert_eq!([batch, conv_dim, conv_kernel], updated_bvk.dims());
                updated_bvk
            };

            // Apply the depthwise convolution manually (one step = dot product
            // of the cached window with the conv weight along the kernel axis).
            let xbc_bv = {
                let conv1d_v1k = self.conv1d.weight.val(); // [conv_dim, 1, conv_kernel]
                assert_eq!([conv_dim, 1, conv_kernel], conv1d_v1k.dims());

                let conv1d_bvk = conv1d_v1k
                    .permute([1, 0, 2]) // conv1d_1vk
                    .expand([batch, conv_dim, conv_kernel]); // conv1d_bvk
                assert_eq!([batch, conv_dim, conv_kernel], conv1d_bvk.dims());

                // Element-wise multiply and sum over the kernel axis.
                let product_bvk = cache.conv_bvk.clone() * conv1d_bvk;
                let mut xbc_bv = product_bvk.sum_dim(2).squeeze_dim(2);
                assert_eq!([batch, conv_dim], xbc_bv.dims());

                // Add the (optional) bias.
                if let Some(bias_v) = &self.conv1d.bias {
                    assert_eq!([conv_dim], bias_v.dims());
                    xbc_bv = xbc_bv + bias_v.val().unsqueeze();
                }

                Silu::new().forward(xbc_bv)
            };
            assert_eq!([batch, conv_dim], xbc_bv.dims());

            // ── Split (x, B, C) ───────────────────────────────────────────────
            assert_eq!(d_inner, nheads * per_head_dim);
            let (x_bhp, b_bgr, c_bgr) = {
                let [x_bi, b_bGR, c_bGR] = crate::utils::split::split_into(
                    xbc_bv,
                    [d_inner, ngroups * state_rank, ngroups * state_rank],
                    1,
                );
                (
                    x_bi.reshape([batch, nheads, per_head_dim]), // x_bhp
                    b_bGR.reshape([batch, ngroups, state_rank]), // b_bgr
                    c_bGR.reshape([batch, ngroups, state_rank]), // c_bgr
                )
            };

            // ── Discretisation ────────────────────────────────────────────────
            // Δₜ = softplus(dt_raw + dt_bias)
            let dt_bias_1h = self.dt_bias_h.val().unsqueeze_dim(0);
            assert_eq!([1, nheads], dt_bias_1h.dims());
            let dt_bh = softplus(dt_raw_bh + dt_bias_1h).clamp(self.dt_limit.0, self.dt_limit.1);
            assert_eq!([batch, nheads], dt_bh.dims());

            // A = -exp(a_log) < 0   (negative, decaying)
            let a_head_decay_h = -self.a_log_h.val().exp();
            assert_eq!([nheads], a_head_decay_h.dims());

            // Āₜ = exp(Δₜ · A) ∈ (0, 1)   scalar per [batch, nheads]
            let dta_bh = (dt_bh.clone() * a_head_decay_h.unsqueeze()).exp();
            assert_eq!([batch, nheads], dta_bh.dims());

            // ── SSM state update:  hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜᵀ ───────────────────
            // The cache holds h_{t-1} with shape _bhpr.
            // Āₜ is a scalar per head, so we broadcast it over per_head_dim and state_rank.
            // B̄ₜ xₜᵀ is an outer product producing a _pr matrix per _bh.

            let ssm_shape_bhpr = [batch, nheads, per_head_dim, state_rank];

            let dta_bhpr = dta_bh.unsqueeze_dims::<4>(&[2, 3]).expand(ssm_shape_bhpr);

            // B̄ₜ xₜᵀ = (Δₜ Bₜ) xₜᵀ
            let heads_per_group = nheads / ngroups;
            let dtbx_bhpr = {
                let x_bhpr = x_bhp.clone().unsqueeze_dim::<4>(3).expand(ssm_shape_bhpr);

                // Expand B from _bgr → _bhpr, matching the SSD forward path:
                // each group's projection is replicated across the heads_per_group heads of
                // that group so that heads 0..(nheads/ngroups) belong to group 0, etc.
                let b_bhpr = b_bgr
                    .unsqueeze_dim::<4>(2) // b_bg1r
                    .expand([batch, ngroups, heads_per_group, state_rank]) // b_bgHr
                    .reshape([batch, nheads, state_rank]) // b_bhr
                    .unsqueeze_dim::<4>(2) // b_bh1r
                    .expand(ssm_shape_bhpr); // b_bhpr

                let dt_bhpr = dt_bh.unsqueeze_dims::<4>(&[2, 3]).expand(ssm_shape_bhpr);

                dt_bhpr * b_bhpr * x_bhpr // B̄ₜ xₜᵀ
            };
            assert_eq!(ssm_shape_bhpr, dtbx_bhpr.dims());

            // hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜᵀ
            cache.ssm_bhpr = cache.ssm_bhpr * dta_bhpr + dtbx_bhpr;
            assert_eq!(ssm_shape_bhpr, cache.ssm_bhpr.dims());

            // ── Output:  yₜ = Cₜ hₜ + D xₜ ──────────────────────────────────
            let y_bi = {
                // Cₜ hₜ:  element-wise multiply C
                // with hₜ, then sum over state_rank.
                let c_bhpr = c_bgr
                    .unsqueeze_dim::<4>(2) // c_bg1r
                    .expand([batch, ngroups, heads_per_group, state_rank]) // c_bgHr
                    .reshape([batch, nheads, state_rank]) // c_bhr
                    .unsqueeze_dim::<4>(2) // c_bh1r
                    .expand(ssm_shape_bhpr); // c_bhpr
                assert_eq!(ssm_shape_bhpr, c_bhpr.dims());

                let ch_bhp = (cache.ssm_bhpr.clone() * c_bhpr).sum_dim(3).squeeze_dim(3);
                assert_eq!([batch, nheads, per_head_dim], ch_bhp.dims());

                // D xₜ:  per-head scalar skip.
                let d_1h1 = self.d_h.val().unsqueeze_dims(&[0, 2]);
                assert_eq!([1, nheads, 1], d_1h1.dims());
                let skip_bhp = d_1h1.expand([batch, nheads, per_head_dim]) * x_bhp;
                assert_eq!([batch, nheads, per_head_dim], skip_bhp.dims());

                let y_bhp = ch_bhp + skip_bhp;
                assert_eq!([batch, nheads, per_head_dim], y_bhp.dims());

                // Flatten heads, then apply gated RMSNorm.
                let y_bi = y_bhp.reshape([batch, d_inner]);
                self.norm.forward(y_bi, z_gate_bi)
            };
            assert_eq!([batch, d_inner], y_bi.dims());

            // ── Out-projection ────────────────────────────────────────────────
            let out_bm = self.out_proj.forward(y_bi);
            assert_eq!([batch, d_model], out_bm.dims());

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

    fn small_config() -> Mamba2Config {
        Mamba2Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
    }

    /// A bundle of input + model-parameter gradients extracted from one
    /// forward+backward run.  Each `check_grads_match` call compares these
    /// across two runs that should be mathematically equivalent.
    struct RunGrads {
        out: Tensor<InnerB, 3>,
        /// Final convolution window from the returned cache.
        final_conv: Tensor<InnerB, 3>,
        /// Final SSM hidden state from the returned cache.
        final_ssm: Tensor<InnerB, 4>,
        d_input: Tensor<InnerB, 3>,
        d_in_proj_w: Tensor<InnerB, 2>,
        d_conv1d_w: Tensor<InnerB, 3>,
        d_dt_bias: Tensor<InnerB, 1>,
        d_a_log: Tensor<InnerB, 1>,
        d_d: Tensor<InnerB, 1>,
        d_norm_gamma: Tensor<InnerB, 1>,
        d_out_proj_w: Tensor<InnerB, 2>,
    }

    /// Fixed (non-tracked) random "downstream heads" used to form a scalar loss
    /// from the output **and** the final cache, so the backward pass exercises
    /// both the output and the state path.
    struct Heads {
        out: Tensor<InnerB, 3>,
        conv: Tensor<InnerB, 3>,
        ssm: Tensor<InnerB, 4>,
    }

    /// Build the initial cache passed to both `forward` and the `step`
    /// unrolling. With `random = false` it is zero (the standard fresh start);
    /// with `random = true` it holds random values, exercising parity from an
    /// arbitrary initial state (conv window + SSM hidden state).
    fn build_init_cache(cfg: &Mamba2Config, batch: usize, random: bool) -> Mamba2Cache<B> {
        let device: Device = Default::default();
        let conv_dim = cfg.conv_dim();
        let conv_kernel = cfg.conv_kernel;
        let nheads = cfg.nheads();
        let per_head_dim = cfg.per_head_dim;
        let state_rank = cfg.state_rank;
        let (conv, ssm) = if random {
            let dist = Distribution::Normal(0.0, 1.0);
            (
                Tensor::<InnerB, 3>::random([batch, conv_dim, conv_kernel], dist, &device),
                Tensor::<InnerB, 4>::random(
                    [batch, nheads, per_head_dim, state_rank],
                    dist,
                    &device,
                ),
            )
        } else {
            (
                Tensor::<InnerB, 3>::zeros([batch, conv_dim, conv_kernel], &device),
                Tensor::<InnerB, 4>::zeros([batch, nheads, per_head_dim, state_rank], &device),
            )
        };
        Mamba2Cache {
            conv_bvk: Tensor::from_inner(conv),
            ssm_bhpr: Tensor::from_inner(ssm),
        }
    }

    /// Compare the output and final cache (conv window + SSM state) of two runs.
    fn assert_outputs_match(label: &str, a: &RunGrads, b: &RunGrads, tol: f32) {
        use crate::utils::test_helpers::max_abs_diff;
        let d_out = max_abs_diff(a.out.clone(), b.out.clone());
        let d_conv = max_abs_diff(a.final_conv.clone(), b.final_conv.clone());
        let d_ssm = max_abs_diff(a.final_ssm.clone(), b.final_ssm.clone());
        assert!(
            d_out < tol,
            "{label}: output max abs diff = {d_out:.6} (tol {tol})"
        );
        assert!(
            d_conv < tol,
            "{label}: final conv window max abs diff = {d_conv:.6} (tol {tol})"
        );
        assert!(
            d_ssm < tol,
            "{label}: final SSM state max abs diff = {d_ssm:.6} (tol {tol})"
        );
    }

    /// Run a closure that produces an output tensor from a model and an input
    /// (wrapped as a `Param` so it has its own autodiff leaf), then derive a
    /// scalar loss with a fixed (non-tracked) random "head" and return the
    /// gradients of the input and a representative set of model parameters.
    fn run_with_grads(
        model: &Mamba2<B>,
        input: &Param<Tensor<B, 3>>,
        heads: &Heads,
        forward: impl FnOnce(&Mamba2<B>, Tensor<B, 3>) -> (Tensor<B, 3>, Mamba2Cache<B>),
    ) -> RunGrads {
        let (out, cache) = forward(model, input.val());
        let out_inner = out.clone().inner();
        let conv = cache.conv_bvk;
        let ssm = cache.ssm_bhpr;
        let final_conv = conv.clone().inner();
        let final_ssm = ssm.clone().inner();

        // Loss couples the output and the final cache (each via its own random
        // head) so parameter gradients reflect both the output and state paths.
        let out_head = Tensor::from_inner(heads.out.clone());
        let conv_head = Tensor::from_inner(heads.conv.clone());
        let ssm_head = Tensor::from_inner(heads.ssm.clone());
        let loss = (out * out_head).sum() + (conv * conv_head).sum() + (ssm * ssm_head).sum();
        let grads = loss.backward();

        RunGrads {
            out: out_inner,
            final_conv,
            final_ssm,
            d_input: input.val().grad(&grads).expect("grad input"),
            d_in_proj_w: model
                .in_proj
                .weight
                .val()
                .grad(&grads)
                .expect("grad in_proj.weight"),
            d_conv1d_w: model
                .conv1d
                .weight
                .val()
                .grad(&grads)
                .expect("grad conv1d.weight"),
            d_dt_bias: model.dt_bias_h.val().grad(&grads).expect("grad dt_bias_h"),
            d_a_log: model.a_log_h.val().grad(&grads).expect("grad a_log_h"),
            d_d: model.d_h.val().grad(&grads).expect("grad d_h"),
            d_norm_gamma: model
                .norm
                .gamma
                .val()
                .grad(&grads)
                .expect("grad norm.gamma"),
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
        check!(d_conv1d_w, "conv1d.weight");
        check!(d_dt_bias, "dt_bias_h");
        check!(d_a_log, "a_log_h");
        check!(d_d, "d_h");
        check!(d_norm_gamma, "norm.gamma");
        check!(d_out_proj_w, "out_proj.weight");
        assert!(
            failures.is_empty(),
            "gradient mismatches:\n  {}",
            failures.join("\n  ")
        );
    }

    /// Helper that builds a fresh `Param<Tensor>` from a stable inner tensor.
    /// A new Param is needed per run so that the autodiff leaf has a fresh
    /// node, isolating each backward pass to its own forward graph.
    fn param_input(input: &Tensor<InnerB, 3>) -> Param<Tensor<B, 3>> {
        Param::from_tensor(Tensor::from_inner(input.clone()))
    }

    /// `forward(x)` is mathematically equivalent to repeatedly calling `step`
    /// token-by-token from the **same** initial cache. Outputs, the final cache
    /// (conv window + SSM state), and parameter gradients must all agree up to
    /// float-summation-order noise.
    ///
    /// With `random_init = true` the shared initial cache is random rather than
    /// zero. Parity from an arbitrary initial state subsumes the chunked-prefill
    /// (split-vs-full) guarantee: if `forward` from any state matches the
    /// recurrent unrolling from that same state — outputs *and* final cache —
    /// then feeding a `forward`-produced cache back in continues correctly.
    fn run_step_matches_forward(cfg: Mamba2Config, ssd_path: Mamba2SsdPath, random_init: bool) {
        let device: Device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        // seq_len >= conv_kernel so the final conv window is fully determined by
        // the sequence (the initial window is flushed out), keeping the window
        // comparison well-defined for both zero and random init.
        let seq_len = 5;
        let d_model = cfg.d_model;
        let normal = Distribution::Normal(0.0, 1.0);

        let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
        let heads = Heads {
            out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
            conv: Tensor::<InnerB, 3>::random(
                [batch, cfg.conv_dim(), cfg.conv_kernel],
                normal,
                &device,
            ),
            ssm: Tensor::<InnerB, 4>::random(
                [batch, cfg.nheads(), cfg.per_head_dim, cfg.state_rank],
                normal,
                &device,
            ),
        };

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
            let mut cache: Option<Mamba2Cache<B>> = Some(cache_step);
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
        // step() and forward() are different reductions of the same SSM, so
        // their per-parameter gradients should also agree, modulo float-
        // summation order noise.
        check_grads_match("step vs forward", &r_fwd, &r_step, 1e-3);

        // ── Guard: the random initial state must actually be consumed ─────
        // Re-run forward from a *zero* initial cache; its output must differ
        // from the random-init output. Otherwise the initial state is being
        // silently ignored and forward/step would match trivially.
        if random_init {
            use crate::utils::test_helpers::max_abs_diff;
            let (out_zero, _) = model.forward(
                Tensor::from_inner(input.clone()),
                Some(build_init_cache(&cfg, batch, false)),
                ssd_path.clone(),
            );
            let d = max_abs_diff(r_fwd.out.clone(), out_zero.inner());
            assert!(
                d > 1e-3,
                "random initial state appears ignored: random-init vs zero-init \
                 output max abs diff = {d:.6} (expected a clear difference)"
            );
        }
    }

    fn cfg_ngroups2() -> Mamba2Config {
        Mamba2Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(16)
            .with_ngroups(2)
    }

    fn cfg_norm_before_gate() -> Mamba2Config {
        Mamba2Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
            .with_is_norm_before_gate(true)
    }

    #[test]
    fn step_matches_forward() {
        run_step_matches_forward(small_config(), Mamba2SsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn step_matches_forward_random_init() {
        run_step_matches_forward(small_config(), Mamba2SsdPath::Minimal(Some(4)), true);
    }

    #[test]
    fn step_matches_forward_ngroups2() {
        run_step_matches_forward(cfg_ngroups2(), Mamba2SsdPath::Minimal(Some(4)), false);
    }

    #[test]
    fn step_matches_forward_ngroups2_random_init() {
        run_step_matches_forward(cfg_ngroups2(), Mamba2SsdPath::Minimal(Some(4)), true);
    }

    // ── is_norm_before_gate = true ───────────────────────────────────────────

    #[test]
    fn step_matches_forward_norm_before_gate() {
        run_step_matches_forward(
            cfg_norm_before_gate(),
            Mamba2SsdPath::Minimal(Some(4)),
            false,
        );
    }

    #[test]
    fn step_matches_forward_norm_before_gate_random_init() {
        run_step_matches_forward(
            cfg_norm_before_gate(),
            Mamba2SsdPath::Minimal(Some(4)),
            true,
        );
    }

    // ── SSD path agreement ───────────────────────────────────────────────────

    /// `Minimal`, `Serial`, and `SerialRecalculated` are chunkwise reformulations
    /// of the same SSD, so their block-level outputs, final caches, and gradients
    /// must agree — from a zero (`random_init = false`) or random initial cache.
    fn run_ssd_paths_agree(cfg: Mamba2Config, random_init: bool) {
        let device: Device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 8;
        let d_model = cfg.d_model;
        let normal = Distribution::Normal(0.0, 1.0);

        let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
        let heads = Heads {
            out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
            conv: Tensor::<InnerB, 3>::random(
                [batch, cfg.conv_dim(), cfg.conv_kernel],
                normal,
                &device,
            ),
            ssm: Tensor::<InnerB, 4>::random(
                [batch, cfg.nheads(), cfg.per_head_dim, cfg.state_rank],
                normal,
                &device,
            ),
        };

        let init_cache = build_init_cache(&cfg, batch, random_init);

        let run = |path: Mamba2SsdPath| {
            let input_p = param_input(&input);
            let cache_p = init_cache.clone();
            run_with_grads(&model, &input_p, &heads, |m, x| {
                m.forward(x, Some(cache_p), path)
            })
        };
        let r_min = run(Mamba2SsdPath::Minimal(Some(4)));
        let r_ser = run(Mamba2SsdPath::Serial(Some(4)));
        let r_rec = run(Mamba2SsdPath::SerialRecalculated(Some(4)));

        assert_outputs_match("Minimal vs Serial", &r_min, &r_ser, 1e-4);
        assert_outputs_match("Minimal vs SerialRecalculated", &r_min, &r_rec, 1e-4);
        check_grads_match("Minimal vs Serial", &r_min, &r_ser, 1e-3);
        check_grads_match("Minimal vs SerialRecalculated", &r_min, &r_rec, 1e-3);

        // ── Guard: the random initial state must actually be consumed ─────
        if random_init {
            use crate::utils::test_helpers::max_abs_diff;
            let (out_zero, _) = model.forward(
                Tensor::from_inner(input.clone()),
                Some(build_init_cache(&cfg, batch, false)),
                Mamba2SsdPath::Minimal(Some(4)),
            );
            let d = max_abs_diff(r_min.out.clone(), out_zero.inner());
            assert!(
                d > 1e-3,
                "random initial state appears ignored: random-init vs zero-init \
                 output max abs diff = {d:.6} (expected a clear difference)"
            );
        }
    }

    #[test]
    fn ssd_paths_agree() {
        run_ssd_paths_agree(small_config(), false);
    }

    #[test]
    fn ssd_paths_agree_random_init() {
        run_ssd_paths_agree(small_config(), true);
    }

    #[test]
    fn ssd_paths_agree_ngroups2() {
        run_ssd_paths_agree(cfg_ngroups2(), false);
    }

    #[test]
    fn ssd_paths_agree_ngroups2_random_init() {
        run_ssd_paths_agree(cfg_ngroups2(), true);
    }
}
