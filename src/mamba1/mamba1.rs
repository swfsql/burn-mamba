//! # Mamba-1 SSM Block — Selective State Space Model
//!
//! This module implements the original selective SSM block from the paper
//! *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"*
//! (Gu & Dao, 2023).
//!
//! ## Pipeline
//!
//! ```text
//!   in_proj      d_model → [x | z]                (split into two d_inner halves)
//!   conv1d       causal depthwise conv over x + SiLU
//!   x_proj       x → [Δ_raw | B | C]
//!   dt_proj      Δ_raw → Δ;  Δ = softplus(Δ)
//!   scan         selective scan (ZOH for A, Euler for B) → y
//!   gate         y = y · SiLU(z)
//!   out_proj     d_inner → d_model
//! ```
//!
//! Unlike Mamba-2/3, the recurrence is run as a plain **sequential selective
//! scan** rather than a chunkwise SSD; there is no pluggable SSD path.  Both
//! [`Mamba1::forward`] (full sequence) and [`Mamba1::step`] (single token)
//! thread the same [`Mamba1Cache`] (convolution window + SSM state).
//!
//! ## Notation / Dimension Keys
//!
//! Tensor names carry a shape suffix (see the crate-level notation table).
//! The letters used here:
//!
//! | Letter | Dimension                         | Typical |
//! |--------|-----------------------------------|---------|
//! | `b`    | `batch`                           | varies  |
//! | `s`    | `sequence` length                 | varies  |
//! | `d`    | `d_model`                         | 768     |
//! | `i`    | `d_inner` = `expand`·`d_model`     | 2·d_model |
//! | `k`    | `conv_kernel`                     | 4       |
//! | `r`    | `state_rank` (latent SSM state)   | 16      |
//!
//! The Δ-projection rank `dt_rank` has no single-letter key; tensors carrying
//! it are annotated with an explicit shape comment.

use crate::mamba1::prelude::*;
use crate::modules::Silu;
use crate::modules::sanity as san;
use crate::modules::split_into;
use burn::backend::Backend;
use burn::prelude::*;
use burn::{
    module::{Module, Param},
    nn::conv::{Conv1d, Conv1dConfig},
    nn::{Initializer, Linear, LinearConfig, PaddingConfig1d},
};

/// The Mamba-1 selective SSM block.
#[derive(Module, Debug)]
pub struct Mamba1 {
    /// Input channel: d_model.
    /// Output channel: 2 * d_inner.
    pub in_proj: Linear,

    /// Input channel: d_inner.
    /// Output channel: d_inner.
    pub conv1d: Conv1d,

    /// Input channel: d_inner.
    /// Output channel: dt_rank + 2 * state_rank.
    pub x_proj: Linear,

    /// Input channel: dt_rank.
    /// Output channel: d_inner.
    pub dt_proj: Linear,

    /// Dims: `[d_inner, state_rank]`.
    pub a_log: Param<Tensor<2>>,

    /// Dims: `[d_inner]`.
    pub d: Param<Tensor<1>>,

    /// Input channel: d_inner.
    /// Output channel: d_model.
    pub out_proj: Linear,
}

/// Configuration / factory for [`Mamba1`].
#[derive(Config, Debug)]
pub struct Mamba1Config {
    /// Hidden dimension.
    pub d_model: usize,

    /// State rank — the latent dimension of the SSM hidden state
    /// (`N` in Algorithm 2 from the Mamba paper).
    #[config(default = 16)]
    pub state_rank: usize,

    /// Causal convolution window length.
    #[config(default = 4)]
    pub conv_kernel: usize,

    /// Expansion factor for `d_inner = expand · d_model`.
    #[config(default = 2)]
    pub expand: usize,

    /// Minimum dt value.
    #[config(default = 1e-3)]
    pub dt_min: f64,

    /// Maximum dt value.
    #[config(default = 1e-1)]
    pub dt_max: f64,

    /// Scale for dt initialization.
    #[config(default = 1.)]
    pub dt_scale: f64,

    /// Floor for dt initialization.
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,

    /// Whether the depthwise convolution should have a bias.
    #[config(default = true)]
    pub has_conv_bias: bool,

    /// Whether in_proj and out_proj should have a bias.
    #[config(default = false)]
    pub has_proj_bias: bool,

    /// Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
    /// Δ or delta: input-dependent step size.
    ///
    /// By default, set to `d_model.div_ceil(state_rank)`.
    pub dt_rank: Option<usize>,

    /// d_model * expand (`D` in Algorithm 2 from the Mamba paper).
    ///
    /// By default, set to expand * d_model.
    pub d_inner: Option<usize>,
}

impl Mamba1Config {
    /// Returns the initialized model.
    pub fn init(&self, device: &Device) -> Mamba1 {
        let d_inner = self.d_inner();
        assert_ne!(self.state_rank, 0);
        assert!(self.d_model + self.state_rank > 0);
        let dt_rank = self.dt_rank();

        // Helper function for PyTorch-style uniform initialization
        let uniform_init = |d_input: usize| {
            let bound = 1.0 / (d_input as f64).sqrt();
            Initializer::Uniform {
                min: -bound,
                max: bound,
            }
        };

        let dt_proj = {
            use burn::tensor::Distribution;
            let weight: Tensor<2> = {
                let dt_init_std = (dt_rank as f64).powf(-0.5) * self.dt_scale;
                Tensor::random(
                    [dt_rank, d_inner],
                    Distribution::Uniform(-dt_init_std, dt_init_std),
                    device,
                )
            };
            assert_eq!([dt_rank, d_inner], weight.dims());
            let bias: Tensor<1> = {
                // note: this placeholder impl may lose precision for very small values,
                // and a Taylor series could approximate it: e^x - 1 = x + x^2/2! + x^3/3! + ⋯
                // but with the clamp at dt_init_floor, this isn't necessary
                let expm1 = |t: Tensor<1>| t.exp() - 1.;
                let dt = Tensor::random([d_inner], Distribution::Uniform(0.0, 1.0), device)
                    * (f64::ln(self.dt_max) - f64::ln(self.dt_min))
                    + f64::ln(self.dt_min);
                let dt = dt.exp().clamp_min(self.dt_init_floor);
                // Inverse of softplus
                dt.clone() + (-expm1(-dt)).log()
            };
            assert_eq!([d_inner], bias.dims());
            Linear {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        };

        let a_log = {
            let a_row: Tensor<1> =
                Tensor::<1, Int>::arange(1..self.state_rank as i64 + 1, device).float();
            assert_eq!([self.state_rank], a_row.dims());
            let a_row = a_row.unsqueeze();
            assert_eq!([1, self.state_rank], a_row.dims());
            let a = a_row.repeat(&[d_inner, 1]);
            assert_eq!([d_inner, self.state_rank], a.dims());
            let a_log = a.log();
            Param::from_tensor(a_log)
        };

        Mamba1 {
            in_proj: LinearConfig::new(self.d_model, 2 * d_inner)
                .with_bias(self.has_proj_bias)
                // follows PyTorch's default initializer
                .with_initializer(uniform_init(self.d_model))
                .init(device),
            conv1d: Conv1dConfig::new(d_inner, d_inner, self.conv_kernel)
                // Causal left-padding is applied manually in `forward` (from the
                // conv cache window), so the convolution itself uses no padding.
                .with_padding(PaddingConfig1d::Valid)
                .with_groups(d_inner)
                .with_bias(self.has_conv_bias)
                // follows PyTorch's default initializer
                // fan_in = in_channels / groups * kernel_size
                .with_initializer(uniform_init(self.conv_kernel))
                .init(device),
            x_proj: LinearConfig::new(d_inner, dt_rank + 2 * self.state_rank)
                .with_bias(false)
                // follows PyTorch's default initializer
                .with_initializer(uniform_init(d_inner))
                .init(device),
            dt_proj,
            a_log,
            d: Initializer::Ones.init([d_inner], device),
            out_proj: LinearConfig::new(d_inner, self.d_model)
                .with_bias(self.has_proj_bias)
                // follows PyTorch's default initializer
                .with_initializer(uniform_init(d_inner))
                .init(device),
        }
    }
    /// Inner (expanded) channel width: the `d_inner` override if set, else
    /// `expand · d_model`.
    pub fn d_inner(&self) -> usize {
        self.d_inner.unwrap_or(self.expand * self.d_model)
    }
    /// Rank of the Δ projection: the `dt_rank` override if set, else
    /// `ceil(d_model / state_rank)`.
    pub fn dt_rank(&self) -> usize {
        self.dt_rank
            .unwrap_or(self.d_model.div_ceil(self.state_rank))
    }
}

impl Mamba1 {
    /// See also [`Self::step`].
    ///
    /// Mirrors [`crate::mamba2::mamba2::Mamba2::forward`]: an optional `cache`
    /// supplies the initial convolution window and SSM state (zero-initialised
    /// when `None`), and the updated cache is returned so a sequence can be
    /// processed in segments (prefill then decode, or chunked prefill).
    ///
    /// # Shapes
    ///   - Input `[batch, sequence, d_model]`
    ///   - Output `[batch, sequence, d_model]`
    pub fn forward(&self, x: Tensor<3>, cache: Option<Mamba1Cache>) -> (Tensor<3>, Mamba1Cache) {
        let [batch, sequence, d_model] = x.dims();
        let [d_inner] = self.d.dims();
        let [_, _, conv_kernel] = self.conv1d.weight.dims();
        let [_d_inner, state_rank] = self.a_log.dims();
        let device = x.device();
        assert!(sequence > 0, "sequence length must be at least 1");

        // Zero-initialise the cache (conv window + SSM state) when not provided.
        let mut cache = cache.unwrap_or_else(|| Mamba1Cache {
            conv_bik: Tensor::zeros([batch, d_inner, conv_kernel], &device),
            ssm_bir: Tensor::zeros([batch, d_inner, state_rank], &device),
        });
        cache.sanity();

        // layer 1 (in_proj): projects the input d_model into 2 * d_inner.
        let [xs_bsi, res_bsi] = {
            let xs_and_res = self.in_proj.forward(x);
            assert_eq!([batch, sequence, 2 * d_inner], xs_and_res.dims());
            split_into(xs_and_res, [d_inner, d_inner], 2)
        };
        assert_eq!([batch, sequence, d_inner], xs_bsi.dims());
        assert_eq!([batch, sequence, d_inner], res_bsi.dims());

        // layer 2 (conv1d) — causal, with the cache window threaded as left context
        let xs_bsi = {
            assert!(conv_kernel > 0);
            let conv_in_bis = xs_bsi.permute([0, 2, 1]);
            assert_eq!([batch, d_inner, sequence], conv_in_bis.dims());

            // Left-pad with the last (conv_kernel - 1) columns of the cached
            // window so the convolution is strictly causal and continues a
            // prior segment.
            let conv_in_padded = if conv_kernel >= 2 {
                let tail = cache.conv_bik.clone().narrow(2, 1, conv_kernel - 1);
                assert_eq!([batch, d_inner, conv_kernel - 1], tail.dims());
                Tensor::cat(vec![tail, conv_in_bis], 2)
            } else {
                conv_in_bis
            };
            assert_eq!(
                [batch, d_inner, (conv_kernel - 1) + sequence],
                conv_in_padded.dims()
            );

            // Update the conv window: the last conv_kernel columns of the padded
            // input.
            cache.conv_bik = conv_in_padded.clone().narrow(2, sequence - 1, conv_kernel);
            assert_eq!([batch, d_inner, conv_kernel], cache.conv_bik.dims());

            let xs = self.conv1d.forward(conv_in_padded);
            assert_eq!([batch, d_inner, sequence], xs.dims());

            // restore original positioning as per before the layer 2
            let xs = xs.permute([0, 2, 1]);
            assert_eq!([batch, sequence, d_inner], xs.dims());

            // activation
            let xs = Silu::new().forward(xs);
            assert_eq!([batch, sequence, d_inner], xs.dims());

            xs
        };
        assert_eq!([batch, sequence, d_inner], xs_bsi.dims());

        let (scan_bsi, final_ssm) = self.ssm(xs_bsi, cache.ssm_bir.clone());
        assert_eq!([batch, sequence, d_inner], scan_bsi.dims());
        cache.ssm_bir = final_ssm;

        // activation
        let ys = scan_bsi * Silu::new().forward(res_bsi);
        assert_eq!([batch, sequence, d_inner], ys.dims());

        let y = self.out_proj.forward(ys);
        assert_eq!([batch, sequence, d_model], y.dims());
        san(&y);

        (y, cache)
    }

    /// Computes the selective-SSM parameters (Δ, A, B, C) from the conv output
    /// and runs the [`Self::selective_scan`] recurrence over the full sequence.
    ///
    /// # Shapes
    ///   - Input u `[batch, sequence, d_inner]`
    ///   - Input init_ssm `[batch, d_inner, state_rank]`
    ///   - Output `[batch, sequence, d_inner]`
    ///   - Output (final state) `[batch, d_inner, state_rank]`
    pub fn ssm(&self, u: Tensor<3>, init_ssm: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let [batch, sequence, d_inner] = u.dims();
        let [_d_inner, state_rank] = self.a_log.dims();
        let [dt_rank, _d_inner] = self.dt_proj.weight.dims();

        // Compute ∆ A B C D, the state space parameters.

        // A
        // this is input independent (see Section 3.5.2 "Interpretation of A" form the Mamba paper for why A isn't selective)
        let a = self.a_log.val().exp().neg();
        assert_eq!([d_inner, state_rank], a.dims());

        let x_dbl = self.x_proj.forward(u.clone());
        assert_eq!([batch, sequence, dt_rank + 2 * state_rank], x_dbl.dims());

        // ∆ (part 1/2)
        // ∆ is input-dependent
        // B and C are input-dependent
        let [delta, b, c] = split_into(x_dbl, [dt_rank, state_rank, state_rank], 2);
        assert_eq!([batch, sequence, dt_rank], delta.dims()); // [batch, sequence, dt_rank]
        assert_eq!([batch, sequence, state_rank], b.dims());
        assert_eq!([batch, sequence, state_rank], c.dims());

        // ∆ (part 2/2)
        // ∆ is input-dependent
        let delta = self.dt_proj.forward(delta);
        assert_eq!([batch, sequence, d_inner], delta.dims());

        let delta = burn::tensor::activation::softplus(delta, 1.);

        let delta = delta.permute([1, 0, 2]);
        assert_eq!([sequence, batch, d_inner], delta.dims());

        let c = c.permute([1, 0, 2]);
        assert_eq!([sequence, batch, state_rank], c.dims());

        Self::selective_scan(delta, a, b, c, self.d.val(), u, init_ssm)
    }

    /// Selective Scan.
    ///
    /// Does selective scan algorithm. See:
    /// - Section 2 State Space Models from the Mamba paper;
    /// - Algorithm 2 in Section 3.2 from the Mamba paper;
    /// - run_SSM(A, B, C, u) from The Annotated S4.
    ///
    /// # Shapes
    ///   - Input delta `[sequence, batch, d_inner]`
    ///   - Input a `[d_inner, state_rank]`
    ///   - Input b `[batch, sequence, state_rank]`
    ///   - Input c `[sequence, batch, state_rank]`
    ///   - Input d `[d_inner]`
    ///   - Input u `[batch, sequence, d_inner]`
    ///   - Input init_ssm `[batch, d_inner, state_rank]`
    ///   - Output `[batch, sequence, d_inner]`
    ///   - Output (final state) `[batch, d_inner, state_rank]`
    pub fn selective_scan(
        delta: Tensor<3>,
        a: Tensor<2>,
        b: Tensor<3>,
        c: Tensor<3>,
        d: Tensor<1>,
        u: Tensor<3>,
        init_ssm: Tensor<3>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [sequence, batch, d_inner] = delta.dims();
        let [_d_inner, state_rank] = a.dims();
        let outer_shape = [sequence, batch, d_inner, state_rank];

        // Discretize continuous parameters (A, B)
        //  - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper)
        //  - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        //    "A is the more important term and the performance doesn't change much with the simplification on B"
        let (delta_a, delta_bu) = {
            let delta = delta.unsqueeze_dim(3);
            assert_eq!([sequence, batch, d_inner, 1], delta.dims());
            let delta = delta.expand(outer_shape);
            assert_eq!(outer_shape, delta.dims());

            let a = a.unsqueeze_dims(&[0, 1]);
            assert_eq!([1, 1, d_inner, state_rank], a.dims());
            let a = a.expand(outer_shape);
            assert_eq!(outer_shape, a.dims());
            let delta_a = (delta.clone() * a).exp();
            assert_eq!(outer_shape, delta_a.dims());

            let b = b.permute([1, 0, 2]);
            assert_eq!([sequence, batch, state_rank], b.dims());
            let b = b.unsqueeze_dim(2);
            assert_eq!([sequence, batch, 1, state_rank], b.dims());
            let b = b.expand(outer_shape);
            assert_eq!(outer_shape, b.dims());
            let delta_b = delta * b;
            assert_eq!(outer_shape, delta_b.dims());

            let u = u.clone().permute([1, 0, 2]);
            assert_eq!([sequence, batch, d_inner], u.dims());
            let u = u.unsqueeze_dim(3);
            assert_eq!([sequence, batch, d_inner, 1], u.dims());
            let u = u.expand(outer_shape);
            assert_eq!(outer_shape, u.dims());
            let delta_bu = delta_b * u;
            assert_eq!(outer_shape, delta_bu.dims());

            (delta_a, delta_bu)
        };
        assert_eq!(outer_shape, delta_a.dims());
        assert_eq!(outer_shape, delta_bu.dims());

        // Perform selective scan (see scan_SSM() from The Annotated S4)
        // Note that the below is sequential, while the official implementation does a much faster parallel scan that
        // is additionally hardware-aware (like FlashAttention).

        // unstack the Sequence axis

        let delta_a = delta_a.split(1, 0);
        assert_eq!(delta_a.len(), sequence);

        let delta_bu = delta_bu.split(1, 0);
        assert_eq!(delta_bu.len(), sequence);

        let c = c.unsqueeze_dim(3);
        assert_eq!([sequence, batch, state_rank, 1], c.dims());
        let c = c.split(1, 0);
        assert_eq!(c.len(), sequence);

        let inner_shape = [batch, d_inner, state_rank];
        assert_eq!(inner_shape, init_ssm.dims());
        let mut xs: Tensor<3> = init_ssm;
        let mut ys = Vec::with_capacity(sequence); // inner shape: [batch, d_inner]
        for ((delta_a, delta_bu), c) in delta_a
            .into_iter()
            .zip(delta_bu.into_iter())
            .zip(c.into_iter())
        {
            let delta_a = delta_a.squeeze_dim(0);
            assert_eq!(inner_shape, delta_a.dims());
            let delta_bu = delta_bu.squeeze_dim(0);
            assert_eq!(inner_shape, delta_bu.dims());
            let c = c.squeeze_dim(0);
            assert_eq!([batch, state_rank, 1], c.dims());

            xs = (xs.clone() * delta_a) + delta_bu;
            let y = xs.clone().matmul(c);
            assert_eq!([batch, d_inner, 1], y.dims());
            let y = y.squeeze_dim(2);
            assert_eq!([batch, d_inner], y.dims());
            ys.push(y);
        }

        let ys = Tensor::stack(ys, 1);
        assert_eq!([batch, sequence, d_inner], ys.dims());

        let d = d.unsqueeze_dims(&[0, 1]);
        assert_eq!([1, 1, d_inner], d.dims());
        let d = d.expand([batch, sequence, d_inner]);

        let ys = ys + (d * u);
        assert_eq!([batch, sequence, d_inner], ys.dims());

        (ys, xs)
    }
}

mod step {
    use super::*;

    impl Mamba1 {
        /// # Shapes
        ///   - Input `[batch, d_model]`
        ///   - Output `[batch, d_model]`
        pub fn step(&self, x: Tensor<2>, cache: Option<Mamba1Cache>) -> (Tensor<2>, Mamba1Cache) {
            let [batch, d_model] = x.dims();
            let [d_inner] = self.d.dims();
            let [_, _, conv_kernel] = self.conv1d.weight.dims();
            let [_d_inner, state_rank] = self.a_log.dims();
            let device = x.device();

            // Zero-initialise the cache (conv window + SSM state) when not
            // provided, mirroring `forward` so `step` is `Option`-coherent with
            // the Mamba-2/3 blocks.
            let mut cache = cache.unwrap_or_else(|| Mamba1Cache {
                conv_bik: Tensor::zeros([batch, d_inner, conv_kernel], &device),
                ssm_bir: Tensor::zeros([batch, d_inner, state_rank], &device),
            });
            cache.sanity();

            // layer 1 (in_proj): projects the input d_model into 2 * d_inner.
            let [xs_bi, res_bi] = {
                let xs_and_res = self.in_proj.forward(x);
                assert_eq!([batch, 2 * d_inner], xs_and_res.dims());
                split_into(xs_and_res, [d_inner, d_inner], 1)
            };
            assert_eq!([batch, d_inner], xs_bi.dims());
            assert_eq!([batch, d_inner], res_bi.dims());

            // layer 2 (conv1d): roll the window leftwards and insert the new
            // token's projection as the newest (rightmost) column.
            cache.conv_bik = {
                let t0 = cache.conv_bik.clone().narrow(2, 1, conv_kernel - 1);
                assert_eq!([batch, d_inner, conv_kernel - 1], t0.dims());

                let conv = Tensor::cat(vec![t0, xs_bi.unsqueeze_dim(2)], 2);
                assert_eq!([batch, d_inner, conv_kernel], conv.dims());

                conv
            };
            let xs_bi = {
                let conv1d = self.conv1d.weight.val();
                // [channels_out, channels_in / groups, kernel_size]
                assert_eq!([d_inner, 1, conv_kernel], conv1d.dims());
                let conv1d = conv1d.permute([1, 0, 2]);
                assert_eq!([1, d_inner, conv_kernel], conv1d.dims());
                let conv1d = conv1d.expand([batch, d_inner, conv_kernel]);
                assert_eq!([batch, d_inner, conv_kernel], conv1d.dims());

                let xs = cache.conv_bik.clone() * conv1d;
                let xs = xs.sum_dim(2);
                assert_eq!([batch, d_inner, 1], xs.dims());
                let xs = xs.squeeze_dim(2);
                assert_eq!([batch, d_inner], xs.dims());

                // conv1d bias
                let conv1d_bias = self.conv1d.bias.as_ref().unwrap().val();
                // [channels_out]
                assert_eq!([d_inner], conv1d_bias.dims());
                let conv1d_bias = conv1d_bias.unsqueeze();
                assert_eq!([1, d_inner], conv1d_bias.dims());
                let xs = xs + conv1d_bias;

                // activation
                let xs = Silu::new().forward(xs);
                assert_eq!([batch, d_inner], xs.dims());

                xs
            };
            assert_eq!([batch, d_inner], xs_bi.dims());

            let (scan_bi, cache) = self.ssm_step(xs_bi, cache);
            assert_eq!([batch, d_inner], scan_bi.dims());

            // activation
            let ys = scan_bi * Silu::new().forward(res_bi);
            assert_eq!([batch, d_inner], ys.dims());

            let y = self.out_proj.forward(ys);
            assert_eq!([batch, d_model], y.dims());

            (y, cache)
        }

        /// Single-token counterpart of [`Mamba1::ssm`]: computes the
        /// selective-SSM parameters (Δ, A, B, C) for one token and advances the
        /// recurrence by one step via [`Self::selective_scan_step`].
        ///
        /// # Shapes
        ///   - Input u `[batch, d_inner]`
        ///   - Output `[batch, d_inner]`
        pub fn ssm_step(&self, u: Tensor<2>, cache: Mamba1Cache) -> (Tensor<2>, Mamba1Cache) {
            let [batch, d_inner, state_rank] = cache.ssm_bir.dims();
            let [dt_rank, _d_inner] = self.dt_proj.weight.dims();

            // Compute ∆ A B C D, the state space parameters.

            // A
            // this is input independent (see Section 3.5.2 "Interpretation of A" form the Mamba paper for why A isn't selective)
            let a = self.a_log.val().exp().neg();
            assert_eq!([d_inner, state_rank], a.dims());

            let x_dbl = self.x_proj.forward(u.clone());
            assert_eq!([batch, dt_rank + 2 * state_rank], x_dbl.dims());

            // ∆ (part 1/2)
            // ∆ is input-dependent
            // B and C are input-dependent
            let [delta, b, c] = split_into(x_dbl, [dt_rank, state_rank, state_rank], 1);
            assert_eq!([batch, dt_rank], delta.dims()); // [batch, dt_rank]
            assert_eq!([batch, state_rank], b.dims());
            assert_eq!([batch, state_rank], c.dims());

            // ∆ (part 2/2)
            // ∆ is input-dependent
            let delta = self.dt_proj.forward(delta);
            assert_eq!([batch, d_inner], delta.dims());
            let delta = burn::tensor::activation::softplus(delta, 1.);

            Self::selective_scan_step(delta, a, b, c, self.d.val(), u, cache)
        }

        /// Selective Scan.
        ///
        /// Does selective scan algorithm. See:
        /// - Section 2 State Space Models from the Mamba paper;
        /// - Algorithm 2 in Section 3.2 from the Mamba paper;
        /// - run_SSM(A, B, C, u) from The Annotated S4.
        ///
        /// # Shapes
        ///   - Input delta `[batch, d_inner]`
        ///   - Input a `[d_inner, state_rank]`
        ///   - Input b `[batch, state_rank]`
        ///   - Input c `[batch, state_rank]`
        ///   - Input d `[d_inner]`
        ///   - Input u `[batch, d_inner]`
        ///   - Output `[batch, d_inner]`
        pub fn selective_scan_step(
            delta: Tensor<2>,
            a: Tensor<2>,
            b: Tensor<2>,
            c: Tensor<2>,
            d: Tensor<1>,
            u: Tensor<2>,
            mut cache: Mamba1Cache,
        ) -> (Tensor<2>, Mamba1Cache) {
            let [batch, d_inner, state_rank] = cache.ssm_bir.dims();
            let outer_shape = [batch, d_inner, state_rank];

            // Discretize continuous parameters (A, B)
            //  - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper)
            //  - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
            //    "A is the more important term and the performance doesn't change much with the simplification on B"
            let (delta_a, delta_bu) = {
                let delta = delta.unsqueeze_dim(2);
                assert_eq!([batch, d_inner, 1], delta.dims());
                let delta = delta.expand(outer_shape);
                assert_eq!(outer_shape, delta.dims());

                let a = a.unsqueeze();
                assert_eq!([1, d_inner, state_rank], a.dims());
                let a = a.expand(outer_shape);
                assert_eq!(outer_shape, a.dims());
                let delta_a = (delta.clone() * a).exp();
                assert_eq!(outer_shape, delta_a.dims());

                let b = b.unsqueeze_dim(1);
                assert_eq!([batch, 1, state_rank], b.dims());
                let b = b.expand(outer_shape);
                assert_eq!(outer_shape, b.dims());
                let delta_b = delta * b;
                assert_eq!(outer_shape, delta_b.dims());

                let u = u.clone().unsqueeze_dim(2);
                assert_eq!([batch, d_inner, 1], u.dims());
                let u = u.expand(outer_shape);
                assert_eq!(outer_shape, u.dims());
                let delta_bu = delta_b * u;
                assert_eq!(outer_shape, delta_bu.dims());

                (delta_a, delta_bu)
            };
            assert_eq!(outer_shape, delta_a.dims());
            assert_eq!(outer_shape, delta_bu.dims());

            cache.ssm_bir = (cache.ssm_bir.clone() * delta_a) + delta_bu;

            let c = c.unsqueeze_dim(2);
            assert_eq!([batch, state_rank, 1], c.dims());

            let y = cache.ssm_bir.clone().matmul(c);
            assert_eq!([batch, d_inner, 1], y.dims());
            let y = y.squeeze_dim(2);
            assert_eq!([batch, d_inner], y.dims());

            let d = d.unsqueeze();
            assert_eq!([1, d_inner], d.dims());
            let d = d.expand([batch, d_inner]);
            assert_eq!([batch, d_inner], d.dims());

            let y = y + (d * u);
            assert_eq!([batch, d_inner], y.dims());

            (y, cache)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
