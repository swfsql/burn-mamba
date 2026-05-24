use crate::mamba1::prelude::*;
use crate::utils::silu::Silu;
use burn::prelude::*;
use burn::{
    module::{Module, Param},
    nn::conv::{Conv1d, Conv1dConfig},
    nn::{Initializer, Linear, LinearConfig, PaddingConfig1d},
};

#[derive(Module, Debug)]
pub struct Mamba1<B: Backend> {
    /// Input channel: d_model.
    /// Output channel: 2 * d_inner.
    pub in_proj: Linear<B>,

    /// Input channel: d_inner.
    /// Output channel: d_inner.
    pub conv1d: Conv1d<B>,

    /// Input channel: d_inner.
    /// Output channel: dt_rank + 2 * d_state.
    pub x_proj: Linear<B>,

    /// Input channel: dt_rank.
    /// Output channel: d_inner.
    pub dt_proj: Linear<B>,

    /// Dims: `[d_inner, d_state]`.
    pub a_log: Param<Tensor<B, 2>>,

    /// Dims: `[d_inner]`.
    pub d: Param<Tensor<B, 1>>,

    /// Input channel: d_inner.
    /// Output channel: d_model.
    pub out_proj: Linear<B>,
}

#[derive(Config, Debug)]
pub struct Mamba1Config {
    /// Hidden dimension.
    pub d_model: usize,

    /// latent state dimension (`N` in Algorithm 2 from the Mamba paper).
    #[config(default = 16)]
    pub d_state: usize,

    #[config(default = 4)]
    pub d_conv: usize,

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

    /// Whether conv1d should have a bias.
    #[config(default = true)]
    pub conv_bias: bool,

    /// Whether in_proj and out_proj should have a bias.
    #[config(default = false)]
    pub bias: bool,

    /// Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
    /// Δ or delta: input-dependent step size.
    ///
    /// By default, set to (d_model + d_state - 1) / d_state.
    pub dt_rank: Option<usize>,

    /// DModel * expand (`D` in Algorithm 2 from the Mamba paper).
    ///
    /// By default, set to expand * d_model.
    pub d_inner: Option<usize>,
}

impl Mamba1Config {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba1<B> {
        let d_inner = self.d_inner();
        debug_assert_ne!(self.d_state, 0);
        debug_assert!(self.d_model + self.d_state > 0);
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
            let weight: Tensor<B, 2> = {
                let dt_init_std = (dt_rank as f64).powf(-0.5) * self.dt_scale;
                Tensor::random(
                    [dt_rank, d_inner],
                    Distribution::Uniform(-dt_init_std, dt_init_std),
                    device,
                )
            };
            debug_assert_eq!([dt_rank, d_inner], weight.dims());
            let bias: Tensor<B, 1> = {
                // note: this placeholder impl may lose precision for very small values,
                // and a Taylor series could approximate it: e^x - 1 = x + x^2/2! + x^3/3! + ⋯
                // but with the clamp at dt_init_floor, this isn't necessary
                let expm1 = |t: Tensor<B, 1>| t.exp() - 1.;
                let dt = Tensor::random([d_inner], Distribution::Uniform(0.0, 1.0), device)
                    * (f64::ln(self.dt_max) - f64::ln(self.dt_min))
                    + f64::ln(self.dt_min);
                let dt = dt.exp().clamp_min(self.dt_init_floor);
                // Inverse of softplus
                let inv_dt = dt.clone() + (-expm1(-dt)).log();
                inv_dt
            };
            debug_assert_eq!([d_inner], bias.dims());
            Linear {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        };

        let a_log = {
            let a_row: Tensor<B, 1> =
                Tensor::<B, 1, Int>::arange(1..self.d_state as i64 + 1, device).float();
            debug_assert_eq!([self.d_state], a_row.dims());
            let a_row = a_row.unsqueeze();
            debug_assert_eq!([1, self.d_state], a_row.dims());
            let a = a_row.repeat(&[d_inner, 1]);
            debug_assert_eq!([d_inner, self.d_state], a.dims());
            let a_log = a.log();
            Param::from_tensor(a_log)
        };

        Mamba1 {
            in_proj: LinearConfig::new(self.d_model, 2 * d_inner)
                .with_bias(self.bias)
                // follows PyTorch's default initializer
                .with_initializer(uniform_init(self.d_model))
                .init(device),
            conv1d: Conv1dConfig::new(d_inner, d_inner, self.d_conv)
                // Causal left-padding is applied manually in `forward` (from the
                // conv cache window), so the convolution itself uses no padding.
                .with_padding(PaddingConfig1d::Valid)
                .with_groups(d_inner)
                .with_bias(self.conv_bias)
                // follows PyTorch's default initializer
                // fan_in = in_channels / groups * kernel_size
                .with_initializer(uniform_init(self.d_conv))
                .init(device),
            x_proj: LinearConfig::new(d_inner, dt_rank + 2 * self.d_state)
                .with_bias(false)
                // follows PyTorch's default initializer
                .with_initializer(uniform_init(d_inner))
                .init(device),
            dt_proj,
            a_log,
            d: Initializer::Ones.init([d_inner], device),
            out_proj: LinearConfig::new(d_inner, self.d_model)
                .with_bias(self.bias)
                // follows PyTorch's default initializer
                .with_initializer(uniform_init(d_inner))
                .init(device),
        }
    }
    pub fn d_inner(&self) -> usize {
        self.d_inner.unwrap_or(self.expand * self.d_model)
    }
    pub fn dt_rank(&self) -> usize {
        self.dt_rank.unwrap_or(self.d_model.div_ceil(self.d_state))
    }
}

impl<B: Backend> Mamba1<B> {
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
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cache: Option<Mamba1Cache<B>>,
    ) -> (Tensor<B, 3>, Mamba1Cache<B>) {
        let [batch, sequence, d_model] = x.dims();
        let [d_inner] = self.d.dims();
        let [_, _, d_conv] = self.conv1d.weight.dims();
        let [_d_inner, d_state] = self.a_log.dims();
        let device = x.device();
        debug_assert!(sequence > 0, "sequence length must be at least 1");

        // Zero-initialise the cache (conv window + SSM state) when not provided.
        let mut cache = cache.unwrap_or_else(|| Mamba1Cache {
            conv: Param::from_tensor(Tensor::zeros([batch, d_inner, d_conv], &device)),
            ssm: Param::from_tensor(Tensor::zeros([batch, d_inner, d_state], &device)),
        });

        // layer 1 (in_proj)
        let (xs, res) = {
            // projects the input d_model into 2 * d_inner
            let xs_and_res = self.in_proj.forward(x);
            debug_assert_eq!([batch, sequence, 2 * d_inner], xs_and_res.dims());

            let mut split = xs_and_res
                .split_with_sizes(vec![d_inner, d_inner], 2)
                .into_iter();
            debug_assert_eq!(split.len(), 2);
            (split.next().unwrap(), split.next().unwrap())
        };
        debug_assert_eq!([batch, sequence, d_inner], xs.dims());
        debug_assert_eq!([batch, sequence, d_inner], res.dims());

        // layer 2 (conv1d) — causal, with the cache window threaded as left context
        let xs = {
            debug_assert!(d_conv > 0);
            // let xs = xs.swap_dims(1, 2);
            let xs_bis = xs.permute([0, 2, 1]);
            debug_assert_eq!([batch, d_inner, sequence], xs_bis.dims());

            // Left-pad with the last (d_conv - 1) columns of the cached window so
            // the convolution is strictly causal and continues a prior segment.
            let xs_padded = if d_conv >= 2 {
                let tail = cache.conv.val().narrow(2, 1, d_conv - 1);
                debug_assert_eq!([batch, d_inner, d_conv - 1], tail.dims());
                Tensor::cat(vec![tail, xs_bis], 2)
            } else {
                xs_bis
            };
            debug_assert_eq!([batch, d_inner, (d_conv - 1) + sequence], xs_padded.dims());

            // Update the conv window: the last d_conv columns of the padded input.
            // `Param::map` (rather than `Param::from_tensor`) is required because
            // the new window is a computed, non-leaf autodiff tensor.
            let new_conv = xs_padded.clone().narrow(2, sequence - 1, d_conv);
            cache.conv = cache.conv.map(|_| new_conv);
            debug_assert_eq!([batch, d_inner, d_conv], cache.conv.dims());

            let xs = self.conv1d.forward(xs_padded);
            debug_assert_eq!([batch, d_inner, sequence], xs.dims());

            // restore original positioning as per before the layer 2
            // let xs = xs.swap_dims(1, 2);
            let xs = xs.permute([0, 2, 1]);
            debug_assert_eq!([batch, sequence, d_inner], xs.dims());

            // activation
            let xs = Silu::new().forward(xs);
            debug_assert_eq!([batch, sequence, d_inner], xs.dims());

            xs
        };
        debug_assert_eq!([batch, sequence, d_inner], xs.dims());

        let (ss, final_ssm) = self.ss(xs, cache.ssm.val());
        debug_assert_eq!([batch, sequence, d_inner], ss.dims());
        cache.ssm = cache.ssm.map(|_| final_ssm);

        // activation
        let ys = ss * Silu::new().forward(res);
        debug_assert_eq!([batch, sequence, d_inner], ys.dims());

        let y = self.out_proj.forward(ys);
        debug_assert_eq!([batch, sequence, d_model], y.dims());

        (y, cache)
    }

    /// # Shapes
    ///   - Input u `[batch, sequence, d_inner]`
    ///   - Input init_ssm `[batch, d_inner, d_state]`
    ///   - Output `[batch, sequence, d_inner]`
    ///   - Output (final state) `[batch, d_inner, d_state]`
    pub fn ss(&self, u: Tensor<B, 3>, init_ssm: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, sequence, d_inner] = u.dims();
        let [_d_inner, d_state] = self.a_log.dims();
        let [dt_rank, _d_inner] = self.dt_proj.weight.dims();

        // Compute ∆ A B C D, the state space parameters.

        // A
        // this is input independent (see Section 3.5.2 "Interpretation of A" form the Mamba paper for why A isn't selective)
        let a = self.a_log.val().exp().neg();
        debug_assert_eq!([d_inner, d_state], a.dims());

        let x_dbl = self.x_proj.forward(u.clone());
        debug_assert_eq!([batch, sequence, dt_rank + 2 * d_state], x_dbl.dims());

        // ∆ (part 1/2)
        // ∆ is input-dependent
        // B and C are input-dependent
        let mut split = x_dbl
            .split_with_sizes(vec![dt_rank, d_state, d_state], 2)
            .into_iter();
        let delta = split.next().unwrap();
        let b = split.next().unwrap();
        let c = split.next().unwrap();
        debug_assert_eq!([batch, sequence, dt_rank], delta.dims());
        debug_assert_eq!([batch, sequence, d_state], b.dims());
        debug_assert_eq!([batch, sequence, d_state], c.dims());

        // ∆ (part 2/2)
        // ∆ is input-dependent
        let delta = self.dt_proj.forward(delta);
        debug_assert_eq!([batch, sequence, d_inner], delta.dims());

        let delta = burn::tensor::activation::softplus(delta, 1.);

        // let delta = delta.swap_dims(0, 1);
        let delta = delta.permute([1, 0, 2]);
        debug_assert_eq!([sequence, batch, d_inner], delta.dims());

        // let c = c.swap_dims(0, 1);
        let c = c.permute([1, 0, 2]);
        debug_assert_eq!([sequence, batch, d_state], c.dims());

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
    ///   - Input a `[d_inner, d_state]`
    ///   - Input b `[batch, sequence, d_state]`
    ///   - Input c `[sequence, batch, d_state]`
    ///   - Input d `[d_inner]`
    ///   - Input u `[batch, sequence, d_inner]`
    ///   - Input init_ssm `[batch, d_inner, d_state]`
    ///   - Output `[batch, sequence, d_inner]`
    ///   - Output (final state) `[batch, d_inner, d_state]`
    pub fn selective_scan(
        delta: Tensor<B, 3>,
        a: Tensor<B, 2>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
        u: Tensor<B, 3>,
        init_ssm: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [sequence, batch, d_inner] = delta.dims();
        let [_d_inner, d_state] = a.dims();
        let outer_shape = [sequence, batch, d_inner, d_state];

        // Discretize continuous parameters (A, B)
        //  - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper)
        //  - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        //    "A is the more important term and the performance doesn't change much with the simplification on B"
        let (delta_a, delta_bu) = {
            let delta = delta.unsqueeze_dim(3);
            debug_assert_eq!([sequence, batch, d_inner, 1], delta.dims());
            let delta = delta.expand(outer_shape);
            debug_assert_eq!(outer_shape, delta.dims());

            let a = a.unsqueeze_dims(&[0, 1]);
            debug_assert_eq!([1, 1, d_inner, d_state], a.dims());
            let a = a.expand(outer_shape);
            debug_assert_eq!(outer_shape, a.dims());
            let delta_a = (delta.clone() * a).exp();
            debug_assert_eq!(outer_shape, delta_a.dims());

            // let b = b.swap_dims(0, 1);
            let b = b.permute([1, 0, 2]);
            debug_assert_eq!([sequence, batch, d_state], b.dims());
            let b = b.unsqueeze_dim(2);
            debug_assert_eq!([sequence, batch, 1, d_state], b.dims());
            let b = b.expand(outer_shape);
            debug_assert_eq!(outer_shape, b.dims());
            let delta_b = delta * b;
            debug_assert_eq!(outer_shape, delta_b.dims());

            // let u = u.clone().swap_dims(0, 1);
            let u = u.clone().permute([1, 0, 2]);
            debug_assert_eq!([sequence, batch, d_inner], u.dims());
            let u = u.unsqueeze_dim(3);
            debug_assert_eq!([sequence, batch, d_inner, 1], u.dims());
            let u = u.expand(outer_shape);
            debug_assert_eq!(outer_shape, u.dims());
            let delta_bu = delta_b * u;
            debug_assert_eq!(outer_shape, delta_bu.dims());

            (delta_a, delta_bu)
        };
        debug_assert_eq!(outer_shape, delta_a.dims());
        debug_assert_eq!(outer_shape, delta_bu.dims());

        // Perform selective scan (see scan_SSM() from The Annotated S4)
        // Note that the below is sequential, while the official implementation does a much faster parallel scan that
        // is additionally hardware-aware (like FlashAttention).

        // unstack the Sequence axis

        let delta_a = delta_a.split(1, 0);
        debug_assert_eq!(delta_a.len(), sequence);

        let delta_bu = delta_bu.split(1, 0);
        debug_assert_eq!(delta_bu.len(), sequence);

        let c = c.unsqueeze_dim(3);
        debug_assert_eq!([sequence, batch, d_state, 1], c.dims());
        let c = c.split(1, 0);
        debug_assert_eq!(c.len(), sequence);

        let inner_shape = [batch, d_inner, d_state];
        debug_assert_eq!(inner_shape, init_ssm.dims());
        let mut xs: Tensor<B, 3> = init_ssm;
        let mut ys = Vec::with_capacity(sequence); // inner shape: [batch, d_inner]
        for ((delta_a, delta_bu), c) in delta_a
            .into_iter()
            .zip(delta_bu.into_iter())
            .zip(c.into_iter())
        {
            let delta_a = delta_a.squeeze_dim(0);
            debug_assert_eq!(inner_shape, delta_a.dims());
            let delta_bu = delta_bu.squeeze_dim(0);
            debug_assert_eq!(inner_shape, delta_bu.dims());
            let c = c.squeeze_dim(0);
            debug_assert_eq!([batch, d_state, 1], c.dims());

            xs = (xs.clone() * delta_a) + delta_bu;
            let y = xs.clone().matmul(c);
            debug_assert_eq!([batch, d_inner, 1], y.dims());
            let y = y.squeeze_dim(2);
            debug_assert_eq!([batch, d_inner], y.dims());
            ys.push(y);
        }

        let ys = Tensor::stack(ys, 1);
        debug_assert_eq!([batch, sequence, d_inner], ys.dims());

        let d = d.unsqueeze_dims(&[0, 1]);
        debug_assert_eq!([1, 1, d_inner], d.dims());
        let d = d.expand([batch, sequence, d_inner]);

        let ys = ys + (d * u);
        debug_assert_eq!([batch, sequence, d_inner], ys.dims());

        (ys, xs)
    }
}

mod step {
    use super::*;

    impl<B: Backend> Mamba1<B> {
        /// # Shapes
        ///   - Input `[batch, d_model]`
        ///   - Output `[batch, d_model]`
        pub fn step(
            &self,
            x: Tensor<B, 2>,
            mut cache: Mamba1Cache<B>,
        ) -> (Tensor<B, 2>, Mamba1Cache<B>) {
            let [batch, d_inner, d_conv] = cache.conv.dims();
            let [_batch, d_model] = x.dims();

            // layer 1 (in_proj)
            let (xs, res) = {
                // projects the input d_model into 2 * d_inner
                let xs_and_res = self.in_proj.forward(x);
                debug_assert_eq!([batch, 2 * d_inner], xs_and_res.dims());

                let mut split = xs_and_res
                    .split_with_sizes(vec![d_inner, d_inner], 1)
                    .into_iter();
                (split.next().unwrap(), split.next().unwrap())
            };
            debug_assert_eq!([batch, d_inner], xs.dims());
            debug_assert_eq!([batch, d_inner], res.dims());

            // layer 2 (conv1d)
            cache.conv = cache.conv.map(|conv| {
                // split-off oldest/first column (i.e. rolling leftwards)
                let t0 = conv.narrow(2, 1, d_conv - 1);
                debug_assert_eq!([batch, d_inner, d_conv - 1], t0.dims());

                // insert xs as a the newest/last column
                let conv = Tensor::cat([t0, xs.unsqueeze_dim(2)].to_vec(), 2);
                debug_assert_eq!([batch, d_inner, d_conv], conv.dims());

                conv
            });
            let xs = {
                let conv1d = self.conv1d.weight.val();
                // [channels_out, channels_in / groups, kernel_size]
                debug_assert_eq!([d_inner, 1, d_conv], conv1d.dims());
                // let conv1d = conv1d.swap_dims(1, 0);
                let conv1d = conv1d.permute([1, 0, 2]);
                debug_assert_eq!([1, d_inner, d_conv], conv1d.dims());
                let conv1d = conv1d.expand([batch, d_inner, d_conv]);
                debug_assert_eq!([batch, d_inner, d_conv], conv1d.dims());

                let xs = cache.conv.val() * conv1d;
                let xs = xs.sum_dim(2);
                debug_assert_eq!([batch, d_inner, 1], xs.dims());
                let xs = xs.squeeze_dim(2);
                debug_assert_eq!([batch, d_inner], xs.dims());

                // conv1d bias
                let conv1d_bias = self.conv1d.bias.as_ref().unwrap().val();
                // [channels_out]
                debug_assert_eq!([d_inner], conv1d_bias.dims());
                let conv1d_bias = conv1d_bias.unsqueeze();
                debug_assert_eq!([1, d_inner], conv1d_bias.dims());
                let xs = xs + conv1d_bias;

                // activation
                let xs = Silu::new().forward(xs);
                debug_assert_eq!([batch, d_inner], xs.dims());

                xs
            };
            debug_assert_eq!([batch, d_inner], xs.dims());

            let (ss, cache) = self.ss_step(xs, cache);
            debug_assert_eq!([batch, d_inner], ss.dims());

            // activation
            let ys = ss * Silu::new().forward(res);
            debug_assert_eq!([batch, d_inner], ys.dims());

            let y = self.out_proj.forward(ys);
            debug_assert_eq!([batch, d_model], y.dims());

            (y, cache)
        }

        /// Runs the SSM. See:
        /// - Algorithm 2 in Section 3.2 from the Mamba paper;
        /// - run_SSM(A, B, C, u) from The Annotated S4.
        ///
        /// # Shapes
        ///   - Input u `[batch, d_inner]`
        ///   - Output `[batch, d_inner]`
        pub fn ss_step(
            &self,
            u: Tensor<B, 2>,
            cache: Mamba1Cache<B>,
        ) -> (Tensor<B, 2>, Mamba1Cache<B>) {
            let [batch, d_inner, d_state] = cache.ssm.dims();
            let [dt_rank, _d_inner] = self.dt_proj.weight.dims();

            // Compute ∆ A B C D, the state space parameters.

            // A
            // this is input independent (see Section 3.5.2 "Interpretation of A" form the Mamba paper for why A isn't selective)
            let a = self.a_log.val().exp().neg();
            debug_assert_eq!([d_inner, d_state], a.dims());

            let x_dbl = self.x_proj.forward(u.clone());
            debug_assert_eq!([batch, dt_rank + 2 * d_state], x_dbl.dims());

            // ∆ (part 1/2)
            // ∆ is input-dependent
            // B and C are input-dependent
            let mut split = x_dbl
                .split_with_sizes(vec![dt_rank, d_state, d_state], 1)
                .into_iter();
            let delta = split.next().unwrap();
            let b = split.next().unwrap();
            let c = split.next().unwrap();
            debug_assert_eq!([batch, dt_rank], delta.dims());
            debug_assert_eq!([batch, d_state], b.dims());
            debug_assert_eq!([batch, d_state], c.dims());

            // ∆ (part 2/2)
            // ∆ is input-dependent
            let delta = self.dt_proj.forward(delta);
            debug_assert_eq!([batch, d_inner], delta.dims());
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
        ///   - Input a `[d_inner, d_state]`
        ///   - Input b `[batch, d_state]`
        ///   - Input c `[batch, d_state]`
        ///   - Input d `[d_inner]`
        ///   - Input u `[batch, d_inner]`
        ///   - Output `[batch, d_inner]`
        pub fn selective_scan_step(
            delta: Tensor<B, 2>,
            a: Tensor<B, 2>,
            b: Tensor<B, 2>,
            c: Tensor<B, 2>,
            d: Tensor<B, 1>,
            u: Tensor<B, 2>,
            mut cache: Mamba1Cache<B>,
        ) -> (Tensor<B, 2>, Mamba1Cache<B>) {
            let [batch, d_inner, d_state] = cache.ssm.dims();
            let outer_shape = [batch, d_inner, d_state];

            // Discretize continuous parameters (A, B)
            //  - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper)
            //  - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
            //    "A is the more important term and the performance doesn't change much with the simplification on B"
            let (delta_a, delta_bu) = {
                let delta = delta.unsqueeze_dim(2);
                debug_assert_eq!([batch, d_inner, 1], delta.dims());
                let delta = delta.expand(outer_shape);
                debug_assert_eq!(outer_shape, delta.dims());

                let a = a.unsqueeze();
                debug_assert_eq!([1, d_inner, d_state], a.dims());
                let a = a.expand(outer_shape);
                debug_assert_eq!(outer_shape, a.dims());
                let delta_a = (delta.clone() * a).exp();
                debug_assert_eq!(outer_shape, delta_a.dims());

                let b = b.unsqueeze_dim(1);
                debug_assert_eq!([batch, 1, d_state], b.dims());
                let b = b.expand(outer_shape);
                debug_assert_eq!(outer_shape, b.dims());
                let delta_b = delta * b;
                debug_assert_eq!(outer_shape, delta_b.dims());

                let u = u.clone().unsqueeze_dim(2);
                debug_assert_eq!([batch, d_inner, 1], u.dims());
                let u = u.expand(outer_shape);
                debug_assert_eq!(outer_shape, u.dims());
                let delta_bu = delta_b * u;
                debug_assert_eq!(outer_shape, delta_bu.dims());

                (delta_a, delta_bu)
            };
            debug_assert_eq!(outer_shape, delta_a.dims());
            debug_assert_eq!(outer_shape, delta_bu.dims());

            cache.ssm = cache.ssm.map(|ssm| (ssm * delta_a) + delta_bu);

            let c = c.unsqueeze_dim(2);
            debug_assert_eq!([batch, d_state, 1], c.dims());

            let y = cache.ssm.val().matmul(c);
            debug_assert_eq!([batch, d_inner, 1], y.dims());
            let y = y.squeeze_dim(2);
            debug_assert_eq!([batch, d_inner], y.dims());

            let d = d.unsqueeze();
            debug_assert_eq!([1, d_inner], d.dims());
            let d = d.expand([batch, d_inner]);
            debug_assert_eq!([batch, d_inner], d.dims());

            let y = y + (d * u);
            debug_assert_eq!([batch, d_inner], y.dims());

            (y, cache)
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

    fn small_config() -> Mamba1Config {
        Mamba1Config::new(32) // d_model = 32
            .with_d_state(8)
            .with_d_conv(4)
            .with_expand(2)
    }

    /// A bundle of input + model-parameter gradients extracted from one
    /// forward+backward run.  Each `check_grads_match` call compares these
    /// across two runs that should be mathematically equivalent.
    struct RunGrads {
        out: Tensor<InnerB, 3>,
        /// Final convolution window from the returned cache.
        final_conv: Tensor<InnerB, 3>,
        /// Final SSM state from the returned cache.
        final_ssm: Tensor<InnerB, 3>,
        d_input: Tensor<InnerB, 3>,
        d_in_proj_w: Tensor<InnerB, 2>,
        d_conv1d_w: Tensor<InnerB, 3>,
        d_x_proj_w: Tensor<InnerB, 2>,
        d_dt_proj_w: Tensor<InnerB, 2>,
        d_dt_proj_b: Tensor<InnerB, 1>,
        d_a_log: Tensor<InnerB, 2>,
        d_d: Tensor<InnerB, 1>,
        d_out_proj_w: Tensor<InnerB, 2>,
    }

    /// Fixed (non-tracked) random "downstream heads" used to form a scalar loss
    /// from the output **and** the final cache, so the backward pass exercises
    /// both the output and the state path.
    struct Heads {
        out: Tensor<InnerB, 3>,
        conv: Tensor<InnerB, 3>,
        ssm: Tensor<InnerB, 3>,
    }

    /// Run a closure that produces an output tensor from a model and an input
    /// (wrapped as a `Param` so it has its own autodiff leaf), then derive a
    /// scalar loss with a fixed (non-tracked) random "head" and return the
    /// gradients of the input and a representative set of model parameters.
    fn run_with_grads(
        model: &Mamba1<B>,
        input: &Param<Tensor<B, 3>>,
        heads: &Heads,
        forward: impl FnOnce(&Mamba1<B>, Tensor<B, 3>) -> (Tensor<B, 3>, Mamba1Cache<B>),
    ) -> RunGrads {
        let (out, cache) = forward(model, input.val());
        let out_inner = out.clone().inner();
        let conv = cache.conv.val();
        let ssm = cache.ssm.val();
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
            d_x_proj_w: model
                .x_proj
                .weight
                .val()
                .grad(&grads)
                .expect("grad x_proj.weight"),
            d_dt_proj_w: model
                .dt_proj
                .weight
                .val()
                .grad(&grads)
                .expect("grad dt_proj.weight"),
            d_dt_proj_b: model
                .dt_proj
                .bias
                .as_ref()
                .expect("dt_proj has bias")
                .val()
                .grad(&grads)
                .expect("grad dt_proj.bias"),
            d_a_log: model.a_log.val().grad(&grads).expect("grad a_log"),
            d_d: model.d.val().grad(&grads).expect("grad d"),
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
        check!(d_x_proj_w, "x_proj.weight");
        check!(d_dt_proj_w, "dt_proj.weight");
        check!(d_dt_proj_b, "dt_proj.bias");
        check!(d_a_log, "a_log");
        check!(d_d, "d");
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

    /// Build the initial cache (conv window + SSM state) passed to both
    /// `forward` and the `step` unrolling. With `random = false` the cache is
    /// zero (the standard fresh start); with `random = true` it holds random
    /// values, exercising forward/step parity from an arbitrary initial state.
    fn build_init_cache(cfg: &Mamba1Config, batch: usize, random: bool) -> Mamba1Cache<B> {
        let device: Device = Default::default();
        let d_inner = cfg.d_inner();
        let d_conv = cfg.d_conv;
        let d_state = cfg.d_state;
        let (conv, ssm) = if random {
            let dist = Distribution::Normal(0.0, 1.0);
            (
                Tensor::<InnerB, 3>::random([batch, d_inner, d_conv], dist, &device),
                Tensor::<InnerB, 3>::random([batch, d_inner, d_state], dist, &device),
            )
        } else {
            (
                Tensor::<InnerB, 3>::zeros([batch, d_inner, d_conv], &device),
                Tensor::<InnerB, 3>::zeros([batch, d_inner, d_state], &device),
            )
        };
        Mamba1Cache {
            conv: Param::from_tensor(Tensor::from_inner(conv)),
            ssm: Param::from_tensor(Tensor::from_inner(ssm)),
        }
    }

    /// `forward(x)` is mathematically equivalent to repeatedly calling `step`
    /// token-by-token from the **same** initial cache: the latter is the
    /// recurrent unrolling of the former. Both the outputs, the final cache
    /// (conv window + SSM state), and the parameter gradients must agree up to
    /// float-summation-order noise.
    ///
    /// With `random_init = true` the shared initial cache is random rather than
    /// zero. Parity from an arbitrary initial state subsumes the chunked-prefill
    /// (split-vs-full) guarantee: if `forward` from any state equals the
    /// recurrent unrolling from that same state — outputs *and* final cache —
    /// then feeding a `forward`-produced cache back in continues correctly.
    fn run_step_matches_forward(cfg: Mamba1Config, random_init: bool) {
        let device: Device = Default::default();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        // seq_len >= d_conv so the final conv window is fully determined by the
        // sequence (the initial window has been flushed out), keeping the
        // window comparison well-defined for both zero and random init.
        let seq_len = 5;
        let d_model = cfg.d_model;
        let d_inner = cfg.d_inner();
        let d_conv = cfg.d_conv;
        let d_state = cfg.d_state;
        assert!(seq_len >= d_conv);

        let normal = Distribution::Normal(0.0, 1.0);
        let input = Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device);
        let heads = Heads {
            out: Tensor::<InnerB, 3>::random([batch, seq_len, d_model], normal, &device),
            conv: Tensor::<InnerB, 3>::random([batch, d_inner, d_conv], normal, &device),
            ssm: Tensor::<InnerB, 3>::random([batch, d_inner, d_state], normal, &device),
        };

        let init_cache = build_init_cache(&cfg, batch, random_init);

        let input_fwd = param_input(&input);
        let cache_fwd = init_cache.clone();
        let r_fwd =
            run_with_grads(&model, &input_fwd, &heads, |m, x| m.forward(x, Some(cache_fwd)));

        let input_step = param_input(&input);
        let cache_step = init_cache;
        let r_step = run_with_grads(&model, &input_step, &heads, |m, x| {
            let mut cache = cache_step;
            let mut outs: Vec<Tensor<B, 2>> = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let token = x.clone().narrow(1, t, 1).squeeze_dim(1);
                let (out_t, new_cache) = m.step(token, cache);
                cache = new_cache;
                outs.push(out_t);
            }
            (Tensor::stack(outs, 1), cache)
        });

        // ── Forward + final-state agreement ──────────────────────────────
        use crate::utils::test_helpers::max_abs_diff;
        let val_tol = 1e-4;
        let d_out = max_abs_diff(r_fwd.out.clone(), r_step.out.clone());
        let d_conv_state = max_abs_diff(r_fwd.final_conv.clone(), r_step.final_conv.clone());
        let d_ssm_state = max_abs_diff(r_fwd.final_ssm.clone(), r_step.final_ssm.clone());
        assert!(d_out < val_tol, "step vs forward: output max abs diff = {d_out:.6}");
        assert!(
            d_conv_state < val_tol,
            "step vs forward: final conv window max abs diff = {d_conv_state:.6}"
        );
        assert!(
            d_ssm_state < val_tol,
            "step vs forward: final SSM state max abs diff = {d_ssm_state:.6}"
        );

        // ── Gradient agreement ───────────────────────────────────────────
        // step() and forward() are different reductions of the same SSM, so
        // their per-parameter gradients should also agree, modulo float-
        // summation order noise.
        check_grads_match("step vs forward", &r_fwd, &r_step, 1e-3);

        // ── Guard: the random initial state must actually be consumed ─────
        // Re-run forward from a *zero* initial cache; its output must differ
        // from the random-init output. Otherwise the initial state is being
        // silently ignored and forward/step would match trivially.
        if random_init {
            let (out_zero, _) = model.forward(
                Tensor::from_inner(input.clone()),
                Some(build_init_cache(&cfg, batch, false)),
            );
            let d = max_abs_diff(r_fwd.out.clone(), out_zero.inner());
            assert!(
                d > 1e-3,
                "random initial state appears ignored: random-init vs zero-init \
                 output max abs diff = {d:.6} (expected a clear difference)"
            );
        }
    }

    #[test]
    fn step_matches_forward() {
        run_step_matches_forward(small_config(), false);
    }

    #[test]
    fn step_matches_forward_random_init() {
        run_step_matches_forward(small_config(), true);
    }

    // ── Varying d_state ─────────────────────────────────────────────────────

    fn cfg_d_state_16() -> Mamba1Config {
        Mamba1Config::new(32)
            .with_d_state(16)
            .with_d_conv(4)
            .with_expand(2)
    }

    #[test]
    fn step_matches_forward_d_state_16() {
        run_step_matches_forward(cfg_d_state_16(), false);
    }

    #[test]
    fn step_matches_forward_d_state_16_random_init() {
        run_step_matches_forward(cfg_d_state_16(), true);
    }

    // ── Varying d_conv (causal convolution window) ──────────────────────────

    fn cfg_d_conv_2() -> Mamba1Config {
        Mamba1Config::new(32)
            .with_d_state(8)
            .with_d_conv(2)
            .with_expand(2)
    }

    #[test]
    fn step_matches_forward_d_conv_2() {
        run_step_matches_forward(cfg_d_conv_2(), false);
    }

    #[test]
    fn step_matches_forward_d_conv_2_random_init() {
        run_step_matches_forward(cfg_d_conv_2(), true);
    }

    // ── Varying expand (inner width) ────────────────────────────────────────

    fn cfg_expand_1() -> Mamba1Config {
        Mamba1Config::new(32)
            .with_d_state(8)
            .with_d_conv(4)
            .with_expand(1)
    }

    #[test]
    fn step_matches_forward_expand_1() {
        run_step_matches_forward(cfg_expand_1(), false);
    }

    #[test]
    fn step_matches_forward_expand_1_random_init() {
        run_step_matches_forward(cfg_expand_1(), true);
    }

    // ── Custom dt_rank (Δ projection rank) ──────────────────────────────────

    fn cfg_custom_dt_rank() -> Mamba1Config {
        Mamba1Config::new(32)
            .with_d_state(8)
            .with_d_conv(4)
            .with_expand(2)
            .with_dt_rank(Some(8))
    }

    #[test]
    fn step_matches_forward_custom_dt_rank() {
        run_step_matches_forward(cfg_custom_dt_rank(), false);
    }

    #[test]
    fn step_matches_forward_custom_dt_rank_random_init() {
        run_step_matches_forward(cfg_custom_dt_rank(), true);
    }
}
