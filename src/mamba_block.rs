use crate::silu::Silu;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Initializer, PaddingConfig1d};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct MambaBlock<B: Backend> {
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

    /// Dims: [d_inner, d_state].
    pub a_log: Param<Tensor<B, 2>>,

    /// Dims: [d_inner].
    pub d: Param<Tensor<B, 1>>,

    /// Input channel: d_inner.
    /// Output channel: d_model.
    pub out_proj: Linear<B>,
}

#[derive(Config, Debug)]
pub struct MambaBlockConfig {
    /// Hidden dimension.
    pub d_model: usize,

    /// latent state dimension (`N` in Algorithm 2 from the Mamba paper).
    #[config(default = 16)]
    pub d_state: usize,

    /// Rank of Δ (See Section 3.6 "Parameterization of ∆" from the Mamba paper).
    /// Δ or delta: input-dependent step size.
    ///
    /// By default, set to (d_model + d_state - 1) / d_state.
    pub dt_rank: Option<usize>,

    #[config(default = 4)]
    pub d_conv: usize,

    /// DModel * expand (`D` in Algorithm 2 from the Mamba paper).
    ///
    /// By default, set to 2 * d_model.
    pub d_inner: Option<usize>,
}

impl MambaBlockConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MambaBlock<B> {
        let d_inner = self.d_inner();
        assert_ne!(self.d_state, 0);
        assert!(self.d_model + self.d_state > 0);
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
                let dt_scale = 1.0;
                let dt_init_std = (dt_rank as f64).powf(-0.5) * dt_scale;
                Tensor::random(
                    [dt_rank, d_inner],
                    Distribution::Uniform(-dt_init_std, dt_init_std),
                    device,
                )
            };
            assert_eq!([dt_rank, d_inner], weight.dims());
            let bias: Tensor<B, 1> = {
                let dt_min = 0.001;
                let dt_max = 0.1;
                let dt_init_floor = 1e-4;
                // note: this placeholder impl may lose precision for very small values,
                // and a Taylor series could approximate it: e^x - 1 = x + x^2/2! + x^3/3! + ⋯
                // but with the clamp at dt_init_floor, this isn't necessary
                let expm1 = |t: Tensor<B, 1>| t.exp() - 1.;
                let dt = Tensor::random([d_inner], Distribution::Uniform(0.0, 1.0), device)
                    * (f32::ln(dt_max) - f32::ln(dt_min))
                    + f32::ln(dt_min);
                let dt = dt.exp().clamp_min(dt_init_floor);
                // Inverse of softplus
                let inv_dt = dt.clone() + (-expm1(-dt)).log();
                inv_dt
            };
            assert_eq!([d_inner], bias.dims());
            Linear {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        };

        let a_log = {
            let a_row: Tensor<B, 1> =
                Tensor::<B, 1, Int>::arange(1..self.d_state as i64 + 1, device).float();
            assert_eq!([self.d_state], a_row.dims());
            let a_row = a_row.unsqueeze();
            assert_eq!([1, self.d_state], a_row.dims());
            let a = a_row.repeat(&[d_inner, 1]);
            assert_eq!([d_inner, self.d_state], a.dims());
            let a_log = a.log();
            Param::from_tensor(a_log)
        };

        MambaBlock {
            in_proj: LinearConfig::new(self.d_model, 2 * d_inner)
                .with_bias(false)
                // follows PyTorch's default initializer
                .with_initializer(uniform_init(self.d_model))
                .init(device),
            conv1d: Conv1dConfig::new(d_inner, d_inner, self.d_conv)
                .with_padding(PaddingConfig1d::Explicit(self.d_conv - 1))
                .with_groups(d_inner)
                .with_bias(true)
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
                .with_bias(false)
                // follows PyTorch's default initializer
                .with_initializer(uniform_init(d_inner))
                .init(device),
        }
    }
    pub fn d_inner(&self) -> usize {
        self.d_inner.unwrap_or(2 * self.d_model)
    }
    pub fn dt_rank(&self) -> usize {
        self.dt_rank
            .unwrap_or((self.d_model + self.d_state - 1) / self.d_state)
    }
}

impl<B: Backend> MambaBlock<B> {
    /// See also [`Self::step`].
    ///
    /// # Shapes
    ///   - Input [batch, sequence, d_model]
    ///   - Output [batch, sequence, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, sequence, d_model] = x.dims();
        let [d_inner] = self.d.dims();
        let [_, _, d_conv] = self.conv1d.weight.dims();

        // layer 1 (in_proj)
        let (xs, res) = {
            // projects the input d_model into 2 * d_inner
            let xs_and_res = self.in_proj.forward(x);
            assert_eq!([batch, sequence, 2 * d_inner], xs_and_res.dims());

            let split = xs_and_res.split_with_sizes(vec![d_inner, d_inner], 2);
            assert_eq!(split.len(), 2);
            (split[0].clone(), split[1].clone())
        };
        assert_eq!([batch, sequence, d_inner], xs.dims());
        assert_eq!([batch, sequence, d_inner], res.dims());

        // layer 2 (conv1d)
        let xs = {
            let xs = xs.movedim(1, 2);
            assert_eq!([batch, d_inner, sequence], xs.dims());

            assert!(d_conv > 0);
            let xs = self.conv1d.forward(xs);
            assert_eq!([batch, d_inner, sequence + d_conv - 1], xs.dims());

            let xs = xs.narrow(2, 0, sequence);
            assert_eq!([batch, d_inner, sequence], xs.dims());

            // restore original positioning as per before the layer 2
            let xs = xs.movedim(1, 2);
            assert_eq!([batch, sequence, d_inner], xs.dims());

            // activation
            let xs = Silu::new().forward(xs);
            assert_eq!([batch, sequence, d_inner], xs.dims());

            xs
        };
        assert_eq!([batch, sequence, d_inner], xs.dims());

        let ss = self.ss(xs);
        assert_eq!([batch, sequence, d_inner], ss.dims());

        // activation
        let ys = ss * Silu::new().forward(res);
        assert_eq!([batch, sequence, d_inner], ys.dims());

        let y = self.out_proj.forward(ys);
        assert_eq!([batch, sequence, d_model], y.dims());

        y
    }

    /// # Shapes
    ///   - Input [batch, sequence, d_inner]
    ///   - Output [batch, sequence, d_inner]
    pub fn ss(&self, u: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, sequence, d_inner] = u.dims();
        let [_d_inner, d_state] = self.a_log.dims();
        let [dt_rank, _d_inner] = self.dt_proj.weight.dims();

        // Compute ∆ A B C D, the state space parameters.

        // A
        // this is input independent (see Section 3.5.2 "Interpretation of A" form the Mamba paper for why A isn't selective)
        let a = self.a_log.val().exp().neg();
        assert_eq!([d_inner, d_state], a.dims());

        let x_dbl = self.x_proj.forward(u.clone());
        assert_eq!([batch, sequence, dt_rank + 2 * d_state], x_dbl.dims());

        // ∆ (part 1/2)
        // ∆ is input-dependent
        // B and C are input-dependent
        let split = x_dbl.split_with_sizes(vec![dt_rank, d_state, d_state], 2);
        let delta = split[0].clone();
        let b = split[1].clone();
        let c = split[2].clone();
        assert_eq!([batch, sequence, dt_rank], delta.dims());
        assert_eq!([batch, sequence, d_state], b.dims());
        assert_eq!([batch, sequence, d_state], c.dims());

        // ∆ (part 2/2)
        // ∆ is input-dependent
        let delta = self.dt_proj.forward(delta);
        assert_eq!([batch, sequence, d_inner], delta.dims());

        let delta = burn::tensor::activation::softplus(delta, 1.);

        let delta = delta.movedim(0, 1);
        assert_eq!([sequence, batch, d_inner], delta.dims());

        let c = c.movedim(0, 1);
        assert_eq!([sequence, batch, d_state], c.dims());

        Self::selective_scan(delta, a, b, c, self.d.val(), u)
    }

    /// Selective Scan.
    ///
    /// Does selective scan algorithm. See:
    /// - Section 2 State Space Models from the Mamba paper;
    /// - Algorithm 2 in Section 3.2 from the Mamba paper;
    /// - run_SSM(A, B, C, u) from The Annotated S4.
    ///
    /// # Shapes
    ///   - Input delta [sequence, batch, d_inner]
    ///   - Input a [d_inner, d_state]
    ///   - Input b [batch, sequence, d_state]
    ///   - Input c [sequence, batch, d_state]
    ///   - Input d [d_inner]
    ///   - Input u [batch, sequence, d_inner]
    ///   - Output [batch, sequence, d_inner]
    pub fn selective_scan(
        delta: Tensor<B, 3>,
        a: Tensor<B, 2>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        d: Tensor<B, 1>,
        u: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let device = &u.device();
        let [sequence, batch, d_inner] = delta.dims();
        let [_d_inner, d_state] = a.dims();
        let outer_shape = [sequence, batch, d_inner, d_state];

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
            assert_eq!([1, 1, d_inner, d_state], a.dims());
            let a = a.expand(outer_shape);
            assert_eq!(outer_shape, a.dims());
            let delta_a = (delta.clone() * a).exp();
            assert_eq!(outer_shape, delta_a.dims());

            let b = b.movedim(1, 0);
            assert_eq!([sequence, batch, d_state], b.dims());
            let b = b.unsqueeze_dim(2);
            assert_eq!([sequence, batch, 1, d_state], b.dims());
            let b = b.expand(outer_shape);
            assert_eq!(outer_shape, b.dims());
            let delta_b = delta * b;
            assert_eq!(outer_shape, delta_b.dims());

            let u = u.clone().movedim(0, 1);
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
        assert_eq!([sequence, batch, d_state, 1], c.dims());
        let c = c.split(1, 0);
        assert_eq!(c.len(), sequence);

        let inner_shape = [batch, d_inner, d_state];
        let mut xs: Tensor<B, 3> = Tensor::zeros(inner_shape, device);
        let mut ys = Vec::with_capacity(sequence); // inner shape: [batch, d_inner]
        for ((delta_a, delta_bu), c) in delta_a
            .into_iter()
            .zip(delta_bu.into_iter())
            .zip(c.into_iter())
        {
            let delta_a = delta_a.squeeze(0);
            assert_eq!(inner_shape, delta_a.dims());
            let delta_bu = delta_bu.squeeze(0);
            assert_eq!(inner_shape, delta_bu.dims());
            let c = c.squeeze(0);
            assert_eq!([batch, d_state, 1], c.dims());

            xs = (xs.clone() * delta_a) + delta_bu;
            let y = xs.clone().matmul(c);
            assert_eq!([batch, d_inner, 1], y.dims());
            let y = y.squeeze(2);
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

        ys
    }
}

pub mod step {
    use super::*;

    #[derive(Module, Debug)]
    pub struct MambaBlockCache<B: Backend> {
        /// # Shape
        /// [batch, d_inner, d_conv]
        pub conv: Param<Tensor<B, 3>>,
        /// # Shape
        /// [batch, d_inner, d_state]
        pub ssm: Param<Tensor<B, 3>>,
    }

    #[derive(Config, Debug)]
    pub struct MambaBlockCacheConfig {
        pub batch: usize,
        pub mamba_block: MambaBlockConfig,
    }

    impl MambaBlockCacheConfig {
        /// Returns the initialized model.
        pub fn init<B: Backend>(&self, device: &B::Device) -> MambaBlockCache<B> {
            let d_inner = self.mamba_block.d_inner();
            let conv =
                Initializer::Zeros.init([self.batch, d_inner, self.mamba_block.d_conv], device);
            let ssm =
                Initializer::Zeros.init([self.batch, d_inner, self.mamba_block.d_state], device);
            MambaBlockCache { conv, ssm }
        }
    }

    impl<B: Backend> MambaBlock<B> {
        /// # Shapes
        ///   - Input [batch, d_model]
        ///   - Output [batch, d_model]
        pub fn step(
            &self,
            x: Tensor<B, 2>,
            mut cache: MambaBlockCache<B>,
        ) -> (Tensor<B, 2>, MambaBlockCache<B>) {
            let [batch, d_inner, d_conv] = cache.conv.dims();
            let [_batch, d_model] = x.dims();

            // layer 1 (in_proj)
            let (xs, res) = {
                // projects the input d_model into 2 * d_inner
                let xs_and_res = self.in_proj.forward(x);
                assert_eq!([batch, 2 * d_inner], xs_and_res.dims());

                let split = xs_and_res.split_with_sizes(vec![d_inner, d_inner], 1);
                (split[0].clone(), split[1].clone())
            };
            assert_eq!([batch, d_inner], xs.dims());
            assert_eq!([batch, d_inner], res.dims());

            // layer 2 (conv1d)
            cache.conv = cache.conv.map(|conv| {
                // split-off oldest/first column (i.e. rolling leftwards)
                let t0 = conv.narrow(2, 1, d_conv - 1);
                assert_eq!([batch, d_inner, d_conv - 1], t0.dims());

                // insert xs as a the newest/last column
                let conv = Tensor::cat([t0, xs.unsqueeze_dim(2)].to_vec(), 2);
                assert_eq!([batch, d_inner, d_conv], conv.dims());

                conv
            });
            let xs = {
                let conv1d = self.conv1d.weight.val();
                // [channels_out, channels_in / groups, kernel_size]
                assert_eq!([d_inner, 1, d_conv], conv1d.dims());
                let conv1d = conv1d.movedim(1, 0);
                assert_eq!([1, d_inner, d_conv], conv1d.dims());
                let conv1d = conv1d.expand([batch, d_inner, d_conv]);
                assert_eq!([batch, d_inner, d_conv], conv1d.dims());

                let xs = cache.conv.val() * conv1d;
                let xs = xs.sum_dim(2);
                assert_eq!([batch, d_inner, 1], xs.dims());
                let xs = xs.squeeze(2);
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
            assert_eq!([batch, d_inner], xs.dims());

            let (ss, cache) = self.ss_step(xs, cache);
            assert_eq!([batch, d_inner], ss.dims());

            // activation
            let ys = ss * Silu::new().forward(res);
            assert_eq!([batch, d_inner], ys.dims());

            let y = self.out_proj.forward(ys);
            assert_eq!([batch, d_model], y.dims());

            (y, cache)
        }

        /// Runs the SSM. See:
        /// - Algorithm 2 in Section 3.2 from the Mamba paper;
        /// - run_SSM(A, B, C, u) from The Annotated S4.
        ///
        /// # Shapes
        ///   - Input u [batch, d_inner]
        ///   - Output [batch, d_inner]
        pub fn ss_step(
            &self,
            u: Tensor<B, 2>,
            cache: MambaBlockCache<B>,
        ) -> (Tensor<B, 2>, MambaBlockCache<B>) {
            let [batch, d_inner, d_state] = cache.ssm.dims();
            let [dt_rank, _d_inner] = self.dt_proj.weight.dims();

            // Compute ∆ A B C D, the state space parameters.

            // A
            // this is input independent (see Section 3.5.2 "Interpretation of A" form the Mamba paper for why A isn't selective)
            let a = self.a_log.val().exp().neg();
            assert_eq!([d_inner, d_state], a.dims());

            let x_dbl = self.x_proj.forward(u.clone());
            assert_eq!([batch, dt_rank + 2 * d_state], x_dbl.dims());

            // ∆ (part 1/2)
            // ∆ is input-dependent
            // B and C are input-dependent
            let split = x_dbl.split_with_sizes(vec![dt_rank, d_state, d_state], 1);
            let delta = split[0].clone();
            let b = split[1].clone();
            let c = split[2].clone();
            assert_eq!([batch, dt_rank], delta.dims());
            assert_eq!([batch, d_state], b.dims());
            assert_eq!([batch, d_state], c.dims());

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
        ///   - Input delta [batch, d_inner]
        ///   - Input a [d_inner, d_state]
        ///   - Input b [batch, d_state]
        ///   - Input c [batch, d_state]
        ///   - Input d [d_inner]
        ///   - Input u [batch, d_inner]
        ///   - Output [batch, d_inner]
        pub fn selective_scan_step(
            delta: Tensor<B, 2>,
            a: Tensor<B, 2>,
            b: Tensor<B, 2>,
            c: Tensor<B, 2>,
            d: Tensor<B, 1>,
            u: Tensor<B, 2>,
            mut cache: MambaBlockCache<B>,
        ) -> (Tensor<B, 2>, MambaBlockCache<B>) {
            let [batch, d_inner, d_state] = cache.ssm.dims();
            let outer_shape = [batch, d_inner, d_state];

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
                assert_eq!([1, d_inner, d_state], a.dims());
                let a = a.expand(outer_shape);
                assert_eq!(outer_shape, a.dims());
                let delta_a = (delta.clone() * a).exp();
                assert_eq!(outer_shape, delta_a.dims());

                let b = b.unsqueeze_dim(1);
                assert_eq!([batch, 1, d_state], b.dims());
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

            cache.ssm = cache.ssm.map(|ssm| (ssm * delta_a) + delta_bu);

            let c = c.unsqueeze_dim(2);
            assert_eq!([batch, d_state, 1], c.dims());

            let y = cache.ssm.val().matmul(c);
            assert_eq!([batch, d_inner, 1], y.dims());
            let y = y.squeeze(2);
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
