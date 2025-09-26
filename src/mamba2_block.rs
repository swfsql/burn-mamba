use crate::rms_norm_gated::{RmsNormGated, RmsNormGatedConfig};
use crate::silu::Silu;
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Mamba2Block<B: Backend> {
    /// Input channel: [`Mamba2BlockConfig::d_model`].
    /// Output channel: z + xbc + dt.
    ///
    /// z: [`Self::d_inner`].
    /// xbc: [`Self::conv_dim`].
    /// dt: [`Self::nheads`].
    pub in_proj: Linear<B>,

    /// Input channel: conv_dim.
    /// Output channel: conv_dim.
    /// Kernel: [`Mamba2BlockConfig::d_conv`].
    /// Padding: [`Mamba2BlockConfig::d_conv`] - 1.
    /// Groups: conv_dim.
    pub conv1d: Conv1d<B>,

    /// Dims: [`Self::nheads`].
    pub dt_bias: Param<Tensor<B, 1>>,

    /// Dims: [`Self::nheads`].
    pub a_log: Param<Tensor<B, 1>>,

    /// Dims: [`Self::nheads`].
    pub d: Param<Tensor<B, 1>>,

    /// Dims: [`Self::d_inner`].
    pub norm: RmsNormGated<B>,

    /// Input channel: [`Self::d_inner`].
    /// Output channel: [`Mamba2BlockConfig::d_model`].
    pub out_proj: Linear<B>,

    /// Dims: [[`Self::nheads`], [`Self::headdim`], [`Mamba2BlockConfig::d_state`]].
    pub init_states: Option<Param<Tensor<B, 3>>>,

    /// [`Mamba2BlockConfig::d_state`].
    pub d_state: usize,

    /// [`Mamba2BlockConfig::ngroups`].
    pub ngroups: usize,
}

impl<B: Backend> Mamba2Block<B> {
    /// d_inner = expand * d_model = nheads * headdim.
    pub fn d_inner(&self) -> usize {
        let [d_inner] = self.norm.gamma.dims();
        d_inner
    }

    /// nheads = d_inner / headdim.
    pub fn nheads(&self) -> usize {
        let [nheads] = self.a_log.dims();
        nheads
    }

    /// headdim = d_inner / n_heads.
    pub fn headdim(&self) -> usize {
        self.d_inner() / self.nheads()
    }

    /// conv_dim = d_inner + 2 * ngroups * d_state.
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
    }
}

#[derive(Config, Debug)]
pub struct Mamba2BlockConfig {
    /// Hidden dimension.
    pub d_model: usize,

    /// latent state dimension.
    #[config(default = 128)]
    pub d_state: usize,

    /// Convolution kernel size.
    #[config(default = 4)]
    pub d_conv: usize,

    /// Expansion factor for `d_inner`.
    #[config(default = 2)]
    pub expand: usize,

    /// Head dimension.
    #[config(default = 64)]
    pub headdim: usize,

    /// Number of groups.
    #[config(default = 1)]
    pub ngroups: usize,

    /// Range for `A` initialization.
    #[config(default = "(1., 16.)")]
    pub a_init_range: (f64, f64),

    #[config(default = false)]
    pub is_norm_before_gate: bool,

    /// Minimum dt value.
    #[config(default = 1e-3)]
    pub dt_min: f64,

    /// Maximum dt value.
    #[config(default = 1e-1)]
    pub dt_max: f64,

    /// Floor for dt initialization.
    #[config(default = 1e-4)]
    pub dt_init_floor: f64,

    /// Range limits for dt.
    #[config(default = "(0., f64::INFINITY)")]
    pub dt_limit: (f64, f64),

    #[config(default = false)]
    pub has_proj_bias: bool,

    #[config(default = true)]
    pub has_conv_bias: bool,

    /// Whether initial states are learnable.
    #[config(default = false)]
    pub has_learnable_init_states: bool,
}

//
impl Mamba2BlockConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2Block<B> {
        let d_inner = self.d_inner();
        assert!(self.headdim > 0);
        let nheads = self.nheads();
        // panic!("{nheads}");
        debug_assert_eq!(nheads * self.headdim, d_inner);

        // Helper function for PyTorch-style uniform initialization
        let uniform_init = |d_input: usize| {
            let bound = 1.0 / (d_input as f64).sqrt();
            Initializer::Uniform {
                min: -bound,
                max: bound,
            }
        };

        // Input projection
        let conv_dim = self.conv_dim();
        let d_in_proj = d_inner + conv_dim + nheads;
        let in_proj = LinearConfig::new(self.d_model, d_in_proj)
            .with_bias(self.has_proj_bias)
            // follows PyTorch's default initializer
            .with_initializer(uniform_init(self.d_model))
            .init::<B>(device);

        // Convolution
        let conv1d = Conv1dConfig::new(conv_dim, conv_dim, self.d_conv)
            // the conv's inputs are padded manually for causality, by self.d_conv - 1
            .with_padding(burn::nn::PaddingConfig1d::Valid)
            .with_groups(conv_dim)
            .with_bias(self.has_conv_bias)
            // follows PyTorch's default initializer
            // fan_in = in_channels / groups * kernel_size
            .with_initializer(uniform_init(self.d_conv))
            .init::<B>(device);

        // dt_bias initialization
        // note: this placeholder impl may lose precision for very small values,
        // and a Taylor series could approximate it: e^x - 1 = x + x^2/2! + x^3/3! + ⋯
        // but with the clamp at dt_init_floor, this isn't necessary
        let expm1 = |t: Tensor<B, 1>| t.exp() - 1.;
        let dt = Tensor::random(
            [nheads],
            burn::tensor::Distribution::Uniform(self.dt_min.ln(), self.dt_max.ln()),
            device,
        )
        .exp();
        let dt = dt.clamp(self.dt_init_floor, f64::MAX);
        let inv_dt = dt.clone() + (-expm1(-dt)).log(); // Inverse softplus
        let dt_bias = Param::from_tensor(inv_dt);

        // A_log initialization
        assert!(self.a_init_range.0 > 0.);
        assert!(self.a_init_range.0 < self.a_init_range.1);
        let a = Tensor::random(
            [nheads],
            burn::tensor::Distribution::Uniform(self.a_init_range.0, self.a_init_range.1),
            device,
        );
        let a_log = Param::from_tensor(a.log());

        // D initialization
        let d = Initializer::Ones.init::<B, 1, _>([nheads], device);

        // Normalization and output projection
        let norm = RmsNormGatedConfig::new(d_inner)
            .with_epsilon(1e-5)
            .with_norm_before_gate(self.is_norm_before_gate)
            .init(device);
        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(self.has_proj_bias)
            // follows PyTorch's default initializer
            .with_initializer(uniform_init(d_inner))
            .init(device);

        // Optional learnable initial states
        let init_states = if self.has_learnable_init_states {
            Some(Initializer::Zeros.init::<B, 3, _>([nheads, self.headdim, self.d_state], device))
        } else {
            None
        };

        Mamba2Block {
            in_proj,
            conv1d,
            dt_bias,
            a_log,
            d,
            norm,
            out_proj,
            init_states,
            d_state: self.d_state,
            ngroups: self.ngroups,
        }
    }

    /// d_inner = expand * d_model.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// nheads = d_inner / headdim.
    pub fn nheads(&self) -> usize {
        // panic!("d_inner: {}, headdim: {}", self.d_inner(), self.headdim);
        self.d_inner() / self.headdim
    }

    /// conv_dim = d_inner + 2 * ngroups * d_state.
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
    }
}

#[derive(Module, Debug)]
pub struct Mamba2BlockCache<B: Backend> {
    /// # Shape
    /// [batch, conv_dim, d_conv]
    pub conv: Param<Tensor<B, 3>>,
    /// # Shape
    /// [batch, nheads, headdim, d_state]
    pub ssm: Param<Tensor<B, 4>>,
}

#[derive(Config, Debug)]
pub struct Mamba2BlockCacheConfig {
    pub batch: usize,
    pub mamba2_block: Mamba2BlockConfig,
}

impl Mamba2BlockCacheConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2BlockCache<B> {
        let config = &self.mamba2_block;
        let conv = Initializer::Zeros.init([self.batch, config.conv_dim(), config.d_conv], device);
        let ssm = Initializer::Zeros.init(
            [self.batch, config.nheads(), config.headdim, config.d_state],
            device,
        );
        Mamba2BlockCache { conv, ssm }
    }
}

impl<B: Backend> Mamba2Block<B> {
    /// See also [`Self::step`].
    ///
    /// `chunk_size`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    /// - Input [batch, sequence, d_model]
    /// - Output.0 value [batch, sequence, d_model]
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        chunk_size: usize,
    ) -> (Tensor<B, 3>, Mamba2BlockCache<B>) {
        let device = &input.device();
        let [batch, _sequence, _d_model] = input.dims();
        let [conv_dim, _, d_conv] = self.conv1d.weight.dims();
        let conv = Initializer::Zeros.init([batch, conv_dim, d_conv], device);
        let ssm =
            Initializer::Zeros.init([batch, self.nheads(), self.headdim(), self.d_state], device);
        self.forward_with_cache(input, Mamba2BlockCache { conv, ssm }, chunk_size)
    }

    /// See also [`Self::step`].
    ///
    /// `chunk_size`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    /// - Input [batch, sequence, d_model]
    /// - Output.0 output [batch, sequence, d_model]
    pub fn forward_with_cache(
        &self,
        input: Tensor<B, 3>,
        mut cache: Mamba2BlockCache<B>,
        chunk_size: usize,
    ) -> (Tensor<B, 3>, Mamba2BlockCache<B>) {
        let [batch, sequence, _d_model] = input.dims();
        let d_inner = self.d_inner();
        let ngroups = self.ngroups;
        let nheads = self.nheads();
        let conv_dim = self.conv_dim();
        let d_state = self.d_state;
        let [_conv_dim, _, d_conv] = self.conv1d.weight.dims();
        let [_d_model, d_in_proj_output] = self.in_proj.weight.dims();

        // input projection
        let (z, xbc, dt) = {
            let z_xbc_dt = self.in_proj.forward(input);
            debug_assert_eq!([batch, sequence, d_in_proj_output], z_xbc_dt.dims());
            debug_assert_eq!(
                [batch, sequence, d_inner + conv_dim + nheads],
                z_xbc_dt.dims()
            );

            let z_xbc_dt = z_xbc_dt.split_with_sizes(vec![d_inner, conv_dim, nheads], 2);
            (
                z_xbc_dt[0].clone(),
                z_xbc_dt[1].clone(),
                z_xbc_dt[2].clone(),
            )
        };
        debug_assert_eq!([batch, sequence, d_inner], z.dims());
        debug_assert_eq!([batch, sequence, conv_dim], xbc.dims());
        debug_assert_eq!([batch, sequence, nheads], dt.dims());

        // convolution and activation
        let xbc = xbc.swap_dims(1, 2);
        debug_assert_eq!([batch, conv_dim, sequence], xbc.dims());
        // split-off oldest/first column (i.e. rolling leftwards) for the causal pad
        let t0 = cache.conv.val().narrow(2, 1, d_conv - 1);
        debug_assert_eq!([batch, conv_dim, d_conv - 1], t0.dims());
        // add the causal pad
        let xbc = Tensor::cat(vec![t0, xbc], 2);
        debug_assert_eq!([batch, conv_dim, (d_conv - 1) + sequence], xbc.dims());
        // get the final state for the causal pad (right-side of pre-xbc)
        cache.conv = Param::from_tensor(xbc.clone().narrow(2, sequence - 1, d_conv));
        debug_assert_eq!([batch, conv_dim, d_conv], cache.conv.dims());
        let xbc = self.conv1d.forward(xbc);
        debug_assert_eq!([batch, conv_dim, sequence + d_conv - 1], xbc.dims());
        let xbc = xbc.narrow(2, 0, sequence); // trim padding
        debug_assert_eq!([batch, conv_dim, sequence], xbc.dims());
        let xbc = xbc.swap_dims(1, 2);
        debug_assert_eq!([batch, sequence, conv_dim], xbc.dims());
        let xbc = Silu::new().forward(xbc);
        debug_assert_eq!([batch, sequence, conv_dim], xbc.dims());

        // split xbc into x, b, c.
        // note: the attention Q,K,V values correspond to c,b,x from ssm/attention duality.
        let (x, b, c) = {
            let xbc = xbc.split_with_sizes(vec![d_inner, ngroups * d_state, ngroups * d_state], 2);
            (xbc[0].clone(), xbc[1].clone(), xbc[2].clone())
        };
        debug_assert_eq!([batch, sequence, d_inner], x.dims());
        debug_assert_eq!([batch, sequence, ngroups * d_state], b.dims());
        debug_assert_eq!([batch, sequence, ngroups * d_state], c.dims());

        // prepare state space parameters
        let dt_bias = self.dt_bias.val().unsqueeze_dims(&[0, 1]);
        debug_assert_eq!([1, 1, nheads], dt_bias.dims());
        let dt = burn::tensor::activation::softplus(dt + dt_bias, 1.);
        debug_assert_eq!([batch, sequence, nheads], dt.dims());
        let a = -self.a_log.val().exp(); // [nheads]
        debug_assert_eq!([nheads], a.dims());

        // Perform chunked selective scan
        let (y, final_state) =
            self.chunked_selective_scan(x, dt, a, b, c, cache.ssm.val(), chunk_size);
        debug_assert_eq!([batch, sequence, d_inner], y.dims());
        cache.ssm = Param::from_tensor(final_state);

        // Normalization
        let y = self.norm.forward(y, z);
        debug_assert_eq!([batch, sequence, d_inner], y.dims());

        // Output projection
        let out = self.out_proj.forward(y);
        debug_assert_eq!([batch, sequence, _d_model], out.dims());
        (out, cache)
    }

    /// Performs a chunked selective scan for Mamba2.
    ///
    /// # Shapes
    /// - Input x [batch, sequence, d_inner]
    /// - Input dt [batch, sequence, nheads]
    /// - Input a [nheads]
    /// - Input b [batch, sequence, ngroups * d_state]
    /// - Input c [batch, sequence, ngroups * d_state]
    /// - Input ssm_initial_state [batch, nheads, headdim, d_state]
    /// - Output.0 y [batch, sequence, d_inner]
    /// - Output.1 ssm_final_state [batch, nheads, headdim, d_state]
    pub fn chunked_selective_scan(
        &self,
        x: Tensor<B, 3>,
        dt: Tensor<B, 3>,
        a: Tensor<B, 1>,
        b: Tensor<B, 3>,
        c: Tensor<B, 3>,
        ssm_initial_state: Tensor<B, 4>,
        chunk_size: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let [batch, sequence, d_inner] = x.dims();
        let nheads = self.nheads();
        let headdim = self.headdim();
        let ngroups = self.ngroups;
        let d_state = self.d_state;
        let num_full_chunks = sequence / chunk_size;
        let remainder_chunk_size = sequence % chunk_size;
        let sequence_full_chunks = num_full_chunks * chunk_size;
        debug_assert_eq!(sequence, sequence_full_chunks + remainder_chunk_size);
        debug_assert_eq!(d_inner, nheads * headdim);

        // reshapes
        let x = x.reshape([batch, sequence, nheads, headdim]);
        let b = b.reshape([batch, sequence, ngroups, d_state]);
        let c = c.reshape([batch, sequence, ngroups, d_state]);

        // expand b and c
        let heads_per_group = nheads / ngroups;
        let b = b
            .unsqueeze_dim::<5>(3)
            .expand([batch, sequence, ngroups, heads_per_group, d_state])
            .reshape([batch, sequence, nheads, d_state]);
        let c = c
            .unsqueeze_dim::<5>(3)
            .expand([batch, sequence, ngroups, heads_per_group, d_state])
            .reshape([batch, sequence, nheads, d_state]);

        let state: Tensor<B, 4> = ssm_initial_state; // cache.ssm shape
        debug_assert_eq!([batch, nheads, headdim, d_state], state.dims());
        let mut y_chunks =
            Vec::with_capacity(num_full_chunks + if remainder_chunk_size == 0 { 0 } else { 1 });
        let mut current_state = state;
        for chunk in 0..num_full_chunks {
            let x_chunk = x
                .clone()
                .narrow(1, chunk * chunk_size, chunk_size)
                .reshape([batch, chunk_size, nheads, headdim]);
            let dt_chunk = dt
                .clone()
                .narrow(1, chunk * chunk_size, chunk_size)
                .reshape([batch, chunk_size, nheads]);
            let b_chunk = b
                .clone()
                .narrow(1, chunk * chunk_size, chunk_size)
                .reshape([batch, chunk_size, nheads, d_state]);
            let c_chunk = c
                .clone()
                .narrow(1, chunk * chunk_size, chunk_size)
                .reshape([batch, chunk_size, nheads, d_state]);

            let (y_chunk, new_state) = scan_chunk(
                x_chunk,
                dt_chunk,
                a.clone(),
                b_chunk,
                c_chunk,
                self.d.val(),
                current_state,
            );
            debug_assert_eq!([batch, chunk_size, nheads, headdim], y_chunk.dims());
            debug_assert_eq!([batch, nheads, headdim, d_state], new_state.dims());

            y_chunks.push(y_chunk);
            current_state = new_state;
        }

        if remainder_chunk_size != 0 {
            let x_chunk = x
                .clone()
                .narrow(1, num_full_chunks * chunk_size, remainder_chunk_size)
                .reshape([batch, remainder_chunk_size, nheads, headdim]);
            let dt_chunk = dt
                .clone()
                .narrow(1, num_full_chunks * chunk_size, remainder_chunk_size)
                .reshape([batch, remainder_chunk_size, nheads]);
            let b_chunk = b
                .clone()
                .narrow(1, num_full_chunks * chunk_size, remainder_chunk_size)
                .reshape([batch, remainder_chunk_size, nheads, d_state]);
            let c_chunk = c
                .clone()
                .narrow(1, num_full_chunks * chunk_size, remainder_chunk_size)
                .reshape([batch, remainder_chunk_size, nheads, d_state]);

            let (y_chunk, new_state) = scan_chunk(
                x_chunk,
                dt_chunk,
                a.clone(),
                b_chunk,
                c_chunk,
                self.d.val(),
                current_state,
            );
            debug_assert_eq!(
                [batch, remainder_chunk_size, nheads, headdim],
                y_chunk.dims()
            );
            debug_assert_eq!([batch, nheads, headdim, d_state], new_state.dims());

            y_chunks.push(y_chunk);
            current_state = new_state;
        }

        let y = Tensor::cat(y_chunks, 1);
        debug_assert_eq!(
            [
                batch,
                num_full_chunks * chunk_size + remainder_chunk_size,
                nheads,
                headdim
            ],
            y.dims()
        );

        let y = y.reshape([
            batch,
            num_full_chunks * chunk_size + remainder_chunk_size,
            nheads * headdim,
        ]);
        debug_assert_eq!([batch, sequence, d_inner], y.dims());
        (y, current_state)
    }
}

/// Performs a selective scan over a single chunk for Mamba2.
///
/// # Shapes
/// - Input `x_chunk`: [batch, chunk_size, nheads, headdim]
/// - Input `dt_chunk`: [batch, chunk_size, nheads]
/// - Input `a`: [nheads]
/// - Input `b_chunk`: [batch, chunk_size, nheads, d_state]
/// - Input `c_chunk`: [batch, chunk_size, nheads, d_state]
/// - Input `d`: [nheads]
/// - Input `initial_state`: [batch, nheads, headdim, d_state]
/// - Output.0 `y`: [batch, chunk_size, nheads, headdim]
/// - Output.1 `final_state`: [batch, nheads, headdim, d_state]
fn scan_chunk<B: Backend>(
    x_chunk: Tensor<B, 4>,
    dt_chunk: Tensor<B, 3>,
    a: Tensor<B, 1>,
    b_chunk: Tensor<B, 4>,
    c_chunk: Tensor<B, 4>,
    d: Tensor<B, 1>,
    initial_state: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, chunk_size, nheads, headdim] = x_chunk.dims();
    let [_batch, _chunk_size, _nheads, d_state] = b_chunk.dims();

    let mut state = initial_state;
    let mut y_list = Vec::with_capacity(chunk_size);
    for t in 0..chunk_size {
        let x_t = x_chunk.clone().narrow(1, t, 1).squeeze(1);
        let dt_t = dt_chunk.clone().narrow(1, t, 1).squeeze(1);
        let b_t = b_chunk.clone().narrow(1, t, 1).squeeze(1);
        let c_t = c_chunk.clone().narrow(1, t, 1).squeeze(1);
        debug_assert_eq!([batch, nheads, headdim], x_t.dims());
        debug_assert_eq!([batch, nheads], dt_t.dims());
        debug_assert_eq!([batch, nheads, d_state], b_t.dims());
        debug_assert_eq!([batch, nheads, d_state], c_t.dims());

        let delta_a = (dt_t.clone() * a.clone().unsqueeze())
            .exp()
            .unsqueeze_dims(&[2, 3]);
        let delta_bu =
            (dt_t.unsqueeze_dim(2) * b_t).unsqueeze_dim(2) * x_t.clone().unsqueeze_dim(3);
        debug_assert_eq!([batch, nheads, 1, 1], delta_a.dims());
        debug_assert_eq!([batch, nheads, headdim, d_state], delta_bu.dims());

        state = state * delta_a + delta_bu;
        debug_assert_eq!([batch, nheads, headdim, d_state], state.dims());

        let y_t = (state.clone() * c_t.unsqueeze_dim(2)).sum_dim(3).squeeze(3);
        debug_assert_eq!([batch, nheads, headdim], y_t.dims());
        let y_t = y_t + d.clone().unsqueeze_dims(&[0, 2]) * x_t;
        debug_assert_eq!([batch, nheads, headdim], y_t.dims());
        y_list.push(y_t);
    }
    debug_assert_eq!([batch, nheads, headdim, d_state], state.dims());

    let y_chunk = Tensor::stack(y_list, 1);
    debug_assert_eq!([batch, chunk_size, nheads, headdim], y_chunk.dims());

    (y_chunk, state)
}

pub mod step {
    use super::*;

    impl<B: Backend> Mamba2Block<B> {
        /// # Shapes
        ///   - Input [batch, d_model]
        ///   - Output [batch, d_model]
        //
        /// Single-step inference for decoding one token at a time.
        ///
        /// # Shapes
        /// - Input: [batch, d_model]
        /// - Output: [batch, d_model]
        pub fn step(
            &self,
            input: Tensor<B, 2>,
            mut cache: Mamba2BlockCache<B>,
        ) -> (Tensor<B, 2>, Mamba2BlockCache<B>) {
            let [batch, d_model] = input.dims();
            let d_inner = self.d_inner();
            let ngroups = self.ngroups;
            let nheads = self.nheads();
            let headdim = self.headdim();
            let conv_dim = self.conv_dim();
            let d_state = self.d_state;
            let [_conv_dim, _, d_conv] = self.conv1d.weight.dims();

            // Input projection
            let z_xbc_dt = self.in_proj.forward(input);
            debug_assert_eq!([batch, d_inner + conv_dim + nheads], z_xbc_dt.dims());
            let z_xbc_dt = z_xbc_dt;

            let (z, xbc, dt) = {
                let split = z_xbc_dt.split_with_sizes(vec![d_inner, conv_dim, nheads], 1);
                (split[0].clone(), split[1].clone(), split[2].clone())
            };
            debug_assert_eq!([batch, d_inner], z.dims());
            debug_assert_eq!([batch, conv_dim], xbc.dims());
            debug_assert_eq!([batch, nheads], dt.dims());

            // Convolution step
            cache.conv = cache.conv.map(|conv| {
                debug_assert_eq!([batch, conv_dim, d_conv], conv.dims());

                // split-off oldest/first column (i.e. rolling leftwards)
                let t0 = conv.narrow(2, 1, d_conv - 1);
                debug_assert_eq!([batch, conv_dim, d_conv - 1], t0.dims());

                // insert xbc as a the newest/last column
                let conv = Tensor::cat([t0, xbc.unsqueeze_dim(2)].to_vec(), 2);
                debug_assert_eq!([batch, conv_dim, d_conv], conv.dims());

                conv
            });

            let xbc = {
                let conv1d = self.conv1d.weight.val();
                // [channels_out, channels_in / groups, kernel_size]
                debug_assert_eq!([conv_dim, 1, d_conv], conv1d.dims());
                let conv1d = conv1d.swap_dims(0, 1);
                debug_assert_eq!([1, conv_dim, d_conv], conv1d.dims());
                let conv1d = conv1d.expand([batch, conv_dim, d_conv]);
                debug_assert_eq!([batch, conv_dim, d_conv], conv1d.dims());

                let xbc = cache.conv.val() * conv1d;
                debug_assert_eq!([batch, conv_dim, d_conv], xbc.dims());
                let mut xbc = xbc.sum_dim(2).squeeze(2);
                debug_assert_eq!([batch, conv_dim], xbc.dims());
                if let Some(bias) = &self.conv1d.bias {
                    debug_assert_eq!([conv_dim], bias.dims());
                    xbc = xbc + bias.val().unsqueeze();
                }
                Silu::new().forward(xbc)
            };
            debug_assert_eq!([batch, conv_dim], xbc.dims());

            // Split xbc
            let (x, b, c) = {
                let split =
                    xbc.split_with_sizes(vec![d_inner, ngroups * d_state, ngroups * d_state], 1);
                (split[0].clone(), split[1].clone(), split[2].clone())
            };
            debug_assert_eq!([batch, d_inner], x.dims());
            debug_assert_eq!([batch, ngroups * d_state], b.dims());
            debug_assert_eq!([batch, ngroups * d_state], c.dims());

            // SSM step
            let ssm_shape = [batch, nheads, headdim, d_state]; // cache.ssm shape
            let dt =
                burn::tensor::activation::softplus(dt + self.dt_bias.val().unsqueeze_dim(0), 1.);
            debug_assert_eq!([batch, nheads], dt.dims());
            let a = -self.a_log.val().exp();
            debug_assert_eq!([nheads], a.dims());

            let x = x.reshape([batch, nheads, headdim]); // d_inner = nheads * headdim
            let b = b.reshape([batch, ngroups, d_state]);
            let c = c.reshape([batch, ngroups, d_state]);

            // dt * a = [batch, nheads] * [nheads]
            let dta = (dt.clone() * a.clone().unsqueeze()).exp();
            debug_assert_eq!([batch, nheads], dta.dims());

            let dta = dta.unsqueeze_dims(&[2, 3]);
            debug_assert_eq!([batch, nheads, 1, 1], dta.dims());
            let dta = dta.expand(ssm_shape);

            // dt * b * x = [batch, nheads] * [batch, ngroups, d_state] * [batch, nheads, headdim]
            let heads_per_group = nheads / ngroups;
            let dtbx = {
                let x = x.clone().unsqueeze_dim(3);
                debug_assert_eq!([batch, nheads, headdim, 1], x.dims());
                let x = x.expand(ssm_shape);

                let b = b.unsqueeze_dims(&[1, 4]);
                debug_assert_eq!([batch, 1, ngroups, d_state, 1], b.dims());
                let b = b
                    .expand([batch, heads_per_group, ngroups, d_state, 1])
                    .reshape([batch, nheads, d_state, 1])
                    .swap_dims(2, 3)
                    .expand(ssm_shape);

                let dt = dt.unsqueeze_dims(&[2, 3]);
                debug_assert_eq!([batch, nheads, 1, 1], dt.dims());
                let dt = dt.expand(ssm_shape);

                dt * b * x
            };
            debug_assert_eq!(ssm_shape, dtbx.dims());

            let c = c.unsqueeze_dims(&[1, 4]);
            debug_assert_eq!([batch, 1, ngroups, d_state, 1], c.dims());
            let c = c
                .expand([batch, heads_per_group, ngroups, d_state, 1])
                .reshape([batch, nheads, d_state, 1])
                .swap_dims(2, 3)
                .expand(ssm_shape);
            let d = self.d.val().unsqueeze_dims(&[0, 2]);
            debug_assert_eq!([1, nheads, 1], d.dims());
            let d = d.expand([batch, nheads, headdim]);
            //
            debug_assert_eq!(ssm_shape, c.dims());

            // Compute state update: state = state * exp(dt * A) + dt * B * x
            cache.ssm = cache.ssm.map(|ssm| ssm * dta + dtbx);
            debug_assert_eq!(ssm_shape, cache.ssm.dims());

            // Compute output: y = C * state + D * x
            let y = {
                let y = cache.ssm.val() * c;
                let y = y.sum_dim(3).squeeze(3);
                debug_assert_eq!([batch, nheads, headdim], y.dims());

                let y = y + d * x;
                debug_assert_eq!([batch, nheads, headdim], y.dims());

                // Apply normalization with z
                let y_flat = y.reshape([batch, d_inner]);

                let y = self.norm.forward(y_flat, z);
                y
            };
            debug_assert_eq!([batch, d_inner], y.dims());

            // Output projection
            let out = self.out_proj.forward(y);
            debug_assert_eq!([batch, d_model], out.dims());

            (out, cache)
        }
    }
}
