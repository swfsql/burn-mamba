use crate::rms_norm_gated::{RmsNormGated, RmsNormGatedConfig};
use crate::silu::Silu;
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Mamba-2 SSM recurrence:
/// - hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ
/// - yₜ = Cₜ hₜ + D xₜ
///
/// Where:
/// - Āₜ = exp(Δₜ A)
/// - B̄ₜ = Δₜ Bₜ
/// - Δₜ = softplus(dtₜ + dt_bias)
/// - A = -exp(A_log)  (scalar per head, fixed)
/// - Bₜ, Cₜ  (time-varying per step)
/// - D  (skip parameter, per head)
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
    #[config(default = 0.1)]
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

        // A_log initialization for Āₜ = exp(Δₜ A)
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
    pub conv: Tensor<B, 3>,
    /// # Shape
    /// [batch, nheads, headdim, d_state]
    pub ssm: Tensor<B, 4>,
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
        let conv = Tensor::zeros(
            Shape::new([self.batch, config.conv_dim(), config.d_conv]),
            device,
        );
        let ssm = Tensor::zeros(
            Shape::new([self.batch, config.nheads(), config.headdim, config.d_state]),
            device,
        );
        Mamba2BlockCache { conv, ssm }
    }
}

/// Stable segment sum computation for creating the 1-semi-separable mask L = exp(segsum(A_input)).
///
/// - x: [..., T] (e.g., A_input per chunk)
/// - Returns: [..., T, T] where out[..., i, j] = sum_{k=j+1 to i} x[..., k] for i >= j, -inf otherwise.
///
/// This avoids numerical instability by using masked differences of cumsums.
fn segsum<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, { D + 1 }> {
    let x_cumsum = x.cumsum(D - 1);

    let x_cumsum_u = x_cumsum.clone().unsqueeze_dim(D);
    let x_cumsum_v = x_cumsum.unsqueeze_dim(D - 1);
    let diff = x_cumsum_u - x_cumsum_v;

    let neg_inf = Tensor::full_like(&diff, f32::NEG_INFINITY);
    let upper_mask = neg_inf.triu(1);

    let x_segsum = diff + upper_mask;

    x_segsum
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
        let conv = Tensor::zeros(Shape::new([batch, conv_dim, d_conv]), device);
        let ssm = Tensor::zeros(
            Shape::new([batch, self.nheads(), self.headdim(), self.d_state]),
            device,
        );
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
        let headdim = self.headdim();
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

            let mut z_xbc_dt = z_xbc_dt.split_with_sizes(vec![d_inner, conv_dim, nheads], 2).into_iter();
            (
                z_xbc_dt.next().unwrap(),
                z_xbc_dt.next().unwrap(),
                z_xbc_dt.next().unwrap(),
            )
        };
        debug_assert_eq!([batch, sequence, d_inner], z.dims());
        debug_assert_eq!([batch, sequence, conv_dim], xbc.dims());
        debug_assert_eq!([batch, sequence, nheads], dt.dims());

        // convolution and activation
        let xbc = xbc.swap_dims(1, 2);
        debug_assert_eq!([batch, conv_dim, sequence], xbc.dims());
        // split-off oldest/first column (i.e. rolling leftwards) for the causal pad
        let t0 = cache.conv.narrow(2, 1, d_conv - 1);
        debug_assert_eq!([batch, conv_dim, d_conv - 1], t0.dims());
        // add the causal pad (only left-pad is necessary)
        let xbc = Tensor::cat(vec![t0, xbc], 2);
        debug_assert_eq!([batch, conv_dim, (d_conv - 1) + sequence], xbc.dims());
        // get the final state for the causal pad (right-side of pre-xbc)
        cache.conv = xbc.clone().narrow(2, sequence - 1, d_conv);
        debug_assert_eq!([batch, conv_dim, d_conv], cache.conv.dims());
        let xbc = self.conv1d.forward(xbc);
        debug_assert_eq!([batch, conv_dim, sequence], xbc.dims());
        let xbc = xbc.swap_dims(1, 2);
        debug_assert_eq!([batch, sequence, conv_dim], xbc.dims());
        let xbc = Silu::new().forward(xbc);
        debug_assert_eq!([batch, sequence, conv_dim], xbc.dims());

        // split xbc into x, B, C.
        // note: the attention Q,K,V values correspond to c,b,x from ssm/attention duality.
        let (x, b, c) = {
            let mut xbc = xbc.split_with_sizes(vec![d_inner, ngroups * d_state, ngroups * d_state], 2).into_iter();
            (xbc.next().unwrap(), xbc.next().unwrap(), xbc.next().unwrap())
        };
        debug_assert_eq!([batch, sequence, d_inner], x.dims());
        debug_assert_eq!([batch, sequence, ngroups * d_state], b.dims());
        debug_assert_eq!([batch, sequence, ngroups * d_state], c.dims());

        // prepare state space parameters
        let dt_bias = self.dt_bias.val().unsqueeze_dims(&[0, 1]);
        debug_assert_eq!([1, 1, nheads], dt_bias.dims());
        let dt = burn::tensor::activation::softplus(dt + dt_bias, 1.); // Δ
        debug_assert_eq!([batch, sequence, nheads], dt.dims());
        let a = -self.a_log.val().exp(); // (scalar per head for Ā = exp(Δ A))
        debug_assert_eq!([nheads], a.dims());

        // reshapes
        let x = x.reshape([batch, sequence, nheads, headdim]);
        let b = b.reshape([batch, sequence, ngroups, d_state]);
        let c = c.reshape([batch, sequence, ngroups, d_state]);

        // Perform chunked selective scan
        let (y, final_state) =
            self.chunked_selective_scan(x.clone(), dt, a, b, c, cache.ssm, chunk_size);
        debug_assert_eq!([batch, sequence, nheads, headdim], y.dims());
        // debug_assert_eq!([batch, sequence, d_inner], y.dims());
        cache.ssm = final_state; // update cache.ssm with final state h_L

        // Add D skip
        let d = self
            .d
            .val()
            .unsqueeze_dims::<4>(&[0, 0, 3])
            .expand([batch, sequence, nheads, 1]);
        let x_reshaped = x.reshape([batch, sequence, nheads, headdim]);
        let y = y + d * x_reshaped;
        let y = y.reshape([batch, sequence, d_inner]);
        debug_assert_eq!([batch, sequence, d_inner], y.dims());

        // Normalization
        let y = self.norm.forward(y, z);
        debug_assert_eq!([batch, sequence, d_inner], y.dims());

        // Output projection
        let out = self.out_proj.forward(y);
        debug_assert_eq!([batch, sequence, _d_model], out.dims());
        (out, cache)
    }

    /// Performs a chunked selective scan for Mamba2 using semi-separable decomposition.
    ///
    /// # Shapes
    /// - Input x [batch, sequence, nheads, headdim]
    /// - Input dt Δ [batch, sequence, nheads]
    /// - Input A [nheads]
    /// - Input B [batch, sequence, ngroups, d_state]
    /// - Input C [batch, sequence, ngroups, d_state]
    /// - Input ssm_initial_state h₀ [batch, nheads, headdim, d_state]
    /// - Output.0 y [batch, sequence, nheads, headdim]
    /// - Output.1 ssm_final_state h_L [batch, nheads, headdim, d_state]
    pub fn chunked_selective_scan(
        &self,
        x: Tensor<B, 4>,
        dt: Tensor<B, 3>,
        a: Tensor<B, 1>,
        b: Tensor<B, 4>,
        c: Tensor<B, 4>,
        ssm_initial_state: Tensor<B, 4>,
        chunk_size: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, sequence, nheads, headdim] = x.dims();
        let ngroups = self.ngroups;
        let d_state = self.d_state;
        let d_inner = nheads * headdim;
        let device = x.device();

        assert!(sequence >= 1);

        let num_full_chunks = sequence / chunk_size;
        let remainder_chunk_size = sequence % chunk_size;
        let sequence_full_chunks = num_full_chunks * chunk_size;
        debug_assert_eq!(sequence, sequence_full_chunks + remainder_chunk_size);
        debug_assert_eq!(d_inner, nheads * headdim);

        // expand B and C to per-head
        let heads_per_group = nheads / ngroups;
        let b_per_head = b
            .clone()
            .unsqueeze_dim::<5>(3)
            .expand([batch, sequence, ngroups, heads_per_group, d_state])
            .reshape([batch, sequence, nheads, d_state]);
        let c_per_head = c
            .clone()
            .unsqueeze_dim::<5>(3)
            .expand([batch, sequence, ngroups, heads_per_group, d_state])
            .reshape([batch, sequence, nheads, d_state]);

        // B̄ = Δ B
        let delta_b = dt.clone().unsqueeze_dim(3) * b_per_head.clone();
        debug_assert_eq!([batch, sequence, nheads, d_state], delta_b.dims());

        // A_input = Δ A
        let a_input = a
            .clone()
            .unsqueeze_dims::<3>(&[0, 1])
            .expand([batch, sequence, nheads]);
        let a_input = dt.clone() * a_input;

        let (y, final_state) = if num_full_chunks > 0 {
            let x_full = x.clone().narrow(1, 0, sequence_full_chunks);
            let a_input_full = a_input.narrow(1, 0, sequence_full_chunks);
            let b_full = delta_b.narrow(1, 0, sequence_full_chunks);
            let c_full = c_per_head.clone().narrow(1, 0, sequence_full_chunks);

            // Rearrange into chunks
            let x_chunk = x_full.reshape([batch, num_full_chunks, chunk_size, nheads, headdim]);
            let a_chunk = a_input_full.reshape([batch, num_full_chunks, chunk_size, nheads]);
            let b_chunk = b_full.reshape([batch, num_full_chunks, chunk_size, nheads, d_state]);
            let c_chunk = c_full.reshape([batch, num_full_chunks, chunk_size, nheads, d_state]);

            let a_chunk = a_chunk.permute([0, 3, 1, 2]);
            assert_eq!([batch, nheads, num_full_chunks, chunk_size], a_chunk.dims());
            let a_cumsum = a_chunk.clone().cumsum(3);

            // reference annotations are usually as:
            // - b == batch
            // - c == num_full_chunks
            // - h == nheads
            // - l/s == chunk_size
            // - n == d_state
            // - p == headdim
            // - z == 1 + num_full_chunks

            // Step 1: Intra-chunk outputs: Y_diag = ∑_d_state ∑_chunk_size C B L X
            let y_diag = {
                // contract C and B over d_state
                let b_chunk = b_chunk.clone().swap_dims(2, 3);
                let c_chunk = c_chunk.clone().swap_dims(2, 3);
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, d_state],
                    b_chunk.dims()
                );
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, d_state],
                    c_chunk.dims()
                );
                let temp1 = c_chunk.matmul(b_chunk.transpose());
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, chunk_size],
                    temp1.dims()
                );
                let temp1 = temp1.swap_dims(2, 3);
                assert_eq!(
                    [batch, num_full_chunks, chunk_size, nheads, chunk_size],
                    temp1.dims()
                );
                // element-wise multiplication with permuted L
                let l = segsum(a_chunk.clone()).exp();
                assert_eq!(
                    [batch, nheads, num_full_chunks, chunk_size, chunk_size],
                    l.dims()
                );
                let l = l.permute([0, 2, 3, 1, 4]);
                let temp2 = temp1 * l;
                assert_eq!(
                    [batch, num_full_chunks, chunk_size, nheads, chunk_size],
                    temp2.dims()
                );
                // final contraction over last chunk_size via batched matmul with X
                let temp2 = temp2.permute([0, 1, 3, 2, 4]);
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, chunk_size],
                    temp2.dims()
                );
                let x_chunk = x_chunk.clone().swap_dims(2, 3);
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, headdim],
                    x_chunk.dims()
                );
                let y_diag = temp2.matmul(x_chunk);
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, headdim],
                    y_diag.dims()
                );
                y_diag.swap_dims(2, 3)
            };
            assert_eq!(
                [batch, num_full_chunks, chunk_size, nheads, headdim],
                y_diag.dims()
            );

            // Step 2: Intra-chunk states: states = ∑_chunk_size B decay_states X
            let states = {
                // element-wise multiply decay_states and X
                let decay_states =
                    (a_cumsum.clone().narrow(3, chunk_size - 1, 1) - a_cumsum.clone()).exp();
                assert_eq!(
                    [batch, nheads, num_full_chunks, chunk_size],
                    decay_states.dims()
                );
                let decay_states = decay_states.permute([0, 2, 3, 1]).unsqueeze_dim(4);
                assert_eq!(
                    [batch, num_full_chunks, chunk_size, nheads, 1],
                    decay_states.dims()
                );
                let temp = decay_states * x_chunk.clone();
                assert_eq!(
                    [batch, num_full_chunks, chunk_size, nheads, headdim],
                    temp.dims()
                );
                // contraction over chunk_size via batched matmul with B
                let temp = temp.permute([0, 1, 3, 4, 2]);
                assert_eq!(
                    [batch, num_full_chunks, nheads, headdim, chunk_size],
                    temp.dims()
                );
                let b_chunk = b_chunk.clone().permute([0, 1, 3, 2, 4]);
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, d_state],
                    b_chunk.dims()
                );
                let states = temp.matmul(b_chunk);
                states
            };
            assert_eq!(
                [batch, num_full_chunks, nheads, headdim, d_state],
                states.dims()
            );

            // Step 3: Inter-chunk state passing: new_states = ∑_num_full_chunks decay_chunks states
            let (states, final_state) = {
                // cat states
                let initial_states = ssm_initial_state.unsqueeze_dim(1);
                assert_eq!([batch, 1, nheads, headdim, d_state], initial_states.dims());
                let states = Tensor::cat(vec![initial_states, states], 1);
                assert_eq!(
                    [
                        batch,
                        1 + num_full_chunks, // 1 + c
                        nheads,
                        headdim,
                        d_state
                    ],
                    states.dims()
                );
                // decay_chunk
                let a_chunk_ends = a_cumsum.clone().narrow(3, chunk_size - 1, 1).squeeze_dim(3);
                assert_eq!([batch, nheads, num_full_chunks], a_chunk_ends.dims());
                let a_chunk_pad = Tensor::cat(
                    vec![
                        Tensor::zeros(Shape::new([batch, nheads, 1]), &device),
                        a_chunk_ends,
                    ],
                    2,
                );
                assert_eq!([batch, nheads, 1 + num_full_chunks], a_chunk_pad.dims());
                let decay_chunk = segsum(a_chunk_pad).exp();
                assert_eq!(
                    [
                        batch,
                        nheads,
                        1 + num_full_chunks, // z
                        1 + num_full_chunks  // 1+c
                    ], // bhz(1+c)
                    decay_chunk.dims()
                );
                // align for batched matmul
                let states = states.clone().permute([0, 2, 1, 3, 4]); // b(1+c)hpn -> bh(1+c)pn
                assert_eq!(
                    [
                        batch,
                        nheads,
                        1 + num_full_chunks, // 1+c
                        headdim,
                        d_state
                    ],
                    states.dims()
                );
                let states =
                    states.reshape([batch, nheads, 1 + num_full_chunks, headdim * d_state]); // bh(1+c)pn -> bh(1+c)(pn)
                assert_eq!(
                    [batch, nheads, 1 + num_full_chunks, headdim * d_state],
                    states.dims()
                );
                // contraction over (1+c): bhz(1+c) @ bh(1+c)(pn) -> bhz(pn)
                let new_states = decay_chunk.matmul(states);
                assert_eq!(
                    [batch, nheads, 1 + num_full_chunks, headdim * d_state], // bhz(pn)
                    new_states.dims()
                );
                let new_states =
                    new_states.reshape([batch, nheads, 1 + num_full_chunks, headdim, d_state]); // bhzpn
                //
                let mut split = new_states.split_with_sizes(vec![num_full_chunks, 1], 2).into_iter();
                let states = split.next().unwrap(); // bhcpn
                let final_state = split.next().unwrap(); // bh1pn
                (states.swap_dims(1, 2), final_state.squeeze_dim(2))
            };
            assert_eq!(
                [batch, num_full_chunks, nheads, headdim, d_state], // bchpn
                states.dims()
            );
            assert_eq!([batch, nheads, headdim, d_state], final_state.dims()); // bhpn

            // Step 4: Inter-chunk outputs: Y_off = state_decay_out ∑_d_state C state
            let y_off = {
                let state_decay_out = a_cumsum.exp();
                assert_eq!(
                    [batch, nheads, num_full_chunks, chunk_size], // bhcl
                    state_decay_out.dims()
                );
                // contract C and states over d_state
                let c_chunk = c_chunk.swap_dims(2, 3); // bclhn -> bchln
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, d_state], // bchln
                    c_chunk.dims()
                );
                let states = states.transpose(); // bchpn -> bchnp
                assert_eq!(
                    [batch, num_full_chunks, nheads, d_state, headdim], // bchnp
                    states.dims()
                );
                let temp = c_chunk.matmul(states);
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, headdim], // bchlp
                    temp.dims()
                );
                // element mul
                let state_decay_out = state_decay_out.permute([0, 2, 1, 3]).unsqueeze_dim(4);
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, 1], // bchl1
                    state_decay_out.dims()
                );
                let temp = temp * state_decay_out; // bchlp
                assert_eq!(
                    [batch, num_full_chunks, nheads, chunk_size, headdim], // bchlp
                    temp.dims()
                );
                temp.swap_dims(2, 3) // bclhp
            };
            assert_eq!(
                [batch, num_full_chunks, chunk_size, nheads, headdim], // bclhp
                y_off.dims()
            );

            // Combine intra and inter-chunk
            let y_full_combined = y_diag + y_off;
            let y_full = y_full_combined.reshape([batch, sequence_full_chunks, nheads, headdim]);

            if remainder_chunk_size > 0 {
                // some full chunks + remainder
                let x_rem = x.narrow(1, sequence_full_chunks, remainder_chunk_size);
                let dt_rem = dt.narrow(1, sequence_full_chunks, remainder_chunk_size);
                let b_rem = b.narrow(1, sequence_full_chunks, remainder_chunk_size);
                let c_rem = c.narrow(1, sequence_full_chunks, remainder_chunk_size);
                // recursively calls chunked_selective_scan with a single chunk_size == remainder
                let (y_rem, final_state) = self.chunked_selective_scan(
                    x_rem,
                    dt_rem,
                    a,
                    b_rem,
                    c_rem,
                    final_state,
                    remainder_chunk_size,
                );
                (Tensor::cat(vec![y_full, y_rem], 1), final_state)
            } else {
                // some full chunks and no remainer
                (y_full, final_state)
            }
        } else {
            // no full chunks and implied remainder
            // recursively calls chunked_selective_scan with a single chunk_size == remainder
            let (y_rem, final_state) = self.chunked_selective_scan(
                x,
                dt,
                a,
                b,
                c,
                ssm_initial_state,
                remainder_chunk_size,
            );
            (y_rem, final_state)
        };
        (y, final_state)
    }
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
                let mut split = z_xbc_dt.split_with_sizes(vec![d_inner, conv_dim, nheads], 1).into_iter();
                (split.next().unwrap(), split.next().unwrap(), split.next().unwrap())
            };
            debug_assert_eq!([batch, d_inner], z.dims());
            debug_assert_eq!([batch, conv_dim], xbc.dims());
            debug_assert_eq!([batch, nheads], dt.dims());

            // Convolution step
            cache.conv = {
                let conv = cache.conv;
                debug_assert_eq!([batch, conv_dim, d_conv], conv.dims());

                // split-off oldest/first column (i.e. rolling leftwards)
                let t0 = conv.narrow(2, 1, d_conv - 1);
                debug_assert_eq!([batch, conv_dim, d_conv - 1], t0.dims());

                // insert xbc as a the newest/last column
                let conv = Tensor::cat([t0, xbc.unsqueeze_dim(2)].to_vec(), 2);
                debug_assert_eq!([batch, conv_dim, d_conv], conv.dims());

                conv
            };

            let xbc = {
                let conv1d = self.conv1d.weight.val();
                // [channels_out, channels_in / groups, kernel_size]
                debug_assert_eq!([conv_dim, 1, d_conv], conv1d.dims());
                let conv1d = conv1d.swap_dims(0, 1);
                debug_assert_eq!([1, conv_dim, d_conv], conv1d.dims());
                let conv1d = conv1d.expand([batch, conv_dim, d_conv]);
                debug_assert_eq!([batch, conv_dim, d_conv], conv1d.dims());

                let xbc = cache.conv.clone() * conv1d;
                debug_assert_eq!([batch, conv_dim, d_conv], xbc.dims());
                let mut xbc = xbc.sum_dim(2).squeeze_dim(2);
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
                let mut split =
                    xbc.split_with_sizes(vec![d_inner, ngroups * d_state, ngroups * d_state], 1).into_iter();
                (split.next().unwrap(), split.next().unwrap(), split.next().unwrap())
            };
            debug_assert_eq!([batch, d_inner], x.dims());
            debug_assert_eq!([batch, ngroups * d_state], b.dims());
            debug_assert_eq!([batch, ngroups * d_state], c.dims());

            // SSM step
            let ssm_shape = [batch, nheads, headdim, d_state]; // cache.ssm shape
            // Δₜ = softplus(dt + dt_bias)
            let dt =
                burn::tensor::activation::softplus(dt + self.dt_bias.val().unsqueeze_dim(0), 1.);
            debug_assert_eq!([batch, nheads], dt.dims());
            let a = -self.a_log.val().exp(); // A
            debug_assert_eq!([nheads], a.dims());

            let x = x.reshape([batch, nheads, headdim]); // xₜ: d_inner = nheads * headdim
            let b = b.reshape([batch, ngroups, d_state]); // Bₜ
            let c = c.reshape([batch, ngroups, d_state]); // Cₜ

            // Āₜ = exp(Δₜ A)
            let dta = (dt.clone() * a.clone().unsqueeze()).exp();
            debug_assert_eq!([batch, nheads], dta.dims());

            let dta = dta.unsqueeze_dims(&[2, 3]);
            debug_assert_eq!([batch, nheads, 1, 1], dta.dims());
            let dta = dta.expand(ssm_shape);

            // dt * b * x = [batch, nheads] * [batch, ngroups, d_state] * [batch, nheads, headdim]
            let heads_per_group = nheads / ngroups;
            // B̄ₜ xₜ = Δₜ Bₜ xₜ
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

            // Compute state update
            // hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ
            cache.ssm = cache.ssm * dta + dtbx;
            debug_assert_eq!(ssm_shape, cache.ssm.dims());

            // Compute output:yₜ = Cₜ hₜ + D xₜ
            let y = {
                // yₜ = Cₜ hₜ (matrix-vector product, without the skip)
                let y = cache.ssm.clone() * c;
                let y = y.sum_dim(3).squeeze_dim(3);
                debug_assert_eq!([batch, nheads, headdim], y.dims());

                // yₜ += D xₜ
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
