use crate::mamba2::*;
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
pub struct Mamba2<B: Backend> {
    /// Input channel: [`Mamba2Config::d_model`].
    /// Output channel: z + xbc + dt.
    ///
    /// z: [`Self::d_inner`].
    /// xbc: [`Self::conv_dim`].
    /// dt: [`Self::nheads`].
    pub in_proj: Linear<B>,

    /// Input channel: conv_dim.
    /// Output channel: conv_dim.
    /// Kernel: [`Mamba2Config::conv_kernel`].
    /// Padding: [`Mamba2Config::conv_kernel`] - 1.
    /// Groups: conv_dim.
    pub conv1d: Conv1d<B>,

    /// Dims: [`Self::nheads`].
    pub dt_bias_h: Param<Tensor<B, 1>>,

    pub dt_limit: (f64, f64),

    /// Dims: [`Self::nheads`].
    pub a_log_h: Param<Tensor<B, 1>>,

    /// Dims: [`Self::nheads`].
    pub d_h: Param<Tensor<B, 1>>,

    /// Dims: [`Self::d_inner`].
    pub norm: RmsNormGated<B>,

    /// Input channel: [`Self::d_inner`].
    /// Output channel: [`Mamba2Config::d_model`].
    pub out_proj: Linear<B>,

    /// Dims: [[`Self::nheads`], [`Self::per_head_dim`], [`Mamba2Config::state_rank`]].
    pub init_states_hpr: Option<Param<Tensor<B, 3>>>,

    /// [`Mamba2Config::state_rank`].
    pub state_rank: usize,

    /// [`Mamba2Config::ngroups`].
    pub ngroups: usize,
}

impl<B: Backend> Mamba2<B> {
    /// d_inner = expand * d_model = nheads * per_head_dim.
    pub fn d_inner(&self) -> usize {
        let [d_inner] = self.norm.gamma.dims();
        d_inner
    }

    /// nheads = d_inner / per_head_dim.
    pub fn nheads(&self) -> usize {
        let [nheads] = self.a_log_h.dims();
        nheads
    }

    /// per_head_dim = d_inner / nheads.
    pub fn per_head_dim(&self) -> usize {
        self.d_inner() / self.nheads()
    }

    /// conv_dim = d_inner + 2 * ngroups * state_rank.
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.state_rank
    }
}

#[derive(Config, Debug)]
pub struct Mamba2Config {
    /// Hidden dimension.
    pub d_model: usize,

    /// latent state dimension.
    #[config(default = 128)]
    pub state_rank: usize,

    /// Convolution kernel size.
    #[config(default = 4)]
    pub conv_kernel: usize,

    /// Expansion factor for `d_inner`.
    #[config(default = 2)]
    pub expand: usize,

    /// Head dimension.
    #[config(default = 64)]
    pub per_head_dim: usize,

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
    ///
    /// Defaults to (0, f16::MAX).
    #[config(default = "(0., 6.5504e+4)")]
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
impl Mamba2Config {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mamba2<B> {
        let d_inner = self.d_inner();
        assert!(self.per_head_dim > 0);
        let nheads = self.nheads();
        debug_assert_eq!(nheads * self.per_head_dim, d_inner);
        assert_eq!(nheads % self.ngroups, 0);

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
        let conv1d = Conv1dConfig::new(conv_dim, conv_dim, self.conv_kernel)
            // the conv's inputs are padded manually for causality, by self.conv_kernel - 1
            // TODO: only left-padding is necessary. Add explicit left-side padding.
            .with_padding(burn::nn::PaddingConfig1d::Valid)
            .with_groups(conv_dim)
            .with_bias(self.has_conv_bias)
            // follows PyTorch's default initializer
            // fan_in = in_channels / groups * kernel_size
            .with_initializer(uniform_init(self.conv_kernel))
            .init::<B>(device);

        // dt_bias initialization
        // note: this placeholder impl may lose precision for very small values,
        // and a Taylor series could approximate it: e^x - 1 = x + x^2/2! + x^3/3! + ⋯
        // but with the clamp at dt_init_floor, this isn't necessary
        let expm1 = |t: Tensor<B, 1>| t.exp() - 1.;
        let dt_h = Tensor::random(
            [nheads],
            burn::tensor::Distribution::Uniform(self.dt_min.ln(), self.dt_max.ln()),
            device,
        )
        .exp();
        let dt_h = dt_h.clamp(self.dt_init_floor, f64::INFINITY);
        let inv_dt_h = dt_h.clone() + (-expm1(-dt_h)).log(); // Inverse softplus
        let dt_bias_h = Param::from_tensor(inv_dt_h);

        // A_log initialization for Āₜ = exp(Δₜ A)
        assert!(self.a_init_range.0 > 0.);
        assert!(self.a_init_range.0 < self.a_init_range.1);
        let a_h = Tensor::random(
            [nheads],
            burn::tensor::Distribution::Uniform(self.a_init_range.0, self.a_init_range.1),
            device,
        );
        let a_log_h = Param::from_tensor(a_h.log());

        // D initialization
        let d_h = Initializer::Ones.init::<B, 1, _>([nheads], device);

        // Normalization and output projection
        let norm = RmsNormGatedConfig::new(d_inner)
            // .with_epsilon(div_eps::<B>().to_f64())
            .with_norm_before_gate(self.is_norm_before_gate)
            .init(device);
        let out_proj = LinearConfig::new(d_inner, self.d_model)
            .with_bias(self.has_proj_bias)
            // follows PyTorch's default initializer
            .with_initializer(uniform_init(d_inner))
            .init(device);

        // Optional learnable initial states
        let init_states_hpr = if self.has_learnable_init_states {
            Some(
                Initializer::Zeros
                    .init::<B, 3, _>([nheads, self.per_head_dim, self.state_rank], device),
            )
        } else {
            None
        };

        Mamba2 {
            in_proj,
            conv1d,
            dt_bias_h,
            dt_limit: self.dt_limit,
            a_log_h,
            d_h,
            norm,
            out_proj,
            init_states_hpr,
            state_rank: self.state_rank,
            ngroups: self.ngroups,
        }
    }

    /// d_inner = expand * d_model.
    pub fn d_inner(&self) -> usize {
        self.expand * self.d_model
    }

    /// nheads = d_inner / per_head_dim.
    pub fn nheads(&self) -> usize {
        self.d_inner() / self.per_head_dim
    }

    /// conv_dim = d_inner + 2 * ngroups * state_rank.
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.state_rank
    }
}

impl<B: Backend> Mamba2<B> {
    /// See also [`Self::step`].
    ///
    /// `chunk_len`: Chunk size for selective scan. Defaults to 256.
    ///
    /// # Shapes
    /// - Input [batch, sequence, d_model]
    /// - Output.0 output [batch, sequence, d_model]
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        input_bsm: Tensor<B, 3>,
        cache: Option<Mamba2Cache<B>>,
        chunk_len: usize,
    ) -> (Tensor<B, 3>, Mamba2Cache<B>) {
        let [batch, sequence, _d_model] = input_bsm.dims();
        let d_inner = self.d_inner();
        let ngroups = self.ngroups;
        let nheads = self.nheads();
        let per_head_dim = self.per_head_dim();
        let conv_dim = self.conv_dim();
        let state_rank = self.state_rank;
        let [_conv_dim, _, conv_kernel] = self.conv1d.weight.dims();
        let [_d_model, d_in_proj_output] = self.in_proj.weight.dims();
        assert_eq!(conv_dim, _conv_dim);
        assert_eq!(nheads % ngroups, 0);
        assert!(sequence > 0);

        let mut cache = cache.unwrap_or_else(|| {
            let device = &input_bsm.device();
            let conv_bvk = Tensor::zeros(Shape::new([batch, conv_dim, conv_kernel]), device);
            let ssm_bhpr = Tensor::zeros(
                Shape::new([batch, nheads, per_head_dim, state_rank]),
                device,
            );
            Mamba2Cache { conv_bvk, ssm_bhpr }
        });

        // input projection
        let (z_gate_bsi, xbc_bsv, dt_raw_bsh) = {
            let z_xbc_dt = self.in_proj.forward(input_bsm);
            debug_assert_eq!([batch, sequence, d_in_proj_output], z_xbc_dt.dims());
            debug_assert_eq!(
                [batch, sequence, d_inner + conv_dim + nheads],
                z_xbc_dt.dims()
            );

            let mut z_xbc_dt = z_xbc_dt
                .split_with_sizes(vec![d_inner, conv_dim, nheads], 2)
                .into_iter();
            (
                z_xbc_dt.next().unwrap(),
                z_xbc_dt.next().unwrap(),
                z_xbc_dt.next().unwrap(),
            )
        };
        debug_assert_eq!([batch, sequence, d_inner], z_gate_bsi.dims());
        debug_assert_eq!([batch, sequence, conv_dim], xbc_bsv.dims());
        debug_assert_eq!([batch, sequence, nheads], dt_raw_bsh.dims());

        // convolution and activation
        let xbc_bvs = xbc_bsv.permute([0, 2, 1]); // xbc_bsv.swap_dims(1, 2)
        debug_assert_eq!([batch, conv_dim, sequence], xbc_bvs.dims());
        // split-off oldest/first column (i.e. rolling leftwards) for the causal pad
        assert!(conv_kernel >= 1);
        let xbc_bvS = if conv_kernel >= 2 {
            let t0_bvK = cache.conv_bvk.slice(s![.., .., 1..]);
            debug_assert_eq!([batch, conv_dim, conv_kernel - 1], t0_bvK.dims());
            // add the causal pad (only left-pad is necessary)
            Tensor::cat(vec![t0_bvK, xbc_bvs], 2)
        } else {
            xbc_bvs
        };
        debug_assert_eq!(
            [batch, conv_dim, (conv_kernel - 1) + sequence],
            xbc_bvS.dims()
        );
        // get the final state for the causal pad (right-side of pre-xbc)
        cache.conv_bvk = xbc_bvS.clone().slice(s![.., .., (sequence - 1)..]);
        debug_assert_eq!([batch, conv_dim, conv_kernel], cache.conv_bvk.dims());
        let xbc_bvs = self.conv1d.forward(xbc_bvS);
        debug_assert_eq!([batch, conv_dim, sequence], xbc_bvs.dims());
        //
        let xbc_bsv = xbc_bvs.permute([0, 2, 1]); // xbc_bvs.swap_dims(1, 2)
        debug_assert_eq!([batch, sequence, conv_dim], xbc_bsv.dims());
        let xbc_bsv = Silu::new().forward(xbc_bsv);
        debug_assert_eq!([batch, sequence, conv_dim], xbc_bsv.dims());

        // split xbc into x, B, C.
        // note: the attention Q,K,V values correspond to c,b,x from ssm/attention duality.
        let (x_bshp, b_bsgr, c_bsgr) = {
            let mut xbc = xbc_bsv
                .split_with_sizes(vec![d_inner, ngroups * state_rank, ngroups * state_rank], 2)
                .into_iter();
            (
                xbc.next()
                    .unwrap() // [batch, sequence, d_inner]
                    .reshape([batch, sequence, nheads, per_head_dim]),
                xbc.next()
                    .unwrap() // [batch, sequence, ngroups * state_rank]
                    .reshape([batch, sequence, ngroups, state_rank]),
                xbc.next()
                    .unwrap() // [batch, sequence, ngroups * state_rank]
                    .reshape([batch, sequence, ngroups, state_rank]),
            )
        };

        // prepare state space parameters
        let dt_bias_11h = self.dt_bias_h.val().unsqueeze_dims(&[0, 1]);
        debug_assert_eq!([1, 1, nheads], dt_bias_11h.dims());
        let dt_bsh = softplus(dt_raw_bsh + dt_bias_11h).clamp(self.dt_limit.0, self.dt_limit.1); // Δ
        debug_assert_eq!([batch, sequence, nheads], dt_bsh.dims());
        let a_head_decay_h = -self.a_log_h.val().exp(); // (scalar per head for Ā = exp(Δ A))
        debug_assert_eq!([nheads], a_head_decay_h.dims());

        // Perform chunked selective scan
        let (y_bshp, final_state_bhpr) = self.chunked_selective_scan(
            x_bshp.clone(),
            dt_bsh,
            a_head_decay_h,
            b_bsgr,
            c_bsgr,
            cache.ssm_bhpr,
            chunk_len,
        );
        debug_assert_eq!([batch, sequence, nheads, per_head_dim], y_bshp.dims());

        // debug_assert_eq!([batch, sequence, d_inner], y.dims());
        cache.ssm_bhpr = final_state_bhpr; // update cache with final state

        // Add D skip
        let d_bsh1 = self
            .d_h
            .val()
            .unsqueeze_dims::<4>(&[0, 0, 3])
            .expand([batch, sequence, nheads, 1]);
        let y_bshp = y_bshp + d_bsh1 * x_bshp;
        let y_bsi = y_bshp.reshape([batch, sequence, d_inner]);
        debug_assert_eq!([batch, sequence, d_inner], y_bsi.dims());

        // Normalization
        let y_bsi = self.norm.forward(y_bsi, z_gate_bsi);
        debug_assert_eq!([batch, sequence, d_inner], y_bsi.dims());

        // Output projection
        let out_bsm = self.out_proj.forward(y_bsi);
        debug_assert_eq!([batch, sequence, _d_model], out_bsm.dims());

        (out_bsm, cache)
    }

    /// Performs a chunked selective scan for Mamba2 using semi-separable decomposition.
    ///
    /// # Shapes
    /// - Input x_bshp x [batch, sequence, nheads, per_head_dim]
    /// - Input dt_bsh Δ [batch, sequence, nheads]
    /// - Input a_head_decay_h A [nheads]
    /// - Input b_bsgr B [batch, sequence, ngroups, state_rank]
    /// - Input c_bsgr C [batch, sequence, ngroups, state_rank]
    /// - Input ssm_initial_state_bhpr h₀ [batch, nheads, per_head_dim, state_rank]
    /// - Output.0 y_bshp [batch, sequence, nheads, per_head_dim]
    /// - Output.1 ssm_final_state_bhpr h [batch, nheads, per_head_dim, state_rank]
    #[allow(non_snake_case)]
    pub fn chunked_selective_scan(
        &self,
        x_bshp: Tensor<B, 4>,
        dt_bsh: Tensor<B, 3>,
        a_head_decay_h: Tensor<B, 1>,
        b_bsgr: Tensor<B, 4>,
        c_bsgr: Tensor<B, 4>,
        ssm_initial_state_bhpr: Tensor<B, 4>,
        chunk_len: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, sequence_unpadded, nheads, per_head_dim] = x_bshp.dims();
        let ngroups = self.ngroups;
        let state_rank = self.state_rank;
        let d_inner = nheads * per_head_dim;
        let device = &x_bshp.device();
        assert_eq!(nheads % ngroups, 0);
        assert!(sequence_unpadded >= 1);
        assert_eq!(d_inner, nheads * per_head_dim);

        // padding
        assert!(chunk_len > 0);
        let sequence_mod = sequence_unpadded % chunk_len;
        let pad = if sequence_mod == 0 {
            0
        } else {
            chunk_len - sequence_mod
        };
        let (x_bshp, dt_bsh, b_bsgr, c_bsgr, sequence) = if pad == 0 {
            (x_bshp, dt_bsh, b_bsgr, c_bsgr, sequence_unpadded)
        } else {
            // we will call both padded and unpadded sequence as `s`,
            // but the output range and the last state should be corrected
            //
            // note: it's important that the pad tensors are zeroes.
            // they imply identity operations for the unpadded final state up to the end of its chunk.
            // (see the end of chunked_selective_scan Step 3 for more info)
            let x_bshp = Tensor::cat(
                vec![
                    x_bshp,
                    Tensor::zeros(Shape::new([batch, pad, nheads, per_head_dim]), device),
                ],
                1,
            );
            let dt_bsh = Tensor::cat(
                vec![
                    dt_bsh,
                    Tensor::zeros(Shape::new([batch, pad, nheads]), device),
                ],
                1,
            );
            let b_bsgr = Tensor::cat(
                vec![
                    b_bsgr,
                    Tensor::zeros(Shape::new([batch, pad, ngroups, state_rank]), device),
                ],
                1,
            );
            let c_bsgr = Tensor::cat(
                vec![
                    c_bsgr,
                    Tensor::zeros(Shape::new([batch, pad, ngroups, state_rank]), device),
                ],
                1,
            );
            (x_bshp, dt_bsh, b_bsgr, c_bsgr, sequence_unpadded + pad)
        };
        assert_eq!(sequence % chunk_len, 0);
        let nchunks = sequence / chunk_len;

        // expand B and C to per-head
        let heads_per_group = nheads / ngroups;
        let b_bshr = b_bsgr
            .clone()
            .unsqueeze_dim::<5>(3) // bsg1r
            .expand([batch, sequence, ngroups, heads_per_group, state_rank])
            .reshape([batch, sequence, nheads, state_rank]);
        let c_bshr = c_bsgr
            .clone()
            .unsqueeze_dim::<5>(3) // bsg1r
            .expand([batch, sequence, ngroups, heads_per_group, state_rank])
            .reshape([batch, sequence, nheads, state_rank]);

        // B̄ = Δ B
        let delta_b_bshr = dt_bsh.clone().unsqueeze_dim(3) * b_bshr.clone();
        debug_assert_eq!([batch, sequence, nheads, state_rank], delta_b_bshr.dims());

        // A_input = Δ A
        let a_bsh = a_head_decay_h
            .clone()
            .unsqueeze_dims::<3>(&[0, 1]) // 11h
            .expand([batch, sequence, nheads]);
        let a_bsh = dt_bsh.clone() * a_bsh;

        let x_bnlhp = x_bshp
            .clone()
            .reshape([batch, nchunks, chunk_len, nheads, per_head_dim]);
        let a_bnlh = a_bsh.reshape([batch, nchunks, chunk_len, nheads]);
        let b_bnlhr = delta_b_bshr.reshape([batch, nchunks, chunk_len, nheads, state_rank]);
        let c_bnlhr = c_bshr
            .clone()
            .reshape([batch, nchunks, chunk_len, nheads, state_rank]);

        let a_bhnl = a_bnlh.permute([0, 3, 1, 2]);
        assert_eq!([batch, nheads, nchunks, chunk_len], a_bhnl.dims());
        let a_cumsum_bhnl = a_bhnl.clone().cumsum(3);

        // Step 1: Intra-chunk outputs: Y_diag = ∑_state_rank ∑_chunk_len C B L X
        let y_diag_bnlhp = {
            // contract C and B over state_rank
            let b_bnhlr = b_bnlhr.clone().permute([0, 1, 3, 2, 4]); // b_bnlhr.clone().swap_dims(2, 3)
            let c_bnhlr = c_bnlhr.clone().permute([0, 1, 3, 2, 4]); // c_bnlhr.clone().swap_dims(2, 3)
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                b_bnhlr.dims()
            );
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                c_bnhlr.dims()
            );
            let b_bnhrl = b_bnhlr.permute([0, 1, 2, 4, 3]); // b_bnhlr.transpose()
            let temp1_bnhll = c_bnhlr.matmul(b_bnhrl);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, chunk_len],
                temp1_bnhll.dims()
            );
            let temp1_bnlhl = temp1_bnhll.permute([0, 1, 3, 2, 4]); // temp1.swap_dims(2, 3)
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, chunk_len],
                temp1_bnlhl.dims()
            );
            // element-wise multiplication with permuted L
            let l_bhnll = segsum(a_bhnl.clone()).exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len, chunk_len],
                l_bhnll.dims()
            );
            let l_bnlhl = l_bhnll.permute([0, 2, 3, 1, 4]);
            let temp2_bnlhl = temp1_bnlhl * l_bnlhl;
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, chunk_len],
                temp2_bnlhl.dims()
            );
            // final contraction over last chunk_len via batched matmul with X
            let temp2_bnhll = temp2_bnlhl.permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, chunk_len],
                temp2_bnhll.dims()
            );
            let x_bnhlp = x_bnlhp.clone().permute([0, 1, 3, 2, 4]); // x_bnlhp.clone().swap_dims(2, 3)
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                x_bnhlp.dims()
            );
            let y_diag_bnhlp = temp2_bnhll.matmul(x_bnhlp);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                y_diag_bnhlp.dims()
            );
            y_diag_bnhlp.permute([0, 1, 3, 2, 4]) // y_diag_bnhlp.swap_dims(2, 3)
        };
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim],
            y_diag_bnlhp.dims()
        );

        // Step 2: Intra-chunk states: states = ∑_chunk_len B decay_states X
        let states_bnhpr = {
            // element-wise multiply decay_states and X
            let a_cumsum_last_bhn1 = a_cumsum_bhnl.clone().slice(s![.., .., .., -1]);
            let decay_states_bhnl = (a_cumsum_last_bhn1 - a_cumsum_bhnl.clone()).exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len],
                decay_states_bhnl.dims()
            );
            let decay_states_bnlh1 = decay_states_bhnl.permute([0, 2, 3, 1]).unsqueeze_dim(4);
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, 1],
                decay_states_bnlh1.dims()
            );
            let temp_bnlhp = decay_states_bnlh1 * x_bnlhp.clone();
            assert_eq!(
                [batch, nchunks, chunk_len, nheads, per_head_dim],
                temp_bnlhp.dims()
            );
            // contraction over chunk_len via batched matmul with B
            let temp_bnhpl = temp_bnlhp.permute([0, 1, 3, 4, 2]);
            assert_eq!(
                [batch, nchunks, nheads, per_head_dim, chunk_len],
                temp_bnhpl.dims()
            );
            let b_bnhlr = b_bnlhr.clone().permute([0, 1, 3, 2, 4]);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                b_bnhlr.dims()
            );
            temp_bnhpl.matmul(b_bnhlr)
        };
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            states_bnhpr.dims()
        );

        // Step 3: Inter-chunk state passing: new_states = ∑_nchunks decay_chunks states
        let (states_bnhpr, final_state_bnpr) = {
            // cat states
            let initial_states_b1hpr = ssm_initial_state_bhpr.unsqueeze_dim(1);
            assert_eq!(
                [batch, 1, nheads, per_head_dim, state_rank],
                initial_states_b1hpr.dims()
            );
            let states_bNhpr = Tensor::cat(vec![initial_states_b1hpr, states_bnhpr], 1);
            assert_eq!(
                [
                    batch,
                    1 + nchunks, // 1+n
                    nheads,
                    per_head_dim,
                    state_rank
                ],
                states_bNhpr.dims()
            );
            // decay_chunk
            let a_cumsum_last_bhn = a_cumsum_bhnl
                .clone()
                .slice(s![.., .., .., -1])
                .squeeze_dim(3);
            assert_eq!([batch, nheads, nchunks], a_cumsum_last_bhn.dims());
            let a_chunk_pad_bhN = Tensor::cat(
                vec![
                    Tensor::zeros(Shape::new([batch, nheads, 1]), device), // bh1
                    a_cumsum_last_bhn,
                ],
                2,
            );
            assert_eq!([batch, nheads, 1 + nchunks], a_chunk_pad_bhN.dims());
            let decay_chunk_bhNN = segsum(a_chunk_pad_bhN).exp();
            assert_eq!(
                [
                    batch,
                    nheads,
                    1 + nchunks, // 1+n
                    1 + nchunks  // 1+n
                ],
                decay_chunk_bhNN.dims()
            );
            // align for batched matmul
            let states_bhNpr = states_bNhpr.clone().permute([0, 2, 1, 3, 4]);
            assert_eq!(
                [
                    batch,
                    nheads,
                    1 + nchunks, // 1+n
                    per_head_dim,
                    state_rank
                ],
                states_bhNpr.dims()
            );
            let flat_state_dim = per_head_dim * state_rank; // f = (pr)
            let states_bhNf = states_bhNpr.reshape([batch, nheads, 1 + nchunks, flat_state_dim]);
            assert_eq!(
                [batch, nheads, 1 + nchunks, flat_state_dim],
                states_bhNf.dims()
            );
            let new_states_bhNf = decay_chunk_bhNN.matmul(states_bhNf);
            assert_eq!(
                [batch, nheads, 1 + nchunks, flat_state_dim],
                new_states_bhNf.dims()
            );
            let new_states_bhNpr =
                new_states_bhNf.reshape([batch, nheads, 1 + nchunks, per_head_dim, state_rank]);
            let states_bhnpr = new_states_bhNpr
                .clone()
                .slice(s![.., .., 0..nchunks, .., ..]);
            let final_state_bhpr = new_states_bhNpr
                // note regarding the sequence padding:
                // new_states_bhNpr contains the last state for each chunk.
                // even if the sequence_unpadded implied an intermediary state inside the last chunk,
                // that intermediary state is carried to the end of that chunk through identity operations:
                // - Āₜ = exp(Δₜ A) = exp(0) = 1
                // - B̄ₜ = Δₜ Bₜ = 0
                // so it's ok to grab the final state of the last chunk, even for padded sequences
                .slice(s![.., .., nchunks, .., ..])
                .squeeze_dim(2);
            (
                states_bhnpr.permute([0, 2, 1, 3, 4]), // states_bhnpr.swap_dims(1, 2)
                final_state_bhpr,
            )
        };
        assert_eq!(
            [batch, nchunks, nheads, per_head_dim, state_rank],
            states_bnhpr.dims()
        );
        assert_eq!(
            [batch, nheads, per_head_dim, state_rank],
            final_state_bnpr.dims()
        );

        // Step 4: Inter-chunk outputs: Y_off = state_decay_out ∑_state_rank C state
        let y_off_bnlhp = {
            let state_decay_out_bhnl = a_cumsum_bhnl.exp();
            assert_eq!(
                [batch, nheads, nchunks, chunk_len],
                state_decay_out_bhnl.dims()
            );
            // contract C and states over state_rank
            let c_bnhlr = c_bnlhr.permute([0, 1, 3, 2, 4]); // c_chunk.swap_dims(2, 3)
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, state_rank],
                c_bnhlr.dims()
            );
            let states_bnhrp = states_bnhpr.permute([0, 1, 2, 4, 3]); // states_bnhpr.transpose()
            assert_eq!(
                [batch, nchunks, nheads, state_rank, per_head_dim],
                states_bnhrp.dims()
            );
            let temp_bnhlp = c_bnhlr.matmul(states_bnhrp);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                temp_bnhlp.dims()
            );
            // element mul
            let state_decay_out_bnhl1 = state_decay_out_bhnl.permute([0, 2, 1, 3]).unsqueeze_dim(4);
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, 1],
                state_decay_out_bnhl1.dims()
            );
            let temp_bnhlp = temp_bnhlp * state_decay_out_bnhl1;
            assert_eq!(
                [batch, nchunks, nheads, chunk_len, per_head_dim],
                temp_bnhlp.dims()
            );
            temp_bnhlp.permute([0, 1, 3, 2, 4]) // temp_bnhlp.swap_dims(2, 3) // bclhp
        };
        assert_eq!(
            [batch, nchunks, chunk_len, nheads, per_head_dim], // bclhp
            y_off_bnlhp.dims()
        );

        // Combine intra and inter-chunk
        let y_bnlhp = y_diag_bnlhp + y_off_bnlhp;
        let y_bshp = y_bnlhp.reshape([batch, sequence, nheads, per_head_dim]);
        let y_bshp = y_bshp.slice(s![.., 0..sequence_unpadded, .., ..]); // may correct for padding

        (y_bshp, final_state_bnpr)
    }
}

mod step {
    use super::*;

    impl<B: Backend> Mamba2<B> {
        /// # Shapes
        ///   - Input [batch, d_model]
        ///   - Output [batch, d_model]
        //
        /// Single-step inference for decoding one token at a time.
        ///
        /// # Shapes
        /// - Input: [batch, d_model]
        /// - Output: [batch, d_model]
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
            let [_d_model, d_in_proj_output] = self.in_proj.weight.dims();
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

            // Input projection
            let (z_gate_bi, xbc_bv, dt_raw_bh) = {
                let z_xbc_dt = self.in_proj.forward(input_bm);
                debug_assert_eq!([batch, d_in_proj_output], z_xbc_dt.dims());
                debug_assert_eq!([batch, d_inner + conv_dim + nheads], z_xbc_dt.dims());

                let mut z_xbc_dt = z_xbc_dt
                    .split_with_sizes(vec![d_inner, conv_dim, nheads], 1)
                    .into_iter();
                (
                    z_xbc_dt.next().unwrap(),
                    z_xbc_dt.next().unwrap(),
                    z_xbc_dt.next().unwrap(),
                )
            };
            debug_assert_eq!([batch, d_inner], z_gate_bi.dims());
            debug_assert_eq!([batch, conv_dim], xbc_bv.dims());
            debug_assert_eq!([batch, nheads], dt_raw_bh.dims());

            // Convolution step
            cache.conv_bvk = {
                let conv_bvk = cache.conv_bvk;
                debug_assert_eq!([batch, conv_dim, conv_kernel], conv_bvk.dims());

                // split-off oldest/first column (i.e. rolling leftwards)
                let t0_bvK = conv_bvk.slice(s![.., .., 1..]);
                debug_assert_eq!([batch, conv_dim, conv_kernel - 1], t0_bvK.dims());

                // insert xbc as a the newest/last column
                let conv_bvk = Tensor::cat([t0_bvK, xbc_bv.unsqueeze_dim(2)].to_vec(), 2);
                debug_assert_eq!([batch, conv_dim, conv_kernel], conv_bvk.dims());

                conv_bvk
            };

            let xbc_bv = {
                let conv1d_v1k = self.conv1d.weight.val();
                // [channels_out, channels_in / groups, kernel_size]
                debug_assert_eq!([conv_dim, 1, conv_kernel], conv1d_v1k.dims());
                let conv1d_1vk = conv1d_v1k.permute([1, 0, 2]); // conv1d_v1k.swap_dims(0, 1)
                debug_assert_eq!([1, conv_dim, conv_kernel], conv1d_1vk.dims());
                let conv1d_bvk = conv1d_1vk.expand([batch, conv_dim, conv_kernel]);
                debug_assert_eq!([batch, conv_dim, conv_kernel], conv1d_bvk.dims());

                let xbc_bvk = cache.conv_bvk.clone() * conv1d_bvk;
                debug_assert_eq!([batch, conv_dim, conv_kernel], xbc_bvk.dims());
                let mut xbc_bv = xbc_bvk.sum_dim(2).squeeze_dim(2);
                debug_assert_eq!([batch, conv_dim], xbc_bv.dims());
                if let Some(bias_v) = &self.conv1d.bias {
                    debug_assert_eq!([conv_dim], bias_v.dims());
                    xbc_bv = xbc_bv + bias_v.val().unsqueeze();
                }
                Silu::new().forward(xbc_bv)
            };
            debug_assert_eq!([batch, conv_dim], xbc_bv.dims());

            // Split (xₜ, Bₜ, Cₜ)
            debug_assert_eq!(d_inner, nheads * per_head_dim);
            let (x_bhp, b_bgr, c_bgr) = {
                let mut split = xbc_bv
                    .split_with_sizes(vec![d_inner, ngroups * state_rank, ngroups * state_rank], 1)
                    .into_iter();
                (
                    // xₜ
                    split
                        .next()
                        .unwrap() // [batch, d_inner]
                        .reshape([batch, nheads, per_head_dim]),
                    split
                        .next()
                        .unwrap() // [batch, ngroups * state_rank]
                        .reshape([batch, ngroups, state_rank]),
                    split
                        .next()
                        .unwrap() // [batch, ngroups * state_rank]
                        .reshape([batch, ngroups, state_rank]),
                )
            };

            // SSM step
            let ssm_shape_bhpr = [batch, nheads, per_head_dim, state_rank]; // cache ssm shape
            // Δₜ = softplus(dt + dt_bias)
            let dt_bias_1h = self.dt_bias_h.val().unsqueeze_dim(0);
            debug_assert_eq!([1, nheads], dt_bias_1h.dims());
            let dt_bh = softplus(dt_raw_bh + dt_bias_1h).clamp(self.dt_limit.0, self.dt_limit.1);
            debug_assert_eq!([batch, nheads], dt_bh.dims());
            let a_head_decay_h = -self.a_log_h.val().exp(); // A
            debug_assert_eq!([nheads], a_head_decay_h.dims());

            // Āₜ = exp(Δₜ A)
            let dta_bh = (dt_bh.clone() * a_head_decay_h.clone().unsqueeze()).exp();
            debug_assert_eq!([batch, nheads], dta_bh.dims());

            let dta_bh11 = dta_bh.unsqueeze_dims(&[2, 3]);
            debug_assert_eq!([batch, nheads, 1, 1], dta_bh11.dims());
            let dta_bhpr = dta_bh11.expand(ssm_shape_bhpr);

            // dt * b * x = [batch, nheads] * [batch, ngroups, state_rank] * [batch, nheads, per_head_dim]
            let heads_per_group = nheads / ngroups;
            // B̄ₜ xₜ = Δₜ Bₜ xₜ
            let dtbx_bhpr = {
                let x_bhpr = x_bhp.clone().unsqueeze_dim::<4>(3).expand(ssm_shape_bhpr);

                let b_b1gr1 = b_bgr.unsqueeze_dims(&[1, 4]);
                debug_assert_eq!([batch, 1, ngroups, state_rank, 1], b_b1gr1.dims());
                let b_bhpr = b_b1gr1
                    .expand([batch, heads_per_group, ngroups, state_rank, 1])
                    .reshape([batch, nheads, state_rank, 1]) // bhr1
                    .permute([0, 1, 3, 2]) // .swap_dims(2, 3) // bh1r
                    .expand(ssm_shape_bhpr);

                let dt_bhpr = dt_bh.unsqueeze_dims::<4>(&[2, 3]).expand(ssm_shape_bhpr);

                dt_bhpr * b_bhpr * x_bhpr
            };
            debug_assert_eq!(ssm_shape_bhpr, dtbx_bhpr.dims());

            // Compute state update
            // hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ
            cache.ssm_bhpr = cache.ssm_bhpr * dta_bhpr + dtbx_bhpr;
            debug_assert_eq!(ssm_shape_bhpr, cache.ssm_bhpr.dims());

            // Compute output:yₜ = Cₜ hₜ + D xₜ
            let y_bi = {
                // yᵢₜ = Cₜ hₜ
                let c_b1gr1 = c_bgr.unsqueeze_dims(&[1, 4]);
                debug_assert_eq!([batch, 1, ngroups, state_rank, 1], c_b1gr1.dims());
                let c_bhpr = c_b1gr1
                    .expand([batch, heads_per_group, ngroups, state_rank, 1])
                    .reshape([batch, nheads, state_rank, 1]) // bhr1
                    .permute([0, 1, 3, 2]) // .swap_dims(2, 3) // bh1r
                    .expand(ssm_shape_bhpr);
                debug_assert_eq!(ssm_shape_bhpr, c_bhpr.dims());
                let y_state_bhpr = cache.ssm_bhpr.clone() * c_bhpr;
                let y_state_bhp = y_state_bhpr.sum_dim(3).squeeze_dim(3);
                debug_assert_eq!([batch, nheads, per_head_dim], y_state_bhp.dims());

                // yⱼₜ = D xₜ
                let d_1h1 = self.d_h.val().unsqueeze_dims(&[0, 2]);
                debug_assert_eq!([1, nheads, 1], d_1h1.dims());
                let d_bhp = d_1h1.expand([batch, nheads, per_head_dim]);
                let y_skip_bhp = d_bhp * x_bhp;

                // yₜ = yᵢₜ + yⱼₜ
                let y_bhp = y_state_bhp + y_skip_bhp;
                debug_assert_eq!([batch, nheads, per_head_dim], y_bhp.dims());

                // Apply normalization with z_gate
                let y_bi = y_bhp.reshape([batch, d_inner]);
                self.norm.forward(y_bi, z_gate_bi)
            };
            debug_assert_eq!([batch, d_inner], y_bi.dims());

            // Output projection
            let out_bm = self.out_proj.forward(y_bi);
            debug_assert_eq!([batch, d_model], out_bm.dims());

            (out_bm, cache)
        }
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

    let x_segsum = {
        let neg_inf = Tensor::full_like(&diff, f32::NEG_INFINITY);
        let upper_mask = neg_inf.triu(1);
        diff + upper_mask
    };

    x_segsum
}
