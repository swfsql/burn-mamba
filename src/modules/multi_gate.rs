//! Multi-Gate Residuals (MGR) — a depth-wise residual scheme replacing the plain
//! additive skip of a [`Layers`](crate::modules::Layers) stack.
//!
//! Instead of one residual stream, MGR keeps **`n_stream` parallel streams**
//! `sᵢ` (all seeded from the stack input). Between layers, one
//! [`MultiGateResidual`] per layer does two convex, norm-bounded operations
//! (paper §"Our Architecture"):
//!
//! 1. **Mixer** (independent sigmoid gate) — each stream is interpolated towards
//!    the current layer output `F_l` by a per-stream gate `βᵢ`:
//!    `sᵢ' = (1−βᵢ)·sᵢ + βᵢ·F_l`, with
//!    `βᵢ = σ( (w⁽ᵝ⁾ · RMSNorm(sᵢ))/√d + b⁽ᵝ⁾ᵢ )`.
//! 2. **Aggregator** (depth-wise attention pooling, "AttnPool") — the updated
//!    streams are pooled into the next layer's input `h` by a softmax over
//!    streams: `αᵢ = softmax_i( (w⁽ᵅ⁾ · RMSNorm(sᵢ'))/√d )`, `h = Σᵢ αᵢ·sᵢ'`.
//!
//! Both `w` vectors are learnable in `ℝ^d` (init zero), the RMSNorm is
//! parameter-free, and `b⁽ᵝ⁾` is a per-stream learnable bias. Only the
//! **independent** (sigmoid) gate is implemented; the paper's competitive
//! (softmax) variant is omitted.
//!
//! MGR is purely **point-wise over `(batch, sequence)`** — the streams only
//! evolve along *depth*, never along the sequence — so `forward` over a sequence
//! equals `step` unrolled token-by-token, and `step` carries no extra state
//! (each token rebuilds its own depth-streams).
//!
//! **Gate-bias initialisation.** Following Highway Networks, a negative
//! `init_bias` biases the gates towards *carry* (small updates) at the start of
//! training. The paper scales it with depth `L`:
//! `b_init = ln( √(L/L_base)·(exp(−b_base)+1) − n )` (with `L_base = 21`,
//! `b_base = −3`); here `init_bias` is taken directly so the caller may apply
//! that formula. Default `0` (gates open at `σ(0)=0.5`).

use crate::modules::bidi::NoOp;
use crate::utils::div_eps;
use burn::config::Config;
use burn::module::Param;
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{DType, f16};

/// One layer's Multi-Gate Residual parameters: the mixer query `w⁽ᵝ⁾` + bias
/// `b⁽ᵝ⁾`, and the aggregator (AttnPool) query `w⁽ᵅ⁾`.
#[derive(Module, Debug)]
pub struct MultiGateResidual {
    /// Mixer query `w⁽ᵝ⁾ ∈ ℝ^d` (the per-stream sigmoid gate), `[d_model]`.
    pub w_beta: Param<Tensor<1>>,
    /// Aggregator query `w⁽ᵅ⁾ ∈ ℝ^d` (the AttnPool softmax), `[d_model]`.
    pub w_alpha: Param<Tensor<1>>,
    /// Per-stream mixer gate bias `b⁽ᵝ⁾`, `[n_stream]`.
    pub b_beta: Param<Tensor<1>>,
    /// Model width `d`.
    #[module(skip)]
    pub d_model: usize,
    /// Number of parallel residual streams `n`.
    #[module(skip)]
    pub n_stream: usize,
}

impl MultiGateResidual {
    fn scale(&self) -> f32 {
        (self.d_model as f32).powf(-0.5)
    }

    /// Parameter-free RMS norm over the last dim (matches [`RmsNorm`] math with
    /// `γ ≡ 1`; the fp16 path uses the same overflow-safe max-rescale).
    ///
    /// [`RmsNorm`]: crate::modules::RmsNorm
    fn rms_norm<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match x.dtype() {
            DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
                let eps = div_eps(x.dtype());
                let rms = (x.clone() * x.clone()).mean_dim(D - 1).sqrt();
                x / (rms + eps)
            }
            DType::F16 => {
                use burn::tensor::ElementConversion;
                let eps: f16 = f16::from_elem(div_eps(x.dtype())) * f16::from_f32(2.);
                let max = x.clone().no_grad().detach().abs().max().expand(x.shape());
                let x_ = x.clone() / (max.clone() + eps); // x_.abs() <= 1
                let rms_partial = (x.clone() * x_).mean_dim(D - 1).sqrt();
                x / (rms_partial + eps) / max.sqrt()
            }
            _ => unreachable!("rms_norm expects a float dtype"),
        }
    }

    /// Full-sequence mix + pool.
    ///
    /// - `layer_output`: this layer's transform `F_l`, `[batch, sequence, d_model]`
    /// - `streams`: the `n_stream` residual streams, `[batch, sequence, n_stream, d_model]`
    ///
    /// Returns `(h, streams')`: the pooled input `h` for the next layer
    /// (`[batch, sequence, d_model]`) and the updated streams (same shape as in).
    pub fn forward(
        &self,
        layer_output: Tensor<3>,
        streams: Tensor<4>,
    ) -> (Tensor<3>, Tensor<4>) {
        let [batch, sequence, n_stream, d_model] = streams.dims();
        assert_eq!(d_model, self.d_model, "stream width must equal d_model");
        assert_eq!(n_stream, self.n_stream, "stream count must equal n_stream");
        let scale = self.scale();

        // Mixer: independent per-stream sigmoid gate.
        let normed_s = self.rms_norm(streams.clone());
        let score_s: Tensor<3> = (normed_s * self.w_beta.val().unsqueeze::<4>())
            .sum_dim(3)
            .squeeze_dim(3);
        let logit = score_s * scale + self.b_beta.val().unsqueeze::<3>();
        let beta = sigmoid(logit); // [batch, sequence, n_stream]
        let delta = layer_output.unsqueeze_dim::<4>(2) - streams.clone();
        let new_streams = streams + beta.unsqueeze_dim::<4>(3) * delta;

        // Aggregator: depth-wise attention pooling (softmax over streams).
        let normed_ns = self.rms_norm(new_streams.clone());
        let score_ns: Tensor<3> = (normed_ns * self.w_alpha.val().unsqueeze::<4>())
            .sum_dim(3)
            .squeeze_dim(3);
        let alpha = softmax(score_ns * scale, 2); // over streams
        let new_h: Tensor<3> = (alpha.unsqueeze_dim::<4>(3) * new_streams.clone())
            .sum_dim(2)
            .squeeze_dim(2);
        (new_h, new_streams)
    }

    /// Single-token mix + pool (the [`Self::forward`] math with the sequence axis
    /// dropped).
    ///
    /// - `layer_output`: `[batch, d_model]`
    /// - `streams`: `[batch, n_stream, d_model]`
    ///
    /// Returns `(h, streams')`: `[batch, d_model]` and `[batch, n_stream, d_model]`.
    pub fn step(&self, layer_output: Tensor<2>, streams: Tensor<3>) -> (Tensor<2>, Tensor<3>) {
        let [batch, n_stream, d_model] = streams.dims();
        assert_eq!(d_model, self.d_model, "stream width must equal d_model");
        assert_eq!(n_stream, self.n_stream, "stream count must equal n_stream");
        let scale = self.scale();

        let normed_s = self.rms_norm(streams.clone());
        let score_s: Tensor<2> = (normed_s * self.w_beta.val().unsqueeze::<3>())
            .sum_dim(2)
            .squeeze_dim(2);
        let logit = score_s * scale + self.b_beta.val().unsqueeze::<2>();
        let beta = sigmoid(logit); // [batch, n_stream]
        let delta = layer_output.unsqueeze_dim::<3>(1) - streams.clone();
        let new_streams = streams + beta.unsqueeze_dim::<3>(2) * delta;

        let normed_ns = self.rms_norm(new_streams.clone());
        let score_ns: Tensor<2> = (normed_ns * self.w_alpha.val().unsqueeze::<3>())
            .sum_dim(2)
            .squeeze_dim(2);
        let alpha = softmax(score_ns * scale, 1);
        let new_h: Tensor<2> = (alpha.unsqueeze_dim::<3>(2) * new_streams.clone())
            .sum_dim(1)
            .squeeze_dim(1);
        (new_h, new_streams)
    }
}

/// Configuration for a single [`MultiGateResidual`].
#[derive(Config, Debug)]
pub struct MultiGateResidualConfig {
    /// Model width `d`.
    pub d_model: usize,
    /// Number of parallel residual streams `n`.
    pub n_stream: usize,
    /// Initial value for every entry of the gate bias `b⁽ᵝ⁾` (see module header).
    #[config(default = 0.0)]
    pub init_bias: f64,
}

impl MultiGateResidualConfig {
    /// Allocate one layer's MGR parameters (`w⁽ᵝ⁾`, `w⁽ᵅ⁾` zero; `b⁽ᵝ⁾` constant).
    pub fn init(&self, device: &Device) -> MultiGateResidual {
        MultiGateResidual {
            w_beta: Initializer::Zeros.init::<1, _>([self.d_model], device),
            w_alpha: Initializer::Zeros.init::<1, _>([self.d_model], device),
            b_beta: Param::from_tensor(Tensor::full([self.n_stream], self.init_bias, device)),
            d_model: self.d_model,
            n_stream: self.n_stream,
        }
    }
}

/// A stack of [`MultiGateResidual`]s — one per *real* layer of the enclosing
/// [`Layers`](crate::modules::Layers) (virtual layers reuse them by real index).
#[derive(Module, Debug)]
pub struct MultiGate {
    /// One MGR module per real layer.
    pub layers: Vec<MultiGateResidual>,
    /// Number of parallel residual streams `n`.
    #[module(skip)]
    pub n_stream: usize,
}

/// How a [`Layers`](crate::modules::Layers) stack threads residuals between
/// layers: the plain additive skip, or Multi-Gate Residuals.
#[derive(Module, Debug)]
pub enum Residuals {
    /// Plain Pre-LN additive residual — each [`Layer`](crate::modules::Layer)
    /// adds its own skip connection.
    Standard(NoOp),
    /// Multi-Gate Residuals: `n_stream` parallel streams with per-layer gated
    /// mixing + attention pooling.
    MultiGate(MultiGate),
}

/// Configuration / factory for [`Residuals`].
#[derive(Config, Debug)]
pub enum ResidualsConfig {
    /// Plain additive Pre-LN residual.
    Standard,
    /// Multi-Gate Residuals over `n_stream` streams.
    MultiGate {
        /// Number of parallel residual streams `n`.
        n_stream: usize,
        /// Initial gate bias (see [`MultiGateResidualConfig::init_bias`]).
        init_bias: f64,
    },
}

impl ResidualsConfig {
    /// Build the runtime [`Residuals`] for a stack of `n_real_layers` layers.
    pub fn init(&self, d_model: usize, n_real_layers: usize, device: &Device) -> Residuals {
        match self {
            ResidualsConfig::Standard => Residuals::Standard(NoOp),
            ResidualsConfig::MultiGate {
                n_stream,
                init_bias,
            } => {
                let layers = (0..n_real_layers)
                    .map(|_| {
                        MultiGateResidualConfig::new(d_model, *n_stream)
                            .with_init_bias(*init_bias)
                            .init(device)
                    })
                    .collect();
                Residuals::MultiGate(MultiGate {
                    layers,
                    n_stream: *n_stream,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests;
