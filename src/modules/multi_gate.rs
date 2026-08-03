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

    /// The parameter-free RMS denominator `d(x) ∈ [‥, 1]` such that the RMSNorm
    /// (matching [`RmsNorm`] math with `γ ≡ 1`) is `x / d(x)`. Returning the
    /// denominator rather than the normalised tensor lets [`Self::normed_score`]
    /// fold it out of the (feature-axis) score reduction, so the full-width
    /// normalised tensor is never built. The fp16 path keeps the same
    /// overflow-safe max-rescale, folded into the same scalar denominator.
    ///
    /// [`RmsNorm`]: crate::modules::RmsNorm
    fn rms_denom<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match x.dtype() {
            DType::F64 | DType::F32 | DType::Flex32 | DType::BF16 => {
                let eps = div_eps(x.dtype());
                (x.clone() * x).mean_dim(D - 1).sqrt() + eps
            }
            DType::F16 => {
                use burn::tensor::ElementConversion;
                let eps: f16 = f16::from_elem(div_eps(x.dtype())) * f16::from_f32(2.);
                // Single global scalar `max`, reshaped to `[1; D]` so it
                // broadcasts against the `[‥, 1]` partial RMS.
                let max = x.clone().no_grad().detach().abs().max().reshape([1; D]);
                let x_ = x.clone() / (max.clone() + eps); // x_.abs() <= 1
                let rms_partial = (x.clone() * x_).mean_dim(D - 1).sqrt();
                (rms_partial + eps) * max.sqrt()
            }
            _ => unreachable!("rms_denom expects a float dtype"),
        }
    }

    /// The RMSNorm-then-dot score `scale · Σ_feat(x · w) / (rms(x)+eps)`,
    /// shape `[‥, 1]`. The RMS denominator is constant over the feature axis, so
    /// it is folded out of the reduction (via [`Self::rms_denom`]) — equal to
    /// `Σ_feat(rms_norm(x) · w) · scale` but without materialising the full-width
    /// normalised tensor.
    fn normed_score<const R: usize>(&self, x: Tensor<R>, w: Tensor<R>) -> Tensor<R> {
        let dot = (x.clone() * w).sum_dim(R - 1);
        dot * self.scale() / self.rms_denom(x)
    }

    /// The shared mix + pool, generic over the streams rank `R` (the *stream*
    /// axis is `R-2`, the *feature* axis `R-1`). [`Self::forward`] (`R = 4`) and
    /// [`Self::step`] (`R = 3`) only differ by that rank, so both lift their
    /// `layer_output` to a singleton stream axis, call this, and drop it again.
    /// All reductions keep their axis (size 1) for broadcasting, so scores/gates
    /// are `[…, n_stream, 1]` throughout.
    ///
    /// - `layer_output`: `F_l` lifted to a unit stream axis, `[…, 1, d_model]`
    /// - `streams`: the `n_stream` residual streams, `[…, n_stream, d_model]`
    ///
    /// Returns `(h, streams')` with `h` still carrying its unit stream axis
    /// (`[…, 1, d_model]`) and `streams'` the same shape as `streams`.
    fn mix_pool<const R: usize>(
        &self,
        layer_output: Tensor<R>,
        streams: Tensor<R>,
    ) -> (Tensor<R>, Tensor<R>) {
        let dims = streams.dims();
        let (stream_axis, feat_axis) = (R - 2, R - 1);
        assert_eq!(
            dims[feat_axis], self.d_model,
            "stream width must equal d_model"
        );
        assert_eq!(
            dims[stream_axis], self.n_stream,
            "stream count must equal n_stream"
        );

        // `b_beta` reshaped to broadcast on the stream axis: `[1, …, n_stream, 1]`.
        let mut bias_shape = [1usize; R];
        bias_shape[stream_axis] = self.n_stream;
        let b_beta = self.b_beta.val().reshape(bias_shape);
        // The query vectors broadcast on the feature axis: `[1, …, 1, d_model]`.
        let w_beta = self.w_beta.val().unsqueeze::<R>();
        let w_alpha = self.w_alpha.val().unsqueeze::<R>();

        // Mixer: independent per-stream sigmoid gate, `β`: `[…, n_stream, 1]`.
        let beta = sigmoid(self.normed_score(streams.clone(), w_beta) + b_beta);
        // Lerp `(1−β)·streams + β·layer_output` (equal to the paper's
        // `streams + β·(layer_output − streams)`) — written so no full-width
        // intermediate is retained: `streams` is the already-saved input and
        // `layer_output` is `[…, 1, d_model]`, so neither `mul` saves a new
        // `[…, n_stream, d_model]` tensor and the `+` saves nothing.
        let new_streams = streams * (-beta.clone() + 1.0) + layer_output * beta;

        // Aggregator: depth-wise attention pooling (softmax over the stream axis).
        let score_ns = self.normed_score(new_streams.clone(), w_alpha);
        let alpha = softmax(score_ns, stream_axis);
        let new_h = (alpha * new_streams.clone()).sum_dim(stream_axis);
        (new_h, new_streams)
    }

    /// Full-sequence mix + pool.
    ///
    /// - `layer_output`: this layer's transform `F_l`, `[batch, sequence, d_model]`
    /// - `streams`: the `n_stream` residual streams, `[batch, sequence, n_stream, d_model]`
    ///
    /// Returns `(h, streams')`: the pooled input `h` for the next layer
    /// (`[batch, sequence, d_model]`) and the updated streams (same shape as in).
    pub fn forward(&self, layer_output: Tensor<3>, streams: Tensor<4>) -> (Tensor<3>, Tensor<4>) {
        let (new_h, new_streams) = self.mix_pool::<4>(layer_output.unsqueeze_dim(2), streams);
        (new_h.squeeze_dim(2), new_streams)
    }

    /// Single-token mix + pool (the [`Self::forward`] math with the sequence axis
    /// dropped).
    ///
    /// - `layer_output`: `[batch, d_model]`
    /// - `streams`: `[batch, n_stream, d_model]`
    ///
    /// Returns `(h, streams')`: `[batch, d_model]` and `[batch, n_stream, d_model]`.
    pub fn step(&self, layer_output: Tensor<2>, streams: Tensor<3>) -> (Tensor<2>, Tensor<3>) {
        let (new_h, new_streams) = self.mix_pool::<3>(layer_output.unsqueeze_dim(1), streams);
        (new_h.squeeze_dim(1), new_streams)
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

/// A stack of [`MultiGateResidual`]s for the enclosing
/// [`Layers`](crate::modules::Layers). When `per_virtual` is `false` there is one
/// module **per real layer** (virtual layers reuse them by real index); when
/// `true` there is one **per virtual layer** (each virtual pass owns its own).
#[derive(Module, Debug)]
pub struct MultiGate {
    /// The MGR modules: length `n_real_layers` (per-real) or `n_virtual_layers`
    /// (per-virtual) — see [`Self::per_virtual`].
    pub layers: Vec<MultiGateResidual>,
    /// Number of parallel residual streams `n`.
    #[module(skip)]
    pub n_stream: usize,
    /// `true` ⇒ one MGR per *virtual* layer (indexed by virtual position);
    /// `false` ⇒ one per *real* layer (reused across virtual passes by real index).
    #[module(skip)]
    pub per_virtual: bool,
}

impl MultiGate {
    /// Index into [`Self::layers`] for a given `(virtual_idx, real_idx)` layer
    /// position: the virtual index when each virtual layer owns its MGR
    /// ([`Self::per_virtual`]), otherwise the real index.
    pub fn module_index(&self, virtual_idx: usize, real_idx: usize) -> usize {
        if self.per_virtual {
            virtual_idx
        } else {
            real_idx
        }
    }
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
        /// `true` ⇒ one MGR per *virtual* layer; `false` ⇒ one per *real* layer
        /// (reused across virtual passes). See [`MultiGate::per_virtual`].
        per_virtual_layer: bool,
    },
}

impl ResidualsConfig {
    /// Build the runtime [`Residuals`] for a stack of `n_real_layers` real weight
    /// sets unrolled over `n_virtual_layers` (virtual) passes. The MGR module
    /// count follows `per_virtual_layer` (one per virtual layer vs one per real
    /// layer).
    pub fn init(
        &self,
        d_model: usize,
        n_real_layers: usize,
        n_virtual_layers: usize,
        device: &Device,
    ) -> Residuals {
        match self {
            ResidualsConfig::Standard => Residuals::Standard(NoOp),
            ResidualsConfig::MultiGate {
                n_stream,
                init_bias,
                per_virtual_layer,
            } => {
                let count = if *per_virtual_layer {
                    n_virtual_layers
                } else {
                    n_real_layers
                };
                let layers = (0..count)
                    .map(|_| {
                        MultiGateResidualConfig::new(d_model, *n_stream)
                            .with_init_bias(*init_bias)
                            .init(device)
                    })
                    .collect();
                Residuals::MultiGate(MultiGate {
                    layers,
                    n_stream: *n_stream,
                    per_virtual: *per_virtual_layer,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests;
