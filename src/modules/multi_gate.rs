//! Multi-Gate Residuals (MGR) — a depth-wise residual scheme replacing the plain
//! additive skip of a [`Layers`](crate::modules::Layers) stack.
//!
//! Instead of one residual stream, MGR keeps up to **`n_stream` parallel
//! streams** `sᵢ`. The stack input is stream 1; the streams then grow and are
//! mixed in two phases (paper §"Our Architecture"), with one
//! [`MultiGateResidual`] per layer:
//!
//! * **Accumulation** — while fewer than `n_stream` streams exist, the layer
//!   output `F_l` is *appended* as a new stream ([`MultiGateResidual::accumulate`]). This is
//!   what makes the streams **distinct**: after the first `n_stream−1` layers
//!   they hold `[x, F₀, …, F_{n−2}]` (a bounded AttnRes over the early layers).
//! * **Mixing** — from then on the stream count is capped and each stream is
//!   interpolated towards `F_l` by its own gate ([`MultiGateResidual::forward`]).
//!
//! Both phases end in the same aggregator, and both are convex (norm-bounded):
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
//! Seeding every stream with a *copy* of the stack input instead would be a
//! symmetry the model can never break — identical streams give identical
//! scores, hence identical gates and gradients, collapsing MGR to a single
//! lerped stream. The accumulation phase is what avoids that.
//!
//! MGR is purely **point-wise over `(batch, sequence)`** — the streams only
//! evolve along *depth*, never along the sequence — so `forward` over a sequence
//! equals `step` unrolled token-by-token, and `step` carries no extra state
//! (each token rebuilds its own depth-streams).
//!
//! **Class tokens / latents.** A class marker spliced into the sequence enters
//! the token stream *and* every residual stream at that position (`k` copies of
//! the one embedding row). Identical streams score alike, so the aggregator —
//! a convex combination — reproduces the row exactly: the marker reaches the
//! layer above just as the plain additive skip would have handed it on. Markers
//! spliced *below* the stack (a network's [`ClassToken`]s, the stack's own
//! [`ClassLatent`]s) simply seed the streams like any other token.
//!
//! [`ClassToken`]: crate::utils::ClassToken
//! [`ClassLatent`]: crate::utils::ClassLatent
//!
//! **Gate-bias initialisation.** Following Highway Networks, a negative
//! `init_bias` biases the gates towards *carry* (small updates) at the start of
//! training. The paper scales it with the number of *mixing* layers `L`
//! (total layers minus the `n−1` accumulating ones) — see
//! [`MultiGateResidualConfig::depth_init_bias`]. `init_bias` is taken directly,
//! so the caller applies that formula (or picks its own). Default `0` (gates
//! open at `σ(0)=0.5`).
//!
//! That formula assumes one *shared* timescale for all `n` streams, which is a
//! long-training-run choice: it buys stability by making every layer's
//! contribution small (`σ(b) ∝ 1/√L`), and the streams then differ only by what
//! they were seeded with. [`MultiGateResidualConfig::init_bias_step`] instead
//! spreads the initial biases over the streams — an update-biased stream down to
//! a carry-biased one — so the set covers several depth-timescales from step
//! zero. `0` keeps the paper's uniform init.
//!
//! **Output scale.** Every step here is a convex combination, and the aggregator
//! is a *mean* over the `n` streams, so the stack's output stays `O(1)` where a
//! plain additive skip grows it with depth (`x + Σ F_l`) — measured at ~15×
//! larger for a 16-layer stack. Nothing downstream is wrong with either, but a
//! head initialised for one sees the other far off its range, and closing that
//! gap by training `out_proj` alone takes many steps. A norm before the head
//! makes it moot: [`VocabNetwork`] always has one, while on a [`LatentNetwork`]
//! it is the opt-in [`final_norm`]. Prefer enabling it when pairing MGR with a
//! latent head, or expect a slow start.
//!
//! [`VocabNetwork`]: crate::modules::network::VocabNetwork
//! [`LatentNetwork`]: crate::modules::network::LatentNetwork
//! [`final_norm`]: crate::modules::network::LatentNetwork::norm_f

use crate::modules::bidi::NoOp;
use crate::modules::{normed_score, score_scale};
use burn::config::Config;
use burn::module::Param;
use burn::nn::Initializer;
use burn::prelude::*;
use burn::tensor::activation::{sigmoid, softmax};

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
    /// This module's score temperature `1/√d`.
    fn scale(&self) -> f64 {
        score_scale(self.d_model)
    }

    /// The RMSNorm-then-dot score of each stream against `w`, shape
    /// `[‥, n_stream, 1]` — [`normed_score`], at this module's width.
    fn normed_score<const R: usize>(&self, x: Tensor<R>, w: Tensor<R>) -> Tensor<R> {
        normed_score(x, w, self.scale())
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
        let stream_axis = R - 2;
        assert_eq!(
            streams.dims()[stream_axis],
            self.n_stream,
            "the mixing phase runs at the full stream count"
        );

        // `b_beta` reshaped to broadcast on the stream axis: `[1, …, n_stream, 1]`.
        let mut bias_shape = [1usize; R];
        bias_shape[stream_axis] = self.n_stream;
        let b_beta = self.b_beta.val().reshape(bias_shape);
        // The query vector broadcasts on the feature axis: `[1, …, 1, d_model]`.
        let w_beta = self.w_beta.val().unsqueeze::<R>();

        // Mixer: independent per-stream sigmoid gate, `β`: `[…, n_stream, 1]`.
        let beta = sigmoid(self.normed_score(streams.clone(), w_beta) + b_beta);
        // Lerp `(1−β)·streams + β·layer_output` (equal to the paper's
        // `streams + β·(layer_output − streams)`) — written so no full-width
        // intermediate is retained: `streams` is the already-saved input and
        // `layer_output` is `[…, 1, d_model]`, so neither `mul` saves a new
        // `[…, n_stream, d_model]` tensor and the `+` saves nothing.
        let new_streams = streams * (-beta.clone() + 1.0) + layer_output * beta;

        (self.attn_pool(new_streams.clone()), new_streams)
    }

    /// The accumulation-phase counterpart of [`Self::mix_pool`]: append
    /// `layer_output` (`[…, 1, d_model]`) to `streams` (`[…, k, d_model]`,
    /// `k < n_stream`) as a new stream, then pool as usual. Returns
    /// `(h, streams')` with `streams'` one stream wider.
    fn append_pool<const R: usize>(
        &self,
        layer_output: Tensor<R>,
        streams: Tensor<R>,
    ) -> (Tensor<R>, Tensor<R>) {
        assert!(
            streams.dims()[R - 2] < self.n_stream,
            "the accumulation phase stops at `n_stream` streams"
        );
        let new_streams = Tensor::cat(vec![streams, layer_output], R - 2);
        (self.attn_pool(new_streams.clone()), new_streams)
    }

    /// Aggregator: depth-wise attention pooling over any stream count `k ≥ 1`
    /// (softmax over the stream axis `R-2`), keeping that axis at size 1 so the
    /// result still broadcasts against the streams.
    fn attn_pool<const R: usize>(&self, streams: Tensor<R>) -> Tensor<R> {
        let (stream_axis, feat_axis) = (R - 2, R - 1);
        assert_eq!(
            streams.dims()[feat_axis],
            self.d_model,
            "stream width must equal d_model"
        );
        let w_alpha = self.w_alpha.val().unsqueeze::<R>();
        let alpha = softmax(self.normed_score(streams.clone(), w_alpha), stream_axis);
        (alpha * streams).sum_dim(stream_axis)
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

    /// Accumulation phase of [`Self::forward`]: `layer_output` becomes a **new**
    /// stream instead of being mixed into the existing ones (the mixer gate is
    /// not used at all). Only valid while `streams` holds fewer than `n_stream`
    /// streams.
    ///
    /// - `layer_output`: this layer's transform `F_l`, `[batch, sequence, d_model]`
    /// - `streams`: `[batch, sequence, k, d_model]` with `k < n_stream`
    ///
    /// Returns `(h, streams')`, `streams'` being `[batch, sequence, k+1, d_model]`.
    pub fn accumulate(
        &self,
        layer_output: Tensor<3>,
        streams: Tensor<4>,
    ) -> (Tensor<3>, Tensor<4>) {
        let (new_h, new_streams) = self.append_pool::<4>(layer_output.unsqueeze_dim(2), streams);
        (new_h.squeeze_dim(2), new_streams)
    }

    /// Single-token [`Self::accumulate`] (the sequence axis dropped), the
    /// accumulation-phase counterpart of [`Self::step`].
    ///
    /// - `layer_output`: `[batch, d_model]`
    /// - `streams`: `[batch, k, d_model]` with `k < n_stream`
    ///
    /// Returns `(h, streams')`: `[batch, d_model]` and `[batch, k+1, d_model]`.
    pub fn accumulate_step(
        &self,
        layer_output: Tensor<2>,
        streams: Tensor<3>,
    ) -> (Tensor<2>, Tensor<3>) {
        let (new_h, new_streams) = self.append_pool::<3>(layer_output.unsqueeze_dim(1), streams);
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
    /// Initial gate bias `b⁽ᵝ⁾` of the **first** stream (see module header).
    #[config(default = 0.0)]
    pub init_bias: f64,
    /// Per-stream offset added on top of [`Self::init_bias`]: stream `i` starts
    /// at `init_bias + i · init_bias_step`, so a negative step gives the streams
    /// **different timescales** — an update-biased first stream down to a
    /// carry-biased last one. `0` (the default) starts them all equal, which is
    /// what the paper prescribes.
    #[config(default = 0.0)]
    pub init_bias_step: f64,
}

impl MultiGateResidualConfig {
    /// The paper's depth-adaptive gate bias for the independent (sigmoid) mixer:
    /// `−ln( √(L/21)·(e³+1) − n )`, where `L = n_mixing_layers` is the number of
    /// layers that actually *lerp* (the stack depth minus the `n−1` accumulating
    /// ones) and `n = n_stream`.
    ///
    /// It keeps the total per-layer increment `O(1)` — `σ(b) ∝ 1/√L` — so the
    /// gates start biased towards *carry*, as in Highway Networks. Panics when
    /// the stack is too shallow for the stream count (`√(L/21)·(e³+1) ≤ n`,
    /// i.e. no carry budget is left to distribute).
    pub fn depth_init_bias(n_mixing_layers: usize, n_stream: usize) -> f64 {
        const L_BASE: f64 = 21.0;
        const B_BASE: f64 = -3.0;
        let inner = (n_mixing_layers as f64 / L_BASE).sqrt() * ((-B_BASE).exp() + 1.0);
        assert!(
            inner > n_stream as f64,
            "depth_init_bias: {n_mixing_layers} mixing layers are too few for \
             {n_stream} streams (pick a smaller `n_stream` or set the bias directly)"
        );
        -(inner - n_stream as f64).ln()
    }

    /// Allocate one layer's MGR parameters (`w⁽ᵝ⁾`, `w⁽ᵅ⁾` zero; `b⁽ᵝ⁾` the
    /// arithmetic ramp `init_bias + i · init_bias_step`).
    pub fn init(&self, device: &Device) -> MultiGateResidual {
        let ramp = Tensor::<1, Int>::arange(0..self.n_stream as i64, device).float();
        MultiGateResidual {
            w_beta: Initializer::Zeros.init::<1, _>([self.d_model], device),
            w_alpha: Initializer::Zeros.init::<1, _>([self.d_model], device),
            b_beta: Param::from_tensor(ramp * self.init_bias_step + self.init_bias),
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
        /// First stream's initial gate bias (see
        /// [`MultiGateResidualConfig::init_bias`]).
        init_bias: f64,
        /// Per-stream gate-bias offset (see
        /// [`MultiGateResidualConfig::init_bias_step`]); `0` starts every
        /// stream equal.
        init_bias_step: f64,
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
                init_bias_step,
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
                            .with_init_bias_step(*init_bias_step)
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
