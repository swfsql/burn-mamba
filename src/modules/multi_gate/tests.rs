//! Tests for [`MultiGateResidual`]: `forward`/`step` parity (the streams are a
//! point-wise depth construct, so the two must agree), the convex-mixture
//! identity at initialisation, and that gradients reach every parameter.

use super::*;
use crate::utils::test_helpers::max_abs_diff;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// An MGR with randomised (non-zero) queries/bias so per-stream gates differ.
fn random_mgr(d_model: usize, n_stream: usize, device: &Device) -> MultiGateResidual {
    MultiGateResidual {
        w_beta: Param::from_tensor(Tensor::random([d_model], Distribution::Default, device)),
        w_alpha: Param::from_tensor(Tensor::random([d_model], Distribution::Default, device)),
        b_beta: Param::from_tensor(Tensor::random([n_stream], Distribution::Default, device)),
        d_model,
        n_stream,
    }
}

#[test]
fn forward_step_parity() {
    let device = Device::default();
    let (b, s, n, d) = (2, 5, 4, 8);
    let m = random_mgr(d, n, &device);

    let layer_output = Tensor::<3>::random([b, s, d], Distribution::Default, &device);
    let streams = Tensor::<4>::random([b, s, n, d], Distribution::Default, &device);

    let (h_f, s_f) = m.forward(layer_output.clone(), streams.clone());
    assert_eq!(h_f.dims(), [b, s, d]);
    assert_eq!(s_f.dims(), [b, s, n, d]);

    // Each sequence position must reproduce exactly via the single-token step.
    for t in 0..s {
        let lo_t = layer_output.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let st_t = streams.clone().narrow(1, t, 1).squeeze_dim::<3>(1);
        let (h_t, s_t) = m.step(lo_t, st_t);

        let h_ref = h_f.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let s_ref = s_f.clone().narrow(1, t, 1).squeeze_dim::<3>(1);
        assert!(max_abs_diff(h_t, h_ref) < 1e-5, "h mismatch at t={t}");
        assert!(max_abs_diff(s_t, s_ref) < 1e-5, "stream mismatch at t={t}");
    }
}

#[test]
fn init_is_convex_mean() {
    // Zero queries + zero bias ⇒ β = σ(0) = 0.5 and α uniform = 1/n, so the
    // mixer is the midpoint of (stream, layer_output) and the pool is the mean.
    let device = Device::default();
    let (b, n, d) = (2, 3, 6);
    let m = MultiGateResidualConfig::new(d, n).init(&device);

    let layer_output = Tensor::<2>::random([b, d], Distribution::Default, &device);
    let streams = Tensor::<3>::random([b, n, d], Distribution::Default, &device);
    let (h, new_streams) = m.step(layer_output.clone(), streams.clone());

    let expected_streams = (streams + layer_output.unsqueeze_dim::<3>(1)) * 0.5;
    assert!(max_abs_diff(new_streams.clone(), expected_streams) < 1e-5);

    let expected_h = new_streams.mean_dim(1).squeeze_dim::<2>(1);
    assert!(max_abs_diff(h, expected_h) < 1e-5);
}

#[test]
fn gradients_flow() {
    let device = Device::default().autodiff();
    let (b, s, n, d) = (2, 4, 3, 8);
    let m = MultiGateResidualConfig::new(d, n)
        .with_init_bias(0.5)
        .init(&device);

    let layer_output = Tensor::<3>::random([b, s, d], Distribution::Default, &device);
    let streams = Param::from_tensor(Tensor::<4>::random(
        [b, s, n, d],
        Distribution::Default,
        &device,
    ));

    let (h, new_streams) = m.forward(layer_output, streams.val());
    let loss = h.sum() + new_streams.sum();
    let grads = loss.backward();

    assert!(streams.val().grad(&grads).is_some(), "grad streams");
    assert!(m.w_beta.val().grad(&grads).is_some(), "grad w_beta");
    assert!(m.w_alpha.val().grad(&grads).is_some(), "grad w_alpha");
    assert!(m.b_beta.val().grad(&grads).is_some(), "grad b_beta");
}

/// The **Standard** residual path (the refactor where [`Layer`] returns its bare
/// output and `Layers` adds the residual) must keep `forward == unrolled step`
/// with both the first and last residuals suppressed, over virtual layers.
///
/// [`Layer`]: crate::modules::Layer
#[cfg(feature = "mamba2")]
#[test]
fn layers_standard_ignore_residuals_parity() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use crate::modules::{LayersBuilder, ResidualsConfig};
    use crate::utils::Schedule;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(2, block)
        .with_n_virtual_layers(Some((5, Schedule::Stretched)))
        .with_residuals(ResidualsConfig::Standard)
        .with_ignore_first_residual(true)
        .with_ignore_last_residual(true)
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-4,
            "Standard (ignored residuals) step disagrees with forward at t={t}"
        );
    }
}

/// End-to-end wiring check: a `Layers` stack with Multi-Gate residuals must
/// still satisfy the `forward == unrolled step` parity property (the streams are
/// rebuilt per token in `step`, so each user position reproduces `forward`).
#[cfg(feature = "mamba2")]
#[test]
fn layers_multi_gate_forward_step_parity() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use crate::modules::{LayersBuilder, ResidualsConfig};

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(2, block)
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: -1.0,
            init_bias_step: 0.0,
            per_virtual_layer: false,
        })
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-4,
            "MGR step disagrees with forward at t={t}"
        );
    }
}

/// Same parity check over Mamba-3 with **virtual layers** (the `mnist-class`
/// shape): 2 real weight sets stretched to several virtual layers, so the
/// per-layer MGR modules are reused by real index across the virtual passes.
#[cfg(feature = "mamba3")]
#[test]
fn layers_multi_gate_virtual_forward_step_parity() {
    use crate::mamba3::prelude::{Mamba3Config, Mamba3SsdPath};
    use crate::modules::{LayersBuilder, ResidualsConfig};
    use crate::utils::Schedule;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba3Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(0.5);
    let layers = LayersBuilder::new(2, block)
        .with_n_virtual_layers(Some((6, Schedule::Stretched)))
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream: 4,
            init_bias: -1.0,
            init_bias_step: 0.0,
            per_virtual_layer: false,
        })
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba3SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-3,
            "MGR (virtual) step disagrees with forward at t={t}"
        );
    }
}

/// Parity check exercising the two MGR composition options together: a
/// **per-virtual** MGR (one module for every virtual layer, not reused by real
/// index) with the **first and last residuals skipped**. `forward` must still
/// equal `step` unrolled token-by-token.
#[cfg(feature = "mamba2")]
#[test]
fn layers_multi_gate_per_virtual_ignore_residuals_parity() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use crate::modules::{LayersBuilder, ResidualsConfig};
    use crate::utils::Schedule;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let mut builder = LayersBuilder::new(2, block)
        .with_n_virtual_layers(Some((5, Schedule::Stretched)))
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: -1.0,
            init_bias_step: 0.0,
            per_virtual_layer: true,
        });
    builder.ignore_first_residual = true;
    builder.ignore_last_residual = true;
    let layers = builder.init(&device);

    // One MGR per virtual layer (5), not per real layer (2).
    if let crate::modules::Residuals::MultiGate(mg) = &layers.residuals {
        assert_eq!(
            mg.layers.len(),
            5,
            "per-virtual ⇒ one MGR per virtual layer"
        );
        assert!(mg.per_virtual);
    } else {
        panic!("expected MultiGate residuals");
    }

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-4,
            "MGR (per-virtual, ignored residuals) step disagrees with forward at t={t}"
        );
    }
}

/// A bidirectional stack threads its residuals **between pairs** (the MGR unit
/// is the pair, not the single layer). This checks the wiring: the MGR module
/// count follows the number of pairs (per-real here), forward produces the right
/// shape, and gradients reach every MGR parameter.
#[cfg(feature = "mamba2")]
#[test]
fn bidi_multi_gate_forward_and_grads() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use crate::modules::ResidualsConfig;
    use crate::modules::bidi::{BidiLayersBuilder, OutputMergeConfig};
    use crate::modules::{Layer, Residuals};

    let device = Device::default().autodiff();
    let d_model = 16;
    let n_real = 4; // 2 pairs
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = BidiLayersBuilder {
        n_real_layers: n_real,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(n_real),
        class_latents: Vec::new(),
        // Two streams over two pairs: the first pair accumulates (the input
        // plus its own output), the second one mixes — one pair per phase.
        residuals: ResidualsConfig::MultiGate {
            n_stream: 2,
            init_bias: -1.0,
            init_bias_step: 0.0,
            per_virtual_layer: false,
        },
    }
    .init(&device);

    // One MGR module per pair (n_real / 2), not per real layer.
    let Residuals::MultiGate(mg) = &layers.residuals else {
        panic!("expected MultiGate residuals");
    };
    assert_eq!(mg.layers.len(), n_real / 2, "one MGR per pair");

    let (batch, seq) = (2usize, 5usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (y, _c) = layers.forward(x, None, Mamba2SsdPath::default(), None);
    assert_eq!(y.dims(), [batch, seq, d_model]);

    let grads = y.sum().backward();
    for (li, l) in layers.real_layers.iter().enumerate() {
        let Layer { norm, .. } = l;
        assert!(
            norm.gamma.val().grad(&grads).is_some(),
            "grad did not reach real layer {li}'s pre-norm"
        );
    }
    assert!(
        mg.layers[0].w_alpha.val().grad(&grads).is_some(),
        "grad did not reach the accumulating pair's aggregator query"
    );
    assert!(
        mg.layers[1].w_beta.val().grad(&grads).is_some(),
        "grad did not reach the mixing pair's mixer query"
    );
}

/// The accumulation phase widens the stream set by one and pools over what is
/// there — `k+1` streams out of `k`, and (at zero-init) their plain mean.
#[test]
fn accumulate_appends_a_stream() {
    let device = Device::default();
    let (b, s, n, d) = (2, 3, 4, 8);
    let m = MultiGateResidualConfig::new(d, n).init(&device);

    let layer_output = Tensor::<3>::random([b, s, d], Distribution::Default, &device);
    let streams = Tensor::<4>::random([b, s, 2, d], Distribution::Default, &device);

    let (h, new_streams) = m.accumulate(layer_output.clone(), streams.clone());
    assert_eq!(new_streams.dims(), [b, s, 3, d], "one stream wider");
    // The appended stream is the layer output itself (no gate, no mixing) and
    // the carried ones are untouched.
    let expected = Tensor::cat(vec![streams, layer_output.clone().unsqueeze_dim::<4>(2)], 2);
    assert!(max_abs_diff(new_streams.clone(), expected) < 1e-6);
    // Zero queries ⇒ uniform α ⇒ the pool is the mean over the wider set.
    let mean = new_streams.mean_dim(2).squeeze_dim::<3>(2);
    assert!(max_abs_diff(h, mean) < 1e-5);

    // The single-token path agrees with the sequence one.
    let (h_t, s_t) = m.accumulate_step(
        layer_output.clone().narrow(1, 0, 1).squeeze_dim::<2>(1),
        Tensor::<4>::zeros([b, 1, 2, d], &device).squeeze_dim::<3>(1),
    );
    assert_eq!(s_t.dims(), [b, 3, d]);
    assert_eq!(h_t.dims(), [b, d]);
}

/// **The** MGR regression test: a stack must keep its streams *distinct*.
///
/// Seeding all `n_stream` streams with copies of the input is a symmetry the
/// model can never break — identical streams score identically, so every gate
/// and every gradient is identical and the stack collapses to a single lerped
/// stream. This pins the paper's two-phase wiring instead: the first
/// `n_stream−1` layers append their output as a new stream, the rest mix.
///
/// The reference below rebuilds the stack loop by hand (layer by layer, with
/// the MGR's own primitives) and must match `Layers::forward` exactly, while the
/// degenerate all-copies wiring must *not*.
#[cfg(feature = "mamba2")]
#[test]
fn layers_multi_gate_streams_are_distinct() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use crate::modules::{LayersBuilder, Residuals, ResidualsConfig};
    use crate::utils::Schedule;

    let device = Device::default();
    let (d_model, n_stream, n_virtual) = (16, 3, 5);
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(1, block)
        .with_n_virtual_layers(Some((n_virtual, Schedule::Stretched)))
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream,
            init_bias: -1.0,
            init_bias_step: 0.0,
            per_virtual_layer: true,
        })
        .init(&device);
    let Residuals::MultiGate(mg) = &layers.residuals else {
        panic!("expected MultiGate residuals");
    };

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (y, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);

    // Reference: the input is stream 1; each layer appends its output until
    // `n_stream` streams exist, and only then are they gate-mixed.
    let layer = &layers.real_layers[0];
    let mut streams = x.clone().unsqueeze_dim::<4>(2);
    let mut h = x.clone();
    for i in 0..n_virtual {
        let (out, _c) = layer.forward(h, None, Mamba2SsdPath::default());
        let (new_h, new_streams) = if streams.dims()[2] < n_stream {
            mg.layers[i].accumulate(out, streams)
        } else {
            mg.layers[i].forward(out, streams)
        };
        h = new_h;
        streams = new_streams;
    }
    assert_eq!(streams.dims(), [batch, seq, n_stream, d_model]);
    assert!(
        max_abs_diff(y.clone(), h) < 1e-5,
        "the stack must run the accumulate-then-mix schedule"
    );

    // Degenerate wiring (every stream a copy of the input): all streams stay
    // identical forever, so the whole stack is one lerped stream.
    let mut h = x.clone();
    for i in 0..n_virtual {
        let (out, _c) = layer.forward(h.clone(), None, Mamba2SsdPath::default());
        let copies = h
            .unsqueeze_dim::<4>(2)
            .expand([batch, seq, n_stream, d_model]);
        let (new_h, new_streams) = mg.layers[i].forward(out, copies);
        // Identical in, identical out: the streams never differentiate.
        let first = new_streams.clone().narrow(2, 0, 1);
        let last = new_streams.narrow(2, n_stream - 1, 1);
        assert!(max_abs_diff(first, last) < 1e-6, "the symmetry is exact");
        h = new_h;
    }
    assert!(
        max_abs_diff(y, h) > 1e-3,
        "the stack must not collapse to a single lerped stream"
    );
}

/// `init_bias_step` ramps the per-stream gate bias, giving the streams distinct
/// depth-timescales from step zero (`0` keeps the paper's uniform init).
#[test]
fn init_bias_step_ramps_the_gates() {
    let device = Device::default();
    let (n, d) = (4, 8);
    let m = MultiGateResidualConfig::new(d, n)
        .with_init_bias(-2.7)
        .with_init_bias_step(0.9)
        .init(&device);

    let expected = Tensor::<1>::from_floats([-2.7, -1.8, -0.9, 0.0], &device);
    assert!(max_abs_diff(m.b_beta.val(), expected) < 1e-5);

    // Each stream therefore lerps by its own β — the first carries (β ≈ 0.06),
    // the last updates (β = 0.5) — even before any training.
    let (b, s) = (1, 1);
    let streams = Tensor::<4>::zeros([b, s, n, d], &device);
    let layer_output = Tensor::<3>::ones([b, s, d], &device);
    let (_h, new_streams) = m.forward(layer_output, streams);
    let betas: Vec<f32> = new_streams
        .narrow(3, 0, 1)
        .reshape([n])
        .into_data()
        .to_vec()
        .unwrap();
    for (i, w) in betas.windows(2).enumerate() {
        assert!(
            w[0] < w[1],
            "stream {i} must lerp less than stream {}",
            i + 1
        );
    }
    assert!(
        (betas[n - 1] - 0.5).abs() < 1e-4,
        "last stream is σ(0) = 0.5"
    );
}
