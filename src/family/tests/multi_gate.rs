//! Tests for [`MultiGateResidual`]: `forward`/`step` parity (the streams are a
//! point-wise depth construct, so the two must agree), the convex-mixture
//! identity at initialisation, and that gradients reach every parameter.

use crate::prelude::*;
use burn::module::Param;
use burn::prelude::*;
use burn_stack::modules::multi_gate::MultiGateResidual;
use burn_stack::utils::test_helpers::max_abs_diff;
#[cfg(feature = "mamba2")]
use burn_stack::utils::test_helpers::max_rel_diff;
use burn::tensor::Distribution;
use burn_stack::utils::test_helpers::{dtype_tol, test_device};

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
    let device = test_device();
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
        assert!(max_abs_diff(h_t, h_ref) < dtype_tol(1e-5), "h mismatch at t={t}");
        assert!(max_abs_diff(s_t, s_ref) < dtype_tol(1e-5), "stream mismatch at t={t}");
    }
}

#[test]
fn init_is_convex_mean() {
    // Zero queries + zero bias ⇒ β = σ(0) = 0.5 and α uniform = 1/n, so the
    // β gate is the midpoint of (stream, layer_output) and the pool is the mean.
    let device = test_device();
    let (b, n, d) = (2, 3, 6);
    let m = MultiGateResidualConfig::new(d, n).init(&device);

    let layer_output = Tensor::<2>::random([b, d], Distribution::Default, &device);
    let streams = Tensor::<3>::random([b, n, d], Distribution::Default, &device);
    let (h, new_streams) = m.step(layer_output.clone(), streams.clone());

    let expected_streams = (streams + layer_output.unsqueeze_dim::<3>(1)) * 0.5;
    assert!(max_abs_diff(new_streams.clone(), expected_streams) < dtype_tol(1e-5));

    let expected_h = new_streams.mean_dim(1).squeeze_dim::<2>(1);
    assert!(max_abs_diff(h, expected_h) < dtype_tol(1e-5));
}

#[test]
fn gradients_flow() {
    let device = test_device().autodiff();
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
/// [`Layer`]: burn_stack::modules::Layer
#[cfg(feature = "mamba2")]
#[test]
fn layers_standard_ignore_residuals_parity() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use burn_stack::modules::{LayersBuilder, ResidualsConfig};
    use burn_stack::utils::Schedule;

    let device = test_device();
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

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None, None);
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_rel_diff(yt, expected) < dtype_tol(1e-4),
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
    use burn_stack::modules::{LayersBuilder, ResidualsConfig};

    let device = test_device();
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
            init_bias_offset: 0.0,
            per_virtual_layer: false,
        })
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None, None);
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_rel_diff(yt, expected) < dtype_tol(1e-4),
            "MGR step disagrees with forward at t={t}"
        );
    }
}

/// Same parity check over Mamba-3 with **virtual layers** (the `mnist-class`
/// shape): 2 real layers stretched to several virtual layers, so the
/// per-layer MGR modules are reused by real index across the virtual passes.
#[cfg(feature = "mamba3")]
#[test]
fn layers_multi_gate_virtual_forward_step_parity() {
    use crate::mamba3::prelude::{Mamba3Config, Mamba3SsdPath};
    use burn_stack::modules::{LayersBuilder, ResidualsConfig};
    use burn_stack::utils::Schedule;

    let device = test_device();
    let d_model = 16;
    let block = Mamba3Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rotation_fraction(0.5);
    let layers = LayersBuilder::new(2, block)
        .with_n_virtual_layers(Some((6, Schedule::Stretched)))
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream: 4,
            init_bias: -1.0,
            init_bias_offset: 0.0,
            per_virtual_layer: false,
        })
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba3SsdPath::default(), None, None);
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < dtype_tol(1e-3),
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
    use burn_stack::modules::{LayersBuilder, ResidualsConfig};
    use burn_stack::utils::Schedule;

    let device = test_device();
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
            init_bias_offset: 0.0,
            per_virtual_layer: true,
        });
    builder.ignore_first_residual = true;
    builder.ignore_last_residual = true;
    let layers = builder.init(&device);

    // One MGR per virtual layer (5), not per real layer (2).
    if let burn_stack::modules::Residuals::MultiGate(mg) = &layers.residuals {
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

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None, None);
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_rel_diff(yt, expected) < dtype_tol(1e-4),
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
    use burn_stack::modules::ResidualsConfig;
    use burn_stack::modules::bidi::{BidiLayersBuilder, OutputMergeConfig};
    use burn_stack::modules::{Layer, Residuals};

    let device = test_device().autodiff();
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
        block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(n_real),
        class_latents: Vec::new(),
        // Two streams over two pairs: the first pair accumulates (the input
        // plus its own output), the second one mixes — one pair per phase.
        residuals: ResidualsConfig::MultiGate {
            n_stream: 2,
            init_bias: -1.0,
            init_bias_offset: 0.0,
            per_virtual_layer: false,
        },
        untied: Vec::new(),
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
    let (y, _c) = layers.forward(x, None, Mamba2SsdPath::default(), None, None);
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
        "grad did not reach the β-gate query of the mixing pair"
    );
}

/// The accumulation phase widens the stream set by one and pools over what is
/// there — `k+1` streams out of `k`, and (at zero-init) their plain mean.
#[test]
fn accumulate_appends_a_stream() {
    let device = test_device();
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
    assert!(max_abs_diff(h, mean) < dtype_tol(1e-5));

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
    use burn_stack::modules::{LayersBuilder, Residuals, ResidualsConfig};
    use burn_stack::utils::Schedule;

    let device = test_device();
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
            init_bias_offset: 0.0,
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
    let (y, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None, None);

    // Reference: the input is stream 1; each layer appends its output until
    // `n_stream` streams exist, and only then are they gate-mixed.
    let layer = &layers.real_layers[0];
    let mut streams = x.clone().unsqueeze_dim::<4>(2);
    let mut h = x.clone();
    for i in 0..n_virtual {
        let (out, _c) = layer.forward(h, None, Mamba2SsdPath::default(), None);
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
        max_abs_diff(y.clone(), h) < dtype_tol(1e-5),
        "the stack must run the accumulate-then-mix schedule"
    );

    // Degenerate wiring (every stream a copy of the input): all streams stay
    // identical forever, so the whole stack is one lerped stream.
    let mut h = x.clone();
    for i in 0..n_virtual {
        let (out, _c) = layer.forward(h.clone(), None, Mamba2SsdPath::default(), None);
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
        max_abs_diff(y, h) > dtype_tol(1e-3),
        "the stack must not collapse to a single lerped stream"
    );
}

/// `init_bias_offset` is the offset between the initial gate biases of two
/// consecutive streams: stream `i` starts at `init_bias + i · init_bias_offset`.
/// A nonzero offset gives the streams different depth timescales before any
/// training. `0` keeps the uniform initial biases of the paper.
#[test]
fn init_bias_offset_ramps_the_gates() {
    let device = test_device();
    let (n, d) = (4, 8);
    let m = MultiGateResidualConfig::new(d, n)
        .with_init_bias(-2.7)
        .with_init_bias_offset(0.9)
        .init(&device);

    let expected = Tensor::<1>::from_floats([-2.7, -1.8, -0.9, 0.0], &device);
    assert!(max_abs_diff(m.b_beta.val(), expected) < dtype_tol(1e-5));

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
        .try_into_vec_as()
        .unwrap();
    for (i, w) in betas.windows(2).enumerate() {
        assert!(
            w[0] < w[1],
            "stream {i} must lerp less than stream {}",
            i + 1
        );
    }
    assert!(
        (betas[n - 1] - 0.5).abs() < dtype_tol(1e-4),
        "last stream is σ(0) = 0.5"
    );
}

// --- class tokens / latents ------------------------------------------------

/// The identity MGR class-marker support rests on: a row present in **every**
/// stream is reproduced exactly by the aggregator. All streams then score
/// alike, so `α` is uniform and the convex pool returns the row — which is why
/// splicing a class latent into the streams hands the layer above the very same
/// embedding a plain additive skip would have handed it.
#[test]
fn attn_pool_reproduces_a_row_present_in_every_stream() {
    let device = test_device();
    let (b, k, d) = (2, 4, 8);
    let m = random_mgr(d, k, &device);

    let row = Tensor::<2>::random([b, d], Distribution::Default, &device);
    let streams = row.clone().unsqueeze_dim::<3>(1).expand([b, k, d]);
    // `attn_pool` keeps the stream axis at size 1 (it broadcasts back).
    let pooled = m.attn_pool(streams).squeeze_dim::<2>(1);
    assert!(max_abs_diff(pooled, row) < dtype_tol(1e-5));
}

/// MGR + **stack-level** class latents: they are spliced below the first layer,
/// so they seed the streams like any other token. `forward` must equal `step`
/// unrolled, with the latent falling on the step whose cursor reaches it.
#[cfg(feature = "mamba2")]
#[test]
fn layers_multi_gate_stack_class_latents_step_matches_forward() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use burn_stack::modules::{LayersBuilder, ResidualsConfig};
    use burn_stack::utils::{ClassCursors, ClassLatent};

    let device = test_device();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(3, block)
        .with_class_latents(vec![ClassLatent::Start, ClassLatent::Custom(2)])
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: -1.0,
            init_bias_offset: -0.5,
            per_virtual_layer: false,
        })
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None, None);
    assert_eq!(y_fwd.dims(), [batch, seq + 2, d_model]);
    // Start, u0, u1, Custom, u2, u3.
    let user_pos = [1usize, 2, 4, 5];

    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    for (t, &pos) in user_pos.iter().enumerate() {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut class));
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, pos, 1).squeeze_dim::<2>(1);
        assert!(
            max_rel_diff(yt, expected) < dtype_tol(1e-4),
            "MGR stack-latent step disagrees with forward at t={t}"
        );
    }
    assert_eq!(class.stack, seq + 2);
}

/// MGR + class latents under a **chunked** `forward`: the same cursors carried
/// across two chunks must place every latent exactly where the single-call
/// `forward` does, streams included.
#[cfg(feature = "mamba2")]
#[test]
fn layers_multi_gate_class_latents_split_forward_matches_single() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use burn_stack::modules::{LayersBuilder, ResidualsConfig};
    use burn_stack::utils::{ClassCursors, ClassLatent};

    let device = test_device();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(3, block)
        .with_class_latents(vec![ClassLatent::Start])
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: -1.0,
            init_bias_offset: 0.0,
            per_virtual_layer: false,
        })
        .init(&device);

    let (batch, seq) = (2usize, 6usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let path = Mamba2SsdPath::Minimal(None);
    let (y_one, _c) = layers.forward(x.clone(), None, path.clone(), None, None);

    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    let mut outs = Vec::new();
    for (at, len) in [(0usize, 2usize), (2, 4)] {
        let chunk = x.clone().narrow(1, at, len);
        let (y, c) = layers.forward(chunk, caches, path.clone(), Some(&mut class), None);
        caches = Some(c);
        outs.push(y);
    }
    assert!(max_rel_diff(Tensor::cat(outs, 1), y_one) < dtype_tol(1e-4));
}

/// MGR + class latents through `prime`: priming the latents waiting for the
/// next token and then stepping it must run exactly what that `step` alone
/// would have — the property `prime` exists for, now on the MGR path too.
#[cfg(feature = "mamba2")]
#[test]
fn layers_multi_gate_prime_then_step_matches_step() {
    use crate::mamba2::prelude::Mamba2Config;
    use burn_stack::modules::{LayersBuilder, ResidualsConfig};
    use burn_stack::utils::{ClassCursors, ClassLatent};

    let device = test_device();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(3, block)
        .with_class_latents(vec![ClassLatent::Start])
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: -1.0,
            init_bias_offset: -0.5,
            per_virtual_layer: false,
        })
        .init(&device);

    let (batch, seq) = (2usize, 3usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let token = |t: usize| x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);

    // Plain stepping.
    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    let mut plain = Vec::new();
    for t in 0..seq {
        let (y, c) = layers.step(token(t), caches, Some(&mut class));
        caches = Some(c);
        plain.push(y);
    }

    // Priming before every step must leave the same outputs and cursors.
    let mut class2 = ClassCursors::new(seq);
    let mut caches2 = None;
    for (t, want) in plain.iter().enumerate() {
        let (primed, c) = layers.prime(batch, caches2, Some(&mut class2));
        caches2 = c;
        // The stack's `Start` is due before the very first token. Nothing waits
        // for the later ones.
        assert_eq!(primed.is_some(), t == 0, "primed latents at t={t}");
        let (y, c) = layers.step(token(t), caches2, Some(&mut class2));
        caches2 = Some(c);
        assert!(
            max_abs_diff(y, want.clone()) < dtype_tol(1e-4),
            "prime+step disagrees with step at t={t}"
        );
    }
    assert_eq!(class, class2);
}

/// `BidiLayers` + MGR + stack-level class latents: the latents are spliced
/// below the first pair, so they lengthen the sequence and seed the streams.
#[cfg(feature = "mamba2")]
#[test]
fn bidi_multi_gate_class_latents_forward() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use burn_stack::modules::ResidualsConfig;
    use burn_stack::modules::bidi::{BidiLayersBuilder, OutputMergeConfig};
    use burn_stack::utils::ClassLatent;

    let device = test_device();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let n_real = 4; // 2 pairs
    let layers = BidiLayersBuilder {
        n_real_layers: n_real,
        n_virtual_layers: None,
        block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(n_real),
        class_latents: vec![ClassLatent::Start, ClassLatent::End],
        residuals: ResidualsConfig::MultiGate {
            n_stream: 3,
            init_bias: -1.0,
            init_bias_offset: 0.0,
            per_virtual_layer: false,
        },
        untied: Vec::new(),
    }
    .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (y, _c) = layers.forward(x, None, Mamba2SsdPath::default(), None, None);
    assert_eq!(y.dims(), [batch, seq + 2, d_model]);
}

/// Both marker levels (a network's [`ClassToken`] and the stack's own
/// [`ClassLatent`]) on top of MGR, driven through the config surface a user
/// actually reaches. `prime` then `step` must reproduce the single `forward`,
/// marker for marker, and the final state with it.
///
/// [`ClassToken`]: burn_stack::utils::ClassToken
/// [`ClassLatent`]: burn_stack::utils::ClassLatent
#[cfg(feature = "mamba2")]
#[test]
fn latent_network_multi_gate_class_markers_prime_step_matches_forward() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use burn_stack::modules::network::LatentNetworkBuilder;
    use burn_stack::modules::{LayersBuilder, ResidualsConfig};
    use burn_stack::utils::{ClassCursors, ClassLatent, ClassToken};

    let device = test_device();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let net = LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder::new(2, block)
            .with_class_latents(vec![ClassLatent::Start])
            .with_residuals(ResidualsConfig::MultiGate {
                n_stream: 2,
                init_bias: -1.0,
                init_bias_offset: -0.5,
                per_virtual_layer: false,
            }),
        output_size: 2,
        // MGR pools convexly, so a latent head wants the norm (see the header).
        final_norm: true,
        class_tokens: vec![ClassToken::Start],
    }
    .init(&device);

    let (batch, seq) = (2usize, 3usize);
    let x = Tensor::<3>::random([batch, seq, 3], Distribution::Normal(0.0, 1.0), &device);

    // forward ⇒ S N u0 u1 u2 (each level opens the one above it).
    let (y_fwd, c_fwd) = net.forward(x.clone(), None, Mamba2SsdPath::default(), None, None);
    assert_eq!(y_fwd.dims(), [batch, seq + 2, 2]);
    let row = |p: usize| y_fwd.clone().narrow(1, p, 1).squeeze_dim::<2>(1);

    // One prime emits both markers. The network's class token is the last.
    let mut class = ClassCursors::new(seq);
    let (y_prime, mut caches) = net.prime(batch, None, Some(&mut class));
    assert!(
        max_rel_diff(y_prime.expect("two markers wait"), row(1)) < dtype_tol(1e-4),
        "prime did not return the network's class token"
    );
    assert_eq!(class.network, 1);
    assert_eq!(class.stack, 2);

    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = net.step(xt, caches, Some(&mut class));
        caches = Some(c);
        assert!(
            max_rel_diff(yt, row(t + 2)) < dtype_tol(1e-4),
            "MGR step {t} disagrees with forward after the prime"
        );
    }
    for (i, (f, s)) in c_fwd.caches.iter().zip(&caches.unwrap().caches).enumerate() {
        assert!(
            max_rel_diff(s.conv_bvk.clone(), f.conv_bvk.clone()) < dtype_tol(1e-4),
            "layer {i} conv state disagrees"
        );
        assert!(
            max_rel_diff(s.ssm_bhpr.clone(), f.ssm_bhpr.clone()) < dtype_tol(1e-4),
            "layer {i} state disagrees"
        );
    }
}
