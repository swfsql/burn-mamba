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
    let streams =
        Param::from_tensor(Tensor::<4>::random([b, s, n, d], Distribution::Default, &device));

    let (h, new_streams) = m.forward(layer_output, streams.val());
    let loss = h.sum() + new_streams.sum();
    let grads = loss.backward();

    assert!(streams.val().grad(&grads).is_some(), "grad streams");
    assert!(m.w_beta.val().grad(&grads).is_some(), "grad w_beta");
    assert!(m.w_alpha.val().grad(&grads).is_some(), "grad w_alpha");
    assert!(m.b_beta.val().grad(&grads).is_some(), "grad b_beta");
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
        })
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random([batch, seq, d_model], Distribution::Normal(0.0, 1.0), &device);

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default());
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None, None);
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
        })
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random([batch, seq, d_model], Distribution::Normal(0.0, 1.0), &device);

    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba3SsdPath::default());
    assert_eq!(y_fwd.dims(), [batch, seq, d_model]);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None, None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-3,
            "MGR (virtual) step disagrees with forward at t={t}"
        );
    }
}
