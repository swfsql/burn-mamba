//! Tests for the optional feed-forward sub-block on [`Layer`].
//!
//! The load-bearing claim is that [`Layer`] returning its *total delta*, plus the
//! single outer add in [`Layers`](burn_stack::modules::Layers), reproduces the two
//! separate residuals of the reference `mamba_ssm` `Block`:
//!
//! ```text
//!   residual = x + mixer(norm(x))          # first residual
//!   out      = residual + mlp(norm2(residual))   # second residual
//! ```
//!
//! Nothing about that is visible in the shapes, so it needs an explicit
//! reference computation to pin it down.

use crate::prelude::*;
use burn::prelude::*;
use burn_stack::modules::{GatedMlpConfig, LayersBuilder, RmsNormConfig};
use burn_stack::utils::test_helpers::max_abs_diff;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

#[cfg(feature = "mamba3")]
fn block_config(d_model: usize) -> crate::mamba3::prelude::Mamba3Config {
    crate::mamba3::prelude::Mamba3Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(0.5)
}

/// A layer with an `mlp` must equal the reference block's two-residual form.
///
/// Recomputed here from the layer's own sub-modules, so a delta folded the wrong
/// way (e.g. feeding the MLP `x` instead of `x + h₁`, or dropping `h₁` from the
/// returned sum) shows up as a numeric mismatch rather than passing silently.
#[cfg(feature = "mamba3")]
#[test]
fn layer_with_mlp_matches_the_two_residual_reference() {
    use crate::mamba3::prelude::Mamba3SsdPath;

    let device = Device::default();
    let (batch, seq, d_model) = (2, 6, 16);
    let layers = LayersBuilder::new(1, block_config(d_model))
        .with_mlp(Some(GatedMlpConfig::new(d_model, 32)))
        .init(&device);
    let layer = &layers.real_layers[0];
    assert!(layer.mlp.is_some() && layer.norm2.is_some());

    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // What the stack actually produces: one outer add over the layer's delta.
    let (got, _caches) = layers.forward(x.clone(), None, Mamba3SsdPath::default(), None);

    // The reference block, spelled out.
    let (h1, _c) = layer.block.block_forward(
        layer.norm.forward(x.clone()),
        None,
        Mamba3SsdPath::default(),
    );
    let residual = x + h1;
    let expected = residual.clone()
        + layer
            .mlp
            .as_ref()
            .unwrap()
            .forward(layer.norm2.as_ref().unwrap().forward(residual));

    assert!(max_abs_diff(got, expected) < 1e-4);
}

/// `forward` and `step` must still agree once the feed-forward is in the loop —
/// the MLP is point-wise, so it may not introduce any cross-token coupling.
#[cfg(feature = "mamba3")]
#[test]
fn mlp_layer_forward_step_parity() {
    use crate::mamba3::prelude::Mamba3SsdPath;

    let device = Device::default();
    let (batch, seq, d_model) = (2, 5, 16);
    let layers = LayersBuilder::new(2, block_config(d_model))
        .with_mlp(Some(GatedMlpConfig::new(d_model, 32)))
        .init(&device);

    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (full, _caches) = layers.forward(x.clone(), None, Mamba3SsdPath::default(), None);

    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, None);
        caches = Some(c);
        let expected = full.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        assert!(max_abs_diff(yt, expected) < 1e-4, "mismatch at t={t}");
    }
}

/// Without an `mlp` the delta is the mixer output alone, so a stack built the old
/// way is bit-for-bit what it was before the field existed. Guards the default
/// path for Mamba-1/2 and for Mamba-3 checkpoints that carry no feed-forward.
#[cfg(feature = "mamba3")]
#[test]
fn mixer_only_layer_is_unchanged() {
    use crate::mamba3::prelude::Mamba3SsdPath;

    let device = Device::default();
    let (batch, seq, d_model) = (2, 4, 16);
    let layers = LayersBuilder::new(1, block_config(d_model)).init(&device);
    let layer = &layers.real_layers[0];
    assert!(layer.mlp.is_none() && layer.norm2.is_none());

    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (got, _caches) = layers.forward(x.clone(), None, Mamba3SsdPath::default(), None);

    let (h1, _c) = layer.block.block_forward(
        layer.norm.forward(x.clone()),
        None,
        Mamba3SsdPath::default(),
    );
    assert!(max_abs_diff(got, x + h1) < 1e-5);
}

/// `norm2` is allocated together with `mlp`, so a hand-built `Layer` that sets
/// only one of the two is a construction bug. It must fail loudly rather than
/// silently skipping the feed-forward.
#[cfg(feature = "mamba3")]
#[test]
#[should_panic(expected = "`norm2` is allocated alongside `mlp`")]
fn mlp_without_norm2_panics() {
    let device = Device::default();
    let d_model = 16;
    let mut layers = LayersBuilder::new(1, block_config(d_model))
        .with_mlp(Some(GatedMlpConfig::new(d_model, 32)))
        .init(&device);
    layers.real_layers[0].norm2 = None;

    let x = Tensor::<3>::zeros([1, 3, d_model], &device);
    let _ = layers.forward(
        x,
        None,
        crate::mamba3::prelude::Mamba3SsdPath::default(),
        None,
    );
}

/// The feed-forward's pre-norm is a distinct parameter set from the mixer's, so
/// the two must not alias — a checkpoint writes different values into each.
#[cfg(feature = "mamba3")]
#[test]
fn norm_and_norm2_are_independent() {
    let device = Device::default();
    let d_model = 16;
    let mut layers = LayersBuilder::new(1, block_config(d_model))
        .with_mlp(Some(GatedMlpConfig::new(d_model, 32)))
        .init(&device);

    let replacement = RmsNormConfig::new(d_model).init(&device);
    layers.real_layers[0].norm2 = Some(RmsNorm {
        gamma: burn::module::Param::from_tensor(replacement.gamma.val() * 3.0),
    });

    let layer = &layers.real_layers[0];
    let norm_gamma = layer.norm.gamma.val().into_data().try_to_vec::<f32>().unwrap();
    let norm2_gamma = layer
        .norm2
        .as_ref()
        .unwrap()
        .gamma
        .val()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert!(norm_gamma.iter().zip(&norm2_gamma).any(|(a, b)| a != b));
}
