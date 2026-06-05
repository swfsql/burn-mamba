//! Tests for the bidirectional stack ([`BidiLayers`](super::BidiLayers)):
//! virtual-layer weight-sharing over the per-pair output merge, and basic
//! forward determinism. MultiGate-on-bidi wiring is covered by the
//! `multi_gate` tests.

use crate::modules::bidi::{BidiLayersBuilder, OutputMerge, OutputMergeConfig};
use crate::modules::ResidualsConfig;
use crate::utils::BidiSchedule;
use burn::prelude::*;
use burn::tensor::Distribution;

/// Regression test for the virtual-layer / output-merge indexing.
///
/// A bidi stack with more virtual than real layers must share each real pair's
/// `CatLinear` merge across the virtual pairs that map to it — indexed by the
/// **real** pair, not the virtual one. Indexing by the virtual pair used to panic
/// (`outputs_merge` has only `n_real_layers / 2` entries). This builds 10 virtual
/// layers (5 pairs) over 4 real layers (2 pairs) and checks that (a) `forward` no
/// longer panics and returns the right shape, and (b) gradients reach **both**
/// real pairs' merge weights (so both are exercised and shared, not dangling).
#[cfg(feature = "mamba2")]
#[test]
fn virtual_layers_share_per_real_pair_merge() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};

    let device = Device::default().autodiff();
    let d_model = 16;
    let n_real = 4; // 2 real pairs
    let n_virtual = 10; // 5 virtual pairs → must wrap onto the 2 real pairs
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);

    let layers = BidiLayersBuilder {
        n_real_layers: n_real,
        n_virtual_layers: Some((n_virtual, BidiSchedule::StridedCyclic)),
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        // CatLinear (weight-bearing) is what surfaces the indexing bug.
        outputs_merge: OutputMergeConfig::cat_linear(n_real),
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
    }
    .init(&device);

    assert_eq!(
        layers.outputs_merge.len(),
        n_real / 2,
        "one merge per real pair"
    );

    let (batch, seq) = (2usize, 6usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    // Pre-fix: this panicked with an out-of-bounds merge index.
    let (y, _c) = layers.forward(x, None, Mamba2SsdPath::default());
    assert_eq!(y.dims(), [batch, seq, d_model]);

    // Every real pair's merge must receive gradient (each is hit by ≥1 virtual
    // pair under StridedCyclic over 5 virtual / 2 real pairs).
    let grads = y.sum().backward();
    for (p, merge) in layers.outputs_merge.iter().enumerate() {
        let OutputMerge::CatLinear(proj) = merge else {
            panic!("expected CatLinear merge");
        };
        assert!(
            proj.weight.val().grad(&grads).is_some(),
            "grad did not reach real pair {p}'s output merge",
        );
    }
}

/// Two identical bidi forward passes must agree (no hidden nondeterminism in the
/// straight/reverse/merge plumbing), including under virtual scheduling.
#[cfg(feature = "mamba2")]
#[test]
fn virtual_forward_is_deterministic() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use crate::utils::test_helpers::max_abs_diff;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: Some((6, BidiSchedule::StridedStretched)),
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(2),
        class_latents: Vec::new(),
        residuals: ResidualsConfig::Standard,
    }
    .init(&device);

    let x = Tensor::<3>::random([2, 5, d_model], Distribution::Normal(0.0, 1.0), &device);
    let (y0, _) = layers.forward(x.clone(), None, Mamba2SsdPath::default());
    let (y1, _) = layers.forward(x, None, Mamba2SsdPath::default());
    assert!(max_abs_diff(y0, y1) < 1e-6, "bidi forward is nondeterministic");
}
