use super::*;
use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
use crate::utils::test_helpers::max_abs_diff;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

/// The full cascade (`VocabNetwork → Layers → Layer → Mamba2`): the per-layer
/// moments of `forward_with_state_moments` must equal what a token-by-token
/// `step` loop accumulates from each layer's `ssm_bhpr` cache — the exact
/// consumption pattern of a state-PR diagnostic. Value-only (no gradients),
/// so everything lives on the plain device.
#[test]
fn vocab_forward_state_moments_match_step() {
    let device: Device = Default::default();
    let block_cfg = Mamba2Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8);
    let n_layers = 2;
    let net = VocabNetworkBuilder {
        vocab_size: 11,
        pad_vocab_size_multiple: 8,
        layers: LayersBuilder::new(n_layers, block_cfg.clone()),
        missing_lm_head: true,
    }
    .init(&device);

    let (batch, seq_len) = (2, 7); // chunk_len 4 ⇒ padding in the last chunk
    let ssd_path = Mamba2SsdPath::SerialRecalculated(Some(4));
    let tokens_bs = Tensor::<2, Int>::random(
        [batch, seq_len],
        Distribution::Uniform(0.0, 11.0),
        &device,
    );

    let (logits, _caches, moments) =
        net.forward_with_state_moments(tokens_bs.clone(), None, ssd_path);
    assert_eq!(logits.dims()[..2], [batch, seq_len]);
    assert_eq!(moments.len(), n_layers, "one moments entry per (virtual) layer");

    // Reference: step token-by-token, accumulating per-layer cache states.
    let (nheads, state_rank) = (block_cfg.nheads(), block_cfg.state_rank);
    let zero4 = || Tensor::<4>::zeros([batch, nheads, state_rank, state_rank], &device);
    let zero3 = || Tensor::<3>::zeros([batch, nheads, state_rank], &device);
    let mut m2s: Vec<Tensor<4>> = (0..n_layers).map(|_| zero4()).collect();
    let mut m1s: Vec<Tensor<3>> = (0..n_layers).map(|_| zero3()).collect();
    let mut caches = None;
    for t in 0..seq_len {
        let token_b = tokens_bs.clone().narrow(1, t, 1).squeeze_dim::<1>(1);
        let (_logits_t, new_caches) = net.step(token_b, caches, None, None);
        for (l, cache) in new_caches.caches.iter().enumerate() {
            let h_bhpr = cache.ssm_bhpr.clone();
            m2s[l] = m2s[l].clone()
                + h_bhpr.clone().permute([0, 1, 3, 2]).matmul(h_bhpr.clone());
            m1s[l] = m1s[l].clone() + h_bhpr.sum_dim(2).reshape([batch, nheads, state_rank]);
        }
        caches = Some(new_caches);
    }

    for (l, m) in moments.iter().enumerate() {
        assert_eq!(m.count, seq_len * block_cfg.per_head_dim);
        let scale = max_abs_diff(m2s[l].clone(), m2s[l].zeros_like()).max(1.0);
        let d2 = max_abs_diff(m.m2_bhrr.clone(), m2s[l].clone());
        let d1 = max_abs_diff(m.m1_bhr.clone(), m1s[l].clone());
        assert!(d2 < 1e-4 * scale, "layer {l}: m2 vs step {d2:.6} (scale {scale:.3})");
        assert!(d1 < 1e-4 * scale, "layer {l}: m1 vs step {d1:.6} (scale {scale:.3})");
    }
}

/// The `_grad` cascade: a moments-only loss through
/// `forward_with_state_moments_grad` must back-propagate into upstream
/// parameters (the embedding — every layer's states depend on it), while
/// leaving parameters that feed only `y` past the last state read
/// (`norm_f.gamma`) untouched. A wiring mistake that detaches the moments
/// (or leaks `y` into them) fails one of the two.
#[test]
fn vocab_forward_state_moments_grad_reaches_params() {
    let device: Device = Default::default();
    let block_cfg = Mamba2Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8);
    let net = VocabNetworkBuilder {
        vocab_size: 11,
        pad_vocab_size_multiple: 8,
        layers: LayersBuilder::new(2, block_cfg),
        missing_lm_head: true,
    }
    .init(&device.clone().autodiff());

    let tokens_bs = Tensor::<2, Int>::random(
        [2, 5],
        Distribution::Uniform(0.0, 11.0),
        &device.clone().autodiff(),
    );
    let (_logits, _caches, moments) = net.forward_with_state_moments_grad(
        tokens_bs,
        None,
        Mamba2SsdPath::SerialRecalculated(Some(4)),
    );
    let loss = moments
        .into_iter()
        .map(|m| m.m2_bhrr.sum() + m.m1_bhr.sum())
        .reduce(|a, b| a + b)
        .expect("two layers of moments");
    let grads = loss.backward();

    assert!(
        net.embedding.weight.val().grad(&grads).is_some(),
        "attached moments must reach the embedding"
    );
    assert!(
        net.norm_f.gamma.val().grad(&grads).is_none(),
        "a moments-only loss must not touch y-only parameters"
    );
}

/// Mamba-3 twin of [`vocab_forward_state_moments_match_step`]: the cascade is
/// the same generic code, but the per-layer moments are the **physical-frame**
/// states — the step reference de-rotates each layer's cache state
/// (`rotation.derotate_state`) before pooling.
#[cfg(feature = "mamba3")]
#[test]
fn vocab_forward_state_moments_match_step_mamba3() {
    use crate::mamba3::prelude::{Mamba3Caches, Mamba3Config, Mamba3SsdPath};

    let device: Device = Default::default();
    let block_cfg = Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8);
    let n_layers = 2;
    let net = VocabNetworkBuilder {
        vocab_size: 11,
        pad_vocab_size_multiple: 8,
        layers: LayersBuilder::new(n_layers, block_cfg.clone()),
        missing_lm_head: true,
    }
    .init(&device);

    let (batch, seq_len) = (2, 7); // chunk_len 4 ⇒ padding in the last chunk
    let ssd_path = Mamba3SsdPath::SerialRecalculated(Some(4));
    let tokens_bs =
        Tensor::<2, Int>::random([batch, seq_len], Distribution::Uniform(0.0, 11.0), &device);

    let (logits, _caches, moments) =
        net.forward_with_state_moments(tokens_bs.clone(), None, ssd_path);
    assert_eq!(logits.dims()[..2], [batch, seq_len]);
    assert_eq!(moments.len(), n_layers, "one moments entry per (virtual) layer");

    // Reference: step token-by-token, pooling each layer's physical state.
    let (nheads, state_rank) = (block_cfg.nheads(), block_cfg.state_rank);
    let rope_dim = block_cfg.rope_dim();
    let zero4 = || Tensor::<4>::zeros([batch, nheads, state_rank, state_rank], &device);
    let zero3 = || Tensor::<3>::zeros([batch, nheads, state_rank], &device);
    let mut m2s: Vec<Tensor<4>> = (0..n_layers).map(|_| zero4()).collect();
    let mut m1s: Vec<Tensor<3>> = (0..n_layers).map(|_| zero3()).collect();
    let mut caches = None;
    for t in 0..seq_len {
        let token_b = tokens_bs.clone().narrow(1, t, 1).squeeze_dim::<1>(1);
        let (_logits_t, new_caches) = net.step(token_b, caches, None, None);
        let Mamba3Caches::SingleSsd(per_layer) = &new_caches else {
            panic!("fresh caches default to single-ssd");
        };
        for (l, cache) in per_layer.caches.iter().enumerate() {
            let h_bhpr = cache.rotation.derotate_state(
                cache.ssm_bhpr.clone(),
                rope_dim,
                block_cfg.mimo_rank == 1,
            );
            m2s[l] =
                m2s[l].clone() + h_bhpr.clone().permute([0, 1, 3, 2]).matmul(h_bhpr.clone());
            m1s[l] = m1s[l].clone() + h_bhpr.sum_dim(2).reshape([batch, nheads, state_rank]);
        }
        caches = Some(new_caches);
    }

    for (l, m) in moments.iter().enumerate() {
        assert_eq!(m.count, seq_len * block_cfg.per_head_dim);
        let scale = max_abs_diff(m2s[l].clone(), m2s[l].zeros_like()).max(1.0);
        let d2 = max_abs_diff(m.m2_bhrr.clone(), m2s[l].clone());
        let d1 = max_abs_diff(m.m1_bhr.clone(), m1s[l].clone());
        assert!(d2 < 1e-4 * scale, "layer {l}: m2 vs step {d2:.6} (scale {scale:.3})");
        assert!(d1 < 1e-4 * scale, "layer {l}: m1 vs step {d1:.6} (scale {scale:.3})");
    }
}

/// Mamba-3 twin of [`vocab_forward_state_moments_grad_reaches_params`]: the
/// attached moments (through the custom recompute node) must reach upstream
/// parameters while leaving y-only parameters untouched.
#[cfg(feature = "mamba3")]
#[test]
fn vocab_forward_state_moments_grad_reaches_params_mamba3() {
    use crate::mamba3::prelude::{Mamba3Config, Mamba3SsdPath};

    let device: Device = Default::default();
    let block_cfg = Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8);
    let net = VocabNetworkBuilder {
        vocab_size: 11,
        pad_vocab_size_multiple: 8,
        layers: LayersBuilder::new(2, block_cfg),
        missing_lm_head: true,
    }
    .init(&device.clone().autodiff());

    let tokens_bs = Tensor::<2, Int>::random(
        [2, 5],
        Distribution::Uniform(0.0, 11.0),
        &device.clone().autodiff(),
    );
    let (_logits, _caches, moments) = net.forward_with_state_moments_grad(
        tokens_bs,
        None,
        Mamba3SsdPath::SerialRecalculated(Some(4)),
    );
    let loss = moments
        .into_iter()
        .map(|m| m.m2_bhrr.sum() + m.m1_bhr.sum())
        .reduce(|a, b| a + b)
        .expect("two layers of moments");
    let grads = loss.backward();

    assert!(
        net.embedding.weight.val().grad(&grads).is_some(),
        "attached moments must reach the embedding"
    );
    assert!(
        net.norm_f.gamma.val().grad(&grads).is_none(),
        "a moments-only loss must not touch y-only parameters"
    );
}
