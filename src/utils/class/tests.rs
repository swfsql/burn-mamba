//! Smoke / parity tests for the family-generic abstraction in
//! [`crate::generic`] — builder wiring, the unifying enums, and the
//! class-token / class-latent insertion + step-injection machinery.

use super::*;
use crate::prelude::*;
use crate::modules::*;
use crate::modules::{bidi::*, network::*};

#[cfg(feature = "mamba2")]
#[test]
fn latent_network_builder_mamba2() {
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let net = LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder::new(2, block),
        output_size: 2,
        class_tokens: Vec::new(),
    }
    .init(&device);

    let (y, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        Mamba2SsdPath::default(),
    );
    assert_eq!([2, 5, 2], y.dims());
    let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None, None, None, None);
    assert_eq!([2, 2], yt.dims());
}

#[cfg(feature = "mamba2")]
#[test]
fn unified_net_config_mamba2() {
    
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let net = MambaLatentNetConfig::Mamba2 {
        input_size: 3,
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        output_size: 2,
        class_tokens: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);

    // Explicit, family-tagged path.
    let (y, caches) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        MambaSsdPath::mamba2_default(),
    );
    assert_eq!([2, 5, 2], y.dims());

    // Thread the returned caches back in (round-trips the enum cache).
    let (y2, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        Some(caches),
        MambaSsdPath::mamba2_default(),
    );
    assert_eq!([2, 5, 2], y2.dims());

    let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None, None, None, None);
    assert_eq!([2, 2], yt.dims());
}

#[cfg(feature = "mamba3")]
#[test]
fn unified_net_config_mamba3() {
    
    let device = Device::default();
    let block = Mamba3Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(0.5);
    let net = MambaLatentNetConfig::Mamba3 {
        input_size: 3,
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        output_size: 2,
        class_tokens: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);

    let (y, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        MambaSsdPath::mamba3_default(),
    );
    assert_eq!([2, 5, 2], y.dims());
    let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None, None, None, None);
    assert_eq!([2, 2], yt.dims());
}

#[cfg(feature = "mamba1")]
#[test]
fn unified_net_config_mamba1() {
    
    let device = Device::default();
    let block = Mamba1Config::new(16).with_state_rank(8);
    let net = MambaLatentNetConfig::Mamba1 {
        input_size: 3,
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        output_size: 2,
        class_tokens: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);

    let (y, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        MambaSsdPath::Mamba1,
    );
    assert_eq!([2, 5, 2], y.dims());
    let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None, None, None, None);
    assert_eq!([2, 2], yt.dims());
}

// --- generic bidirectional stack ------------------------------------

#[cfg(feature = "mamba2")]
#[test]
fn bidi_layers_mamba2() {
    
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    // 2 real layers = 1 pair; CatLinear exercises the merge's params.
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::cat_linear(2),
        class_latents: Vec::new(),
    }
    .init(&device);
    let (y, _c) = layers.forward(
        Tensor::<3>::zeros([2, 5, 16], &device),
        None,
        Mamba2SsdPath::default(),
    );
    assert_eq!([2, 5, 16], y.dims());
}

#[cfg(feature = "mamba3")]
#[test]
fn bidi_layers_mamba3() {
    
    let device = Device::default();
    let block = Mamba3Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(0.5);
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(2),
        class_latents: Vec::new(),
    }
    .init(&device);
    let (y, _c) = layers.forward(
        Tensor::<3>::zeros([2, 5, 16], &device),
        None,
        Mamba3SsdPath::default(),
    );
    assert_eq!([2, 5, 16], y.dims());
}

// Mamba-1 gains bidirectional support for free via the generic stack
// (historically bidi was Mamba-2/3-only).
#[cfg(feature = "mamba1")]
#[test]
fn bidi_layers_mamba1() {
    
    let device = Device::default();
    let block = Mamba1Config::new(16).with_state_rank(8);
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::cat_linear(2),
        class_latents: Vec::new(),
    }
    .init(&device);
    let (y, _c) = layers.forward(Tensor::<3>::zeros([2, 5, 16], &device), None, ());
    assert_eq!([2, 5, 16], y.dims());
}

// --- unifying MambaBidiLayers enum ----------------------------------

#[cfg(feature = "mamba2")]
#[test]
fn unified_bidi_config_mamba2() {
    
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = MambaBidiLayersConfig::Mamba2 {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(2),
        class_latents: Vec::new(),
    }
    .init(&device);
    let (y, _c) = layers.forward(
        Tensor::<3>::zeros([2, 5, 16], &device),
        None,
        MambaSsdPath::mamba2_default(),
    );
    assert_eq!([2, 5, 16], y.dims());
}

/// Smoke test for the bidi residual refactor (the pair returns its merged output
/// and [`BidiLayers`] adds the skip): the residual-suppressed path
/// (`ignore_first/last_residual`) runs and preserves the sequence shape.
#[cfg(feature = "mamba2")]
#[test]
fn bidi_residual_skip_smoke() {
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: true,
        ignore_last_residual: true,
        outputs_merge: OutputMergeConfig::cat_linear(2),
        class_latents: Vec::new(),
    }
    .init(&device);
    let (y, _c) = layers.forward(
        Tensor::<3>::zeros([2, 5, 16], &device),
        None,
        Mamba2SsdPath::default(),
    );
    assert_eq!([2, 5, 16], y.dims());
}

// --- class tokens / latents -----------------------------------------

// Start/Middle/End class latents lengthen the sequence and land at the
// documented output positions.
#[cfg(feature = "mamba2")]
#[test]
fn class_latents_lengthen_and_index() {
    
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(1, block)
        .with_class_latents(vec![
            ClassLatent::Start,
            ClassLatent::Middle,
            ClassLatent::End,
        ])
        .init(&device);

    // L = 4 ⇒ Start→0, Middle→ floor(4/2)=2 (after the leading prefix), End→ end.
    assert_eq!(layers.class_latent_output_indices(4), vec![0, 3, 6]);

    let (y, _c) = layers.forward(
        Tensor::<3>::zeros([2, 4, 16], &device),
        None,
        Mamba2SsdPath::default(),
    );
    assert_eq!([2, 7, 16], y.dims()); // 4 original + 3 class latents
}

// `Custom(index)` is inserted last at its explicit index.
#[cfg(feature = "mamba2")]
#[test]
fn class_latents_custom_index() {
    let markers = vec![ClassLatent::Custom(1), ClassLatent::Custom(3)];
    // L = 5: a token before original index 1 (output pos 1) and one before
    // index 3 (output pos 4, shifted by the first insertion).
    assert_eq!(class_marker_output_indices(&markers, 5), vec![1, 4]);
}

// A network's class tokens lengthen its output sequence too.
#[cfg(feature = "mamba2")]
#[test]
fn class_tokens_on_latent_network() {
    
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let net = LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder::new(2, block),
        output_size: 2,
        class_tokens: vec![ClassToken::End],
    }
    .init(&device);
    let (y, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        Mamba2SsdPath::default(),
    );
    assert_eq!([2, 6, 2], y.dims()); // 5 + 1 class token
}

// `Middle`/`End` class latents are incompatible with single-token `step()`.
#[cfg(feature = "mamba2")]
#[test]
#[should_panic(expected = "not compatible with step")]
fn class_latents_step_panics() {
    
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(1, block)
        .with_class_latents(vec![ClassLatent::Middle])
        .init(&device);
    let _ = layers.step(Tensor::<2>::zeros([2, 16], &device), None, None, None);
}

// Stepping with the stack-level cursor injects the class latents at exactly the
// `forward` positions: the per-user-token step outputs match `forward`'s
// user-position slices, and the cursor lands past every emitted token. Two real
// layers exercise the cascade — a stack class latent must propagate through both
// layers' recurrences (as it does in `forward`), not be absorbed by the first.
#[cfg(feature = "mamba2")]
#[test]
fn class_latents_step_matches_forward() {
    
    use crate::utils::test_helpers::max_abs_diff;
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(2, block)
        .with_class_latents(vec![ClassLatent::Start, ClassLatent::Custom(2)])
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, 16],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    // forward → length seq + 2; class tokens at [0, 3], user tokens at [1,2,4,5].
    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default());
    assert_eq!(y_fwd.dims(), [batch, seq + 2, 16]);
    let user_pos = [1usize, 2, 4, 5];

    // step the user tokens with the stack-level class cursor; the class latents
    // are injected automatically as the cursor reaches their positions.
    let mut cursor = 0usize;
    let mut caches = None;
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut cursor), None);
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, user_pos[t], 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-4,
            "stepped user token {t} disagrees with forward"
        );
    }
    // Start, u0, u1, Custom, u2, u3 ⇒ cursor advanced by 6.
    assert_eq!(cursor, 6);
}

// Per-layer class latents in a 3-layer stack — A: `Custom(2)`, B: none, C:
// `Start` — with NO stack-level latents. A class latent grows the sequence the
// *next* layer sees, so `step` can only match `forward` via the cascade (each
// token a layer emits, its class latents included, must flow into the next
// layer in order). Checks results, final state, AND gradients all agree between
// a length-3 `forward` and 3 `step`s.
#[cfg(feature = "mamba2")]
#[test]
fn per_layer_class_latents_step_matches_forward() {
    
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let adev = device.clone().autodiff();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);

    // A,B,C; A=Custom(2), C=Start, B none. (Per-layer latents aren't builder-
    // configurable, so set them directly on the real layers.)
    let mut layers = LayersBuilder::new(3, block).init(&adev);
    layers.real_layers[0].class_latents = vec![ClassLatent::Custom(2)];
    layers.real_layers[0].class_latents_emb = init_class_emb(1, d_model, &adev);
    layers.real_layers[2].class_latents = vec![ClassLatent::Start];
    layers.real_layers[2].class_latents_emb = init_class_emb(1, d_model, &adev);

    let (batch, seq) = (2usize, 3usize);
    let dist = Distribution::Normal(0.0, 1.0);
    // Stable values reused (as fresh autodiff leaves) by both runs.
    let x_inner = Tensor::<3>::random([batch, seq, d_model], dist, &device);
    let out_head_inner = Tensor::<3>::random([batch, seq, d_model], dist, &device);

    // forward output is length seq + 2 (A adds one, C adds one); the user tokens
    // land at [1, 2, 4]: C_cls@0, u0@1, u1@2, A_cls@3, u2@4.
    let user_pos = [1usize, 2, 4];
    let path = Mamba2SsdPath::Minimal(None);

    // One run (forward or stepwise). Returns the user output + final per-layer
    // state (inner tensors) and the gradients of the input and a few params.
    type Run = (
        Tensor<3>,                   // user output
        Vec<(Tensor<3>, Tensor<4>)>, // per-layer (conv, ssm) final state
        Tensor<3>,                   // d input
        Tensor<2>,                   // d layer-0 in_proj weight
        Tensor<2>,                   // d layer-A (Custom) class emb
        Tensor<2>,                   // d layer-C (Start) class emb
    );
    let run = |stepwise: bool| -> Run {
        let x = Param::from_tensor(Tensor::from_inner(x_inner.clone()));
        let (out_user, caches) = if stepwise {
            let mut cursors = vec![0usize; 3];
            let mut caches = None;
            let mut outs = Vec::new();
            for t in 0..seq {
                let xt = x.val().narrow(1, t, 1).squeeze_dim::<2>(1);
                let (yt, c) = layers.step(xt, caches, None, Some(&mut cursors));
                caches = Some(c);
                outs.push(yt.unsqueeze_dim::<3>(1));
            }
            (Tensor::cat(outs, 1), caches.unwrap())
        } else {
            let (out_full, caches) = layers.forward(x.val(), None, path.clone());
            let parts: Vec<_> = user_pos
                .iter()
                .map(|&p| out_full.clone().narrow(1, p, 1))
                .collect();
            (Tensor::cat(parts, 1), caches)
        };

        // Loss couples the user output (via a fixed head) with the final state
        // (sum of squares), so gradients run through both the output and state.
        let out_head = Tensor::from_inner(out_head_inner.clone());
        let mut loss = (out_user.clone() * out_head).sum();
        for c in &caches.caches {
            loss = loss + (c.conv_bvk.clone() * c.conv_bvk.clone()).sum();
            loss = loss + (c.ssm_bhpr.clone() * c.ssm_bhpr.clone()).sum();
        }
        let grads = loss.backward();

        let state: Vec<(Tensor<3>, Tensor<4>)> = caches
            .caches
            .iter()
            .map(|c| (c.conv_bvk.clone().inner(), c.ssm_bhpr.clone().inner()))
            .collect();
        let d_emb = |i: usize| {
            layers.real_layers[i]
                .class_latents_emb
                .as_ref()
                .unwrap()
                .val()
                .grad(&grads)
                .expect("class emb grad")
        };
        (
            out_user.inner(),
            state,
            x.val().grad(&grads).expect("input grad"),
            layers.real_layers[0]
                .mamba_block
                .in_proj
                .weight
                .val()
                .grad(&grads)
                .expect("in_proj grad"),
            d_emb(0),
            d_emb(2),
        )
    };

    let f = run(false);
    let s = run(true);

    // Results + final state.
    assert!(max_abs_diff(f.0, s.0) < 1e-4, "user outputs disagree");
    for (i, ((cf, sf), (cs, ss))) in f.1.iter().zip(&s.1).enumerate() {
        assert!(
            max_abs_diff(cf.clone(), cs.clone()) < 1e-4,
            "layer {i} conv state disagrees"
        );
        assert!(
            max_abs_diff(sf.clone(), ss.clone()) < 1e-4,
            "layer {i} ssm state disagrees"
        );
    }
    // Gradients (input, a block weight, and both class-latent embeddings).
    assert!(max_abs_diff(f.2, s.2) < 1e-3, "input grads disagree");
    assert!(max_abs_diff(f.3, s.3) < 1e-3, "in_proj grads disagree");
    assert!(
        max_abs_diff(f.4, s.4) < 1e-3,
        "Custom class-emb grads disagree"
    );
    assert!(
        max_abs_diff(f.5, s.5) < 1e-3,
        "Start class-emb grads disagree"
    );
}
