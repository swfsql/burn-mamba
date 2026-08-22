//! Smoke / parity tests for the family-generic abstraction in
//! [`crate::modules`] — builder wiring, the unifying enums, and the
//! class-token / class-latent insertion + step-injection machinery.

use super::*;
use crate::modules::*;
use crate::modules::{bidi::*, network::*};
use crate::prelude::*;

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
        final_norm: false,
        class_tokens: Vec::new(),
    }
    .init(&device);

    let (y, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        Mamba2SsdPath::default(),
        None,
    );
    assert_eq!([2, 5, 2], y.dims());
    let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None, None);
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
        grad_horizon: None,
        mamba_block: block,
        output_size: 2,
        final_norm: false,
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: crate::modules::ResidualsConfig::Standard,
        mlp: None,
    }
    .init(&device);

    // Explicit, family-tagged path.
    let (y, caches) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        MambaSsdPath::mamba2_default(),
        None,
    );
    assert_eq!([2, 5, 2], y.dims());

    // Thread the returned caches back in (round-trips the enum cache).
    let (y2, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        Some(caches),
        MambaSsdPath::mamba2_default(),
        None,
    );
    assert_eq!([2, 5, 2], y2.dims());

    let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None, None);
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
        .with_rope_fraction(Some(0.5));
    let net = MambaLatentNetConfig::Mamba3 {
        input_size: 3,
        n_real_layers: 2,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: block,
        output_size: 2,
        final_norm: false,
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: crate::modules::ResidualsConfig::Standard,
        mlp: None,
    }
    .init(&device);

    let (y, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        MambaSsdPath::mamba3_default(),
        None,
    );
    assert_eq!([2, 5, 2], y.dims());
    let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None, None);
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
        grad_horizon: None,
        mamba_block: block,
        output_size: 2,
        final_norm: false,
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: crate::modules::ResidualsConfig::Standard,
        mlp: None,
    }
    .init(&device);

    let (y, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        MambaSsdPath::Mamba1,
        None,
    );
    assert_eq!([2, 5, 2], y.dims());
    let (yt, _c) = net.step(Tensor::<2>::zeros([2, 3], &device), None, None);
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
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);
    let (y, _c) = layers.forward(
        Tensor::<3>::zeros([2, 5, 16], &device),
        None,
        Mamba2SsdPath::default(),
        None,
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
        .with_rope_fraction(Some(0.5));
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::mean(2),
        class_latents: Vec::new(),
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);
    let (y, _c) = layers.forward(
        Tensor::<3>::zeros([2, 5, 16], &device),
        None,
        Mamba3SsdPath::default(),
        None,
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
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);
    let (y, _c) = layers.forward(Tensor::<3>::zeros([2, 5, 16], &device), None, (), None);
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
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);
    let (y, _c) = layers.forward(
        Tensor::<3>::zeros([2, 5, 16], &device),
        None,
        MambaSsdPath::mamba2_default(),
        None,
    );
    assert_eq!([2, 5, 16], y.dims());
}

/// `BidiLayers::forward` determinism (see [`assert_bidi_deterministic`]).
#[cfg(feature = "mamba2")]
#[test]
fn bidi_forward_is_deterministic_mamba2() {
    use burn::tensor::Distribution;

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
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);

    let x = Tensor::<3>::random([2, 5, 16], Distribution::Normal(0.0, 1.0), &device);
    let (y1, _) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    let (y2, _) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!([2, 5, 16], y1.dims());
    assert_bidi_deterministic(y1, y2);
}

/// Mamba-1 counterpart of [`bidi_forward_is_deterministic_mamba2`].
#[cfg(feature = "mamba1")]
#[test]
fn bidi_forward_is_deterministic_mamba1() {
    use burn::tensor::Distribution;

    let device = Device::default();
    let block = Mamba1Config::new(16).with_state_rank(8);
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: true,
        ignore_last_residual: true,
        outputs_merge: OutputMergeConfig::cat_linear(2),
        class_latents: Vec::new(),
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);

    let x = Tensor::<3>::random([2, 5, 16], Distribution::Normal(0.0, 1.0), &device);
    let (y1, _) = layers.forward(x.clone(), None, (), None);
    let (y2, _) = layers.forward(x.clone(), None, (), None);
    assert_eq!([2, 5, 16], y1.dims());
    assert_bidi_deterministic(y1, y2);
}

/// Mamba-3 counterpart of [`bidi_forward_is_deterministic_mamba2`].
#[cfg(feature = "mamba3")]
#[test]
fn bidi_forward_is_deterministic_mamba3() {
    use burn::tensor::Distribution;

    let device = Device::default();
    let block = Mamba3Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(Some(0.5));
    let layers = BidiLayersBuilder {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        ignore_first_residual: true,
        ignore_last_residual: true,
        outputs_merge: OutputMergeConfig::cat_linear(2),
        class_latents: Vec::new(),
        residuals: crate::modules::ResidualsConfig::Standard,
    }
    .init(&device);

    let x = Tensor::<3>::random([2, 5, 16], Distribution::Normal(0.0, 1.0), &device);
    let (y1, _) = layers.forward(x.clone(), None, Mamba3SsdPath::default(), None);
    let (y2, _) = layers.forward(x.clone(), None, Mamba3SsdPath::default(), None);
    assert_eq!([2, 5, 16], y1.dims());
    assert_bidi_deterministic(y1, y2);
}

/// Two identical bidi forward passes must match: `BidiLayers::forward` runs its
/// real layers by reference rather than cloning them per call. Cloning a
/// lazily-initialised Burn `Param` re-runs its random initializer, so a
/// per-forward clone used to resample fresh weights every call — these tests
/// guard against that regression across all three families. The configs also
/// exercise the residual-suppressed (`ignore_first/last_residual`) path.
#[cfg(any(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
fn assert_bidi_deterministic(y1: Tensor<3>, y2: Tensor<3>) {
    assert!(
        crate::utils::test_helpers::max_abs_diff(y1, y2) < 1e-6,
        "two identical bidi forward passes diverged (weights resampled per call?)"
    );
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
        None,
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
        final_norm: false,
        class_tokens: vec![ClassToken::End],
    }
    .init(&device);
    let (y, _c) = net.forward(
        Tensor::<3>::zeros([2, 5, 3], &device),
        None,
        Mamba2SsdPath::default(),
        None,
    );
    assert_eq!([2, 6, 2], y.dims()); // 5 + 1 class token
}

// `Middle`/`End` class latents cannot be placed without a full-length hint.
#[cfg(feature = "mamba2")]
#[test]
#[should_panic(expected = "need a full-length hint")]
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
    let _ = layers.step(Tensor::<2>::zeros([2, 16], &device), None, None);
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
    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq + 2, 16]);
    let user_pos = [1usize, 2, 4, 5];

    // step the user tokens with the stack-level class cursor; the class latents
    // are injected automatically as the cursor reaches their positions.
    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    for (t, &pos) in user_pos.iter().enumerate() {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut class));
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, pos, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-4,
            "stepped user token {t} disagrees with forward"
        );
    }
    // Start, u0, u1, Custom, u2, u3 ⇒ the stack cursor advanced by 6.
    assert_eq!(class.stack, 6);
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
            let mut cursors = ClassCursors::new(seq);
            let mut caches = None;
            let mut outs = Vec::new();
            for t in 0..seq {
                let xt = x.val().narrow(1, t, 1).squeeze_dim::<2>(1);
                let (yt, c) = layers.step(xt, caches, Some(&mut cursors));
                caches = Some(c);
                outs.push(yt.unsqueeze_dim::<3>(1));
            }
            (Tensor::cat(outs, 1), caches.unwrap())
        } else {
            let (out_full, caches) = layers.forward(x.val(), None, path.clone(), None);
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

// --- prime (class-only steps, no user token) -------------------------------

// The prime planner, checked directly against the chunk planner it splits off
// from: a prime emits the markers waiting for the next user token — the run
// sitting at the cursor — and leaves everything else exactly where `step` finds
// it. `End` is never primed, and a cursor at the announced end primes nothing.
#[test]
fn class_prime_plan_emits_only_what_waits_for_the_next_token() {
    let markers = vec![ClassLatent::Start, ClassLatent::Custom(2), ClassLatent::End];
    // L = 3 ⇒ S u0 u1 C u2 E, i.e. positions [0, 3, 5].
    assert_eq!(class_marker_output_indices(&markers, 3), vec![0, 3, 5]);

    // A prime at the start emits `Start` alone — the others wait for a token.
    let mut cursor = ClassCursor::whole(3);
    assert_eq!(
        class_prime_plan(&markers, 0, &mut cursor, "test"),
        vec![(0, 0)]
    );
    assert_eq!(cursor.offset, 1);
    // Priming again finds nothing…
    assert_eq!(class_prime_plan(&markers, 0, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 1);
    // …and the two steps that follow carry their user token alone (`Start` is
    // behind the cursor, the `Custom` still waits for u2).
    assert_eq!(class_chunk_plan(&markers, 1, &mut cursor, "test"), vec![]);
    assert_eq!(class_chunk_plan(&markers, 1, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 3);
    // Now the `Custom` is what u2 would have been preceded by, so it primes.
    assert_eq!(
        class_prime_plan(&markers, 0, &mut cursor, "test"),
        vec![(0, 1)]
    );
    assert_eq!(cursor.offset, 4);
    // The last user token closes the sequence and `End` trails it there — never
    // on a prime, and the prime that follows has no next token to serve.
    assert_eq!(
        class_chunk_plan(&markers, 1, &mut cursor, "test"),
        vec![(1, 2)]
    );
    assert_eq!(cursor.offset, 6);
    assert_eq!(class_prime_plan(&markers, 0, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 6);

    // A prime reaching an upper level carries the tokens the level below just
    // emitted, and flushes what the *next* one is due to be preceded by — the
    // one difference from `class_chunk_plan` on the same tokens.
    let markers = vec![ClassLatent::Custom(1)];
    let mut cursor = ClassCursor::whole(4);
    assert_eq!(class_chunk_plan(&markers, 1, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 1);
    let mut cursor = ClassCursor::whole(4);
    assert_eq!(
        class_prime_plan(&markers, 1, &mut cursor, "test"),
        vec![(1, 0)]
    );
    assert_eq!(cursor.offset, 2);

    // An open-ended stream — the seedless case with no announced length: a
    // further token is always still to come, so `Start`/`Custom` prime exactly.
    let markers = vec![ClassLatent::Start, ClassLatent::Custom(1)];
    let mut cursor = ClassCursor::default();
    assert_eq!(
        class_prime_plan(&markers, 0, &mut cursor, "test"),
        vec![(0, 0)]
    );
    assert_eq!(cursor.offset, 1);
    assert_eq!(class_chunk_plan(&markers, 1, &mut cursor, "test"), vec![]);
    assert_eq!(cursor.offset, 2);
    assert_eq!(
        class_prime_plan(&markers, 0, &mut cursor, "test"),
        vec![(0, 1)]
    );
    assert_eq!(cursor.offset, 3);
}

// `prime` is `step`'s opening half: priming before every step must run exactly
// the sequence the steps alone run, while handing back the class outputs those
// steps drop. Pinned against a hand-built reference (as in
// [`step_output_and_state_follow_the_last_emitted_token`]):
//
// ```text
//   markers   Start, Custom(2), End                  L = 3 user tokens
//   sequence  S u0 u1 C u2 E                         ← built explicitly below
//   prime 0   S      → returns S            step 0   u0 → returns u0
//   prime 1   (none) → returns None         step 1   u1 → returns u1
//   prime 2   C      → returns C            step 2   u2 E → returns E
//   prime 3   (none) → returns None (the sequence is closed; `End` was the
//                     closing step's business, never a prime's)
// ```
#[cfg(feature = "mamba2")]
#[test]
fn prime_emits_the_class_markers_a_step_would_drop() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let (batch, seq) = (2usize, 3usize);
    let mut layers = LayersBuilder::new(2, block)
        .with_class_latents(vec![
            ClassLatent::Start,
            ClassLatent::Custom(2),
            ClassLatent::End,
        ])
        .init(&device);
    let path = Mamba2SsdPath::default();
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // The reference sequence, spliced by hand: S u0 u1 C u2 E.
    let emb = layers.class_latents_emb.as_ref().unwrap().val();
    let cls = |i: usize| {
        emb.clone()
            .narrow(0, i, 1)
            .unsqueeze_dim::<3>(0)
            .expand([batch, 1, d_model])
    };
    let tok = |t: usize| x.clone().narrow(1, t, 1);
    let reference = Tensor::cat(vec![cls(0), tok(0), tok(1), cls(1), tok(2), cls(2)], 1);
    assert_eq!(layers.class_latent_output_indices(seq), vec![0, 3, 5]);

    // Prime before every user token, and once more at the end.
    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    let mut got = Vec::new();
    for t in 0..seq {
        let (yp, c) = layers.prime(batch, caches, Some(&mut class));
        caches = c;
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut class));
        caches = Some(c);
        got.push((yp, yt));
    }
    let (y_last, caches) = layers.prime(batch, caches, Some(&mut class));
    assert!(
        y_last.is_none(),
        "the closed sequence has nothing left to prime"
    );
    // S u0 u1 C u2 E — every marker landed exactly once.
    assert_eq!(class.stack, seq + 3);

    // Same weights over the hand-built reference, markers cleared.
    layers.class_latents = Vec::new();
    let (y_ref, c_ref) = layers.forward(reference, None, path, None);

    let primed = [Some(0usize), None, Some(3)]; // S, (none), C
    let stepped = [1usize, 2, 5]; // u0, u1, E (closing the sequence)
    for (t, (yp, yt)) in got.into_iter().enumerate() {
        let row = |p: usize| y_ref.clone().narrow(1, p, 1).squeeze_dim::<2>(1);
        match (yp, primed[t]) {
            (Some(yp), Some(p)) => assert!(
                max_abs_diff(yp, row(p)) < 1e-4,
                "prime {t} did not return its class marker"
            ),
            (None, None) => {}
            _ => panic!("prime {t} emitted the wrong number of markers"),
        }
        assert!(
            max_abs_diff(yt, row(stepped[t])) < 1e-4,
            "step {t} disagrees with the reference once primed"
        );
    }

    // The primes moved the recurrence, never disturbed it: the final state is
    // the reference's, exactly as it is without them.
    for (i, (f, s)) in c_ref.caches.iter().zip(&caches.unwrap().caches).enumerate() {
        assert!(
            max_abs_diff(f.conv_bvk.clone(), s.conv_bvk.clone()) < 1e-4,
            "layer {i} conv state disagrees"
        );
        assert!(
            max_abs_diff(f.ssm_bhpr.clone(), s.ssm_bhpr.clone()) < 1e-4,
            "layer {i} ssm state disagrees"
        );
    }
}

// Seedless start: with a class latent on the *upper* layer only, a prime has no
// bottom stream at all — layer 0 runs nothing and layer 1 still emits its own
// waiting latent, exactly as the cascade does in `forward`. The caches come back
// complete (layer 0 holding the zero state it never left), so the steps that
// follow can be threaded straight on.
#[cfg(feature = "mamba2")]
#[test]
fn prime_runs_a_per_layer_latent_with_an_empty_stream() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let mut layers = LayersBuilder::new(2, block).init(&device);
    // Only the second layer carries a latent (not builder-configurable).
    layers.real_layers[1].class_latents = vec![ClassLatent::Start];
    layers.real_layers[1].class_latents_emb = init_class_emb(1, d_model, &device);

    let (batch, seq) = (2usize, 3usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // forward ⇒ the latent opens the sequence layer 1 sees, so the output is
    // [latent, u0, u1, u2].
    let (y_fwd, c_fwd) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq + 1, d_model]);

    let mut class = ClassCursors::new(seq);
    let (y_prime, mut caches) = layers.prime(batch, None, Some(&mut class));
    let row = |p: usize| y_fwd.clone().narrow(1, p, 1).squeeze_dim::<2>(1);
    assert!(
        max_abs_diff(
            y_prime.expect("the upper layer's latent was waiting"),
            row(0)
        ) < 1e-4,
        "prime did not return the per-layer latent"
    );
    // The bottom layer ran nothing, the top one one token.
    assert_eq!(class.stack, 0);
    assert_eq!(class.per_layer, vec![0, 1]);

    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut class));
        caches = Some(c);
        assert!(
            max_abs_diff(yt, row(t + 1)) < 1e-4,
            "step {t} disagrees with forward after the prime"
        );
    }
    for (i, (f, s)) in c_fwd.caches.iter().zip(&caches.unwrap().caches).enumerate() {
        assert!(
            max_abs_diff(f.conv_bvk.clone(), s.conv_bvk.clone()) < 1e-4,
            "layer {i} conv state disagrees"
        );
        assert!(
            max_abs_diff(f.ssm_bhpr.clone(), s.ssm_bhpr.clone()) < 1e-4,
            "layer {i} ssm state disagrees"
        );
    }
}

// The same, for Mamba-3: `prime` is family-generic, and the caches it completes
// for the layers that ran nothing must stay pathway-compatible with the ones a
// cacheless `step` builds (both single-ssd).
#[cfg(feature = "mamba3")]
#[test]
fn prime_runs_a_per_layer_latent_mamba3() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba3Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(Some(0.5));
    let mut layers = LayersBuilder::new(2, block).init(&device);
    layers.real_layers[1].class_latents = vec![ClassLatent::Start];
    layers.real_layers[1].class_latents_emb = init_class_emb(1, d_model, &device);

    let (batch, seq) = (2usize, 3usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba3SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq + 1, d_model]);
    let row = |p: usize| y_fwd.clone().narrow(1, p, 1).squeeze_dim::<2>(1);

    let mut class = ClassCursors::new(seq);
    let (y_prime, mut caches) = layers.prime(batch, None, Some(&mut class));
    assert!(
        max_abs_diff(
            y_prime.expect("the upper layer's latent was waiting"),
            row(0)
        ) < 1e-4,
        "prime did not return the per-layer latent"
    );
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut class));
        caches = Some(c);
        assert!(
            max_abs_diff(yt, row(t + 1)) < 1e-4,
            "step {t} disagrees with forward after the prime"
        );
    }
}

// A network primes all three class levels at once: its own class token runs a
// full pass (the stack splicing its latents around it, per-layer ones included),
// and the layers flush whatever is still waiting above it. What comes back is
// the last marker's output — the seedless generation seed.
#[cfg(feature = "mamba2")]
#[test]
fn prime_on_a_network_covers_every_class_level() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let mut net = LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder::new(2, block).with_class_latents(vec![ClassLatent::Start]),
        output_size: 2,
        final_norm: false,
        class_tokens: vec![ClassToken::Start],
    }
    .init(&device);
    net.layers.real_layers[1].class_latents = vec![ClassLatent::Start];
    net.layers.real_layers[1].class_latents_emb = init_class_emb(1, d_model, &device);

    let (batch, seq) = (2usize, 3usize);
    let x = Tensor::<3>::random([batch, seq, 3], Distribution::Normal(0.0, 1.0), &device);

    // forward ⇒ L S N u0 u1 u2: the layer-1 latent below the stack latent, both
    // below the network's class token (each level opens the one above it).
    let (y_fwd, c_fwd) = net.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq + 3, 2]);
    let row = |p: usize| y_fwd.clone().narrow(1, p, 1).squeeze_dim::<2>(1);

    // One prime emits all three markers; the network's class token is the last
    // of them, so its output is what comes back.
    let mut class = ClassCursors::new(seq);
    let (y_prime, mut caches) = net.prime(batch, None, Some(&mut class));
    assert!(
        max_abs_diff(y_prime.expect("three markers were waiting"), row(2)) < 1e-4,
        "prime did not return the network's class token"
    );
    assert_eq!(class.network, 1);
    assert_eq!(class.stack, 2);
    assert_eq!(class.per_layer, vec![2, 3]);

    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = net.step(xt, caches, Some(&mut class));
        caches = Some(c);
        assert!(
            max_abs_diff(yt, row(t + 3)) < 1e-4,
            "step {t} disagrees with forward after the prime"
        );
    }
    for (i, (f, s)) in c_fwd.caches.iter().zip(&caches.unwrap().caches).enumerate() {
        assert!(
            max_abs_diff(f.conv_bvk.clone(), s.conv_bvk.clone()) < 1e-4,
            "layer {i} conv state disagrees"
        );
        assert!(
            max_abs_diff(f.ssm_bhpr.clone(), s.ssm_bhpr.clone()) < 1e-4,
            "layer {i} ssm state disagrees"
        );
    }
}

// A bare `Layer` primes too, handing back the latent's delta *and* the latent
// itself so the caller can close the residual it owns. Priming then stepping
// must equal the single cursored step that would have emitted both.
#[cfg(feature = "mamba2")]
#[test]
fn layer_prime_returns_the_latent_and_its_delta() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let mut layers = LayersBuilder::new(1, block).init(&device);
    layers.real_layers[0].class_latents = vec![ClassLatent::Start];
    layers.real_layers[0].class_latents_emb = init_class_emb(1, d_model, &device);
    let layer = &layers.real_layers[0];

    let (batch, seq) = (2usize, 2usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let x0 = x.clone().narrow(1, 0, 1).squeeze_dim::<2>(1);

    // One cursored step emits the latent and then the token, returning the
    // token's delta.
    let mut c_step = ClassCursor::whole(seq);
    let (y_step, cache_step) = layer.step(x0.clone(), None, Some(&mut c_step));

    // The same, split: prime the latent, then step the token.
    let mut c_prime = ClassCursor::whole(seq);
    let (primed, cache) = layer.prime(batch, None, Some(&mut c_prime));
    let (delta, latent) = primed.expect("the Start latent was waiting");
    let (y_prime, cache_prime) = layer.step(x0, cache, Some(&mut c_prime));
    assert_eq!(c_prime, c_step);
    assert!(
        max_abs_diff(y_step, y_prime.clone()) < 1e-4,
        "priming the latent changed the step that followed"
    );
    assert!(
        max_abs_diff(cache_step.ssm_bhpr, cache_prime.ssm_bhpr.clone()) < 1e-4,
        "priming the latent changed the state"
    );

    // `(delta, latent)` is the layer's own row and what it produced from it: the
    // sequence `[latent, x0]` run through `forward` gives both deltas.
    let reference = Tensor::cat(
        vec![latent.unsqueeze_dim::<3>(1), x.clone().narrow(1, 0, 1)],
        1,
    );
    let (y_ref, c_ref) = layer.forward(reference, None, Mamba2SsdPath::default());
    assert!(
        max_abs_diff(delta, y_ref.clone().narrow(1, 0, 1).squeeze_dim::<2>(1)) < 1e-4,
        "the primed delta is not the latent's"
    );
    assert!(
        max_abs_diff(y_prime, y_ref.narrow(1, 1, 1).squeeze_dim::<2>(1)) < 1e-4,
        "the stepped delta is not the token's"
    );
    assert!(max_abs_diff(c_ref.ssm_bhpr, cache_prime.ssm_bhpr) < 1e-4);
}

// The runtime enums prime as well, threading the tagged caches straight into
// the `step` that follows — the seedless loop over the unified API. The vocab
// LM runs the primed latent through `norm_f` and the head, so what comes back
// is logits to sample the first token from.
#[cfg(feature = "mamba2")]
#[test]
fn prime_through_the_runtime_enums() {
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let (batch, seq, vocab) = (2usize, 3usize, 8usize);

    let net = MambaLatentNetConfig::Mamba2 {
        input_size: 3,
        n_real_layers: 1,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: block.clone(),
        output_size: 2,
        final_norm: false,
        class_tokens: vec![ClassToken::Start],
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: crate::modules::ResidualsConfig::Standard,
        mlp: None,
    }
    .init(&device);

    let mut class = ClassCursors::new(seq);
    let (y, caches) = net.prime(batch, None, Some(&mut class));
    assert_eq!(y.expect("the class token was waiting").dims(), [batch, 2]);
    let (yt, _c) = net.step(
        Tensor::<2>::zeros([batch, 3], &device),
        caches,
        Some(&mut class),
    );
    assert_eq!(yt.dims(), [batch, 2]);

    // The vocab counterpart: class latents live on its stack (the LM has no
    // class tokens of its own).
    let lm = MambaVocabNet::Mamba2(
        VocabNetworkBuilder {
            vocab_size: vocab,
            pad_vocab_size_multiple: 1,
            layers: LayersBuilder::new(1, block).with_class_latents(vec![ClassLatent::Start]),
            missing_lm_head: true,
        }
        .init(&device),
    );
    let mut class = ClassCursors::new(seq);
    let (logits, caches) = lm.prime(batch, None, Some(&mut class));
    assert_eq!(
        logits.expect("the class latent was waiting").dims(),
        [batch, vocab]
    );
    // `Start` fires once: the next prime has nothing left to emit.
    let (again, caches) = lm.prime(batch, caches, Some(&mut class));
    assert!(again.is_none(), "the Start latent was primed twice");
    let (logits, _c) = lm.step(
        Tensor::<1, Int>::zeros([batch], &device),
        caches,
        Some(&mut class),
    );
    assert_eq!(logits.dims(), [batch, vocab]);
}

// --- streamed placement (chunked forward / cursored step) ------------------

// The chunk planner is the one place placement is decided, so check it directly:
// all four marker kinds, whole vs split, and an open-ended stream.
#[test]
fn class_chunk_plan_splits_a_sequence() {
    let markers = vec![
        ClassLatent::Start,
        ClassLatent::Middle,
        ClassLatent::End,
        ClassLatent::Custom(1),
    ];
    // L = 6 ⇒ S u0 C u1 u2 M u3 u4 u5 E, i.e. positions [0, 5, 9, 2] in `Vec`
    // order (Start, Middle, End, Custom).
    assert_eq!(class_marker_output_indices(&markers, 6), vec![0, 5, 9, 2]);

    // One call for the whole sequence: `(at, marker)`, `at` counting the chunk's
    // own tokens placed before the marker.
    let mut cursor = ClassCursor::whole(6);
    assert_eq!(
        class_chunk_plan(&markers, 6, &mut cursor, "test"),
        vec![(0, 0), (1, 3), (3, 1), (6, 2)]
    );
    assert_eq!(cursor.offset, 10);

    // Split 4 + 2: identical placement, `End` waiting for the closing chunk.
    let mut cursor = ClassCursor::whole(6);
    assert_eq!(
        class_chunk_plan(&markers, 4, &mut cursor, "test"),
        vec![(0, 0), (1, 3), (3, 1)]
    );
    assert_eq!(cursor.offset, 7);
    assert_eq!(
        class_chunk_plan(&markers, 2, &mut cursor, "test"),
        vec![(2, 2)]
    );
    assert_eq!(cursor.offset, 10);

    // `End` closes; `Custom` never does. At (or past) the end a `Custom` has no
    // token to precede, so it waits — for a caller that keeps going, or forever.
    let markers = vec![ClassLatent::End, ClassLatent::Custom(3)];
    // L = 3 ⇒ u0 u1 u2 E, the Custom's slot (4) staying unemitted.
    assert_eq!(class_marker_output_indices(&markers, 3), vec![3, 4]);
    let mut cursor = ClassCursor::whole(3);
    assert_eq!(
        class_chunk_plan(&markers, 3, &mut cursor, "test"),
        vec![(3, 0)]
    );
    assert_eq!(cursor.offset, 4);
    // One token past the announced length and the `Custom` does land — still
    // *before* that token, as `Custom` always does.
    assert_eq!(
        class_chunk_plan(&markers, 1, &mut cursor, "test"),
        vec![(0, 1)]
    );
    assert_eq!(cursor.offset, 6);

    // An open-ended stream (no hint): `Start`/`Custom` still place exactly, and
    // nothing is ever treated as trailing.
    let markers = vec![ClassLatent::Start, ClassLatent::Custom(1)];
    let mut cursor = ClassCursor::default();
    assert_eq!(
        class_chunk_plan(&markers, 1, &mut cursor, "test"),
        vec![(0, 0)]
    );
    assert_eq!(cursor.offset, 2);
    assert_eq!(
        class_chunk_plan(&markers, 1, &mut cursor, "test"),
        vec![(0, 1)]
    );
    assert_eq!(cursor.offset, 4);
}

// Splitting one `forward` in two must place every class marker exactly where a
// single call over the whole sequence does — the shared cursors decide, not the
// chunk length. Stack latents at all four kinds of position plus a per-layer
// latent, so `Start` (behind the second chunk's cursor), `Custom`/`Middle`
// (interior) and `End` (only on the chunk that closes the sequence) are all
// exercised, at both class levels. Outputs **and** final caches must agree.
#[cfg(feature = "mamba2")]
#[test]
fn class_markers_split_forward_matches_single_forward() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let mut layers = LayersBuilder::new(2, block)
        .with_class_latents(vec![
            ClassLatent::Start,
            ClassLatent::Middle,
            ClassLatent::End,
            ClassLatent::Custom(1),
        ])
        .init(&device);
    // A per-layer latent as well (those aren't builder-configurable).
    layers.real_layers[0].class_latents = vec![ClassLatent::Custom(2)];
    layers.real_layers[0].class_latents_emb = init_class_emb(1, d_model, &device);

    let (batch, seq, split) = (2usize, 6usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let path = Mamba2SsdPath::default();

    // One call over the whole sequence (what `None` cursors mean).
    let (y_full, c_full) = layers.forward(x.clone(), None, path.clone(), None);
    assert_eq!(y_full.dims(), [batch, seq + 5, d_model]);

    // The same sequence in two chunks, sharing one cursor set.
    let mut class = ClassCursors::new(seq);
    let (y_a, caches) = layers.forward(
        x.clone().narrow(1, 0, split),
        None,
        path.clone(),
        Some(&mut class),
    );
    let (y_b, c_split) = layers.forward(
        x.narrow(1, split, seq - split),
        Some(caches),
        path,
        Some(&mut class),
    );
    // Chunk 1 carries Start, Custom and Middle; chunk 2 the trailing End.
    assert_eq!(y_a.dims(), [batch, split + 4, d_model]);
    assert_eq!(y_b.dims(), [batch, seq - split + 1, d_model]);

    let y_split = Tensor::cat(vec![y_a, y_b], 1);
    assert_eq!(y_split.dims(), y_full.dims());
    assert!(
        max_abs_diff(y_full, y_split) < 1e-4,
        "chunked class placement disagrees with the single forward"
    );
    for (i, (f, s)) in c_full.caches.iter().zip(&c_split.caches).enumerate() {
        assert!(
            max_abs_diff(f.conv_bvk.clone(), s.conv_bvk.clone()) < 1e-4,
            "layer {i} conv state disagrees"
        );
        assert!(
            max_abs_diff(f.ssm_bhpr.clone(), s.ssm_bhpr.clone()) < 1e-4,
            "layer {i} ssm state disagrees"
        );
    }
    // Every marker landed exactly once: 4 stack latents, then 1 per-layer one.
    assert_eq!(class.stack, seq + 4);
    assert_eq!(class.per_layer, vec![seq + 5, seq + 5]);
}

// With a full-length hint `step` places `Middle`/`End` too: the interior latent
// opens the step that carries the token it precedes, while the closing one is
// stepped *after* the last user token — and being the sequence's true last
// token, it is that step's returned output. Every step's output is `forward`'s
// row for the last token it emitted, and the final caches must match too.
#[cfg(feature = "mamba2")]
#[test]
fn class_markers_step_matches_forward_with_full_len() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(2, block)
        .with_class_latents(vec![ClassLatent::Middle, ClassLatent::End])
        .init(&device);

    let (batch, seq) = (2usize, 4usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // forward ⇒ u0 u1 M u2 u3 E. Each step emits up to its last token: u0, u1,
    // then (M, u2), then (u3, E) — so the closing latent, not u3, ends the run.
    let (y_fwd, c_fwd) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq + 2, d_model]);
    let last_pos = [0usize, 1, 3, 5];

    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    for (t, &pos) in last_pos.iter().enumerate() {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut class));
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, pos, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-4,
            "step {t} disagrees with forward at its last emitted token"
        );
    }
    assert_eq!(class.stack, seq + 2);

    // The `End` latent was stepped after the last token, so the final state
    // matches the one `forward` produced over the full lengthened sequence.
    let c_step = caches.unwrap();
    for (i, (f, s)) in c_fwd.caches.iter().zip(&c_step.caches).enumerate() {
        assert!(
            max_abs_diff(f.conv_bvk.clone(), s.conv_bvk.clone()) < 1e-4,
            "layer {i} conv state disagrees"
        );
        assert!(
            max_abs_diff(f.ssm_bhpr.clone(), s.ssm_bhpr.clone()) < 1e-4,
            "layer {i} ssm state disagrees"
        );
    }
}

// A network's own class tokens stream the same way (they are spliced at input
// width, before `in_proj`): two chunks must equal one call.
#[cfg(feature = "mamba2")]
#[test]
fn class_tokens_split_forward_matches_single_forward() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let net = LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder::new(2, block).with_class_latents(vec![ClassLatent::End]),
        output_size: 2,
        final_norm: false,
        class_tokens: vec![ClassToken::Start, ClassToken::End],
    }
    .init(&device);

    let (batch, seq, split) = (2usize, 5usize, 3usize);
    let x = Tensor::<3>::random([batch, seq, 3], Distribution::Normal(0.0, 1.0), &device);
    let path = Mamba2SsdPath::default();

    // S u0 u1 u2 u3 u4 E, then the stack's own End latent after it.
    let (y_full, _c) = net.forward(x.clone(), None, path.clone(), None);
    assert_eq!(y_full.dims(), [batch, seq + 3, 2]);

    let mut class = ClassCursors::new(seq);
    let (y_a, caches) = net.forward(
        x.clone().narrow(1, 0, split),
        None,
        path.clone(),
        Some(&mut class),
    );
    let (y_b, _c) = net.forward(
        x.narrow(1, split, seq - split),
        Some(caches),
        path,
        Some(&mut class),
    );
    let y_split = Tensor::cat(vec![y_a, y_b], 1);
    assert_eq!(y_split.dims(), y_full.dims());
    assert!(
        max_abs_diff(y_full, y_split) < 1e-4,
        "chunked class-token placement disagrees with the single forward"
    );
    // 2 class tokens at the network level, and the stack's End latent above them.
    assert_eq!(class.network, seq + 2);
    assert_eq!(class.stack, seq + 3);
}

// `Middle`/`End` cannot be placed against an open-ended stream, in `forward`
// either — the hint is what makes them well-defined.
#[cfg(feature = "mamba2")]
#[test]
#[should_panic(expected = "need a full-length hint")]
fn class_markers_without_full_len_panic_in_forward() {
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(1, block)
        .with_class_latents(vec![ClassLatent::End])
        .init(&device);
    let mut class = ClassCursors::stream();
    let _ = layers.forward(
        Tensor::<3>::zeros([2, 4, 16], &device),
        None,
        Mamba2SsdPath::default(),
        Some(&mut class),
    );
}

// The two modes share one cursor set, so a prefill `forward` can hand over to
// `step` mid-sequence: the markers left in the tail land on the steps that
// follow, and the stepped user tokens still match the single forward.
#[cfg(feature = "mamba2")]
#[test]
fn class_markers_forward_then_step() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(2, block)
        .with_class_latents(vec![
            ClassLatent::Start,
            ClassLatent::Middle,
            ClassLatent::End,
        ])
        .init(&device);

    let (batch, seq, prefill) = (2usize, 4usize, 2usize);
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // forward ⇒ S u0 u1 M u2 u3 E; the decoded steps end on u2 (M opens that
    // step) and on E (which closes the sequence after u3).
    let (y_fwd, _c) = layers.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq + 3, d_model]);
    let last_pos = [1usize, 2, 4, 6];

    // Prefill the first two tokens (Start comes along), then decode the rest.
    let mut class = ClassCursors::new(seq);
    let (y_pre, mut caches) = layers.forward(
        x.clone().narrow(1, 0, prefill),
        None,
        Mamba2SsdPath::default(),
        Some(&mut class),
    );
    assert_eq!(y_pre.dims(), [batch, prefill + 1, d_model]);
    assert_eq!(class.stack, prefill + 1);

    for (t, &pos) in last_pos.iter().enumerate().skip(prefill) {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, Some(caches), Some(&mut class));
        caches = c;
        let expected = y_fwd.clone().narrow(1, pos, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-4,
            "step {t} disagrees with forward at its last emitted token"
        );
    }
    // Middle opened the third step, End closed the last one.
    assert_eq!(class.stack, seq + 3);
}

// A network's class tokens `step` too, at both ends: `Start` is stepped (as a
// full network pass, so the stack splices its own latents around it) before the
// first user token, `End` after the last one — and `End`, closing the sequence,
// is what that final step returns.
#[cfg(feature = "mamba2")]
#[test]
fn class_tokens_step_matches_forward_with_full_len() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let net = LatentNetworkBuilder {
        input_size: 3,
        layers: LayersBuilder::new(2, block).with_class_latents(vec![ClassLatent::Start]),
        output_size: 2,
        final_norm: false,
        class_tokens: vec![ClassToken::Start, ClassToken::End],
    }
    .init(&device);

    let (batch, seq) = (2usize, 3usize);
    let x = Tensor::<3>::random([batch, seq, 3], Distribution::Normal(0.0, 1.0), &device);

    // forward ⇒ L S u0 u1 u2 E (the stack latent below the network's tokens);
    // the steps end on u0, u1, then E (the closing token, after u2).
    let (y_fwd, _c) = net.forward(x.clone(), None, Mamba2SsdPath::default(), None);
    assert_eq!(y_fwd.dims(), [batch, seq + 3, 2]);
    let last_pos = [2usize, 3, 5];

    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    for (t, &pos) in last_pos.iter().enumerate() {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = net.step(xt, caches, Some(&mut class));
        caches = Some(c);
        let expected = y_fwd.clone().narrow(1, pos, 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt, expected) < 1e-4,
            "step {t} disagrees with forward at its last emitted token"
        );
    }
    assert_eq!(class.network, seq + 2);
    assert_eq!(class.stack, seq + 3);
}

// `End` is the *only* marker that closes a sequence; `Custom` always precedes
// the token it names, whatever the index. Two `End`s and a `Custom(L)` over
// `L = 3` announced tokens, checked against the hand-built reference:
//
// ```text
//   sequence   u0 u1 u2 E1 E2 [C u3]     ← C and u3 only if the caller goes on
//   step 0     u0            → returns u0
//   step 1     u1            → returns u1
//   step 2     u2 E1 E2      → returns E2 (both closers ran; C did not)
//   step 3     C u3          → returns u3 (past the announced L: C finally has
//                               a token to precede, and still precedes it)
// ```
#[cfg(feature = "mamba2")]
#[test]
fn end_closes_the_sequence_custom_never_does() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let (batch, seq) = (2usize, 3usize);
    let mut layers = LayersBuilder::new(2, block)
        // Both `End`s sit at index `seq` and trail, in `Vec` order; `Custom`
        // names the same index but has no token there to precede.
        .with_class_latents(vec![
            ClassLatent::End,
            ClassLatent::End,
            ClassLatent::Custom(seq),
        ])
        .init(&device);
    let path = Mamba2SsdPath::default();
    // One token more than the announced length, fed at the very end.
    let x = Tensor::<3>::random(
        [batch, seq + 1, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // The reference, spliced by hand: u0 u1 u2 E1 E2 C u3.
    let emb = layers.class_latents_emb.as_ref().unwrap().val();
    let cls = |i: usize| {
        emb.clone()
            .narrow(0, i, 1)
            .unsqueeze_dim::<3>(0)
            .expand([batch, 1, d_model])
    };
    let tok = |t: usize| x.clone().narrow(1, t, 1);
    let reference = Tensor::cat(
        vec![tok(0), tok(1), tok(2), cls(0), cls(1), cls(2), tok(3)],
        1,
    );

    // `Custom(seq)` reports a position past the emitted sequence ⇒ it does not
    // land, and `forward` over the announced tokens is `u0 u1 u2 E1 E2`.
    assert_eq!(layers.class_latent_output_indices(seq), vec![3, 4, 5]);
    let (y_marked, _c) = layers.forward(x.clone().narrow(1, 0, seq), None, path.clone(), None);
    assert_eq!(y_marked.dims(), [batch, seq + 2, d_model]);

    // Step all four tokens — the last one runs past the announced length.
    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    let mut got = Vec::new();
    for t in 0..seq + 1 {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut class));
        got.push((yt, class.stack));
        caches = Some(c);
    }
    // u0 u1 u2 E1 E2 C u3 — every marker landed exactly once, `Custom` last.
    assert_eq!(class.stack, seq + 4);

    // Same weights over the hand-built reference, markers cleared.
    layers.class_latents = Vec::new();
    let (y_ref, c_ref) = layers.forward(reference.clone(), None, path.clone(), None);
    assert!(
        max_abs_diff(y_marked, y_ref.clone().narrow(1, 0, seq + 2)) < 1e-4,
        "the markers did not land where the reference splices them"
    );

    let last = [0usize, 1, 4, 6]; // u0, u1, E2, u3
    let dropped = [0usize, 1, 3, 5]; // (itself), (itself), E1, C
    for (t, (yt, cursor)) in got.into_iter().enumerate() {
        let want = y_ref.clone().narrow(1, last[t], 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt.clone(), want) < 1e-4,
            "step {t} did not return its last emitted token"
        );
        if dropped[t] != last[t] {
            let other = y_ref.clone().narrow(1, dropped[t], 1).squeeze_dim::<2>(1);
            assert!(
                max_abs_diff(yt, other) > 1e-3,
                "step {t} returned a token it should have dropped"
            );
        }
        assert_eq!(cursor, last[t] + 1, "step {t} left the cursor elsewhere");
    }

    // Every token of the reference ran through the recurrence.
    let c_step = caches.unwrap();
    for (i, (f, s)) in c_ref.caches.iter().zip(&c_step.caches).enumerate() {
        assert!(
            max_abs_diff(f.conv_bvk.clone(), s.conv_bvk.clone()) < 1e-4,
            "layer {i} conv state disagrees"
        );
        assert!(
            max_abs_diff(f.ssm_bhpr.clone(), s.ssm_bhpr.clone()) < 1e-4,
            "layer {i} ssm state disagrees"
        );
    }
}

// What a `step` hands back, per marker kind — pinned against a **hand-built**
// reference sequence instead of index arithmetic. The class rows are spliced
// into the input by hand where the four kinds claim to land, and the same
// weights (markers cleared, so nothing is inserted twice) are run over it:
//
// ```text
//   markers   Start, Custom(1), Middle, End         L = 4 user tokens
//   sequence  S u0 C u1 M u2 u3 E                   ← built explicitly below
//   step 0    S u0   → returns u0, state after 2 reference tokens
//   step 1    C u1   → returns u1, state after 4
//   step 2    M u2   → returns u2, state after 6
//   step 3    u3 E   → returns E (not u3!), state after 8
// ```
//
// `Start`/`Middle`/`Custom(k<L)` open the step carrying the token they precede,
// so the user token still ends it; `End` (like any `Custom(L)`) closes the
// sequence and ends the step instead — it is then the token whose output comes
// back, and the state is the one after it.
#[cfg(feature = "mamba2")]
#[test]
fn step_output_and_state_follow_the_last_emitted_token() {
    use crate::utils::test_helpers::max_abs_diff;
    use burn::tensor::Distribution;

    let device = Device::default();
    let d_model = 16;
    let block = Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let (batch, seq) = (2usize, 4usize);
    let mut layers = LayersBuilder::new(2, block)
        .with_class_latents(vec![
            ClassLatent::Start,
            ClassLatent::Middle,
            ClassLatent::End,
            ClassLatent::Custom(1),
        ])
        .init(&device);
    let path = Mamba2SsdPath::default();
    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // The reference sequence, spliced by hand: S u0 C u1 M u2 u3 E.
    let emb = layers.class_latents_emb.as_ref().unwrap().val();
    let cls = |i: usize| {
        emb.clone()
            .narrow(0, i, 1)
            .unsqueeze_dim::<3>(0)
            .expand([batch, 1, d_model])
    };
    let tok = |t: usize| x.clone().narrow(1, t, 1);
    let reference = Tensor::cat(
        vec![
            cls(0), // Start
            tok(0),
            cls(3), // Custom(1)
            tok(1),
            cls(1), // Middle
            tok(2),
            tok(3),
            cls(2), // End
        ],
        1,
    );
    assert_eq!(reference.dims(), [batch, seq + 4, d_model]);

    let state = |c: &Mamba2Caches| -> Vec<(Tensor<3>, Tensor<4>)> {
        c.caches
            .iter()
            .map(|c| (c.conv_bvk.clone(), c.ssm_bhpr.clone()))
            .collect()
    };
    let assert_state = |a: &[(Tensor<3>, Tensor<4>)], b: &[(Tensor<3>, Tensor<4>)], who: &str| {
        for (i, ((ca, sa), (cb, sb))) in a.iter().zip(b).enumerate() {
            assert!(
                max_abs_diff(ca.clone(), cb.clone()) < 1e-4,
                "{who}: layer {i} conv state disagrees"
            );
            assert!(
                max_abs_diff(sa.clone(), sb.clone()) < 1e-4,
                "{who}: layer {i} ssm state disagrees"
            );
        }
    };

    // Step the four user tokens, keeping what each step handed back.
    let mut class = ClassCursors::new(seq);
    let mut caches = None;
    let mut got = Vec::new();
    for t in 0..seq {
        let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (yt, c) = layers.step(xt, caches, Some(&mut class));
        got.push((yt, state(&c), class.stack));
        caches = Some(c);
    }

    // The marked `forward` claims to *be* that reference sequence…
    let (y_marked, c_marked) = layers.forward(x.clone(), None, path.clone(), None);
    // …so drop the markers (same weights) and run the reference itself.
    layers.class_latents = Vec::new();
    let (y_ref, c_ref) = layers.forward(reference.clone(), None, path.clone(), None);
    assert!(
        max_abs_diff(y_marked, y_ref.clone()) < 1e-4,
        "the markers did not land where the reference splices them"
    );
    assert_state(&state(&c_marked), &state(&c_ref), "forward");

    // Each step: the token it returned, the one it dropped, and its state.
    let last = [1usize, 3, 5, 7]; // u0, u1, u2, End
    let dropped = [0usize, 2, 4, 6]; // Start, Custom, Middle, u3
    let consumed = [2usize, 4, 6, 8]; // reference tokens emitted by then
    for (t, (yt, st, cursor)) in got.into_iter().enumerate() {
        assert_eq!(cursor, consumed[t], "step {t} left the cursor elsewhere");
        let want = y_ref.clone().narrow(1, last[t], 1).squeeze_dim::<2>(1);
        let other = y_ref.clone().narrow(1, dropped[t], 1).squeeze_dim::<2>(1);
        assert!(
            max_abs_diff(yt.clone(), want) < 1e-4,
            "step {t} did not return its last emitted token"
        );
        assert!(
            max_abs_diff(yt, other) > 1e-3,
            "step {t} returned the token it should have dropped"
        );
        // The state must be the reference's after exactly those tokens — so the
        // closing `End` of step 3 really did run through the recurrence.
        let (_y, c) = layers.forward(
            reference.clone().narrow(1, 0, consumed[t]),
            None,
            path.clone(),
            None,
        );
        assert_state(&st, &state(&c), &format!("step {t}"));
    }
}
