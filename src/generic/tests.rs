//! Smoke / parity tests for the family-generic abstraction in
//! [`crate::generic`] — builder wiring, the unifying enums, and the
//! class-token / class-latent insertion + step-injection machinery.

use super::*;

#[cfg(feature = "mamba2")]
#[test]
fn latent_network_builder_mamba2() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
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
    use crate::mamba2::prelude::Mamba2Config;
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
    use crate::mamba3::prelude::Mamba3Config;
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
    use crate::mamba1::prelude::Mamba1Config;
    let device = Device::default();
    let block = Mamba1Config::new(16).with_state_rank(8);
    let net = MambaLatentNetConfig::Mamba1 {
        input_size: 3,
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: block,
        output_size: 2,
        class_tokens: Vec::new(),
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
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
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
    use crate::mamba3::prelude::{Mamba3Config, Mamba3SsdPath};
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
    use crate::mamba1::prelude::Mamba1Config;
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
    use crate::mamba2::prelude::Mamba2Config;
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = MambaBidiLayersConfig::Mamba2 {
        n_real_layers: 2,
        mamba_block: block,
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

// --- class tokens / latents -----------------------------------------

// Start/Middle/End class latents lengthen the sequence and land at the
// documented output positions.
#[cfg(feature = "mamba2")]
#[test]
fn class_latents_lengthen_and_index() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(1, block)
        .with_class_latents(vec![ClassLatent::Start, ClassLatent::Middle, ClassLatent::End])
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
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
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
    use crate::mamba2::prelude::Mamba2Config;
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
// user-position slices, and the cursor lands past every emitted token.
#[cfg(feature = "mamba2")]
#[test]
fn class_latents_step_matches_forward() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
    use crate::utils::test_helpers::max_abs_diff;
    let device = Device::default();
    let block = Mamba2Config::new(16)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4);
    let layers = LayersBuilder::new(1, block)
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
