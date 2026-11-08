//! The `{ shape, block }` configs. Each container and each family survives a
//! JSON round trip, and the parsed config builds a network of the same size.

use crate::prelude::*;
use burn::config::config_to_json;
use burn::prelude::*;
use burn_stack::modules::hnet::{HNetShape, HNetStageShape, HNetVocabShape};
use burn_stack::utils::test_helpers::test_device;

/// One block config of each enabled family.
fn block_configs() -> Vec<MambaBlockConfig> {
    let mut blocks = Vec::new();
    #[cfg(feature = "mamba1")]
    blocks.push(MambaBlockConfig::Mamba1(Mamba1Config::new(16).with_state_rank(8)));
    #[cfg(feature = "mamba2")]
    blocks.push(MambaBlockConfig::Mamba2(
        Mamba2Config::new(16).with_expand(2).with_per_head_dim(4).with_state_rank(8),
    ));
    #[cfg(feature = "mamba3")]
    blocks.push(MambaBlockConfig::Mamba3(
        Mamba3Config::new(16).with_expand(2).with_per_head_dim(4).with_state_rank(8),
    ));
    blocks
}

fn multi_gate() -> ResidualsConfig {
    ResidualsConfig::MultiGate {
        n_stream: 2,
        init_bias: 0.0,
        init_bias_offset: 0.0,
        per_virtual_layer: false,
    }
}

/// A stack with each knob away from its default.
fn stack() -> NetworkShape {
    NetworkShape::new(2)
        .with_n_virtual_layers(Some((4, Schedule::Stretched)))
        .with_grad_horizon(Some(GradHorizon::Depth(1)))
        .with_class_latents(vec![ClassLatent::Start])
        .with_ignore_first_residual(true)
        .with_ignore_last_residual(true)
        .with_residuals(multi_gate())
        .with_mlp(Some(GatedMlpConfig::new(16, 32).with_multiple_of(16)))
        .with_untied(vec![LayerUntied::Norm])
        .with_init(Some(InitPolicy::new().with_std(0.01)))
}

fn assert_stack_knobs(stack: &NetworkShape) {
    assert_eq!(stack.n_real_layers, 2);
    assert!(matches!(stack.n_virtual_layers, Some((4, Schedule::Stretched))));
    assert!(matches!(stack.grad_horizon, Some(GradHorizon::Depth(1))));
    assert_eq!(stack.class_latents.len(), 1);
    assert!(stack.ignore_first_residual && stack.ignore_last_residual);
    assert!(matches!(stack.residuals, ResidualsConfig::MultiGate { n_stream: 2, .. }));
    assert!(stack.mlp.is_some());
    assert_eq!(stack.untied.len(), 1);
    assert_eq!(stack.init.as_ref().map(|init| init.std), Some(0.01));
}

/// Writes `config` as JSON and parses that JSON. The parsed config must write
/// the same JSON, and the JSON holds the family tag of the block.
fn json_round_trip<C: Config>(config: &C) -> C {
    let json = config_to_json(config);
    let parsed = C::load_binary(json.as_bytes()).expect("the JSON of a config parses");
    assert_eq!(config_to_json(&parsed), json);
    parsed
}

#[test]
fn latent_net_config_survives_a_json_round_trip() {
    let device = test_device();
    for block in block_configs() {
        let shape = LatentShape::new(3, 5, stack())
            .with_final_norm(true)
            .with_class_tokens(vec![ClassToken::Start]);
        let config = MambaLatentNetConfig::new(shape, block);
        let parsed = json_round_trip(&config);

        assert_eq!((parsed.shape.input_size, parsed.shape.output_size), (3, 5));
        assert!(parsed.shape.final_norm);
        assert_eq!(parsed.shape.class_tokens.len(), 1);
        assert_stack_knobs(&parsed.shape.stack);
        assert_eq!(parsed.init(&device).num_params(), config.init(&device).num_params());
    }
}

#[test]
fn vocab_net_config_survives_a_json_round_trip() {
    let device = test_device();
    for block in block_configs() {
        let shape = VocabShape::new(20, stack())
            .with_pad_vocab_size_multiple(8)
            .with_missing_lm_head(false);
        let config = MambaVocabNetConfig::new(shape, block);
        let parsed = json_round_trip(&config);

        assert_eq!(parsed.shape.vocab_size, 20);
        assert_eq!(parsed.shape.pad_vocab_size_multiple, 8);
        assert!(!parsed.shape.missing_lm_head);
        assert_stack_knobs(&parsed.shape.stack);
        assert_eq!(parsed.init(&device).num_params(), config.init(&device).num_params());
    }
}

#[test]
fn bidi_layers_config_survives_a_json_round_trip() {
    let device = test_device();
    for block in block_configs() {
        let shape = BidiShape::new(2, OutputMergeConfig::cat_linear(2))
            .with_n_virtual_layers(Some((4, BidiSchedule::StridedStretched)))
            .with_ignore_first_residual(true)
            .with_ignore_last_residual(true)
            .with_class_latents(vec![ClassLatent::Middle])
            .with_residuals(multi_gate())
            .with_untied(vec![LayerUntied::Norm]);
        let config = MambaBidiLayersConfig::new(shape, block);
        let parsed = json_round_trip(&config);

        assert_eq!(parsed.shape.n_real_layers, 2);
        assert_eq!(parsed.shape.outputs_merge.len(), 1);
        assert!(matches!(
            parsed.shape.n_virtual_layers,
            Some((4, BidiSchedule::StridedStretched))
        ));
        assert!(parsed.shape.ignore_first_residual && parsed.shape.ignore_last_residual);
        assert_eq!(parsed.shape.class_latents.len(), 1);
        assert!(matches!(parsed.shape.residuals, ResidualsConfig::MultiGate { n_stream: 2, .. }));
        assert_eq!(parsed.shape.untied.len(), 1);
        assert_eq!(parsed.init(&device).num_params(), config.init(&device).num_params());
    }
}

/// Each pair of families: the stages take one, and the main stack takes the
/// other. `hnet.rs` of these tests builds and runs this config.
#[test]
fn hnet_vocab_net_config_survives_a_json_round_trip() {
    for stage in block_configs() {
        for main in block_configs() {
            let stack = || NetworkShape::new(2);
            let hnet = HNetShape::new(vec![HNetStageShape::new(stack(), stack())], stack());
            let shape = HNetVocabShape::new(hnet, 20)
                .with_pad_vocab_size_multiple(8)
                .with_missing_lm_head(true);
            let config = MambaHNetVocabNetConfig::new(shape, vec![stage.clone()], main);
            let parsed = json_round_trip(&config);

            assert_eq!(parsed.shape.vocab_size, 20);
            assert_eq!(parsed.shape.pad_vocab_size_multiple, 8);
            assert!(parsed.shape.missing_lm_head);
            assert_eq!(parsed.stage_blocks.len(), 1);
        }
    }
}

/// A path of another family panics in `forward`.
#[test]
fn default_ssd_path_fits_the_family_of_the_block() {
    let device = test_device();
    for block in block_configs() {
        let path = block.default_ssd_path();
        let net = MambaLatentNetConfig::new(LatentShape::new(3, 2, NetworkShape::new(1)), block)
            .init(&device);
        let (y, _) = net.forward(Tensor::<3>::zeros([2, 5, 3], &device), None, path, None, None);
        assert_eq!(y.dims(), [2, 5, 2]);
    }
}
