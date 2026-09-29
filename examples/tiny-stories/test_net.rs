//! A tiny network for the tests of the example: five real layers with the
//! settings of the search models, at a tiny width and state.
//! `model::model_config` follows the parameter search, so the tests do not use
//! it.

use crate::dataset::VOCAB_SIZE;
use burn::module::{ModuleMapper, Param};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_mamba::prelude::*;
use burn_stack::utils::ClassLatent;

/// The class latents of the tiny network and of the search models.
pub const N_LATENTS: usize = 4;

/// Adds Normal noise to every float parameter.
struct Jitter;

impl ModuleMapper for Jitter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|t| {
            let noise = Tensor::random(t.shape(), Distribution::Normal(0.0, 0.2), &t.device());
            t + noise
        })
    }
}

/// MultiGate with 4 streams, projection bias, output norm, an LM head of its
/// own, full rotation, the SISO decode kernels, and [`N_LATENTS`] `Start`
/// latents.
///
/// Noise moves every parameter off its init. The zero-initialized tensors of a
/// fresh block make each layer add nothing, so no state (the latents included)
/// could change a logit.
pub fn tiny_net(device: &Device) -> MambaVocabNet {
    let mamba_block = Mamba3Config::new(16)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        .with_rope_fraction(1.0)
        .with_siso_specialization_decode(true)
        .with_rotation(RotationKind::Complex2D);
    MambaVocabNetConfig::Mamba3 {
        n_real_layers: 5,
        n_virtual_layers: None,
        grad_horizon: None,
        vocab_size: VOCAB_SIZE,
        pad_vocab_size_multiple: 1,
        mamba_block,
        missing_lm_head: false,
        class_latents: vec![ClassLatent::Start; N_LATENTS],
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: ResidualsConfig::MultiGate {
            n_stream: 4,
            init_bias: 0.0,
            init_bias_step: 0.0,
            per_virtual_layer: false,
        },
        mlp: None,
        untied: Vec::new(),
    }
    .init(device)
    .map(&mut Jitter)
}
