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
        .with_rotation_fraction(1.0)
        .with_siso_specialization_decode(true)
        .with_rotation(RotationKind::Complex2D);
    let stack = NetworkShape::new(5)
        .with_class_latents(vec![ClassLatent::Start; N_LATENTS])
        .with_residuals(ResidualsConfig::MultiGate {
            n_stream: 4,
            init_bias: 0.0,
            init_bias_offset: 0.0,
            per_virtual_layer: false,
        });
    MambaVocabNetConfig::new(
        VocabShape::new(VOCAB_SIZE, stack).with_missing_lm_head(false),
        MambaBlockConfig::Mamba3(mamba_block),
    )
    .init(device)
    .map(&mut Jitter)
}
