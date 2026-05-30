//! The model for the `mnist-ae` example — a symmetric, fully **bidirectional**
//! autoencoder over the 784-pixel MNIST sequence.
//!
//! Both halves are [`MambaBidiLayers`] stacks of pure real Mamba-3 layers
//! (`Complex2D` rotation). The encoder reads the image both ways and, **in place
//! of mean-pooling**, splices a single learnable [`ClassLatent::Middle`] token
//! into the middle of the sequence; after the bidi stack (where every position
//! has seen the whole image) that token's output is read back out as the image
//! summary and projected to a small **latent** `z` (the configurable
//! bottleneck). The decoder is the interesting part: it reconstructs the whole
//! image in **one parallel pass, reading only from `z`** — at every output
//! position its input is a learned positional embedding plus the broadcast
//! latent, so no ground-truth pixel ever reaches it (the bottleneck is real, the
//! reconstruction has to flow through `z`). This is the MAE-style decoder: a
//! bidirectional, positional-query, latent-conditioned generator.
//!
//! ```text
//!   img[b,784,1] ─enc_in_proj→ [b,784,d] ─enc_layers(bidi)→ mean_t → [b,d] ─enc_to_z→ z[b,n_latent]
//!   dec_in = pos_emb[b,784,d] + z_to_dec(z)[b,1,d]   ─dec_layers(bidi)→ [b,784,d] ─dec_out→ logits[b,784,1]
//! ```
//!
//! Refinement across depth is **implicit** here (the residual stack sharpens the
//! representation layer by layer; the pixels only materialise at `dec_out`). A
//! later iteration can add explicit per-layer deep supervision.

use crate::common::mnist::dataset::{HEIGHT, WIDTH};
use crate::common::model::ModelConfigExt;
use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn_mamba::generic::OutputMergeConfig;
use burn_mamba::prelude::{
    ClassLatent, Mamba3Config, Mamba3SsdPath, MambaBidiLayers, MambaBidiLayersConfig, MambaSsdPath,
    RotationKind,
};

/// A symmetric bidirectional MNIST autoencoder (see the module docs).
#[derive(Module, Debug)]
pub struct AeModel {
    /// Encoder input projection `input_size → d_model`.
    pub enc_in_proj: Linear,
    /// Bidirectional Mamba-3 encoder stack.
    pub enc_layers: MambaBidiLayers,
    /// Maps the pooled encoder output `d_model → n_latent` (the bottleneck).
    pub enc_to_z: Linear,
    /// Maps the latent back up `n_latent → d_model`, broadcast over the sequence.
    pub z_to_dec: Linear,
    /// Learned per-position query embedding, `[sequence, d_model]`.
    pub dec_pos_sd: Param<Tensor<2>>,
    /// Bidirectional Mamba-3 decoder stack.
    pub dec_layers: MambaBidiLayers,
    /// Decoder output projection `d_model → input_size` (pixel logits).
    pub dec_out: Linear,
    /// Sequence length (`HEIGHT * WIDTH`); the fixed reconstruction canvas size.
    pub sequence: usize,
}

impl AeModel {
    /// Memory-saving SSD path shared by both stacks (saves ~⅓ vram vs `Minimal`).
    fn ssd_path() -> MambaSsdPath {
        MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None))
    }

    /// Encode the image sequence into the latent `z`.
    ///
    /// `img`: `[batch, sequence, input_size]` → `z`: `[batch, n_latent]`.
    pub fn encode(&self, img_bsi: Tensor<3>) -> Tensor<2> {
        let [batch, sequence, _input_size] = img_bsi.dims();
        let h_bsd = self.enc_in_proj.forward(img_bsi); // [batch, sequence, d_model]
        // The bidi stack splices a `Middle` class latent (lengthening the
        // sequence by one): `enc_out` is `[batch, sequence + 1, d_model]`.
        let (enc_out, _caches) = self.enc_layers.forward(h_bsd, None, Self::ssd_path());
        // Read the summary out of the class-latent position — with a bidi encoder
        // that token has attended to the whole image, so it is an order-agnostic
        // summary (a learned alternative to mean-pooling). Fall back to mean-pool
        // if no class latent is configured.
        let pooled_bd = match self.enc_layers.class_latent_output_indices(sequence).first() {
            Some(&i) => enc_out.narrow(1, i, 1).squeeze_dim::<2>(1), // [batch, d_model]
            None => enc_out.mean_dim(1).squeeze_dim::<2>(1),
        };
        let z_bl = self.enc_to_z.forward(pooled_bd);
        let [batch_z, _n_latent] = z_bl.dims();
        assert_eq!(batch, batch_z);
        z_bl
    }

    /// Decode the latent `z` back into pixel logits, reading **only** from `z`.
    ///
    /// `z`: `[batch, n_latent]` → logits: `[batch, sequence, input_size]`.
    pub fn decode(&self, z_bl: Tensor<2>) -> Tensor<3> {
        let [_batch, _n_latent] = z_bl.dims();

        // Per-position learned queries: positions 0..sequence, the same for every
        // sample (they carry *where*, not *what*).
        let pos_1sd = self.dec_pos_sd.val().unsqueeze_dim::<3>(0); // [1, sequence, d_model]

        // Inject the latent at every position (broadcast add). This is the only
        // sample-specific signal the decoder sees.
        let seed_b1d = self.z_to_dec.forward(z_bl).unsqueeze_dim::<3>(1); // [batch, 1, d_model]
        let d_bsd = pos_1sd + seed_b1d; // broadcast → [batch, sequence, d_model]

        let (d_bsd, _caches) = self.dec_layers.forward(d_bsd, None, Self::ssd_path());
        let logits_bsi = self.dec_out.forward(d_bsd); // [batch, sequence, input_size]
        logits_bsi
    }

    /// Full autoencoder pass: `img` → reconstruction logits (same shape).
    pub fn forward(&self, img: Tensor<3>) -> Tensor<3> {
        let [batch, sequence, input_size] = img.dims();
        assert_eq!(sequence, self.sequence);
        let z = self.encode(img);
        let logits = self.decode(z);
        assert_eq!([batch, sequence, input_size], logits.dims());
        logits
    }
}

/// Configuration / factory for [`AeModel`].
#[derive(Config, Debug)]
pub struct AeConfig {
    /// Width of a pixel token (1 for grayscale MNIST).
    pub input_size: usize,
    /// Sequence length / canvas size (`HEIGHT * WIDTH`).
    pub sequence: usize,
    /// Model width shared by both stacks (must equal `mamba_block.d_model`).
    pub d_model: usize,
    /// The latent bottleneck width — the configurable "number of latents".
    pub n_latent: usize,
    /// Number of real encoder layers (must be even — bidi pairs).
    pub n_enc_layers: usize,
    /// Number of real decoder layers (must be even — bidi pairs).
    pub n_dec_layers: usize,
    /// Shared Mamba-3 block config for both stacks.
    pub mamba_block: Mamba3Config,
    /// Encoder class latents, spliced into the sequence.
    #[config(default = "Vec::new()")]
    pub enc_class_latents: Vec<ClassLatent>,
}

impl AeConfig {
    /// Allocate and initialise the autoencoder on `device`.
    pub fn init(&self, device: &Device) -> AeModel {
        assert_eq!(
            self.d_model, self.mamba_block.d_model,
            "AeConfig.d_model must match the Mamba-3 block d_model"
        );
        let d = self.d_model;

        // The encoder carries the class latents (its summary readout); the
        // decoder reconstructs at the original resolution, so it has none.
        let enc_layers = MambaBidiLayersConfig::Mamba3 {
            n_real_layers: self.n_enc_layers,
            mamba_block: self.mamba_block.clone(),
            outputs_merge: OutputMergeConfig::mean(self.n_enc_layers),
            class_latents: self.enc_class_latents.clone(),
        }
        .init(device);
        let dec_layers = MambaBidiLayersConfig::Mamba3 {
            n_real_layers: self.n_dec_layers,
            mamba_block: self.mamba_block.clone(),
            outputs_merge: OutputMergeConfig::mean(self.n_dec_layers),
            class_latents: Vec::new(),
        }
        .init(device);

        // Learned per-position decoder query embedding, `[sequence, d_model]`.
        let dec_pos_sd = Initializer::Normal { mean: 0.0, std: 0.02 }.init([self.sequence, d], device);

        AeModel {
            enc_in_proj: LinearConfig::new(self.input_size, d).init(device),
            enc_layers,
            enc_to_z: LinearConfig::new(d, self.n_latent).init(device),
            z_to_dec: LinearConfig::new(self.n_latent, d).init(device),
            dec_pos_sd,
            dec_layers,
            dec_out: LinearConfig::new(d, self.input_size).init(device),
            sequence: self.sequence,
        }
    }
}

impl ModelConfigExt for AeConfig {
    type Model = AeModel;
    fn init(&self, device: &Device) -> Self::Model {
        // Inherent `init` wins over this trait method in call syntax, so this
        // delegates rather than recursing.
        self.init(device)
    }
}

/// The example model config: a deliberately **tiny** (fibonacci-scale) AE.
///
/// `d_model = 32`, `expand = 2` (`d_inner = 64`), `per_head_dim = 32`
/// (`nheads = 2`), `state_rank = 64`, two real bidi layers per half, full RoPE,
/// `Complex2D` rotation. `n_latent` is the caller-chosen bottleneck width
/// (`-- --latents N`, default 32). The encoder uses a single `Middle` class
/// latent in place of mean-pooling.
pub fn model_config(n_latent: usize) -> AeConfig {
    let d_model = 32;
    let mamba_block = Mamba3Config::new(d_model)
        .with_state_rank(64)
        .with_expand(2)
        .with_per_head_dim(32) // d_inner = 64, nheads = 64/32 = 2
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(1.0)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        .with_rotation(RotationKind::Complex2D);

    AeConfig::new(
        1,
        HEIGHT * WIDTH,
        d_model,
        n_latent,
        2,
        2,
        mamba_block,
    )
    .with_enc_class_latents(vec![ClassLatent::Middle])
}
