//! The model for the `mnist-ae` example — a symmetric, fully **bidirectional**
//! **patch** autoencoder over MNIST (a ViT/MAE-style design with the attention
//! blocks replaced by Mamba-3 blocks).
//!
//! Instead of a 784-long sequence of single-pixel tokens, the 28×28 image is cut
//! into non-overlapping `patch×patch` tiles (default 7×7 ⇒ a length-16 sequence
//! of 49-pixel tokens). Short, content-rich tokens are far easier for the SSD
//! scan to route — and the ~16× shorter sequence frees the VRAM the old
//! single-pixel design spent on length.
//!
//! Both halves are [`MambaBidiLayers`] stacks of Mamba-3 layers.
//!
//! - **Encoder**: `patchify → enc_in_proj → +patch_pos → bidi stack → mean-pool →
//!   enc_to_z`. The pooled summary is projected to the small **latent** `z` (the
//!   configurable bottleneck). With a bidi encoder every patch has seen the whole
//!   image, so the mean is an order-agnostic summary. (A `Middle` [`ClassLatent`]
//!   readout is available as an optional alternative to mean-pooling.)
//! - **Decoder**: reconstructs every patch in **one parallel pass, reading only
//!   from `z`**. Each output position is a learned positional query
//!   ([`dec_pos`](AeModel::dec_pos)) **modulated by `z` via FiLM**
//!   (`(1+scale(z))·query + shift(z)`), so the only sample-specific signal is `z`
//!   (the bottleneck is real). The bidi decoder stack then refines, and
//!   `dec_out → unpatchify` lays the patches back onto the 28×28 canvas as pixel
//!   logits.
//!
//! ```text
//!   img[b,28,28,1] ─patchify→ [b,np,p²] ─enc_in_proj+pos→ [b,np,d] ─enc(bidi)→ mean → [b,d] ─enc_to_z→ z[b,n_latent]
//!   dec_in[b,np,d] = (1+scale(z))·pos + shift(z)   ─dec(bidi)→ [b,np,d] ─dec_out→ [b,np,p²] ─unpatchify→ logits[b,784]
//! ```

use crate::common::mnist::dataset::{HEIGHT, WIDTH};
use crate::common::model::ModelConfigExt;
use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn_mamba::modules::bidi::OutputMergeConfig;
use burn_mamba::prelude::{
    ClassLatent, Mamba3Config, Mamba3SsdPath, MambaBidiLayers, MambaBidiLayersConfig, MambaSsdPath,
    RotationKind,
};

/// How the decoder conditions its per-position queries on the latent `z`.
///
/// A plain `Copy` enum (no `Module`/`Config` derive) usable both as a
/// `#[derive(Module)]` constant field and a `#[derive(Config)]` field — same
/// recipe as `RotationKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DecoderCond {
    /// FiLM: `dec_in = (1 + scale(z)) · query + shift(z)` — `z` multiplicatively
    /// modulates and biases each positional query channel (recommended).
    Film,
    /// Additive broadcast: `dec_in = query + project(z)` — `z` added identically
    /// at every position (the original, weaker conditioning).
    Add,
}

/// Cut a `[batch, height, width, 1]` image into a `[batch, npatch, patch²]`
/// sequence of non-overlapping `patch×patch` tiles (row-major over patches).
pub fn patchify(img_bhw1: Tensor<4>, patch: usize) -> Tensor<3> {
    let [b, h, w, _c] = img_bhw1.dims();
    assert_eq!(h % patch, 0, "patch must divide height");
    assert_eq!(w % patch, 0, "patch must divide width");
    let (n_h, n_w) = (h / patch, w / patch);
    img_bhw1
        .reshape([b, h, w]) // drop the trailing channel
        .reshape([b, n_h, patch, n_w, patch]) // split rows & cols into patches
        .swap_dims(2, 3) // [b, n_h, n_w, patch, patch]
        .reshape([b, n_h * n_w, patch * patch]) // [b, npatch, patch²]
}

/// Inverse of [`patchify`]: lay a `[batch, npatch, patch²]` sequence back onto a
/// flat `[batch, height*width]` canvas.
pub fn unpatchify(x_bnp: Tensor<3>, patch: usize, height: usize, width: usize) -> Tensor<2> {
    let [b, _np, _pp] = x_bnp.dims();
    let (n_h, n_w) = (height / patch, width / patch);
    x_bnp
        .reshape([b, n_h, n_w, patch, patch])
        .swap_dims(2, 3) // [b, n_h, patch, n_w, patch]
        .reshape([b, height * width]) // [b, height*width]
}

/// A symmetric bidirectional MNIST **patch** autoencoder (see the module docs).
#[derive(Module, Debug)]
pub struct AeModel {
    /// Encoder patch projection `patch² → d_model`.
    pub enc_in_proj: Linear,
    /// Learned per-patch positional embedding, `[npatch, d_model]`.
    pub enc_pos: Param<Tensor<2>>,
    /// Bidirectional Mamba-3 encoder stack.
    pub enc_layers: MambaBidiLayers,
    /// Maps the pooled encoder output `d_model → n_latent` (the bottleneck).
    pub enc_to_z: Linear,
    /// Learned per-patch decoder query embedding, `[npatch, d_model]`.
    pub dec_pos: Param<Tensor<2>>,
    /// FiLM scale projection `n_latent → d_model` (used when `cond = Film`).
    pub z_to_scale: Linear,
    /// FiLM shift projection `n_latent → d_model` (also the additive seed when
    /// `cond = Add`).
    pub z_to_shift: Linear,
    /// Bidirectional Mamba-3 decoder stack.
    pub dec_layers: MambaBidiLayers,
    /// Decoder output projection `d_model → patch²` (per-patch pixel logits).
    pub dec_out: Linear,
    /// How the decoder conditions on `z`.
    #[module(skip)]
    pub cond: DecoderCond,
    /// Patch side length (must divide `HEIGHT` and `WIDTH`).
    pub patch: usize,
    /// Number of patches per image (`(HEIGHT/patch)·(WIDTH/patch)`).
    pub npatch: usize,
}

impl AeModel {
    /// Memory-saving SSD path shared by both stacks (saves ~⅓ vram vs `Minimal`).
    fn ssd_path() -> MambaSsdPath {
        MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None))
    }

    /// Encode the image into the latent `z`.
    ///
    /// `img`: `[batch, HEIGHT, WIDTH, 1]` → `z`: `[batch, n_latent]`.
    pub fn encode(&self, img_bhw1: Tensor<4>) -> Tensor<2> {
        let [batch, _h, _w, _c] = img_bhw1.dims();
        let patches_bsp = patchify(img_bhw1, self.patch); // [batch, npatch, patch²]
        let h_bsd = self.enc_in_proj.forward(patches_bsp); // [batch, npatch, d_model]
        let h_bsd = h_bsd + self.enc_pos.val().unsqueeze_dim::<3>(0); // + patch position
        // The bidi stack may splice class latents (lengthening the sequence).
        let (enc_out, _caches) = self.enc_layers.forward(h_bsd, None, Self::ssd_path());
        // Read the summary out of the class-latent position if configured (with a
        // bidi encoder that token has seen the whole image), else mean-pool over
        // the patches — both are order-agnostic image summaries.
        let pooled_bd = match self
            .enc_layers
            .class_latent_output_indices(self.npatch)
            .first()
        {
            Some(&i) => enc_out.narrow(1, i, 1).squeeze_dim::<2>(1), // [batch, d_model]
            None => enc_out.mean_dim(1).squeeze_dim::<2>(1),
        };
        let z_bl = self.enc_to_z.forward(pooled_bd);
        let [batch_z, _n_latent] = z_bl.dims();
        assert_eq!(batch, batch_z);
        z_bl
    }

    /// Decode the latent `z` into flat pixel logits, reading **only** from `z`.
    ///
    /// `z`: `[batch, n_latent]` → logits: `[batch, HEIGHT*WIDTH]`.
    pub fn decode(&self, z_bl: Tensor<2>) -> Tensor<2> {
        // Per-position learned queries: the same for every sample (they carry
        // *where*, not *what*).
        let pos_1sd = self.dec_pos.val().unsqueeze_dim::<3>(0); // [1, npatch, d_model]

        // The only sample-specific signal the decoder sees, injected from `z`.
        let dec_in = match self.cond {
            DecoderCond::Film => {
                // (1 + scale(z)) · query + shift(z): z modulates the canvas.
                let scale_b1d = self.z_to_scale.forward(z_bl.clone()).unsqueeze_dim::<3>(1);
                let shift_b1d = self.z_to_shift.forward(z_bl).unsqueeze_dim::<3>(1);
                pos_1sd.mul(scale_b1d.add_scalar(1.0)).add(shift_b1d) // → [batch, npatch, d_model]
            }
            DecoderCond::Add => {
                let seed_b1d = self.z_to_shift.forward(z_bl).unsqueeze_dim::<3>(1);
                pos_1sd.add(seed_b1d) // broadcast → [batch, npatch, d_model]
            }
        };

        let (d_bsd, _caches) = self.dec_layers.forward(dec_in, None, Self::ssd_path());
        let patches_bsp = self.dec_out.forward(d_bsd); // [batch, npatch, patch²]
        unpatchify(patches_bsp, self.patch, HEIGHT, WIDTH) // [batch, HEIGHT*WIDTH]
    }

    /// Full autoencoder pass: `img [b,H,W,1]` → flat reconstruction logits `[b,H*W]`.
    pub fn forward(&self, img_bhw1: Tensor<4>) -> Tensor<2> {
        let [batch, _h, _w, _c] = img_bhw1.dims();
        let z = self.encode(img_bhw1);
        let logits = self.decode(z);
        assert_eq!([batch, HEIGHT * WIDTH], logits.dims());
        logits
    }
}

/// Configuration / factory for [`AeModel`].
#[derive(Config, Debug)]
pub struct AeConfig {
    /// Patch side length (must divide `HEIGHT`=`WIDTH`=28: 2, 4, 7, or 14).
    /// `7` ⇒ 16 tokens of 49 px; `4` ⇒ 49 tokens of 16 px (finer detail).
    #[config(default = 7)]
    pub patch: usize,
    /// Model width shared by both stacks (must equal `mamba_block.d_model`).
    pub d_model: usize,
    /// The latent bottleneck width — the configurable "number of latents".
    pub n_latent: usize,
    /// Number of real encoder layers (bidi pairs).
    pub n_enc_layers: usize,
    /// Number of real decoder layers (bidi pairs).
    pub n_dec_layers: usize,
    /// Shared Mamba-3 block config for both stacks.
    pub mamba_block: Mamba3Config,
    /// Encoder class latents, spliced into the sequence (empty ⇒ mean-pool).
    #[config(default = "Vec::new()")]
    pub enc_class_latents: Vec<ClassLatent>,
    /// How the decoder conditions its per-position queries on `z`.
    #[config(default = "DecoderCond::Film")]
    pub cond: DecoderCond,
}

impl AeConfig {
    /// Allocate and initialise the autoencoder on `device`.
    pub fn init(&self, device: &Device) -> AeModel {
        assert_eq!(
            self.d_model, self.mamba_block.d_model,
            "AeConfig.d_model must match the Mamba-3 block d_model"
        );
        assert_eq!(HEIGHT % self.patch, 0, "patch must divide HEIGHT");
        assert_eq!(WIDTH % self.patch, 0, "patch must divide WIDTH");
        let d = self.d_model;
        let pdim = self.patch * self.patch;
        let npatch = (HEIGHT / self.patch) * (WIDTH / self.patch);

        // The encoder carries any class latents (its summary readout); the
        // decoder reconstructs at patch resolution, so it has none.
        let enc_layers = MambaBidiLayersConfig::Mamba3 {
            n_real_layers: self.n_enc_layers,
            n_virtual_layers: None,
            mamba_block: self.mamba_block.clone(),
            // first input (before flip) is non-bidi
            ignore_first_residual: true,
            // last output comes only from the state
            ignore_last_residual: true,
            outputs_merge: OutputMergeConfig::cat_linear(self.n_enc_layers),
            class_latents: self.enc_class_latents.clone(),
            residuals: burn_mamba::modules::ResidualsConfig::Standard,
        }
        .init(device);
        let dec_layers = MambaBidiLayersConfig::Mamba3 {
            n_real_layers: self.n_dec_layers,
            n_virtual_layers: None,
            mamba_block: self.mamba_block.clone(),
            // first input (before flip) is bidi
            ignore_first_residual: false,
            // last output comes only from the state
            ignore_last_residual: true,
            outputs_merge: OutputMergeConfig::cat_linear(self.n_dec_layers),
            class_latents: Vec::new(),
            residuals: burn_mamba::modules::ResidualsConfig::Standard,
        }
        .init(device);

        let pos_init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        AeModel {
            enc_in_proj: LinearConfig::new(pdim, d).init(device),
            enc_pos: pos_init.init([npatch, d], device),
            enc_layers,
            enc_to_z: LinearConfig::new(d, self.n_latent).init(device),
            dec_pos: pos_init.init([npatch, d], device),
            z_to_scale: LinearConfig::new(self.n_latent, d).init(device),
            z_to_shift: LinearConfig::new(self.n_latent, d).init(device),
            dec_layers,
            dec_out: LinearConfig::new(d, pdim).init(device),
            cond: self.cond,
            patch: self.patch,
            npatch,
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

/// The example model config: a small ViT/MAE-style patch AE sized to reach a
/// usable reconstruction in well under one epoch while staying within ~5GB VRAM.
///
/// `patch = 7` ⇒ a length-16 patch sequence; `d_model = 128`, `expand = 2`
/// (`d_inner = 256`), `per_head_dim = 32` (`nheads = 8`), `state_rank = 128`,
/// four real bidi layers per half, full RoPE, `Complex2D` rotation, FiLM decoder
/// conditioning. `n_latent` is the caller-chosen bottleneck width (`-- --latents
/// N`, default 64).
pub fn model_config(n_latent: usize) -> AeConfig {
    let d_model = 128;
    let mamba_block = Mamba3Config::new(d_model)
        .with_state_rank(128)
        .with_expand(2)
        // d_inner = expand·d_model = 2·128 = 256
        // per_head_dim = 32
        // nheads = d_inner/per_head_dim = 256/32 = 8
        .with_per_head_dim(32)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(1.0)
        .with_has_proj_bias(true)
        .with_has_outproj_norm(true)
        .with_rotation(RotationKind::Complex2D);

    AeConfig::new(d_model, n_latent, 4, 4, mamba_block)
}
