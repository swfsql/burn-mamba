use crate::modules::RmsNorm;
use crate::prelude::*;
use crate::utils::ClassLatent;
use crate::utils::class::{assert_step_compatible, class_step_injections, insert_class_markers};
use burn::module::Param;
use burn::prelude::*;

/// A single Pre-LN residual block: `output = x·residual_scale + M(RMSNorm(x))`.
///
/// May carry its own [`ClassLatent`]s, spliced into the sequence at the start of
/// `forward` (so the residual and the inner block both see the lengthened
/// sequence). They are independent of any class latents on the enclosing
/// [`Layers`].
#[derive(Module, Debug)]
pub struct Layer<M: Module> {
    /// Pre-norm applied before the inner block.
    pub norm: RmsNorm,
    /// The inner Mamba-x SSM block.
    pub mamba_block: M,
    /// Positions of this layer's class latents (empty ⇒ none).
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The class-latent embeddings, `[num_class_latents, d_model]` (`None` ⇒ none).
    pub class_latents_emb: Option<Param<Tensor<2>>>,
}

impl<M: MambaBlock> Layer<M> {
    /// Splice this layer's class latents into `x` (no-op when there are none).
    fn insert_latents(&self, x: Tensor<3>) -> Tensor<3> {
        if self.class_latents_emb.is_none() {
            return x;
        }
        insert_class_markers(x, &self.class_latents, self.class_latents_emb.as_ref()).0
    }

    /// Full-sequence Pre-LN residual pass.
    pub fn forward(
        &self,
        x: Tensor<3>,
        cache: Option<M::Cache>,
        ssd_path: M::SsdPath,
        residual_scale: f32,
    ) -> (Tensor<3>, M::Cache) {
        let x = self.insert_latents(x);
        let res = x.clone() * residual_scale;
        let normed = self.norm.forward(x);
        let (out, cache) = self.mamba_block.block_forward(normed, cache, ssd_path);
        (out + res, cache)
    }

    /// Single-token Pre-LN residual step.
    ///
    /// `index` is the running cursor into this layer's *output* sequence. With
    /// `Some`, whenever it lands on one of this layer's class-latent positions
    /// those latents are stepped first (each advancing `index`, recursing with
    /// `None`), then the user token (also advancing `index`); only the user
    /// token's output and cache are returned. With `None` no class latents are
    /// injected — and `Middle`/`End` latents panic (their positions need the full
    /// sequence; use `forward`).
    pub fn step(
        &self,
        x: Tensor<2>,
        cache: Option<M::Cache>,
        residual_scale: f32,
        index: Option<&mut usize>,
    ) -> (Tensor<2>, M::Cache) {
        let Some(cursor) = index else {
            // The actual one-token work (no class injection).
            assert_step_compatible(&self.class_latents, "Layer");
            let res = x.clone() * residual_scale;
            let normed = self.norm.forward(x);
            let (out, cache) = self.mamba_block.block_step(normed, cache);
            return (out + res, cache);
        };
        let [batch, d_model] = x.dims();
        let inj = class_step_injections(&self.class_latents, "Layer");
        let emb = self.class_latents_emb.as_ref();
        let mut cache = cache;
        while let Some(i) = inj.iter().position(|&p| p == *cursor) {
            let row = emb.unwrap().val().narrow(0, i, 1).expand([batch, d_model]);
            let (_discard, c) = self.step(row, cache, residual_scale, None);
            cache = Some(c);
            *cursor += 1;
        }
        let (out, cache) = self.step(x, cache, residual_scale, None);
        *cursor += 1;
        (out, cache)
    }
}
