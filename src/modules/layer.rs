use crate::modules::RmsNorm;
use crate::prelude::*;
use crate::utils::ClassLatent;
use crate::utils::class::{assert_step_compatible, class_step_injections, insert_class_markers};
use burn::module::Param;
use burn::prelude::*;

/// A single Pre-LN block wrapper computing `M(RMSNorm(x))` — the residual is
/// **not** applied here. The enclosing [`Layers`](crate::modules::Layers) owns
/// that decision (add the input back, suppress it on the first/last layer, or
/// thread it through Multi-Gate streams), so no input clone / zero-add is wasted
/// when no residual is wanted.
///
/// May carry its own [`ClassLatent`]s. In `step` they are spliced via the
/// `index` cursor; in `forward` the caller splices them first (via
/// [`Self::insert_latents`]) so the residual it adds sees the same lengthened
/// sequence. They are independent of any class latents on the enclosing
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
    /// Public to the crate so [`Layers`](crate::modules::Layers) can lengthen the
    /// sequence itself (and add the matching residual) before calling
    /// [`Self::forward`].
    pub(crate) fn insert_latents(&self, x: Tensor<3>) -> Tensor<3> {
        if self.class_latents_emb.is_none() {
            return x;
        }
        insert_class_markers(x, &self.class_latents, self.class_latents_emb.as_ref()).0
    }

    /// Full-sequence Pre-LN block **without** the residual: `M(RMSNorm(x))`.
    ///
    /// The caller owns any class-latent insertion ([`Self::insert_latents`]) and
    /// the residual.
    pub fn forward(
        &self,
        x: Tensor<3>,
        cache: Option<M::Cache>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Cache) {
        let normed = self.norm.forward(x);
        self.mamba_block.block_forward(normed, cache, ssd_path)
    }

    /// Single-token Pre-LN block step **without** the residual.
    ///
    /// `index` is the running cursor into this layer's *output* sequence. With
    /// `Some`, whenever it lands on one of this layer's class-latent positions
    /// those latents are stepped first (each advancing `index`, recursing with
    /// `None`); only the user token's output and cache are returned. With `None`
    /// no class latents are injected — and `Middle`/`End` latents panic (their
    /// positions need the full sequence; use `forward`). The residual is the
    /// caller's responsibility.
    pub fn step(
        &self,
        x: Tensor<2>,
        cache: Option<M::Cache>,
        index: Option<&mut usize>,
    ) -> (Tensor<2>, M::Cache) {
        let Some(cursor) = index else {
            // The actual one-token work (no class injection, no residual).
            assert_step_compatible(&self.class_latents, "Layer");
            let normed = self.norm.forward(x);
            return self.mamba_block.block_step(normed, cache);
        };
        let [batch, d_model] = x.dims();
        let inj = class_step_injections(&self.class_latents, "Layer");
        let emb = self.class_latents_emb.as_ref();
        let mut cache = cache;
        while let Some(i) = inj.iter().position(|&p| p == *cursor) {
            let row = emb.unwrap().val().narrow(0, i, 1).expand([batch, d_model]);
            let (_discard, c) = self.step(row, cache, None);
            cache = Some(c);
            *cursor += 1;
        }
        let (out, cache) = self.step(x, cache, None);
        *cursor += 1;
        (out, cache)
    }
}
