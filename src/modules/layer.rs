use crate::modules::{GatedMlp, RmsNorm};
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
/// With [`Self::mlp`] set the layer additionally runs a second Pre-LN sub-block,
/// the SwiGLU feed-forward of `mamba_ssm`'s `d_intermediate > 0` checkpoints. It
/// has a residual of its own, *inside* the layer, which is the reason the
/// methods below return the layer's **total delta** rather than the mixer output:
///
/// ```text
///   h₁ = M(norm(x))                     the mixer sub-block
///   h₂ = mlp(norm2(x + h₁))             the feed-forward sub-block
///   return h₁ + h₂                      so that Layers' `x + delta` is
///                                       (x + h₁) + h₂ — both residuals
/// ```
///
/// Folding it this way keeps [`Layers`] the single owner of the *outer* residual
/// (and of the `ignore_first/last_residual` ablations, which therefore govern
/// only that outer add — the feed-forward's inner residual is intrinsic to the
/// sub-block and always applies). Without an `mlp` the delta is just `h₁` and
/// nothing changes for Mamba-1/2 or for Mamba-3 checkpoints that carry no MLP.
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
    /// Pre-norm of the feed-forward sub-block. `Some` exactly when [`Self::mlp`]
    /// is (`norm2` in the reference checkpoints).
    pub norm2: Option<RmsNorm>,
    /// Optional SwiGLU feed-forward sub-block run after the mixer, with its own
    /// residual. `None` ⇒ the layer is mixer-only.
    pub mlp: Option<GatedMlp>,
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

    /// The layer input, kept only when the feed-forward sub-block needs it for
    /// its inner residual — otherwise `None`, so the mixer-only path still moves
    /// `x` straight into the pre-norm with no clone.
    fn mlp_residual<const D: usize>(&self, x: &Tensor<D>) -> Option<Tensor<D>> {
        self.mlp.as_ref().map(|_| x.clone())
    }

    /// Completes the layer's total delta: `h₁ ↦ h₁ + mlp(norm2(x + h₁))`.
    ///
    /// `residual` is whatever [`Self::mlp_residual`] captured, so a `None` here
    /// means there is no feed-forward and the delta is the mixer output alone.
    fn add_mlp_delta<const D: usize>(
        &self,
        residual: Option<Tensor<D>>,
        h1: Tensor<D>,
    ) -> Tensor<D> {
        let Some(mlp) = self.mlp.as_ref() else {
            return h1;
        };
        let x = residual.expect("`mlp_residual` captures the input whenever `mlp` is present");
        let norm2 = self
            .norm2
            .as_ref()
            .expect("`norm2` is allocated alongside `mlp`");
        let h2 = mlp.forward(norm2.forward(x + h1.clone()));
        h1 + h2
    }

    /// Full-sequence Pre-LN block **without** the outer residual: the layer's
    /// total delta `M(RMSNorm(x))`, plus the feed-forward sub-block's own
    /// contribution when [`Self::mlp`] is set (see the type docs).
    ///
    /// The caller owns any class-latent insertion ([`Self::insert_latents`]) and
    /// the outer residual.
    pub fn forward(
        &self,
        x: Tensor<3>,
        cache: Option<M::Cache>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Cache) {
        let residual = self.mlp_residual(&x);
        let normed = self.norm.forward(x);
        let (h1, cache) = self.mamba_block.block_forward(normed, cache, ssd_path);
        (self.add_mlp_delta(residual, h1), cache)
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
            // The actual one-token work (no class injection, no outer residual).
            assert_step_compatible(&self.class_latents, "Layer");
            let residual = self.mlp_residual(&x);
            let normed = self.norm.forward(x);
            let (h1, cache) = self.mamba_block.block_step(normed, cache);
            return (self.add_mlp_delta(residual, h1), cache);
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

    /// Stationary fixed point of the Pre-LN block under a constant token,
    /// **without** the residual: the `step` counterpart of infinitely many
    /// identical tokens (closed form, no cache — see
    /// [`MambaBlock::block_step_infinite`]). Cursorless: class latents are not
    /// injected (`Middle`/`End` latents panic, as in a `None`-cursor `step`).
    /// The feed-forward sub-block is point-wise, so it composes with the limit:
    /// once the mixer output settles, `x + h₁` is constant and so is `h₂`.
    pub fn step_infinite(&self, x: Tensor<2>) -> Tensor<2> {
        assert_step_compatible(&self.class_latents, "Layer");
        let residual = self.mlp_residual(&x);
        let h1 = self.mamba_block.block_step_infinite(self.norm.forward(x));
        self.add_mlp_delta(residual, h1)
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
