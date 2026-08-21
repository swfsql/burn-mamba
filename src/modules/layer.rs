use crate::modules::{GatedMlp, RmsNorm};
use crate::prelude::*;
use crate::utils::class::{
    assert_full_len_known, class_chunk_plan, class_emb_width, class_prime_plan, class_row,
    insert_class_markers,
};
use crate::utils::{ClassCursor, ClassLatent};
use burn::module::Param;
use burn::prelude::*;

/// A single Pre-LN block wrapper computing `M(RMSNorm(x))` — the residual is
/// **not** applied here. The enclosing [`Layers`] owns
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
/// May carry its own [`ClassLatent`]s, placed from a [`ClassCursor`]: `step`
/// splices them around the token it is given, while in `forward` the caller
/// splices them first (via [`Self::insert_latents`]) so the residual it adds
/// sees the same lengthened sequence; [`Self::prime`] steps the ones waiting for
/// the next token *without* that token. They are independent of any class
/// latents on the enclosing [`Layers`].
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
    /// Splice this layer's class latents into the chunk `x` (no-op when there
    /// are none), advancing `class` past it.
    ///
    /// Public so a caller driving a bare [`Layer`] can lengthen the sequence
    /// itself (and add the matching residual) before calling [`Self::forward`].
    /// `None` cursors ⇒ this chunk is the whole sequence. [`Layers`] splices its
    /// layers' latents itself, since under
    /// [`MultiGate`](crate::modules::MultiGate) residuals the same rows must
    /// also enter the carried streams.
    pub fn insert_latents(&self, x: Tensor<3>, class: Option<&mut ClassCursor>) -> Tensor<3> {
        let mut whole = ClassCursor::whole(x.dims()[1]);
        let cursor = class.unwrap_or(&mut whole);
        insert_class_markers(
            x,
            &self.class_latents,
            self.class_latents_emb.as_ref(),
            cursor,
            "Layer",
        )
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
    /// `class` is this layer's own class-latent cursor. With `Some`, every
    /// latent whose position falls on this token is stepped around it — before
    /// it (`Start`/`Middle`/`Custom`, which precede a token) or after it (`End`,
    /// which closes the sequence) — each a step of its own. What comes back is
    /// the **last** token the step emitted (see
    /// [`ClassCursors`](crate::utils::ClassCursors)): the user token, unless an
    /// `End` latent follows it, that latent being then the sequence's true last
    /// token. With `None` no class latents are injected — and `Middle`/`End`
    /// latents panic (their positions need the full sequence length). The
    /// residual is the caller's responsibility.
    pub fn step(
        &self,
        x: Tensor<2>,
        cache: Option<M::Cache>,
        class: Option<&mut ClassCursor>,
    ) -> (Tensor<2>, M::Cache) {
        let Some(cursor) = class else {
            assert_full_len_known(&self.class_latents, None, "Layer");
            return self.step_one(x, cache);
        };
        let plan = class_chunk_plan(&self.class_latents, 1, cursor, "Layer");
        if plan.is_empty() {
            return self.step_one(x, cache);
        }
        // `at == 0` ⇒ the latent precedes the user token, `at == 1` ⇒ it is an
        // `End` closing the sequence, and follows it.
        let [batch, d_model] = x.dims();
        let row = |i: usize| class_row(self.class_latents_emb.as_ref(), i, batch, d_model);
        let (before, after): (Vec<_>, Vec<_>) = plan.into_iter().partition(|&(at, _)| at == 0);
        let mut cache = cache;
        for (_, i) in before {
            let (_discard, c) = self.step_one(row(i), cache);
            cache = Some(c);
        }
        let (mut out, mut cache) = self.step_one(x, cache);
        for (_, i) in after {
            // A closing `End` *is* the sequence's last token — its output, not
            // the user token's, is what this step produced.
            let (o, c) = self.step_one(row(i), Some(cache));
            out = o;
            cache = c;
        }
        (out, cache)
    }

    /// Step the class latents this layer has waiting for its next token — with
    /// **no** token of its own, so nothing but class data is consumed.
    ///
    /// This is [`Self::step`]'s opening half on its own (see
    /// [`ClassCursors`](crate::utils::ClassCursors)): the latents that would
    /// have preceded the next token are stepped now, in the same order, so a
    /// `prime` followed by a `step` runs exactly the sequence that `step` alone
    /// would have. `End` latents are never primed — closing the sequence, they
    /// belong to the step carrying its last token.
    ///
    /// Returns the **last** latent stepped, as the pair `(delta, latent)` — this
    /// layer's own embedding row alongside the delta it produced, since the
    /// caller has no other way to complete the residual (`delta + latent`, as it
    /// does with the token it hands to [`Self::step`]). `None` ⇒ nothing was
    /// waiting, and the cache comes back exactly as it went in (`None` included:
    /// a layer that stepped nothing has the state it already had).
    pub fn prime(
        &self,
        batch: usize,
        cache: Option<M::Cache>,
        class: Option<&mut ClassCursor>,
    ) -> (Option<(Tensor<2>, Tensor<2>)>, Option<M::Cache>) {
        let Some(cursor) = class else {
            // No cursor ⇒ nothing is injected, exactly as in a `None` step.
            assert_full_len_known(&self.class_latents, None, "Layer");
            return (None, cache);
        };
        let plan = class_prime_plan(&self.class_latents, 0, cursor, "Layer");
        if plan.is_empty() {
            return (None, cache);
        }
        let width = class_emb_width(self.class_latents_emb.as_ref());
        let mut cache = cache;
        let mut last = None;
        for (_at, i) in plan {
            let row = class_row(self.class_latents_emb.as_ref(), i, batch, width);
            let (out, c) = self.step_one(row.clone(), cache);
            last = Some((out, row));
            cache = Some(c);
        }
        (last, cache)
    }

    /// The actual one-token work: no class injection, no outer residual.
    ///
    /// `pub(crate)` for [`Layers`]'s cascade, which places this layer's class
    /// latents from the stack-wide [`ClassCursors`](crate::utils::ClassCursors)
    /// itself and so must bypass [`Self::step`]'s cursorless guard (that guard
    /// rejects `Middle`/`End`, which the cascade has already resolved).
    pub(crate) fn step_one(&self, x: Tensor<2>, cache: Option<M::Cache>) -> (Tensor<2>, M::Cache) {
        let residual = self.mlp_residual(&x);
        let normed = self.norm.forward(x);
        let (h1, cache) = self.mamba_block.block_step(normed, cache);
        (self.add_mlp_delta(residual, h1), cache)
    }

    /// Stationary fixed point of the Pre-LN block under a constant token,
    /// **without** the residual: the `step` counterpart of infinitely many
    /// identical tokens (closed form, no cache — see
    /// [`MambaBlock::block_step_infinite`]). Cursorless: class latents are not
    /// injected (`Middle`/`End` latents panic, as in a `None`-cursor `step`).
    /// The feed-forward sub-block is point-wise, so it composes with the limit:
    /// once the mixer output settles, `x + h₁` is constant and so is `h₂`.
    pub fn step_infinite(&self, x: Tensor<2>) -> Tensor<2> {
        assert_full_len_known(&self.class_latents, None, "Layer");
        let residual = self.mlp_residual(&x);
        let h1 = self.mamba_block.block_step_infinite(self.norm.forward(x));
        self.add_mlp_delta(residual, h1)
    }
}

#[cfg(all(test, feature = "_dev-test"))]
mod tests;
