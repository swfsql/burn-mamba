use crate::modules::RmsNormConfig;
use crate::prelude::*;
use crate::utils::ClassLatent;
use crate::utils::Schedule;
use crate::utils::class::{
    assert_step_compatible, class_marker_output_indices, class_step_injections, init_class_emb,
    insert_class_markers,
};
use burn::module::Param;
use burn::prelude::*;

/// A stack of [`Layer`]s with optional virtual-layer scheduling — one struct for
/// every Mamba-x family.
#[derive(Module, Debug)]
pub struct Layers<M: Module> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,
    /// Optional `(n_virtual_layers, schedule)` for weight-sharing.
    #[module(skip)]
    pub n_virtual_layers: Option<(usize, Schedule)>,
    /// The weight-bearing layers, length `n_real_layers`.
    pub real_layers: Vec<Layer<M>>,
    /// Zero the first virtual layer's residual when `true`.
    pub ignore_first_residual: bool,
    /// Zero the last virtual layer's residual when `true`.
    pub ignore_last_residual: bool,
    /// Positions of the stack-level class latents, spliced into the sequence
    /// once before the first virtual layer (independent of any per-[`Layer`]
    /// class latents). Empty ⇒ none.
    #[module(skip)]
    pub class_latents: Vec<ClassLatent>,
    /// The stack-level class-latent embeddings, `[num_class_latents, d_model]`.
    pub class_latents_emb: Option<Param<Tensor<2>>>,
}

impl<M: MambaBlock> Layers<M>
where
    M::SsdPath: Clone,
{
    /// Output positions of the stack-level class latents for an `orig_len` input.
    pub fn class_latent_output_indices(&self, orig_len: usize) -> Vec<usize> {
        class_marker_output_indices(&self.class_latents, orig_len)
    }

    /// Splice this layers' class latents into `x` (no-op when there are none).
    fn insert_latents(&self, x: Tensor<3>) -> Tensor<3> {
        if self.class_latents_emb.is_none() {
            return x;
        }
        insert_class_markers(x, &self.class_latents, self.class_latents_emb.as_ref()).0
    }

    fn n_virtual_count(&self) -> usize {
        self.n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers)
    }

    fn real_idx(&self, virtual_idx: usize) -> usize {
        if let Some((n, schedule)) = &self.n_virtual_layers {
            schedule.real_idx(virtual_idx, *n, self.n_real_layers)
        } else {
            virtual_idx
        }
    }

    fn residual_scale(&self, i: usize, n: usize) -> f32 {
        let first = self.ignore_first_residual && i == 0;
        let last = self.ignore_last_residual && i + 1 == n;
        if first || last { 0.0 } else { 1.0 }
    }

    /// Full-sequence pass through every (virtual) layer.
    pub fn forward(
        &self,
        x: Tensor<3>,
        caches: Option<M::Caches>,
        ssd_path: M::SsdPath,
    ) -> (Tensor<3>, M::Caches) {
        let mut x = self.insert_latents(x);
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_3d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");

        let mut slots = caches.into_slots();
        for i in 0..n {
            let layer = &self.real_layers[self.real_idx(i)];
            let rs = self.residual_scale(i, n);
            let cache = slots[i].take().unwrap();
            let (x_, c_) = layer.forward(x, Some(cache), ssd_path.clone(), rs);
            x = x_;
            slots[i] = Some(c_);
        }
        (x, M::Caches::from_slots(slots))
    }

    /// Single-token step through every (virtual) layer.
    ///
    /// Two independent class-latent cursors:
    /// - `own_index` — the stack-level [`Self::class_latents`] (spliced once
    ///   before the first layer in `forward`). When it lands on a stack position
    ///   that latent enters the bottom of the stack as an extra input token.
    /// - `layer_indices` — one cursor **per virtual layer**
    ///   (`len == n_virtual_layers`), the per-[`Layer`] cursor so each layer
    ///   splices its own class latents into the token stream it receives.
    ///
    /// Because a layer's class latents grow the sequence the *next* layer sees
    /// (exactly as in `forward`), a single user step is a **cascade**: the bottom
    /// input stream (any stack latents at `own_index`, then the user token) is
    /// threaded up the stack, each layer expanding it with its own class latents
    /// at `layer_indices[i]`. Every layer's recurrence therefore sees the same
    /// token order as `forward`, so `forward` and `step` agree. Only the user
    /// token's (fully propagated) output is returned — it is emitted last.
    ///
    /// A `None` cursor skips that level's injection; `Middle`/`End` latents panic
    /// for the cursored level (their positions need the full sequence — use
    /// `forward`).
    pub fn step(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        own_index: Option<&mut usize>,
        mut layer_indices: Option<&mut Vec<usize>>,
    ) -> (Tensor<2>, M::Caches) {
        let [batch, d_model] = x.dims();
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_2d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");
        if let Some(v) = layer_indices.as_deref() {
            assert_eq!(v.len(), n, "one class-latent cursor per virtual layer");
        }
        let mut slots = caches.into_slots();

        // Bottom input stream for this user step: the stack-level class latents
        // that fall at the cursor (fed through the whole stack like ordinary
        // inputs), then the user token. Without a cursor, just the user token.
        let mut stream: Vec<Tensor<2>> = Vec::new();
        if let Some(own_cursor) = own_index {
            let positions = class_step_injections(&self.class_latents, "Layers");
            let emb = self.class_latents_emb.as_ref();
            while let Some(i) = positions.iter().position(|&p| p == *own_cursor) {
                stream.push(emb.unwrap().val().narrow(0, i, 1).expand([batch, d_model]));
                *own_cursor += 1;
            }
            stream.push(x);
            *own_cursor += 1;
        } else {
            assert_step_compatible(&self.class_latents, "Layers");
            stream.push(x);
        }

        // Propagate the stream up through each virtual layer, each layer splicing
        // its own class latents into the stream it receives.
        for pos in 0..n {
            let layer = &self.real_layers[self.real_idx(pos)];
            let rs = self.residual_scale(pos, n);
            let mut layer_cursor = layer_indices.as_deref_mut().map(|v| &mut v[pos]);
            let positions = if layer_cursor.is_some() {
                class_step_injections(&layer.class_latents, "Layer")
            } else {
                assert_step_compatible(&layer.class_latents, "Layer");
                Vec::new()
            };
            let emb = layer.class_latents_emb.as_ref();
            let mut cache = slots[pos].take();
            let mut next: Vec<Tensor<2>> = Vec::with_capacity(stream.len());
            for token in stream {
                // Splice this layer's class latents that fall before this token.
                if let Some(cursor) = layer_cursor.as_deref_mut() {
                    while let Some(i) = positions.iter().position(|&p| p == *cursor) {
                        let row = emb.unwrap().val().narrow(0, i, 1).expand([batch, d_model]);
                        let (out, c) = layer.step(row, cache, rs, None);
                        next.push(out);
                        cache = Some(c);
                        *cursor += 1;
                    }
                }
                let (out, c) = layer.step(token, cache, rs, None);
                next.push(out);
                cache = Some(c);
                if let Some(cursor) = layer_cursor.as_deref_mut() {
                    *cursor += 1;
                }
            }
            slots[pos] = cache;
            stream = next;
        }

        // The user token entered last, so its fully-propagated output is last.
        let out = stream.pop().expect("the user token is always emitted");
        (out, M::Caches::from_slots(slots))
    }
}

/// Plain (non-serde) factory for [`Layers`]. The serializable surface is the
/// concrete `MambaLatentNetConfig` enum; this is just the generic builder it
/// delegates to.
pub struct LayersBuilder<C> {
    /// Number of real (weight-bearing) layers.
    pub n_real_layers: usize,
    /// Optional virtual-layer scheduling.
    pub n_virtual_layers: Option<(usize, Schedule)>,
    /// Shared block config.
    pub mamba_block: C,
    /// Zero the first virtual layer's residual.
    pub ignore_first_residual: bool,
    /// Zero the last virtual layer's residual.
    pub ignore_last_residual: bool,
    /// Stack-level class latents (spliced once before the first virtual layer).
    pub class_latents: Vec<ClassLatent>,
}

impl<C: MambaBlockConfig> LayersBuilder<C> {
    /// Builder with no virtual scheduling, no class latents, residuals enabled.
    pub fn new(n_real_layers: usize, mamba_block: C) -> Self {
        Self {
            n_real_layers,
            n_virtual_layers: None,
            mamba_block,
            ignore_first_residual: false,
            ignore_last_residual: false,
            class_latents: Vec::new(),
        }
    }

    /// Set the optional virtual-layer scheduling.
    pub fn with_n_virtual_layers(mut self, n: Option<(usize, Schedule)>) -> Self {
        self.n_virtual_layers = n;
        self
    }

    /// Set the stack-level class latents.
    #[cfg(test)]
    pub fn with_class_latents(mut self, class_latents: Vec<ClassLatent>) -> Self {
        self.class_latents = class_latents;
        self
    }

    /// Allocate and initialise the stack on `device`.
    pub fn init(&self, device: &Device) -> Layers<C::Block> {
        let d_model = self.mamba_block.d_model();
        let real_layers = (0..self.n_real_layers)
            .map(|_| Layer {
                norm: RmsNormConfig::new(d_model).init(device),
                mamba_block: self.mamba_block.init_block(device),
                class_latents: Vec::new(),
                class_latents_emb: None,
            })
            .collect();
        Layers {
            n_real_layers: self.n_real_layers,
            n_virtual_layers: self.n_virtual_layers.clone(),
            real_layers,
            ignore_first_residual: self.ignore_first_residual,
            ignore_last_residual: self.ignore_last_residual,
            class_latents_emb: init_class_emb(self.class_latents.len(), d_model, device),
            class_latents: self.class_latents.clone(),
        }
    }
}
