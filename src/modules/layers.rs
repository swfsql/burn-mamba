use crate::modules::{Residuals, ResidualsConfig, RmsNormConfig};
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
    /// How residuals are threaded between layers (plain additive vs Multi-Gate).
    pub residuals: Residuals,
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

    /// Whether (virtual) layer `i` of `n` suppresses its residual — the first
    /// layer when `ignore_first_residual`, the last when `ignore_last_residual`.
    fn skip_residual(&self, i: usize, n: usize) -> bool {
        (self.ignore_first_residual && i == 0) || (self.ignore_last_residual && i + 1 == n)
    }

    /// Full-sequence pass through every (virtual) layer.
    ///
    /// [`Layer`] returns only `F_l = Block(RMSNorm(·))`; the residual is added
    /// here. With [`Residuals::Standard`] each layer adds the input skip (unless
    /// suppressed). With [`Residuals::MultiGate`] the skip is dropped and
    /// `n_stream` parallel streams — seeded from `x` — carry the residual: each
    /// layer reads their attention-pooled aggregate as input and its output is
    /// gated back into every stream (see [`MultiGate`]).
    ///
    /// `ignore_first/last_residual` apply to **both** paths: skipping the first
    /// restarts the residual carry from the first layer's output (the input is
    /// read but not carried); skipping the last makes the stack output the last
    /// layer's transform `F_l` alone (no input-dependent carry). Class latents
    /// apply to the Standard path only (MultiGate forbids them, panicking if any
    /// are present).
    ///
    /// [`MultiGate`]: crate::modules::MultiGate
    /// [`Layer`]: crate::modules::Layer
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

        // MultiGate keeps `n_stream` parallel streams (seeded from the input);
        // Standard threads the single tensor `x` directly (streams stays `None`).
        let mut streams = self.multi_gate_streams_seed(&x);

        for i in 0..n {
            let real = self.real_idx(i);
            let layer = &self.real_layers[real];
            let cache = slots[i].take().unwrap();
            let first = self.ignore_first_residual && i == 0;
            let last = self.ignore_last_residual && i + 1 == n;
            match &self.residuals {
                Residuals::Standard(_noop) => {
                    // Splice this layer's class latents, then add the residual
                    // (the lengthened input) here — unless suppressed, in which
                    // case the input is moved straight in (no clone, no add).
                    let x_l = layer.insert_latents(x);
                    let (out, c_) = if first || last {
                        layer.forward(x_l, Some(cache), ssd_path.clone())
                    } else {
                        let (out, c_) = layer.forward(x_l.clone(), Some(cache), ssd_path.clone());
                        (out + x_l, c_)
                    };
                    x = out;
                    slots[i] = Some(c_);
                }
                Residuals::MultiGate(mg) => {
                    assert!(
                        layer.class_latents_emb.is_none(),
                        "MultiGate residuals do not support per-layer class latents"
                    );
                    let (out, c_) = layer.forward(x, Some(cache), ssd_path.clone());
                    slots[i] = Some(c_);
                    let s = streams.take().unwrap();
                    // A skipped residual here is equivalent to forcing the MGR
                    // mixer gate β ≡ 1 (`new_streams = out`): the carried streams
                    // are dropped, and the aggregator over the resulting identical
                    // streams collapses to `F_l`. Both branches shortcut that.
                    if last {
                        // Output depends purely on the last layer's transform.
                        x = out;
                        streams = Some(s);
                    } else if first {
                        // Drop the input seed: restart the streams from `F_0`.
                        let [b, seq, d] = out.dims();
                        streams = Some(out.clone().unsqueeze_dim::<4>(2).expand([
                            b,
                            seq,
                            mg.n_stream,
                            d,
                        ]));
                        x = out;
                    } else {
                        let idx = mg.module_index(i, real);
                        let (new_h, new_streams) = mg.layers[idx].forward(out, s);
                        x = new_h;
                        streams = Some(new_streams);
                    }
                }
            }
        }
        (x, M::Caches::from_slots(slots))
    }

    /// Seed the MultiGate streams from a full-sequence input — `n_stream` copies
    /// of `x` as `[batch, sequence, n_stream, d_model]` — or `None` for the
    /// Standard path. Panics if MultiGate is paired with stack-level class latents.
    fn multi_gate_streams_seed(&self, x: &Tensor<3>) -> Option<Tensor<4>> {
        let Residuals::MultiGate(mg) = &self.residuals else {
            return None;
        };
        assert!(
            self.class_latents_emb.is_none(),
            "MultiGate residuals do not support stack-level class latents"
        );
        let [batch, sequence, d_model] = x.dims();
        Some(
            x.clone()
                .unsqueeze_dim::<4>(2)
                .expand([batch, sequence, mg.n_stream, d_model]),
        )
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
        if let Residuals::MultiGate(mg) = &self.residuals {
            return self.step_multi_gate(x, caches, mg);
        }
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
            let skip = self.skip_residual(pos, n);
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
            // One token through the layer, adding its residual here unless
            // suppressed (then the token is moved straight in — no clone/add).
            let run = |token: Tensor<2>, cache: Option<M::Cache>| {
                if skip {
                    layer.step(token, cache, None)
                } else {
                    let (out, c) = layer.step(token.clone(), cache, None);
                    (out + token, c)
                }
            };
            for token in stream {
                // Splice this layer's class latents that fall before this token.
                if let Some(cursor) = layer_cursor.as_deref_mut() {
                    while let Some(i) = positions.iter().position(|&p| p == *cursor) {
                        let row = emb.unwrap().val().narrow(0, i, 1).expand([batch, d_model]);
                        let (out, c) = run(row, cache);
                        next.push(out);
                        cache = Some(c);
                        *cursor += 1;
                    }
                }
                let (out, c) = run(token, cache);
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

    /// Stationary fixed point of the whole stack under a constant token, with
    /// **no caches** involved: under a constant input each layer's output
    /// converges (its decay damps the transient, and the readout phase of the
    /// rotation cancels), so the downstream layer's input converges too and
    /// the limit composes **exactly**, layer by layer — even though every
    /// layer's SSM state keeps rotating forever. Residual handling mirrors
    /// [`Self::step`]; cursorless (class latents are not injected).
    pub fn step_infinite(&self, x: Tensor<2>) -> Tensor<2> {
        if let Residuals::MultiGate(mg) = &self.residuals {
            return self.step_infinite_multi_gate(x, mg);
        }
        assert_step_compatible(&self.class_latents, "Layers");
        let n = self.n_virtual_count();
        let mut h = x;
        for i in 0..n {
            let layer = &self.real_layers[self.real_idx(i)];
            h = if self.skip_residual(i, n) {
                layer.step_infinite(h)
            } else {
                layer.step_infinite(h.clone()) + h
            };
        }
        h
    }

    /// Multi-Gate counterpart of [`Self::step_infinite`]. The streams are a
    /// per-token depth construct (as in [`Self::step_multi_gate`]), so applying
    /// the mixers to the layers' fixed-point outputs *is* the fixed point of
    /// the whole stack.
    fn step_infinite_multi_gate(&self, x: Tensor<2>, mg: &crate::modules::MultiGate) -> Tensor<2> {
        assert_step_compatible(&self.class_latents, "Layers");
        let [batch, d_model] = x.dims();
        let n = self.n_virtual_count();
        let mut streams = x
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([batch, mg.n_stream, d_model]);
        let mut h = x;
        for i in 0..n {
            let real = self.real_idx(i);
            let layer = &self.real_layers[real];
            assert_step_compatible(&layer.class_latents, "Layer");
            let out = layer.step_infinite(h);
            if self.ignore_last_residual && i + 1 == n {
                h = out;
            } else if self.ignore_first_residual && i == 0 {
                let [b, d] = out.dims();
                streams = out
                    .clone()
                    .unsqueeze_dim::<3>(1)
                    .expand([b, mg.n_stream, d]);
                h = out;
            } else {
                let idx = mg.module_index(i, real);
                let (new_h, new_streams) = mg.layers[idx].step(out, streams);
                h = new_h;
                streams = new_streams;
            }
        }
        h
    }

    /// **Approximate** jump of `n_steps` consecutive constant-token
    /// [`Self::step`] calls (cursorless), in O(1) per layer.
    ///
    /// Each (virtual) layer jumps in closed form with its input held constant
    /// at the *previous layer's step-`n` output*. The first layer's jump is
    /// exact; deeper layers ignore the upstream transient, an error that
    /// decays geometrically in `n_steps` (the `n → ∞` limit is exact — see
    /// [`Self::step_infinite`]). `n_steps = 1` is exactly one `step`.
    pub fn step_n_approx(
        &self,
        x: Tensor<2>,
        n_steps: usize,
        caches: Option<M::Caches>,
    ) -> (Tensor<2>, M::Caches) {
        if let Residuals::MultiGate(mg) = &self.residuals {
            return self.step_n_approx_multi_gate(x, n_steps, caches, mg);
        }
        assert_step_compatible(&self.class_latents, "Layers");
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_2d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");
        let mut slots = caches.into_slots();

        let mut h = x;
        for i in 0..n {
            let layer = &self.real_layers[self.real_idx(i)];
            let cache = slots[i].take();
            let (out, c) = if self.skip_residual(i, n) {
                layer.step_n_approx(h, n_steps, cache)
            } else {
                let (out, c) = layer.step_n_approx(h.clone(), n_steps, cache);
                (out + h, c)
            };
            h = out;
            slots[i] = Some(c);
        }
        (h, M::Caches::from_slots(slots))
    }

    /// Multi-Gate counterpart of [`Self::step_n_approx`] (mirrors
    /// [`Self::step_multi_gate`]; the mixers see each layer's step-`n` output).
    fn step_n_approx_multi_gate(
        &self,
        x: Tensor<2>,
        n_steps: usize,
        caches: Option<M::Caches>,
        mg: &crate::modules::MultiGate,
    ) -> (Tensor<2>, M::Caches) {
        assert_step_compatible(&self.class_latents, "Layers");
        let [batch, d_model] = x.dims();
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_2d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");

        let mut slots = caches.into_slots();
        let mut streams = x
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([batch, mg.n_stream, d_model]);
        let mut h = x;
        for i in 0..n {
            let real = self.real_idx(i);
            let layer = &self.real_layers[real];
            let cache = slots[i].take();
            let (out, c_) = layer.step_n_approx(h, n_steps, cache);
            slots[i] = Some(c_);
            if self.ignore_last_residual && i + 1 == n {
                h = out;
            } else if self.ignore_first_residual && i == 0 {
                let [b, d] = out.dims();
                streams = out
                    .clone()
                    .unsqueeze_dim::<3>(1)
                    .expand([b, mg.n_stream, d]);
                h = out;
            } else {
                let idx = mg.module_index(i, real);
                let (new_h, new_streams) = mg.layers[idx].step(out, streams);
                h = new_h;
                streams = new_streams;
            }
        }
        (h, M::Caches::from_slots(slots))
    }

    /// Single-token Multi-Gate Residual step — the recurrent counterpart of
    /// [`Self::forward_multi_gate`]. The streams are a per-token *depth*
    /// construct (rebuilt from `x` each step, never carried between tokens), so
    /// no extra state crosses steps and `forward`/`step` agree. Class latents are
    /// unsupported.
    fn step_multi_gate(
        &self,
        x: Tensor<2>,
        caches: Option<M::Caches>,
        mg: &crate::modules::MultiGate,
    ) -> (Tensor<2>, M::Caches) {
        assert_step_compatible(&self.class_latents, "Layers");
        let [batch, d_model] = x.dims();
        let n = self.n_virtual_count();
        let caches =
            caches.unwrap_or_else(|| self.real_layers[0].mamba_block.zero_caches_2d(&x, n));
        assert_eq!(caches.slot_count(), n, "one cache per virtual layer");

        let mut slots = caches.into_slots();
        let mut streams = x
            .clone()
            .unsqueeze_dim::<3>(1)
            .expand([batch, mg.n_stream, d_model]);
        let mut h = x;
        for i in 0..n {
            let real = self.real_idx(i);
            let layer = &self.real_layers[real];
            assert_step_compatible(&layer.class_latents, "Layer");
            let cache = slots[i].take();
            let (out, c_) = layer.step(h, cache, None);
            slots[i] = Some(c_);
            // As in `forward`, a skipped residual is β ≡ 1 in the mixer
            // (`new_streams = out`), the aggregator then collapsing to `F_l`.
            if self.ignore_last_residual && i + 1 == n {
                // Output depends purely on the last layer's transform.
                h = out;
            } else if self.ignore_first_residual && i == 0 {
                // Drop the input seed: restart the streams from `F_0`.
                let [b, d] = out.dims();
                streams = out
                    .clone()
                    .unsqueeze_dim::<3>(1)
                    .expand([b, mg.n_stream, d]);
                h = out;
            } else {
                let idx = mg.module_index(i, real);
                let (new_h, new_streams) = mg.layers[idx].step(out, streams);
                h = new_h;
                streams = new_streams;
            }
        }
        (h, M::Caches::from_slots(slots))
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
    /// Inter-layer residual scheme (defaults to plain additive).
    pub residuals: ResidualsConfig,
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
            residuals: ResidualsConfig::Standard,
        }
    }

    /// Set the optional virtual-layer scheduling.
    pub fn with_n_virtual_layers(mut self, n: Option<(usize, Schedule)>) -> Self {
        self.n_virtual_layers = n;
        self
    }

    /// Set the inter-layer residual scheme (plain additive vs Multi-Gate).
    pub fn with_residuals(mut self, residuals: ResidualsConfig) -> Self {
        self.residuals = residuals;
        self
    }

    /// Suppress the first virtual layer's residual (see [`Layers`]).
    pub fn with_ignore_first_residual(mut self, ignore: bool) -> Self {
        self.ignore_first_residual = ignore;
        self
    }

    /// Suppress the last virtual layer's residual (see [`Layers`]).
    pub fn with_ignore_last_residual(mut self, ignore: bool) -> Self {
        self.ignore_last_residual = ignore;
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
        let n_virtual = self
            .n_virtual_layers
            .as_ref()
            .map(|(l, _)| *l)
            .unwrap_or(self.n_real_layers);
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
            residuals: self
                .residuals
                .init(d_model, self.n_real_layers, n_virtual, device),
            class_latents_emb: init_class_emb(self.class_latents.len(), d_model, device),
            class_latents: self.class_latents.clone(),
        }
    }
}
