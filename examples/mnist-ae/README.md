# MNIST Autoencoder

A symmetric, fully **bidirectional** ViT/MAE-style **patch** autoencoder over
MNIST. It uses the same Mamba-3 stack as
[`mnist-class`](../mnist-class/README.md). But it cuts the 28×28 image into
`patch×patch` tiles instead of a sequence of 784 single pixels. For example, 4×4
tiles give a sequence of 49 tokens of 16 pixels. The SSD scan routes short,
content-rich tokens much more easily. The much shorter sequence also frees the
VRAM that a single-pixel sequence spends on length.

This example is a work in progress. Its sizes and hyperparameters are
placeholders until a parameter search.

- **Encoder**: `patchify → in_proj (patch² → d_model) → +patch_pos → bidirectional
  Mamba-3 stack → mean-pool over patches → Linear (d_model → n_latent)`. The
  latent `z` is the configurable bottleneck. A `Middle` class-latent readout is
  an optional alternative to the mean-pool.
- **Decoder (generator)**: reconstructs every patch in **one parallel pass that
  reads only from `z`**. Each output position is a *learned positional query*,
  **FiLM-modulated by `z`** (`(1 + scale(z))·query + shift(z)`). No ground-truth
  pixel goes in, so all reconstruction information must go through `z`. The bidi
  decoder then refines the queries. `dec_out → unpatchify` puts the patches back
  on the 28×28 canvas as pixel logits.

```text
img[b,28,28,1] ─patchify→ [b,np,p²] ─in_proj+pos→ [b,np,d] ─enc(bidi)→ mean → [b,d] ─enc_to_z→ z[b,n_latent]
dec_in[b,np,d] = (1+scale(z))·pos + shift(z)  ─dec(bidi)→ [b,np,d] ─dec_out→ [b,np,p²] ─unpatchify→ logits[b,784]
```

The block uses the `Quaternion4D` rotation. The loss is a binary cross-entropy
reconstruction loss: the pixels are Bernoulli probabilities, and the model emits
raw logits.

## Tuning knobs

If the loss stalls, change one knob at a time. Each is one line in `model.rs`:

- `patch`: `7` ⇒ 16 tokens, `4` ⇒ 49 tokens (finer detail).
- `cond` (config, default `DecoderCond::Film`): `Add` selects the weaker
  additive-broadcast conditioning.
- `enc_class_latents` (config, default empty ⇒ mean-pool): add
  `ClassLatent::Middle` for a learned pooled readout.
- `--latents N` (default 16): the bottleneck width (try 128 for a sharper
  reconstruction).
- `d_model`, the real layer counts and `n_virtual_layers` in `model_config()`.
  `num_epochs` is in `main.rs`.

Training writes original-vs-reconstruction PNGs into
`<artifacts>/epoch-{e}-batch-{b}/` at every small validation check (every 100
mini-batches). Inference writes them into `<artifacts>/inference/`.

## Usage

`-- --latents N` (default 16) sets the latent width of a fresh model config. On
reload, the saved config wins.

```bash
# debug check in flex (fp32)
cargo check --example mnist-ae

# train + reconstruct on CUDA (long-running), 16-latent bottleneck
cargo run --release --example mnist-ae --features "backend-cuda,fusion" -- --training --inference
```

Inference prints a few test digits as ASCII art and writes PNGs (the original
next to the reconstruction).

- See `burn-mamba/Cargo.toml` for other backends/features.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

On CUDA, each validation pass captures its forward in one graph at its first
batch (burn-stack's `CapturedStep`), as in `mnist-class`. The reconstruction PNGs
run eagerly. The loss is the same in both modes. The graph holds its own memory.
`--no-graph` runs the forward eagerly. Under `--sgd` (plain SGD instead of AdamW,
see `mnist-class`), the training step replays in the same way: the forward, the
backward and the update, captured at the first batch.
