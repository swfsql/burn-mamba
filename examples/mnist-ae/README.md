# MNIST Autoencoder

A symmetric, fully **bidirectional** ViT/MAE-style **patch** autoencoder over
MNIST — the same Mamba-3 stack as [`mnist-class`](../mnist-class/README.md), but
the 28×28 image is cut into `patch×patch` tiles (the example uses 4×4 ⇒ a
length-49 sequence of 16-pixel tokens) instead of a 784-long single-pixel
sequence. Short, content-rich tokens are far easier for the SSD scan to route,
and the much shorter sequence frees the VRAM the single-pixel design spent on
length.

- **Encoder**: `patchify → in_proj (patch² → d_model) → +patch_pos → bidirectional
  Mamba-3 stack → mean-pool over patches → Linear (d_model → n_latent)`. The
  latent `z` is the configurable bottleneck. (A `Middle` class-latent readout is
  available as an optional alternative to mean-pooling.)
- **Decoder (generator)**: reconstructs every patch in **one parallel pass,
  reading only from `z`**. Each output position is a *learned positional query*
  **FiLM-modulated by `z`** (`(1 + scale(z))·query + shift(z)`) — no ground-truth
  pixel is ever fed in, so all reconstruction information must flow through `z`.
  The bidi decoder then refines, and `dec_out → unpatchify` lays the patches back
  onto the 28×28 canvas as pixel logits.

```text
img[b,28,28,1] ─patchify→ [b,np,p²] ─in_proj+pos→ [b,np,d] ─enc(bidi)→ mean → [b,d] ─enc_to_z→ z[b,n_latent]
dec_in[b,np,d] = (1+scale(z))·pos + shift(z)  ─dec(bidi)→ [b,np,d] ─dec_out→ [b,np,p²] ─unpatchify→ logits[b,784]
```

`Complex2D` rotation, binary cross-entropy reconstruction loss (pixels treated as
Bernoulli probabilities; the model emits raw logits).

## Tuning knobs

The example is built to be bisected if the loss stalls (each is a one-liner in
`model.rs`):

- `patch` (`model_config` uses `4`): `7` ⇒ 16 tokens; `4` ⇒ 49 tokens (finer detail).
- `cond` (config, default `DecoderCond::Film`): `Add` falls back to the weaker
  additive-broadcast conditioning.
- `enc_class_latents` (config, default empty ⇒ mean-pool): add `ClassLatent::Middle`
  for a learned pooled readout instead.
- `--latents N` (default 128): the bottleneck width (try 256 for sharper recon).
- `d_model` / layer counts (`4, 4` → `6, 6`) in `model_config()`; `num_epochs` in
  `main.rs`.

Training writes original-vs-reconstruction PNGs into
`<artifacts>/epoch-{e}-batch-{b}/` at every small validation check (every 100
mini-batches), and inference writes them into `<artifacts>/inference/`.

## Usage

The latent width is chosen with `-- --latents N` (default 128) and baked into a
fresh model config (a persisted config wins on reload).

```bash
# debug check in flex (fp32)
cargo check --example mnist-ae --features "backend-flex"

# train + reconstruct on CUDA (long-running), 128-latent bottleneck
cargo run --release --example mnist-ae --features "backend-cuda,fusion" -- --training --inference
```

Inference prints a few test digits as ASCII art and writes PNGs to disk
(original beside reconstruction).

- See `burn-mamba/Cargo.toml` for other backends/features.
- See `burn-mamba/examples/README.md` for the CLI usage overview.
