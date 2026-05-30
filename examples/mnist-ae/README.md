# MNIST Autoencoder

A symmetric, fully **bidirectional** Mamba-3 autoencoder over MNIST read as a
length-784 sequence of single-pixel tokens (same data handling as
[`mnist-class`](../mnist-class/README.md)).

- **Encoder**: `in_proj (1 → d_model)` → bidirectional Mamba-3 stack → mean-pool
  over the sequence → `Linear (d_model → n_latent)`. The latent `z` is the
  configurable bottleneck.
- **Decoder (generator)**: reconstructs the whole image in **one parallel pass,
  reading only from `z`**. At every output position the decoder input is a
  *learned positional query* plus the *broadcast latent* — no ground-truth pixel
  is ever fed in, so all reconstruction information must flow through `z`. This
  is the MAE-style decoder: bidirectional, positional-query, latent-conditioned.

```text
img[b,784,1] ─enc_in_proj→ [b,784,d] ─enc(bidi)→ mean_t → [b,d] ─enc_to_z→ z[b,n_latent]
dec_in = pos_emb[b,784,d] + z_to_dec(z)[b,1,d]  ─dec(bidi)→ [b,784,d] ─dec_out→ logits[b,784,1]
```

Pure real layers, `Complex2D` rotation, binary cross-entropy reconstruction loss
(pixels treated as Bernoulli probabilities; the model emits raw logits).

Refinement across depth is currently **implicit** (the residual stack sharpens
the representation; pixels materialise only at the final projection). A planned
follow-up adds explicit per-layer deep supervision, and a VAE variant for
sampling novel digits from the latent prior.

## Usage

The latent width is chosen with `-- --latents N` (default 32) and baked into a
fresh model config (a persisted config wins on reload).

```bash
# debug check in flex (fp32)
cargo check --example mnist-ae --features "backend-flex"

# train + reconstruct on CUDA (long-running), 16-latent bottleneck
cargo run --release --example mnist-ae --features "backend-cuda" -- --training --inference -- --latents 16
```

Inference prints a few test digits as ASCII art, original beside reconstruction.

- See `burn-mamba/Cargo.toml` for other backends/features.
- See `burn-mamba/examples/README.md` for the CLI usage overview.
