# MNIST Classification

The dataset is mostly based on [burn-dataset/vision/mnist](https://github.com/tracel-ai/burn/blob/fa4f9845a6b2279cd8de68bf7ca5a7eb76dec96d/crates/burn-dataset/src/vision/mnist.rs) and [book/data](https://burn.dev/books/burn/basic-workflow/data.html#data). It is mnist as flat (sequential) pixels, with sequence length of 28 * 28 = 784. The model reads the pixel sequence and predicts the classification label at the last input.

Inference samples a few test digits and, for each, prints the digit as ASCII art beside a text bar chart of the 10 class probabilities, and writes a PNG (the digit next to its probability bars; the true label and prediction are in the file name). Training also dumps these prediction PNGs into `<artifacts>/epoch-{e}-batch-{b}/` at every small validation check.

## Usage

The dataset is first downloaded and stored in `${CACHEDIR}/burn-dataset/mnist/train/`. The files are the following:

- train-images-idx3-ubyte (9.45 MB)
- train-labels-idx1-ubyte (28.20 KB)
- tk10-images-idx3-ubyte (1.57 MB)
- tk10-labels-idx1-ubyte (4.44 KB)

Note: "CACHEDIR" per [`dirs::cache_dir`](https://docs.rs/dirs/6.0.0/dirs/fn.cache_dir.html).

##### Usage Example

```bash
# debug check in flex (fp32)
cargo check --example mnist-class

# training and running inference in wgpu (fp32)
# note: the following requires ~7GB vram during training by default
cargo run --release --example mnist-class --features "backend-wgpu" -- --training --inference
```

- See `burn-mamba/Cargo.toml` for other features or backend information.  
- See `burn-mamba/examples/README.md` for the CLI usage overview.

## Optimizer: AdamW vs. AdamW + Muon

This example carries one downstream flag, `--muon` (after the trailing `--`),
which puts the block's hidden weight matrices on
[Muon](https://kellerjordan.github.io/posts/muon/) instead of AdamW:

```bash
# baseline: AdamW on every parameter
cargo run --release --example mnist-class --features "backend-wgpu" -- --training -a /tmp/mc-adamw
# AdamW + Muon on the hidden matrices
cargo run --release --example mnist-class --features "backend-wgpu" -- --training -a /tmp/mc-muon -- --muon
```

Everything else is identical: Muon uses the same LR schedule and weight decay
(`AdjustLrFn::MatchRmsAdamW` sizes its orthogonalised update to AdamW's RMS), and
every parameter the plan does not claim keeps its AdamW state. The flag is
recorded in `<artifacts>/training_config.json`, so a resumed run keeps it — and a
persisted config wins over the flag on reload.

Which weights move (see `burn_mamba::optim`, and the `muon_plan()` on the model
config): the block `out_proj`, and the Muon-owned segments of the fused
`in_proj` — `z`, `x`, `B`, `C` and the rotation channels. The per-head Δ/`A`/`λ`
channels of the same tensor stay on AdamW, as do every 1-D/3-D parameter, the
network's own `in_proj`/`out_proj`, and any class-token table. For this config
that is **~91% of the parameters** on Muon.

The fused `in_proj` is **split per sub-projection before Muon sees it** — the
model keeps its single fused GEMM, but the optimizer orthogonalises each
sub-matrix on its own, as if they had been separate `Linear`s.
