# MNIST Classification

The dataset is mostly based on [burn-dataset/vision/mnist](https://github.com/tracel-ai/burn/blob/fa4f9845a6b2279cd8de68bf7ca5a7eb76dec96d/crates/burn-dataset/src/vision/mnist.rs) and [book/data](https://burn.dev/books/burn/basic-workflow/data.html#data). It is mnist as flat (sequential) pixels, with sequence length of 28 * 28 = 784. The model reads the pixel sequence and predicts the classification label at the last input.

Inference samples a few test digits and, for each, prints the digit as ASCII art beside a text bar chart of the 10 class probabilities, and writes a PNG (the digit next to its probability bars; the true label and prediction are in the file name). Training also dumps these prediction PNGs into `<artifacts>/epoch-{e}-batch-{b}/` at every small validation check.

## Model

`model.rs` configures a deliberately tiny Mamba-3 network (954 parameters): one real layer applied as 4 virtual layers, a quaternion transition rotation, one B/C group per head and multi-gate residuals, each knob commented where it is set. With the training config in `main.rs` (batch 16, fp32, one cosine LR schedule over 4 epochs) it reaches ~90% validation accuracy.

## Usage

The dataset is first downloaded and stored in `${HOME}/.cache/burn-dataset/mnist/{train,test}/`. The files are the following:

- train-images-idx3-ubyte (9.45 MB)
- train-labels-idx1-ubyte (28.20 KB)
- t10k-images-idx3-ubyte (1.57 MB)
- t10k-labels-idx1-ubyte (4.44 KB)

Note: "HOME" per [`dirs::home_dir`](https://docs.rs/dirs/6.0.0/dirs/fn.home_dir.html).

##### Usage Example

```bash
# debug check in flex (fp32)
cargo check --example mnist-class

# training and running inference in wgpu (fp32)
cargo run --release --example mnist-class --features "backend-wgpu" -- --training --inference
```

- See `burn-mamba/Cargo.toml` for other features or backend information.  
- See `burn-mamba/examples/README.md` for the CLI usage overview.

On CUDA each validation pass replays its forward from one graph captured at the
pass's first batch (burn-stack's `CapturedStep`) instead of launching it anew — a
model this small is bound by the host enqueueing its launches. The metrics are the
same either way; the graph pins memory of its own, and `MNIST_GRAPH=0` runs the
forward eagerly.

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

Which weights move (see `burn_stack::optim`, and the `muon_plan()` on the model
config): the block `out_proj`, and the Muon-owned segments of its in-projection
— `z`, `x`, `B`, `C` and the rotation channels. The per-head Δ/`A`/`λ` channels
stay on AdamW, as do every 1-D/3-D parameter, the network's own
`in_proj`/`out_proj`, and any class-token table.

The fused `in_proj` is **split per sub-projection before Muon sees it** — the
model keeps its single fused GEMM, but the optimizer orthogonalises each
sub-matrix on its own, as if they had been separate `Linear`s.
