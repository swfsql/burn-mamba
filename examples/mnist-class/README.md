# MNIST Classification

MNIST as flat (sequential) pixels: a sequence of 28 × 28 = 784 pixels. The model
reads the pixel sequence and predicts the class label at the last input. The
dataset code is mostly from
[burn-dataset/vision/mnist](https://github.com/tracel-ai/burn/blob/fa4f9845a6b2279cd8de68bf7ca5a7eb76dec96d/crates/burn-dataset/src/vision/mnist.rs)
and [book/data](https://burn.dev/books/burn/basic-workflow/data.html#data).

Inference takes a few test digits. For each digit, it prints the digit as ASCII
art next to a text bar chart of the 10 class probabilities. It also writes a PNG
of the digit next to its probability bars. The file name holds the true label and
the prediction. Training writes the same PNGs into
`<artifacts>/epoch-{e}-batch-{b}/` at every small validation check.

## Model

`model.rs` configures a very small Mamba-3 network (954 parameters):

- one real layer, applied as 4 virtual layers,
- a quaternion transition rotation,
- one B/C group per head,
- multi-gate residuals.

A comment at each knob explains it. With the training config of `main.rs`
(batch 16, fp32, one cosine LR schedule over 4 epochs), it gets ~90% validation
accuracy.

## Usage

The first run downloads the dataset into
`${HOME}/.cache/burn-dataset/mnist/{train,test}/`. The files are:

- train-images-idx3-ubyte (9.45 MB)
- train-labels-idx1-ubyte (28.20 KB)
- t10k-images-idx3-ubyte (1.57 MB)
- t10k-labels-idx1-ubyte (4.44 KB)

"HOME" is as [`dirs::home_dir`](https://docs.rs/dirs/6.0.0/dirs/fn.home_dir.html)
gives it.

```bash
# debug check in flex (fp32)
cargo check --example mnist-class

# training and running inference in wgpu (fp32)
cargo run --release --example mnist-class --features "backend-wgpu" -- --training --inference
```

- See `burn-mamba/Cargo.toml` for other features or backend information.
- See `burn-mamba/examples/README.md` for the CLI usage overview.

On CUDA, each validation pass captures its forward in one graph at its first
batch (burn-stack's `CapturedStep`). The other batches replay that graph and do
not launch the forward again. A model this small is bound by the host that
enqueues its launches. The metrics are the same in both modes. The graph holds
its own memory. `--no-graph` runs the forward eagerly. Under SGD (below), the
training step replays in the same way.

## Optimizer: AdamW vs. AdamW + Muon vs. SGD

The shared optimizer flags select the optimizer (AdamW by default). `--muon` puts
the hidden weight matrices of the block on
[Muon](https://kellerjordan.github.io/posts/muon/) instead of AdamW:

```bash
# baseline: AdamW on every parameter
cargo run --release --example mnist-class --features "backend-wgpu" -- --training -a /tmp/mc-adamw
# AdamW + Muon on the hidden matrices
cargo run --release --example mnist-class --features "backend-wgpu" -- --training -a /tmp/mc-muon --muon
```

Everything else is the same:

- Muon uses the same LR schedule and weight decay. `AdjustLrFn::MatchRmsAdamW`
  scales its orthogonalised update to the RMS of AdamW.
- Every parameter that the plan does not claim keeps its AdamW state.
- `<artifacts>/training_config.json` records the choice, so a resumed run keeps
  it. A flag that names a different optimizer replaces it, unless that would
  ignore the saved optimizer state (see the CLI overview).

The weights that Muon trains (see `burn_stack::optim`, and `muon_plan()` on the
model config):

- the block `out_proj`,
- the Muon segments of its in-projection: `z`, `x`, `B`, `C` and the rotation
  channels.

The per-head Δ/`A`/`λ` channels stay on AdamW. So do all 1-D/3-D parameters, the
`in_proj`/`out_proj` of the network, and any class-token table.

The fused `in_proj` is **split per sub-projection before Muon sees it**. The
model keeps its single fused GEMM, but the optimizer orthogonalises each
sub-matrix separately, as if each were a separate `Linear`.

`--sgd` trains every parameter with plain SGD (`burn_stack::optim::SgdConfig`:
gradient clipping at 1.0, no momentum, no weight decay). It uses the same cosine
schedule, with a peak of 5e-2 instead of the 9.6e-3 of AdamW.

SGD is the one optimizer whose training step replays from a captured graph. The
first batch records the forward, the backward and the update once. Every batch of
that shape then replays them. The learning rate is an input, so the schedule
continues to move. The other optimizers of Burn put their host-side state into a
graph ([tracel-ai/burn#5779](https://github.com/tracel-ai/burn/issues/5779)), so
they train eagerly. Muon + SGD (`--muon --sgd`, at the SGD rate) also trains
eagerly, because Muon keeps momentum. A replayed run trains exactly as the eager
one (`--no-graph`). The graph holds the memory of one training step for as long
as it exists.

```bash
cargo run --release --example mnist-class --features "backend-cuda" -- --training -a /tmp/mc-sgd --sgd
```
