# Mamba Examples

#### List of Examples

- **`reset-*`** ([`reset/README.md`](reset/README.md), six examples): a ladder
  on one `+`/`-`/`R`-shaped stream. Each rung is the smallest task that its
  block is *needed* for, and that the rung below cannot solve:
  - `reset-majority`: the selective decay alone (`Real1D`).
  - `reset-rotor`: the complex transition (`Z₃`).
  - `reset-spinor`: the non-abelian quaternion transition (`Q₈`).
  - `reset-swap`: the full two-sided `SO(4)` (`S₃`).
  - `spinor-product`: the *other* dial, `Mamba3Config::micro_steps`
    (MambaProduct's `u`). It reads the `Q₈` stream of `reset-spinor` **two
    symbols per token**. The rotation generator of one step is affine in the
    token, so the two generators of a pair can only *add* where the group
    multiplies. `u = 2` makes the token two recurrence steps, and its
    transition the product. A second layer (`-- --layers 2 --micro-steps 1`)
    is the contrast: it turns a second state, so it has to *learn* the product.
  - `reset-quintic`: the question of `reset-swap` over five items. `A₅` (the
    rotations of the icosahedron, the smallest non-solvable group) is one
    `Rotor4D` block at the width of `reset-swap`. `S₅` is the transition group
    of no single Mamba-3 layer at any size (its swaps would have to be
    reflections), and it takes two.

  Each rung has a hand-built exact solution and the ablations that separate
  it from the rung below.
- **`tally-*`** ([`tally/README.md`](tally/README.md), three examples): a ladder
  on the *other* axis. The plant stays at the floor of `reset-majority` (a
  scalar real state). What climbs is the **positive system** beside it: the
  per-head scalar recurrence of `src/mamba3/positive/`, which reads only the
  input and sets the coefficients of the plant.
  - `tally-depth` (a counter with a floor) needs a `Tropical::MaxPlus`
    register for its **growth**: a decay above one, which the plant's
    `α = exp(Δ·A) ≤ 1` cannot give.
  - `tally-record` (a running maximum over twelve values) needs the same
    register for its **range**. The one route without a register, a sum of
    `exp(S·v)`, needs `e^{45.8}` inside a single channel.
  - `tally-drift` (age evidence by how much of it there is) needs the
    `Gain::KalmanProjectedNoise` filter. No projected decay matches its
    hyperbolic discount.

  Each rung has a hand-built exact solution, ablation sweeps, and a trained
  ablation at equal budget. That last row sets the alphabet of the second
  rung: at six values, a trained stock block solves it.
- **`mnist-class`**: a small Mamba-3 model that classifies MNIST digits.
- **`mnist-ae`** (work in progress): a symmetric bidirectional Mamba-3
  autoencoder over a sequence of MNIST patches. The decoder rebuilds the whole
  image in one parallel pass, and reads only a configurable latent
  (`-- --latents N`).
- **`tiny-stories`** (work in progress): a tiny character-level Mamba-3
  language model on the cleaned
  [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
  corpus, with a tied 48-character embedding at both ends. Its README covers
  the case-folded alphabet, the download, the class latents that mark the
  start of a story (and that `prime()` replays for seedless sampling), the
  prefill-`forward()` / decode-`step()` sampler, and the runs of windows that
  carry the state across a story.

#### Examples Structure

Each example has its own directory. The `reset-*` examples are one level
deeper, under `reset/`, with one shared README. The three `tally-*` examples
are under `tally/`, and they also share a `shared/` module (the dataset,
loops and hand-built helpers of the ladder). Cargo autodiscovers only
`examples/<name>/main.rs`, so `Cargo.toml` declares these nine as explicit
`[[example]]` targets.

An example usually has:

- `model.rs`: the model (it can also give the training requirements and the
  expected accuracy),
- `dataset.rs`: the dataset (if applicable),
- `training.rs`: the training procedure,
- `inference.rs`: the inference procedure (if applicable),
- `cli.rs`: its own command-line flags (empty when it has none),
- `main.rs`: the launch procedure.

`main.rs` parses the command line: first the shared flags (training and/or
inference, and more), then the flags of the example, after `--`. The examples
read no environment variables of their own (Burn itself reads `BURN_DEVICE`,
see below). Training usually validates every few batches. The README of each
example gives the training goal.

`common/mod.rs` holds the shared definitions, imported as an outside module by
each example. It is a thin shim. The CLI, the runtime device selection, the
training config, and both datasets with their epoch loops (sequential-MNIST
classification, character-level TinyStories language modelling) are in
**`burn_stack::examples`** (feature `examples-common`, dev-only), shared with
`burn-deltanet`. The `config → module` seam is
`burn_stack::modules::ModelConfigExt`, which the network configs of this crate
implement. `common/mod.rs` re-exports those under the `common::*` paths, and
adds:

- `ARTIFACT_PREFIX`, which must expand in the example crate,
- `parse_rotation`, the `--rotation` value of the rotating `reset-*` rungs (a
  flag that names a `burn-mamba` type, which `burn-stack` cannot do).

##### Model Definition

Most examples use the lib-generic `MambaLatentNet` (configured by
`MambaLatentNetConfig`, in `src/unified/network.rs`). It is a continuous-I/O
network: input and output projections (linear layers) around a generic
`Layers<M>` stack, where `M` is the SSM core (`Mamba1`/`Mamba2`/`Mamba3`). The
token-based example (`tiny-stories`) uses `MambaVocabNet` (embedding →
`Layers<M>` → LM head). `src/unified/network.rs` implements `ModelConfigExt`
(config enum → `Module`, plus the Muon plan) on those configs. Examples do not
define their own network types.

##### Optimizer

`burn_stack::examples::training` defines `OptimizerConfig { fallback, muon }`,
held by `TrainingConfig`:

- The fallback is AdamW or plain SGD. `muon = None` puts every parameter on
  it. Plain SGD is the one optimizer that a captured training step can replay:
  under `--sgd`, every example except `tiny-stories` runs through the
  `examples::trainer::Trainer` of burn-stack. It captures the forward, backward
  and update at the first batch, and masks the loss (not a gather), so the
  shapes never change.
- `muon` moves the hidden weight matrices to
  [Muon](https://kellerjordan.github.io/posts/muon/), driven by the
  `muon_plan()` of the model config (`ModelConfigExt::muon_plan`, backed by
  `burn_stack::optim`). Muon gets only rank-2 hidden matrices. Each fused
  projection (`in_proj`, `fc1`, …) is first split into its independent
  sub-projections, so the orthogonalisation is per linear map, not per
  allocation.

Every example takes the choice from the shared `--adamw` / `--sgd` / `--muon`
flags (below). Without them, `tiny-stories` trains Muon + AdamW and the other
examples train AdamW.

#### Backend Selection

Features select the backend, for example `backend-flex` (the default). See the
`[features]` section of `burn-mamba/Cargo.toml` for the backend list. You can
compile in several backends at once. `Device::default()` resolves which one to
use (the `BURN_DEVICE` environment variable can override it). Other "dev"
features select the float precision (default f32, or `dev-f16`) and whether
fusion and/or autotune are enabled.

#### Examples CLI

All examples use the CLI of `burn_stack::examples::cli`, re-exported as
`common::cli`: the flags that every example shares (below). The arguments
after a second `--` belong to the example, and its `cli.rs` parses them.
`-- --help` lists them.

##### Usage Example

```bash
# training the simplest example on flex (fp32) and running inference:
cargo run --example reset-majority --features "backend-flex" -- --training --inference

# assume /tmp/reset-majority-abcd-0 got created:
ARTIFACTS="/tmp/reset-majority-abcd-0"

# running only the inference from the trained model:
cargo run --example reset-majority --features "backend-flex" -- --inference --artifacts-path "$ARTIFACTS"

# a short run: stop after 600 mini-batches, however many epochs that spans
# (checkpoints and the end-of-epoch validation still happen before it stops)
cargo run --example reset-majority --features "backend-flex" -- --training --max-batches 600

# continue it for another 600: the LR schedule, the epoch and the position in it pick up
# where the checkpoint left them (the rest of that epoch drawn from a fresh shuffle)
cargo run --example reset-majority --features "backend-flex" -- --training --artifacts-path "$ARTIFACTS" --max-batches 600 --resume

# or stop after 10 minutes of training, wherever in the epoch that lands
cargo run --example reset-majority --features "backend-flex" -- --training --max-seconds 600

# Muon on the hidden matrices, plain SGD on the rest; and plain SGD alone, whose training
# step replays from a captured CUDA graph (--no-graph steps it eagerly)
cargo run --example mnist-class --features "backend-flex" -- --training --muon --sgd
cargo run --release --example mnist-class --features "backend-cuda" -- --training --sgd

# an example's own flags, after a second `--`, and their help
cargo run --example reset-spinor --features "backend-flex" -- --training -- --rotation complex
cargo run --example reset-spinor --features "backend-flex" -- -- --help

# the per-step and per-validation metrics of every run so far, one JSON object per line:
jq -c 'select(.event == "valid")' "$ARTIFACTS/metrics.jsonl"

# assume /some/path/ contains a different training config file, e.g. with a different LR schedule:
TCONFIG="/some/path/training_config.json"

# continue training from another training config
# warning: "$ARTIFACTS/training_config.json" gets overwritten by "$TCONFIG"
cargo run --example reset-majority --features "backend-flex" -- --training --artifacts-path "$ARTIFACTS" --training-config "$TCONFIG"
```

##### CLI Help Message

```txt
Burn Example

A command-line tool to train machine learning models and/or to run inference with them.
An artifacts directory keeps the models, the optimizers and the configurations.

USAGE:
    example-name [OPTIONS] [-- <EXTRA_ARGS>...]

Without --training or --inference, the program handles the configurations and then exits.

BEHAVIOR OVERVIEW
- The program manages two configurations: the training config and the model config.
- With --training-config or --model-config, the program loads that config from the given file and saves it to the artifacts directory (it overwrites the existing file).
- Without an explicit config file, the program loads the config from the artifacts directory. If that file is absent, the program creates a default config and saves it.
- The program reads and writes the model weights, the optimizer state and the configurations in the artifacts directory (--artifacts-path). Without --artifacts-path, the program creates a new temporary directory and prints its path.
- With --remove-artifacts and --training, the program deletes the model and optimizer files (and the saved progress) in the artifacts directory before training.
- The program loads the model and optimizer weights from the artifacts directory if they are present. Otherwise it creates new ones and saves them.
- --seed, --epochs, --batch-size and --max-lr replace the value in the training config (loaded or created) before the program saves the config. So later runs from the same artifacts directory inherit the value. --epochs also rescales the length of a cosine LR schedule by the same factor, so the schedule still spans the run. --batch-size rescales its length and warmup by the inverse factor (an epoch has that many fewer steps).
- --adamw, --sgd and --muon choose the optimizer. --muon puts the hidden weight matrices of the model on Muon, and the other flag (default --adamw) optimizes every other parameter. --adamw and --sgd are exclusive. Without these flags, a new training config gets the default optimizer of the example.
- In a loaded config, these flags replace the optimizer and keep the LR schedule (see --max-lr). If the new optimizer would ignore the optimizer state that the old optimizer saved, the program panics. (--remove-artifacts with --training removes that state.)
- Only plain SGD (--sgd alone) has a training step that replays from a captured graph.
- An example that supports it replays its fixed-shape passes (training steps under plain SGD, validation, decoding) from captured CUDA graphs. --no-graph runs every pass eagerly.
- The program saves the optimizer state together with the progress of the run: the LR-schedule step, the epoch, and the batch within it. A run that loads this state starts again at step 0 of epoch 1. With --resume, the run continues from the saved progress. The interrupted epoch then trains only its remaining batches, drawn from a new shuffle (the batch order of the dataloader workers cannot be replayed).
- Training makes a checkpoint at every epoch end and when it stops. --checkpoint-every adds a checkpoint every N optimizer steps. --valid-every adds a validation every N optimizer steps. --valid-batches caps the batches that a validation reads. Each example has its own defaults for these three flags.
- Every training step and every validation appends one JSON line to metrics.jsonl in the artifacts directory. Each run starts with a "start" line.
- With both --training and --inference, training runs first. Inference then uses the trained model.
- With --max-batches, training stops after that many mini-batches in total (counted across epochs), and makes a checkpoint as usual before it returns. One mini-batch is one optimizer step. For the character LM, this is one window of a run, not one dataloader item. --max-seconds stops training in the same way, when that much wall-clock time has passed since its first step.
- The program forwards all the arguments after -- unchanged to the flags of the example (-- --help lists them).

FLAGS:
    -h, --help                  Show this help message and exit

OPTIONS:
    -t, --training              Run training (creates or updates the model and the optimizer)
    -i, --inference             Run inference: after training (with both flags), or immediately (with this flag only)
    -r, --remove-artifacts      Delete the model and optimizer files (and the progress) in the artifacts directory
                                before training (no effect without --training)
    -c, --training-config <PATH>
                                Load the training config from this file (overrides the config in the artifacts directory)
    -m, --model-config <PATH>   Load the model config from this file (overrides the config in the artifacts directory)
    -b, --max-batches <N>       Stop training after N mini-batches in total (across epochs), for any configured
                                number of epochs. Unlimited when absent.
        --max-seconds <S>       Stop training when S seconds have passed since its first step. Unlimited when absent.
    -s, --seed <N>              Replace the RNG seed of the training config (model init, data shuffle, sampling)
        --epochs <N>            Replace the number of epochs of the training config (rescales a cosine LR schedule)
        --batch-size <N>        Replace the mini-batch size of the training config (rescales a cosine LR schedule)
        --max-lr <LR>           Replace the peak rate of the LR schedule (the only rate of a constant schedule)
        --adamw                 Optimize with AdamW (with --muon: every parameter that Muon does not own)
        --sgd                   Optimize with plain SGD (with --muon: every parameter that Muon does not own)
        --muon                  Put the hidden weight matrices on Muon
        --no-graph              Run every pass eagerly, not from captured CUDA graphs
        --resume                Continue from the progress saved with the optimizer state (schedule step, epoch,
                                batch), not from step 0 (no effect on a new optimizer)
        --checkpoint-every <N>  Also make a checkpoint every N optimizer steps (0: only at epoch ends)
        --valid-every <N>       Validate every N optimizer steps (0: no periodic validation)
        --valid-batches <N>     The batches that a periodic validation reads
    -a, --artifacts-path <PATH>
                                Directory to save and load the configs, the model weights and the optimizer state.
                                The program creates it if it does not exist.
                                Default: a new temporary directory (the program prints its path).

ARGS:
    -- <EXTRA_ARGS>             The program forwards all the arguments after -- unchanged to the flags of the example.
                                -h or --help there shows the help of the example.
```
