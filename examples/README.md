# Mamba Examples

#### List of Examples:

- `reset-*`: A four-rung ladder on the same `+`/`-`/`R`-shaped stream, each rung the smallest task its block is *needed* for and that the rung below cannot solve: `reset-majority` (the selective decay alone, `Real1D`), `reset-rotor` (the complex transition, `Z₃`), `reset-spinor` (the non-abelian quaternion one, `Q₈`), `reset-swap` (the full two-sided `SO(4)`, `S₃`). Each carries a hand-built exact solution and the ablations that wall off the rung below. `spinor-product` is the coda that moves the *other* dial: `reset-spinor`'s `Q₈` stream read **two symbols per token**, the smallest task `Mamba3Config::micro_steps` (MambaProduct's `u`) is needed for — one step's rotation generator is affine in the token, so a pair's two generators can only *add* where the group multiplies, while `u = 2` makes the token two recurrence steps and its transition the product. It also contrasts against the other way to put two rotations in a token — a second layer (`-- --layers 2 --micro-steps 1`), which turns a second state and so has to *learn* the product instead of being handed it. `reset-quintic` asks `reset-swap`'s question over five items: `A₅` (the icosahedron's rotations, the smallest non-solvable group) is one `Rotor4D` block at `reset-swap`'s width, while `S₅` is the transition group of no single Mamba-3 layer at any size — its swaps would have to be reflections — and takes two. The single [`reset/README.md`](reset/README.md) covers all six.
- `tally-*`: A three-rung ladder on the *other* axis from `reset-*`: the plant is held at `reset-majority`'s floor (a scalar real state) and what climbs is the **positive system** beside it — the per-head scalar recurrence of `src/mamba3/positive/` that reads only the input and sets the plant's coefficients. `tally-depth` (a counter with a floor) needs a `Tropical::MaxPlus` register for its **growth**, a decay above one, which the plant's `α = exp(Δ·A) ≤ 1` cannot be; `tally-record` (a running maximum over twelve values) needs the same register for its **range**, since the one route that needs no register — a sum of `exp(S·v)` — asks for `e^{45.8}` inside a single channel; `tally-drift` (age evidence by how much of it there is) needs the `Gain::KalmanProjectedNoise` filter, whose hyperbolic discount no projected decay matches. Each rung carries a hand-built exact solution, ablation sweeps, and a trained ablation at equal budget — the row that sizes the second rung's alphabet, since at six values a trained stock block solves it. The single [`tally/README.md`](tally/README.md) covers all three.
- `mnist-class`: A small Mamba-3 model training to classify mnist digits.
- `mnist-ae`: A symmetric bidirectional Mamba-3 autoencoder over the 784-pixel MNIST sequence; the decoder reconstructs the whole image in one parallel pass reading only from a configurable latent (`-- --latents N`).
- `tiny-stories`: A tiny character-level Mamba-3 language model on the cleaned [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) corpus, with a tied 48-character embedding at both ends. Its README covers the case-folded alphabet (the corpus's own inventory), the whole-parquet download, the class latents that mark a story's start (and that `prime()` replays for seedless sampling), the prefill-`forward()` / decode-`step()` sampler, the runs of windows that carry the state across a story behind a frontier gate, and a measured table on what truncated BPTT costs a language model.

#### Examples Structure

Each example lives in its own directory (the `reset-*` ladder, `spinor-product` and `reset-quintic` one level deeper, under `reset/`, sharing one README — those six are declared as explicit `[[example]]` targets in `Cargo.toml`, since cargo only autodiscovers `examples/<name>/main.rs`; the three `tally-*` rungs likewise, under `tally/`, where they also share a `shared/` module holding the ladder's dataset, loops and hand-built helpers). An example usually defines a model in `model.rs`, a dataset (if applicable) in `dataset.rs`, a training procedure in `training.rs`, an inference procedure (if applicable) in `inference.rs`, its own command-line flags in `cli.rs` (empty when it has none) and a launching procedure in `main.rs`.

The lauching procedure first parses the command line — the shared flags, which set whether training and/or inference should run, and then the example's own, after `--` — reading no environment variables. The training often run validations every couple of batches, and each example's README may inform what the training goal is. The `model.rs` may also indicate the training requirements and expected resulting accuracy.

There are shared definitions in `common/mod.rs`, imported as an outside module by each example. It is a thin shim: the CLI, the runtime device selection, the training config, and both datasets with their epoch loops (sequential-MNIST classification, character-level TinyStories language modelling) all live in **`burn_stack::examples`** (feature `examples-common`, dev-only), shared verbatim with `burn-deltanet`, and the `config → module` seam is `burn_stack::modules::ModelConfigExt`, implemented by this crate's network configs. `common/mod.rs` re-exports those under the `common::*` paths and adds `ARTIFACT_PREFIX`, which has to be expanded in the example crate, and `parse_rotation`, the `--rotation` value the rotating `reset-*` rungs share (a flag naming a `burn-mamba` type, which `burn-stack` cannot).

##### Model Definition

The overall model used throughout the examples is the lib-generic `MambaLatentNet` (configured via `MambaLatentNetConfig`), defined in `burn-mamba`'s `src/unified/network.rs`. It is a continuous-I/O network: input and output projections (linear layers) around a generic `Layers<M>` stack, where `M` is the chosen SSM core (`Mamba1`/`Mamba2`/`Mamba3`). Token-based examples (`tiny-stories`) use the lib's `MambaVocabNet` (embedding → `Layers<M>` → LM head) instead. `ModelConfigExt` (config enum → `Module`, plus the Muon plan) is implemented on those configs in `src/unified/network.rs`; examples no longer define their own network types.

##### Optimizer

`burn_stack::examples::training` defines `OptimizerConfig { fallback, muon }`,
held by `TrainingConfig`. The fallback is AdamW or plain SGD, the one optimizer
a captured training step can replay (under `--sgd`, every example but
`tiny-stories`, through burn-stack's `examples::trainer::Trainer`: forward,
backward and update captured at the first batch, the loss masked rather than
gathered so the shapes never change); `muon = None` puts every parameter on it. Setting `muon` moves the hidden weight matrices to
[Muon](https://kellerjordan.github.io/posts/muon/), driven by the model config's
`muon_plan()` (`ModelConfigExt::muon_plan`, backed by `burn_stack::optim`): Muon
only ever gets rank-2 hidden matrices, and each fused projection (`in_proj`,
`fc1`, …) is split into its independent sub-projections first, so the
orthogonalisation is per linear map rather than per allocation. Every example
takes the choice from the shared `--adamw` / `--sgd` / `--muon` flags (below);
without them, `tiny-stories` trains Muon + AdamW and the rest AdamW.

#### Backend Selection

A single backend must be enabled, and features are used to select it -- e.g. `backend-flex`. See `burn-mamba/Cargo.toml` > `[features]` section for the backend list. Some extra "dev" features are also available for selection, them being the float precision selection (default f32 vs `dev-f16`) and whether fusion and/or autotune should be enabled.  
If no backend is selected, you should get a compile error message.

#### Examples CLI

All examples use a CLI defined in `burn_stack::examples::cli`, re-exported as `common::cli`: the flags every example shares (below). The arguments after a second `--` are the example's own, parsed by its `cli.rs`; `-- --help` lists them.

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

A command-line tool for training and/or running inference with machine learning models.
Models, optimizers, and configurations are persisted in an artifacts directory.

USAGE:
    example-name [OPTIONS] [-- <EXTRA_ARGS>...]

When no --training or --inference flag is provided, the program exits after handling configuration logic.

BEHAVIOR OVERVIEW
- The program manages two configurations: training config and model config.
- If --training-config or --model-config is given, the corresponding config is loaded from the specified file and saved to the artifacts directory (overwriting any existing file).
- If no explicit config file is provided for a component, the program attempts to load it from the artifacts directory; if absent, a default configuration is created and saved.
- The artifacts directory (--artifacts-path) is used to read/write model weights, optimizer state, and configurations. If not specified, a new temporary directory is created and its path is printed.
- With --remove-artifacts, any existing model and optimizer files (and the saved progress) in the artifacts directory are deleted before training (if --training is active).
- Model and optimizer weights are loaded from the artifacts directory if present; otherwise new ones are created and saved.
- With --seed, --epochs, --batch-size or --max-lr, the given value replaces the training config's (loaded or created) before the config is saved, so later runs from the same artifacts directory inherit it. --epochs also rescales a cosine LR schedule's length by the same factor, so the schedule still spans the run; --batch-size rescales its length and warmup by the inverse one (an epoch has that many fewer steps).
- --adamw, --sgd and --muon choose the optimizer. --muon puts the model's hidden weight matrices on Muon and the other flag (default --adamw) optimizes every other parameter; --adamw and --sgd are exclusive. Without any of them a new training config gets the example's own default. A loaded config's optimizer is replaced by the flags' choice (its LR schedule is kept: see --max-lr), unless optimizer state saved under the old optimizer would then be ignored, which panics instead (the state is removed by --remove-artifacts with --training). Only plain SGD (--sgd alone) has a training step that replays from a captured graph.
- An example that supports it replays its fixed-shape passes (training steps under plain SGD, validation, decoding) from captured CUDA graphs; --no-graph runs every pass eagerly.
- The optimizer state is saved together with the run's progress: the LR-schedule step, the epoch, and the batch within it. A run that loads it starts over at step 0 of epoch 1, unless --resume is given, which continues from the saved progress. The interrupted epoch then trains only the batches it has left, drawn from a fresh shuffle (the dataloader workers' batch order cannot be replayed).
- Training checkpoints at every epoch end and when it stops. --checkpoint-every adds a checkpoint every that many optimizer steps, --valid-every a periodic validation, and --valid-batches caps the batches that validation reads; each example has its own defaults for these three.
- Every training step and validation is appended as one JSON line to metrics.jsonl in the artifacts directory; each run opens with a "start" line.
- If both --training and --inference are specified, training executes first, followed by inference using the trained model.
- With --max-batches, training stops after that many mini-batches in total (counted across epochs), checkpointing as usual before it returns. One mini-batch is one optimizer step, which for the character LM is one window of a run rather than one dataloader item. --max-seconds stops it the same way once that much wall-clock time has passed since its first step.
- Any arguments following -- are captured as-is and forwarded to the example's own flags (-- --help lists them).

FLAGS:
    -h, --help                  Show this help message and exit

OPTIONS:
    -t, --training              Run training (creates or updates model / optimizer)
    -i, --inference             Run inference after training (if both flags are used) or immediately (if only inference is requested)
    -r, --remove-artifacts      Delete existing model and optimizer files from the artifacts directory before training
                                (has no effect if --training is not used)
    -c, --training-config <PATH>
                                Load training configuration from this file (overrides any config in artifacts directory)
    -m, --model-config <PATH>   Load model configuration from this file (overrides any config in artifacts directory)
    -b, --max-batches <N>       Stop training after N mini-batches in total (across epochs), regardless of the
                                configured number of epochs. Unlimited when absent.
        --max-seconds <S>       Stop training once S seconds have passed since its first step. Unlimited when absent.
    -s, --seed <N>              Replace the training config's RNG seed (model init, data shuffling, sampling)
        --epochs <N>            Replace the training config's number of epochs (rescaling a cosine LR schedule)
        --batch-size <N>        Replace the training config's mini-batch size (rescaling a cosine LR schedule)
        --max-lr <LR>           Replace the LR schedule's peak rate (a constant schedule's only one)
        --adamw                 Optimize with AdamW (with --muon: every parameter Muon does not own)
        --sgd                   Optimize with plain SGD (with --muon: every parameter Muon does not own)
        --muon                  Put the hidden weight matrices on Muon
        --no-graph              Run every pass eagerly instead of replaying captured CUDA graphs
        --resume                Continue from the progress saved with the optimizer state (schedule step, epoch,
                                batch) instead of from step 0 (has no effect on a new optimizer)
        --checkpoint-every <N>  Also checkpoint every N optimizer steps (0: only at epoch ends)
        --valid-every <N>       Validate every N optimizer steps (0: no periodic validation)
        --valid-batches <N>     Batches a periodic validation reads
    -a, --artifacts-path <PATH>
                                Directory where configurations, model weights, and optimizer state are saved and loaded.
                                If the directory does not exist, it will be created.
                                Defaults to a newly created temporary directory (path will be printed).

ARGS:
    -- <EXTRA_ARGS>             All arguments after -- are forwarded verbatim to the example's own flags.
                                Passing -h or --help there displays its help information.
```
