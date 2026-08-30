# Mamba Examples

#### List of Examples:

- `reset-*`: A four-rung ladder on the same `+`/`-`/`R`-shaped stream, each rung the smallest task its block is *needed* for and that the rung below cannot solve: `reset-majority` (Mamba-2's selective decay), `reset-rotor` (the complex transition, `Z₃`), `reset-spinor` (the non-abelian quaternion one, `Q₈`), `reset-swap` (the full two-sided `SO(4)`, `S₃`). Each carries a hand-built exact solution and the ablations that wall off the rung below; the single [`reset/README.md`](reset/README.md) covers all four.
- `mnist-class`: A small Mamba-3 model training to classify mnist digits.
- `mnist-ae`: A symmetric bidirectional Mamba-3 autoencoder over the 784-pixel MNIST sequence; the decoder reconstructs the whole image in one parallel pass reading only from a configurable latent (`-- --latents N`).
- `tiny-stories`: A tiny character-level Mamba-3 language model on the cleaned [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) corpus, with a tied 48-character embedding at both ends. Its README covers the case-folded alphabet (the corpus's own inventory), the datasets-server paging that avoids a 673MB parquet, the prefill-`forward()` / decode-`step()` sampler, and a measured table on what truncated BPTT costs a language model.

#### Examples Structure

Each example lives in its own directory (the `reset-*` ladder one level deeper, under `reset/`, sharing one README — those four are declared as explicit `[[example]]` targets in `Cargo.toml`, since cargo only autodiscovers `examples/<name>/main.rs`). An example usually defines a model in `model.rs`, a dataset (if applicable) in `dataset.rs`, a training procedure in `training.rs`, an inference procedure (if applicable) in `inference.rs` and a launching procedure in `main.rs`.

The lauching procedure first triggers some basic command arguments parsing, which sets whether training and/or inference should run. The training often run validations every couple of batches, and each example's README may inform what the training goal is. The `model.rs` may also indicate the training requirements and expected resulting accuracy.

There are shared definitions in `common/mod.rs`, imported as an outside module by each example. Importantly, a common model definition and the backend selection is shared among all examples. Some dataset and helpers for training may be also defined under `common`.

##### Model Definition

The overall model used throughout the examples is the lib-generic `MambaLatentNet` (configured via `MambaLatentNetConfig`), defined in `burn-mamba`'s `src/unified/network.rs`. It is a continuous-I/O network: input and output projections (linear layers) around a generic `Layers<M>` stack, where `M` is the chosen SSM core (`Mamba1`/`Mamba2`/`Mamba3`). Token-based examples (`tiny-stories`) use the lib's `MambaVocabNet` (embedding → `Layers<M>` → LM head) instead. `common/model.rs` only supplies the `ModelConfigExt` glue (config enum → `Module`); examples no longer define their own network types.

##### Optimizer

`common/training.rs` defines `OptimizerConfig { adamw, muon }`, held by
`TrainingConfig`. `muon = None` (the default) is plain AdamW on every parameter.
Setting `muon` moves the hidden weight matrices to
[Muon](https://kellerjordan.github.io/posts/muon/), driven by the model config's
`muon_plan()` (`ModelConfigExt::muon_plan`, backed by `burn_stack::optim`): Muon
only ever gets rank-2 hidden matrices, and each fused projection (`in_proj`,
`fc1`, …) is split into its independent sub-projections first, so the
orthogonalisation is per linear map rather than per allocation. The `mnist-class`
example exposes this as a `-- --muon` flag; see its README.

#### Backend Selection

A single backend must be enabled, and features are used to select it -- e.g. `backend-flex`. See `burn-mamba/Cargo.toml` > `[features]` section for the backend list. Some extra "dev" features are also available for selection, them being the float precision selection (default f32 vs `dev-f16`) and whether fusion and/or autotune should be enabled.  
If no backend is selected, you should get a compile error message.

#### Examples CLI

All examples use a CLI defined in `common/cli.rs`.

##### Usage Example

```bash
# training the simplest example on flex (fp32) and running inference:
cargo run --example reset-majority --features "backend-flex" -- --training --inference

# assume /tmp/reset-majority-abcd-0 got created:
ARTIFACTS="/tmp/reset-majority-abcd-0"

# running only the inference from the trained model:
cargo run --example reset-majority --features "backend-flex" -- --inference --artifacts-path "$ARTIFACTS"

# assume /some/path/ contains a different training config file, e.g. with a different seed:
TCONFIG="/some/path/training_config.json"

# continue training from another training config
# warning: "$ARTIFACTS/training_config.json" gets overwritten by "$TCONFIG"
cargo run --example reset-majority --features "backend-flex" -- --training --artifacts-path "$ARTIFACTS" --training-config "$TCONFIG"
```

##### CLI Help Message

```txt
Burn Mamba Example

A command-line tool for training and/or running inference with machine learning models.
Models, optimizers, and configurations are persisted in an artifacts directory.

USAGE:
    example-name [OPTIONS]

When no --training or --inference flag is provided, the program exits after handling configuration logic.

BEHAVIOR OVERVIEW
- The program manages two configurations: training config and model config.
- If --training-config or --model-config is given, the corresponding config is loaded from the specified file and saved to the artifacts directory (overwriting any existing file).
- If no explicit config file is provided for a component, the program attempts to load it from the artifacts directory; if absent, a default configuration is created and saved.
- The artifacts directory (--artifacts-path) is used to read/write model weights, optimizer state, and configurations. If not specified, a new temporary directory is created and its path is printed.
- With --remove-artifacts, any existing model and optimizer files in the artifacts directory are deleted before training (if --training is active).
- Model and optimizer weights are loaded from the artifacts directory if present; otherwise new ones are created and saved.
- If both --training and --inference are specified, training executes first, followed by inference using the trained model.

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
    -a, --artifacts-path <PATH>
                                Directory where configurations, model weights, and optimizer state are saved and loaded.
                                If the directory does not exist, it will be created.
                                Defaults to a newly created temporary directory (path will be printed).
```
