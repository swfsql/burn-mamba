# Mamba Examples

#### List of Examples:

- `reset-majority`: One Mamba-2 block (two scalar states, no conv, no residual) on the sign of a running vote since the last reset — the smallest task the block is *required* for. Its README carries a hand-built exact solution and a sweep showing no fixed decay reaches it.
- `reset-rotor`: The Mamba-3 corollary of `reset-majority` — the same `+`/`-`/`R` stream read as a three-detent rotor (the turn count since the last reset, mod 3), the smallest task needing the *complex* transition. Its README carries a hand-built exact solution plus sweeps showing that neither a fixed rotation nor a real (rotation-free) state reaches it.
- `reset-spinor`: The next rung — the same stream with two *non-commuting* turns, read as the running product in the quaternion group `Q₈`, the smallest task needing the **non-abelian** (`Quaternion4D`) rotation. Hand-built exact solution plus the abelian twin (`-- --rotation complex`) and the order-blind ceilings.
- `reset-swap`: The rung above that — the running order of three items (the `S₃` word problem), the smallest group with **more than one involution**, which no `SU(2)` state can hold: the left-isoclinic twin tracks the double cover `2D₃` and cannot present it to a linear head. Needs the full `SO(4)` rotation (`-- --rotation rotor|quaternion|complex`).
- `mnist-class`: A small Mamba-3 model training to classify mnist digits.
- `state-tracking`: A tiny Mamba-3 model on the `A₅` word problem, contrasting the abelian `Complex2D` rotation against the non-abelian `Quaternion4D` (`-- --rotation complex|quaternion|rotor`, the last being the full `SO(4)` rotation).
- `mnist-ae`: A symmetric bidirectional Mamba-3 autoencoder over the 784-pixel MNIST sequence; the decoder reconstructs the whole image in one parallel pass reading only from a configurable latent (`-- --latents N`).
- `grokking`: A small Mamba-2 LM on k-summand modular addition (the classic grokking task), grown into an experimentation/ablation platform: participation-ratio diagnostics (state + weight spectra + embedding-frequency), differentiable rank/norm/noise loss terms with schedules, and an SGD probe path. Its README is a standalone report of the findings with reproduction commands for every claim.

#### Examples Structure

Each example usually defines a model in `model.rs`, a dataset (if applicable) in `dataset.rs`, a training procedure in `training.rs`, an inference procedure (if applicable) in `inference.rs` and a launching procedure in `main.rs`.

The lauching procedure first triggers some basic command arguments parsing, which sets whether training and/or inference should run. The training often run validations every couple of batches, and each example's README may inform what the training goal is. The `model.rs` may also indicate the training requirements and expected resulting accuracy.

There are shared definitions in `common/mod.rs`, imported as an outside module by each example. Importantly, a common model definition and the backend selection is shared among all examples. Some dataset and helpers for training may be also defined under `common`.

##### Model Definition

The overall model used throughout the examples is the lib-generic `MambaLatentNet` (configured via `MambaLatentNetConfig`), defined in `burn-mamba`'s `src/generic.rs`. It is a continuous-I/O network: input and output projections (linear layers) around a generic `Layers<M>` stack, where `M` is the chosen SSM core (`Mamba1`/`Mamba2`/`Mamba3`). Token-based examples (`grokking`) use the lib's `MambaVocabNet` (embedding → `Layers<M>` → LM head) instead. `common/model.rs` only supplies the `ModelConfigExt` glue (config enum → `Module`); examples no longer define their own network types.

##### Optimizer

`common/training.rs` defines `OptimizerConfig { adamw, muon }`, held by
`TrainingConfig`. `muon = None` (the default) is plain AdamW on every parameter.
Setting `muon` moves the hidden weight matrices to
[Muon](https://kellerjordan.github.io/posts/muon/), driven by the model config's
`muon_plan()` (`ModelConfigExt::muon_plan`, backed by `burn_mamba::optim`): Muon
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
