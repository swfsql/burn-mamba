# Mamba Examples

#### List of Examples:

- `fibonacci`: Very small model training on a fibonacci-like sequence.
- `mnist-class`: A small model training to classify mnist digits.

#### Examples Structure

Each example usually defines a model in `model.rs`, a dataset (if applicable) in `dataset.rs`, a training procedure in `training.rs`, an inference procedure (if applicable) in `inference.rs` and a launching procedure in `main.rs`.

The lauching procedure first triggers some basic command arguments parsing, which sets whether training and/or inference should run. The training often run validations every couple of batches, and each example's README may inform what the training goal is. The `model.rs` may also indicate the training requirements and expected resulting accuracy.

There are shared definitions in `common/mod.rs`, imported as an outside module by each example. Importantly, a common model definition and the backend selection is shared among all examples. Some dataset and helpers for training may be also defined under `common`.

##### Model Definition

The overall model used throughout the examples is the `Mamba2Network`, defined in `common/model.rs`. It contains some input and output projections (linear layers) and a `Mamba2Layers`, which is defined in `burn-mamba`. Helpers for configuring that model is also defined alogside it.

#### Backend Selection

A single backend must be enabled, and features are used to select it -- e.g. `dev-ndarray`. See `burn-mamba/Cargo.toml` > `[features]` section for the backend list. Some extra "dev" features are also available for selection, them being the float precision selection (default f32 vs `dev-f16`) and whether fusion and/or autotune should be enabled.  
If no backend is selected, you should get a compile error message.

#### Examples CLI

All examples use a CLI defined in `common/cli.rs`.

##### Usage Example

```bash
# training the simplest example on ndarray (fp32) and running inference:
cargo run --example fibonacci --features "dev-ndarray" -- --training --inference

# assume /tmp/fibonacci-abcd-0 got created:
ARTIFACTS="/tmp/fibonacci-abcd-0"

# running only the inference from the trained model:
cargo run --example fibonacci --features "dev-ndarray" -- --inference --artifacts-path "$ARTIFACTS"

# assume /some/path/ contains a different training config file, e.g. with a different seed:
TCONFIG="/some/path/training_config.json"

# continue training from another training config
# warning: "$ARTIFACTS/training_config.json" gets overwritten by "$TCONFIG"
cargo run --example fibonacci --features "dev-ndarray" -- --training --artifacts-path "$ARTIFACTS" --training-config "$TCONFIG"
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
