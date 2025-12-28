# Mamba Examples

#### List of Examples:

- `fibonacci`: Very small model training on fibonacci-like sequence.
- `mnist-class`: A small model training to classify mnist digits.

#### Examples Structure

Each example usually defines a model in `model.rs`, a dataset (if applicable) in `dataset.rs`, a CLI training procedure in `training.rs`, an inference procedure (if applicable) in `inference.rs` and a launching procedure in `main.rs`. The lauching procedure first triggers training (note: it removes any previously produced files) and then triggers the inference (if applicable). The training often run validations every couple of batches, and each example's README may inform what the training goal is. The `model.rs` may also indicate the training requirements and expected resulting accuracy.

There are shared definitions in `common/mod.rs`, imported as an outside module by each example. Importantly, a common model definition and the backend selection is shared among all examples. Some dataset and helpers for training may be also defined under `common`.

#### Backend Selection

A single backend must be enabled, and features are used to select it -- e.g. `dev-ndarray`. See `burn-mamba/Cargo.toml` > `[features]` section for the backend list. Some extra features are also available for selection, them being the float precision selection (default f32 vs `dev-f16`) and whether fusion and/or autotune should be enabled.  
If no backend is selected, you should get a compile error indicating this.

Running the simplest example on ndarray (f32):

```bash
cargo run --example fibonacci --features "dev-ndarray"
```

#### Model Definition

The overall model used throughout the examples is the `Mamba2Network`, defined in `common/model.rs`. It contains some input and output projections (linear layers) and a `Mamba2Layers`, which is defined in `burn-mamba`. Helpers for configuring that model is also defined alogside it.
