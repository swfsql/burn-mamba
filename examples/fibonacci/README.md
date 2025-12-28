# Fibonacci-like Sequence

This is based on [burn/examples/modern-lstm](https://github.com/tracel-ai/burn/tree/fa4f9845a6b2279cd8de68bf7ca5a7eb76dec96d/examples/modern-lstm), where the lstm network was not copied over to here, but the overall setting is very comparable.

The dataset is generated based on two initial numbers between `0.0` and `1.0`, and each new value is the sum of the last two values (plus noise), thus resulting in a Fibonacci-like sequence. The prediction isn't considered for every timestep -- even if those values could be available -- but only for the last value of the sequence.

## Usage

The model is first trained, saved, then loaded and then used to run inference.

The trained model is saved to `/tmp/burn-mamba/fibonacci/`, which is later loaded for inference usage.  
WARNING: All files under `/tmp/burn-mamba/fibonacci/**` are removed at the start of the training.

#### NdArray backend (f32)

```sh
cargo run --example fibonacci --features "dev-ndarray"
```

See `burn-mamba/Cargo.toml` for other features or backend information.
