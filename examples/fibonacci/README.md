# Fibonacci-like Sequence

This is based on [burn/examples/modern-lstm](https://github.com/tracel-ai/burn/tree/fa4f9845a6b2279cd8de68bf7ca5a7eb76dec96d/examples/modern-lstm), where the lstm network was not copied over to here, but the overall setting is very comparable.

The dataset is generated based on two initial numbers between `0.0` and `1.0`, and each new value is the sum of the last two values (plus noise), thus resulting in a Fibonacci-like sequence. The prediction isn't considered for every timestep -- even if those values could be available -- but only for the last value of the sequence.

## Usage

##### Usage Example

```bash
# debug check in ndarray (fp32)
cargo check --example fibonacci --features "dev-ndarray"

# training and running inference in ndarray (fp32)
cargo run --release --example fibonacci --features "dev-ndarray" -- --training --inference
```

- See `burn-mamba/Cargo.toml` for other features or backend information.  
- See `burn-mamba/examples/README.md` for the CLI usage overview.
