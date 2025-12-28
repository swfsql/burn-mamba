# MNIST Classification

The dataset is mostly based on [burn-dataset/vision/mnist](https://github.com/tracel-ai/burn/blob/fa4f9845a6b2279cd8de68bf7ca5a7eb76dec96d/crates/burn-dataset/src/vision/mnist.rs) and [book/data](https://burn.dev/books/burn/basic-workflow/data.html#data). It is mnist as flat (sequential) pixels, with sequence length of 28 * 28 = 784. The model reads the pixel sequence and predicts the classification label at the last input.

## Usage

The dataset is first downloaded and stored in `$HOME/.cache/burn-dataset/mnist/train/`. The files are the following:

- train-images-idx3-ubyte (9.45 MB)
- train-labels-idx1-ubyte (28.20 KB)
- tk10-images-idx3-ubyte (1.57 MB)
- tk10-labels-idx1-ubyte (4.44 KB)

Then the model is trained and saved to `/tmp/burn-mamba/mnist-class/`.  
WARNING: All files under `/tmp/burn-mamba/mnist-class/**` are removed at the start of the training.

#### Cuda backend (fp16)

```sh
cargo run --release --example mnist-class --features "dev-cuda,dev-f16"
```

See `burn-mamba/Cargo.toml` for other features or backend information.  
See `model.rs` file for vram system requirements.
