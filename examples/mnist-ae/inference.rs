//! Inference for the MNIST autoencoder: loads the trained model, reconstructs a
//! handful of test images, and prints each original next to its reconstruction
//! as ASCII art (a quick visual check that the latent bottleneck carries the
//! digit).

use crate::AppArgs;
use crate::common::device::FloatElement;
use crate::common::mnist::dataset::{HEIGHT, MnistBatcher, MnistDataset, WIDTH};
use crate::model::{AeConfig, AeModel};
use burn::tensor::ElementConversion;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

/// Number of test images to reconstruct and display.
const NUM_SHOWN: usize = 8;

/// Load the trained model and print originals vs. reconstructions as ASCII art.
pub fn infer(model_config: AeConfig, infer_device: Device, app_args: &AppArgs) {
    let model: AeModel = app_args
        .load_model(&model_config, &infer_device)
        .expect("failed to load model");

    // Grab the first `NUM_SHOWN` test images.
    let dataset = MnistDataset::test();
    let items: Vec<_> = (0..NUM_SHOWN).filter_map(|i| dataset.get(i)).collect();
    let labels: Vec<u8> = items.iter().map(|it| it.label).collect();

    let batcher = MnistBatcher::default();
    let batch = batcher.batch(items, &infer_device);

    let input = batch.images_norm(); // [n, H, W, 1] in [0, 1]
    let [n, _h, _w, _c] = input.dims();
    let input = input.reshape([n, HEIGHT * WIDTH, 1]);

    let logits = model.forward(input.clone()); // [n, 784, 1]
    let recon = burn::tensor::activation::sigmoid(logits).reshape([n, HEIGHT * WIDTH]);
    let original = input.reshape([n, HEIGHT * WIDTH]);

    let recon = to_host(recon);
    let original = to_host(original);

    for i in 0..n {
        let off = i * HEIGHT * WIDTH;
        let orig = &original[off..off + HEIGHT * WIDTH];
        let rec = &recon[off..off + HEIGHT * WIDTH];
        println!("\n--- digit {} (label {}) ---", i, labels[i]);
        println!("{}", render_side_by_side(orig, rec));
    }
}

/// Read a float tensor back to a host `Vec<f32>` (dtype-agnostic).
fn to_host<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .to_vec::<FloatElement>()
        .unwrap()
        .into_iter()
        .map(|x| x.elem::<f32>())
        .collect()
}

/// Render two `[HEIGHT * WIDTH]` intensity buffers as side-by-side 28×28 ASCII.
fn render_side_by_side(left: &[f32], right: &[f32]) -> String {
    // Intensity ramp from empty to full.
    const RAMP: &[u8] = b" .:-=+*#%@";
    let pixel = |v: f32| -> char {
        let v = v.clamp(0.0, 1.0);
        let idx = (v * (RAMP.len() - 1) as f32).round() as usize;
        RAMP[idx] as char
    };
    let mut out = String::new();
    out.push_str("  original                     reconstruction\n");
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            out.push(pixel(left[row * WIDTH + col]));
        }
        out.push_str("   ");
        for col in 0..WIDTH {
            out.push(pixel(right[row * WIDTH + col]));
        }
        out.push('\n');
    }
    out
}
