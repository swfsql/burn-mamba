//! Inference / sampling for the MNIST autoencoder.
//!
//! [`infer`] loads the trained model, reconstructs a handful of test images,
//! prints each original next to its reconstruction as ASCII art, and writes them
//! out as PNGs. [`save_reconstructions`] is the reusable PNG writer — the
//! training loop calls it at every small validation check to dump
//! original-vs-reconstruction samples into a fresh `epoch-{e}-batch-{b}/`
//! directory, so the reconstruction quality can be watched over time.

use crate::AppArgs;
use crate::common::device::FloatElement;
use crate::common::mnist::dataset::{HEIGHT, MnistBatcher, MnistDataset, WIDTH};
use crate::model::{AeConfig, AeModel};
use burn::tensor::ElementConversion;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use image::{GrayImage, Luma};
use std::path::Path;

/// Number of test images to reconstruct and display.
const NUM_SHOWN: usize = 8;

/// Nearest-neighbour upscale factor for the saved PNGs (28px → 224px).
const SCALE: u32 = 8;

/// Width of the gray separator column between original and reconstruction.
const SEP: usize = 2;

/// Load the trained model, print originals vs. reconstructions as ASCII art, and
/// write them out as PNGs under `<artifacts>/inference/`.
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

    // ASCII art to the terminal for a quick look.
    let logits = model.forward(input.clone()); // [n, H*W]
    let recon = burn::tensor::activation::sigmoid(logits);
    let recon_host = to_host(recon);
    let original_host = to_host(input.clone().reshape([n, HEIGHT * WIDTH]));
    for i in 0..n {
        let off = i * HEIGHT * WIDTH;
        println!("\n--- digit {} (label {}) ---", i, labels[i]);
        println!(
            "{}",
            render_side_by_side(
                &original_host[off..off + HEIGHT * WIDTH],
                &recon_host[off..off + HEIGHT * WIDTH],
            )
        );
    }

    // PNGs to disk.
    let out_dir = app_args.artifacts_path.join("inference");
    save_reconstructions(&model, input, &labels, &out_dir);
    println!("\nsaved {n} reconstruction PNGs to {out_dir:?}");
}

/// Run the model on `images_bhw1` and write one side-by-side
/// (original | reconstruction) PNG per sample into `out_dir` (created if
/// missing). Reusable by both [`infer`] and the training-time sampling.
///
/// `images_bhw1`: `[n, HEIGHT, WIDTH, 1]` in `[0, 1]`; `labels`: one per sample.
pub fn save_reconstructions(
    model: &AeModel,
    images_bhw1: Tensor<4>,
    labels: &[u8],
    out_dir: &Path,
) {
    let [n, _h, _w, _c] = images_bhw1.dims();
    let logits = model.forward(images_bhw1.clone()); // [n, H*W]
    let recon = burn::tensor::activation::sigmoid(logits);
    let recon_host = to_host(recon);
    let original_host = to_host(images_bhw1.reshape([n, HEIGHT * WIDTH]));

    std::fs::create_dir_all(out_dir).expect("failed to create sample directory");
    for i in 0..n {
        let off = i * HEIGHT * WIDTH;
        let img = side_by_side_png(
            &original_host[off..off + HEIGHT * WIDTH],
            &recon_host[off..off + HEIGHT * WIDTH],
        );
        let label = labels.get(i).copied().unwrap_or(0);
        let path = out_dir.join(format!("sample-{i:02}-label-{label}.png"));
        img.save(&path).expect("failed to save reconstruction PNG");
    }
}

/// Build a nearest-upscaled side-by-side grayscale PNG: original on the left, a
/// gray separator, then the reconstruction on the right.
fn side_by_side_png(orig: &[f32], recon: &[f32]) -> GrayImage {
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let native_w = (WIDTH * 2 + SEP) as u32;
    let mut img = GrayImage::new(native_w, HEIGHT as u32);
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            img.put_pixel(col as u32, row as u32, Luma([to_u8(orig[row * WIDTH + col])]));
            let rcol = (WIDTH + SEP + col) as u32;
            img.put_pixel(rcol, row as u32, Luma([to_u8(recon[row * WIDTH + col])]));
        }
        for s in 0..SEP {
            img.put_pixel((WIDTH + s) as u32, row as u32, Luma([90])); // separator
        }
    }
    image::imageops::resize(
        &img,
        native_w * SCALE,
        HEIGHT as u32 * SCALE,
        image::imageops::FilterType::Nearest,
    )
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
