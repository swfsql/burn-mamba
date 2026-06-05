//! Inference / sampling for the sequential-MNIST classifier.
//!
//! [`infer`] loads the trained model, classifies a handful of test digits,
//! prints each digit as ASCII art beside its 10-bin class-probability bar chart,
//! and writes a PNG per digit (the image plus the probability bars).
//! [`save_predictions`] is the reusable PNG writer — the training loop calls it
//! at every small validation check to dump labelled samples into a fresh
//! `epoch-{e}-batch-{b}/` directory, so predictions can be watched over time.

use crate::AppArgs;
use crate::common::device::FloatElement;
use crate::common::mnist::dataset::{HEIGHT, MnistBatch, MnistBatcher, MnistDataset, WIDTH};
use burn::tensor::ElementConversion;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use burn_mamba::prelude::{Mamba3SsdPath, MambaLatentNet, MambaLatentNetConfig, MambaSsdPath};
use image::{GrayImage, Luma};
use std::path::Path;

/// Number of test digits to classify and display.
const NUM_SHOWN: usize = 8;

/// Number of classes (digits 0–9).
const NUM_CLASSES: usize = 10;

/// Nearest-neighbour upscale factor for the saved PNGs.
const SCALE: u32 = 8;

/// Width of the gray separator column between the digit and the bar chart.
const SEP: usize = 2;

/// Load the trained classifier, print each digit + its class distribution as
/// ASCII/text, and write the PNGs under `<artifacts>/inference/`.
pub fn infer(model_config: MambaLatentNetConfig, infer_device: Device, app_args: &AppArgs) {
    let model: MambaLatentNet = app_args
        .load_model(&model_config, &infer_device)
        .expect("failed to load model");

    // Grab the first `NUM_SHOWN` test digits.
    let dataset = MnistDataset::test();
    let items: Vec<_> = (0..NUM_SHOWN).filter_map(|i| dataset.get(i)).collect();
    let labels: Vec<u8> = items.iter().map(|it| it.label).collect();

    let batcher = MnistBatcher::default();
    let batch = batcher.batch(items, &infer_device);
    let images_norm = batch.images_norm(); // [n, H, W, 1] in [0, 1]
    let [n, _h, _w, _c] = images_norm.dims();

    // Terminal view: digit ASCII art + a text bar chart of the 10 probabilities.
    let probs = predict(&model, images_norm.clone()); // [n, 10]
    let probs_host = to_host(probs);
    let digits_host = to_host(images_norm.clone().reshape([n, HEIGHT * WIDTH]));
    for i in 0..n {
        let off = i * HEIGHT * WIDTH;
        let p = &probs_host[i * NUM_CLASSES..i * NUM_CLASSES + NUM_CLASSES];
        let pred = argmax(p);
        println!(
            "\n--- digit {i} (true label {}, predicted {pred}) ---",
            labels[i]
        );
        println!("{}", render_digit_ascii(&digits_host[off..off + HEIGHT * WIDTH]));
        println!("{}", render_prediction(p, labels[i] as usize, pred));
    }

    let out_dir = app_args.artifacts_path.join("inference");
    save_predictions(&model, images_norm, &labels, &out_dir);
    println!("\nsaved {n} prediction PNGs to {out_dir:?}");
}

/// Run the model on `images_norm` (`[n, H, W, 1]` in `[0, 1]`) and write one
/// "digit + 10-bin probability bar chart" PNG per sample into `out_dir` (created
/// if missing). The true label and prediction are encoded in each file name.
/// Reusable by both [`infer`] and the training-time sampling.
pub fn save_predictions(
    model: &MambaLatentNet,
    images_norm: Tensor<4>,
    labels: &[u8],
    out_dir: &Path,
) {
    let [n, _h, _w, _c] = images_norm.dims();
    let probs = predict(model, images_norm.clone()); // [n, 10]
    let probs_host = to_host(probs);
    let digits_host = to_host(images_norm.reshape([n, HEIGHT * WIDTH]));

    std::fs::create_dir_all(out_dir).expect("failed to create sample directory");
    for i in 0..n {
        let off = i * HEIGHT * WIDTH;
        let p = &probs_host[i * NUM_CLASSES..i * NUM_CLASSES + NUM_CLASSES];
        let true_label = labels.get(i).copied().unwrap_or(0) as usize;
        let pred = argmax(p);
        let img = digit_with_bars_png(&digits_host[off..off + HEIGHT * WIDTH], p, true_label, pred);
        let path = out_dir.join(format!("sample-{i:02}-label-{true_label}-pred-{pred}.png"));
        img.save(&path).expect("failed to save prediction PNG");
    }
}

/// Forward the classifier and return per-class probabilities `[n, 10]`.
///
/// `images_norm`: `[n, H, W, 1]` in `[0, 1]`; the model is fed the z-scored
/// pixels (matching training), the last timestep's logits are softmaxed.
fn predict(model: &MambaLatentNet, images_norm: Tensor<4>) -> Tensor<2> {
    let [n, h, w, _c] = images_norm.dims();
    // z-score to match training (see `MnistBatch::images_z_score`).
    let zscored = images_norm
        .sub_scalar(MnistBatch::MEAN)
        .div_scalar(MnistBatch::STDDEV)
        .reshape([n, h * w, 1]);
    let ssd_path = MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None));
    let (output, _caches) = model.forward(zscored, None, ssd_path); // [n, seq, 10]
    let seq = h * w;
    let last = output.narrow(1, seq - 1, 1).squeeze_dim::<2>(1); // [n, 10]
    burn::tensor::activation::softmax(last, 1)
}

/// Build a nearest-upscaled grayscale PNG: the digit on the left, a separator,
/// then a 10-bar probability chart (bars left→right are classes 0–9). The
/// predicted bar is full-white; the true class is marked by a faint full-height
/// column behind its bar.
fn digit_with_bars_png(digit: &[f32], probs: &[f32], true_label: usize, pred: usize) -> GrayImage {
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let (bar_w, gap, lead) = (2usize, 1usize, 1usize);
    let chart_w = lead + NUM_CLASSES * (bar_w + gap);
    let native_w = (WIDTH + SEP + chart_w) as u32;
    let h = HEIGHT as u32;
    let mut img = GrayImage::new(native_w, h);

    // Digit panel.
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            img.put_pixel(col as u32, row as u32, Luma([to_u8(digit[row * WIDTH + col])]));
        }
        for s in 0..SEP {
            img.put_pixel((WIDTH + s) as u32, row as u32, Luma([90])); // separator
        }
    }

    // Bar chart panel.
    let base_x = WIDTH + SEP + lead;
    // Faint full-height marker behind the true class.
    let true_x = base_x + true_label.min(NUM_CLASSES - 1) * (bar_w + gap);
    for xx in 0..bar_w {
        for row in 0..HEIGHT {
            img.put_pixel((true_x + xx) as u32, row as u32, Luma([60]));
        }
    }
    // Bars: height ∝ probability; predicted class brightest.
    for c in 0..NUM_CLASSES {
        let x0 = base_x + c * (bar_w + gap);
        let hbar = (probs[c].clamp(0.0, 1.0) * (HEIGHT as f32 - 1.0)).round() as usize;
        let shade = if c == pred { 255 } else { 140 };
        for xx in 0..bar_w {
            for row in (HEIGHT - hbar)..HEIGHT {
                img.put_pixel((x0 + xx) as u32, row as u32, Luma([shade]));
            }
        }
    }

    image::imageops::resize(
        &img,
        native_w * SCALE,
        h * SCALE,
        image::imageops::FilterType::Nearest,
    )
}

/// Index of the maximum value (the predicted class).
fn argmax(probs: &[f32]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
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

/// Render a `[HEIGHT * WIDTH]` intensity buffer as 28×28 ASCII.
fn render_digit_ascii(digit: &[f32]) -> String {
    const RAMP: &[u8] = b" .:-=+*#%@";
    let pixel = |v: f32| -> char {
        let v = v.clamp(0.0, 1.0);
        RAMP[(v * (RAMP.len() - 1) as f32).round() as usize] as char
    };
    let mut out = String::new();
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            out.push(pixel(digit[row * WIDTH + col]));
        }
        out.push('\n');
    }
    out
}

/// Render the 10 class probabilities as a text bar chart, marking the predicted
/// and true classes.
fn render_prediction(probs: &[f32], true_label: usize, pred: usize) -> String {
    let mut out = String::from("  class  prob\n");
    for (c, &p) in probs.iter().enumerate() {
        let bar = "#".repeat((p * 20.0).round() as usize);
        let mark = match (c == pred, c == true_label) {
            (true, true) => " <- pred ✓",
            (true, false) => " <- pred",
            (false, true) => " (true)",
            (false, false) => "",
        };
        out.push_str(&format!("  {c:>5}  {:>5.1}% |{bar:<20}|{mark}\n", p * 100.0));
    }
    out
}
