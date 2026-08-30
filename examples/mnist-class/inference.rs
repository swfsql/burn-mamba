//! Inference / sampling for the sequential-MNIST classifier.
//!
//! [`infer`] loads the trained model, classifies a handful of test digits,
//! prints each digit as ASCII art beside its 10-bin class-probability bar chart,
//! and writes a PNG per digit (the image plus the probability bars). The
//! rendering itself is `burn_stack::examples::mnist::render`, shared with
//! `burn-deltanet`; what is Mamba's here is [`predict`], which the training
//! loop also calls at every small validation check to dump labelled samples
//! into a fresh `epoch-{e}-batch-{b}/` directory.

use crate::AppArgs;
use crate::common::mnist::dataset::{MnistBatch, MnistBatcher, MnistDataset};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use burn_mamba::prelude::{MambaLatentNet, MambaLatentNetConfig};
use burn_stack::examples::mnist::render;

/// Number of test digits to classify and display.
const NUM_SHOWN: usize = 8;

/// Load the trained classifier, print each digit + its class distribution as
/// ASCII/text, and write the PNGs under `<artifacts>/inference/`.
pub fn infer(model_config: MambaLatentNetConfig, infer_device: Device, app_args: &AppArgs) {
    let model: MambaLatentNet = app_args
        .load_model(&model_config, &infer_device)
        .expect("failed to load model");

    // Grab the first `NUM_SHOWN` test digits.
    let dataset = MnistDataset::test();
    let items: Vec<_> = (0..NUM_SHOWN).filter_map(|i| dataset.get(i).ok()).collect();
    let labels: Vec<u8> = items.iter().map(|it| it.label).collect();

    let batcher = MnistBatcher::default();
    let batch = batcher.batch(items, &infer_device);
    let images_norm = batch.images_norm(); // [n, H, W, 1] in [0, 1]
    let [n, _h, _w, _c] = images_norm.dims();

    // Terminal view: digit ASCII art + a text bar chart of the 10 probabilities.
    let probs = predict(&model, images_norm.clone()); // [n, 10]
    render::print_predictions(probs.clone(), images_norm.clone(), &labels);

    let out_dir = app_args.artifacts_path.join("inference");
    render::save_predictions(probs, images_norm, &labels, &out_dir);
    println!("\nsaved {n} prediction PNGs to {out_dir:?}");
}

/// Forward the classifier and return per-class probabilities `[n, 10]`.
///
/// `images_norm`: `[n, H, W, 1]` in `[0, 1]`; the model is fed the z-scored
/// pixels (matching training), the last timestep's logits are softmaxed.
pub fn predict(model: &MambaLatentNet, images_norm: Tensor<4>) -> Tensor<2> {
    let [n, h, w, _c] = images_norm.dims();
    // z-score to match training (see `MnistBatch::images_z_score`).
    let zscored = images_norm
        .sub_scalar(MnistBatch::MEAN)
        .div_scalar(MnistBatch::STDDEV)
        .reshape([n, h * w, 1]);
    let (output, _caches) = model.forward(zscored, None, crate::training::ssd_path(), None);
    // The class latents lengthen the sequence (all `Start`, so the last position
    // is still the last pixel) — see `model::OUTPUT_SEQUENCE_EXTRA`.
    let seq = h * w + crate::model::OUTPUT_SEQUENCE_EXTRA;
    let last = output.narrow(1, seq - 1, 1).squeeze_dim::<2>(1); // [n, 10]
    burn::tensor::activation::softmax(last, 1)
}
