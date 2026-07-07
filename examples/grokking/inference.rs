//! Inference for the grokking example: loads the trained model, reports
//! train/test accuracy on the same deterministic split used in training, and
//! prints a few sample test-pair predictions.

use crate::common::cli::AppArgs;
use crate::dataset;
use crate::training::{GrokkingConfig, eval_accuracies, final_logits};
use burn::prelude::*;
use burn_mamba::prelude::*;

/// Evaluate the trained model on the training-time split and print a handful
/// of held-out predictions.
pub fn infer(
    config: &GrokkingConfig,
    model_config: MambaVocabNetConfig,
    device: Device,
    app_args: &AppArgs,
) {
    let model: MambaVocabNet = app_args
        .load_model(&model_config, &device)
        .expect("no trained model in the artifacts directory; run with --training first");

    let (train_split, test_split) =
        dataset::build(config.p, config.train_fraction, config.split_seed);
    let (train_acc, test_acc) =
        eval_accuracies(&model, &train_split, &test_split, &device, config.stepwise);
    println!(
        "train acc {train_acc:.4} ({} pairs), test acc {test_acc:.4} ({} pairs), chance ≈ {:.4}",
        train_split.len(),
        test_split.len(),
        1.0 / config.p as f64,
    );

    // A few held-out examples.
    let n_samples = 8.min(test_split.len());
    let sample = dataset::Split {
        p: test_split.p,
        pairs: test_split.pairs[..n_samples].to_vec(),
        labels: test_split.labels[..n_samples].to_vec(),
    };
    let logits_bc = final_logits(&model, &sample.inputs_tensor(&device), config.stepwise);
    let [b, _classes] = logits_bc.dims();
    let preds = logits_bc
        .argmax(1)
        .reshape([b])
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    for (([a, y], label), pred) in sample.pairs.iter().zip(&sample.labels).zip(&preds) {
        let mark = if pred == label { "✓" } else { "✗" };
        println!("  {a:>2} + {y:>2} ≡ {pred:>2} (mod {})  [expected {label:>2}] {mark}", config.p);
    }
}
