//! Inference for the grokking example: loads the trained model, reports
//! train/test accuracy on the same deterministic split used in training, and
//! prints a few sample test-pair predictions.

use crate::common::cli::AppArgs;
use crate::dataset;
use crate::diagnostics;
use crate::training::{GrokkingConfig, eval_accuracies, final_logits, format_prs};
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
        dataset::build(config.p, config.k, config.train_fraction, config.split_seed);
    let (train_acc, test_acc) =
        eval_accuracies(&model, &train_split, &test_split, &device, config.stepwise);
    println!(
        "train acc {train_acc:.4} ({} seqs), test acc {test_acc:.4} ({} seqs), chance ≈ {:.4}",
        train_split.len(),
        test_split.len(),
        1.0 / config.p as f64,
    );

    let diag_inputs = dataset::diagnostic_set(config.p, config.k, 10_000, config.split_seed)
        .inputs_tensor(&device);
    let state_prs = diagnostics::state_pr(&model, &diag_inputs);
    let weight_prs = diagnostics::weight_pr(&model, config.p);
    println!("{}", format_prs(&state_prs, &weight_prs));

    // A few held-out examples.
    let sample = test_split.head(8);
    let logits_bc = final_logits(&model, &sample.inputs_tensor(&device), config.stepwise);
    let [b, _classes] = logits_bc.dims();
    let preds = logits_bc
        .argmax(1)
        .reshape([b])
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    for ((seq, label), pred) in sample
        .seqs
        .chunks_exact(sample.k)
        .zip(&sample.labels)
        .zip(&preds)
    {
        let lhs = seq
            .iter()
            .map(|x| format!("{x:>2}"))
            .collect::<Vec<_>>()
            .join(" + ");
        let mark = if pred == label { "✓" } else { "✗" };
        println!("  {lhs} ≡ {pred:>2} (mod {})  [expected {label:>2}] {mark}", config.p);
    }
}
