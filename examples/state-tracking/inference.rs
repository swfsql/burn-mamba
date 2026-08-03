//! Inference for the state-tracking example: loads the trained model, reports
//! train/test accuracy on the training-time split, the per-position depth
//! curve, the full PR diagnostics panel (both state axes), and a few sample
//! held-out words.

use crate::common::cli::AppArgs;
use crate::common::protocol;
use crate::dataset::{self, NUM_CLASSES, NUM_SYMBOLS};
use crate::diagnostics;
use crate::training::{
    TrackingConfig, eval_accuracies, format_per_position, format_prs, per_position_accuracy,
};
use burn::prelude::*;
use burn_mamba::prelude::*;

/// Evaluate the trained model on the training-time split, print the depth
/// curve and the diagnostics panel, and show a handful of held-out words.
pub fn infer(
    config: &TrackingConfig,
    model_config: MambaVocabNetConfig,
    device: Device,
    app_args: &AppArgs,
) {
    let model: MambaVocabNet = app_args
        .load_model(&model_config, &device)
        .expect("no trained model in the artifacts directory; run with --training first");
    if config.frontier {
        return infer_frontier(config, &model, device);
    }
    let chunk = config.chunk_len();

    let (train_split, test_split) =
        dataset::build(config.seq_len, config.train_fraction, config.split_seed);
    let (train_acc, test_acc) = eval_accuracies(
        &model,
        &train_split,
        &test_split,
        &device,
        config.stepwise,
        chunk,
    );
    println!(
        "train acc {train_acc:.4} ({} words), test acc {test_acc:.4} ({} words), chance ≈ {:.4}",
        train_split.len(),
        test_split.len(),
        1.0 / NUM_CLASSES as f64,
    );
    let pos_acc = per_position_accuracy(
        &model,
        &test_split.inputs_tensor(&device),
        &test_split.pos_targets_tensor(&device),
        chunk,
    );
    println!("test per-position acc: {}", format_per_position(&pos_acc));

    let diag_inputs = dataset::diagnostic_set(config.seq_len, 10_000, config.split_seed)
        .inputs_tensor(&device);
    let state_prs = diagnostics::state_pr(&model, &diag_inputs);
    let weight_prs = diagnostics::weight_pr(&model, NUM_SYMBOLS);
    println!("{}", format_prs(&state_prs, &weight_prs));
    for s in diagnostics::state_pr_p_axis(&model, &diag_inputs) {
        println!(
            "state PR p-axis [L{}H{} pooled {:.2}/{:.2}c (m{:.1e}), final {:.2}/{:.2}c]",
            s.layer,
            s.head,
            s.pooled_uncentered,
            s.pooled_centered,
            s.pooled_trace,
            s.final_uncentered,
            s.final_centered,
        );
    }

    print_samples(config, &model, &test_split, &device, chunk);
}

/// Frontier-mode panel: fresh sampled sets at the in-range and extrapolation
/// lengths (same seeds as training's fixed eval sets), frontier curves, and
/// the PR diagnostics on the in-range set.
fn infer_frontier(config: &TrackingConfig, model: &MambaVocabNet, device: Device) {
    println!(
        "frontier mode (curriculum {}..[{}→{}]), chance ≈ {:.4}",
        config.min_len,
        config.max_len_start,
        config.max_len_end,
        1.0 / NUM_CLASSES as f64,
    );
    for (name, len, seed) in [
        ("in-range", config.max_len_end, config.split_seed ^ 0xE7A1_0000),
        ("extrap  ", config.eval_len, config.split_seed ^ 0xE7A2_0000),
    ] {
        let split = dataset::sample_split(256, len, seed);
        let chunk = Some(len + 1);
        let pos = per_position_accuracy(
            model,
            &split.inputs_tensor(&device),
            &split.pos_targets_tensor(&device),
            chunk,
        );
        let mean = pos.iter().sum::<f64>() / pos.len() as f64;
        println!("{name} (len {len:>3}): per-token acc {mean:.4}");
        println!("  frontier: {}", format_per_position(&pos));
        if name.trim() == "in-range" {
            let inputs = split.inputs_tensor(&device);
            let state_prs = diagnostics::state_pr(model, &inputs);
            let weight_prs = diagnostics::weight_pr(model, NUM_SYMBOLS);
            println!("{}", format_prs(&state_prs, &weight_prs));
            for s in diagnostics::state_pr_p_axis(model, &inputs) {
                println!(
                    "state PR p-axis [L{}H{} pooled {:.2}/{:.2}c (m{:.1e}), final {:.2}/{:.2}c]",
                    s.layer,
                    s.head,
                    s.pooled_uncentered,
                    s.pooled_centered,
                    s.pooled_trace,
                    s.final_uncentered,
                    s.final_centered,
                );
            }
            print_samples(config, model, &split, &device, chunk);
        }
    }
}

/// Print a handful of words with their predicted vs expected final product.
fn print_samples(
    config: &TrackingConfig,
    model: &MambaVocabNet,
    split: &dataset::Split,
    device: &Device,
    chunk: Option<usize>,
) {
    let sample = split.head(8);
    let logits_bv = protocol::final_logits(
        model,
        &sample.inputs_tensor(device),
        config.stepwise,
        chunk,
    );
    let [b, _vocab] = logits_bv.dims();
    let preds = logits_bv
        .narrow(1, dataset::CLASS_BASE, NUM_CLASSES)
        .argmax(1)
        .reshape([b])
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    let s = sample.tokens();
    for ((word, label), pred) in sample.seqs.chunks_exact(s).zip(&sample.labels).zip(&preds) {
        // skip the anchor token when printing the word
        let word_str: String = word[1..]
            .iter()
            .map(|g| char::from(b'a' + *g as u8))
            .collect();
        let mark = if pred == label { "✓" } else { "✗" };
        println!("  {word_str} ↦ class {pred:>2} [expected {label:>2}] {mark}");
    }
}
