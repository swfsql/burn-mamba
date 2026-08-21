//! Inference for the reset-spinor example: loads the trained model and reports
//! accuracy on each evaluation family, plus a decoded sample sequence.

use crate::AppArgs;
use crate::dataset::{
    EVAL_SEED, Family, NUM_CLASSES, NUM_EVAL, RESET, ResetSpinorBatcher, ResetSpinorDataset,
    ResetSpinorItem, SEQ_LENGTH, TURN_I, TURN_J,
};
use crate::training::{EVAL_FAMILIES, ssd_path};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use burn_mamba::prelude::*;

/// Load the trained model and report per-family accuracy on fresh eval sets.
pub fn infer(model_config: MambaLatentNetConfig, infer_device: Device, app_args: &AppArgs) {
    let model: MambaLatentNet = app_args
        .load_model(&model_config, &infer_device)
        .expect("failed to load model");
    let batcher = ResetSpinorBatcher::default();

    println!("chance ≈ {:.1}%", 100.0 / NUM_CLASSES as f32);
    for (name, family) in EVAL_FAMILIES {
        let items: Vec<ResetSpinorItem> =
            ResetSpinorDataset::new(NUM_EVAL, SEQ_LENGTH, family, EVAL_SEED)
                .iter()
                .map(|item| item.expect("dataset item"))
                .collect();
        let sample = items[0].clone();
        let batch = batcher.batch(items, &infer_device);
        let [batch_size, seq, _] = batch.inputs.dims();

        let (output, _caches) = model.forward(batch.inputs, None, ssd_path(), None);
        assert_eq!([batch_size, seq, NUM_CLASSES], output.dims());

        let pred = argmax_classes(output);
        let target = batch
            .targets
            .reshape([batch_size * seq])
            .into_data()
            .try_to_vec::<i32>()
            .unwrap();
        let correct = pred.iter().zip(&target).filter(|(p, t)| p == t).count();
        println!(
            "{name:<9} acc {:6.2}%   ({correct}/{})   per element {}",
            100.0 * correct as f32 / target.len() as f32,
            target.len(),
            per_class(&pred, &target),
        );

        if family == Family::Shuffle {
            println!("  sample     {}", render_symbols(&sample.symbols));
            println!("  target     {}", render_classes(&sample.targets));
            let shown: Vec<i64> = pred[..seq].iter().map(|&p| i64::from(p)).collect();
            println!("  predicted  {}", render_classes(&shown));
        }
    }
}

/// Accuracy split by target group element, rendered as `1 99% i 99% …`.
fn per_class(pred: &[i32], target: &[i32]) -> String {
    let mut hit = [0u64; NUM_CLASSES];
    let mut all = [0u64; NUM_CLASSES];
    for (p, t) in pred.iter().zip(target) {
        all[*t as usize] += 1;
        if p == t {
            hit[*t as usize] += 1;
        }
    }
    (0..NUM_CLASSES)
        .map(|c| {
            let pct = if all[c] == 0 {
                0.0
            } else {
                100.0 * hit[c] as f32 / all[c] as f32
            };
            format!("{} {pct:.0}%", ELEMENTS[c])
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn argmax_classes(output: Tensor<3>) -> Vec<i32> {
    let [batch, seq, classes] = output.dims();
    output
        .reshape([batch * seq, classes])
        .argmax(1)
        .reshape([batch * seq])
        .into_data()
        .try_to_vec::<i32>()
        .unwrap()
}

/// The eight classes, in index order (`unit + 4·negative`).
const ELEMENTS: [&str; NUM_CLASSES] = ["1", "i", "j", "k", "-1", "-i", "-j", "-k"];

fn render_symbols(symbols: &[usize]) -> String {
    symbols
        .iter()
        .map(|&s| match s {
            TURN_I => 'i',
            TURN_J => 'j',
            RESET => 'R',
            _ => '?',
        })
        .collect()
}

/// One char per class: `1 i j k` for the positives, upper-case (and `~` for
/// `−1`) for their negatives.
fn render_classes(classes: &[i64]) -> String {
    const CHARS: [char; NUM_CLASSES] = ['1', 'i', 'j', 'k', '~', 'I', 'J', 'K'];
    classes.iter().map(|&c| CHARS[c as usize]).collect()
}
