//! Inference for the reset-majority example: loads the trained model and reports
//! accuracy on each evaluation family, plus a decoded sample sequence.

use crate::AppArgs;
use crate::dataset::{
    EVAL_SEED, Family, IGNORE, MINUS, NUM_CLASSES, NUM_EVAL, PLUS, RESET, ResetMajorityBatcher,
    ResetMajorityDataset, ResetMajorityItem, SEQ_LENGTH,
};
use crate::training::EVAL_FAMILIES;
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
    let batcher = ResetMajorityBatcher::default();

    println!("chance ≈ {:.1}%", 100.0 / NUM_CLASSES as f32);
    for (name, family) in EVAL_FAMILIES {
        let items: Vec<ResetMajorityItem> =
            ResetMajorityDataset::new(NUM_EVAL, SEQ_LENGTH, family, EVAL_SEED)
                .iter()
                .map(|item| item.expect("dataset item"))
                .collect();
        let sample = items[0].clone();
        let batch = batcher.batch(items, &infer_device);
        let [batch_size, seq, _] = batch.inputs.dims();

        let (output, _caches) = model.forward(
            batch.inputs,
            None,
            MambaSsdPath::Mamba2(Mamba2SsdPath::Minimal(None)),
            None,
        );
        assert_eq!([batch_size, seq, NUM_CLASSES], output.dims());

        let pred = argmax_classes(output);
        let target = batch
            .targets
            .reshape([batch_size * seq])
            .into_data()
            .try_to_vec::<i32>()
            .unwrap();
        // Only the positions with a sign to report are scored (`dataset::IGNORE`).
        let scored: Vec<(i32, i32)> = pred
            .iter()
            .zip(&target)
            .filter(|(_, t)| i64::from(**t) != IGNORE)
            .map(|(p, t)| (*p, *t))
            .collect();
        let correct = scored.iter().filter(|(p, t)| p == t).count();
        println!(
            "{name:<12} acc {:6.2}%   ({correct}/{})   per class {}",
            100.0 * correct as f32 / scored.len() as f32,
            scored.len(),
            per_class(&scored),
        );

        if family == Family::LongPrefix {
            println!("  sample     {}", render_symbols(&sample.symbols));
            println!("  target     {}", render_classes(&sample.targets));
            // unscored positions are blanked out, so what shows is what counts
            let shown: Vec<i64> = pred[..seq]
                .iter()
                .zip(&sample.targets)
                .map(|(&p, &t)| if t == IGNORE { IGNORE } else { i64::from(p) })
                .collect();
            println!("  predicted  {}", render_classes(&shown));
        }
    }
}

/// Accuracy split by target class, rendered as `neg 99% pos 99%`.
fn per_class(scored: &[(i32, i32)]) -> String {
    let names = ["neg", "pos"];
    let mut hit = [0u64; NUM_CLASSES];
    let mut all = [0u64; NUM_CLASSES];
    for (p, t) in scored {
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
            format!("{} {pct:.0}%", names[c])
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

fn render_symbols(symbols: &[usize]) -> String {
    symbols
        .iter()
        .map(|&s| match s {
            MINUS => '-',
            PLUS => '+',
            RESET => 'R',
            _ => '?',
        })
        .collect()
}

fn render_classes(classes: &[i64]) -> String {
    classes
        .iter()
        .map(|&c| match c {
            0 => 'n',
            1 => 'p',
            _ => '.',
        })
        .collect()
}
