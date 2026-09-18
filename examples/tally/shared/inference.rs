//! Inference: load the trained model and report accuracy per family (overall
//! and per class), with one decoded sample.

use super::data::{EVAL_SEED, IGNORE, NUM_CLASSES, NUM_EVAL, TallyBatcher, TallyDataset, Task};
use super::handmade::predictions;
use crate::common::cli::AppArgs;
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use burn_mamba::prelude::*;

/// Load the trained model and report per-family accuracy on fresh eval sets.
pub fn infer(task: &Task, model_config: MambaLatentNetConfig, device: Device, app_args: &AppArgs) {
    let model: MambaLatentNet = app_args
        .load_model(&model_config, &device)
        .expect("failed to load model");
    let batcher = TallyBatcher {
        num_symbols: task.num_symbols,
    };
    println!("chance ≈ {:.1}%", 100.0 / NUM_CLASSES as f32);
    for (i, (name, generator)) in task.families.iter().enumerate() {
        let items = TallyDataset::new(task, *generator, NUM_EVAL, EVAL_SEED).items();
        let sample = items[0].clone();
        let pred = predictions(&model, &batcher.batch(items.clone(), &device).inputs);
        let target: Vec<i64> = items.iter().flat_map(|i| i.targets.clone()).collect();
        let scored: Vec<(i64, i64)> = pred
            .iter()
            .zip(&target)
            .filter(|(_, t)| **t != IGNORE)
            .map(|(p, t)| (*p, *t))
            .collect();
        println!("{name:<12} {}", summary(task, &scored));
        if i == 0 {
            let seq = task.seq_length;
            let render = |classes: &[i64]| -> String {
                classes
                    .iter()
                    .map(|&c| if c == IGNORE { '.' } else { task.class_glyphs[c as usize] })
                    .collect()
            };
            let shown: Vec<i64> = pred[..seq]
                .iter()
                .zip(&sample.targets)
                .map(|(&p, &t)| if t == IGNORE { IGNORE } else { p })
                .collect();
            let symbols: String = sample.symbols.iter().map(|&s| task.glyphs[s]).collect();
            println!("  sample     {symbols}");
            println!("  target     {}", render(&sample.targets));
            println!("  predicted  {}", render(&shown));
        }
    }
}

/// `acc 99.00% (n/m)  class-a 99% class-b 99%`.
pub fn summary(task: &Task, scored: &[(i64, i64)]) -> String {
    let mut hit = [0u64; NUM_CLASSES];
    let mut all = [0u64; NUM_CLASSES];
    for (p, t) in scored {
        all[*t as usize] += 1;
        hit[*t as usize] += u64::from(p == t);
    }
    let correct: u64 = hit.iter().sum();
    let per_class: Vec<String> = (0..NUM_CLASSES)
        .map(|c| {
            let pct = if all[c] == 0 { 0.0 } else { 100.0 * hit[c] as f64 / all[c] as f64 };
            format!("{} {pct:.0}%", task.class_names[c])
        })
        .collect();
    format!(
        "acc {:6.2}%   ({correct}/{})   {}",
        100.0 * correct as f64 / scored.len().max(1) as f64,
        scored.len(),
        per_class.join(" ")
    )
}
