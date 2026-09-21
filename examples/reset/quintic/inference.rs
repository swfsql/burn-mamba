//! Inference for the reset-quintic example: loads the trained model and reports
//! accuracy on each evaluation family, plus a decoded sample sequence.

use crate::AppArgs;
use crate::dataset::{
    EVAL_LENGTHS, EVAL_SEED, Family, Group, NUM_EVAL, QuinticBatcher, QuinticDataset, QuinticItem,
    SEQ_LENGTH,
};
use crate::training::{EVAL_FAMILIES, ssd_path};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use burn_mamba::prelude::*;

/// Load the trained model and report per-family accuracy on fresh eval sets.
pub fn infer(
    group: Group,
    model_config: MambaLatentNetConfig,
    infer_device: Device,
    app_args: &AppArgs,
) {
    let model: MambaLatentNet = app_args
        .load_model(&model_config, &infer_device)
        .expect("failed to load model");
    let batcher = QuinticBatcher::default();
    let num_classes = group.num_classes();

    println!("chance ≈ {:.2}%", 100.0 / num_classes as f32);
    for length in EVAL_LENGTHS {
        println!("— {length} symbols —");
        for (name, family) in EVAL_FAMILIES {
            let items: Vec<QuinticItem> =
                QuinticDataset::new(group, NUM_EVAL, length, family, EVAL_SEED)
                    .iter()
                    .map(|item| item.expect("dataset item"))
                    .collect();
            let sample = items[0].clone();
            let batch = batcher.batch(items, &infer_device);
            let [batch_size, seq, _] = batch.inputs.dims();

            let (output, _caches) = model.forward(batch.inputs, None, ssd_path(), None, None);
            assert_eq!([batch_size, seq, num_classes], output.dims());

            let pred = argmax_classes(output);
            let target = batch
                .targets
                .reshape([batch_size * seq])
                .into_data()
                .try_to_vec::<i32>()
                .unwrap();
            let correct = pred.iter().zip(&target).filter(|(p, t)| p == t).count();
            let (worst, seen) = worst_class(&pred, &target, num_classes);
            println!(
                "{name:<9} acc {:6.2}%   ({correct}/{})   worst element {worst:.0}% ({seen}/{num_classes} elements seen)",
                100.0 * correct as f32 / target.len() as f32,
                target.len(),
            );

            if family == Family::Shuffle && length == SEQ_LENGTH {
                let symbols: String =
                    sample.symbols.iter().map(|&s| group.symbol_char(s)).collect();
                let misses: String = pred[..seq]
                    .iter()
                    .zip(&target[..seq])
                    .map(|(p, t)| if p == t { ' ' } else { '^' })
                    .collect();
                println!("  sample     {symbols}");
                println!("  wrong at   {misses}");
            }
        }
    }
}

/// The lowest per-element accuracy, over the elements that occur at all, and
/// how many do.
fn worst_class(pred: &[i32], target: &[i32], num_classes: usize) -> (f32, usize) {
    let mut hit = vec![0u64; num_classes];
    let mut all = vec![0u64; num_classes];
    for (p, t) in pred.iter().zip(target) {
        all[*t as usize] += 1;
        if p == t {
            hit[*t as usize] += 1;
        }
    }
    let seen = all.iter().filter(|&&n| n > 0).count();
    let worst = (0..num_classes)
        .filter(|&c| all[c] > 0)
        .map(|c| 100.0 * hit[c] as f32 / all[c] as f32)
        .fold(100.0f32, f32::min);
    (worst, seen)
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
