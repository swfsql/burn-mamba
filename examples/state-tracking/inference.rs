//! Inference for the state-tracking example: loads the trained model and
//! reports the per-token accuracy on a fresh evaluation set — the headline
//! number contrasting the abelian (`Complex2D`) and non-abelian
//! (`Quaternion4D`) rotations.

use crate::AppArgs;
pub use crate::common::model::{MyMamba3Network, MyMamba3NetworkConfig};
use crate::dataset::{
    EVAL_SEED, NUM_CLASSES, NUM_EVAL, SEQ_LENGTH, StateTrackingBatcher, StateTrackingDataset,
    StateTrackingItem,
};
use crate::training::{accumulate_per_position, format_per_position};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use burn_mamba::prelude::*;

/// Load the trained model and report per-token (and per-position) accuracy on a
/// fresh eval set.
pub fn infer(model_config: MyMamba3NetworkConfig, infer_device: Device, app_args: &AppArgs) {
    let rotation = model_config.layers.mamba_block.rotation;

    // load model
    let model: MyMamba3Network = app_args
        .load_model(&model_config, &infer_device)
        .expect("failed to load model");

    let dataset = StateTrackingDataset::new(NUM_EVAL, SEQ_LENGTH, EVAL_SEED);
    let items: Vec<StateTrackingItem> = dataset.iter().collect();

    let batcher = StateTrackingBatcher::default();
    // Put all items in one batch.
    let batch = batcher.batch(items, &infer_device);
    let [batch_size, sequence_size, _num_sym] = batch.inputs.dims();

    let (output, _caches) = model.forward(batch.inputs, None, Mamba3SsdPath::Minimal(None));
    assert_eq!([batch_size, sequence_size, NUM_CLASSES], output.dims());

    // Per-position accuracy: early positions have few distinct input prefixes
    // (≤ NUM_GENERATORS^t) so even an abelian model memorises them; the gap shows
    // up in the deeper positions, where genuine A₅ composition is required.
    let mut correct: Vec<u64> = Vec::new();
    let mut total: Vec<u64> = Vec::new();
    accumulate_per_position(
        output.reshape([batch_size * sequence_size, NUM_CLASSES]),
        batch.targets.reshape([batch_size * sequence_size]),
        sequence_size,
        &mut correct,
        &mut total,
    );

    let overall = correct.iter().sum::<u64>() as f32 / total.iter().sum::<u64>() as f32;
    println!(
        "Eval per-token accuracy ({rotation:?}): {:.1}%  (chance ≈ {:.1}%)",
        overall * 100.0,
        100.0 / NUM_CLASSES as f32,
    );
    println!(
        "per-position acc: {}",
        format_per_position(&correct, &total)
    );
}
