use crate::common::backend::Element;
use crate::{
    common::model::Mamba2Network,
    common::training::TrainingConfig,
    dataset::{NOISE_LEVEL, SEQ_LENGTH, SequenceBatcher, SequenceDataset, SequenceDatasetItem},
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use burn_mamba::prelude::Mamba2BlockCachesConfig;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    // Loading model
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model: Mamba2Network<B> = config.model.init(&device).load_record(record);

    let dataset = SequenceDataset::new(config.batch_size, SEQ_LENGTH, NOISE_LEVEL);
    let items: Vec<SequenceDatasetItem> = dataset.iter().collect();

    let batcher = SequenceBatcher::default();
    // Put all items in one batch
    let batch = batcher.batch(items, &device);
    let caches = Mamba2BlockCachesConfig::new_from_block_config(
        config.model.n_layer,
        config.batch_size,
        config.model.mamba_block,
    )
    .init(&device);
    let (predicted, _caches) = model.forward(batch.sequences, caches, Some(4));
    assert_eq!([config.batch_size, SEQ_LENGTH + 1, 1], predicted.dims());
    let last_predicted = predicted.narrow(1, SEQ_LENGTH, 1);
    assert_eq!([config.batch_size, 1, 1], last_predicted.dims());
    let targets = batch.targets;
    assert_eq!([config.batch_size, 1], targets.dims());

    // Display the predicted vs expected values
    let show = config.batch_size.min(10);
    let last_predicted = last_predicted.squeeze_dims::<1>(&[1, 2]).narrow(0, 0, show);
    let expected = targets.squeeze_dim::<1>(1).narrow(0, 0, show);
    println!("predicted/expected:");
    for i in 0..show {
        println!(
            "- {:04.2?}/{:04.2?}",
            last_predicted
                .clone()
                .narrow(0, i, 1)
                .into_data()
                .to_vec::<Element>()
                .unwrap()[0],
            expected
                .clone()
                .narrow(0, i, 1)
                .into_data()
                .to_vec::<Element>()
                .unwrap()[0]
        );
    }
}
