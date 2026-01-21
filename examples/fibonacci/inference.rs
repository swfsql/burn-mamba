use crate::AppArgs;
pub use crate::common::{
    backend::FloatElement,
    model::{Mamba2Network, Mamba2NetworkConfig},
};
use crate::dataset::{
    NOISE_LEVEL, SEQ_LENGTH, SequenceBatcher, SequenceDataset, SequenceDatasetItem,
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

pub fn infer<B: Backend>(
    model_config: Mamba2NetworkConfig,
    batch_size: usize,
    infer_device: B::Device,
    app_args: &AppArgs,
) {
    // load model
    let model: Mamba2Network<B> = app_args
        .load_model(&model_config, &infer_device)
        .expect("failed to load model");

    let dataset = SequenceDataset::new(batch_size, SEQ_LENGTH, NOISE_LEVEL);
    let items: Vec<SequenceDatasetItem> = dataset.iter().collect();

    let batcher = SequenceBatcher::default();
    // Put all items in one batch
    let batch = batcher.batch(items, &infer_device);
    let (predicted, _caches) = model.forward(batch.sequences, None, Some(4));
    assert_eq!([batch_size, SEQ_LENGTH + 1, 1], predicted.dims());
    let last_predicted = predicted.narrow(1, SEQ_LENGTH, 1);
    assert_eq!([batch_size, 1, 1], last_predicted.dims());
    let targets = batch.targets;
    assert_eq!([batch_size, 1], targets.dims());

    // Display the predicted vs expected values
    let show = 10;
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
                .to_vec::<FloatElement>()
                .unwrap()[0],
            expected
                .clone()
                .narrow(0, i, 1)
                .into_data()
                .to_vec::<FloatElement>()
                .unwrap()[0]
        );
    }
}
