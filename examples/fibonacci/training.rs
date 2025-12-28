use crate::common::model::{Mamba2Network, Mamba2NetworkConfig};
use crate::common::training::{TrainingConfig, create_artifact_dir};
use crate::dataset::{
    NOISE_LEVEL, NUM_SEQUENCES, SEQ_LENGTH, SequenceBatch, SequenceBatcher, SequenceDataset,
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::AutodiffModule,
    nn::loss::Reduction,
    optim::{AdamW, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::RegressionOutput,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
};
use burn_mamba::prelude::*;

pub const RANDOM_SEED: u64 = 5;

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Save training config
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(&device, RANDOM_SEED);

    // Create the model and optimizer
    let mut model = Wrap(config.model.init::<B>(&device), config.model.clone());
    let mut optim = config.optimizer.init::<B, Mamba2Network<B>>();

    // Create the batcher
    let batcher = SequenceBatcher::default();

    // Create the dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(RANDOM_SEED)
        .num_workers(config.num_workers)
        .build(SequenceDataset::new(NUM_SEQUENCES, SEQ_LENGTH, NOISE_LEVEL));
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(RANDOM_SEED)
        .num_workers(config.num_workers)
        // 20% size of training
        .build(SequenceDataset::new(
            NUM_SEQUENCES / 5,
            SEQ_LENGTH,
            NOISE_LEVEL,
        ));

    let train_num_items = dataloader_train.num_items();

    let mut metric_meta = MetricMetadata {
        progress: burn::data::dataloader::Progress::new(0, train_num_items),
        epoch: 1,
        epoch_total: config.num_epochs,
        iteration: 0,
        lr: Some(config.lr),
    };

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in 1..config.num_epochs + 1 {
        metric_meta.epoch = epoch;

        model.0 = epoch_train::<B>(
            artifact_dir,
            std::sync::Arc::clone(&dataloader_train),
            std::sync::Arc::clone(&dataloader_valid),
            model.0,
            &config,
            &mut optim,
            &mut metric_meta,
            None,
            Some(10),
        );
    }
    println!("Training finished.");

    println!("Saving model and running full validation...");
    // Save the trained model
    model
        .0
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
    println!("Model saved");

    // run validation
    epoch_valid::<B::InnerBackend>(
        std::sync::Arc::clone(&dataloader_valid),
        model.0.valid(),
        &config,
        config.num_epochs,
        Some(10), // validate against 10 batches
    );
}

type Dataloader<B> = std::sync::Arc<dyn DataLoader<B, SequenceBatch<B>> + 'static>;

pub fn epoch_train<B: AutodiffBackend>(
    artifact_dir: &str,
    dataloader_train: Dataloader<B>,
    dataloader_valid: Dataloader<B::InnerBackend>,
    train_model: Mamba2Network<B>,
    config: &TrainingConfig,
    optim: &mut OptimizerAdaptor<AdamW, Mamba2Network<B>, B>,
    metric_meta: &mut MetricMetadata,
    train_loop_limit: Option<usize>,
    valid_loop_limit: Option<usize>,
) -> Mamba2Network<B> {
    let train_loop_limit = train_loop_limit.unwrap_or(usize::MAX);
    let mut loss_metric = burn::train::metric::LossMetric::<B>::new();

    let mut train_model = Wrap(train_model, config.model.clone());

    // training loop
    for (mut b, batch) in dataloader_train.iter().enumerate().take(train_loop_limit) {
        b += 1;
        let input = batch.sequences;
        let targets = batch.targets;
        let [batch_size, sequence_size, _input_size] = input.dims();
        assert_eq!([batch_size, 1], targets.dims());
        assert!(sequence_size >= 1);

        metric_meta.iteration += 1;
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = train_model.forward_regression(input, targets);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);

        let loss = pre_metrics.loss.clone();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &train_model.0);
        train_model.0 = optim.step(config.lr, train_model.0, grads);

        println!(
            "Epoch {}/{}, Batch {b}/{}, Loss {:.4}",
            metric_meta.epoch,
            metric_meta.epoch_total,
            dataloader_train.num_items() / config.batch_size + 1,
            loss_metric.value().current(),
        );

        if b % 10 == 0 {
            println!("Saving model and running validation (batch limit: {valid_loop_limit:?})");
            train_model
                .0
                .clone()
                .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
                .expect("Trained model should be saved successfully");
            epoch_valid::<B::InnerBackend>(
                std::sync::Arc::clone(&dataloader_valid),
                train_model.0.valid(),
                &config,
                metric_meta.epoch,
                valid_loop_limit,
            );
        }
    }

    // Display the averaged training metrics
    println!(
        "Epoch {}/{}, Avg Loss {:.4}",
        metric_meta.epoch,
        metric_meta.epoch_total,
        loss_metric.running_value().current(),
    );

    train_model.0
}

pub fn epoch_valid<B: Backend>(
    dataloader_valid: Dataloader<B>,
    valid_model: Mamba2Network<B>,
    config: &TrainingConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
) {
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = burn::train::metric::MetricMetadata {
        progress: burn::data::dataloader::Progress::new(0, valid_num_items),
        epoch,
        epoch_total: config.num_epochs,
        iteration: 0,
        lr: Some(config.lr),
    };

    let mut loss_metric = burn::train::metric::LossMetric::<B>::new();

    let valid_model = Wrap(valid_model, config.model.clone());

    // validation loop
    for (_b, batch) in dataloader_valid.iter().enumerate().take(valid_loop_limit) {
        let input = batch.sequences;
        let targets = batch.targets;
        let [batch_size, sequence_size, _input_size] = input.dims();
        assert_eq!([batch_size, 1], targets.dims());
        assert!(sequence_size >= 1);

        metric_meta.iteration += 1;
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = valid_model.forward_regression(input, targets);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
    }

    // Display the averaged validation metrics
    println!(
        "Epoch {}/{}, Avg Valid Loss {:.4}",
        metric_meta.epoch,
        metric_meta.epoch_total,
        loss_metric.running_value().current(),
    );
}

/// Wrapper over [`Mamba2Network`] for custom implementations.
pub struct Wrap<B: Backend>(pub Mamba2Network<B>, pub Mamba2NetworkConfig);

impl<B: Backend> Wrap<B> {
    pub fn forward_regression(
        &self,
        input: Tensor<B, 3>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let model = &self.0;
        let config = &self.1;
        let [batch_size, sequence_size, _input_size] = input.dims();
        assert_eq!([batch_size, 1], targets.dims());
        assert!(sequence_size >= 1);

        let device = model.in_proj.weight.device();

        // prepare caches (once per batch)
        let caches = Mamba2BlockCachesConfig::new_from_block_config(
            config.n_layer,
            batch_size,
            config.mamba_block.clone(),
        )
        .init(&device);

        let (output, _caches) = model.forward(input.clone(), caches, Some(4));
        assert_eq!([batch_size, sequence_size, 1], output.dims());
        let last_output = output.narrow(1, sequence_size - 1, 1).squeeze_dim(1);
        assert_eq!([batch_size, 1], last_output.dims());

        let loss = burn_mamba::mse::MseLoss::new().forward(
            last_output.clone(),
            targets.clone(),
            Reduction::Mean,
        );

        RegressionOutput::new(loss.clone(), last_output, targets)
    }
}
