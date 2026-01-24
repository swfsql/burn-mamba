pub use crate::common::{
    cli::AppArgs,
    model::{MyMamba2Network, MyMamba2NetworkConfig},
    training::TrainingConfig,
};
use crate::dataset::{
    NOISE_LEVEL, NUM_SEQUENCES, SEQ_LENGTH, SequenceBatch, SequenceBatcher, SequenceDataset,
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::AutodiffModule,
    nn::loss::Reduction,
    optim::{AdamW, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::backend::AutodiffBackend,
    train::RegressionOutput,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
};

pub fn train<AutoB: AutodiffBackend>(
    training_config: TrainingConfig,
    model_config: MyMamba2NetworkConfig,
    training_device: AutoB::Device,
    app_args: &AppArgs,
) {
    AutoB::seed(&training_device, training_config.seed);

    // load (or init and save) model and optim
    let model: MyMamba2Network<AutoB> =
        app_args.load_or_save_model(&model_config, &training_device);
    let mut optim = app_args.load_or_save_optim(&training_config.optimizer, &training_device);

    let mut model = Wrap(model, model_config.clone());

    // Create the batcher
    let batcher = SequenceBatcher::default();

    // Create the dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(SequenceDataset::new(NUM_SEQUENCES, SEQ_LENGTH, NOISE_LEVEL));
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        // 20% size of training
        .build(SequenceDataset::new(
            NUM_SEQUENCES / 5,
            SEQ_LENGTH,
            NOISE_LEVEL,
        ));

    let training_num_items = dataloader_train.num_items();

    let mut metric_meta = MetricMetadata {
        progress: burn::data::dataloader::Progress::new(0, training_num_items),
        epoch: 1,
        epoch_total: training_config.num_epochs,
        iteration: 0,
        lr: Some(training_config.lr),
    };

    println!("running small initial validation...");
    epoch_valid::<AutoB::InnerBackend>(
        std::sync::Arc::clone(&dataloader_valid),
        model.0.valid(),
        &training_config,
        &model_config,
        metric_meta.epoch,
        Some(10),
    );

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in 1..training_config.num_epochs + 1 {
        metric_meta.epoch = epoch;

        model.0 = epoch_train::<AutoB>(
            std::sync::Arc::clone(&dataloader_train),
            std::sync::Arc::clone(&dataloader_valid),
            model.0,
            &training_config,
            &model_config,
            &mut optim,
            &mut metric_meta,
            None,
            Some(10),
            app_args,
        );

        // save assets
        app_args.save_model(&model.0);
        app_args.save_optim(&optim);

        let valid_loop_limit = Some(10);
        println!("running validation (batch iteration limit: {valid_loop_limit:?})");
        epoch_valid::<AutoB::InnerBackend>(
            std::sync::Arc::clone(&dataloader_valid),
            model.0.valid(),
            &training_config,
            &model_config,
            metric_meta.epoch,
            valid_loop_limit, // validate against 10 batches
        );
    }
    println!("Training finished.");
}

type Dataloader<B> = std::sync::Arc<dyn DataLoader<B, SequenceBatch<B>> + 'static>;

pub fn epoch_train<B: AutodiffBackend>(
    dataloader_train: Dataloader<B>,
    dataloader_valid: Dataloader<B::InnerBackend>,
    training_model: MyMamba2Network<B>,
    training_config: &TrainingConfig,
    model_config: &MyMamba2NetworkConfig,
    optim: &mut OptimizerAdaptor<AdamW, MyMamba2Network<B>, B>,
    metric_meta: &mut MetricMetadata,
    training_loop_limit: Option<usize>,
    valid_loop_limit: Option<usize>,
    app_args: &AppArgs,
) -> MyMamba2Network<B> {
    let training_loop_limit = training_loop_limit.unwrap_or(usize::MAX);
    let mut loss_metric = burn::train::metric::LossMetric::<B>::new();

    let mut training_model = Wrap(training_model, model_config.clone());

    // training loop
    for (mut b, batch) in dataloader_train
        .iter()
        .enumerate()
        .take(training_loop_limit)
    {
        b += 1;
        let input = batch.sequences;
        let targets = batch.targets;
        let [batch_size, sequence_size, _input_size] = input.dims();
        assert_eq!([batch_size, 1], targets.dims());
        assert!(sequence_size >= 1);

        metric_meta.iteration += 1;
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = training_model.forward_regression(input, targets);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);

        let loss = pre_metrics.loss.clone();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &training_model.0);
        training_model.0 = optim.step(training_config.lr, training_model.0, grads);

        println!(
            "Epoch {}/{}, Batch {b}/{}, Loss {:.4}",
            metric_meta.epoch,
            metric_meta.epoch_total,
            dataloader_train.num_items() / training_config.batch_size + 1,
            loss_metric.value().current(),
        );

        if b % 10 == 0 {
            // save assets
            app_args.save_model(&training_model.0);
            app_args.save_optim(optim);

            println!("running validation (batch iteration limit: {valid_loop_limit:?})");
            epoch_valid::<B::InnerBackend>(
                std::sync::Arc::clone(&dataloader_valid),
                training_model.0.valid(),
                &training_config,
                &model_config,
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

    training_model.0
}

pub fn epoch_valid<B: Backend>(
    dataloader_valid: Dataloader<B>,
    valid_model: MyMamba2Network<B>,
    training_config: &TrainingConfig,
    model_config: &MyMamba2NetworkConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
) {
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = burn::train::metric::MetricMetadata {
        progress: burn::data::dataloader::Progress::new(0, valid_num_items),
        epoch,
        epoch_total: training_config.num_epochs,
        iteration: 0,
        lr: Some(training_config.lr),
    };

    let mut loss_metric = burn::train::metric::LossMetric::<B>::new();

    let valid_model = Wrap(valid_model, model_config.clone());

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
pub struct Wrap<B: Backend>(pub MyMamba2Network<B>, pub MyMamba2NetworkConfig);

impl<B: Backend> Wrap<B> {
    pub fn forward_regression(
        &self,
        input: Tensor<B, 3>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let model = &self.0;
        let _model_config = &self.1;
        let [batch_size, sequence_size, _input_size] = input.dims();
        assert_eq!([batch_size, 1], targets.dims());
        assert!(sequence_size >= 1);

        let (output, _caches) = model.forward(input.clone(), None, Some(4));
        assert_eq!([batch_size, sequence_size, 1], output.dims());
        let last_output = output.narrow(1, sequence_size - 1, 1).squeeze_dim(1);
        assert_eq!([batch_size, 1], last_output.dims());

        let loss = burn_mamba::utils::loss::mse::MseLoss::new().forward(
            last_output.clone(),
            targets.clone(),
            Reduction::Mean,
        );

        RegressionOutput::new(loss.clone(), last_output, targets)
    }
}
