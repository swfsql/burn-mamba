pub use crate::common::{
    cli::AppArgs,
    mnist::dataset::{HEIGHT, MnistBatch, MnistBatcher, MnistDataset, WIDTH},
    model::{MyMamba2Network, MyMamba2NetworkConfig},
    training::TrainingConfig,
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    module::AutodiffModule,
    optim::{AdamW, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::backend::AutodiffBackend,
    train::ClassificationOutput,
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
    let batcher = MnistBatcher::default();

    // Create the dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(MnistDataset::train());
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .num_workers(training_config.num_workers)
        .build(MnistDataset::test());

    let training_num_items = dataloader_train.num_items();

    let mut metric_meta = burn::train::metric::MetricMetadata {
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

        println!("running full validation...");
        epoch_valid::<AutoB::InnerBackend>(
            std::sync::Arc::clone(&dataloader_valid),
            model.0.valid(),
            &training_config,
            &model_config,
            metric_meta.epoch,
            None,
        );
    }
    println!("Training finished.");
}

type Dataloader<B> = std::sync::Arc<dyn DataLoader<B, MnistBatch<B>> + 'static>;

pub fn epoch_train<AutoB: AutodiffBackend>(
    dataloader_train: Dataloader<AutoB>,
    dataloader_valid: Dataloader<AutoB::InnerBackend>,
    training_model: MyMamba2Network<AutoB>,
    training_config: &TrainingConfig,
    model_config: &MyMamba2NetworkConfig,
    optim: &mut OptimizerAdaptor<AdamW, MyMamba2Network<AutoB>, AutoB>,
    metric_meta: &mut MetricMetadata,
    training_loop_limit: Option<usize>,
    valid_loop_limit: Option<usize>,
    app_args: &AppArgs,
) -> MyMamba2Network<AutoB> {
    let training_loop_limit = training_loop_limit.unwrap_or(usize::MAX);
    let mut loss_metric = burn::train::metric::LossMetric::<AutoB>::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::<AutoB>::new();

    let mut training_model = Wrap(training_model, model_config.clone());

    // training loop
    for (mut b, batch) in dataloader_train
        .iter()
        .enumerate()
        .take(training_loop_limit)
    {
        b += 1;
        let input = batch.images_z_score(); // values mean=0, stddev=1
        let [batch_size, HEIGHT, WIDTH, 1] = input.dims() else {
            panic!()
        };
        let input = input.reshape([batch_size, HEIGHT * WIDTH, 1]);
        let [_batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(sequence_size, HEIGHT * WIDTH);
        assert_eq!(input_size, 1);
        let targets = batch.targets;

        metric_meta.iteration += 1;
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = training_model.forward_classification(input, targets);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);

        let loss = pre_metrics.loss.clone();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &training_model.0);
        training_model.0 = optim.step(training_config.lr, training_model.0, grads);

        println!(
            "Epoch {}/{}, Batch {b:0>4}/{}, Loss {:.4}, Acc {:0>6.2}",
            metric_meta.epoch,
            metric_meta.epoch_total,
            dataloader_train.num_items() / training_config.batch_size + 1,
            loss_metric.value().current(),
            acc_metric.value().current(),
        );

        if b % 100 == 0 {
            // save assets
            app_args.save_model(&training_model.0);
            app_args.save_optim(optim);

            println!("running validation (batch iteration limit: {valid_loop_limit:?})");
            epoch_valid::<AutoB::InnerBackend>(
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
        "Epoch {}/{}, Avg Loss {:.4}, Avg Acc: {}",
        metric_meta.epoch,
        metric_meta.epoch_total,
        loss_metric.running_value().current(),
        acc_metric.running_value().current(),
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
    let mut acc_metric = burn::train::metric::AccuracyMetric::<B>::new();

    let valid_model = Wrap(valid_model, model_config.clone());

    // validation loop
    for (_b, batch) in dataloader_valid.iter().enumerate().take(valid_loop_limit) {
        let input = batch.images_z_score(); // values mean=0, stddev=1
        let [batch_size, HEIGHT, WIDTH, 1] = input.dims() else {
            panic!()
        };
        let input = input.reshape([batch_size, HEIGHT * WIDTH, 1]);
        let [_batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(sequence_size, HEIGHT * WIDTH);
        assert_eq!(input_size, 1);
        let targets = batch.targets;

        metric_meta.iteration += 1;
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = valid_model.forward_classification(input, targets);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
    }

    // Display the averaged validation metrics
    println!(
        "Epoch {}/{}, Avg Valid Loss {:.4}, Avg Valid Acc: {}",
        metric_meta.epoch,
        metric_meta.epoch_total,
        loss_metric.running_value().current(),
        acc_metric.running_value().current(),
    );
}

/// Wrapper over [`MyMamba2Network`] for custom implementations.
pub struct Wrap<B: Backend>(pub MyMamba2Network<B>, pub MyMamba2NetworkConfig);

impl<B: Backend> Wrap<B> {
    pub fn forward_classification(
        &self,
        input: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let model = &self.0;
        let _config = &self.1;
        let [batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(sequence_size, HEIGHT * WIDTH);
        assert_eq!(input_size, 1);
        assert_eq!([batch_size], targets.dims());

        let (output, _caches) = model.forward(input.clone(), None, Some(128));
        assert_eq!([batch_size, sequence_size, 10], output.dims());
        let last_output = output.narrow(1, sequence_size - 1, 1).squeeze_dim(1);
        assert_eq!([batch_size, 10], last_output.dims());

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&last_output.device())
            .forward(last_output.clone(), targets.clone());

        ClassificationOutput::new(loss.clone(), last_output, targets)
    }
}
