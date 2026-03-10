pub use crate::common::{
    cli::AppArgs,
    mnist::dataset::{HEIGHT, MnistBatch, MnistBatcher, MnistDataset, WIDTH},
    model::{MyMamba2Network, MyMamba2NetworkConfig},
    training::TrainingConfig,
};
use crate::model::{AutoEncoderNetwork, AutoEncoderNetworkConfig};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    module::AutodiffModule,
    optim::{AdamW, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::backend::AutodiffBackend,
    train::RegressionOutput,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
};

pub fn train<AutoB: AutodiffBackend>(
    training_config: TrainingConfig,
    model_config: AutoEncoderNetworkConfig,
    training_device: AutoB::Device,
    app_args: &AppArgs,
) {
    AutoB::seed(&training_device, training_config.seed);

    // load (or init and save) model and optim
    let model: AutoEncoderNetwork<AutoB> =
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
    let global_training_num_items = training_num_items * training_config.num_epochs;

    let mut metric_meta = burn::train::metric::MetricMetadata {
        progress: Progress::new(0, training_num_items),
        global_progress: Progress::new(0, global_training_num_items),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0)),
    };

    println!("running small initial validation...");
    epoch_valid::<AutoB::InnerBackend>(
        std::sync::Arc::clone(&dataloader_valid),
        model.0.valid(),
        &training_config,
        &model_config,
        0,
        Some(10),
    );

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in 1..training_config.num_epochs + 1 {
        model.0 = epoch_train::<AutoB>(
            std::sync::Arc::clone(&dataloader_train),
            std::sync::Arc::clone(&dataloader_valid),
            model.0,
            &training_config,
            &model_config,
            &mut optim,
            &mut metric_meta,
            epoch,
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
            epoch,
            None,
        );
    }
    println!("Training finished.");
}

type Dataloader<B> = std::sync::Arc<dyn DataLoader<B, MnistBatch<B>> + 'static>;

pub fn epoch_train<AutoB: AutodiffBackend>(
    dataloader_train: Dataloader<AutoB>,
    dataloader_valid: Dataloader<AutoB::InnerBackend>,
    training_model: AutoEncoderNetwork<AutoB>,
    training_config: &TrainingConfig,
    model_config: &AutoEncoderNetworkConfig,
    optim: &mut OptimizerAdaptor<AdamW, AutoEncoderNetwork<AutoB>, AutoB>,
    metric_meta: &mut MetricMetadata,
    epoch: usize,
    training_loop_limit: Option<usize>,
    valid_loop_limit: Option<usize>,
    app_args: &AppArgs,
) -> AutoEncoderNetwork<AutoB> {
    let training_loop_limit = training_loop_limit.unwrap_or(usize::MAX);
    let mut loss_metric = burn::train::metric::LossMetric::<AutoB>::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

    let mut training_model = Wrap(training_model, model_config.clone());

    // training loop
    for (mut b, batch) in dataloader_train
        .iter()
        .enumerate()
        .take(training_loop_limit)
    {
        b += 1;
        let input = batch.images_norm(); // values in between 0 and 1
        let [batch_size, HEIGHT, WIDTH, 1] = input.dims() else {
            panic!()
        };
        let input = input.squeeze_dim(3);
        let [_batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(
            [_batch_size, sequence_size, input_size],
            [batch_size, HEIGHT, WIDTH]
        );
        let _targets = batch.targets; // ignore the digit labels
        let targets = input.clone();

        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = training_model.forward_regression(input, targets);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
        iteration_speed_metric.update(&pre_metrics.adapt(), &metric_meta);

        let loss = pre_metrics.loss.clone();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &training_model.0);
        let lr = training_config.lr.get_lr(metric_meta.iteration.unwrap());
        training_model.0 = optim.step(lr, training_model.0, grads);

        println!(
            "Epoch {}/{}, Batch {b:0>4}/{}, Loss {:.4}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            training_config.num_epochs,
            dataloader_train.num_items() / training_config.batch_size + 1,
            loss_metric.value().current(),
            iteration_speed_metric.value().current(),
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
                epoch,
                valid_loop_limit,
            );
        }
    }

    // Display the averaged training metrics
    println!(
        "Epoch {}/{}, Avg Loss {:.4}",
        epoch,
        training_config.num_epochs,
        loss_metric.running_value().current(),
    );

    training_model.0
}

pub fn epoch_valid<B: Backend>(
    dataloader_valid: Dataloader<B>,
    valid_model: AutoEncoderNetwork<B>,
    training_config: &TrainingConfig,
    model_config: &AutoEncoderNetworkConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
) {
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = burn::train::metric::MetricMetadata {
        progress: Progress::new(0, valid_num_items),
        global_progress: Progress::new(0, valid_num_items),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0)),
    };

    let mut loss_metric = burn::train::metric::LossMetric::<B>::new();

    let valid_model = Wrap(valid_model, model_config.clone());

    // validation loop
    for (_b, batch) in dataloader_valid.iter().enumerate().take(valid_loop_limit) {
        let input = batch.images_norm(); // values in between 0 and 1
        let [batch_size, HEIGHT, WIDTH, 1] = input.dims() else {
            panic!()
        };
        let input = input.squeeze_dim(3);
        let [_batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(
            [_batch_size, sequence_size, input_size],
            [batch_size, HEIGHT, WIDTH]
        );
        let _targets = batch.targets; // ignore the digit labels
        let targets = input.clone();

        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = valid_model.forward_regression(input, targets);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
    }

    // Display the averaged validation metrics
    println!(
        "Epoch {}/{}, Avg Valid Loss {:.4}",
        epoch,
        training_config.num_epochs,
        loss_metric.running_value().current(),
    );
}

/// Wrapper over [`AutoEncoderNetwork`] for custom implementations.
pub struct Wrap<B: Backend>(pub AutoEncoderNetwork<B>, pub AutoEncoderNetworkConfig);

impl<B: Backend> Wrap<B> {
    pub fn forward_regression(
        &self,
        input: Tensor<B, 3>,
        targets: Tensor<B, 3>,
    ) -> RegressionOutput<B> {
        let model = &self.0;
        let _config = &self.1;
        let [batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(
            [batch_size, sequence_size, input_size],
            [batch_size, HEIGHT, WIDTH]
        );
        assert_eq!([batch_size, sequence_size, WIDTH], targets.dims());

        let (output, _encoder_caches, _decoder_caches) =
            model.forward(input.clone(), None, None, Some(7));
        assert_eq!([batch_size, sequence_size, input_size], output.dims());

        // if contains_nan_or_inf(&output) {
        //     panic!();
        // }
        let output = burn::tensor::activation::sigmoid(output);
        // if contains_nan_or_inf(&output) {
        //     panic!();
        // }
        assert!(
            targets
                .clone()
                .greater_equal(Tensor::zeros_like(&targets))
                .all()
                .into_scalar()
                .to_bool()
        );
        assert!(
            targets
                .clone()
                .lower_equal(Tensor::ones_like(&targets))
                .all()
                .into_scalar()
                .to_bool()
        );
        assert!(
            output
                .clone()
                .greater_equal(Tensor::zeros_like(&output))
                .all()
                .into_scalar()
                .to_bool()
        );
        assert!(
            output
                .clone()
                .lower_equal(Tensor::ones_like(&output))
                .all()
                .into_scalar()
                .to_bool()
        );

        let loss = burn_mamba::utils::loss::bce::BinaryCrossEntropyLossConfig::new()
            // .with_logits(true)
            .with_logits(false)
            .init()
            .forward(output.clone(), targets.clone());

        // let loss = burn_mamba::mse::MseLoss::new().forward(
        //     output.clone(),
        //     targets.clone(),
        //     Reduction::Mean,
        // );

        RegressionOutput::new(
            loss.clone(),
            output.reshape([batch_size * sequence_size, input_size]),
            targets.reshape([batch_size * sequence_size, input_size]),
        )
    }
}
