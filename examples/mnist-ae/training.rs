//! Training loop for the MNIST autoencoder: builds the dataloaders, runs the
//! train/validate epochs, and checkpoints the model and optimizer. The [`Wrap`]
//! newtype adapts [`AeModel`] to Burn's `TrainStep` / `InferenceStep` via a
//! pixel-reconstruction objective (binary cross-entropy on the normalized image,
//! computed from raw logits).

pub use crate::common::{
    cli::AppArgs,
    mnist::dataset::{HEIGHT, MnistBatch, MnistBatcher, MnistDataset, WIDTH},
    training::TrainingConfig,
};
use crate::model::{AeConfig, AeModel};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    module::AutodiffModule,
    optim::{AdamW, Optimizer, adaptor::OptimizerAdaptor},
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};
use burn_mamba::utils::loss::bce::BinaryCrossEntropyLossConfig;

/// Run the full training routine: load/init the model and optimizer, then train
/// for the configured number of epochs (validating and checkpointing along the
/// way).
pub fn train(
    training_config: TrainingConfig,
    model_config: AeConfig,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(training_config.seed);

    // load (or init and save) model and optim
    let model: AeModel = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let mut optim = app_args.load_or_save_optim::<AeModel>(&training_config.optimizer);

    let mut model = Wrap(model, model_config.clone());

    // Create the batcher
    let batcher = MnistBatcher::default();

    // Create the dataloaders. Training batches must live on the autodiff device
    // (to match the model weights); validation runs on the inner backend.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone())
        .build(MnistDataset::train());
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone().inner())
        .build(MnistDataset::test());

    let training_num_items = dataloader_train.num_items();
    let global_training_num_items = training_num_items * training_config.num_epochs;

    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, training_num_items),
        global_progress: Progress::new(0, global_training_num_items),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0)),
    };

    println!("running small initial validation...");
    epoch_valid(
        std::sync::Arc::clone(&dataloader_valid),
        model.0.valid(),
        &training_config,
        &model_config,
        0,
        Some(10),
    );

    println!("Starting training...");
    for epoch in 1..training_config.num_epochs + 1 {
        model.0 = epoch_train(
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
        epoch_valid(
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

type Dataloader = std::sync::Arc<dyn DataLoader<MnistBatch> + 'static>;

/// Train for a single epoch, stepping the optimizer per batch and periodically
/// validating + checkpointing; returns the updated model.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train(
    dataloader_train: Dataloader,
    dataloader_valid: Dataloader,
    training_model: AeModel,
    training_config: &TrainingConfig,
    model_config: &AeConfig,
    optim: &mut OptimizerAdaptor<AdamW, AeModel>,
    metric_meta: &mut MetricMetadata,
    epoch: usize,
    training_loop_limit: Option<usize>,
    valid_loop_limit: Option<usize>,
    app_args: &AppArgs,
) -> AeModel {
    let training_loop_limit = training_loop_limit.unwrap_or(usize::MAX);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

    let mut training_model = Wrap(training_model, model_config.clone());

    // training loop
    for (mut b, batch) in dataloader_train
        .iter()
        .enumerate()
        .take(training_loop_limit)
    {
        b += 1;
        let [batch_size, _, _, _] = batch.images.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;
        metric_meta.global_progress.items_processed += batch_size;

        let train_output = TrainStep::step(&training_model, batch);
        let pre_metrics = &train_output.item;

        loss_metric.update(&pre_metrics.adapt(), metric_meta);
        iteration_speed_metric.update(&pre_metrics.adapt(), metric_meta);

        let lr = training_config.lr.get_lr(metric_meta.iteration.unwrap());
        training_model.0 = optim.step(lr, training_model.0, train_output.grads);

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
            epoch_valid(
                std::sync::Arc::clone(&dataloader_valid),
                training_model.0.valid(),
                training_config,
                model_config,
                epoch,
                valid_loop_limit,
            );
        }
    }

    println!(
        "Epoch {}/{}, Avg Loss {:.4}",
        epoch,
        training_config.num_epochs,
        loss_metric.running_value().current(),
    );

    training_model.0
}

/// Run validation over (up to `valid_loop_limit`) batches and report the average
/// reconstruction loss.
pub fn epoch_valid(
    dataloader_valid: Dataloader,
    valid_model: AeModel,
    training_config: &TrainingConfig,
    model_config: &AeConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
) {
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, valid_num_items),
        global_progress: Progress::new(0, valid_num_items),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0)),
    };

    let mut loss_metric = burn::train::metric::LossMetric::new();

    let valid_model = Wrap(valid_model, model_config.clone());

    for (_b, batch) in dataloader_valid.iter().enumerate().take(valid_loop_limit) {
        let [batch_size, _, _, _] = batch.images.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;
        metric_meta.global_progress.items_processed += batch_size;

        let pre_metrics = InferenceStep::step(&valid_model, batch);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
    }

    println!(
        "Epoch {}/{}, Avg Valid Loss {:.4}",
        epoch,
        training_config.num_epochs,
        loss_metric.running_value().current(),
    );
}

/// Wrapper over [`AeModel`] for the train/infer step impls.
pub struct Wrap(pub AeModel, pub AeConfig);

impl TrainStep for Wrap {
    type Input = MnistBatch;
    type Output = RegressionOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pre_metrics = InferenceStep::step(self, batch);
        let grads = pre_metrics.loss.backward();
        TrainOutput::new(&self.0, grads, pre_metrics)
    }
}

impl InferenceStep for Wrap {
    type Input = MnistBatch;
    type Output = RegressionOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let input = batch.images_norm(); // pixels in [0, 1] (Bernoulli targets)
        let [batch_size, HEIGHT, WIDTH, 1] = input.dims() else {
            panic!()
        };
        let input = input.reshape([batch_size, HEIGHT * WIDTH, 1]);
        self.forward_reconstruction(input)
    }
}

impl Wrap {
    /// Forward the autoencoder and compute the binary cross-entropy
    /// reconstruction loss (from raw logits) against the normalized image.
    pub fn forward_reconstruction(&self, input: Tensor<3>) -> RegressionOutput {
        let model = &self.0;
        let [batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(sequence_size, HEIGHT * WIDTH);
        assert_eq!(input_size, 1);

        let logits = model.forward(input.clone()); // [batch, seq, 1]
        assert_eq!([batch_size, sequence_size, 1], logits.dims());

        // Flatten the (single-channel) pixels for the loss / metric.
        let logits_flat = logits.reshape([batch_size, sequence_size]);
        let targets_flat = input.reshape([batch_size, sequence_size]);

        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init()
            // `forward` carries an (otherwise unused) backend type param that
            // can't be inferred; name the global Dispatch backend explicitly.
            .forward::<2, burn::backend::Dispatch>(logits_flat.clone(), targets_flat.clone());

        RegressionOutput::new(loss, logits_flat, targets_flat)
    }
}
