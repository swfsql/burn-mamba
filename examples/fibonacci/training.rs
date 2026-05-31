//! Training loop for the fibonacci example: builds the dataloaders, runs the
//! train/validate epochs, and periodically saves the model and optimizer. The
//! [`Wrap`] newtype adapts the example network to Burn's `TrainStep` /
//! `InferenceStep` via an MSE regression head on the last timestep.

pub use crate::common::{cli::AppArgs, training::TrainingConfig};
use crate::dataset::{
    NOISE_LEVEL, NUM_SEQUENCES, SEQ_LENGTH, SequenceBatch, SequenceBatcher, SequenceDataset,
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    module::AutodiffModule,
    nn::loss::Reduction,
    optim::{AdamW, Optimizer, adaptor::OptimizerAdaptor},
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{InferenceStep, RegressionOutput, TrainOutput, TrainStep},
};
use burn_mamba::prelude::*;

/// Run the full training routine: load/init the model and optimizer, then train
/// for the configured number of epochs (validating and checkpointing along the
/// way).
pub fn train(
    training_config: TrainingConfig,
    model_config: MambaLatentNetConfig,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(training_config.seed);

    // load (or init and save) model and optim
    let model: MambaLatentNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let mut optim = app_args.load_or_save_optim::<MambaLatentNet>(&training_config.optimizer);

    let mut model = Wrap(model, model_config.clone());

    // Create the batcher
    let batcher = SequenceBatcher::default();

    // Create the dataloaders. Training batches must live on the autodiff device
    // (to match the model weights); validation runs on the inner backend.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone())
        .build(SequenceDataset::new(NUM_SEQUENCES, SEQ_LENGTH, NOISE_LEVEL));
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone().inner())
        // 20% size of training
        .build(SequenceDataset::new(
            NUM_SEQUENCES / 5,
            SEQ_LENGTH,
            NOISE_LEVEL,
        ));

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
    // Iterate over our training for X epochs
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

        let valid_loop_limit = Some(10);
        println!("running validation (batch iteration limit: {valid_loop_limit:?})");
        epoch_valid(
            std::sync::Arc::clone(&dataloader_valid),
            model.0.valid(),
            &training_config,
            &model_config,
            epoch,
            valid_loop_limit, // validate against 10 batches
        );
    }
    println!("Training finished.");
}

type Dataloader = std::sync::Arc<dyn DataLoader<SequenceBatch> + 'static>;

/// Train for a single epoch, stepping the optimizer per batch and periodically
/// validating + checkpointing; returns the updated model.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train(
    dataloader_train: Dataloader,
    dataloader_valid: Dataloader,
    training_model: MambaLatentNet,
    training_config: &TrainingConfig,
    model_config: &MambaLatentNetConfig,
    optim: &mut OptimizerAdaptor<AdamW, MambaLatentNet>,
    metric_meta: &mut MetricMetadata,
    epoch: usize,
    training_loop_limit: Option<usize>,
    valid_loop_limit: Option<usize>,
    app_args: &AppArgs,
) -> MambaLatentNet {
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
        let [batch_size, _, _] = batch.sequences.dims();
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

        if b % 10 == 0 {
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

    // Display the averaged training metrics
    println!(
        "Epoch {}/{}, Avg Loss {:.4}",
        epoch,
        training_config.num_epochs,
        loss_metric.running_value().current(),
    );

    training_model.0
}

/// Run validation over (up to `valid_loop_limit`) batches and report the
/// average loss.
pub fn epoch_valid(
    dataloader_valid: Dataloader,
    valid_model: MambaLatentNet,
    training_config: &TrainingConfig,
    model_config: &MambaLatentNetConfig,
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

    let mut loss_metric = burn::train::metric::LossMetric::new();

    let valid_model = Wrap(valid_model, model_config.clone());

    // validation loop
    for (_b, batch) in dataloader_valid.iter().enumerate().take(valid_loop_limit) {
        let [batch_size, _, _] = batch.sequences.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;
        metric_meta.global_progress.items_processed += batch_size;

        let pre_metrics = InferenceStep::step(&valid_model, batch);
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

/// Wrapper over [`MambaLatentNet`] for custom implementations.
pub struct Wrap(pub MambaLatentNet, pub MambaLatentNetConfig);

impl TrainStep for Wrap {
    type Input = SequenceBatch;
    type Output = RegressionOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pre_metrics = InferenceStep::step(self, batch);
        let grads = pre_metrics.loss.backward();

        TrainOutput::new(&self.0, grads, pre_metrics)
    }
}

impl InferenceStep for Wrap {
    type Input = SequenceBatch;
    type Output = RegressionOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let input = batch.sequences;
        let targets = batch.targets;
        let [batch_size, sequence_size, _input_size] = input.dims();
        assert_eq!([batch_size, 1], targets.dims());
        assert!(sequence_size >= 1);

        self.forward_regression(input, targets)
    }
}

impl Wrap {
    /// Forward the model and compute the MSE regression loss against the last
    /// timestep's prediction.
    pub fn forward_regression(&self, input: Tensor<3>, targets: Tensor<2>) -> RegressionOutput {
        let model = &self.0;
        let _model_config = &self.1;
        let [batch_size, sequence_size, _input_size] = input.dims();
        assert_eq!([batch_size, 1], targets.dims());
        assert!(sequence_size >= 1);

        let ssd_path = MambaSsdPath::Mamba2(Mamba2SsdPath::Minimal(None));
        // ::Mamba2(Mamba2SsdPath::Serial(None));
        // ::Mamba2(Mamba2SsdPath::SerialRecalculated(None)); // saves vram
        //
        let (output, _caches) = model.forward(input.clone(), None, ssd_path);

        assert_eq!([batch_size, sequence_size, 1], output.dims());
        let last_output = output.narrow(1, sequence_size - 1, 1).squeeze_dim(1);
        assert_eq!([batch_size, 1], last_output.dims());

        let loss = burn_mamba::modules::loss::mse::MseLoss::new().forward(
            last_output.clone(),
            targets.clone(),
            Reduction::Mean,
        );

        RegressionOutput::new(loss.clone(), last_output, targets)
    }
}
