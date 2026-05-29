//! Training loop for the state-tracking example: builds the dataloaders, runs
//! the train/validate epochs, and checkpoints the model and optimizer. The
//! [`Wrap`] newtype adapts the example network to Burn's `TrainStep` /
//! `InferenceStep` via a cross-entropy classification head over **every**
//! position (the running `A₅` product).

pub use crate::common::{
    cli::AppArgs,
    model::{MyMamba3Network, MyMamba3NetworkConfig},
    training::TrainingConfig,
};
use crate::dataset::{
    EVAL_SEED, NUM_CLASSES, NUM_EVAL, NUM_TRAIN, SEQ_LENGTH, StateTrackingBatch,
    StateTrackingBatcher, StateTrackingDataset, TRAIN_SEED,
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    module::AutodiffModule,
    optim::{AdamW, Optimizer, adaptor::OptimizerAdaptor},
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_mamba::prelude::*;

/// Run the full training routine: load/init the model and optimizer, then train
/// for the configured number of epochs (validating and checkpointing along the
/// way).
pub fn train(
    training_config: TrainingConfig,
    model_config: MyMamba3NetworkConfig,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(training_config.seed);

    // load (or init and save) model and optim
    let model: MyMamba3Network = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let mut optim = app_args.load_or_save_optim::<MyMamba3Network>(&training_config.optimizer);

    let mut model = Wrap(model, model_config.clone());

    // Create the batcher
    let batcher = StateTrackingBatcher::default();

    // Create the dataloaders. Training batches must live on the autodiff device
    // (to match the model weights); validation runs on the inner backend.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone())
        .build(StateTrackingDataset::new(NUM_TRAIN, SEQ_LENGTH, TRAIN_SEED));
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone().inner())
        .build(StateTrackingDataset::new(NUM_EVAL, SEQ_LENGTH, EVAL_SEED));

    let training_num_items = dataloader_train.num_items();
    let global_training_num_items = training_num_items * training_config.num_epochs;

    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, training_num_items),
        global_progress: Progress::new(0, global_training_num_items),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0)),
    };

    println!("running initial validation (chance ≈ {:.1}%)...", 100.0 / NUM_CLASSES as f32);
    epoch_valid(
        std::sync::Arc::clone(&dataloader_valid),
        model.0.valid(),
        &training_config,
        &model_config,
        0,
        None,
    );

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in 1..training_config.num_epochs + 1 {
        model.0 = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            model.0,
            &training_config,
            &model_config,
            &mut optim,
            &mut metric_meta,
            epoch,
            None,
        );

        // save assets
        app_args.save_model(&model.0);
        app_args.save_optim(&optim);

        if epoch % 5 == 0 || epoch == 1 || epoch == training_config.num_epochs {
            println!("running validation...");
            epoch_valid(
                std::sync::Arc::clone(&dataloader_valid),
                model.0.valid(),
                &training_config,
                &model_config,
                epoch,
                None,
            );
        }
    }
    println!("Training finished.");
}

type Dataloader = std::sync::Arc<dyn DataLoader<StateTrackingBatch> + 'static>;

/// Train for a single epoch, stepping the optimizer per batch; returns the
/// updated model.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train(
    dataloader_train: Dataloader,
    training_model: MyMamba3Network,
    training_config: &TrainingConfig,
    model_config: &MyMamba3NetworkConfig,
    optim: &mut OptimizerAdaptor<AdamW, MyMamba3Network>,
    metric_meta: &mut MetricMetadata,
    epoch: usize,
    training_loop_limit: Option<usize>,
) -> MyMamba3Network {
    let training_loop_limit = training_loop_limit.unwrap_or(usize::MAX);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

    let mut training_model = Wrap(training_model, model_config.clone());

    // training loop
    for (mut b, batch) in dataloader_train
        .iter()
        .enumerate()
        .take(training_loop_limit)
    {
        b += 1;
        let [batch_size, _, _] = batch.inputs.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;
        metric_meta.global_progress.items_processed += batch_size;

        let train_output = TrainStep::step(&training_model, batch);
        let pre_metrics = &train_output.item;

        loss_metric.update(&pre_metrics.adapt(), metric_meta);
        acc_metric.update(&pre_metrics.adapt(), metric_meta);
        iteration_speed_metric.update(&pre_metrics.adapt(), metric_meta);

        let lr = training_config.lr.get_lr(metric_meta.iteration.unwrap());
        training_model.0 = optim.step(lr, training_model.0, train_output.grads);

        println!(
            "Epoch {}/{}, Batch {b:0>4}/{}, Loss {:.4}, Acc {:0>6.2}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            training_config.num_epochs,
            dataloader_train.num_items() / training_config.batch_size + 1,
            loss_metric.value().current(),
            acc_metric.value().current(),
            iteration_speed_metric.value().current(),
        );
    }

    // Display the averaged training metrics
    println!(
        "Epoch {}/{}, Avg Loss {:.4}, Avg Acc: {}",
        epoch,
        training_config.num_epochs,
        loss_metric.running_value().current(),
        acc_metric.running_value().current(),
    );

    training_model.0
}

/// Run validation over (up to `valid_loop_limit`) batches and report the
/// average loss and per-token accuracy.
pub fn epoch_valid(
    dataloader_valid: Dataloader,
    valid_model: MyMamba3Network,
    training_config: &TrainingConfig,
    model_config: &MyMamba3NetworkConfig,
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
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    // Per-position correct/total counts (the diagnostic that reveals how deep the
    // model actually tracks — the headline gap is in the deep positions).
    let mut pos_correct: Vec<u64> = Vec::new();
    let mut pos_total: Vec<u64> = Vec::new();

    let valid_model = Wrap(valid_model, model_config.clone());

    // validation loop
    for (_b, batch) in dataloader_valid.iter().enumerate().take(valid_loop_limit) {
        let [batch_size, seq, _] = batch.inputs.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;
        metric_meta.global_progress.items_processed += batch_size;

        let pre_metrics = InferenceStep::step(&valid_model, batch);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
        accumulate_per_position(
            pre_metrics.output.clone(),
            pre_metrics.targets.clone(),
            seq,
            &mut pos_correct,
            &mut pos_total,
        );
    }

    // Display the averaged validation metrics
    println!(
        "Epoch {}/{}, Avg Valid Loss {:.4}, Avg Valid Acc: {}",
        epoch,
        training_config.num_epochs,
        loss_metric.running_value().current(),
        acc_metric.running_value().current(),
    );
    println!("  per-position acc: {}", format_per_position(&pos_correct, &pos_total));
}

/// Accumulate per-position correct/total counts from flattened logits/targets.
///
/// `logits_flat` is `[batch·seq, num_classes]` and `targets_flat` is `[batch·seq]`
/// (the [`ClassificationOutput`] layout, row-major over `(batch, position)`); the
/// running tallies are grown to `seq` on first use.
pub fn accumulate_per_position(
    logits_flat: Tensor<2>,
    targets_flat: Tensor<1, Int>,
    seq: usize,
    correct: &mut Vec<u64>,
    total: &mut Vec<u64>,
) {
    let [bseq, _classes] = logits_flat.dims();
    let batch = bseq / seq;
    let pred = logits_flat
        .argmax(1)
        .reshape([bseq])
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    let tgt = targets_flat.into_data().to_vec::<i32>().unwrap();
    if correct.len() < seq {
        correct.resize(seq, 0);
        total.resize(seq, 0);
    }
    for b in 0..batch {
        for t in 0..seq {
            let i = b * seq + t;
            if pred[i] == tgt[i] {
                correct[t] += 1;
            }
            total[t] += 1;
        }
    }
}

/// Format per-position accuracy tallies as `"100% 73% …"` (one entry per position).
pub fn format_per_position(correct: &[u64], total: &[u64]) -> String {
    correct
        .iter()
        .zip(total)
        .map(|(c, t)| {
            let pct = if *t == 0 { 0.0 } else { 100.0 * *c as f32 / *t as f32 };
            format!("{pct:.0}%")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Wrapper over [`MyMamba3Network`] for custom implementations.
pub struct Wrap(pub MyMamba3Network, pub MyMamba3NetworkConfig);

impl TrainStep for Wrap {
    type Input = StateTrackingBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pre_metrics = InferenceStep::step(self, batch);
        let grads = pre_metrics.loss.backward();

        TrainOutput::new(&self.0, grads, pre_metrics)
    }
}

impl InferenceStep for Wrap {
    type Input = StateTrackingBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch.inputs, batch.targets)
    }
}

impl Wrap {
    /// Forward the model and compute the cross-entropy classification loss over
    /// **every** position (the running `A₅` product at each step).
    pub fn forward_classification(
        &self,
        inputs: Tensor<3>,
        targets: Tensor<2, Int>,
    ) -> ClassificationOutput {
        let model = &self.0;
        let _config = &self.1;
        let [batch_size, sequence_size, _num_gen] = inputs.dims();
        assert_eq!([batch_size, sequence_size], targets.dims());

        let ssd_path = Mamba3SsdPath::Minimal(None);
        //
        let (output, _caches) = model.forward(inputs, None, ssd_path);
        assert_eq!([batch_size, sequence_size, NUM_CLASSES], output.dims());

        // Score every position: flatten (batch, seq) into a single axis so the
        // loss/accuracy are per-token.
        let logits = output.reshape([batch_size * sequence_size, NUM_CLASSES]);
        let targets = targets.reshape([batch_size * sequence_size]);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss.clone(), logits, targets)
    }
}
