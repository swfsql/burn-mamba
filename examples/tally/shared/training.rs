//! The training loop every rung runs: dataloaders over the rung's [`Task`],
//! train/validate epochs, checkpoints, and a cross-entropy head over the scored
//! positions.

use super::data::{
    EVAL_SEED, NUM_CLASSES, NUM_EVAL, NUM_TRAIN, TRAIN_SEED, TallyBatch, TallyBatcher, TallyDataset,
    Task,
};
use crate::common::{
    cli::AppArgs,
    model::ModelConfigExt,
    training::{BatchBudget, TrainingConfig, metric_current},
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    module::AutodiffModule,
    optim::ModuleOptimizer,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_mamba::prelude::*;

/// The SSD pathway used everywhere on this ladder: the models are tiny and the
/// sequences short, so the simplest (autodiff) variant.
pub fn ssd_path() -> MambaSsdPath {
    MambaSsdPath::Mamba3(Mamba3SsdPath::Minimal(None))
}

type Dataloader = std::sync::Arc<dyn DataLoader<TallyBatch> + 'static>;

/// Train for the configured number of epochs, validating per family and
/// checkpointing along the way.
pub fn train(
    task: &Task,
    training_config: TrainingConfig,
    model_config: MambaLatentNetConfig,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(training_config.seed);

    let model: MambaLatentNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let muon_plan = ModelConfigExt::muon_plan(&model_config);
    if training_config.optimizer.muon.is_some() {
        print!("{}", muon_plan.describe(&model));
    }
    let mut optim = app_args.load_or_save_optim(training_config.optimizer.init(&muon_plan));

    let batcher = TallyBatcher {
        num_symbols: task.num_symbols,
    };
    let dataloader_train: Dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone())
        .build(TallyDataset::new(task, task.train, NUM_TRAIN, TRAIN_SEED));
    let valid_loaders: Vec<(&str, Dataloader)> = task
        .families
        .iter()
        .map(|(name, generator)| {
            let loader: Dataloader = DataLoaderBuilder::new(batcher.clone())
                .batch_size(training_config.batch_size)
                .num_workers(training_config.num_workers)
                .set_device(training_device.clone().inner())
                .build(TallyDataset::new(task, *generator, NUM_EVAL, EVAL_SEED));
            (*name, loader)
        })
        .collect();

    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, dataloader_train.num_items(), None),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0).into()),
    };
    let mut batch_budget = app_args.batch_budget();

    println!("running initial validation (chance ≈ {:.1}%)...", 100.0 / NUM_CLASSES as f32);
    let mut model = model;
    validate_all(&valid_loaders, model.valid(), 0);

    println!("Starting training...");
    for epoch in 1..training_config.num_epochs + 1 {
        model = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            model,
            &training_config,
            &mut optim,
            &mut metric_meta,
            epoch,
            &mut batch_budget,
        );
        app_args.save_model(&model);
        app_args.save_optim(&optim);

        if epoch % 5 == 0 || epoch == 1 || epoch == training_config.num_epochs {
            println!("running validation...");
            validate_all(&valid_loaders, model.valid(), epoch);
        }
        if batch_budget.is_exhausted() {
            println!("reached the --max-batches limit; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

/// Train for a single epoch; returns the updated model.
fn epoch_train(
    dataloader_train: Dataloader,
    training_model: MambaLatentNet,
    training_config: &TrainingConfig,
    optim: &mut ModuleOptimizer,
    metric_meta: &mut MetricMetadata,
    epoch: usize,
    batch_budget: &mut BatchBudget,
) -> MambaLatentNet {
    let limit = batch_budget.take_limit();
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut training_model = Wrap(training_model);

    for (mut b, batch) in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .enumerate()
        .take(limit)
    {
        b += 1;
        batch_budget.spend();
        let [batch_size, _, _] = batch.inputs.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

        let train_output = TrainStep::step(&training_model, batch);
        loss_metric.update(&train_output.item.adapt(), metric_meta);
        acc_metric.update(&train_output.item.adapt(), metric_meta);

        let lr = training_config.lr.get_lr(metric_meta.iteration.unwrap());
        training_model.0 = optim.step(lr, training_model.0, train_output.grads);

        if b % 16 == 0 {
            println!(
                "Epoch {epoch}/{}, Batch {b:0>4}/{}, Loss {:.4}, Acc {:0>6.2}, lr {lr:0>6.2e}",
                training_config.num_epochs,
                dataloader_train.num_items() / training_config.batch_size + 1,
                metric_current(loss_metric.value()),
                metric_current(acc_metric.value()),
            );
        }
    }
    println!(
        "Epoch {epoch}/{}, Avg Loss {:.4}, Avg Acc: {}",
        training_config.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );
    training_model.0
}

/// Validate on every family in turn, one line each.
fn validate_all(loaders: &[(&str, Dataloader)], valid_model: MambaLatentNet, epoch: usize) {
    let valid_model = Wrap(valid_model);
    for (name, loader) in loaders {
        let meta = MetricMetadata {
            progress: Progress::new(0, loader.num_items(), None),
            iteration: Some(0),
            lr: None,
        };
        let mut loss_metric = burn::train::metric::LossMetric::new();
        let mut acc_metric = burn::train::metric::AccuracyMetric::new();
        for batch in loader.iter().map(|b| b.expect("dataloader batch")) {
            let out = InferenceStep::step(&valid_model, batch);
            loss_metric.update(&out.adapt(), &meta);
            acc_metric.update(&out.adapt(), &meta);
        }
        println!(
            "  epoch {epoch}, {name:<12} loss {:.4}, acc {:6.2}%",
            metric_current(loss_metric.running_value()),
            metric_current(acc_metric.running_value()),
        );
    }
}

/// [`MambaLatentNet`] with a scored-positions classification head.
pub struct Wrap(pub MambaLatentNet);

impl TrainStep for Wrap {
    type Input = TallyBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let out = InferenceStep::step(self, batch);
        let grads = out.loss.backward();
        TrainOutput::new(&self.0, grads, out)
    }
}

impl InferenceStep for Wrap {
    type Input = TallyBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let [batch_size, seq, _] = batch.inputs.dims();
        let (output, _caches) = self.0.forward(batch.inputs, None, ssd_path(), None);
        let n = batch_size * seq;
        let logits = output.reshape([n, NUM_CLASSES]).select(0, batch.scored.clone());
        let targets = batch.targets.reshape([n]).select(0, batch.scored);
        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());
        ClassificationOutput::new(loss, logits, targets)
    }
}
