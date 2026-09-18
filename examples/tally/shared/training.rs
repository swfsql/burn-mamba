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
    session::{Cadence, Session},
    training::{TrainingConfig, metric_current},
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
    let (mut optim, progress) =
        app_args.load_or_save_optim(training_config.optimizer.init(&muon_plan));

    let batcher = TallyBatcher {
        num_symbols: task.num_symbols,
    };
    let dataloader_train: Dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(progress.shuffle_seed(training_config.seed))
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

    // Resume position, `--max-batches` budget, cadence and metrics log: by
    // default a validation every five epochs and no mid-epoch checkpoint.
    let batches = dataloader_train.num_items().div_ceil(training_config.batch_size);
    let cadence = Cadence {
        valid_every: Some(5 * batches),
        ..Cadence::default()
    };
    let mut session = app_args.session(
        progress,
        &training_config,
        cadence,
        dataloader_train.num_items(),
    );

    println!("running initial validation (chance ≈ {:.1}%)...", 100.0 / NUM_CLASSES as f32);
    let mut model = model;
    validate_all(&valid_loaders, model.valid(), 0, &mut session);

    println!("Starting training...");
    for epoch in session.epochs(training_config.num_epochs) {
        model = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            model,
            &training_config,
            &mut optim,
            &mut session,
            epoch,
            &valid_loaders,
            app_args,
        );
        app_args.save_model(&model);
        app_args.save_optim(&optim, session.progress());

        let last = epoch == training_config.num_epochs || session.is_exhausted();
        if last && !session.validated_now() {
            println!("running final validation...");
            validate_all(&valid_loaders, model.valid(), epoch, &mut session);
        }
        if session.is_exhausted() {
            println!("reached the --max-batches limit; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

/// Train for (the rest of) one epoch, checkpointing and validating (on
/// `valid_loaders`) at the `session`'s cadence; returns the updated model.
#[allow(clippy::too_many_arguments)]
fn epoch_train(
    dataloader_train: Dataloader,
    training_model: MambaLatentNet,
    training_config: &TrainingConfig,
    optim: &mut ModuleOptimizer,
    session: &mut Session,
    epoch: usize,
    valid_loaders: &[(&str, Dataloader)],
    app_args: &AppArgs,
) -> MambaLatentNet {
    let batches = dataloader_train.num_items().div_ceil(training_config.batch_size);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut training_model = Wrap(training_model);

    for batch in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .take(session.batch_limit(batches))
    {
        let b = session.begin_batch();
        let [batch_size, _, _] = batch.inputs.dims();
        let (_step, lr) = session.begin_step(batch_size);

        let train_output = TrainStep::step(&training_model, batch);
        loss_metric.update(&train_output.item.adapt(), session.meta());
        acc_metric.update(&train_output.item.adapt(), session.meta());

        training_model.0 = optim.step(lr, training_model.0, train_output.grads);

        let (loss, acc) = (
            metric_current(loss_metric.value()),
            metric_current(acc_metric.value()),
        );
        session.log_train(&[("loss", loss), ("acc", acc)]);
        if b % 16 == 0 {
            println!(
                "Epoch {epoch}/{}, Batch {b:0>4}/{batches}, Loss {loss:.4}, Acc {acc:0>6.2}, lr {lr:0>6.2e}",
                training_config.num_epochs,
            );
        }

        if session.checkpoint_due() {
            app_args.save_model(&training_model.0);
            app_args.save_optim(optim, session.progress());
        }
        if session.valid_due() {
            println!("running validation...");
            validate_all(valid_loaders, training_model.0.valid(), epoch, session);
        }
    }
    println!(
        "Epoch {epoch}/{}, Avg Loss {:.4}, Avg Acc: {}",
        training_config.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );
    session.end_epoch(batches);
    training_model.0
}

/// Validate on every family in turn (each capped at the session's
/// `valid_batches`), one line and one metrics-log entry each.
fn validate_all(
    loaders: &[(&str, Dataloader)],
    valid_model: MambaLatentNet,
    epoch: usize,
    session: &mut Session,
) {
    let valid_model = Wrap(valid_model);
    let limit = session.cadence().valid_batches.unwrap_or(usize::MAX);
    for (name, loader) in loaders {
        let meta = MetricMetadata {
            progress: Progress::new(0, loader.num_items(), None),
            iteration: Some(0),
            lr: None,
        };
        let mut loss_metric = burn::train::metric::LossMetric::new();
        let mut acc_metric = burn::train::metric::AccuracyMetric::new();
        for batch in loader.iter().map(|b| b.expect("dataloader batch")).take(limit) {
            let out = InferenceStep::step(&valid_model, batch);
            loss_metric.update(&out.adapt(), &meta);
            acc_metric.update(&out.adapt(), &meta);
        }
        let (loss, acc) = (
            metric_current(loss_metric.running_value()),
            metric_current(acc_metric.running_value()),
        );
        session.log_valid(name, &[("loss", loss), ("acc", acc)]);
        println!("  epoch {epoch}, {name:<12} loss {loss:.4}, acc {acc:6.2}%");
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
