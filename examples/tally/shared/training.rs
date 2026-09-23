//! The training loop every rung runs: dataloaders over the rung's [`Task`],
//! train/validate epochs, checkpoints, and a cross-entropy head over the scored
//! positions ([`forward_classification`]), stepped by a `Trainer`: under plain
//! SGD (unless `--no-graph`) the whole training step replays from a graph
//! captured at the first batch, under any other optimizer it steps eagerly.

use super::data::{
    EVAL_SEED, IGNORE, NUM_CLASSES, NUM_EVAL, NUM_TRAIN, TRAIN_SEED, TallyBatch, TallyBatcher,
    TallyDataset, Task,
};
use crate::common::{
    cli::AppArgs,
    device::loader_device,
    model::ModelConfigExt,
    session::{Cadence, Session},
    training::{TrainingConfig, metric_current},
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    optim::ModuleOptimizer,
    tensor::activation::log_softmax,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep},
};
use burn_mamba::prelude::*;
use burn_stack::examples::trainer::Trainer;

/// The SSD pathway used everywhere on this ladder: the models are tiny and the
/// sequences short, so the simplest (autodiff) variant.
pub fn ssd_path() -> MambaSsdPath {
    MambaSsdPath::Mamba3(Mamba3SsdPath::Minimal(None))
}

type Dataloader = std::sync::Arc<dyn DataLoader<TallyBatch> + 'static>;

/// What trains the model: `(inputs, targets)` in, the loss and the scored
/// `(logits, targets)` out ([`train_loss`]).
type ClassTrainer =
    Trainer<MambaLatentNet, (Tensor<3>, Tensor<2, Int>), (Tensor<2>, Tensor<1, Int>)>;

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
    // The workers build batches on the host, and the loops move them to the
    // device: a worker uploading to the GPU from its own thread can invalidate
    // a graph being captured (see `device::loader_device`).
    let dataloader_train: Dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(progress.shuffle_seed(training_config.seed))
        .num_workers(training_config.num_workers)
        .set_device(loader_device(&training_device))
        .build(TallyDataset::new(task, task.train, NUM_TRAIN, TRAIN_SEED));
    let valid_loaders: Vec<(&str, Dataloader)> = task
        .families
        .iter()
        .map(|(name, generator)| {
            let loader: Dataloader = DataLoaderBuilder::new(batcher.clone())
                .batch_size(training_config.batch_size)
                .num_workers(training_config.num_workers)
                .set_device(loader_device(&training_device))
                .build(TallyDataset::new(task, *generator, NUM_EVAL, EVAL_SEED));
            (*name, loader)
        })
        .collect();

    // Resume position, budget, cadence and metrics log: by
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
    let mut trainer = Trainer::new(model, train_loss, &training_config.optimizer, app_args.graphs());
    validate_all(&valid_loaders, trainer.module().valid(), 0, &mut session);

    println!("Starting training...");
    for epoch in session.epochs(training_config.num_epochs) {
        epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            &mut trainer,
            &training_config,
            &mut optim,
            &mut session,
            epoch,
            &valid_loaders,
            app_args,
        );
        app_args.save_model(&trainer.module());
        app_args.save_optim(&optim, session.progress());

        let last = epoch == training_config.num_epochs || session.is_exhausted();
        if last && !session.validated_now() {
            println!("running final validation...");
            validate_all(&valid_loaders, trainer.module().valid(), epoch, &mut session);
        }
        if session.is_exhausted() {
            println!("reached the training budget; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

/// Train `trainer` for (the rest of) one epoch, checkpointing and validating
/// (on `valid_loaders`) at the `session`'s cadence.
#[allow(clippy::too_many_arguments)]
fn epoch_train(
    dataloader_train: Dataloader,
    trainer: &mut ClassTrainer,
    training_config: &TrainingConfig,
    optim: &mut ModuleOptimizer,
    session: &mut Session,
    epoch: usize,
    valid_loaders: &[(&str, Dataloader)],
    app_args: &AppArgs,
) {
    let batches = dataloader_train.num_items().div_ceil(training_config.batch_size);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new().with_pad_token(NUM_CLASSES);

    for batch in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .take(session.batch_limit(batches))
    {
        let b = session.begin_batch();
        let [batch_size, _, _] = batch.inputs.dims();
        let (_step, lr) = session.begin_step(batch_size);

        let batch = batch.to_device(trainer.device());
        let batch = (batch.inputs, batch.targets);
        let (loss, (logits, targets)) = trainer.step(batch, optim, lr);
        let output = ClassificationOutput::new(loss, logits, targets);
        loss_metric.update(&output.adapt(), session.meta());
        acc_metric.update(&output.adapt(), session.meta());

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
            app_args.save_model(&trainer.module());
            app_args.save_optim(optim, session.progress());
        }
        if session.valid_due() {
            println!("running validation...");
            validate_all(valid_loaders, trainer.module().valid(), epoch, session);
        }
        if session.is_exhausted() {
            break;
        }
    }
    println!(
        "Epoch {epoch}/{}, Avg Loss {:.4}, Avg Acc: {}",
        training_config.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );
    session.end_epoch(batches);
}

/// Validate on every family in turn (each capped at the session's
/// `valid_batches`), one line and one metrics-log entry each.
fn validate_all(
    loaders: &[(&str, Dataloader)],
    valid_model: MambaLatentNet,
    epoch: usize,
    session: &mut Session,
) {
    let device = valid_model.devices().remove(0);
    let valid_model = Wrap(valid_model);
    let limit = session.cadence().valid_batches.unwrap_or(usize::MAX);
    for (name, loader) in loaders {
        let meta = MetricMetadata {
            progress: Progress::new(0, loader.num_items(), None),
            iteration: Some(0),
            lr: None,
        };
        let mut loss_metric = burn::train::metric::LossMetric::new();
        let mut acc_metric = burn::train::metric::AccuracyMetric::new().with_pad_token(NUM_CLASSES);
        for batch in loader.iter().map(|b| b.expect("dataloader batch")).take(limit) {
            let out = InferenceStep::step(&valid_model, batch.to_device(&device));
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

/// [`MambaLatentNet`] with a scored-positions classification head, for the
/// validation step.
pub struct Wrap(pub MambaLatentNet);

impl InferenceStep for Wrap {
    type Input = TallyBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        forward_classification(&self.0, batch.inputs, batch.targets)
    }
}

/// The training step's [`forward_classification`] on `(inputs, targets)`
/// (inner backend): the loss, and the scored `(logits, targets)`.
fn train_loss(
    model: &MambaLatentNet,
    (inputs, targets): (Tensor<3>, Tensor<2, Int>),
) -> (Tensor<1>, (Tensor<2>, Tensor<1, Int>)) {
    let output = forward_classification(model, inputs.autodiff(), targets.autodiff());
    (output.loss, (output.output.inner(), output.targets.inner()))
}

/// Forward the model and score every position whose target is not [`IGNORE`].
///
/// The others reach neither the loss nor the accuracy: masked out of the mean,
/// and targeted at the pad class `NUM_CLASSES`, which no prediction matches and
/// `AccuracyMetric::with_pad_token(NUM_CLASSES)` leaves out. Masked rather than
/// dropped, so every shape is the batch's whatever it scores — a captured
/// training step replays one.
fn forward_classification(
    model: &MambaLatentNet,
    inputs: Tensor<3>,
    targets: Tensor<2, Int>,
) -> ClassificationOutput {
    let [batch_size, seq, _] = inputs.dims();
    let (output, _caches) = model.forward(inputs, None, ssd_path(), None, None);
    let n = batch_size * seq;
    let logits = output.reshape([n, NUM_CLASSES]);
    let targets = targets.reshape([n]);
    let ignored = targets.clone().equal_elem(IGNORE);
    let nll = log_softmax(logits.clone(), 1)
        .gather(1, targets.clone().mask_fill(ignored.clone(), 0).reshape([n, 1]))
        .reshape([n])
        .neg()
        .mask_fill(ignored.clone(), 0);
    let loss = nll.sum() / ignored.clone().bool_not().float().sum();
    let targets = targets.mask_fill(ignored, NUM_CLASSES as i64);
    ClassificationOutput::new(loss, logits, targets)
}
