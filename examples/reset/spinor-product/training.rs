//! Training loop for the `spinor-product` example: builds the dataloaders, runs the
//! train/validate epochs, and checkpoints the model and optimizer. The objective
//! is a cross-entropy head over **every** token (the group element after each
//! pair of symbols, [`forward_classification`]), stepped by a `Trainer`: under
//! plain SGD (unless `--no-graph`) the whole training step replays from a graph
//! captured at the first batch, under any other optimizer it steps eagerly.

pub use crate::common::{
    cli::AppArgs,
    device::loader_device,
    model::ModelConfigExt,
    session::{Cadence, Session},
    training::{TrainingConfig, metric_current},
};
use crate::dataset::{
    EVAL_SEED, Family, NUM_CLASSES, NUM_EVAL, NUM_TRAIN, ProductBatch, ProductBatcher,
    ProductDataset, SEQ_LENGTH, TRAIN_SEED,
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    optim::ModuleOptimizer,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep},
};
use burn_mamba::prelude::*;
use burn_stack::examples::trainer::Trainer;

/// The evaluation splits, reported separately: `shuffle` is where the word runs
/// the whole sequence and only the order is left, `runs` is where the counts
/// come closest to sufficing, so a single averaged number would hide the point.
pub const EVAL_FAMILIES: [(&str, Family); 3] = [
    ("random", Family::Random),
    ("shuffle", Family::Shuffle),
    ("runs", Family::Runs),
];

/// The SSD pathway used everywhere in this example. The model is tiny and the
/// sequences are short, so the simplest (autodiff) variant is the right one.
pub fn ssd_path() -> MambaSsdPath {
    MambaSsdPath::Mamba3(Mamba3SsdPath::Minimal(None))
}

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

    let model: MambaLatentNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let muon_plan = ModelConfigExt::muon_plan(&model_config);
    if training_config.optimizer.muon.is_some() {
        print!("{}", muon_plan.describe(&model));
    }
    let (mut optim, progress) =
        app_args.load_or_save_optim(training_config.optimizer.init(&muon_plan));

    let mut trainer = Trainer::new(model, train_loss, &training_config.optimizer, app_args.graphs());
    let batcher = ProductBatcher::default();

    // The workers build batches on the host, and the loops move them to the
    // device: a worker uploading to the GPU from its own thread can invalidate
    // a graph being captured (see `device::loader_device`).
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(progress.shuffle_seed(training_config.seed))
        .num_workers(training_config.num_workers)
        .set_device(loader_device(&training_device))
        .build(ProductDataset::new(
            NUM_TRAIN,
            SEQ_LENGTH,
            Family::Mixed,
            TRAIN_SEED,
        ));
    let valid_loaders: Vec<(&str, Dataloader)> = EVAL_FAMILIES
        .iter()
        .map(|(name, family)| {
            let loader: Dataloader = DataLoaderBuilder::new(batcher.clone())
                .batch_size(training_config.batch_size)
                .num_workers(training_config.num_workers)
                .set_device(loader_device(&training_device))
                .build(ProductDataset::new(
                    NUM_EVAL, SEQ_LENGTH, *family, EVAL_SEED,
                ));
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

    println!(
        "running initial validation (chance ≈ {:.1}%)...",
        100.0 / NUM_CLASSES as f32
    );
    validate_all(&valid_loaders, trainer.module().valid(), &model_config, 0, &mut session);

    println!("Starting training...");
    for epoch in session.epochs(training_config.num_epochs) {
        epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            &mut trainer,
            &training_config,
            &model_config,
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
            validate_all(&valid_loaders, trainer.module().valid(), &model_config, epoch, &mut session);
        }

        if session.is_exhausted() {
            println!("reached the training budget; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

type Dataloader = std::sync::Arc<dyn DataLoader<ProductBatch> + 'static>;

/// What trains the model: `(inputs, targets)` in, the loss and the scored
/// `(logits, targets)` out ([`train_loss`]).
pub type ClassTrainer =
    Trainer<MambaLatentNet, (Tensor<3>, Tensor<2, Int>), (Tensor<2>, Tensor<1, Int>)>;

/// Train `trainer` for (the rest of) one epoch, one step per batch,
/// checkpointing and validating (on `valid_loaders`) at the `session`'s
/// cadence. Ends early once the session's budget (`--max-batches` /
/// `--max-seconds`) runs out.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train(
    dataloader_train: Dataloader,
    trainer: &mut ClassTrainer,
    training_config: &TrainingConfig,
    model_config: &MambaLatentNetConfig,
    optim: &mut ModuleOptimizer,
    session: &mut Session,
    epoch: usize,
    valid_loaders: &[(&str, Dataloader)],
    app_args: &AppArgs,
) {
    let batches = dataloader_train.num_items().div_ceil(training_config.batch_size);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

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
        let pre_metrics = ClassificationOutput::new(loss, logits, targets);

        loss_metric.update(&pre_metrics.adapt(), session.meta());
        acc_metric.update(&pre_metrics.adapt(), session.meta());
        iteration_speed_metric.update(&pre_metrics.adapt(), session.meta());

        let (loss, acc) = (
            metric_current(loss_metric.value()),
            metric_current(acc_metric.value()),
        );
        session.log_train(&[("loss", loss), ("acc", acc)]);
        println!(
            "Epoch {}/{}, Batch {b:0>4}/{batches}, Loss {loss:.4}, Acc {acc:0>6.2}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            training_config.num_epochs,
            metric_current(iteration_speed_metric.value()),
        );

        if session.checkpoint_due() {
            app_args.save_model(&trainer.module());
            app_args.save_optim(optim, session.progress());
        }
        if session.valid_due() {
            println!("running validation...");
            let valid_model = trainer.module().valid();
            validate_all(valid_loaders, valid_model, model_config, epoch, session);
        }
        if session.is_exhausted() {
            break;
        }
    }

    println!(
        "Epoch {}/{}, Avg Loss {:.4}, Avg Acc: {}",
        epoch,
        training_config.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );
    session.end_epoch(batches);
}

/// Validate on every family in turn (each capped at the session's
/// `valid_batches`), one line and one metrics-log entry each.
pub fn validate_all(
    loaders: &[(&str, Dataloader)],
    valid_model: MambaLatentNet,
    model_config: &MambaLatentNetConfig,
    epoch: usize,
    session: &mut Session,
) {
    let valid_model = Wrap(valid_model, model_config.clone());
    let limit = session.cadence().valid_batches.unwrap_or(usize::MAX);
    for (name, loader) in loaders {
        let (loss, acc) = evaluate(std::sync::Arc::clone(loader), &valid_model, limit);
        session.log_valid(name, &[("loss", loss), ("acc", acc)]);
        println!("  epoch {epoch}, {name:<10} loss {loss:.4}, acc {acc:6.2}%");
    }
}

/// Average loss and accuracy of `model` over (up to `limit` batches of) one
/// dataloader.
pub fn evaluate(dataloader: Dataloader, model: &Wrap, limit: usize) -> (f64, f64) {
    let device = model.0.devices().remove(0);
    let metric_meta = MetricMetadata {
        progress: Progress::new(0, dataloader.num_items(), None),
        iteration: Some(0),
        lr: None,
    };
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();

    for batch in dataloader.iter().map(|b| b.expect("dataloader batch")).take(limit) {
        let pre_metrics = InferenceStep::step(model, batch.to_device(&device));
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
    }
    (
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    )
}

/// Wrapper over [`MambaLatentNet`] for the validation step.
pub struct Wrap(pub MambaLatentNet, pub MambaLatentNetConfig);

impl InferenceStep for Wrap {
    type Input = ProductBatch;
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

/// Forward the model and score the group element at **every** token.
pub fn forward_classification(
    model: &MambaLatentNet,
    inputs: Tensor<3>,
    targets: Tensor<2, Int>,
) -> ClassificationOutput {
    let [batch_size, sequence_size, _input_size] = inputs.dims();
    assert_eq!([batch_size, sequence_size], targets.dims());

    let (output, _caches) = model.forward(inputs, None, ssd_path(), None, None);
    assert_eq!([batch_size, sequence_size, NUM_CLASSES], output.dims());

    let n = batch_size * sequence_size;
    let logits = output.reshape([n, NUM_CLASSES]);
    let targets = targets.reshape([n]);

    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(logits.clone(), targets.clone());

    ClassificationOutput::new(loss.clone(), logits, targets)
}
