//! Training loop for the reset-majority example. It builds the dataloaders,
//! runs the train and validation epochs, and saves the model and the optimizer.
//! The objective is a cross-entropy head over **every** position (the running
//! vote at each step, [`forward_classification`]). A `Trainer` steps it. Under
//! plain SGD (and without `--no-graph`), the whole training step replays from a
//! graph captured at the first batch. Under any other optimizer, it steps
//! eagerly.

pub use crate::common::{
    cli::AppArgs,
    device::loader_device,
    model::ModelConfigExt,
    session::{Cadence, Session},
    training::{TrainingConfig, metric_current},
};
use crate::dataset::{
    EVAL_SEED, Family, IGNORE, NUM_CLASSES, NUM_EVAL, NUM_TRAIN, ResetMajorityBatch,
    ResetMajorityBatcher, ResetMajorityDataset, SEQ_LENGTH, TRAIN_SEED,
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

/// The evaluation splits, reported separately. A non-selective state fails on
/// the two adversarial families, so one averaged number would hide the result
/// (see [`crate::dataset`]).
pub const EVAL_FAMILIES: [(&str, Family); 3] = [
    ("random", Family::Random),
    ("long-prefix", Family::LongPrefix),
    ("long-suffix", Family::LongSuffix),
];

/// The SSD path of every call in this example. The model is tiny and the
/// sequences are short, so the simplest variant (autodiff backward) is right.
pub fn ssd_path() -> MambaSsdPath {
    MambaSsdPath::Mamba3(Mamba3SsdPath::Minimal(None))
}

/// Run the full training routine. Load or initialize the model and the
/// optimizer, then train for the configured number of epochs, with validations
/// and checkpoints on the way.
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
    let batcher = ResetMajorityBatcher::default();

    // The workers build batches on the host, and the loops move them to the
    // device. A worker that uploads to the GPU from its own thread can
    // invalidate a graph capture (see `device::loader_device`).
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(progress.shuffle_seed(training_config.seed))
        .num_workers(training_config.num_workers)
        .set_device(loader_device(&training_device))
        .build(ResetMajorityDataset::new(
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
                .build(ResetMajorityDataset::new(
                    NUM_EVAL, SEQ_LENGTH, *family, EVAL_SEED,
                ));
            (*name, loader)
        })
        .collect();

    // The session holds the resume position, the budget, the cadence and the
    // metrics log. By default, it validates every five epochs and saves no
    // mid-epoch checkpoint.
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

type Dataloader = std::sync::Arc<dyn DataLoader<ResetMajorityBatch> + 'static>;

/// What trains the model: `(inputs, targets)` in, the loss and the scored
/// `(logits, targets)` out ([`train_loss`]).
pub type ClassTrainer =
    Trainer<MambaLatentNet, (Tensor<3>, Tensor<2, Int>), (Tensor<2>, Tensor<1, Int>)>;

/// Train `trainer` for (the rest of) one epoch, one step per batch. It saves
/// checkpoints and validates (on `valid_loaders`) at the cadence of the
/// `session`. It stops early at the end of the session budget
/// (`--max-batches` / `--max-seconds`).
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
    let mut acc_metric = burn::train::metric::AccuracyMetric::new().with_pad_token(NUM_CLASSES);
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

/// Validate on each family in turn, each one capped at the `valid_batches` of
/// the session. Each family gets one line and one metrics-log entry.
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
        println!("  epoch {epoch}, {name:<12} loss {loss:.4}, acc {acc:6.2}%");
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
    let mut acc_metric = burn::train::metric::AccuracyMetric::new().with_pad_token(NUM_CLASSES);

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
    type Input = ResetMajorityBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        forward_classification(&self.0, batch.inputs, batch.targets)
    }
}

/// [`forward_classification`] for the training step, on `(inputs, targets)`.
/// It returns the loss, and the scored `(logits, targets)` on the inner
/// backend.
fn train_loss(
    model: &MambaLatentNet,
    (inputs, targets): (Tensor<3>, Tensor<2, Int>),
) -> (Tensor<1>, (Tensor<2>, Tensor<1, Int>)) {
    let output = forward_classification(model, inputs.autodiff(), targets.autodiff());
    (output.loss, (output.output.inner(), output.targets.inner()))
}

/// Forward the model, and score the running vote at **every** position that
/// has a sign to report.
///
/// The zero-vote positions reach neither the loss nor the accuracy (see
/// `dataset::IGNORE`). The loss masks them out of the mean. Their targets go to
/// the pad class `NUM_CLASSES`, which no prediction matches and which
/// `AccuracyMetric::with_pad_token(NUM_CLASSES)` leaves out. They are masked,
/// not dropped, so every shape is the shape of the batch, whatever its votes.
/// A captured training step needs this: it replays one fixed shape.
pub fn forward_classification(
    model: &MambaLatentNet,
    inputs: Tensor<3>,
    targets: Tensor<2, Int>,
) -> ClassificationOutput {
    let [batch_size, sequence_size, _num_symbols] = inputs.dims();
    assert_eq!([batch_size, sequence_size], targets.dims());

    let (output, _caches) = model.forward(inputs, None, ssd_path(), None, None);
    assert_eq!([batch_size, sequence_size, NUM_CLASSES], output.dims());

    let n = batch_size * sequence_size;
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
