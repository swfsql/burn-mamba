//! Training loop for the reset-majority example: builds the dataloaders, runs
//! the train/validate epochs, and checkpoints the model and optimizer. The
//! [`Wrap`] newtype adapts the example network to Burn's `TrainStep` /
//! `InferenceStep` via a cross-entropy head over **every** position (the
//! running vote at each step).

pub use crate::common::{
    cli::AppArgs,
    model::ModelConfigExt,
    training::{BatchBudget, TrainingConfig, metric_current},
};
use crate::dataset::{
    EVAL_SEED, Family, NUM_CLASSES, NUM_EVAL, NUM_TRAIN, ResetMajorityBatch, ResetMajorityBatcher,
    ResetMajorityDataset, SEQ_LENGTH, TRAIN_SEED,
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

/// The evaluation splits, reported separately: the two adversarial families are
/// where a non-selective state fails, so a single averaged number would hide the
/// whole point (see [`crate::dataset`]).
pub const EVAL_FAMILIES: [(&str, Family); 3] = [
    ("random", Family::Random),
    ("long-prefix", Family::LongPrefix),
    ("long-suffix", Family::LongSuffix),
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
    let mut optim = app_args.load_or_save_optim(training_config.optimizer.init(&muon_plan));

    let mut model = Wrap(model, model_config.clone());
    let batcher = ResetMajorityBatcher::default();

    // Training batches live on the autodiff device (to match the weights);
    // validation runs on the inner backend.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(training_device.clone())
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
                .set_device(training_device.clone().inner())
                .build(ResetMajorityDataset::new(
                    NUM_EVAL, SEQ_LENGTH, *family, EVAL_SEED,
                ));
            (*name, loader)
        })
        .collect();

    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, dataloader_train.num_items(), None),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0).into()),
    };

    // `--max-batches`: an optional cap on the whole run, spent across epochs.
    let mut batch_budget = app_args.batch_budget();

    println!(
        "running initial validation (chance ≈ {:.1}%)...",
        100.0 / NUM_CLASSES as f32
    );
    validate_all(&valid_loaders, model.0.valid(), &model_config, 0);

    println!("Starting training...");
    for epoch in 1..training_config.num_epochs + 1 {
        model.0 = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            model.0,
            &training_config,
            &model_config,
            &mut optim,
            &mut metric_meta,
            epoch,
            &mut batch_budget,
        );

        app_args.save_model(&model.0);
        app_args.save_optim(&optim);

        if epoch % 5 == 0 || epoch == 1 || epoch == training_config.num_epochs {
            println!("running validation...");
            validate_all(&valid_loaders, model.0.valid(), &model_config, epoch);
        }

        if batch_budget.is_exhausted() {
            println!("reached the --max-batches limit; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

type Dataloader = std::sync::Arc<dyn DataLoader<ResetMajorityBatch> + 'static>;

/// Train for a single epoch, stepping the optimizer per batch; returns the
/// updated model. Ends early once `batch_budget` (`--max-batches`) runs out.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train(
    dataloader_train: Dataloader,
    training_model: MambaLatentNet,
    training_config: &TrainingConfig,
    model_config: &MambaLatentNetConfig,
    optim: &mut ModuleOptimizer,
    metric_meta: &mut MetricMetadata,
    epoch: usize,
    batch_budget: &mut BatchBudget,
) -> MambaLatentNet {
    let training_loop_limit = batch_budget.take_limit();
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

    let mut training_model = Wrap(training_model, model_config.clone());

    for (mut b, batch) in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .enumerate()
        .take(training_loop_limit)
    {
        b += 1;
        batch_budget.spend();
        let [batch_size, _, _] = batch.inputs.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

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
            metric_current(loss_metric.value()),
            metric_current(acc_metric.value()),
            metric_current(iteration_speed_metric.value()),
        );
    }

    println!(
        "Epoch {}/{}, Avg Loss {:.4}, Avg Acc: {}",
        epoch,
        training_config.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );

    training_model.0
}

/// Validate on every family in turn, one line each.
pub fn validate_all(
    loaders: &[(&str, Dataloader)],
    valid_model: MambaLatentNet,
    model_config: &MambaLatentNetConfig,
    epoch: usize,
) {
    let valid_model = Wrap(valid_model, model_config.clone());
    for (name, loader) in loaders {
        let (loss, acc) = evaluate(std::sync::Arc::clone(loader), &valid_model);
        println!("  epoch {epoch}, {name:<12} loss {loss:.4}, acc {acc:6.2}%");
    }
}

/// Average loss and accuracy of `model` over one dataloader.
pub fn evaluate(dataloader: Dataloader, model: &Wrap) -> (f64, f64) {
    let metric_meta = MetricMetadata {
        progress: Progress::new(0, dataloader.num_items(), None),
        iteration: Some(0),
        lr: None,
    };
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();

    for batch in dataloader.iter().map(|b| b.expect("dataloader batch")) {
        let pre_metrics = InferenceStep::step(model, batch);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
    }
    (
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    )
}

/// Wrapper over [`MambaLatentNet`] for custom implementations.
pub struct Wrap(pub MambaLatentNet, pub MambaLatentNetConfig);

impl TrainStep for Wrap {
    type Input = ResetMajorityBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pre_metrics = InferenceStep::step(self, batch);
        let grads = pre_metrics.loss.backward();
        TrainOutput::new(&self.0, grads, pre_metrics)
    }
}

impl InferenceStep for Wrap {
    type Input = ResetMajorityBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_classification(batch.inputs, batch.targets, batch.scored)
    }
}

impl Wrap {
    /// Forward the model and score the running vote at **every** position.
    pub fn forward_classification(
        &self,
        inputs: Tensor<3>,
        targets: Tensor<2, Int>,
        scored: Tensor<1, Int>,
    ) -> ClassificationOutput {
        let model = &self.0;
        let [batch_size, sequence_size, _num_symbols] = inputs.dims();
        assert_eq!([batch_size, sequence_size], targets.dims());

        let (output, _caches) = model.forward(inputs, None, ssd_path(), None);
        assert_eq!([batch_size, sequence_size, NUM_CLASSES], output.dims());

        // Keep only the positions that have a sign to report; the zero-vote ones
        // reach neither the loss nor the accuracy (see `dataset::IGNORE`).
        let n = batch_size * sequence_size;
        let logits = output.reshape([n, NUM_CLASSES]).select(0, scored.clone());
        let targets = targets.reshape([n]).select(0, scored);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss.clone(), logits, targets)
    }
}
