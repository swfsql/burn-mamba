//! Training loop for the character-level TinyStories LM: builds the window
//! dataloaders, runs the train/validate epochs, checkpoints the model and
//! optimizer, and samples a story at every validation point so the text can be
//! watched growing legible. The [`Wrap`] newtype adapts the network to Burn's
//! `TrainStep` / `InferenceStep` via next-character cross-entropy over **every**
//! position of the window.

pub use crate::common::{
    cli::AppArgs,
    model::ModelConfigExt,
    training::{TrainingConfig, metric_current},
};
use crate::dataset::{Split, TinyStoriesBatch, TinyStoriesBatcher, TinyStoriesDataset, VOCAB_SIZE};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder, Progress},
    module::AutodiffModule,
    optim::ModuleOptimizer,
    train::metric::{Adaptor, Metric, MetricMetadata, Numeric},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_mamba::prelude::*;

/// The example's configuration: the shared training hyperparameters plus the
/// corpus knobs (which decide both the dataloader and what gets downloaded).
#[derive(Config, Debug)]
pub struct TinyStoriesConfig {
    /// Optimizer, epochs, batch size, LR schedule, seed.
    pub training: TrainingConfig,
    /// Characters per training window (the BPTT length).
    #[config(default = 256)]
    pub seq_len: usize,
    /// Stories pulled from the train split (~820 characters each).
    #[config(default = 4096)]
    pub train_stories: usize,
    /// Stories pulled from the validation split.
    #[config(default = 256)]
    pub valid_stories: usize,
    /// Characters generated at each sampling point.
    #[config(default = 400)]
    pub sample_chars: usize,
    /// Softmax temperature for those samples.
    #[config(default = 0.8)]
    pub sample_temperature: f64,
}

/// Run the full training routine: load/init the model and optimizer, then train
/// for the configured number of epochs (validating, sampling and checkpointing
/// along the way).
pub fn train(
    config: TinyStoriesConfig,
    model_config: MambaVocabNetConfig,
    training_device: Device,
    app_args: &AppArgs,
) {
    training_device.seed(config.training.seed);

    // load (or init and save) model and optim
    let model: MambaVocabNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let muon_plan = ModelConfigExt::muon_plan(&model_config);
    if config.training.optimizer.muon.is_some() {
        // Which weights Muon took over (and where the fused ones split).
        print!("{}", muon_plan.describe(&model));
    }
    let mut optim = app_args.load_or_save_optim(config.training.optimizer.init(&muon_plan));

    let mut model = model;

    // Create the batcher
    let batcher = TinyStoriesBatcher::default();

    // Create the dataloaders. Training batches must live on the autodiff device
    // (to match the model weights); validation runs on the inner backend.
    let train_set = TinyStoriesDataset::new(Split::Train, config.train_stories, config.seq_len);
    let valid_set = TinyStoriesDataset::new(Split::Valid, config.valid_stories, config.seq_len);
    println!(
        "corpus: {} train / {} valid characters ({} / {} windows of {})",
        train_set.num_tokens(),
        valid_set.num_tokens(),
        burn_dataset::Dataset::len(&train_set),
        burn_dataset::Dataset::len(&valid_set),
        config.seq_len,
    );
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.training.batch_size)
        .shuffle(config.training.seed)
        .num_workers(config.training.num_workers)
        .set_device(training_device.clone())
        .build(train_set);
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.training.batch_size)
        .shuffle(config.training.seed)
        .num_workers(config.training.num_workers)
        .set_device(training_device.clone().inner())
        .build(valid_set);

    let training_num_items = dataloader_train.num_items();

    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, training_num_items, None),
        iteration: Some(0),
        lr: Some(config.training.lr.get_lr(0).into()),
    };

    println!("running small initial validation...");
    epoch_valid(
        std::sync::Arc::clone(&dataloader_valid),
        model.valid(),
        &config,
        0,
        Some(10),
    );

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in 1..config.training.num_epochs + 1 {
        model = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            std::sync::Arc::clone(&dataloader_valid),
            model,
            &config,
            &mut optim,
            &mut metric_meta,
            epoch,
            None,
            Some(10),
            app_args,
            training_device.clone().inner(),
        );

        // save assets
        app_args.save_model(&model);
        app_args.save_optim(&optim);

        println!("running full validation...");
        epoch_valid(
            std::sync::Arc::clone(&dataloader_valid),
            model.valid(),
            &config,
            epoch,
            None,
        );
    }
    println!("Training finished.");
}

type Dataloader = std::sync::Arc<dyn DataLoader<TinyStoriesBatch> + 'static>;

/// Train for a single epoch, stepping the optimizer per batch and periodically
/// validating, sampling and checkpointing; returns the updated model.
#[allow(clippy::too_many_arguments)]
pub fn epoch_train(
    dataloader_train: Dataloader,
    dataloader_valid: Dataloader,
    training_model: MambaVocabNet,
    config: &TinyStoriesConfig,
    optim: &mut ModuleOptimizer,
    metric_meta: &mut MetricMetadata,
    epoch: usize,
    training_loop_limit: Option<usize>,
    valid_loop_limit: Option<usize>,
    app_args: &AppArgs,
    valid_device: Device,
) -> MambaVocabNet {
    let training_loop_limit = training_loop_limit.unwrap_or(usize::MAX);
    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();
    let mut iteration_speed_metric = burn::train::metric::IterationSpeedMetric::new();

    let mut training_model = Wrap(training_model);

    // training loop
    for (mut b, batch) in dataloader_train
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .enumerate()
        .take(training_loop_limit)
    {
        b += 1;
        let [batch_size, _seq_len] = batch.inputs.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

        let train_output = TrainStep::step(&training_model, batch);
        let pre_metrics = &train_output.item;

        loss_metric.update(&pre_metrics.adapt(), metric_meta);
        acc_metric.update(&pre_metrics.adapt(), metric_meta);
        iteration_speed_metric.update(&pre_metrics.adapt(), metric_meta);

        let lr = config.training.lr.get_lr(metric_meta.iteration.unwrap());
        training_model.0 = optim.step(lr, training_model.0, train_output.grads);

        let loss = metric_current(loss_metric.value());
        println!(
            "Epoch {}/{}, Batch {b:0>4}/{}, Loss {loss:.4} ({:.3} bits/char), \
             Acc {:0>6.2}, lr {lr:0>6.2e}, it/s {:.2}",
            epoch,
            config.training.num_epochs,
            dataloader_train.num_items() / config.training.batch_size + 1,
            loss / std::f64::consts::LN_2,
            metric_current(acc_metric.value()),
            metric_current(iteration_speed_metric.value()),
        );

        if b % 100 == 0 {
            // save assets
            app_args.save_model(&training_model.0);
            app_args.save_optim(optim);

            println!("running validation (batch iteration limit: {valid_loop_limit:?})");
            let valid_model = training_model.0.valid();
            epoch_valid(
                std::sync::Arc::clone(&dataloader_valid),
                valid_model.clone(),
                config,
                epoch,
                valid_loop_limit,
            );

            // Sample a story into a fresh per-step file, to watch the text
            // sharpen from noise into words into sentences. The sampler is
            // re-seeded identically every time, so successive samples differ by
            // the model alone.
            let sample_path = app_args
                .artifacts_path
                .join(format!("sample-epoch-{epoch}-batch-{b}.txt"));
            let sample = crate::inference::generate(
                &valid_model,
                &valid_device,
                crate::dataset::STORY_SEPARATOR,
                config.sample_chars,
                config.sample_temperature,
                config.training.seed,
            );
            std::fs::write(&sample_path, &sample).expect("failed to write the sample");
            println!("--- sample ---\n{sample}\n--- saved to {sample_path:?} ---");
        }
    }

    // Display the averaged training metrics
    println!(
        "Epoch {}/{}, Avg Loss {:.4}, Avg Acc: {}",
        epoch,
        config.training.num_epochs,
        metric_current(loss_metric.running_value()),
        metric_current(acc_metric.running_value()),
    );

    training_model.0
}

/// Run validation over (up to `valid_loop_limit`) batches and report the average
/// loss (also as bits per character) and next-character accuracy.
pub fn epoch_valid(
    dataloader_valid: Dataloader,
    valid_model: MambaVocabNet,
    config: &TinyStoriesConfig,
    epoch: usize,
    valid_loop_limit: Option<usize>,
) {
    let valid_loop_limit = valid_loop_limit.unwrap_or(usize::MAX);
    let valid_num_items = dataloader_valid.num_items();
    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, valid_num_items, None),
        iteration: Some(0),
        lr: Some(config.training.lr.get_lr(0).into()),
    };

    let mut loss_metric = burn::train::metric::LossMetric::new();
    let mut acc_metric = burn::train::metric::AccuracyMetric::new();

    let valid_model = Wrap(valid_model);

    // validation loop
    for (_b, batch) in dataloader_valid
        .iter()
        .map(|batch| batch.expect("dataloader batch"))
        .enumerate()
        .take(valid_loop_limit)
    {
        let [batch_size, _seq_len] = batch.inputs.dims();
        metric_meta.iteration = Some(metric_meta.iteration.unwrap() + 1);
        metric_meta.progress.items_processed += batch_size;

        let pre_metrics = InferenceStep::step(&valid_model, batch);
        loss_metric.update(&pre_metrics.adapt(), &metric_meta);
        acc_metric.update(&pre_metrics.adapt(), &metric_meta);
    }

    // Display the averaged validation metrics
    let loss = metric_current(loss_metric.running_value());
    println!(
        "Epoch {}/{}, Avg Valid Loss {loss:.4} ({:.3} bits/char), Avg Valid Acc: {}",
        epoch,
        config.training.num_epochs,
        loss / std::f64::consts::LN_2,
        metric_current(acc_metric.running_value()),
    );
}

/// Wrapper over [`MambaVocabNet`] for custom implementations.
pub struct Wrap(pub MambaVocabNet);

impl TrainStep for Wrap {
    type Input = TinyStoriesBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pre_metrics = InferenceStep::step(self, batch);
        let grads = pre_metrics.loss.backward();

        TrainOutput::new(&self.0, grads, pre_metrics)
    }
}

impl InferenceStep for Wrap {
    type Input = TinyStoriesBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_lm(batch.inputs, batch.targets)
    }
}

impl Wrap {
    /// Forward the LM and score **every** position of the window against its
    /// next character: `[batch, seq]` ids in, `[batch·seq, VOCAB_SIZE]` logits
    /// out, flattened so one window contributes `seq_len` classification
    /// examples (which is also what makes the accuracy metric per-character).
    pub fn forward_lm(
        &self,
        inputs: Tensor<2, Int>,
        targets: Tensor<2, Int>,
    ) -> ClassificationOutput {
        let [batch_size, seq_len] = inputs.dims();
        assert_eq!([batch_size, seq_len], targets.dims());

        // saves ~1/3 vram against Minimal
        let ssd_path = MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None));
        let (logits, _caches) = self.0.forward(inputs, None, ssd_path, None);
        assert_eq!([batch_size, seq_len, VOCAB_SIZE], logits.dims());

        // One example per (window, position): the LM's loss is the mean over
        // every next-character prediction, not just the window's last one.
        let logits = logits.reshape([batch_size * seq_len, VOCAB_SIZE]);
        let targets = targets.reshape([batch_size * seq_len]);

        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss, logits, targets)
    }
}
