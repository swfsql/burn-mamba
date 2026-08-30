//! Training loop for the sequential-MNIST classifier: builds the dataloaders,
//! runs the train/validate epochs, and checkpoints the model and optimizer.
//!
//! The epoch loops themselves are `burn_stack::examples::mnist::classify`,
//! shared with `burn-deltanet`. What is Mamba's here is the [`Wrap`] newtype:
//! it adapts the example network to Burn's `TrainStep` / `InferenceStep` via a
//! cross-entropy classification head on the last timestep, and supplies the
//! `MnistModel` seam the shared loops build against.

pub use crate::common::{
    cli::AppArgs,
    mnist::dataset::{HEIGHT, MnistBatch, MnistBatcher, MnistDataset, WIDTH},
    model::ModelConfigExt,
    training::TrainingConfig,
};
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoaderBuilder, Progress},
    module::AutodiffModule,
    optim::{GradientsParams, ModuleOptimizer},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_mamba::prelude::*;
use burn_stack::examples::mnist::classify::{self, MnistModel, epoch_train, epoch_valid};

use crate::model::OUTPUT_SEQUENCE_EXTRA;

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
    let muon_plan = ModelConfigExt::muon_plan(&model_config);
    if training_config.optimizer.muon.is_some() {
        // Which weights Muon took over (and where the fused ones split).
        print!("{}", muon_plan.describe(&model));
    }
    let mut optim = app_args.load_or_save_optim(training_config.optimizer.init(&muon_plan));

    let mut model = Wrap(model);

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

    let mut metric_meta = burn::train::metric::MetricMetadata {
        progress: Progress::new(0, training_num_items, None),
        iteration: Some(0),
        lr: Some(training_config.lr.get_lr(0).into()),
    };

    println!("running small initial validation...");
    epoch_valid(
        std::sync::Arc::clone(&dataloader_valid),
        &model.valid(),
        &training_config,
        0,
        Some(10),
    );

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in 1..training_config.num_epochs + 1 {
        model = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            std::sync::Arc::clone(&dataloader_valid),
            model,
            &training_config,
            &mut optim,
            &mut metric_meta,
            epoch,
            None,
            Some(10),
            app_args,
            training_device.clone().inner(),
        );

        // save assets
        app_args.save_model(&model.0);
        app_args.save_optim(&optim);

        println!("running full validation...");
        epoch_valid(
            std::sync::Arc::clone(&dataloader_valid),
            &model.valid(),
            &training_config,
            epoch,
            None,
        );
    }
    println!("Training finished.");
}

/// Wrapper over [`MambaLatentNet`] for custom implementations.
pub struct Wrap(pub MambaLatentNet);

/// The forward path used for both training and inference: it saves ~1/3 of the
/// vram against `Minimal`.
pub fn ssd_path() -> MambaSsdPath {
    MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None))
}

impl MnistModel for Wrap {
    type Valid = Wrap;

    fn valid(&self) -> Self::Valid {
        Wrap(self.0.valid())
    }

    fn optim_step(self, optim: &mut ModuleOptimizer, lr: f64, grads: GradientsParams) -> Self {
        Wrap(optim.step(lr, self.0, grads))
    }

    fn save(&self, app_args: &AppArgs) {
        app_args.save_model(&self.0);
    }

    fn predict(valid: &Self::Valid, images_norm: Tensor<4>) -> Tensor<2> {
        crate::inference::predict(&valid.0, images_norm)
    }
}

impl TrainStep for Wrap {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let pre_metrics = InferenceStep::step(self, batch);
        let grads = pre_metrics.loss.backward();

        TrainOutput::new(&self.0, grads, pre_metrics)
    }
}

impl InferenceStep for Wrap {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let input = batch.images_z_score(); // values mean=0, stddev=1
        let [batch_size, HEIGHT, WIDTH, 1] = input.dims() else {
            panic!()
        };
        let input = input.reshape([batch_size, HEIGHT * WIDTH, 1]);
        let [_batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(sequence_size, HEIGHT * WIDTH);
        assert_eq!(input_size, 1);
        let targets = batch.targets;

        self.forward_classification(input, targets)
    }
}

impl Wrap {
    /// Forward the model and compute the cross-entropy classification loss from
    /// the last timestep's logits.
    pub fn forward_classification(
        &self,
        input: Tensor<3>,
        targets: Tensor<1, Int>,
    ) -> ClassificationOutput {
        let model = &self.0;
        let [batch_size, sequence_size, input_size] = input.dims();
        assert_eq!(sequence_size, HEIGHT * WIDTH);
        assert_eq!(input_size, 1);
        assert_eq!([batch_size], targets.dims());

        let (output, _caches) = model.forward(input.clone(), None, ssd_path(), None);
        // The model's class latents lengthen the sequence; they are all `Start`,
        // so the last position is still the last pixel — just further along.
        let output_size = sequence_size + OUTPUT_SEQUENCE_EXTRA;
        assert_eq!([batch_size, output_size, 10], output.dims());
        let last_output = output.narrow(1, output_size - 1, 1).squeeze_dim(1);
        assert_eq!([batch_size, 10], last_output.dims());

        classify::classification_output(last_output, targets)
    }
}
