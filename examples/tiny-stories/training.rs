//! Training loop for the character-level TinyStories LM: builds the window
//! dataloaders, runs the train/validate epochs, checkpoints the model and
//! optimizer, and samples a story at every validation point so the text can be
//! watched growing legible.
//!
//! The epoch loops themselves are `burn_stack::examples::tiny_stories::lm`,
//! shared with `burn-deltanet`. What is Mamba's here is the [`Wrap`] newtype: it
//! adapts the network to Burn's `TrainStep` / `InferenceStep` via
//! next-character cross-entropy over **every** position of the window, and
//! supplies the `LmModel` seam the shared loops build against.

pub use crate::common::{
    cli::AppArgs,
    model::ModelConfigExt,
    tiny_stories::lm::TinyStoriesConfig,
    training::{TrainingConfig, metric_current},
};
use crate::dataset::TinyStoriesBatch;
use burn::prelude::*;
use burn::{
    data::dataloader::Progress,
    module::AutodiffModule,
    optim::{GradientsParams, ModuleOptimizer},
    train::metric::MetricMetadata,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_mamba::prelude::*;
use burn_stack::examples::tiny_stories::lm::{self, LmModel, dataloaders, epoch_train, epoch_valid};

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

    let mut model = Wrap(model);

    // Create the dataloaders (downloading the corpus on the first run).
    let (dataloader_train, dataloader_valid) = dataloaders(&config, &training_device);

    let training_num_items = dataloader_train.num_items();

    let mut metric_meta = MetricMetadata {
        progress: Progress::new(0, training_num_items, None),
        iteration: Some(0),
        lr: Some(config.training.lr.get_lr(0).into()),
    };

    println!("running small initial validation...");
    epoch_valid(
        std::sync::Arc::clone(&dataloader_valid),
        &model.valid(),
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
        app_args.save_model(&model.0);
        app_args.save_optim(&optim);

        println!("running full validation...");
        epoch_valid(
            std::sync::Arc::clone(&dataloader_valid),
            &model.valid(),
            &config,
            epoch,
            None,
        );
    }
    println!("Training finished.");
}

/// The SSD path used for both training and inference; the recalculated serial
/// scan saves ~1/3 vram against `Minimal`.
pub fn ssd_path() -> MambaSsdPath {
    MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None))
}

/// Wrapper over [`MambaVocabNet`] for custom implementations.
pub struct Wrap(pub MambaVocabNet);

impl LmModel for Wrap {
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

    fn generate(
        valid: &Self::Valid,
        device: &Device,
        prompt: &str,
        n_chars: usize,
        temperature: f64,
        seed: u64,
    ) -> String {
        crate::inference::generate(&valid.0, device, prompt, n_chars, temperature, seed)
    }
}

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
    /// next character (see
    /// [`lm_output`](burn_stack::examples::tiny_stories::lm::lm_output)).
    pub fn forward_lm(
        &self,
        inputs: Tensor<2, Int>,
        targets: Tensor<2, Int>,
    ) -> ClassificationOutput {
        let (logits, _caches) = self.0.forward(inputs, None, ssd_path(), None);
        lm::lm_output(logits, targets)
    }
}
