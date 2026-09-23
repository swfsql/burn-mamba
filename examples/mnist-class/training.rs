//! Training loop for the sequential-MNIST classifier: builds the dataloaders,
//! runs the train/validate epochs, and checkpoints the model and optimizer.
//!
//! The epoch loops themselves are `burn_stack::examples::mnist::classify`,
//! shared with `burn-deltanet`. What is Mamba's here is the [`Wrap`] type: it
//! trains the example network through a cross-entropy classification head on
//! the last timestep, and supplies the `MnistModel` seam the shared loops build
//! against. Under SGD its training step replays from a captured graph (a
//! `Trainer`), and its validation side is [`Valid`], which replays the forward
//! from one. `--no-graph` turns both off.

pub use crate::common::{
    cli::AppArgs,
    device::loader_device,
    mnist::dataset::{HEIGHT, MnistBatch, MnistBatcher, MnistDataset, WIDTH},
    model::ModelConfigExt,
    training::TrainingConfig,
};
use burn::prelude::*;
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::ModuleOptimizer,
    train::{ClassificationOutput, InferenceStep},
};
use burn_mamba::prelude::*;
use burn_stack::examples::mnist::classify::{self, MnistModel, epoch_train, epoch_valid};
use burn_stack::examples::trainer::Trainer;
use burn_stack::utils::CapturedStep;
use std::cell::RefCell;

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
    let (mut optim, progress) =
        app_args.load_or_save_optim(training_config.optimizer.init(&muon_plan));

    let mut model = Wrap::new(model, &training_config, app_args.graphs());

    // Create the batcher
    let batcher = MnistBatcher::default();

    // Create the dataloaders. The workers build batches on the host, and the
    // loops move them to the device: a worker uploading to the GPU from its own
    // thread can invalidate a graph being captured (see `loader_device`).
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(training_config.batch_size)
        .shuffle(progress.shuffle_seed(training_config.seed))
        .num_workers(training_config.num_workers)
        .set_device(loader_device(&training_device))
        .build(MnistDataset::train());
    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .set_device(loader_device(&training_device))
        .build(MnistDataset::test());

    // Resume position, budget, cadence and metrics log.
    let mut session = app_args.session(
        progress,
        &training_config,
        classify::CADENCE,
        dataloader_train.num_items(),
    );

    println!("running small initial validation...");
    epoch_valid(
        std::sync::Arc::clone(&dataloader_valid),
        &model.valid(),
        &training_device.clone().inner(),
        &training_config,
        0,
        session.cadence().valid_batches,
        &mut session,
    );

    println!("Starting training...");
    // Iterate over our training for X epochs
    for epoch in session.epochs(training_config.num_epochs) {
        model = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            std::sync::Arc::clone(&dataloader_valid),
            model,
            &training_config,
            &mut optim,
            &mut session,
            epoch,
            app_args,
            training_device.clone().inner(),
        );

        // save assets
        model.save(app_args);
        app_args.save_optim(&optim, session.progress());

        println!("running full validation...");
        epoch_valid(
            std::sync::Arc::clone(&dataloader_valid),
            &model.valid(),
            &training_device.clone().inner(),
            &training_config,
            epoch,
            None,
            &mut session,
        );

        if session.is_exhausted() {
            println!("reached the training budget; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

/// The training-side classifier: [`MambaLatentNet`] on the autodiff backend,
/// stepped by a `Trainer` — under plain SGD (and unless `--no-graph`) the whole
/// training step replays from a graph captured at the first batch, under any
/// other optimizer it steps eagerly.
pub struct Wrap {
    trainer: Trainer<MambaLatentNet, (Tensor<4>, Tensor<1, Int>), Tensor<2>>,
    /// Whether validation replays its forward from a graph ([`Valid`]).
    graphs: bool,
}

impl Wrap {
    /// `net` trained under `training_config`'s optimizer, replaying passes from
    /// graphs if `graphs`.
    pub fn new(net: MambaLatentNet, training_config: &TrainingConfig, graphs: bool) -> Self {
        Self {
            trainer: Trainer::new(net, train_loss, &training_config.optimizer, graphs),
            graphs,
        }
    }

    /// The current weights (a copy, when a graph holds them).
    pub fn net(&self) -> MambaLatentNet {
        self.trainer.module()
    }
}

/// The training step's loss on `(images, targets)` (inner backend), and the
/// logits it read.
fn train_loss(net: &MambaLatentNet, (images, targets): (Tensor<4>, Tensor<1, Int>)) -> (Tensor<1>, Tensor<2>) {
    let batch = MnistBatch {
        images: images.autodiff(),
        targets: targets.autodiff(),
    };
    let output = classify::classification_output(logits(net, pixel_sequence(&batch)), batch.targets);
    (output.loss, output.output.inner())
}

/// The forward path used for both training and inference: it saves ~1/3 of the
/// vram against `Minimal`.
pub fn ssd_path() -> MambaSsdPath {
    MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(None))
}

impl MnistModel for Wrap {
    type Valid = Valid;

    fn valid(&self) -> Self::Valid {
        Valid::new(self.net().valid(), self.graphs)
    }

    fn train_step(&mut self, batch: MnistBatch, optim: &mut ModuleOptimizer, lr: f64) -> ClassificationOutput {
        let targets = batch.targets;
        let (loss, logits) = self.trainer.step((batch.images, targets.clone()), optim, lr);
        ClassificationOutput::new(loss, logits, targets)
    }

    fn save(&self, app_args: &AppArgs) {
        app_args.save_model(&self.net());
    }

    fn predict(valid: &Self::Valid, images_norm: Tensor<4>) -> Tensor<2> {
        crate::inference::predict(&valid.net, images_norm)
    }
}

/// The validation-side classifier: [`Wrap`]'s network on the inner backend,
/// whose forward is captured at the first batch's shape and replayed from the
/// graph for every later batch of that shape (any other shape — a short last
/// batch — runs eagerly). `--no-graph` turns the capture off.
///
/// One capture per validation pass, since the weights change between them, each
/// costing the 4 forwards `CapturedStep::capture` runs before it records.
/// Without hardware graphs (flex) the capture falls back to eager, and the
/// recording runs too: 5 forwards per pass is all it costs there.
pub struct Valid {
    net: MambaLatentNet,
    captured: RefCell<Option<CapturedLogits>>,
    capture: bool,
}

/// [`logits`] over a fixed-shape pixel sequence, captured.
type CapturedLogits = CapturedStep<'static, Tensor<3>, Tensor<2>, ()>;

impl Valid {
    fn new(net: MambaLatentNet, capture: bool) -> Self {
        Self {
            net,
            captured: RefCell::new(None),
            capture,
        }
    }

    fn logits(&self, input: Tensor<3>) -> Tensor<2> {
        if !self.capture {
            return logits(&self.net, input);
        }
        let mut slot = self.captured.borrow_mut();
        let captured = slot.get_or_insert_with(|| {
            let net = self.net.clone();
            // Safety: the forward reads nothing but its argument and `net`,
            // which it owns and never changes.
            unsafe {
                CapturedStep::capture(&input.device(), input.clone(), (), move |x, ()| {
                    (logits(&net, x), ())
                })
            }
        });
        if captured.input_dims() != input.dims() {
            return logits(&self.net, input);
        }
        // Copied out of the graph's output buffer, which the next replay overwrites.
        let y = captured.step(input);
        y.empty_like().slice_assign(y.dims().map(|d| 0..d), y.clone())
    }
}

impl InferenceStep for Valid {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        let logits = self.logits(pixel_sequence(&batch));
        classify::classification_output(logits, batch.targets)
    }
}

/// A batch's z-scored pixels as a sequence of single-pixel tokens,
/// `[batch, HEIGHT * WIDTH, 1]`.
fn pixel_sequence(batch: &MnistBatch) -> Tensor<3> {
    let input = batch.images_z_score(); // values mean=0, stddev=1
    let [batch_size, HEIGHT, WIDTH, 1] = input.dims() else {
        panic!()
    };
    input.reshape([batch_size, HEIGHT * WIDTH, 1])
}

/// Forward the model and return the last timestep's `[batch, 10]` logits, which
/// the cross-entropy classification loss reads.
pub fn logits(model: &MambaLatentNet, input: Tensor<3>) -> Tensor<2> {
    let [batch_size, sequence_size, input_size] = input.dims();
    assert_eq!(sequence_size, HEIGHT * WIDTH);
    assert_eq!(input_size, 1);

    let (output, _caches) = model.forward(input, None, ssd_path(), None, None);
    // The model's class latents lengthen the sequence; the readout is its
    // last position (see `OUTPUT_SEQUENCE_EXTRA`).
    let output_size = sequence_size + OUTPUT_SEQUENCE_EXTRA;
    assert_eq!([batch_size, output_size, 10], output.dims());
    let last_output = output.narrow(1, output_size - 1, 1).squeeze_dim(1);
    assert_eq!([batch_size, 10], last_output.dims());
    last_output
}
