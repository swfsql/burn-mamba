//! Training loop for the character-level H-Net. The epoch loops are those of
//! `burn_stack::examples::tiny_stories::lm`, shared with `tiny-stories`. The
//! H-Net part is the [`Wrap`] type, which implements their `LmModel`:
//!
//! - A window runs from the carried caches. The first window of a story has
//!   none. It opens the story with the start rows of the network
//!   (`forward_opened`). `lm_output` scores the last of them against the
//!   first character.
//! - The loss that back-propagates is the next-character loss plus `α` times
//!   the ratio loss of each stage. The metrics (nats and bits per character)
//!   report the next-character loss alone.
//! - The optimizer step gives each stage its own learning rate: the rate of
//!   the schedule times `λˢ` (see [`crate::model::lr_multipliers`]).

pub use crate::common::{cli::AppArgs, tiny_stories::lm::TinyStoriesConfig};
use crate::cli::Cli;
use crate::dataset::{PackLayout, TinyStoriesBatch};
use crate::model::HNetStoriesConfig;
use burn::lr_scheduler::module_lr_scheduler::ModuleLrScheduler;
use burn::module::ParamGroup;
use burn::prelude::*;
use burn::{
    optim::{GradientsParams, ModuleOptimizer},
    train::{ClassificationOutput, TrainOutput},
};
use burn_mamba::prelude::*;
use burn_stack::examples::tiny_stories::lm::{self, Frontier, LmModel, dataloaders_for, epoch_train, epoch_valid};
use burn_stack::examples::{prof, vram};
use burn_stack::modules::ModelConfigExt;
use burn_stack::modules::hnet::{Routing, StepMode};
use burn_stack::utils::ClassCursors;
use std::sync::atomic::{AtomicU64, Ordering};

/// Run the full training routine. Load or initialize the model and the
/// optimizer, then train for the configured number of epochs, with
/// validations, samples and checkpoints on the way.
pub fn train(
    config: TinyStoriesConfig,
    model_config: HNetStoriesConfig,
    training_device: Device,
    app_args: &AppArgs,
    run: Run,
) {
    training_device.seed(config.training.seed);

    let model: MambaHNetVocabNet = app_args.load_or_save_model(&model_config, &training_device);
    println!("Number of parameters: {}", model.num_params());
    let muon_plan = ModelConfigExt::muon_plan(&model_config);
    if config.training.optimizer.muon.is_some() {
        print!("{}", muon_plan.describe(&model));
    }
    let (mut optim, progress) = app_args.load_or_save_optim(config.training.optimizer.init(&muon_plan));
    let mut model = Wrap(model, run);

    let (dataloader_train, dataloader_valid) =
        dataloaders_for(&config, &training_device, &progress, PackLayout::default());
    let mut session = app_args.session(progress, &config.training, lm::CADENCE, dataloader_train.num_items());
    let mut frontier = Frontier::new(config.frontier.clone());

    println!("running small initial validation...");
    epoch_valid::<Wrap>(
        std::sync::Arc::clone(&dataloader_valid),
        &model.valid(),
        &training_device.clone().inner(),
        &config,
        0,
        session.cadence().valid_batches,
        &mut session,
    );

    println!("Starting training...");
    for epoch in session.epochs(config.training.num_epochs) {
        model = epoch_train(
            std::sync::Arc::clone(&dataloader_train),
            std::sync::Arc::clone(&dataloader_valid),
            model,
            &config,
            &mut optim,
            &mut session,
            &mut frontier,
            epoch,
            app_args,
            training_device.clone().inner(),
        );
        app_args.save_model(&model.0);
        app_args.save_optim(&optim, session.progress());

        println!("running full validation...");
        epoch_valid::<Wrap>(
            std::sync::Arc::clone(&dataloader_valid),
            &model.valid(),
            &training_device.clone().inner(),
            &config,
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

/// How the network runs and trains, in training and inference alike.
#[derive(Clone, Debug)]
pub struct Run {
    /// The SSD paths of every chunkwise `forward`: (the stages, the main
    /// network), from `--ssd-path` and `--main-ssd-path`.
    pub paths: (MambaSsdPath, MambaSsdPath),
    /// `--max-vram`, in MiB: the cap of [`vram`], with its cleanups.
    pub max_vram_mib: Option<u64>,
    /// `--vram-cleanup-every`: the period of those cleanups (`None`: the
    /// default schedule of [`vram::cleanup`]).
    pub vram_cleanup_every: Option<u64>,
    /// `--step-mode`: how a decode step runs the inner network of a stage.
    pub step_mode: StepMode,
    /// Whether decode steps replay a captured graph (`--no-graph` runs them
    /// eagerly). Only a masked step can.
    pub graphs: bool,
    /// `--show-chunks`: mark the chunk starts in each sample.
    pub show_chunks: bool,
    /// `--chunk-log`: print the chunk ratios once per this many windows (`0`
    /// never).
    pub chunk_log: usize,
    /// `Nˢ` of each stage.
    pub ratio_targets: Vec<f64>,
    /// `α`.
    pub ratio_weight: f64,
    /// `λˢ` of each stage, then of the main network.
    pub lr_multipliers: Vec<f64>,
}

impl Run {
    /// The settings of a run of `config`, with the flags of this invocation.
    pub fn new(config: &HNetStoriesConfig, cli: &Cli, graphs: bool) -> Self {
        let n_stages = config.model.stage_blocks.len();
        assert_eq!(config.ratio_targets.len(), n_stages, "one ratio target per stage");
        assert_eq!(config.lr_multipliers.len(), n_stages + 1, "one LR multiplier per stage, then the main network");
        assert_eq!(
            config.model.families(),
            ("mamba3", "mamba3"),
            "this example runs Mamba-3 in every stack"
        );
        Self {
            paths: cli.paths.clone(),
            max_vram_mib: cli.step.max_vram_mib,
            vram_cleanup_every: cli.step.vram_cleanup_every,
            step_mode: cli.step_mode,
            graphs,
            show_chunks: cli.show_chunks,
            chunk_log: cli.chunk_log,
            ratio_targets: config.ratio_targets.clone(),
            ratio_weight: config.ratio_weight,
            lr_multipliers: config.lr_multipliers.clone(),
        }
    }
}

/// The training windows so far: the cadence of `--chunk-log` and of the
/// `--max-vram` cleanups.
static TRAIN_WINDOWS: AtomicU64 = AtomicU64::new(0);

/// The H-Net with its [`Run`] settings.
pub struct Wrap(pub MambaHNetVocabNet, pub Run);

impl Wrap {
    /// One window: the logits, scored against the next characters, the
    /// caches, and the routing of each stage.
    fn forward_lm(
        &self,
        batch: TinyStoriesBatch,
        caches: Option<MambaHNetCaches>,
    ) -> (ClassificationOutput, MambaHNetCaches, Vec<Routing>) {
        let TinyStoriesBatch { inputs, targets, scored, packed, .. } = batch;
        assert!(packed.is_none(), "an H-Net takes no packed rows");
        let [batch_size, seq_len, 2] = inputs.dims() else { unreachable!() };
        // The rows past the scored count of a slot are padding.
        let pad = scored.iter().any(|&n| n < seq_len).then(|| {
            let device = inputs.device();
            let scored: Vec<i32> = scored.iter().map(|&n| n as i32).collect();
            Tensor::<1, Int>::arange(0..seq_len as i64, &device)
                .reshape([1, seq_len])
                .expand([batch_size, seq_len])
                .greater_equal(
                    Tensor::<1, Int>::from_ints(scored.as_slice(), &device)
                        .reshape([batch_size, 1])
                        .expand([batch_size, seq_len]),
                )
        });
        // A window with no carried cache is the first of its story.
        let (logits, caches, routing) = match caches {
            None => self.0.forward_opened(inputs.clone(), self.1.paths.clone(), pad),
            Some(caches) => self.0.forward(inputs.clone(), Some(caches), self.1.paths.clone(), pad),
        };
        (lm::lm_output(logits, inputs, targets, &scored), caches, routing)
    }

    /// `α · Σₛ L_ratioˢ`, `[1]`.
    fn ratio_loss(&self, routing: &[Routing]) -> Tensor<1> {
        routing
            .iter()
            .zip(&self.1.ratio_targets)
            .map(|(r, &target)| r.ratio_loss(target))
            .reduce(|a, b| a + b)
            .expect("an H-Net of this example has at least one stage")
            * self.1.ratio_weight
    }
}

/// Print the chunk ratio `F` of each stage (the share of its rows that start
/// a chunk), against its target `1/Nˢ`.
fn log_chunks(routing: &[Routing], targets: &[f64]) {
    let stages: Vec<String> = routing
        .iter()
        .zip(targets)
        .enumerate()
        .map(|(s, (r, target))| {
            let (f, g) = r.fractions();
            format!(
                "L{}/L{s} = {:.3} (target {:.3}, mean p {:.3})",
                s + 1,
                f.into_scalar::<f32>(),
                1.0 / target,
                g.into_scalar::<f32>()
            )
        })
        .collect();
    println!("chunks: {}", stages.join(", "));
}

impl LmModel for Wrap {
    type Valid = Wrap;
    type Caches = MambaHNetCaches;

    fn valid(&self) -> Self::Valid {
        Wrap(self.0.valid(), self.1.clone())
    }

    /// The class cursors are not used: the start rows open the first window
    /// of each story, which is the window with no carried cache.
    fn train_window(
        &self,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
        _class: &mut ClassCursors,
    ) -> (TrainOutput<ClassificationOutput>, Self::Caches) {
        let device = batch.inputs.device();
        let window = TRAIN_WINDOWS.fetch_add(1, Ordering::Relaxed);
        if self.1.max_vram_mib.is_some() {
            vram::cleanup(&device, window, self.1.vram_cleanup_every);
        }
        let t0 = prof::window_start();
        let (output, caches, routing) = self.forward_lm(batch, caches);
        if self.1.chunk_log > 0 && window % self.1.chunk_log as u64 == 0 {
            log_chunks(&routing, &self.1.ratio_targets);
        }
        let loss = output.loss.clone() + self.ratio_loss(&routing);
        let t1 = prof::mark(&device);
        let grads = GradientsParams::from_grads(loss.backward(), &self.0);
        let t2 = prof::mark(&device);
        if let Some(max_mib) = self.1.max_vram_mib {
            vram::check(&device, max_mib);
        }
        prof::window_end(&device, t0, t1, t2);
        (TrainOutput { grads, item: output }, caches)
    }

    fn detach_caches(caches: Self::Caches) -> Self::Caches {
        caches.detach()
    }

    fn valid_window(
        valid: &Self::Valid,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
        _class: &mut ClassCursors,
    ) -> (ClassificationOutput, Self::Caches) {
        let (output, caches, _) = valid.forward_lm(batch, caches);
        (output, caches)
    }

    /// One step at the rate `lr · λˢ` for the parameters of each stage, and
    /// `lr · λᴹ` for those of the main network. The embedding and the heads
    /// take the rate of stage 0.
    fn optim_step(self, optim: &mut ModuleOptimizer, lr: f64, grads: GradientsParams) -> Self {
        let multipliers = &self.1.lr_multipliers;
        let n_stages = multipliers.len() - 1;
        let mut rates = ModuleLrScheduler::new(lr * multipliers[0]);
        for (s, multiplier) in multipliers.iter().enumerate().skip(1) {
            let scope = match s == n_stages {
                true => "hnet.main.".to_string(),
                false => format!("hnet.stages.{s}."),
            };
            rates = rates.with_group(ParamGroup::from_predicate(scope), lr * multiplier);
        }
        let t3 = prof::opt_start();
        let model = Wrap(optim.step(rates.step(), self.0, grads), self.1);
        prof::opt_end(t3);
        model
    }

    fn save(&self, app_args: &AppArgs) {
        app_args.save_model(&self.0);
    }

    fn generate(
        valid: &Self::Valid,
        device: &Device,
        prompt: Option<&str>,
        n_chars: usize,
        temperature: f64,
        seed: u64,
    ) -> String {
        crate::inference::generate(&valid.0, &valid.1, device, prompt, n_chars, temperature, seed)
    }
}
