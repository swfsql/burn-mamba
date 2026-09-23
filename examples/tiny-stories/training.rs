//! Training loop for the character-level TinyStories LM. It builds the window
//! dataloaders, runs the train and validation epochs, and saves the model and
//! the optimizer. It samples a story at every validation point, so the text
//! shows how the model improves.
//!
//! The epoch loops themselves are `burn_stack::examples::tiny_stories::lm`,
//! shared with `burn-deltanet`. The Mamba part here is the [`Wrap`] type. It
//! adapts the network to the `TrainStep` / `InferenceStep` of Burn, with a
//! next-character cross-entropy over **every** position of the window. It also
//! supplies the `LmModel` interface that the shared loops use, including its
//! two cache-aware halves. The loops train a *run* of windows, and they carry
//! [`MambaCaches`] from each window into the next (see "Runs, carried state,
//! and the frontier" in that module).

pub use crate::common::{
    cli::AppArgs,
    model::ModelConfigExt,
    tiny_stories::lm::TinyStoriesConfig,
    training::{TrainingConfig, metric_current},
};
use crate::dataset::TinyStoriesBatch;
use burn::prelude::*;
use burn::{
    optim::{GradientsParams, ModuleOptimizer},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_mamba::prelude::*;
use burn_stack::examples::tiny_stories::lm::{
    self, Frontier, LmModel, dataloaders, epoch_train, epoch_valid,
};
use burn_stack::utils::ClassCursors;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Run the full training routine. Load or initialize the model and the
/// optimizer, then train for the configured number of epochs, with validations,
/// samples and checkpoints on the way.
pub fn train(
    config: TinyStoriesConfig,
    model_config: MambaVocabNetConfig,
    training_device: Device,
    app_args: &AppArgs,
    run: Run,
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
    let (mut optim, progress) =
        app_args.load_or_save_optim(config.training.optimizer.init(&muon_plan));

    let mut model = Wrap(model, run);

    // Create the dataloaders (downloading the corpus on the first run).
    let (dataloader_train, dataloader_valid) =
        dataloaders(&config, &training_device, &progress);

    // The session: resume position, budget, cadence and metrics log.
    let mut session = app_args.session(
        progress,
        &config.training,
        lm::CADENCE,
        dataloader_train.num_items(),
    );

    // The frontier gate lives longer than the epochs: its depth statistics
    // accumulate over the whole training run, not per epoch.
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
    // Train for the configured number of epochs.
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

        // save assets
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

/// How the network runs, in training and inference alike.
#[derive(Clone, Debug)]
pub struct Run {
    /// The SSD path of every chunkwise `forward` (`--ssd-path`).
    pub ssd_path: MambaSsdPath,
    /// Whether decode steps and prefill chunks replay captured graphs
    /// (`--no-graph` runs them eagerly).
    pub graphs: bool,
}

/// Wrapper over [`MambaVocabNet`] for custom implementations, with its [`Run`]
/// settings.
pub struct Wrap(pub MambaVocabNet, pub Run);

impl LmModel for Wrap {
    type Valid = Wrap;
    type Caches = MambaCaches;

    fn valid(&self) -> Self::Valid {
        Wrap(self.0.valid(), self.1.clone())
    }

    fn train_window(
        &self,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
        class: &mut ClassCursors,
    ) -> (TrainOutput<ClassificationOutput>, Self::Caches) {
        let device = batch.inputs.device();
        let t0 = prof::window_start();
        let (pre_metrics, caches) = self.forward_lm(batch, caches, class);
        let t1 = prof::mark(&device);
        let grads = pre_metrics.loss.backward();
        let t2 = prof::mark(&device);
        let output = TrainOutput::new(&self.0, grads, pre_metrics);
        prof::window_end(&device, t0, t1, t2);
        (output, caches)
    }

    fn detach_caches(caches: Self::Caches) -> Self::Caches {
        caches.detach()
    }

    fn valid_window(
        valid: &Self::Valid,
        batch: TinyStoriesBatch,
        caches: Option<Self::Caches>,
        class: &mut ClassCursors,
    ) -> (ClassificationOutput, Self::Caches) {
        valid.forward_lm(batch, caches, class)
    }

    fn optim_step(self, optim: &mut ModuleOptimizer, lr: f64, grads: GradientsParams) -> Self {
        let t3 = prof::opt_start();
        let model = Wrap(optim.step(lr, self.0, grads), self.1);
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
        let run = &valid.1;
        crate::inference::generate(&valid.0, run, device, prompt, n_chars, temperature, seed, None)
    }
}

/// Opt-in wall-clock timers over the phases of a training step. They show where
/// the time of a step goes, and whether a phase grows over a run.
/// `--profile <N>` prints the mean milliseconds of each phase once per `N`
/// windows. `--profile-sync` also syncs the device after each phase, so a phase
/// then includes its GPU execution, not only its enqueueing. Inert until
/// [`init`](prof::init).
///
/// Phases:
///
/// - `fwd`: forward + loss.
/// - `bwd`: backward.
/// - `gap`: window end → the optimizer. These are the metric reads of the loop,
///   that is its implicit sync.
/// - `opt`: the optimizer step.
/// - `out`: optimizer → next window (logging, next batch).
pub mod prof {
    use super::*;

    struct State {
        every: usize,
        sync: bool,
        n: usize,
        sums: [f64; 5],
        window_end: Option<Instant>,
        opt_end: Option<Instant>,
        device: Option<Device>,
    }

    static STATE: OnceLock<Mutex<State>> = OnceLock::new();

    /// Start timing: print the means of the phases once per `every` windows,
    /// and sync the device after each phase if `sync` is true.
    pub fn init(every: usize, sync: bool) {
        let state = Mutex::new(State {
            every,
            sync,
            n: 0,
            sums: [0.0; 5],
            window_end: None,
            opt_end: None,
            device: None,
        });
        assert!(STATE.set(state).is_ok(), "the profiler starts once");
    }

    fn ms(from: Instant, to: Instant) -> f64 {
        (to - from).as_secs_f64() * 1e3
    }

    pub fn window_start() -> Instant {
        let now = Instant::now();
        if let Some(state) = STATE.get() {
            let mut s = state.lock().unwrap();
            if let Some(end) = s.opt_end.take() {
                s.sums[4] += ms(end, now);
            }
        }
        now
    }

    pub fn mark(device: &Device) -> Instant {
        if let Some(state) = STATE.get()
            && state.lock().unwrap().sync
        {
            device.sync().expect("device sync");
        }
        Instant::now()
    }

    pub fn window_end(device: &Device, t0: Instant, t1: Instant, t2: Instant) {
        if let Some(state) = STATE.get() {
            let mut s = state.lock().unwrap();
            s.sums[0] += ms(t0, t1);
            s.sums[1] += ms(t1, t2);
            s.window_end = Some(Instant::now());
            s.device = Some(device.clone());
        }
    }

    pub fn opt_start() -> Instant {
        let now = Instant::now();
        if let Some(state) = STATE.get() {
            let mut s = state.lock().unwrap();
            if let Some(end) = s.window_end.take() {
                s.sums[2] += ms(end, now);
            }
        }
        now
    }

    pub fn opt_end(t3: Instant) {
        let Some(state) = STATE.get() else {
            return;
        };
        let mut s = state.lock().unwrap();
        if s.sync
            && let Some(device) = &s.device
        {
            device.sync().expect("device sync");
        }
        let now = Instant::now();
        s.sums[3] += ms(t3, now);
        s.opt_end = Some(now);
        s.n += 1;
        if s.n % s.every == 0 {
            let [f, b, g, o, u] = s.sums.map(|sum| sum / s.every as f64);
            // Live allocations: flat unless some launch shape varies. Each new
            // shape pins a cached metadata buffer (burn#5751), but the bytes
            // barely move.
            let allocs = match s.device.as_ref().and_then(Device::memory_pool_usage) {
                Some(usage) => format!(
                    " allocs {} ({:.1} MB)",
                    usage.number_allocs,
                    usage.bytes_in_use as f64 / 1e6
                ),
                None => String::new(),
            };
            println!(
                "prof n={} fwd {f:.1} bwd {b:.1} gap {g:.1} opt {o:.1} out {u:.1} total {:.1} ms{allocs}",
                s.n,
                f + b + g + o + u
            );
            s.sums = [0.0; 5];
        }
    }
}

impl TrainStep for Wrap {
    type Input = TinyStoriesBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        LmModel::train_window(self, batch, None, &mut ClassCursors::stream()).0
    }
}

impl InferenceStep for Wrap {
    type Input = TinyStoriesBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_lm(batch, None, &mut ClassCursors::stream()).0
    }
}

impl Wrap {
    /// Forward the LM from `caches` (`None` ⇒ a zero state), and score every
    /// **real** position of the window against its next character. In the
    /// window that opened the story, also score the readout of the class latents
    /// against the first character (see
    /// [`lm_output`](burn_stack::examples::tiny_stories::lm::lm_output)).
    /// Return the final state of the window too.
    ///
    /// `class` is the cursor of the run: it splices the latents into the first
    /// window of a story, and into no other window.
    pub fn forward_lm(
        &self,
        batch: TinyStoriesBatch,
        caches: Option<MambaCaches>,
        class: &mut ClassCursors,
    ) -> (ClassificationOutput, MambaCaches) {
        let TinyStoriesBatch {
            inputs,
            targets,
            scored,
            ..
        } = batch;
        let (logits, caches) = self
            .0
            .forward(inputs.clone(), caches, self.1.ssd_path.clone(), Some(class), None);
        (lm::lm_output(logits, inputs, targets, &scored), caches)
    }
}
