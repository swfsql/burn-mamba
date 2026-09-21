//! Training loop for the character-level TinyStories LM: builds the window
//! dataloaders, runs the train/validate epochs, checkpoints the model and
//! optimizer, and samples a story at every validation point so the text can be
//! watched growing legible.
//!
//! The epoch loops themselves are `burn_stack::examples::tiny_stories::lm`,
//! shared with `burn-deltanet`. What is Mamba's here is the [`Wrap`] newtype: it
//! adapts the network to Burn's `TrainStep` / `InferenceStep` via
//! next-character cross-entropy over **every** position of the window, and
//! supplies the `LmModel` seam the shared loops build against — including the
//! two cache-aware halves of it, since the loops train a *run* of windows and
//! carry [`MambaCaches`] from each window into the next (see that module's
//! "Runs, carried state, and the frontier").

pub use crate::common::{
    cli::AppArgs,
    model::ModelConfigExt,
    tiny_stories::lm::TinyStoriesConfig,
    training::{TrainingConfig, metric_current},
};
use crate::dataset::TinyStoriesBatch;
use burn::prelude::*;
use burn::{
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    data::dataset::DatasetError,
    optim::{GradientsParams, ModuleOptimizer},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use burn_mamba::prelude::*;
use burn_stack::examples::tiny_stories::lm::{
    self, Frontier, LmModel, dataloaders, epoch_train, epoch_valid,
};
use burn_stack::utils::ClassCursors;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

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
    let (mut optim, progress) =
        app_args.load_or_save_optim(config.training.optimizer.init(&muon_plan));

    let mut model = Wrap(model);

    // Create the dataloaders (downloading the corpus on the first run).
    let (dataloader_train, dataloader_valid) =
        dataloaders(&config, &training_device, &progress);
    // Opt-in wall-clock budget: the train loader stops yielding runs once
    // `TS_TRAIN_SECONDS` have passed since its first one.
    let deadline = Deadline::from_env();
    let dataloader_train = deadline.wrap(dataloader_train);

    // Resume position, `--max-batches` budget, cadence and metrics log.
    let mut session = app_args.session(
        progress,
        &config.training,
        lm::CADENCE,
        dataloader_train.num_items(),
    );

    // The frontier gate outlives the epochs: its depth statistics are cumulative
    // over the whole run of training, not per epoch.
    let mut frontier = Frontier::new(config.frontier.clone());

    println!("running small initial validation...");
    epoch_valid::<Wrap>(
        std::sync::Arc::clone(&dataloader_valid),
        &model.valid(),
        &config,
        0,
        session.cadence().valid_batches,
        &mut session,
    );

    println!("Starting training...");
    // Iterate over our training for X epochs
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
            &config,
            epoch,
            None,
            &mut session,
        );

        if deadline.passed() {
            println!("reached the TS_TRAIN_SECONDS limit; stopping training");
            break;
        }
        if session.is_exhausted() {
            println!("reached the --max-batches limit; stopping training");
            break;
        }
    }
    println!("Training finished.");
}

/// The SSD path used for both training and inference; the recalculated serial
/// scan saves ~1/3 vram against `Minimal`. Opt-in override:
/// `TS_SSD_PATH=serial|minimal|recalc` (the last is the default).
pub fn ssd_path() -> MambaSsdPath {
    let path = match std::env::var("TS_SSD_PATH").as_deref() {
        Ok("serial") => Mamba3SsdPath::Serial(None),
        Ok("minimal") => Mamba3SsdPath::Minimal(None),
        Ok("recalc") | Err(_) => Mamba3SsdPath::SerialRecalculated(None),
        Ok(other) => panic!("TS_SSD_PATH: unknown path {other:?}"),
    };
    MambaSsdPath::Mamba3(path)
}

/// Wrapper over [`MambaVocabNet`] for custom implementations.
pub struct Wrap(pub MambaVocabNet);

impl LmModel for Wrap {
    type Valid = Wrap;
    type Caches = MambaCaches;

    fn valid(&self) -> Self::Valid {
        Wrap(self.0.valid())
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
        let model = Wrap(optim.step(lr, self.0, grads));
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
        crate::inference::generate(&valid.0, device, prompt, n_chars, temperature, seed)
    }
}

/// The opt-in wall-clock budget of one training invocation (`TS_TRAIN_SECONDS`),
/// counted from the train loader's first run, so corpus loading and the initial
/// validation are excluded.
#[derive(Clone)]
struct Deadline {
    budget: Option<Duration>,
    start: Arc<OnceLock<Instant>>,
}

impl Deadline {
    fn from_env() -> Self {
        let budget = std::env::var("TS_TRAIN_SECONDS").ok().map(|secs| {
            Duration::from_secs_f64(secs.parse().expect("TS_TRAIN_SECONDS: seconds"))
        });
        Self {
            budget,
            start: Arc::default(),
        }
    }

    fn passed(&self) -> bool {
        match (self.budget, self.start.get()) {
            (Some(budget), Some(start)) => start.elapsed() >= budget,
            _ => false,
        }
    }

    /// A loader that ends the epoch at the deadline: the loop then closes the
    /// epoch, checkpoints and validates exactly as at its natural end (so a
    /// resumed run continues in the next epoch — keep `num_epochs` large).
    fn wrap(&self, inner: lm::Dataloader) -> lm::Dataloader {
        match self.budget {
            None => inner,
            Some(_) => Arc::new(DeadlineLoader {
                inner,
                deadline: self.clone(),
            }),
        }
    }
}

struct DeadlineLoader {
    inner: lm::Dataloader,
    deadline: Deadline,
}

impl DataLoader<TinyStoriesBatch> for DeadlineLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<TinyStoriesBatch> + 'a> {
        self.deadline.start.get_or_init(Instant::now);
        Box::new(DeadlineIter {
            inner: self.inner.iter(),
            deadline: &self.deadline,
        })
    }

    fn num_items(&self) -> usize {
        self.inner.num_items()
    }

    fn to_device(&self, device: &Device) -> Arc<dyn DataLoader<TinyStoriesBatch>> {
        Arc::new(DeadlineLoader {
            inner: self.inner.to_device(device),
            deadline: self.deadline.clone(),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<TinyStoriesBatch>> {
        Arc::new(DeadlineLoader {
            inner: self.inner.slice(start, end),
            deadline: self.deadline.clone(),
        })
    }
}

struct DeadlineIter<'a> {
    inner: Box<dyn DataLoaderIterator<TinyStoriesBatch> + 'a>,
    deadline: &'a Deadline,
}

impl Iterator for DeadlineIter<'_> {
    type Item = Result<TinyStoriesBatch, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.deadline.passed() {
            true => None,
            false => self.inner.next(),
        }
    }
}

impl DataLoaderIterator<TinyStoriesBatch> for DeadlineIter<'_> {
    fn progress(&self) -> Progress {
        self.inner.progress()
    }
}

/// Opt-in wall-clock timers over the phases of a training step, to see where a
/// step's time goes and whether a phase grows over a run. `TS_PROFILE=<N>`
/// prints each phase's mean milliseconds once per `N` windows;
/// `TS_PROFILE_SYNC=1` also syncs the device after each phase, so a phase then
/// includes its GPU execution rather than only its enqueueing. Unset ⇒ inert.
///
/// Phases: `fwd` (forward + loss), `bwd` (backward), `gap` (window end → the
/// optimizer: the loop's metric reads, i.e. its implicit sync), `opt` (the
/// optimizer step), `out` (optimizer → next window: logging, next batch).
mod prof {
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

    static STATE: LazyLock<Option<Mutex<State>>> = LazyLock::new(|| {
        let every = std::env::var("TS_PROFILE").ok()?.parse().ok()?;
        let sync = std::env::var("TS_PROFILE_SYNC").is_ok_and(|v| v == "1");
        Some(Mutex::new(State {
            every,
            sync,
            n: 0,
            sums: [0.0; 5],
            window_end: None,
            opt_end: None,
            device: None,
        }))
    });

    fn ms(from: Instant, to: Instant) -> f64 {
        (to - from).as_secs_f64() * 1e3
    }

    pub fn window_start() -> Instant {
        let now = Instant::now();
        if let Some(state) = STATE.as_ref() {
            let mut s = state.lock().unwrap();
            if let Some(end) = s.opt_end.take() {
                s.sums[4] += ms(end, now);
            }
        }
        now
    }

    pub fn mark(device: &Device) -> Instant {
        if let Some(state) = STATE.as_ref()
            && state.lock().unwrap().sync
        {
            device.sync().expect("device sync");
        }
        Instant::now()
    }

    pub fn window_end(device: &Device, t0: Instant, t1: Instant, t2: Instant) {
        if let Some(state) = STATE.as_ref() {
            let mut s = state.lock().unwrap();
            s.sums[0] += ms(t0, t1);
            s.sums[1] += ms(t1, t2);
            s.window_end = Some(Instant::now());
            s.device = Some(device.clone());
        }
    }

    pub fn opt_start() -> Instant {
        let now = Instant::now();
        if let Some(state) = STATE.as_ref() {
            let mut s = state.lock().unwrap();
            if let Some(end) = s.window_end.take() {
                s.sums[2] += ms(end, now);
            }
        }
        now
    }

    pub fn opt_end(t3: Instant) {
        let Some(state) = STATE.as_ref() else {
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
            // Live allocations: flat unless some launch shape varies, each new
            // one pinning a cached metadata buffer (burn#5751) — the bytes
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
    /// Forward the LM from `caches` (`None` ⇒ a zero state) and score every
    /// **real** position of the window against its next character — plus, in the
    /// window that opened the story, the class latents' readout against its first
    /// one (see [`lm_output`](burn_stack::examples::tiny_stories::lm::lm_output))
    /// — returning the window's final state alongside.
    ///
    /// `class` is the run's cursor: it splices the latents into the first window
    /// of a story and into no other.
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
            .forward(inputs.clone(), caches, ssd_path(), Some(class), None);
        (lm::lm_output(logits, inputs, targets, &scored), caches)
    }
}
