//! # TinyStories character-level language model
//!
//! An auto-regressive Mamba-3 LM over single **characters** of the
//! [TinyStories-GPT4-clean] corpus: Mamba-3 blocks (cycled to a virtual stack
//! over Multi-Gate residuals) between a **tied** 48-character embedding and its
//! transpose. This example is a work in progress: its sizes and
//! hyperparameters are placeholders until a parameter search.
//!
//! [TinyStories-GPT4-clean]: https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
//!
//! One item is one **story**. Four class latents start it, not a separator
//! character.
//!
//! - Training scores every position of a window against its next character. In
//!   the first window of the story, it also scores the readout of the latents
//!   against the first character.
//! - Training walks the windows of the story. It carries the (detached) state of
//!   each window into the next, while the frontier gate accepts it. See
//!   "Story boundaries" and "Runs and the frontier" in the README.
//! - Inference replays the latents with one `prime` (which already returns the
//!   distribution of the first character). It prefills an optional prompt with
//!   chunkwise `forward`s, then samples one character per `step`.
//!
//! The corpus knobs, the SSD path and the step profiler ([`cli`]) go after the
//! trailing `--`. The corpus knobs are written into the `training_config.json`
//! of the artifacts, so a resumed run keeps them:
//!
//! ```bash
//! # train and then sample (downloads the 673MB parquet once, if not cached yet)
//! cargo run --release --example tiny-stories --features backend-flex -- --training --inference
//! # a bigger corpus and a longer window
//! cargo run --release --example tiny-stories --features backend-flex -- --training \
//!     -- --train-stories 32768 --seq-len 512
//! ```

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    tiny_stories::dataset,
    tiny_stories::lm::TinyStoriesConfig,
    training::{CosineAnnealingLr, Lr, OptimizerConfig, OptimizerKind, TrainingConfig},
};

/// The example's own flags (corpus knobs, SSD path, profiler).
pub mod cli;
/// Sampling from the trained LM.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Training entry point for the LM.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the LM.
pub fn launch(app_args: &AppArgs) {
    let cli = cli::Cli::parse(app_args);
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (and it
    // honours the `BURN_DEVICE` env override). `configure_dtype` installs
    // fp16/i32 when `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    // setup training and model configs (placeholder values, see the header)
    let batch_size = 8;
    let num_epochs = 16;
    let loaded = app_args.load_training_config::<TinyStoriesConfig>();
    let is_fresh = loaded.is_none();
    let mut config = loaded.unwrap_or_else(|| {
        println!("Initializing new training config");
        // Muon on the hidden weight matrices of the block, AdamW on everything
        // else. `--adamw` selects plain AdamW.
        let optimizer = app_args.optimizer_or(OptimizerKind::MuonAdamW);
        TinyStoriesConfig::new(
            TrainingConfig::new(OptimizerConfig::of(optimizer, dtype))
                .with_num_epochs(num_epochs)
                // The schedule below is sized from it.
                .with_batch_size(app_args.batch_size.unwrap_or(batch_size))
                .with_num_workers(2),
        )
    });
    cli.overrides.apply(&mut config);
    if is_fresh {
        // The cosine schedule spans the whole run, so it can be sized only after
        // the corpus knobs are known. It counts *windows*, not dataloader items.
        // The training loop also charges the schedule for the windows that the
        // frontier gate skipped. So a stalling gate shortens the run, and the
        // cosine still finishes.
        //
        // One item is one story, and a batch runs the windows of its *longest*
        // story. So the schedule uses that maximum, not the mean: ~1200
        // characters is the expected longest of `batch_size = 8` draws (the mean
        // of the corpus is ~820, its 90th percentile 1103, its longest story
        // 4149).
        const CHARS_PER_LONGEST_STORY: usize = 1200;
        let batches_per_epoch = config.train_stories / config.training.batch_size;
        let iterations_per_epoch =
            batches_per_epoch * CHARS_PER_LONGEST_STORY.div_ceil(config.seq_len);
        config.training.lr = Lr::CosineAnnealing(
            CosineAnnealingLr::new(config.training.num_epochs * iterations_per_epoch)
                .with_max_lr(12e-3)
                .with_min_lr(12e-4)
                .with_warmup_steps(iterations_per_epoch / 20), // 5% of an epoch
        );
    }
    // After the sizing, so `--epochs` rescales the schedule and `--max-lr`
    // replaces its peak on a fresh config too.
    app_args.override_training_config(&mut config.training, dtype);
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config()
    });
    // save configs
    app_args.save_training_config(&config);
    app_args.save_model_config(&model_config);

    let run = training::Run {
        ssd_path: cli.ssd_path,
        graphs: app_args.graphs(),
    };
    if let Some((every, sync)) = cli.profile {
        training::prof::init(every, sync);
    }

    if app_args.training {
        training::train(
            config.clone(),
            model_config.clone(),
            autodiff_device,
            app_args,
            run.clone(),
        );
    }

    if app_args.inference {
        inference::infer(model_config, device, app_args, &run);
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    launch(&app_args);
}
