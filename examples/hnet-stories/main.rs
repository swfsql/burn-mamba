//! # TinyStories character-level H-Net
//!
//! An auto-regressive H-Net (Hwang, Wang & Gu, 2025) over single
//! **characters** of the [TinyStories-GPT4-clean] corpus, the corpus of the
//! `tiny-stories` example. The model learns where its chunks start. Each
//! stage keeps the characters where its router starts a chunk, runs the next
//! level on that shorter sequence, and expands the result back. Every stack is
//! Mamba-3. This example is a work in progress: its sizes and hyperparameters
//! are placeholders until a parameter search.
//!
//! [TinyStories-GPT4-clean]: https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
//!
//! - Training scores every position of a window against its next character.
//!   It also adds the ratio loss of each stage, which steers the mean chunk
//!   length toward a target. Each stage trains at its own learning rate.
//! - The first window of a story opens with learnable start rows, not a
//!   separator character. So an unprompted sample needs no seed character.
//! - Inference prefills an optional prompt with one chunkwise `forward`, then
//!   samples one character per `step`. `-- --show-chunks` marks where each
//!   stage starts a chunk.
//!
//! The example's own flags ([`cli`]) go after the trailing `--`. The run
//! writes the H-Net recipe into the `model_config.json` of the artifacts. The
//! recipe is the model, the targets, the weight of the ratio loss and the
//! learning-rate multipliers. So a resumed run keeps it:
//!
//! ```bash
//! # train and then sample (downloads the 673MB parquet once, if not cached yet)
//! cargo run --release --example hnet-stories --features backend-flex -- --training --inference
//! # a two-stage H-Net, sampled with the chunks marked
//! cargo run --release --example hnet-stories --features backend-flex -- --training --inference \
//!     -- --stages 2 --show-chunks
//! ```

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::{
    cli::AppArgs,
    tiny_stories::dataset,
    tiny_stories::lm::TinyStoriesConfig,
    training::{CosineAnnealingLr, Lr, OptimizerConfig, OptimizerKind, TrainingConfig},
};

/// The example's own flags.
pub mod cli;
/// Sampling from the trained H-Net, and the chunk view.
pub mod inference;
/// The example's model config: the H-Net and its training recipe.
pub mod model;
/// Training entry point.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow.
pub fn launch(app_args: &AppArgs) {
    let cli = cli::Cli::parse(app_args);
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (and it
    // honours the `BURN_DEVICE` env override). `configure_dtype` installs
    // fp16/i32 when `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    let autodiff_device = cli.step.training_device(&device);
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    // setup training and model configs (placeholder values, see the header)
    let batch_size = 8;
    let num_epochs = 16;
    // Characters per training window. Most stories (~820 characters on
    // average) fit in one window, so most runs are one window long.
    let seq_len = 1024;
    let loaded = app_args.load_training_config::<TinyStoriesConfig>();
    let is_fresh = loaded.is_none();
    let mut config = loaded.unwrap_or_else(|| {
        println!("Initializing new training config");
        // Muon on the hidden weight matrices of every stack, AdamW on
        // everything else. `--adamw` selects plain AdamW.
        let optimizer = app_args.optimizer_or(OptimizerKind::MuonAdamW);
        TinyStoriesConfig::new(
            TrainingConfig::new(OptimizerConfig::of(optimizer, dtype))
                .with_num_epochs(num_epochs)
                .with_batch_size(app_args.batch_size.unwrap_or(batch_size))
                .with_num_workers(2),
        )
        .with_seq_len(seq_len)
    });
    cli.overrides.apply(&mut config);
    assert!(
        config.pack.is_none(),
        "an H-Net takes no packed rows: its chunks would move the resets. Use --no-pack"
    );
    if is_fresh {
        // The schedule counts windows. One batch runs the windows of its
        // longest story: ~1200 characters for 8 draws (see `tiny-stories`).
        const CHARS_PER_LONGEST_STORY: usize = 1200;
        let batches_per_epoch = config.train_stories / config.training.batch_size;
        let iterations_per_epoch =
            batches_per_epoch * CHARS_PER_LONGEST_STORY.div_ceil(config.seq_len);
        // The base rate. The stages multiply it (see `model::lr_multipliers`).
        config.training.lr = Lr::CosineAnnealing(
            CosineAnnealingLr::new(config.training.num_epochs * iterations_per_epoch)
                .with_max_lr(4e-3)
                .with_min_lr(4e-4)
                .with_warmup_steps(iterations_per_epoch / 20), // 5% of an epoch
        );
    }
    app_args.override_training_config(&mut config.training, dtype);
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config(&cli.model)
    });
    app_args.save_training_config(&config);
    app_args.save_model_config(&model_config);

    let run = training::Run::new(&model_config, &cli, app_args.graphs());
    cli.step.start_profile();

    if app_args.training {
        training::train(config.clone(), model_config.clone(), autodiff_device, app_args, run.clone());
    }

    if app_args.inference {
        inference::infer(model_config, device, app_args, &run, &cli.prompting);
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
