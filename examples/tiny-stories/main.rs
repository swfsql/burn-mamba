//! # TinyStories character-level language model
//!
//! An auto-regressive Mamba-3 LM over single **characters** of the
//! [TinyStories-GPT4-clean] corpus: two Mamba-3 blocks (cycled to an 8-deep
//! virtual stack over Multi-Gate residuals) between a **tied** 48-character
//! embedding and its transpose, 39,632 parameters all told.
//!
//! [TinyStories-GPT4-clean]: https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
//!
//! One item is one **story**, opened by four class latents rather than by a
//! separator character. Training scores every position of a 256-character window
//! against its next character — plus, in the story's first window, the latents'
//! readout against its first character — and walks the story's windows carrying
//! the (detached) state from each into the next for as long as the frontier gate
//! admits it; see the README's "Story boundaries" and "Runs and the frontier".
//! Inference replays those latents with one `prime` (which is already the first
//! character's distribution), optionally prefills a prompt with one chunkwise
//! `forward`, and then samples one character per `step`.
//!
//! Corpus knobs are forwarded after the trailing `--` (they are written into the
//! artifacts' `training_config.json`, so resuming a run keeps them):
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
    tiny_stories::lm::{Overrides, TinyStoriesConfig},
    training::{CosineAnnealingLr, Lr, TrainingConfig},
};

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
    let overrides = Overrides::parse(&app_args.extra_args);
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (honouring
    // the `BURN_DEVICE` env override); `configure_dtype` installs fp16/i32 when
    // `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    // setup training and model configs
    // Batch size is the single largest lever on this model's final loss, and it
    // is *not* a capacity knob: halving it from 16 to 8 beat the entire learning
    // rate ladder (a 6x increase), and it kept paying at 4. 8 is where the
    // accuracy-per-minute stops being worth it — batch 4 costs twice the wall
    // clock for a fraction of the gain.
    let batch_size = 8;
    // The validation curve is still improving at epoch 14 and flattens at 15-16,
    // so this is the schedule's own natural length, not an arbitrary budget.
    let num_epochs = 16;
    let loaded = app_args.load_training_config::<TinyStoriesConfig>();
    let is_fresh = loaded.is_none();
    let mut config = loaded.unwrap_or_else(|| {
        println!("Initializing new training config");
        // Muon on the block's hidden weight matrices, AdamW on everything else.
        // It is the smallest of this example's optimizer wins but it stacks with
        // the other two (higher LR, smaller batch) rather than overlapping them.
        // `--no-muon` returns to plain AdamW.
        let optimizer = common::training::OptimizerConfig::adamw_only(dtype)
            .with_muon_defaults(ADAMW_WEIGHT_DECAY);
        TinyStoriesConfig::new(
            TrainingConfig::new(optimizer)
                .with_num_epochs(num_epochs)
                .with_batch_size(batch_size)
                .with_num_workers(2),
        )
    });
    overrides.apply(&mut config);
    if is_fresh {
        // The cosine schedule spans the whole run, so it can only be sized once
        // the corpus knobs are settled. It is counted in *windows*, not in
        // dataloader items, and the training loop charges the schedule for the
        // windows the frontier gate skipped too — so a stalling gate shortens
        // the run rather than leaving the cosine unfinished.
        //
        // One item is one story, and a batch runs the windows of its *longest*
        // story — so the schedule is sized from that maximum, not from the mean:
        // ~1200 characters is the expected longest of `batch_size = 8` draws
        // (the corpus's mean is ~820, its 90th percentile 1103, its longest
        // story 4149).
        const CHARS_PER_LONGEST_STORY: usize = 1200;
        let batches_per_epoch = config.train_stories / config.training.batch_size;
        let iterations_per_epoch =
            batches_per_epoch * CHARS_PER_LONGEST_STORY.div_ceil(config.seq_len);
        config.training.lr = Lr::CosineAnnealing(
            CosineAnnealingLr::new(config.training.num_epochs * iterations_per_epoch)
                // The model is optimization-limited, not capacity-limited, and
                // this is where that shows: 2e-3 (the obvious default) leaves a
                // lot on the table. The ladder improves monotonically to 16e-3,
                // is flat to 24e-3, and only turns over at 32e-3.
                .with_max_lr(12e-3)
                .with_min_lr(12e-4)
                .with_warmup_steps(iterations_per_epoch / 20), // 5% of an epoch
        );
    }
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config()
    });
    // save configs
    app_args.save_training_config(&config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(
            config.clone(),
            model_config.clone(),
            autodiff_device,
            app_args,
        );
    }

    if app_args.inference {
        inference::infer(model_config, device, app_args);
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

/// AdamW's default weight decay, mirrored into the Muon group so the two arms
/// decay the same weights by the same amount.
const ADAMW_WEIGHT_DECAY: f32 = 1e-4;

fn main() {
    let app_args = AppArgs::parse(common::ARTIFACT_PREFIX).unwrap();
    launch(&app_args);
}
