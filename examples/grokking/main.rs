//! # Grokking example
//!
//! Modular addition `(a + b) mod p` with a small Mamba-2 LM — the classic
//! grokking task (Power et al. 2022): train accuracy saturates early while
//! test accuracy sits at chance, then jumps to ~100% much later under weight
//! decay. This example is the substrate for the state-participation-ratio
//! diagnostic (does the effective rank of the recurrent state collapse at the
//! memorize→generalize transition?).
//!
//! Sweep knobs are forwarded after `--`:
//!
//! ```text
//! cargo run --release --example grokking -- --training \
//!     -- --wd 0.1 --lr 1e-3 --steps 100000 --train-fraction 0.5
//! ```
//!
//! `--wd 0` (the default) is the memorization control arm. Metrics land in
//! `metrics.csv` inside the artifacts directory.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::cli::AppArgs;
use std::ffi::OsString;
use training::{ConstantLr, GrokkingConfig, Lr};

/// The modular-addition dataset and its deterministic pair split.
pub mod dataset;
/// Post-training evaluation and sample predictions.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Full-batch training loop + `GrokkingConfig`.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the grokking task.
pub fn launch(app_args: &AppArgs) {
    let overrides = Overrides::parse(&app_args.extra_args);
    app_args.create_artifact_dir();

    // `Device::default()` resolves to the enabled `backend-*` feature (honouring
    // the `BURN_DEVICE` env override); `configure_dtype` installs fp16/i32 when
    // `dev-f16` is on.
    let mut device = burn::prelude::Device::default();
    common::device::configure_dtype(&mut device);
    // training needs an autodiff-enabled device; inference uses the plain one.
    let autodiff_device = device.clone().autodiff();
    let dtype = burn::tensor::Tensor::<1>::zeros([1], &device).dtype();

    let mut training_config = app_args.load_training_config().unwrap_or_else(|| {
        println!("Initializing new training config");
        GrokkingConfig::new(
            common::training::optimizer_config(dtype)
                // Plain decoupled decay for literature fidelity: cautious decay
                // masks exactly the pressure grokking relies on. wd defaults to
                // 0 (the memorization control arm); sweep it via `-- --wd`.
                .with_cautious_weight_decay(false)
                .with_weight_decay(0.0),
        )
    });
    overrides.apply(&mut training_config);
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config(training_config.p)
    });
    // save configs
    app_args.save_training_config(&training_config);
    app_args.save_model_config(&model_config);

    if app_args.training {
        training::train(
            training_config.clone(),
            model_config.clone(),
            autodiff_device,
            app_args,
        );
    }

    if app_args.inference {
        inference::infer(&training_config, model_config, device, app_args);
    }

    if !app_args.inference && !app_args.training {
        println!("neither training nor inference were enabled");
        println!("{}", common::cli::HELP);
    }
}

/// Sweep-knob overrides forwarded after `--`; each applies on top of the
/// loaded/created [`GrokkingConfig`] (and is then persisted with it).
struct Overrides {
    /// `--wd <f32>`: AdamW decoupled weight decay.
    wd: Option<f32>,
    /// `--lr <f64>`: constant learning rate.
    lr: Option<f64>,
    /// `--steps <usize>`: full-batch optimizer steps.
    steps: Option<usize>,
    /// `--train-fraction <f64>`: fraction of the `p²` pairs used for training.
    train_fraction: Option<f64>,
    /// `--chunked`: use the chunkwise `forward()` instead of the (default,
    /// faster) token-by-token `step()` mode.
    chunked: bool,
}

impl Overrides {
    fn parse(extra_args: &[OsString]) -> Self {
        let mut pargs = pico_args::Arguments::from_vec(extra_args.to_vec());
        let overrides = Overrides {
            wd: pargs.opt_value_from_str("--wd").unwrap(),
            lr: pargs.opt_value_from_str("--lr").unwrap(),
            steps: pargs.opt_value_from_str("--steps").unwrap(),
            train_fraction: pargs.opt_value_from_str("--train-fraction").unwrap(),
            chunked: pargs.contains("--chunked"),
        };
        let remaining = pargs.finish();
        assert!(remaining.is_empty(), "unused extra arguments: {remaining:?}");
        overrides
    }

    fn apply(&self, config: &mut GrokkingConfig) {
        if let Some(wd) = self.wd {
            config.optimizer = config.optimizer.clone().with_weight_decay(wd);
        }
        if let Some(lr) = self.lr {
            config.lr = Lr::Constant(ConstantLr::new().with_lr(lr));
        }
        if let Some(steps) = self.steps {
            config.num_steps = steps;
        }
        if let Some(train_fraction) = self.train_fraction {
            config.train_fraction = train_fraction;
        }
        if self.chunked {
            config.stepwise = false;
        }
    }
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
