//! # State-tracking example — the `A₅` word problem in the grokking protocol
//!
//! The successor substrate to the `grokking` example (its README §6 retires
//! mod-p addition for Mamba-3): tracking composition in the **non-solvable**
//! group `A₅`, whose word problem is `NC¹`-complete by Barrington's theorem.
//! An abelian (`Complex2D`, `SO(2)`) rotation composes only commutatively; the
//! non-abelian `SU(2)` quaternion rotation can represent the binary
//! icosahedral group `2I = SL(2,5)` — a double cover of `A₅` — so it can
//! compose the task natively in the state. The contrast is `--quat`.
//!
//! Protocol (transposed from `grokking`): all `2^seq_len` generator words,
//! split by fraction; full-batch AdamW; cross-entropy on the **final running
//! product only**; PR diagnostics + differentiable rank penalties; per-position
//! accuracy kept as an eval-only depth probe. Sweep knobs after `--`:
//!
//! ```text
//! cargo run --release --example state-tracking --features backend-cuda,fusion -- \
//!     --training -a <dir> -- --wd 1.0 --train-fraction 0.5 --steps 20000 [--quat]
//! ```
//!
//! Metrics land in `metrics.csv`/`pr.csv`/`weights.csv`/`positions.csv` inside
//! the artifacts directory. See the README for the hypotheses and knob table.

#![allow(clippy::let_and_return)]
#![allow(clippy::module_inception)]

pub use common::cli::AppArgs;
use std::ffi::OsString;
use training::{ConstantLr, Lr, PrPenaltyTarget, TrackingConfig};

/// The `A₅` word-problem dataset (enumerated, fraction-split).
pub mod dataset;
/// State/weight participation-ratio diagnostics (shared, `common/`).
pub use common::diagnostics;
/// Post-training evaluation, depth curve, and sample predictions.
pub mod inference;
/// The example's `model_config()`.
pub mod model;
/// Full-batch training loop + `TrackingConfig`.
pub mod training;

/// Shared example infrastructure (included by path).
#[path = "../common/mod.rs"]
pub mod common;

/// Wire up the device, configs, and the train/infer flow for the task.
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
        TrackingConfig::new(
            common::training::optimizer_config(dtype)
                // Plain decoupled decay, as in the grokking example: cautious
                // decay masks exactly the pressure grokking relies on. wd
                // defaults to 0 (the memorization control arm); sweep via
                // `-- --wd`.
                .with_cautious_weight_decay(false)
                .with_weight_decay(0.0),
        )
    });
    overrides.apply(&mut training_config);
    let model_config = app_args.load_model_config().unwrap_or_else(|| {
        println!("Initializing new model config");
        model::model_config(
            overrides.d_model.unwrap_or(64),
            overrides.expand.unwrap_or(1),
            overrides.state_rank.unwrap_or(32),
            overrides.n_layers.unwrap_or(1),
            model::RotationArm {
                quaternion: overrides.quat,
                rope_fraction: overrides.rope_fraction.unwrap_or(1.0),
            },
        )
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
/// loaded/created [`TrackingConfig`] (and is then persisted with it).
struct Overrides {
    /// `--wd <f32>`: AdamW decoupled weight decay.
    wd: Option<f32>,
    /// `--lr <f64>`: constant learning rate.
    lr: Option<f64>,
    /// `--steps <usize>`: full-batch optimizer steps.
    steps: Option<usize>,
    /// `--train-fraction <f64>`: fraction of the enumerated words used for
    /// training.
    train_fraction: Option<f64>,
    /// `--seq-len <usize>`: generators per word (space = `2^seq_len`).
    seq_len: Option<usize>,
    /// `--stepwise`: token-by-token `step()` forwards instead of the (default)
    /// chunkwise `forward()` (chunkwise = recompute backward, needed by the
    /// state-PR penalty).
    stepwise: bool,
    /// `--no-diag`: skip the PR diagnostics at eval points.
    no_diag: bool,
    /// `--no-state-pr`: keep the weight-PR diagnostics but skip the (costly)
    /// state-PR pass.
    no_state_pr: bool,
    /// `--pr-lambda <f64>`: weight-PR penalty coefficient (0 = off; spell
    /// negatives as `--pr-lambda=-0.01`).
    pr_lambda: Option<f64>,
    /// `--pr-target <emb|emb-head|bc|all>`: which weights the penalty targets.
    pr_target: Option<String>,
    /// `--pr-sine-period <usize>`: sine-"breathing" period for the penalty
    /// coefficient (0 = constant).
    pr_sine_period: Option<usize>,
    /// `--step-offset <usize>`: offset added to logged step numbers (resumes).
    step_offset: Option<usize>,
    /// `--pr-start-step <usize>`: keep the penalty off until this raw step.
    pr_start_step: Option<usize>,
    /// `--l2-lambda <f64>`: plain L2 loss penalty on the `pr_target` matrices.
    l2_lambda: Option<f64>,
    /// `--noise-lambda <f64>`: weight-independent noise-gradient loss term.
    noise_lambda: Option<f64>,
    /// `--state-pr-lambda <f64>`: state-PR penalty coefficient (0 = off;
    /// needs the chunkwise path; spell negatives as `--state-pr-lambda=-0.01`).
    state_pr_lambda: Option<f64>,
    /// `--d-model <usize>`: model width (fresh model configs only).
    d_model: Option<usize>,
    /// `--expand <usize>`: `d_inner = expand·d_model` (fresh configs only).
    expand: Option<usize>,
    /// `--state-rank <usize>`: SSM state rank (multiple of 4 for `--quat`;
    /// fresh configs only).
    state_rank: Option<usize>,
    /// `--n-layers <usize>`: number of layers (fresh configs only).
    n_layers: Option<usize>,
    /// `--quat`: the non-abelian `Quaternion4D` rotation instead of the
    /// default `Complex2D` (fresh configs only) — the contrast arm.
    quat: bool,
    /// `--rope-fraction <f64>`: rotated fraction of `state_rank`
    /// (0.0 | 0.5 | 1.0; default 1.0; fresh configs only).
    rope_fraction: Option<f64>,
}

impl Overrides {
    fn parse(extra_args: &[OsString]) -> Self {
        let mut pargs = pico_args::Arguments::from_vec(extra_args.to_vec());
        let overrides = Overrides {
            wd: pargs.opt_value_from_str("--wd").unwrap(),
            lr: pargs.opt_value_from_str("--lr").unwrap(),
            steps: pargs.opt_value_from_str("--steps").unwrap(),
            train_fraction: pargs.opt_value_from_str("--train-fraction").unwrap(),
            seq_len: pargs.opt_value_from_str("--seq-len").unwrap(),
            stepwise: pargs.contains("--stepwise"),
            no_diag: pargs.contains("--no-diag"),
            no_state_pr: pargs.contains("--no-state-pr"),
            pr_lambda: pargs.opt_value_from_str("--pr-lambda").unwrap(),
            pr_target: pargs.opt_value_from_str("--pr-target").unwrap(),
            pr_sine_period: pargs.opt_value_from_str("--pr-sine-period").unwrap(),
            step_offset: pargs.opt_value_from_str("--step-offset").unwrap(),
            pr_start_step: pargs.opt_value_from_str("--pr-start-step").unwrap(),
            l2_lambda: pargs.opt_value_from_str("--l2-lambda").unwrap(),
            noise_lambda: pargs.opt_value_from_str("--noise-lambda").unwrap(),
            state_pr_lambda: pargs.opt_value_from_str("--state-pr-lambda").unwrap(),
            d_model: pargs.opt_value_from_str("--d-model").unwrap(),
            expand: pargs.opt_value_from_str("--expand").unwrap(),
            state_rank: pargs.opt_value_from_str("--state-rank").unwrap(),
            n_layers: pargs.opt_value_from_str("--n-layers").unwrap(),
            quat: pargs.contains("--quat"),
            rope_fraction: pargs.opt_value_from_str("--rope-fraction").unwrap(),
        };
        let remaining = pargs.finish();
        assert!(remaining.is_empty(), "unused extra arguments: {remaining:?}");
        overrides
    }

    fn apply(&self, config: &mut TrackingConfig) {
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
        if let Some(seq_len) = self.seq_len {
            config.seq_len = seq_len;
        }
        if self.stepwise {
            config.stepwise = true;
        }
        if self.no_diag {
            config.diagnostics = false;
        }
        if self.no_state_pr {
            config.state_diagnostics = false;
        }
        if let Some(pr_lambda) = self.pr_lambda {
            config.pr_lambda = pr_lambda;
        }
        if let Some(pr_target) = &self.pr_target {
            config.pr_target = match pr_target.as_str() {
                "emb" => PrPenaltyTarget::Emb,
                "emb-head" => PrPenaltyTarget::EmbHead,
                "bc" => PrPenaltyTarget::Bc,
                "all" => PrPenaltyTarget::All,
                other => panic!("unknown --pr-target {other:?} (emb|emb-head|bc|all)"),
            };
        }
        if let Some(pr_sine_period) = self.pr_sine_period {
            config.pr_sine_period = pr_sine_period;
        }
        if let Some(step_offset) = self.step_offset {
            config.step_offset = step_offset;
        }
        if let Some(pr_start_step) = self.pr_start_step {
            config.pr_start_step = pr_start_step;
        }
        if let Some(l2_lambda) = self.l2_lambda {
            config.l2_lambda = l2_lambda;
        }
        if let Some(noise_lambda) = self.noise_lambda {
            config.noise_lambda = noise_lambda;
        }
        if let Some(state_pr_lambda) = self.state_pr_lambda {
            config.state_pr_lambda = state_pr_lambda;
        }
    }
}

fn main() {
    let app_args = AppArgs::parse().unwrap();
    launch(&app_args);
}
