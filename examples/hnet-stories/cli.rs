//! The example's own flags, forwarded after the trailing `--`: the corpus knobs
//! (`Overrides`, shared with `tiny-stories`), the shape of a new model, and the
//! sampling.

use crate::common::cli::{AppArgs, finish_extra};
use crate::common::tiny_stories::lm::Overrides;
use crate::inference::{PROMPT_TEMPERATURE, Prompting};
use burn_stack::modules::hnet::StepMode;

/// `-- --help`, below the corpus knobs.
const HELP: &str = concat!(
    "    --stages <N>           A new model: the number of stages, 1 or 2 (default 1)\n",
    "    --step-mode <M>        How a decode step runs the inner network of a stage: masked (default, capturable) or gathered (only the rows that start a chunk, read on the host)\n",
    "    --show-chunks          Mark the chunk starts in each sample: | where stage 0 starts a chunk, ‖ where stage 1 also does\n",
    "    --chunk-log <N>        Print the chunk ratios of each stage once per N training windows (default 100, 0 never)\n",
    "    --prompt <TEXT>        At inference, continue TEXT instead of the built-in prompts. Give the flag again for each prompt\n",
    "    --temperature <T>      At inference, the sampling temperature of the prompted continuations (default 0.8)",
);

/// The shape of a new model (a saved model config ignores these).
#[derive(Clone, Debug)]
pub struct ModelFlags {
    /// `--stages`.
    pub stages: usize,
}

/// The parsed flags.
pub struct Cli {
    /// The corpus knobs, applied onto the training config.
    pub overrides: Overrides,
    /// The shape of a new model.
    pub model: ModelFlags,
    /// `--step-mode`.
    pub step_mode: StepMode,
    /// `--show-chunks`.
    pub show_chunks: bool,
    /// `--chunk-log`: `0` never prints.
    pub chunk_log: usize,
    /// `--prompt` (repeatable) and `--temperature`.
    pub prompting: Prompting,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let help = format!("hnet-stories' own flags (after --):\n{}\n{HELP}", Overrides::HELP);
        let mut pargs = app_args.extra(&help);
        let overrides = Overrides::parse(&mut pargs);
        let model = ModelFlags { stages: pargs.opt_value_from_str("--stages").unwrap().unwrap_or(1) };
        assert!(matches!(model.stages, 1 | 2), "--stages must be 1 or 2");
        let step_mode = pargs
            .opt_value_from_fn("--step-mode", |value: &str| match value {
                "masked" => Ok(StepMode::Masked),
                "gathered" => Ok(StepMode::Gathered),
                other => Err(format!("--step-mode must be 'masked' or 'gathered', got {other:?}")),
            })
            .unwrap()
            .unwrap_or_default();
        let show_chunks = pargs.contains("--show-chunks");
        let chunk_log = pargs.opt_value_from_str("--chunk-log").unwrap().unwrap_or(100);
        let prompting = Prompting {
            prompts: pargs.values_from_str("--prompt").unwrap(),
            temperature: pargs
                .opt_value_from_str("--temperature")
                .unwrap()
                .unwrap_or(PROMPT_TEMPERATURE),
        };
        finish_extra(pargs);
        Self { overrides, model, step_mode, show_chunks, chunk_log, prompting }
    }
}
