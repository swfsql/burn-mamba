//! The example's own flags, forwarded after the trailing `--`: the corpus knobs
//! (`Overrides`, shared with `tiny-stories`), the flags of a training step
//! (`StepFlags`: the profiler, the VRAM cap, the gradient checkpointing), the
//! SSD paths, the shape of a new model, the sampling, and the context-use
//! measure.

use crate::common::cli::{AppArgs, StepFlags, finish_extra};
use crate::common::context::BATCH;
use crate::common::parse_ssd_path;
use crate::common::tiny_stories::lm::Overrides;
use crate::inference::{PROMPT_TEMPERATURE, Prompting};
use burn_mamba::prelude::{Mamba3SsdPath, MambaSsdPath};
use burn_stack::modules::hnet::StepMode;

/// `-- --help`, below the corpus knobs.
const HELP: &str = concat!(
    "    --ssd-path <PATH[:L]>  The SSD path of every chunkwise forward of the stages: recalc (default), serial or minimal. L sets the chunk length (default: from the block shape)\n",
    "    --main-ssd-path <PATH[:L]>  The SSD path of the main network (default: that of --ssd-path)\n",
    "    --stages <N>           A new model: the number of stages, 1 or 2 (default 1)\n",
    "    --step-mode <M>        How a decode step runs the inner network of a stage: masked (default, capturable) or gathered (only the rows that start a chunk, read on the host)\n",
    "    --show-chunks          Mark the chunk starts in each sample: | where stage 0 starts a chunk, ‖ where stage 1 also does\n",
    "    --chunk-log <N>        Print the chunk ratios of each stage once per N training windows (default 100, 0 never)\n",
    "    --prompt <TEXT>        At inference, continue TEXT instead of the built-in prompts. Give the flag again for each prompt\n",
    "    --continuations <N>    At inference, the number of continuations of each prompt, each with its own seed (default 1)\n",
    "    --temperature <T>      At inference, the sampling temperature of the prompted continuations (default 0.8)\n",
    "    --context-use          Measure how much the trained model uses the earlier part of the validation stories\n",
    "    --context-batch <N>    With --context-use or --long-stories, the sequences per forward (default 4)\n",
    "    --context-stories <N>  With --context-use, read the first N validation stories (default: the stories of the validation loss)\n",
    "    --long-stories         Score whole validation stories: those longer than 2048 characters (per segment of 1024 characters), and those of the validation loss",
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
    /// `--profile`, `--max-vram`, `--grad-checkpointing`.
    pub step: StepFlags,
    /// `--ssd-path` and `--main-ssd-path`: the SSD paths of the stages and of
    /// the main network.
    pub paths: (MambaSsdPath, MambaSsdPath),
    /// The shape of a new model.
    pub model: ModelFlags,
    /// `--step-mode`.
    pub step_mode: StepMode,
    /// `--show-chunks`.
    pub show_chunks: bool,
    /// `--chunk-log`: `0` never prints.
    pub chunk_log: usize,
    /// `--prompt` (repeatable), `--continuations` and `--temperature`.
    pub prompting: Prompting,
    /// `--context-use`: run
    /// [`context_use`](crate::common::context::context_use).
    pub context_use: bool,
    /// `--context-batch`: the sequences per `forward` of `--context-use` and
    /// of `--long-stories`.
    pub context_batch: usize,
    /// `--context-stories`: the validation stories that `--context-use` reads
    /// (`None`: the stories of the validation loss).
    pub context_stories: Option<usize>,
    /// `--long-stories`: run
    /// [`long_stories`](crate::common::long_stories::long_stories).
    pub long_stories: bool,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let help = format!(
            "hnet-stories' own flags (after --):\n{}\n{}\n{HELP}",
            Overrides::HELP,
            StepFlags::HELP
        );
        let mut pargs = app_args.extra(&help);
        let overrides = Overrides::parse(&mut pargs);
        let step = StepFlags::parse(&mut pargs);
        let stage_path = pargs
            .opt_value_from_fn("--ssd-path", parse_ssd_path)
            .unwrap()
            .unwrap_or(Mamba3SsdPath::SerialRecalculated(None));
        let main_path =
            pargs.opt_value_from_fn("--main-ssd-path", parse_ssd_path).unwrap().unwrap_or_else(|| stage_path.clone());
        let paths = (MambaSsdPath::Mamba3(stage_path), MambaSsdPath::Mamba3(main_path));
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
            continuations: pargs.opt_value_from_str("--continuations").unwrap().unwrap_or(1),
            temperature: pargs
                .opt_value_from_str("--temperature")
                .unwrap()
                .unwrap_or(PROMPT_TEMPERATURE),
        };
        assert!(prompting.continuations >= 1, "--continuations must be at least 1");
        let context_use = pargs.contains("--context-use");
        let context_batch = pargs.opt_value_from_str("--context-batch").unwrap().unwrap_or(BATCH);
        assert!(context_batch >= 1, "--context-batch must be at least 1");
        let context_stories: Option<usize> = pargs.opt_value_from_str("--context-stories").unwrap();
        assert!(context_stories != Some(0), "--context-stories must be at least 1");
        let long_stories = pargs.contains("--long-stories");
        finish_extra(pargs);
        Self {
            overrides,
            step,
            paths,
            model,
            step_mode,
            show_chunks,
            chunk_log,
            prompting,
            context_use,
            context_batch,
            context_stories,
            long_stories,
        }
    }
}
