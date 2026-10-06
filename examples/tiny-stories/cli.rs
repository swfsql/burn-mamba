//! The example's own flags, forwarded after the trailing `--`: the corpus knobs
//! (`Overrides`, shared with `burn-deltanet`), the SSD path, the step profiler
//! and the VRAM cap.

use crate::common::cli::{AppArgs, finish_extra};
use crate::common::tiny_stories::lm::Overrides;
use crate::context::BATCH;
use crate::contrast::ContrastOverrides;
use crate::inference::{PROMPT_TEMPERATURE, Prompting};
use burn_mamba::prelude::*;

/// `-- --help`, below the corpus knobs.
const HELP: &str = concat!(
    "    --ssd-path <PATH[:L]>  The SSD path of every chunkwise forward: recalc (default), serial or minimal. L sets the chunk length, in folded positions (default: from the block shape)\n",
    "    --pack-align <N>       With --pack, start each story at a multiple of N tokens. N must be a multiple of the chunk length (default: the chunk length)\n",
    "    --prompt <TEXT>        At inference, continue TEXT instead of the built-in prompts. Give the flag again for each prompt\n",
    "    --continuations <N>    At inference, the number of continuations of each prompt, each with its own seed (default 1)\n",
    "    --temperature <T>      At inference, the sampling temperature of the prompted continuations (default 0.8)\n",
    "    --context-use          Measure how much the trained model uses the earlier part of the validation stories\n",
    "    --context-batch <N>    With --context-use, the sequences per forward (default 4)\n",
    "    --context-stories <N>  With --context-use, read the first N validation stories (default: the stories of the validation loss)\n",
    "    --contrast-eval <N>    Measure how strongly the contrast pushes the trained model: N batches per cut kind, no backward\n",
    "    --contrast-profile     With --contrast-eval, run only the cut and the negative of the flags, and write the scores of each character to contrast-profile.jsonl\n",
    "    --profile <N>          Print the mean milliseconds of each training-step phase once per N windows\n",
    "    --profile-sync         Also sync the device after each phase (with --profile)\n",
    "    --max-vram <MiB>       Stop the process with an error when the memory pools hold more than this, after a backward. Also clean up the memory before windows 0, 1, 2, 4, 8, ... and before each 256th window",
);

/// The parsed flags.
pub struct Cli {
    /// The corpus knobs, applied onto the training config.
    pub overrides: Overrides,
    /// `--ssd-path`: the recalculated serial scan (the default) uses about 1/3
    /// less vram than `Minimal`.
    pub ssd_path: MambaSsdPath,
    /// `--pack-align`: see `training::Run::pack_align`.
    pub pack_align: Option<usize>,
    /// `--profile` (and `--profile-sync`): see `training::prof`.
    pub profile: Option<(usize, bool)>,
    /// `--max-vram`, in MiB: the cap on the bytes that the memory pools
    /// reserve (the CUDA context is not included). See `training::Run`.
    pub max_vram_mib: Option<u64>,
    /// `--prompt` (repeatable), `--continuations` and `--temperature`: the
    /// prompted continuations of inference.
    pub prompting: Prompting,
    /// `--context-use`: run [`context_use`](crate::context::context_use).
    pub context_use: bool,
    /// `--context-batch`: the sequences per `forward` of `--context-use`.
    pub context_batch: usize,
    /// `--context-stories`: the validation stories that `--context-use` reads
    /// (`None`: the stories of the validation loss).
    pub context_stories: Option<usize>,
    /// The `--contrast*` flags: see [`crate::contrast`].
    pub contrast: ContrastOverrides,
    /// `--contrast-eval`: the batches per cut kind of
    /// [`contrast_eval`](crate::contrast::contrast_eval).
    pub contrast_eval: Option<usize>,
    /// `--contrast-profile`: the `profile` of
    /// [`contrast_eval`](crate::contrast::contrast_eval).
    pub contrast_profile: bool,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let help = format!(
            "tiny-stories' own flags (after --):\n{}\n{HELP}\n{}",
            Overrides::HELP,
            ContrastOverrides::HELP
        );
        let mut pargs = app_args.extra(&help);
        let overrides = Overrides::parse(&mut pargs);
        let ssd_path = pargs
            .opt_value_from_fn("--ssd-path", parse_ssd_path)
            .unwrap()
            .unwrap_or(Mamba3SsdPath::SerialRecalculated(None));
        let pack_align: Option<usize> = pargs.opt_value_from_str("--pack-align").unwrap();
        assert!(pack_align != Some(0), "--pack-align must be at least 1");
        let every: Option<usize> = pargs.opt_value_from_str("--profile").unwrap();
        let sync = pargs.contains("--profile-sync");
        assert!(every.is_some() || !sync, "--profile-sync needs --profile");
        let max_vram_mib = pargs.opt_value_from_str("--max-vram").unwrap();
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
        let contrast = ContrastOverrides::parse(&mut pargs);
        let contrast_eval: Option<usize> = pargs.opt_value_from_str("--contrast-eval").unwrap();
        assert!(contrast_eval != Some(0), "--contrast-eval must be at least 1");
        let contrast_profile = pargs.contains("--contrast-profile");
        assert!(contrast_eval.is_some() || !contrast_profile, "--contrast-profile needs --contrast-eval");
        finish_extra(pargs);
        Self {
            overrides,
            ssd_path: MambaSsdPath::Mamba3(ssd_path),
            pack_align,
            profile: every.map(|every| (every, sync)),
            max_vram_mib,
            prompting,
            context_use,
            context_batch,
            context_stories,
            contrast,
            contrast_eval,
            contrast_profile,
        }
    }
}

/// `<path>[:<chunk_len>]`. The chunk length counts folded positions, as
/// `Mamba3SsdPath` does.
fn parse_ssd_path(arg: &str) -> Result<Mamba3SsdPath, String> {
    let (path, chunk_len) = match arg.split_once(':') {
        None => (arg, None),
        Some((path, len)) => match len.parse::<usize>() {
            Ok(len) if len >= 1 => (path, Some(len)),
            _ => return Err(format!("chunk length {len:?} is not a positive integer")),
        },
    };
    match path {
        "recalc" => Ok(Mamba3SsdPath::SerialRecalculated(chunk_len)),
        "serial" => Ok(Mamba3SsdPath::Serial(chunk_len)),
        "minimal" => Ok(Mamba3SsdPath::Minimal(chunk_len)),
        other => Err(format!("unknown SSD path {other:?} (recalc | serial | minimal)")),
    }
}
