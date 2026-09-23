//! The example's own flags, forwarded after the trailing `--`: the corpus knobs
//! (`Overrides`, shared with `burn-deltanet`), the SSD path and the step
//! profiler.

use crate::common::cli::{AppArgs, finish_extra};
use crate::common::tiny_stories::lm::Overrides;
use burn_mamba::prelude::*;

/// `-- --help`, below the corpus knobs.
const HELP: &str = concat!(
    "    --ssd-path <PATH>      The SSD path of every chunkwise forward: recalc (default), serial or minimal\n",
    "    --profile <N>          Print the mean milliseconds of each training-step phase once per N windows\n",
    "    --profile-sync         Also sync the device after each phase (with --profile)",
);

/// The parsed flags.
pub struct Cli {
    /// The corpus knobs, applied onto the training config.
    pub overrides: Overrides,
    /// `--ssd-path`: the recalculated serial scan (the default) uses about 1/3
    /// less vram than `Minimal`.
    pub ssd_path: MambaSsdPath,
    /// `--profile` (and `--profile-sync`): see `training::prof`.
    pub profile: Option<(usize, bool)>,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let help = format!("tiny-stories' own flags (after --):\n{}\n{HELP}", Overrides::HELP);
        let mut pargs = app_args.extra(&help);
        let overrides = Overrides::parse(&mut pargs);
        let ssd_path = pargs
            .opt_value_from_fn("--ssd-path", parse_ssd_path)
            .unwrap()
            .unwrap_or(Mamba3SsdPath::SerialRecalculated(None));
        let every: Option<usize> = pargs.opt_value_from_str("--profile").unwrap();
        let sync = pargs.contains("--profile-sync");
        assert!(every.is_some() || !sync, "--profile-sync needs --profile");
        finish_extra(pargs);
        Self {
            overrides,
            ssd_path: MambaSsdPath::Mamba3(ssd_path),
            profile: every.map(|every| (every, sync)),
        }
    }
}

fn parse_ssd_path(path: &str) -> Result<Mamba3SsdPath, String> {
    match path {
        "recalc" => Ok(Mamba3SsdPath::SerialRecalculated(None)),
        "serial" => Ok(Mamba3SsdPath::Serial(None)),
        "minimal" => Ok(Mamba3SsdPath::Minimal(None)),
        other => Err(format!("unknown SSD path {other:?} (recalc | serial | minimal)")),
    }
}
