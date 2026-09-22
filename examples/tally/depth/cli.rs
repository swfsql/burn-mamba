//! The example's own flags, forwarded after the trailing `--`.

use crate::common::cli::{AppArgs, finish_extra};

/// `-- --help`.
const HELP: &str = "\
tally-depth's own flags (after --):
    --stock    Remove the tropical register from a fresh model config (the rung's ablation)";

/// The parsed flags.
pub struct Cli {
    /// `--stock`: a fresh model config without the tropical register, the
    /// ablation this rung is about. Once a model config is persisted, it wins
    /// on reload.
    pub stock: bool,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let mut pargs = app_args.extra(HELP);
        let cli = Self {
            stock: pargs.contains("--stock"),
        };
        finish_extra(pargs);
        cli
    }
}
