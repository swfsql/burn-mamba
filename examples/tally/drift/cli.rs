//! The example's own flags, forwarded after the trailing `--`.

use crate::common::cli::{AppArgs, finish_extra};

/// `-- --help`.
const HELP: &str = "\
tally-drift's own flags (after --):
    --stock    Replace the computed (Kalman) decay of a fresh model config by the projected one (the ablation of the rung)";

/// The parsed flags.
pub struct Cli {
    /// `--stock`: a fresh model config with the stock projected decay instead
    /// of the computed one, the ablation that this rung is about. On reload, a
    /// persisted model config wins.
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
