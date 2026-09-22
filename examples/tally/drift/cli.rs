//! The example's own flags, forwarded after the trailing `--`.

use crate::common::cli::{AppArgs, finish_extra};

/// `-- --help`.
const HELP: &str = "\
tally-drift's own flags (after --):
    --stock    Replace the computed (Kalman) decay of a fresh model config by the projected one (the rung's ablation)";

/// The parsed flags.
pub struct Cli {
    /// `--stock`: a fresh model config with the stock projected decay instead
    /// of the computed one, the ablation this rung is about. Once a model
    /// config is persisted, it wins on reload.
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
