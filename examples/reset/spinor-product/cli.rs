//! The example's own flags, forwarded after the trailing `--`. Both shape a
//! fresh model config; once one is persisted, it wins on reload.

use crate::common::cli::{AppArgs, finish_extra};

/// `-- --help`.
const HELP: &str = "\
spinor-product's own flags (after --):
    --micro-steps <N>    Micro-steps per token of a fresh model config (default 2; 1 is stock Mamba-3)
    --layers <N>         Layers of a fresh model config (default 1; 2 is the depth contrast)";

/// The parsed flags.
pub struct Cli {
    /// `--micro-steps`, defaulting to the 2 this example is about (`1` is the
    /// ablation: stock Mamba-3, one rotation per token).
    pub micro_steps: usize,
    /// `--layers`, defaulting to the one block the whole ladder runs. `2` is
    /// the depth contrast: a second rotation per token, applied to a second
    /// state.
    pub layers: usize,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let mut pargs = app_args.extra(HELP);
        let cli = Self {
            micro_steps: pargs
                .opt_value_from_str("--micro-steps")
                .unwrap()
                .unwrap_or(crate::dataset::PAIR),
            layers: pargs.opt_value_from_str("--layers").unwrap().unwrap_or(1),
        };
        finish_extra(pargs);
        cli
    }
}
