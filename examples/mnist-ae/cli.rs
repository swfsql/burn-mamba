//! The example's own flags, forwarded after the trailing `--`.

use crate::common::cli::{AppArgs, finish_extra};

/// `-- --help`.
const HELP: &str = "\
mnist-ae's own flags (after --):
    --latents <N>    The latent bottleneck width of a fresh model config (default 16)";

/// The parsed flags.
pub struct Cli {
    /// `--latents`: the bottleneck width of a fresh model config. (On reload, a
    /// persisted model config wins.)
    pub latents: usize,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let mut pargs = app_args.extra(HELP);
        let cli = Self {
            latents: pargs.opt_value_from_str("--latents").unwrap().unwrap_or(16),
        };
        finish_extra(pargs);
        cli
    }
}
