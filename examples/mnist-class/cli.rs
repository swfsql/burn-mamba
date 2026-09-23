//! The example's own flags, forwarded after the trailing `--`: none. (The
//! optimizer flags and `--no-graph` belong to `AppArgs`, which every example
//! shares.)

use crate::common::cli::{AppArgs, finish_extra};

/// `-- --help`.
const HELP: &str = "\
mnist-class takes no flags of its own after --.";

/// The parsed flags.
pub struct Cli;

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        finish_extra(app_args.extra(HELP));
        Self
    }
}
