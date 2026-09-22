//! The example's own flags, forwarded after the trailing `--`: none. (The
//! optimizer and `--no-graph` are `AppArgs`', shared by every example.)

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
