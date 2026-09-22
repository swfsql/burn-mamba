//! The example's own flags, forwarded after the trailing `--`.

use crate::common::cli::{AppArgs, finish_extra};
use burn_mamba::prelude::RotationKind;

/// `-- --help`.
const HELP: &str = "\
reset-spinor's own flags (after --):
    --rotation <KIND>    The rotation baked into a fresh model config: quaternion (default), complex or rotor";

/// The parsed flags.
pub struct Cli {
    /// `--rotation`: the rotation of a fresh model config, defaulting to the
    /// quaternion one this example is about (`rotor` is the full-`SO(4)` kind,
    /// which contains it). Once a model config is persisted, it wins on reload.
    pub rotation: RotationKind,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let mut pargs = app_args.extra(HELP);
        let cli = Self {
            rotation: pargs
                .opt_value_from_fn("--rotation", crate::common::parse_rotation)
                .unwrap()
                .unwrap_or(RotationKind::Quaternion4D),
        };
        finish_extra(pargs);
        cli
    }
}
