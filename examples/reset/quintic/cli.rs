//! The example's own flags, forwarded after the trailing `--` (listed in the
//! crate docs). `--group` is not persisted, so pass it on every run; the width,
//! depth and rotation shape a fresh model config, and a persisted one wins on
//! reload.

use crate::common::cli::{AppArgs, finish_extra};
use crate::dataset::{self, Group};
use crate::model;
use burn_mamba::prelude::RotationKind;

/// `-- --help`.
const HELP: &str = "\
reset-quintic's own flags (after --):
    --group <G>            The group the stream is written in: s5 (default) or a5 (not persisted)
    --rotation <KIND>      The rotation of a fresh model config: rotor (default), quaternion or complex
    --layers <N>           Layers of a fresh model config (default: the fewest that hold the group)
    --train-length <N>     Length of the training sequences
    --d-model <N>          Width of a fresh model config
    --expand <N>           Its expansion factor (default 1)
    --heads <N>            Its heads
    --mimo-rank <N>        Its MIMO rank";

/// The parsed flags.
pub struct Cli {
    /// `--group`, defaulting to `S₅`, the one that needs the second layer.
    pub group: Group,
    /// `--rotation`, defaulting to the full `SO(4)`.
    pub rotation: RotationKind,
    /// `--layers`, defaulting to the fewest that hold the group.
    pub layers: usize,
    /// `--train-length`.
    pub train_length: usize,
    /// `--d-model`.
    pub d_model: usize,
    /// `--expand`.
    pub expand: usize,
    /// `--heads`.
    pub heads: usize,
    /// `--mimo-rank`.
    pub mimo_rank: usize,
}

impl Cli {
    /// Parse the arguments forwarded after `--`.
    pub fn parse(app_args: &AppArgs) -> Self {
        let mut pargs = app_args.extra(HELP);
        let group = pargs
            .opt_value_from_fn("--group", parse_group)
            .unwrap()
            .unwrap_or(Group::Symmetric);
        let rotation = pargs
            .opt_value_from_fn("--rotation", crate::common::parse_rotation)
            .unwrap()
            .unwrap_or(RotationKind::Rotor4D);
        let mut value = |flag: &'static str, default: usize| -> usize {
            pargs.opt_value_from_str(flag).unwrap().unwrap_or(default)
        };
        let (d_model, heads, mimo_rank) = model::DEFAULT_WIDTH;
        let cli = Self {
            group,
            rotation,
            layers: value("--layers", model::default_layers(group)),
            train_length: value("--train-length", dataset::SEQ_LENGTH),
            d_model: value("--d-model", d_model),
            expand: value("--expand", 1),
            heads: value("--heads", heads),
            mimo_rank: value("--mimo-rank", mimo_rank),
        };
        finish_extra(pargs);
        cli
    }
}

fn parse_group(value: &str) -> Result<Group, String> {
    match value {
        "s5" => Ok(Group::Symmetric),
        "a5" => Ok(Group::Alternating),
        other => Err(format!("--group must be 'a5' or 's5', got {other:?}")),
    }
}
