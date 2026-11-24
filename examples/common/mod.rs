//! Shared infrastructure for the burn-mamba examples.
//!
//! Almost nothing is here. These parts are in `burn_stack::examples`, shared
//! verbatim with `burn-deltanet`:
//!
//! - the CLI and the artifact handling,
//! - the runtime device selection,
//! - the training config,
//! - the two datasets (sequential-MNIST and the character-level TinyStories
//!   corpus), each with its epoch loops.
//!
//! The `config → module` interface is `burn_stack::modules::ModelConfigExt`,
//! which the network configs of this crate implement. This module only
//! re-exports those items under the `common::*` paths that the examples use.
//! It adds the things that must be *here*, in the example crate:
//!
//! - [`ARTIFACT_PREFIX`], the one constant that must expand in this crate.
//! - [`parse_rotation`], [`parse_ssd_path`], [`parse_ssd_chunk_len`] and
//!   [`parse_pathway`], the parsers of the flags of the rotation and of the
//!   scan options. These flags are specific to Mamba. `burn-stack` is
//!   family-agnostic, so it cannot have these parsers.
//!
//! It also holds two measures that the two TinyStories examples of this crate
//! (`tiny-stories`, `hnet-stories`) share: [`context`] (the context use) and
//! [`long_stories`] (the bits of whole long stories). They read a model only
//! through a logits closure.
//!
//! With the Dispatch-based architecture, no module here has a generic backend
//! type. `Tensor`/`Device`/`Module` are pinned to the global `Dispatch`
//! backend, and the device selects the concrete runtime backend.

#![allow(dead_code)]

pub use burn_stack::examples::{cli, device, mnist, session, tiny_stories, training};

/// How much a trained LM uses the earlier part of a story (`--context-use`),
/// for any model of the TinyStories examples.
pub mod context;

/// The bits of whole long validation stories (`--long-stories`), for any model
/// of the TinyStories examples.
pub mod long_stories;

/// The `ModelConfigExt` interface, under its usual `common::model` path.
pub mod model {
    pub use burn_stack::modules::ModelConfigExt;
}

/// Prefix of the artifacts directory that a run creates when `--artifacts-path`
/// is absent, for example `burn-mamba-mnist-class-`. Both halves belong to this
/// example target, so it must expand here, not in `burn-stack`.
pub const ARTIFACT_PREFIX: &str = concat!(
    std::env!("CARGO_PKG_NAME"), // burn-mamba
    "-",
    std::env!("CARGO_BIN_NAME"), // e.g. reset-majority
    "-"
);

/// A `--rotation complex|quaternion|rotor` value (`quat` and `so4` are
/// aliases), for `pico_args::Arguments::opt_value_from_fn`.
pub fn parse_rotation(value: &str) -> Result<burn_mamba::prelude::RotationKind, String> {
    use burn_mamba::prelude::RotationKind;
    match value {
        "complex" => Ok(RotationKind::Complex2D),
        "quaternion" | "quat" => Ok(RotationKind::Quaternion4D),
        "rotor" | "so4" => Ok(RotationKind::Rotor4D),
        other => Err(format!(
            "--rotation must be 'complex', 'quaternion' or 'rotor', got {other:?}"
        )),
    }
}

/// A `--ssd-path <p>` value (`recalc | serial | minimal`), for
/// `pico_args::Arguments::opt_value_from_fn`.
pub fn parse_ssd_path(arg: &str) -> Result<burn_mamba::prelude::SsdPath, String> {
    use burn_mamba::prelude::SsdPath;
    match arg {
        "recalc" => Ok(SsdPath::SerialRecalculated),
        "serial" => Ok(SsdPath::Serial),
        "minimal" => Ok(SsdPath::Minimal),
        other => Err(format!("unknown SSD path {other:?} (recalc | serial | minimal)")),
    }
}

/// A `--pathway <p>` value (`single | double`), for
/// `pico_args::Arguments::opt_value_from_fn`.
pub fn parse_pathway(arg: &str) -> Result<burn_mamba::prelude::Pathway, String> {
    use burn_mamba::prelude::Pathway;
    match arg {
        "single" => Ok(Pathway::SingleSsd),
        "double" => Ok(Pathway::DoubleSsd),
        other => Err(format!("unknown pathway {other:?} (single | double)")),
    }
}

/// A `--ssd-chunk-len <l>` value: a chunk length of at least 1, in write
/// positions, for `pico_args::Arguments::opt_value_from_fn`.
pub fn parse_ssd_chunk_len(arg: &str) -> Result<usize, String> {
    match arg.parse::<usize>() {
        Ok(chunk_len) if chunk_len >= 1 => Ok(chunk_len),
        _ => Err(format!("chunk length {arg:?} is not a positive integer")),
    }
}
