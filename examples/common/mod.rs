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
//! It adds two things that must be *here*, in the example crate:
//!
//! - [`ARTIFACT_PREFIX`], the one constant that must expand in this crate.
//! - [`parse_rotation`], the parser of a flag that names a type of this crate.
//!   `burn-stack` is family-agnostic, so it cannot have this parser.
//!
//! With the Dispatch-based architecture, no module here has a generic backend
//! type. `Tensor`/`Device`/`Module` are pinned to the global `Dispatch`
//! backend, and the device selects the concrete runtime backend.

#![allow(dead_code)]

pub use burn_stack::examples::{cli, device, mnist, session, tiny_stories, training};

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
