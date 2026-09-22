//! Shared infrastructure for the burn-mamba examples.
//!
//! Almost nothing lives here: the CLI + artifact handling, the runtime device
//! selection, the training config and the two datasets (sequential-MNIST and
//! the character-level TinyStories corpus, each with its epoch loops) are
//! `burn_stack::examples`, shared verbatim with `burn-deltanet`, and the
//! `config → module` seam is `burn_stack::modules::ModelConfigExt` (implemented
//! by this crate's network configs). This module only re-exports those under the
//! `common::*` paths the examples use, plus the one constant that has to be
//! expanded *here*, in the example crate: [`ARTIFACT_PREFIX`], and the parsers
//! of the flags that name something of this crate's (which `burn-stack`, being
//! family-agnostic, cannot): [`parse_rotation`].
//!
//! With the Dispatch-based architecture, no module here carries a backend type
//! generic — `Tensor`/`Device`/`Module` are pinned to the global `Dispatch`
//! backend, and the device chooses the concrete runtime backend.

#![allow(dead_code)]

pub use burn_stack::examples::{cli, device, mnist, session, tiny_stories, training};

/// The `ModelConfigExt` seam, under its usual `common::model` path.
pub mod model {
    pub use burn_stack::modules::ModelConfigExt;
}

/// Prefix of the artifacts directory created when `--artifacts-path` is absent,
/// e.g. `burn-mamba-mnist-class-`. Both halves belong to this example target, so
/// it must be expanded here rather than in `burn-stack`.
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
