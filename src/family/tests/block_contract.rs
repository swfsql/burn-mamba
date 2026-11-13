//! The block contracts of burn-stack on each block setting of this crate.
//!
//! burn-stack proves "block contracts ⇒ the containers work". This module
//! proves "the Mamba blocks ⇒ block contracts". The module header of
//! `burn_stack::block_contract` states each contract.
//!
//! Each test runs one contract on each block setting. It collects each
//! failure with the name of its setting, and at the end it fails with all of
//! them. So one failure hides no other. The name of a test is the name of its
//! contract, and the panic of a contract also names the contract.
//!
//! ## The block settings
//!
//! - Mamba-1.
//! - Mamba-2 on each SSD path, and on the default SSD path.
//! - Mamba-3 on each SSD path, and on the default SSD path.
//! - Mamba-3 on the base setting, and on settings that change the rotation,
//!   the tap pattern, `mimo_rank` or `micro_steps` of the base. The base has
//!   the `SerialRecalculated` path, the `Complex2D` rotation, the default tap
//!   pattern, `mimo_rank` 1 and `micro_steps` 1.
//! - [`MambaBlock`] over one family: the first enabled family of Mamba-3,
//!   Mamba-2 and Mamba-1.
//!
//! A window of a contract has 7 tokens. So each SSD path has a chunk length
//! of [`CHUNK_TOKENS`] tokens, and a window holds a chunk boundary. A scan
//! error between two chunks then shows. The default SSD path is
//! `SerialRecalculated` with the optimal chunk length. Its settings keep that
//! length, so they cover the choice of the length.
//!
//! At `micro_steps` 1, the tap patterns `Vertical` and
//! `VerticalPlusHorizontalCarryOver` are the default pattern (see
//! [`TapPattern`]). So their settings also set `micro_steps` to 2.
//!
//! ## The limits
//!
//! - No contract starts from a double-SSD cache, and the cache variant of
//!   Mamba-3 selects the pathway. So no setting reaches the double-SSD
//!   pathway.
//! - `cache_traversal_moves_every_field` reads outputs, not fields. A field
//!   that one more step does not change stays unseen. `capture.rs` compares
//!   each field directly.

use crate::prelude::*;
use burn_stack::block_contract;
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

const D_MODEL: usize = 16;
/// The chunk length of each SSD path except the default SSD path, in tokens.
/// It is shorter than a window of a contract.
#[cfg(any(feature = "mamba2", feature = "mamba3"))]
const CHUNK_TOKENS: usize = 4;

/// The fp32 tolerance of Mamba-1. A contract passes it through `dtype_tol`.
#[cfg(feature = "mamba1")]
const MAMBA1_TOL: f32 = 1e-4;
/// The fp32 tolerance of Mamba-2. A contract passes it through `dtype_tol`.
#[cfg(feature = "mamba2")]
const MAMBA2_TOL: f32 = 1e-4;
/// The fp32 tolerance of Mamba-3. A contract passes it through `dtype_tol`.
#[cfg(feature = "mamba3")]
const MAMBA3_TOL: f32 = 1e-4;

/// The options of the block of the config `C`.
type Options<C> = <<C as BlockConfig>::Block as Block>::Options;

/// One block setting: its name, its config, its options, and the fp32
/// tolerance of its family.
struct Setting<C: BlockConfig> {
    name: String,
    config: C,
    options: Options<C>,
    f32_tol: f32,
}

/// One block contract of burn-stack, for each block config.
trait Contract {
    fn run<C>(config: &C, options: Options<C>, f32_tol: f32)
    where
        C: BlockConfig + Clone,
        Options<C>: Clone;
}

/// One test for each contract. The test has the name of the contract, and it
/// runs the contract on each block setting.
macro_rules! contract_tests {
    ($($(#[$attr:meta])* $contract:ident,)*) => {$(
        $(#[$attr])*
        #[test]
        fn $contract() {
            struct ThisContract;
            impl Contract for ThisContract {
                fn run<C>(config: &C, options: Options<C>, f32_tol: f32)
                where
                    C: BlockConfig + Clone,
                    Options<C>: Clone,
                {
                    block_contract::$contract(config, options, f32_tol);
                }
            }
            on_each_setting::<ThisContract>();
        }
    )*};
}

contract_tests! {
    forward_equals_step_unrolled,
    forward_splits_into_windows_through_the_cache,
    padded_rows_equal_each_row_alone,
    packed_segments_equal_each_segment_alone,
    record_round_trip_keeps_the_block,
    cache_traversal_moves_every_field,
    #[cfg(feature = "optim")]
    muon_projections_name_existing_rank_2_weights,
    #[cfg(feature = "optim")]
    muon_projections_select_only_weights_of_the_block,
}

/// Run the contract `K` on each block setting. Panic at the end with each
/// failure, if there are failures.
fn on_each_setting<K: Contract>() {
    let mut failures = Vec::new();
    #[cfg(feature = "mamba1")]
    run_on::<K, _>(mamba1_settings(), &mut failures);
    #[cfg(feature = "mamba2")]
    run_on::<K, _>(mamba2_settings(), &mut failures);
    #[cfg(feature = "mamba3")]
    run_on::<K, _>(mamba3_settings(), &mut failures);
    run_on::<K, _>(mamba_block_settings(), &mut failures);
    assert!(failures.is_empty(), "The contract fails on these block settings:\n{}", failures.join("\n"));
}

/// Run the contract `K` on each setting of `settings`. Add each failure to
/// `failures`, with the name of its setting.
fn run_on<K: Contract, C>(settings: Vec<Setting<C>>, failures: &mut Vec<String>)
where
    C: BlockConfig + Clone,
    Options<C>: Clone,
{
    for setting in settings {
        let Setting { name, config, options, f32_tol } = setting;
        let result = catch_unwind(AssertUnwindSafe(|| K::run(&config, options, f32_tol)));
        if let Err(payload) = result {
            failures.push(format!("- {name}: {}", panic_message(payload)));
        }
    }
}

/// The text of a panic.
fn panic_message(payload: Box<dyn Any + Send>) -> String {
    match (payload.downcast_ref::<String>(), payload.downcast_ref::<&str>()) {
        (Some(message), _) => message.clone(),
        (None, Some(message)) => message.to_string(),
        (None, None) => "a panic with no text".to_string(),
    }
}

#[cfg(feature = "mamba1")]
fn m1() -> Mamba1Config {
    Mamba1Config::new(D_MODEL).with_state_rank(8)
}

#[cfg(feature = "mamba2")]
fn m2() -> Mamba2Config {
    Mamba2Config::new(D_MODEL)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4)
}

/// The base setting of Mamba-3, without its SSD path.
#[cfg(feature = "mamba3")]
fn m3() -> Mamba3Config {
    Mamba3Config::new(D_MODEL)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rotation_fraction(0.5)
}

#[cfg(feature = "mamba1")]
fn mamba1_settings() -> Vec<Setting<Mamba1Config>> {
    vec![Setting { name: "Mamba-1".to_string(), config: m1(), options: (), f32_tol: MAMBA1_TOL }]
}

#[cfg(feature = "mamba2")]
fn mamba2_settings() -> Vec<Setting<Mamba2Config>> {
    let paths = [
        Mamba2SsdPath::default(),
        Mamba2SsdPath::Minimal(Some(CHUNK_TOKENS)),
        Mamba2SsdPath::Serial(Some(CHUNK_TOKENS)),
        Mamba2SsdPath::SerialRecalculated(Some(CHUNK_TOKENS)),
    ];
    paths
        .into_iter()
        .map(|path| Setting { name: format!("Mamba-2, {path:?}"), config: m2(), options: path, f32_tol: MAMBA2_TOL })
        .collect()
}

/// The SSD path `path` of a Mamba-3 block, with a chunk length of
/// [`CHUNK_TOKENS`] tokens. A chunk length of Mamba-3 counts write positions,
/// and a token has `micro_steps` write positions.
#[cfg(feature = "mamba3")]
fn m3_path(path: fn(Option<usize>) -> Mamba3SsdPath, config: &Mamba3Config) -> Mamba3SsdPath {
    path(Some(CHUNK_TOKENS * config.micro_steps))
}

#[cfg(feature = "mamba3")]
fn mamba3_settings() -> Vec<Setting<Mamba3Config>> {
    use Mamba3SsdPath::{Minimal, Serial, SerialRecalculated};
    let u2 = || m3().with_micro_steps(2);
    let short_chunks: Vec<(Mamba3Config, fn(Option<usize>) -> Mamba3SsdPath)> = vec![
        (m3(), Minimal),
        (m3(), Serial),
        (m3(), SerialRecalculated),
        (m3().with_rotation(RotationKind::Real1D), SerialRecalculated),
        (m3().with_rotation(RotationKind::Quaternion4D), SerialRecalculated),
        (m3().with_tap_pattern(TapPattern::None), SerialRecalculated),
        (u2().with_tap_pattern(TapPattern::Vertical), SerialRecalculated),
        (u2().with_tap_pattern(TapPattern::VerticalPlusHorizontalCarryOver), SerialRecalculated),
        (m3().with_mimo_rank(2), SerialRecalculated),
        (u2(), SerialRecalculated),
    ];
    let mut settings = vec![(m3(), Mamba3SsdPath::default())];
    settings.extend(short_chunks.into_iter().map(|(config, path)| {
        let path = m3_path(path, &config);
        (config, path)
    }));
    settings
        .into_iter()
        .map(|(config, path)| Setting {
            name: format!(
                "Mamba-3, {path:?}, {:?}, {:?}, mimo_rank {}, micro_steps {}",
                config.rotation, config.tap_pattern, config.mimo_rank, config.micro_steps,
            ),
            config,
            options: path,
            f32_tol: MAMBA3_TOL,
        })
        .collect()
}

/// [`MambaBlock`] over one family. The enum gives each call to the block of
/// its family, so one family is enough.
fn mamba_block_settings() -> Vec<Setting<MambaBlockConfig>> {
    let families = vec![
        #[cfg(feature = "mamba3")]
        (
            MambaBlockConfig::Mamba3(m3()),
            MambaSsdPath::Mamba3(m3_path(Mamba3SsdPath::SerialRecalculated, &m3())),
            MAMBA3_TOL,
        ),
        #[cfg(feature = "mamba2")]
        (
            MambaBlockConfig::Mamba2(m2()),
            MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(Some(CHUNK_TOKENS))),
            MAMBA2_TOL,
        ),
        #[cfg(feature = "mamba1")]
        (MambaBlockConfig::Mamba1(m1()), MambaSsdPath::Mamba1, MAMBA1_TOL),
    ];
    families
        .into_iter()
        .take(1)
        .map(|(config, path, f32_tol)| Setting { name: format!("MambaBlock, {path:?}"), config, options: path, f32_tol })
        .collect()
}
