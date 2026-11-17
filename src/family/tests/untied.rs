//! The untie list of Mamba-3: the layout that it gives, and the config checks
//! of the Kalman gain and of the tropical register.
//!
//! The contract `untied_stack_equals_the_stack_of_its_application_views`
//! (`block_contract.rs`) checks the untied settings of each family in a
//! stack. This file checks the split of the `in_proj` tail of Mamba-3: it
//! cuts the stock `in_proj` again, and it adds no new layout. It also checks
//! the untie list against the Kalman gain and the tropical register. The init
//! panics when the list holds a tensor that the config does not choose.

use super::{D_MODEL, m3};
use crate::mamba3::prelude::Mamba3Untied::{DtBias, InProjTail, KalmanKappa, KalmanRead, TropicalReadout};
use crate::mamba3::prelude::{Gain, Tropical};
use burn::module::Module;
use burn_stack::utils::test_helpers::{panic_text, test_device};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The layout follows only the untie list. With no untied tensor, the block
/// is the stock block at each count of applications. One application of an
/// untied block holds the stock count of parameters, because the split of
/// the tail cuts `in_proj` and copies nothing. `n` applications hold `n`
/// copies of each untied tensor.
#[test]
fn mamba3_layout_follows_the_untie_list() {
    let device = test_device();
    let stock = m3().init(&device).num_params();
    assert_eq!(m3().init_applications(3, &device).num_params(), stock);

    let config = m3().with_untied(vec![InProjTail, DtBias]);
    assert_eq!(config.init_applications(1, &device).num_params(), stock);

    let block = config.init_applications(3, &device);
    let tail = config.d_in_proj_tail();
    assert_eq!(block.in_proj.weight.dims(), [D_MODEL, config.d_in_proj() - tail]);
    let tail_weight = &block.in_proj_tail.as_ref().expect("split off").weight;
    assert_eq!(tail_weight.dims(), [D_MODEL, 3 * tail]);
    assert_eq!(block.num_params(), stock + 2 * (D_MODEL * tail + config.nheads()));
}

/// An untie choice (an entry of the untie list) of the Kalman gain or of the
/// tropical register needs its tensor. Without the tensor, the init panics
/// with the message of its config check:
///
/// - `KalmanKappa` and `KalmanRead` with `Gain::Projected`,
/// - `TropicalReadout` with `Tropical::None`.
///
/// The test collects each failure with the name of its untie choice.
#[test]
fn mamba3_untie_choice_without_its_tensor_panics() {
    let device = test_device();
    let cases = [
        (
            KalmanKappa,
            m3().with_gain(Gain::Projected),
            "Mamba3Untied::KalmanKappa unties the κ of the Kalman gain, and the gain is Gain::Projected",
        ),
        (
            KalmanRead,
            m3().with_gain(Gain::Projected),
            "Mamba3Untied::KalmanRead unties the Kalman read exponent, and the gain is Gain::Projected",
        ),
        (
            TropicalReadout,
            m3().with_tropical(Tropical::None),
            "Mamba3Untied::TropicalReadout unties the tropical readout, and tropical is Tropical::None",
        ),
    ];
    let mut failures = Vec::new();
    for (part, config, expected) in cases {
        let config = config.with_untied(vec![part]);
        let result = catch_unwind(AssertUnwindSafe(|| config.init(&device)));
        match result.map_err(|payload| panic_text(payload.as_ref())) {
            Ok(_) => failures.push(format!("- {part:?}: the init does not panic")),
            Err(Some(text)) if text == expected => {}
            Err(Some(text)) => {
                failures.push(format!("- {part:?}: the init panics with `{text}`, not with `{expected}`"))
            }
            Err(None) => failures.push(format!("- {part:?}: the init panics with no text")),
        }
    }
    assert!(failures.is_empty(), "These untie choices fail:\n{}", failures.join("\n"));
}
