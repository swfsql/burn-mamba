//! The untie lists of the families: the tensors of each untie choice, the
//! layout of Mamba-3, and its untie checks.
//!
//! The contract `untied_stack_equals_the_stack_of_its_application_views`
//! (`block_contract.rs`) checks the untied settings of each family in a
//! stack. The layer tiles only the tensors that `untied_params` names. So the
//! contract does not see a tensor that `untied_params` misses, or a wrong
//! axis. The table tests compare `untied_params` with a table, one test for
//! each family. The table has one case for each untie choice:
//!
//! - `mamba1_each_untie_choice_names_its_tensors`,
//! - `mamba2_each_untie_choice_names_its_tensors`,
//! - `mamba3_each_untie_choice_names_its_tensors`.
//!
//! This file also checks the split of the `in_proj` tail of Mamba-3: it cuts
//! the stock `in_proj` again, and it adds no new layout. It also checks the
//! untie list against the Kalman gain and the tropical register. The init
//! panics when the list holds a tensor that the config does not choose.

#[cfg(feature = "mamba1")]
use super::m1;
#[cfg(feature = "mamba3")]
use super::{D_MODEL, m3};
#[cfg(feature = "mamba2")]
use super::m2;
#[cfg(feature = "mamba1")]
use crate::mamba1::prelude::Mamba1Config;
#[cfg(feature = "mamba2")]
use crate::mamba2::prelude::Mamba2Config;
#[cfg(feature = "mamba3")]
use crate::mamba3::prelude::Mamba3Untied::{DtBias, InProjTail, KalmanKappa, KalmanRead, TropicalReadout};
#[cfg(feature = "mamba3")]
use crate::mamba3::prelude::{Gain, Mamba3Config, Tropical};
use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use burn_stack::modules::Block;
use burn_stack::utils::test_helpers::test_device;
#[cfg(feature = "mamba3")]
use burn_stack::utils::{test_helpers::panic_text, untied};
use std::collections::HashMap;
use std::fmt::Debug;
#[cfg(feature = "mamba3")]
use std::panic::{AssertUnwindSafe, catch_unwind};

// ---------------------------------------------------------------------------
// The table tests
// ---------------------------------------------------------------------------

/// The path and the axis of each tensor of a case, in the order of
/// `untied_params`. A path is the module path of the parameter in the block.
type PathAxes = &'static [(&'static str, usize)];

/// Check the table of one family. For each untie choice of `parts`, `build`
/// builds the block with that one choice. It also gives the path and the axis
/// of each tensor of the case of that choice. The check fails with one line
/// for each untie choice whose `untied_params` does not give these paths and
/// axes.
fn check_table<P: Copy + Debug, B: Block>(parts: &[P], build: impl Fn(P) -> (B, PathAxes)) {
    let failures: Vec<String> = parts
        .iter()
        .filter_map(|&part| {
            let (block, want) = build(part);
            let paths = param_paths(&block);
            let got: Vec<(&str, usize)> = block
                .untied_params()
                .iter()
                .map(|param| {
                    let path = paths.get(&param.id).map_or("<not a parameter of the block>", String::as_str);
                    (path, param.axis)
                })
                .collect();
            (got != want).then(|| format!("- {part:?}: `untied_params` gives {got:?}, and the case gives {want:?}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "`untied_params` does not give the tensors of the case of these untie choices:\n{}",
        failures.join("\n")
    );
}

/// The module path of each float parameter of `module`, by its id.
fn param_paths(module: &impl Module) -> HashMap<ParamId, String> {
    #[derive(Default)]
    struct Collect {
        path: Vec<String>,
        paths: HashMap<ParamId, String>,
    }
    impl ModuleVisitor for Collect {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.path.push(name.to_string());
        }
        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.paths.insert(param.id, self.path.join("."));
        }
    }
    let mut collect = Collect::default();
    module.visit(&mut collect);
    collect.paths
}

/// `Mamba1::untied_params` names the tensors of each untie choice, each with
/// the axis of its copies.
#[test]
#[cfg(feature = "mamba1")]
fn mamba1_each_untie_choice_names_its_tensors() {
    use crate::mamba1::prelude::Mamba1Untied::{self, *};
    /// The case of `part`: a config that enables each tensor of `part`, and
    /// the path and the axis of each such tensor.
    fn case(part: Mamba1Untied) -> (Mamba1Config, PathAxes) {
        match part {
            Conv1d => (m1().with_has_conv_bias(true), &[("conv1d.weight", 0), ("conv1d.bias", 0)]),
            XProj => (m1(), &[("x_proj.weight", 1)]),
            DtProj => (m1(), &[("dt_proj.weight", 1), ("dt_proj.bias", 0)]),
            ALog => (m1(), &[("a_log", 0)]),
            D => (m1(), &[("d", 0)]),
        }
    }
    let device = test_device();
    check_table(&[Conv1d, XProj, DtProj, ALog, D], |part| {
        let (config, want) = case(part);
        (config.with_untied(vec![part]).init(&device), want)
    });
}

/// `Mamba2::untied_params` names the tensors of each untie choice, each with
/// the axis of its copies.
#[test]
#[cfg(feature = "mamba2")]
fn mamba2_each_untie_choice_names_its_tensors() {
    use crate::mamba2::prelude::Mamba2Untied::{self, *};
    /// The case of `part`: a config that enables each tensor of `part`, and
    /// the path and the axis of each such tensor.
    fn case(part: Mamba2Untied) -> (Mamba2Config, PathAxes) {
        match part {
            InProjTail => (m2().with_has_proj_bias(true), &[("in_proj_tail.weight", 1), ("in_proj_tail.bias", 0)]),
            Conv1d => (m2().with_has_conv_bias(true), &[("conv1d.weight", 0), ("conv1d.bias", 0)]),
            DtBias => (m2(), &[("dt_bias_h", 0)]),
            ALog => (m2(), &[("a_log_h", 0)]),
            D => (m2(), &[("d_h", 0)]),
            Norm => (m2(), &[("norm.gamma", 0)]),
            InitState => (m2().with_has_learnable_init_state(true), &[("init_state_hpr", 0)]),
        }
    }
    let device = test_device();
    check_table(&[InProjTail, Conv1d, DtBias, ALog, D, Norm, InitState], |part| {
        let (config, want) = case(part);
        (config.with_untied(vec![part]).init(&device), want)
    });
}

/// `Mamba3::untied_params` names the tensors of each untie choice, each with
/// the axis of its copies.
#[test]
#[cfg(feature = "mamba3")]
fn mamba3_each_untie_choice_names_its_tensors() {
    use crate::mamba3::prelude::Mamba3Untied::{self, *};
    /// The case of `part`: a config that enables each tensor of `part`, and
    /// the path and the axis of each such tensor.
    fn case(part: Mamba3Untied) -> (Mamba3Config, PathAxes) {
        match part {
            InProjTail => (m3().with_has_proj_bias(true), &[("in_proj_tail.weight", 1), ("in_proj_tail.bias", 0)]),
            DtBias => (m3(), &[("dt_bias_h", 0)]),
            D => (m3(), &[("d_h", 0)]),
            BNorm => (m3(), &[("b_norm.gamma", 0)]),
            CNorm => (m3(), &[("c_norm.gamma", 0)]),
            BBias => (m3(), &[("b_bias_hmr", 0)]),
            CBias => (m3(), &[("c_bias_hmr", 0)]),
            MimoX => (m3().with_mimo_rank(2), &[("mimo_x_hmp", 0)]),
            MimoZ => (m3().with_mimo_rank(2), &[("mimo_z_hmp", 0)]),
            MimoO => (m3().with_mimo_rank(2), &[("mimo_o_hmp", 0)]),
            OutNorm => (m3().with_has_outproj_norm(true), &[("out_norm.gamma", 0)]),
            InitState => (m3().with_has_learnable_init_state(true), &[("init_state_hpr", 0)]),
            KalmanKappa => (m3().with_gain(Gain::Kalman), &[("kalman_log_kappa_h", 0)]),
            KalmanRead => (m3().with_gain(Gain::Kalman), &[("kalman_read_h", 0)]),
            TropicalReadout => (m3().with_tropical(Tropical::MaxPlus), &[("tropical_readout_hp", 0)]),
        }
    }
    let device = test_device();
    let parts = [
        InProjTail, DtBias, D, BNorm, CNorm, BBias, CBias, MimoX, MimoZ, MimoO, OutNorm, InitState, KalmanKappa,
        KalmanRead, TropicalReadout,
    ];
    check_table(&parts, |part| {
        let (config, want) = case(part);
        (config.with_untied(vec![part]).init(&device), want)
    });
}

// ---------------------------------------------------------------------------
// The layout of Mamba-3, and its untie checks
// ---------------------------------------------------------------------------

/// The layout follows only the untie list. With no untied tensor, the block
/// is the stock block at each count of applications. One application of an
/// untied block holds the stock count of parameters, because the split of
/// the tail cuts `in_proj` and copies nothing. `n` applications hold `n`
/// copies of each untied tensor. The test tiles the block for 3 applications,
/// as the layer does.
#[test]
#[cfg(feature = "mamba3")]
fn mamba3_layout_follows_the_untie_list() {
    let device = test_device();
    let tiled = |config: &Mamba3Config| {
        let block = config.init(&device);
        let params = block.untied_params();
        untied::tile(block, &params, 3)
    };
    let stock = m3().init(&device).num_params();
    assert_eq!(tiled(&m3()).num_params(), stock);

    let config = m3().with_untied(vec![InProjTail, DtBias]);
    assert_eq!(config.init(&device).num_params(), stock);

    let block = tiled(&config);
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
#[cfg(feature = "mamba3")]
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
