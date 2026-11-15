//! The layout of the untied parameters of Mamba-3.
//!
//! The contract `untied_stack_equals_the_stack_of_its_application_views`
//! (`block_contract.rs`) checks the untied settings of each family in a
//! stack. This file checks the split of the `in_proj` tail of Mamba-3: it
//! cuts the stock `in_proj` again, and it adds no new layout.

use super::{D_MODEL, m3};
use crate::mamba3::prelude::Mamba3Untied::{DtBias, InProjTail};
use burn::module::Module;
use burn_stack::utils::test_helpers::test_device;

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
