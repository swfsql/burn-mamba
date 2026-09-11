//! Untied parameters through the Mamba families.
//!
//! The container contract — copies start tied and split the tied gradient, a
//! cut through an untied layer panics, an `InitPolicy` keeps copies tied — is
//! `burn-stack`'s, pinned against its reference block. What a family owes on
//! top is that its `untied_params` names exactly what its config tiles, that
//! `forward` and `step` read the same copies, that the `in_proj` tail split is
//! a re-cut of the stock layout rather than a new one, and that the optimizer
//! gets one weight per application.

use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_stack::modules::{Block, BlockConfig, LayerUntied, Layers, LayersBuilder};
use burn_stack::utils::Schedule;
use burn_stack::utils::test_helpers::max_abs_diff;

type Device = burn::prelude::Device;

const D_MODEL: usize = 16;
const BATCH: usize = 2;
const SEQ: usize = 4;
/// Two real layers over five virtual ones: `Cyclic` gives them three
/// applications and two.
const N_REAL: usize = 2;
const N_VIRTUAL: usize = 5;
const TOL: f32 = 1e-4;

/// The small SISO Mamba-3 the other suites use.
#[cfg(feature = "mamba3")]
fn mamba3_config() -> crate::mamba3::prelude::Mamba3Config {
    crate::mamba3::prelude::Mamba3Config::new(D_MODEL)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_rope_fraction(0.5)
}

/// Adds noise to every parameter, so an untied stack's copies stop agreeing.
struct Jitter;

impl ModuleMapper for Jitter {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|t| {
            let noise = Tensor::random(t.shape(), Distribution::Normal(0.0, 0.05), &t.device());
            t + noise
        })
    }
}

/// An untied stack is the *unshared* stack of its application views: virtual
/// layer `i` runs application `index[i]` of its real layer, in `forward` and in
/// `step`. The copies are jittered apart first, so reading the wrong one — or a
/// tiled tensor the family forgot to list — shows.
fn assert_untied_stack_is_its_unshared_views<C>(block: C, options: <C::Block as Block>::Options)
where
    C: BlockConfig,
    <C::Block as Block>::Options: Clone,
{
    let device: Device = Default::default();
    let schedule = Schedule::Cyclic;
    let layers = LayersBuilder::new(N_REAL, block)
        .with_n_virtual_layers(Some((N_VIRTUAL, schedule.clone())))
        .with_untied(vec![LayerUntied::Norm])
        .init(&device)
        .map(&mut Jitter);
    let apps = schedule.applications(N_VIRTUAL, N_REAL);
    let unshared = Layers {
        n_real_layers: N_VIRTUAL,
        n_virtual_layers: None,
        real_layers: (0..N_VIRTUAL)
            .map(|i| {
                let real = schedule.real_idx(i, N_VIRTUAL, N_REAL);
                layers.real_layers[real].application(apps.index[i]).into_owned()
            })
            .collect(),
        ..layers.clone()
    };
    let x = Tensor::<3>::random([BATCH, SEQ, D_MODEL], Distribution::Normal(0.0, 1.0), &device);

    let (y, _) = layers.forward(x.clone(), None, options.clone(), None);
    let (want, _) = unshared.forward(x.clone(), None, options.clone(), None);
    let diff = max_abs_diff(y.clone(), want);
    assert!(diff < TOL, "forward reads the wrong copies: {diff}");

    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..SEQ {
        let x_t = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
        let (y_t, c) = layers.step(x_t, caches, None);
        caches = Some(c);
        ys.push(y_t.unsqueeze_dim::<3>(1));
    }
    let diff = max_abs_diff(y.clone(), Tensor::cat(ys, 1));
    assert!(diff < TOL, "step reads other copies than forward: {diff}");

    // Neither comparison proves anything unless the copies really differ.
    let tied = Layers {
        real_layers: layers.real_layers.iter().map(|l| l.application(0).into_owned()).collect(),
        ..layers.clone()
    };
    let (y_tied, _) = tied.forward(x, None, options, None);
    assert!(max_abs_diff(y, y_tied) > TOL, "the jitter left the copies equal");
}

/// Every Mamba-3 tensor that can be untied, untied — MIMO and the output norm on
/// so they exist. The learnable initial state is left out: only the minimal SSD
/// kernel reads it, and `step` never does.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_untied_stack_is_its_unshared_views() {
    use crate::mamba3::prelude::{Mamba3SsdPath, Mamba3Untied::*};

    let block = mamba3_config()
        .with_mimo_rank(2)
        .with_has_outproj_norm(true)
        .with_untied(vec![
            InProjTail, DtBias, D, BNorm, CNorm, BBias, CBias, MimoX, MimoZ, MimoO, OutNorm,
        ]);
    assert_untied_stack_is_its_unshared_views(block, Mamba3SsdPath::Minimal(Some(4)));
}

/// Every Mamba-2 tensor that can be untied, bar the learnable initial state (as
/// for Mamba-3).
#[cfg(feature = "mamba2")]
#[test]
fn mamba2_untied_stack_is_its_unshared_views() {
    use crate::mamba2::prelude::{Mamba2Config, Mamba2SsdPath, Mamba2Untied::*};

    let block = Mamba2Config::new(D_MODEL)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_untied(vec![InProjTail, Conv1d, DtBias, ALog, D, Norm]);
    assert_untied_stack_is_its_unshared_views(block, Mamba2SsdPath::Minimal(Some(4)));
}

/// Every Mamba-1 tensor that can be untied.
#[cfg(feature = "mamba1")]
#[test]
fn mamba1_untied_stack_is_its_unshared_views() {
    use crate::mamba1::prelude::{Mamba1Config, Mamba1Untied::*};

    let block = Mamba1Config::new(D_MODEL)
        .with_state_rank(4)
        .with_untied(vec![Conv1d, XProj, DtProj, ALog, D]);
    assert_untied_stack_is_its_unshared_views(block, ());
}

/// The layout follows the untie list alone: nothing untied is the stock block at
/// any count, a single application of an untied block holds the stock parameter
/// count (the tail split re-cuts `in_proj`, it copies nothing), and `n`
/// applications hold `n` copies of each untied tensor.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_layout_follows_the_untie_list() {
    use crate::mamba3::prelude::Mamba3Untied::{DtBias, InProjTail};

    let device: Device = Default::default();
    let stock = mamba3_config().init(&device).num_params();
    assert_eq!(mamba3_config().init_applications(3, &device).num_params(), stock);

    let config = mamba3_config().with_untied(vec![InProjTail, DtBias]);
    assert_eq!(config.init_applications(1, &device).num_params(), stock);

    let block = config.init_applications(3, &device);
    let tail = config.d_in_proj_tail();
    assert_eq!(block.in_proj.weight.dims(), [D_MODEL, config.d_in_proj() - tail]);
    let tail_weight = &block.in_proj_tail.as_ref().expect("split off").weight;
    assert_eq!(tail_weight.dims(), [D_MODEL, 3 * tail]);
    assert_eq!(block.num_params(), stock + 2 * (D_MODEL * tail + config.nheads()));
}

/// An untied `in_proj` tail is one weight per application to the optimizer: the
/// plan tiles its spec, each copy's rotation block is stepped on its own, and
/// the copies part under training.
#[cfg(all(feature = "mamba3", feature = "optim"))]
#[test]
fn mamba3_untied_tail_trains_one_copy_per_application() {
    use crate::mamba3::prelude::{Mamba3SsdPath, Mamba3Untied};
    use burn::optim::{AdamWConfig, GradientsParams};
    use burn_stack::optim::{MuonPlan, muon_config};

    let device = Device::default().autodiff();
    let block = mamba3_config().with_untied(vec![Mamba3Untied::InProjTail]);
    let layers = LayersBuilder::new(1, block.clone())
        .with_n_virtual_layers(Some((3, Schedule::Cyclic)))
        .init(&device);
    let plan = MuonPlan::new(block.muon_projections());
    let report = plan.describe(&layers);
    assert!(
        report
            .lines()
            .any(|l| l.contains("in_proj_tail.weight") && l.contains("rotation") && l.ends_with("×3")),
        "{report}",
    );

    let mut optim = plan.build(&AdamWConfig::new(), &muon_config(0.0));
    let x = Tensor::<3>::random([BATCH, SEQ, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let (y, _) = layers.forward(x, None, Mamba3SsdPath::Minimal(Some(4)), None);
    let grads = GradientsParams::from_grads(y.sum().backward(), &layers);
    let layers = optim.step(1e-2, layers, grads);

    let tail = layers.real_layers[0].block.in_proj_tail.as_ref().expect("untied tail");
    let copies = tail.weight.val().chunk(3, 1);
    assert!(
        max_abs_diff(copies[0].clone(), copies[1].clone()) > 0.0,
        "the copies part under training",
    );
}
