//! The H-Net of `burn-stack` over real blocks, through [`MambaHNetVocabNet`]:
//! forward = step unrolled (logits, caches, gradients), right padding, and a
//! Muon step over the scoped plan (each stack at its own width).

use crate::prelude::*;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_stack::modules::hnet::{HNetShape, HNetStageShape, HNetVocabShape, StepMode};
use burn_stack::modules::{CacheTensors, NetworkShape, TensorZip};
use burn_stack::utils::test_helpers::{dtype_tol, max_rel_diff, test_device};

const VOCAB: usize = 11;
const TOL: f32 = 1e-4;

#[cfg(feature = "mamba2")]
fn mamba2(d_model: usize) -> MambaBlockConfig {
    MambaBlockConfig::Mamba2(
        Mamba2Config::new(d_model)
            .with_expand(2)
            .with_per_head_dim(4)
            .with_state_rank(8)
            .with_ngroups(1)
            .with_conv_kernel(4),
    )
}

#[cfg(feature = "mamba3")]
fn mamba3(d_model: usize) -> MambaBlockConfig {
    MambaBlockConfig::Mamba3(
        Mamba3Config::new(d_model)
            .with_expand(2)
            .with_per_head_dim(4)
            .with_state_rank(8)
            .with_ngroups(1)
            .with_mimo_rank(1)
            .with_rope_fraction(0.5),
    )
}

/// An H-Net with one stage per entry of `stage_blocks`, two layers per stack.
fn config(stage_blocks: Vec<MambaBlockConfig>, main_block: MambaBlockConfig) -> MambaHNetVocabNetConfig {
    let stack = || NetworkShape::new(2);
    let hnet = HNetShape::new(
        stage_blocks.iter().map(|_| HNetStageShape::new(stack(), stack())).collect(),
        stack(),
    );
    MambaHNetVocabNetConfig::new(
        HNetVocabShape::new(hnet, VOCAB).with_missing_lm_head(true),
        stage_blocks,
        main_block,
    )
}

fn default_path(block: &MambaBlockConfig) -> MambaSsdPath {
    match block {
        #[cfg(feature = "mamba1")]
        MambaBlockConfig::Mamba1(_) => MambaSsdPath::Mamba1,
        #[cfg(feature = "mamba2")]
        MambaBlockConfig::Mamba2(_) => MambaSsdPath::mamba2_default(),
        #[cfg(feature = "mamba3")]
        MambaBlockConfig::Mamba3(_) => MambaSsdPath::mamba3_default(),
    }
}

fn paths(config: &MambaHNetVocabNetConfig) -> (MambaSsdPath, MambaSsdPath) {
    let stage = config.stage_blocks.first().unwrap_or(&config.main_block);
    (default_path(stage), default_path(&config.main_block))
}

fn tokens(batch: usize, sequence: usize, device: &Device) -> Tensor<3, Int> {
    let ids = Tensor::<2, Int>::random([batch, sequence], Distribution::Uniform(0.0, VOCAB as f64), device);
    let flags = Tensor::<2, Int>::random([batch, sequence], Distribution::Uniform(0.0, 2.0), device);
    Tensor::stack(vec![ids, flags], 2)
}

/// The largest relative difference over every tensor pair of two caches.
fn cache_diff<C: CacheTensors>(a: C, b: C) -> f32 {
    struct MaxDiff(f32);
    impl TensorZip for MaxDiff {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            self.0 = self.0.max(max_rel_diff(a.clone(), b));
            a
        }
    }
    let mut z = MaxDiff(0.0);
    a.zip_tensors(b, &mut z);
    z.0
}

fn step_unrolled(
    net: &MambaHNetVocabNet,
    x: Tensor<3, Int>,
    mode: StepMode,
) -> (Tensor<3>, MambaHNetCaches) {
    let [_batch, sequence, _two] = x.dims();
    let mut caches = None;
    let mut ys = Vec::new();
    for t in 0..sequence {
        let (y, c, _) = net.step(x.clone().narrow(1, t, 1).squeeze_dim(1), caches, mode);
        caches = Some(c);
        ys.push(y.unsqueeze_dim(1));
    }
    (Tensor::cat(ys, 1), caches.unwrap())
}

/// Mamba-2 stages around a Mamba-3 main network (wider than the stage):
/// `forward` equals `step` unrolled on the logits and the caches, in both
/// step modes.
#[cfg(all(feature = "mamba2", feature = "mamba3"))]
#[test]
fn mamba2_stages_mamba3_main_forward_equals_step() {
    let device = test_device();
    let config = config(vec![mamba2(8), mamba2(8)], mamba3(16));
    let net = config.init(&device);
    assert_eq!(net.n_stages(), 2);
    let x = tokens(2, 11, &device);
    let (y_fwd, c_fwd, _) = net.forward(x.clone(), None, paths(&config), None);
    for mode in [StepMode::Masked, StepMode::Gathered] {
        let (y_step, c_step) = step_unrolled(&net, x.clone(), mode);
        assert!(max_rel_diff(y_fwd.clone(), y_step) < dtype_tol(TOL), "{mode:?}: logits");
        assert!(cache_diff(c_fwd.clone(), c_step) < dtype_tol(TOL), "{mode:?}: caches");
    }
}

/// The gradients of `forward` equal those of `step` unrolled, over every
/// parameter of a Mamba-3 H-Net.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_forward_and_step_have_the_same_gradients() {
    use burn::module::{ModuleVisitor, Param};
    let device = test_device().autodiff();
    let config = config(vec![mamba3(8)], mamba3(16));
    let net = config.init(&device);
    let x = tokens(2, 9, &device);
    let w = Tensor::<3>::random([2, 9, VOCAB + 1], Distribution::Normal(0.0, 1.0), &device);
    let (y_fwd, _, _) = net.forward(x.clone(), None, paths(&config), None);
    let g_fwd = (y_fwd * w.clone()).sum().backward();
    let (y_step, _) = step_unrolled(&net, x, StepMode::Masked);
    let g_step = (y_step * w).sum().backward();

    struct Compare<'a> {
        a: &'a burn::tensor::Gradients,
        b: &'a burn::tensor::Gradients,
        worst: f32,
        seen: usize,
    }
    impl ModuleVisitor for Compare<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let val = param.val();
            match (val.grad(self.a), val.grad(self.b)) {
                (Some(a), Some(b)) => {
                    self.worst = self.worst.max(max_rel_diff(a, b));
                    self.seen += 1;
                }
                (None, None) => {}
                (a, b) => panic!("a gradient in one run only ({} vs {})", a.is_some(), b.is_some()),
            }
        }
    }
    let mut cmp = Compare { a: &g_fwd, b: &g_step, worst: 0.0, seen: 0 };
    net.visit(&mut cmp);
    assert!(cmp.seen > 20, "only {} parameters got a gradient", cmp.seen);
    assert!(cmp.worst < dtype_tol(1e-3), "max rel gradient diff {}", cmp.worst);
}

/// A right-padded batch: each slot gives what it gives alone (the logits of
/// its real rows, and its caches). A slot with no real row (a later window of
/// a short story) leaves the other slots alone, and its caches stay finite.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_padded_slots_match_each_slot_alone() {
    let device = test_device();
    let config = config(vec![mamba3(8)], mamba3(16));
    let net = config.init(&device);
    let lens = [10usize, 6, 0];
    let x = tokens(3, 10, &device);
    let pad = Tensor::<1, Int>::arange(0..10, &device)
        .reshape([1, 10])
        .expand([3, 10])
        .greater_equal(Tensor::<2, Int>::from_data([[10], [6], [0]], &device).expand([3, 10]));
    let (y, caches, _) = net.forward(x.clone(), None, paths(&config), Some(pad));
    let caches = caches.into_owned_buffers();
    struct Finite(bool);
    impl TensorZip for Finite {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            self.0 &= !a.clone().is_nan().any().into_scalar::<bool>() && !b.is_inf().any().into_scalar::<bool>();
            a
        }
    }
    let mut finite = Finite(true);
    caches.clone().zip_tensors(caches.clone(), &mut finite);
    assert!(finite.0, "a cache holds a NaN or an Inf");
    for (b, &len) in lens.iter().enumerate().filter(|(_, len)| **len > 0) {
        let (y_b, c_b, _) = net.forward(x.clone().narrow(0, b, 1).narrow(1, 0, len), None, paths(&config), None);
        let y_real = y.clone().narrow(0, b, 1).narrow(1, 0, len);
        assert!(max_rel_diff(y_real, y_b) < dtype_tol(TOL), "slot {b}: logits");
        // Row `b` of the batched caches against the slot run alone.
        struct Row(usize, f32);
        impl TensorZip for Row {
            fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
                self.1 = self.1.max(max_rel_diff(a.clone().narrow(0, self.0, 1), b));
                a
            }
        }
        let mut row = Row(b, 0.0);
        caches.clone().zip_tensors(c_b, &mut row);
        assert!(row.1 < dtype_tol(TOL), "slot {b}: caches differ by {}", row.1);
    }
}

/// The scoped Muon plan fits the real weights of every stack (a Muon step
/// panics on a column layout of the wrong width), and it gives the fused
/// `in_proj` of each stack to Muon.
#[cfg(all(feature = "mamba2", feature = "mamba3", feature = "optim"))]
#[test]
fn the_muon_plan_fits_every_stack() {
    use burn::optim::GradientsParams;
    use burn_stack::optim::{FallbackConfig, muon_config};
    let device = test_device().autodiff();
    let config = config(vec![mamba2(8)], mamba3(16));
    let net = config.init(&device);
    let plan = config.muon_plan();
    let report = plan.describe(&net);
    for path in ["stages.0.encoder.real_layers.0.block", "main.layers.real_layers.0.block"] {
        let line = report
            .lines()
            .find(|l| l.contains(path) && l.contains("in_proj.weight"))
            .unwrap_or_else(|| panic!("no in_proj under {path} in\n{report}"));
        assert!(line.contains("muon["), "{line}");
    }
    let mut optim = plan.build(&FallbackConfig::AdamW(burn::optim::AdamWConfig::new()), &muon_config(0.0));
    let (y, _, routing) = net.forward(tokens(2, 8, &device), None, paths(&config), None);
    let loss = y.sum() + routing[0].ratio_loss(3.0).sum();
    let grads = GradientsParams::from_grads(loss.backward(), &net);
    let _net = optim.step(1e-3, net, grads);
}
