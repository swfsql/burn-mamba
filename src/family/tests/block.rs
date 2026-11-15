//! Tests of what only the enum [`MambaBlock`] can break:
//!
//! - the derives, the records and the path segment of a variant,
//! - exact outputs, and the class markers through the enum,
//! - the empty slots of the flat caches, and the zero cache of Mamba-3,
//! - an H-Net with stages of one family and a main stack of another,
//! - the mismatch panics and the mixed-stack rule.
//!
//! `block_contract.rs` runs the block contracts on the enum.

// With one family enabled, the enum has one variant, and each `let .. else`
// here is irrefutable.
#![allow(irrefutable_let_patterns)]

use super::D_MODEL;
#[cfg(feature = "mamba1")]
use super::m1;
#[cfg(feature = "mamba2")]
use super::m2;
#[cfg(all(feature = "mamba2", feature = "mamba3", feature = "optim"))]
use super::m2_of_width;
#[cfg(feature = "mamba3")]
use super::m3;
use crate::prelude::*;
use burn::module::{ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn_stack::modules::TensorZip;
use burn_stack::utils::ClassCursors;
use burn_stack::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff, test_device};
#[cfg(all(feature = "mamba2", feature = "mamba3"))]
use std::panic::{AssertUnwindSafe, catch_unwind};

const TOL: f32 = 1e-4;

/// Each enabled family: its tag, its block config and its SSD path.
fn families() -> Vec<(&'static str, MambaBlockConfig, MambaSsdPath)> {
    vec![
        #[cfg(feature = "mamba1")]
        ("Mamba1", MambaBlockConfig::Mamba1(m1()), MambaSsdPath::Mamba1),
        #[cfg(feature = "mamba2")]
        ("Mamba2", MambaBlockConfig::Mamba2(m2()), MambaSsdPath::mamba2_default()),
        #[cfg(feature = "mamba3")]
        ("Mamba3", MambaBlockConfig::Mamba3(m3()), MambaSsdPath::mamba3_default()),
    ]
}

fn latent_shape(n_real: usize) -> LatentShape {
    LatentShape::new(3, 5, NetworkShape::new(n_real))
}

fn input(batch: usize, seq: usize, device: &Device) -> Tensor<3> {
    Tensor::random([batch, seq, 3], Distribution::Normal(0.0, 1.0), device)
}

/// The path and the shape of each parameter.
#[derive(Default)]
struct ParamShapes {
    path: Vec<String>,
    shapes: Vec<(String, Vec<usize>)>,
}

impl ModuleVisitor for ParamShapes {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const N: usize>(&mut self, param: &Param<Tensor<N>>) {
        self.shapes.push((self.path.join("."), param.val().shape().dims::<N>().to_vec()));
    }
}

fn param_shapes(module: &impl Module) -> Vec<(String, Vec<usize>)> {
    let mut visitor = ParamShapes::default();
    module.visit(&mut visitor);
    visitor.shapes
}

struct Jitter;

impl ModuleMapper for Jitter {
    fn map_float<const N: usize>(&mut self, param: Param<Tensor<N>>) -> Param<Tensor<N>> {
        param.map(|t| {
            let noise = Tensor::random(t.shape(), Distribution::Normal(0.0, 0.05), &t.device());
            t + noise
        })
    }
}

/// The largest relative difference over each tensor pair of two caches.
fn cache_diff<C: CacheTensors>(a: C, b: C) -> f32 {
    struct MaxDiff(f32);
    impl TensorZip for MaxDiff {
        fn zip<const N: usize>(&mut self, a: Tensor<N>, b: Tensor<N>) -> Tensor<N> {
            self.0 = self.0.max(max_rel_diff(a.clone(), b));
            a
        }
    }
    let mut z = MaxDiff(0.0);
    a.zip_tensors(b, &mut z);
    z.0
}

// ---------------------------------------------------------------------------
// The derives
// ---------------------------------------------------------------------------

/// The `Module` surface of a network over the enum, for each family.
#[test]
fn module_surface() {
    let device = test_device();
    for (tag, block, path) in families() {
        let net: MambaLatentNet = latent_shape(2).init(block, &device);
        assert!(net.num_params() > 0, "{tag}");
        assert!(!format!("{net}").is_empty(), "{tag}: Display");
        assert!(!format!("{net:?}").is_empty(), "{tag}: Debug");
        assert!(!net.devices().is_empty(), "{tag}: devices");
        let moved = net.clone().fork(&device).to_device(&device).no_grad();
        let x = input(2, 6, &device);
        let (a, _) = net.forward(x.clone(), None, path.clone(), None, None);
        let (b, _) = moved.forward(x.clone(), None, path.clone(), None, None);
        assert_eq!(max_abs_diff(a.clone(), b), 0.0, "{tag}: clone, fork, to_device");
        let jittered = net.clone().map(&mut Jitter);
        let (c, _) = jittered.forward(x, None, path, None, None);
        assert!(max_abs_diff(a, c) > 0.0, "{tag}: a mapper reaches the block");
    }
}

/// A variant adds one path segment, and nothing else. A family-typed network
/// has the same parameters.
fn assert_only_the_variant_segment_differs<C: BlockConfig>(
    family: C,
    tagged: MambaBlockConfig,
    variant: &str,
) {
    let device = test_device();
    let shape = latent_shape(2);
    let tagged_shapes = param_shapes(&shape.init(tagged, &device));
    let family_shapes = param_shapes(&shape.init(family, &device));
    assert_eq!(tagged_shapes.len(), family_shapes.len(), "{variant}");
    let segment = format!(".{variant}");
    for ((tagged_path, tagged_dims), (family_path, family_dims)) in tagged_shapes.iter().zip(&family_shapes) {
        assert_eq!(&tagged_path.replace(&segment, ""), family_path);
        assert_eq!(tagged_dims, family_dims, "{tagged_path}");
    }
    let under_block = format!("block.{variant}.");
    assert!(tagged_shapes.iter().any(|(path, _)| path.contains(&under_block)), "{tagged_shapes:?}");
}

#[test]
fn param_paths_add_only_the_variant() {
    #[cfg(feature = "mamba1")]
    assert_only_the_variant_segment_differs(m1(), MambaBlockConfig::Mamba1(m1()), "Mamba1");
    #[cfg(feature = "mamba2")]
    assert_only_the_variant_segment_differs(m2(), MambaBlockConfig::Mamba2(m2()), "Mamba2");
    #[cfg(feature = "mamba3")]
    assert_only_the_variant_segment_differs(m3(), MambaBlockConfig::Mamba3(m3()), "Mamba3");
}

/// A record of one family does not load into a block of another.
#[test]
#[cfg(all(feature = "mamba2", feature = "mamba3"))]
fn record_of_another_family_panics() {
    let device = test_device();
    let shape = latent_shape(1);
    let a: MambaLatentNet = shape.init(MambaBlockConfig::Mamba2(m2()), &device);
    let b: MambaLatentNet = shape.init(MambaBlockConfig::Mamba3(m3()), &device);
    let record = a.into_record();
    let result = catch_unwind(AssertUnwindSafe(move || b.load_record(record)));
    assert!(result.is_err(), "a record of another family must not load");
}

// ---------------------------------------------------------------------------
// The enum changes no math
// ---------------------------------------------------------------------------

/// The enum and the family block give the same bytes.
#[test]
fn block_output_is_exact() {
    let device = test_device();
    let x = Tensor::<3>::random([2, 6, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    #[cfg(feature = "mamba1")]
    {
        let block = BlockConfig::init_block(&m1(), 1, &device);
        let tagged = MambaBlock::Mamba1(block.clone());
        let (want, _) = block.block_forward(x.clone(), None, (), None);
        let (got, _) = tagged.block_forward(x.clone(), None, MambaSsdPath::Mamba1, None);
        assert_eq!(max_abs_diff(want, got), 0.0);
    }
    #[cfg(feature = "mamba2")]
    {
        let block = BlockConfig::init_block(&m2(), 1, &device);
        let tagged = MambaBlock::Mamba2(block.clone());
        let (want, _) = block.block_forward(x.clone(), None, Mamba2SsdPath::default(), None);
        let (got, _) = tagged.block_forward(x.clone(), None, MambaSsdPath::mamba2_default(), None);
        assert_eq!(max_abs_diff(want, got), 0.0);
    }
    #[cfg(feature = "mamba3")]
    {
        let block = BlockConfig::init_block(&m3(), 1, &device);
        let tagged = MambaBlock::Mamba3(block.clone());
        let (want, _) = block.block_forward(x.clone(), None, Mamba3SsdPath::default(), None);
        let (got, _) = tagged.block_forward(x, None, MambaSsdPath::mamba3_default(), None);
        assert_eq!(max_abs_diff(want, got), 0.0);
    }
}

/// A network over the enum, with the weights of a family-typed network, gives
/// the same bytes and the same gradients.
#[test]
#[cfg(feature = "mamba3")]
fn network_output_and_gradients_are_exact() {
    let device = test_device().autodiff();
    let shape = latent_shape(2);
    let family_net = shape.init(m3(), &device);
    let mut tagged_net: MambaLatentNet = shape.init(MambaBlockConfig::Mamba3(m3()), &device);
    tagged_net.in_proj = family_net.in_proj.clone();
    tagged_net.out_proj = family_net.out_proj.clone();
    for (tagged, family) in tagged_net.layers.real_layers.iter_mut().zip(&family_net.layers.real_layers) {
        tagged.norm = family.norm.clone();
        tagged.block = MambaBlock::Mamba3(family.block.clone());
    }
    let x = input(2, 6, &device);
    let (want, _) = family_net.forward(x.clone(), None, Mamba3SsdPath::default(), None, None);
    let (got, _) = tagged_net.forward(x, None, MambaSsdPath::mamba3_default(), None, None);
    assert_eq!(max_abs_diff(want.clone(), got.clone()), 0.0, "values");
    let grads_want = want.sum().backward();
    let grads_got = got.sum().backward();
    // The parameters are the same tensors, in two graphs.
    let MambaBlock::Mamba3(block) = &tagged_net.layers.real_layers[0].block else { unreachable!() };
    let want = family_net.layers.real_layers[0].block.in_proj.weight.val().grad(&grads_want).unwrap();
    let got = block.in_proj.weight.val().grad(&grads_got).unwrap();
    assert_eq!(max_abs_diff(want, got), 0.0, "gradients");
}

// ---------------------------------------------------------------------------
// The caches
// ---------------------------------------------------------------------------

/// The flat caches hold empty slots. Each slot has its own family tag, so
/// `from_slots` needs no family for a slot of `None`.
#[test]
fn the_flat_caches_keep_empty_slots() {
    let empty = MambaCaches::from_slots(vec![None, None]);
    assert_eq!(empty.slot_count(), 2);
    assert!(empty.into_slots().iter().all(Option::is_none), "a slot of `None` must stay empty");
}

/// The zero cache of a Mamba-3 block in the enum is a single-SSD cache,
/// because a missing cache selects that pathway.
#[test]
#[cfg(feature = "mamba3")]
fn the_zero_cache_of_mamba3_is_single_ssd() {
    let device = test_device();
    let block = MambaBlock::Mamba3(BlockConfig::init_block(&m3(), 1, &device));
    let zero = block.zero_cache(3, &device);
    assert!(
        matches!(zero, MambaCache::Mamba3(Mamba3Cache::SingleSsd(_))),
        "the zero cache of Mamba-3 must be a single-SSD cache",
    );
}

// ---------------------------------------------------------------------------
// The class markers
// ---------------------------------------------------------------------------

/// The enum runs a class latent of the stack and a class token of the network.
/// A `prime` emits both, and each `step` after it gives what `forward` gives,
/// in the outputs and in the final caches.
#[test]
fn prime_and_class_tokens_through_the_enum() {
    let device = test_device();
    for (tag, block, path) in families() {
        let shape = LatentShape::new(3, 5, NetworkShape::new(2).with_class_latents(vec![ClassLatent::Start]))
            .with_class_tokens(vec![ClassToken::Start]);
        let net: MambaLatentNet = shape.init(block, &device);
        let (batch, seq) = (2, 3);
        let x = input(batch, seq, &device);

        // The stack latent is below the class token of the network, so the
        // positions are `S N u0 u1 u2`.
        let (whole, caches_forward) = net.forward(x.clone(), None, path, None, None);
        assert_eq!(whole.dims(), [batch, seq + 2, 5], "{tag}");
        let at = |p: usize| whole.clone().narrow(1, p, 1).squeeze_dim::<2>(1);

        let mut class = ClassCursors::new(seq);
        let (primed, mut caches) = net.prime(batch, None, Some(&mut class));
        let d = max_rel_diff(primed.expect("the prime must emit the two markers"), at(1));
        assert!(d < dtype_tol(TOL), "{tag}: the prime differs by {d}");
        assert_eq!((class.network, class.stack), (1, 2), "{tag}");
        for t in 0..seq {
            let xt = x.clone().narrow(1, t, 1).squeeze_dim::<2>(1);
            let (y, c) = net.step(xt, caches, Some(&mut class));
            caches = Some(c);
            let d = max_rel_diff(y, at(t + 2));
            assert!(d < dtype_tol(TOL), "{tag}: step {t} differs by {d}");
        }
        let d = cache_diff(caches_forward, caches.unwrap());
        assert!(d < dtype_tol(TOL), "{tag}: the final caches differ by {d}");
    }
}

// ---------------------------------------------------------------------------
// The H-Net
// ---------------------------------------------------------------------------

/// One network type holds stages of one family and a main stack of another.
#[test]
#[cfg(all(feature = "mamba2", feature = "mamba3", feature = "optim"))]
fn hnet_mixes_stage_and_main_family() {
    use burn::optim::{AdamWConfig, GradientsParams};
    use burn_stack::modules::hnet::{HNetShape, HNetStageShape, HNetVocabShape, StepMode};
    use burn_stack::optim::FallbackConfig;

    const VOCAB: usize = 11;
    let device = test_device().autodiff();
    let wide = MambaBlockConfig::Mamba3(m3());
    let narrow = MambaBlockConfig::Mamba2(m2_of_width(8));
    let stack = || NetworkShape::new(2);
    let shape = HNetShape::new(vec![HNetStageShape::new(stack(), stack())], stack());
    let config = MambaHNetVocabNetConfig::new(
        HNetVocabShape::new(shape, VOCAB).with_missing_lm_head(true),
        vec![narrow],
        wide,
    );
    let net: MambaHNetVocabNet = config.init(&device);

    let pairs: Vec<i64> = (0..16i64).flat_map(|i| [(i * 7 + 3) % VOCAB as i64, i % 2]).collect();
    let x = Tensor::<1, Int>::from_ints(pairs.as_slice(), &device).reshape([2, 8, 2]);
    let paths = (MambaSsdPath::mamba2_default(), MambaSsdPath::mamba3_default());
    let (whole, _, routing) = net.forward(x.clone(), None, paths, None);

    let mut caches = None;
    for t in 0..8 {
        let (y, c, _) = net.step(x.clone().narrow(1, t, 1).squeeze_dim(1), caches, StepMode::Masked);
        caches = Some(c);
        let d = max_rel_diff(y, whole.clone().narrow(1, t, 1).squeeze_dim(1));
        assert!(d < dtype_tol(TOL), "position {t} differs by {d}");
    }

    let plan = config.muon_plan();
    let report = plan.describe(&net);
    assert!(report.contains("muon["), "{report}");
    let mut optim = plan.build(&FallbackConfig::AdamW(AdamWConfig::new()), &muon_config(0.0));
    let loss = whole.sum() + routing[0].ratio_loss(3.0).sum();
    let grads = GradientsParams::from_grads(loss.backward(), &net);
    let _net = optim.step(1e-3, net, grads);
}

// ---------------------------------------------------------------------------
// The mismatch rules
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "the SSD path family does not match the block")]
#[cfg(all(feature = "mamba2", feature = "mamba3"))]
fn wrong_path_panics() {
    let device = test_device();
    let net: MambaLatentNet = latent_shape(1).init(MambaBlockConfig::Mamba3(m3()), &device);
    let _ = net.forward(input(2, 6, &device), None, MambaSsdPath::mamba2_default(), None, None);
}

#[test]
#[should_panic(expected = "the cache family does not match the block")]
#[cfg(all(feature = "mamba2", feature = "mamba3"))]
fn wrong_cache_panics() {
    let device = test_device();
    let a: MambaLatentNet = latent_shape(1).init(MambaBlockConfig::Mamba2(m2()), &device);
    let b: MambaLatentNet = latent_shape(1).init(MambaBlockConfig::Mamba3(m3()), &device);
    let x = input(2, 6, &device);
    let (_, caches) = a.forward(x.clone(), None, MambaSsdPath::mamba2_default(), None, None);
    let _ = b.forward(x, Some(caches), MambaSsdPath::mamba3_default(), None, None);
}

/// A hand-built mixed stack panics at its first `forward` (one `Options` value)
/// and at a `step` with no cache (the first block sizes each zero cache). Only
/// a `step` from caches that the caller built slot by slot runs.
#[test]
#[cfg(all(feature = "mamba2", feature = "mamba3"))]
fn mixed_stack_rules() {
    let device = test_device();
    let mut net: MambaLatentNet = latent_shape(2).init(MambaBlockConfig::Mamba3(m3()), &device);
    net.layers.real_layers[0].block = MambaBlock::Mamba2(BlockConfig::init_block(&m2(), 1, &device));
    let x = input(2, 6, &device);
    let x0 = x.clone().narrow(1, 0, 1).squeeze_dim(1);

    let forward = catch_unwind(AssertUnwindSafe(|| net.forward(x, None, MambaSsdPath::mamba3_default(), None, None)));
    assert!(forward.is_err(), "a mixed stack must panic at its first forward");
    let step = catch_unwind(AssertUnwindSafe(|| net.step(x0.clone(), None, None)));
    assert!(step.is_err(), "a mixed stack must panic at a step with no cache");

    let slots = net.layers.real_layers.iter().map(|layer| Some(layer.block.zero_cache(2, &device))).collect();
    let (y, _) = net.step(x0, Some(MambaCaches::from_slots(slots)), None);
    assert_eq!(y.dims(), [2, 5]);
}
