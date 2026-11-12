//! Tests of what only the enum [`MambaBlock`] can break:
//!
//! - the derives, the records and the path segment of a variant,
//! - exact outputs and the flat caches,
//! - the mismatch panics and the mixed-stack rule.
//!
//! The other tests of this directory cover the containers.

// With one family enabled, the enum has one variant, and each `let .. else`
// here is irrefutable.
#![allow(irrefutable_let_patterns)]

use crate::prelude::*;
use burn::module::{ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::{Distribution, Gradients};
use burn_stack::modules::TensorZip;
use burn_stack::utils::ClassCursors;
use burn_stack::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff, test_device};
#[cfg(all(feature = "mamba2", feature = "mamba3"))]
use std::panic::{AssertUnwindSafe, catch_unwind};

const D_MODEL: usize = 16;
const TOL: f32 = 1e-4;

#[cfg(feature = "mamba1")]
fn m1() -> Mamba1Config {
    Mamba1Config::new(D_MODEL).with_state_rank(8)
}

#[cfg(feature = "mamba2")]
fn m2_of_width(d_model: usize) -> Mamba2Config {
    Mamba2Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4)
}

#[cfg(feature = "mamba2")]
fn m2() -> Mamba2Config {
    m2_of_width(D_MODEL)
}

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

struct GradCount<'a> {
    grads: &'a Gradients,
    some: usize,
    all: usize,
}

impl ModuleVisitor for GradCount<'_> {
    fn visit_float<const N: usize>(&mut self, param: &Param<Tensor<N>>) {
        self.all += 1;
        if param.val().grad(self.grads).is_some() {
            self.some += 1;
        }
    }
}

fn grad_count(module: &impl Module, grads: &Gradients) -> (usize, usize) {
    let mut visitor = GradCount { grads, some: 0, all: 0 };
    module.visit(&mut visitor);
    (visitor.some, visitor.all)
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

/// A record survives a round trip in memory and through a file. A fresh network
/// differs until it loads.
#[test]
fn record_round_trip() {
    let device = test_device();
    let dir = std::env::temp_dir().join(format!("burn-mamba-record-round-trip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (tag, block, path) in families() {
        let shape = latent_shape(2);
        let net: MambaLatentNet = shape.init(block.clone(), &device);
        let x = input(2, 6, &device);
        let (want, _) = net.forward(x.clone(), None, path.clone(), None, None);

        let fresh: MambaLatentNet = shape.init(block.clone(), &device);
        let (other, _) = fresh.forward(x.clone(), None, path.clone(), None, None);
        assert!(max_abs_diff(want.clone(), other) > 0.0, "{tag}");
        let loaded = fresh.load_record(net.clone().into_record());
        let (got, _) = loaded.forward(x.clone(), None, path.clone(), None, None);
        assert_eq!(max_abs_diff(want.clone(), got), 0.0, "{tag}: in memory");

        let file = dir.join(tag);
        net.save_file(file.clone()).unwrap();
        let fresh: MambaLatentNet = shape.init(block, &device);
        let loaded = fresh.load_file(file);
        let (got, _) = loaded.forward(x, None, path, None, None);
        assert_eq!(max_abs_diff(want, got), 0.0, "{tag}: through a file");
    }
    std::fs::remove_dir_all(&dir).unwrap();
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

#[test]
#[cfg(feature = "autodiff")]
fn cache_stack_and_traversal() {
    struct Count(usize);
    impl TensorZip for Count {
        fn zip<const N: usize>(&mut self, a: Tensor<N>, _b: Tensor<N>) -> Tensor<N> {
            self.0 += 1;
            a
        }
    }
    let device = test_device().autodiff();
    for (tag, block, path) in families() {
        let net: MambaLatentNet = latent_shape(2).init(block, &device);
        let x = input(2, 6, &device);
        let (_, caches) = net.forward(x.clone(), None, path.clone(), None, None);
        assert_eq!(caches.slot_count(), 2, "{tag}");

        // Take the slots, and rebuild the caches from them.
        let rebuilt = MambaCaches::from_slots(caches.clone().into_slots());
        let mut count = Count(0);
        let same = rebuilt.zip_tensors(caches.clone(), &mut count);
        assert!(count.0 >= 2, "{tag}: the traversal visits the tensors");

        // All-`None` slots rebuild with no guess.
        let empty = MambaCaches::from_slots(vec![None, None]);
        assert_eq!(empty.slot_count(), 2);

        // Move the caches to the inner backend and return them. Then run the
        // two captured-step operations.
        let detached = same.detach();
        let owned = detached.clone().into_owned_buffers();
        let _assigned = owned.assign_in_place(detached.clone());
        let (_, again) = net.forward(x, Some(detached), path, None, None);
        assert_eq!(again.slot_count(), 2, "{tag}: a forward continues from detached caches");

        // The stack builds the zero cache of each virtual layer through the
        // block. A step from these caches checks the family tag of each one.
        // The batch (3) differs from the count of virtual layers (2).
        let zeros = net.layers.zero_caches(3, &device);
        assert_eq!(zeros.slot_count(), 2, "{tag}");
        // A missing cache selects the single-SSD pathway.
        #[cfg(feature = "mamba3")]
        if let Some(Some(MambaCache::Mamba3(zero))) = zeros.slots.first() {
            assert!(matches!(zero, Mamba3Cache::SingleSsd(_)), "{tag}: a zero cache is single-SSD");
        }
        let (_, c) = net.step(Tensor::zeros([3, 3], &device), Some(zeros), None);
        assert_eq!(c.slot_count(), 2, "{tag}");
    }
}

/// A captured step runs over the flat caches. On flex it replays eagerly, so
/// this checks the bounds and the write-back.
#[test]
#[cfg(feature = "mamba3")]
fn captured_step_over_flat_caches() {
    use burn_stack::utils::capture::CapturedStep;

    let device = test_device();
    let net: MambaLatentNet = latent_shape(2).init(MambaBlockConfig::Mamba3(m3()), &device);
    let xs: Vec<Tensor<2>> = (0..4)
        .map(|_| Tensor::random([2, 3], Distribution::Normal(0.0, 1.0), &device))
        .collect();
    let mut eager = Vec::new();
    let mut caches = None;
    for x in &xs {
        let (y, c) = net.step(x.clone(), caches, None);
        eager.push(y);
        caches = Some(c);
    }
    let (y0, caches) = net.step(xs[0].clone(), None, None);
    assert_eq!(max_abs_diff(y0, eager[0].clone()), 0.0);
    // Safety: the step reads only its arguments and `net`, by reference.
    let mut captured =
        unsafe { CapturedStep::capture(&device, xs[1].clone(), caches, |x, c| net.step(x, Some(c), None)) };
    for k in 1..4 {
        let y = captured.step(xs[k].clone()).clone();
        let d = max_abs_diff(y, eager[k].clone());
        assert_eq!(d, 0.0, "step {k} differs by {d}");
    }
}

// ---------------------------------------------------------------------------
// The knobs of the stack
// ---------------------------------------------------------------------------

/// The enum runs with these knobs of the stack: virtual layers, a gradient
/// horizon (the inner-backend hop through the caches and `Module::valid`),
/// multi-gate residuals and class latents.
#[test]
#[cfg(feature = "autodiff")]
fn stack_knobs() {
    let device = test_device().autodiff();
    for (tag, block, path) in families() {
        let stack = NetworkShape::new(2)
            .with_n_virtual_layers(Some((4, Schedule::Cyclic)))
            .with_grad_horizon(Some(GradHorizon::last(2, 4)))
            .with_class_latents(vec![ClassLatent::Start; 1])
            .with_residuals(ResidualsConfig::MultiGate {
                n_stream: 2,
                init_bias: 0.0,
                init_bias_offset: 0.0,
                per_virtual_layer: false,
            });
        let net: MambaLatentNet = LatentShape::new(3, 5, stack).init(block, &device);
        let (y, caches) = net.forward(input(2, 6, &device), None, path, None, None);
        assert_eq!(caches.slot_count(), 4, "{tag}");
        let grads = y.sum().backward();
        let (some, all) = grad_count(&net.layers.real_layers[1].block, &grads);
        assert!(some > 0 && all > 0, "{tag}: {some} of {all}");
    }
}

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

/// `Layer::application` narrows the untied copies through the enum, in
/// `forward` and in `step`.
#[test]
#[cfg(feature = "mamba3")]
fn untied_through_the_enum() {
    let device = test_device().autodiff();
    let stack = NetworkShape::new(2)
        .with_n_virtual_layers(Some((5, Schedule::Cyclic)))
        .with_untied(vec![LayerUntied::Norm]);
    let untied = || m3().with_untied(vec![Mamba3Untied::DtBias, Mamba3Untied::D]);
    let net: MambaLatentNet =
        LatentShape::new(3, 5, stack.clone()).init(MambaBlockConfig::Mamba3(untied()), &device);
    let MambaBlock::Mamba3(_) = &net.layers.real_layers[0].block else { unreachable!() };
    assert!(!Block::untied_params(&net.layers.real_layers[0].block).is_empty());

    let tagged_dims: Vec<_> = param_shapes(&net).into_iter().map(|(_, dims)| dims).collect();
    let family_dims: Vec<_> = param_shapes(&LatentShape::new(3, 5, stack).init(untied(), &device))
        .into_iter()
        .map(|(_, dims)| dims)
        .collect();
    assert_eq!(tagged_dims, family_dims, "the same tiled shapes as the family type");

    let net = net.map(&mut Jitter);
    let x = input(2, 4, &device);
    let (whole, _) = net.forward(x.clone(), None, MambaSsdPath::mamba3_default(), None, None);
    let mut caches = None;
    for t in 0..4 {
        let (y, c) = net.step(x.clone().narrow(1, t, 1).squeeze_dim(1), caches, None);
        caches = Some(c);
        let d = max_rel_diff(y, whole.clone().narrow(1, t, 1).squeeze_dim(1));
        assert!(d < dtype_tol(TOL), "position {t} differs by {d}");
    }
}

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
// The optimizer
// ---------------------------------------------------------------------------

/// The Muon plan selects the same weights through the variant segment.
#[test]
#[cfg(all(feature = "mamba3", feature = "optim"))]
fn muon_plan_matches_the_family_type() {
    use burn::optim::{AdamWConfig, GradientsParams};
    use burn_stack::optim::FallbackConfig;

    let device = test_device().autodiff();
    let shape = latent_shape(2);
    let config = MambaLatentNetConfig::new(shape.clone(), MambaBlockConfig::Mamba3(m3()));
    let net = config.init(&device);
    let plan = config.muon_plan();
    let report = plan.describe(&net);
    let line = report
        .lines()
        .find(|l| l.contains("block.Mamba3") && l.contains("in_proj.weight"))
        .unwrap_or_else(|| panic!("no in_proj under the variant in\n{report}"));
    assert!(line.contains("muon["), "{line}");

    let f = shape.init(m3(), &device);
    let f_report = shape.stack.muon_plan(&m3()).describe(&f);
    assert_eq!(report.matches("muon[").count(), f_report.matches("muon[").count());

    let mut optim = plan.build(&FallbackConfig::AdamW(AdamWConfig::new()), &muon_config(0.0));
    let (y, _) = net.forward(input(2, 6, &device), None, MambaSsdPath::mamba3_default(), None, None);
    let grads = GradientsParams::from_grads(y.sum().backward(), &net);
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
