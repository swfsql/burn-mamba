//! The stress of the spike: the derives and every container over `MambaBlock`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::{Distribution, Gradients};
use burn_mamba::prelude::MambaSsdPath;
#[cfg(feature = "mamba1")]
use burn_mamba::mamba1::prelude::Mamba1Config;
#[cfg(feature = "mamba2")]
use burn_mamba::mamba2::prelude::{Mamba2Config, Mamba2SsdPath};
#[cfg(feature = "mamba3")]
use burn_mamba::mamba3::prelude::{Mamba3Config, Mamba3SsdPath, Mamba3Untied};
use burn_stack::examples::trainer::Trainer;
use burn_stack::examples::training::OptimizerConfig;
use burn_stack::modules::bidi::OutputMergeConfig;
use burn_stack::modules::hnet::{HNetShape, HNetStageShape, HNetVocabShape, StepMode};
use burn_stack::modules::{
    BidiShape, Block, BlockConfig, CacheStack, CacheTensors, LatentShape, LayerUntied, ModelConfigExt, NetworkShape,
    ResidualsConfig, TensorZip, VocabShape,
};
use burn_stack::optim::{FallbackConfig, muon_config};
use burn_stack::utils::capture::CapturedStep;
use burn_stack::utils::test_helpers::{dtype_tol, max_abs_diff, max_rel_diff, test_device};
use burn_stack::utils::{ClassLatent, GradHorizon, Packed, Schedule};
use spike_block_enum::*;

const D: usize = 16;
const VOCAB: usize = 11;
const TOL: f32 = 1e-4;

#[cfg(feature = "mamba1")]
fn m1() -> Mamba1Config {
    Mamba1Config::new(D).with_state_rank(8)
}
#[cfg(feature = "mamba2")]
fn m2() -> Mamba2Config {
    Mamba2Config::new(D)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_conv_kernel(4)
}
#[cfg(feature = "mamba3")]
fn m3() -> Mamba3Config {
    Mamba3Config::new(D)
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
struct Rows {
    path: Vec<String>,
    rows: Vec<(String, Vec<usize>)>,
}
impl ModuleVisitor for Rows {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const N: usize>(&mut self, param: &Param<Tensor<N>>) {
        self.rows.push((self.path.join("."), param.val().shape().dims::<N>().to_vec()));
    }
}
fn rows(module: &impl Module) -> Vec<(String, Vec<usize>)> {
    let mut v = Rows::default();
    module.visit(&mut v);
    v.rows
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
    let mut v = GradCount { grads, some: 0, all: 0 };
    module.visit(&mut v);
    (v.some, v.all)
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

// ---------------------------------------------------------------------------
// The derives
// ---------------------------------------------------------------------------

/// The `Module` surface of a network over the enum, for each family.
#[test]
fn module_surface() {
    let device = test_device();
    for (tag, block, path) in families() {
        let shape = latent_shape(2);
        let net: MambaLatentNet = shape.init(block, &device);
        assert!(net.num_params() > 0, "{tag}");
        assert!(!format!("{net}").is_empty(), "{tag}: Display");
        assert!(!format!("{net:?}").is_empty(), "{tag}: Debug");
        assert!(!net.devices().is_empty(), "{tag}: devices");
        let copy = net.clone();
        let moved = copy.fork(&device).to_device(&device).no_grad();
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
#[test]
#[cfg(all(feature = "mamba1", feature = "mamba2", feature = "mamba3"))]
fn param_paths_add_only_the_variant() {
    let device = test_device();
    let shape = latent_shape(2);
    let enum_rows = rows(&shape.init(MambaBlockConfig::Mamba3(m3()), &device));
    let fam_rows = rows(&shape.init(m3(), &device));
    assert_eq!(enum_rows.len(), fam_rows.len());
    for (e, f) in enum_rows.iter().zip(&fam_rows) {
        assert_eq!(e.0.replace(".Mamba3", ""), f.0);
        assert_eq!(e.1, f.1, "{}", e.0);
    }
    assert!(enum_rows.iter().any(|(p, _)| p.contains("block.Mamba3.in_proj")), "{enum_rows:?}");

    let enum_rows = rows(&shape.init(MambaBlockConfig::Mamba2(m2()), &device));
    assert_eq!(enum_rows.len(), rows(&shape.init(m2(), &device)).len());
    let enum_rows = rows(&shape.init(MambaBlockConfig::Mamba1(m1()), &device));
    assert_eq!(enum_rows.len(), rows(&shape.init(m1(), &device)).len());
}

/// In memory and through a file. A fresh network differs until it loads.
#[test]
fn record_round_trip() {
    let device = test_device();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("records");
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
        net.clone().save_file(file.clone()).unwrap();
        let fresh: MambaLatentNet = shape.init(block, &device);
        let loaded = fresh.load_file(file);
        let (got, _) = loaded.forward(x, None, path, None, None);
        assert_eq!(max_abs_diff(want, got), 0.0, "{tag}: through a file");
    }
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

/// The clean break of Q2 fails with an error. A record of today's enum (the tag
/// at the root) does not load into the new form, in either direction, and it
/// does not leave random weights behind in silence.
#[test]
#[cfg(feature = "mamba3")]
fn a_record_of_the_current_form_fails_with_an_error() {
    let device = test_device();
    let current = burn_mamba::prelude::MambaLatentNetConfig::Mamba3 {
        input_size: 3,
        output_size: 5,
        final_norm: false,
        n_real_layers: 2,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: m3(),
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: ResidualsConfig::Standard,
        mlp: None,
        untied: Vec::new(),
    }
    .init(&device);
    let fresh = || -> MambaLatentNet { latent_shape(2).init(MambaBlockConfig::Mamba3(m3()), &device) };

    let old = current.clone().into_record();
    let result = fresh().try_load_record(old);
    assert!(result.is_err(), "a record of the current form must not load into the new form");
    println!("old into new: {}", result.err().unwrap());

    let new = fresh().into_record();
    let result = current.try_load_record(new);
    assert!(result.is_err(), "a record of the new form must not load into the current form");
    println!("new into old: {}", result.err().unwrap());
}

/// `{ shape, block }` through JSON, for each container and each family.
#[test]
fn configs_round_trip_and_drive() {
    fn drive<C: ModelConfigExt>(config: &C, device: &Device) -> usize {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("records");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("config.json");
        config.save(&file).unwrap();
        let loaded = C::load(&file).unwrap();
        let model = loaded.init(device);
        let report = loaded.muon_plan().describe(&model);
        assert!(report.lines().count() > 1);
        model.num_params()
    }
    let device = test_device();
    for (tag, block, _) in families() {
        let n = drive(&MambaLatentNetConfig::new(latent_shape(2), block.clone()), &device);
        assert!(n > 0, "{tag}");
        let vocab = VocabShape::new(VOCAB, NetworkShape::new(2));
        let n = drive(&MambaVocabNetConfig::new(vocab, block.clone()), &device);
        assert!(n > 0, "{tag}");
        let bidi = BidiShape::new(2, OutputMergeConfig::cat_linear(2));
        let n = drive(&MambaBidiLayersConfig::new(bidi, block.clone()), &device);
        assert!(n > 0, "{tag}");
    }
}

// ---------------------------------------------------------------------------
// No math moves
// ---------------------------------------------------------------------------

/// The enum and the family block give the same bytes.
#[test]
fn block_output_is_exact() {
    let device = test_device();
    let x = Tensor::<3>::random([2, 6, D], Distribution::Normal(0.0, 1.0), &device);
    #[cfg(feature = "mamba2")]
    {
        let b = BlockConfig::init_block(&m2(), 1, &device);
        let e = MambaBlock::Mamba2(b.clone());
        let (a, _) = b.block_forward(x.clone(), None, Mamba2SsdPath::default(), None);
        let (c, _) = e.block_forward(x.clone(), None, MambaSsdPath::Mamba2(Mamba2SsdPath::default()), None);
        assert_eq!(max_abs_diff(a, c), 0.0);
    }
    #[cfg(feature = "mamba3")]
    {
        let b = BlockConfig::init_block(&m3(), 1, &device);
        let e = MambaBlock::Mamba3(b.clone());
        let (a, _) = b.block_forward(x.clone(), None, Mamba3SsdPath::default(), None);
        let (c, _) = e.block_forward(x.clone(), None, MambaSsdPath::Mamba3(Mamba3SsdPath::default()), None);
        assert_eq!(max_abs_diff(a, c), 0.0);
    }
    #[cfg(feature = "mamba1")]
    {
        let b = BlockConfig::init_block(&m1(), 1, &device);
        let e = MambaBlock::Mamba1(b.clone());
        let (a, _) = b.block_forward(x.clone(), None, (), None);
        let (c, _) = e.block_forward(x, None, MambaSsdPath::Mamba1, None);
        assert_eq!(max_abs_diff(a, c), 0.0);
    }
}

/// A network over the enum with the weights of a family-typed network gives
/// the same bytes and the same gradients.
#[test]
#[cfg(feature = "mamba3")]
fn network_output_and_gradients_are_exact() {
    let device = test_device().autodiff();
    let shape = latent_shape(2);
    let f = shape.init(m3(), &device);
    let mut e: MambaLatentNet = shape.init(MambaBlockConfig::Mamba3(m3()), &device);
    e.in_proj = f.in_proj.clone();
    e.out_proj = f.out_proj.clone();
    for (el, fl) in e.layers.real_layers.iter_mut().zip(&f.layers.real_layers) {
        el.norm = fl.norm.clone();
        el.block = MambaBlock::Mamba3(fl.block.clone());
    }
    let x = input(2, 6, &device);
    let (a, _) = f.forward(x.clone(), None, Mamba3SsdPath::default(), None, None);
    let (b, _) = e.forward(x, None, MambaSsdPath::mamba3_default(), None, None);
    assert_eq!(max_abs_diff(a.clone(), b.clone()), 0.0, "values");
    let ga = a.sum().backward();
    let gb = b.sum().backward();
    // The parameters are the same tensors, in two graphs.
    let MambaBlock::Mamba3(block) = &e.layers.real_layers[0].block else { unreachable!() };
    let want = f.layers.real_layers[0].block.in_proj.weight.val().grad(&ga).unwrap();
    let got = block.in_proj.weight.val().grad(&gb).unwrap();
    assert_eq!(max_abs_diff(want, got), 0.0, "gradients");
}

/// `forward` = `forward` + `step` unrolled, through the flat caches. Every
/// parameter of the block gets a gradient.
#[test]
fn forward_step_parity_and_gradients() {
    let device = test_device().autodiff();
    for (tag, block, path) in families() {
        let net: MambaLatentNet = latent_shape(2).init(block, &device);
        let n = 8;
        let k = 5;
        let x = input(2, n, &device);
        let (whole, _) = net.forward(x.clone(), None, path.clone(), None, None);
        let (_, caches) = net.forward(x.clone().narrow(1, 0, k), None, path.clone(), None, None);
        let mut caches = Some(caches);
        for t in k..n {
            let xt = x.clone().narrow(1, t, 1).squeeze_dim(1);
            let (y, c) = net.step(xt, caches, None);
            caches = Some(c);
            let want = whole.clone().narrow(1, t, 1).squeeze_dim(1);
            let d = max_rel_diff(y, want);
            assert!(d < dtype_tol(TOL), "{tag}: position {t} differs by {d}");
        }
        let grads = whole.sum().backward();
        let (some, all) = grad_count(&net.layers.real_layers[0].block, &grads);
        assert!(some > 0 && all > 0, "{tag}: {some} of {all} block parameters have a gradient");
    }
}

// ---------------------------------------------------------------------------
// The caches
// ---------------------------------------------------------------------------

#[test]
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

        // Slots out and back.
        let rebuilt = MambaCaches::from_slots(caches.clone().into_slots());
        let mut count = Count(0);
        let same = rebuilt.zip_tensors(caches.clone(), &mut count);
        assert!(count.0 >= 2, "{tag}: the traversal visits the tensors");

        // All-`None` slots rebuild with no guess.
        let empty = MambaCaches::from_slots(vec![None, None]);
        assert_eq!(empty.slot_count(), 2);

        // Inner backend and back, and the two captured-step operations.
        let detached = same.detach();
        let owned = detached.clone().into_owned_buffers();
        let _assigned = owned.assign_in_place(detached.clone());
        let (_, again) = net.forward(x, Some(detached), path, None, None);
        assert_eq!(again.slot_count(), 2, "{tag}: a forward continues from detached caches");

        // Zero caches from the block.
        let zeros = net.layers.real_layers[0].block.zero_caches_3d(&Tensor::zeros([2, 6, D], &device), 3);
        assert_eq!(zeros.slot_count(), 3, "{tag}");
        let zeros = net.layers.real_layers[0].block.zero_caches_2d(&Tensor::zeros([2, D], &device), 2);
        let (_, c) = net.step(Tensor::zeros([2, 3], &device), Some(zeros), None);
        assert_eq!(c.slot_count(), 2, "{tag}");
    }
}

// ---------------------------------------------------------------------------
// The containers and their knobs
// ---------------------------------------------------------------------------

/// Virtual layers, a gradient horizon (the inner-backend hop through the
/// caches and `Module::valid`), multi-gate residuals and class latents.
#[test]
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
        let x = input(2, 6, &device);
        let (y, caches) = net.forward(x, None, path, None, None);
        assert_eq!(caches.slot_count(), 4, "{tag}");
        let grads = y.sum().backward();
        let (some, all) = grad_count(&net.layers.real_layers[1].block, &grads);
        assert!(some > 0 && all > 0, "{tag}: {some} of {all}");
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
    let block = MambaBlockConfig::Mamba3(m3().with_untied(vec![Mamba3Untied::DtBias, Mamba3Untied::D]));
    let net: MambaLatentNet = LatentShape::new(3, 5, stack.clone()).init(block, &device);
    let MambaBlock::Mamba3(_) = &net.layers.real_layers[0].block else { unreachable!() };
    assert!(!Block::untied_params(&net.layers.real_layers[0].block).is_empty());

    let e_rows = rows(&net);
    let f_rows = rows(&LatentShape::new(3, 5, stack).init(
        m3().with_untied(vec![Mamba3Untied::DtBias, Mamba3Untied::D]),
        &device,
    ));
    let e_dims: Vec<_> = e_rows.iter().map(|r| r.1.clone()).collect();
    let f_dims: Vec<_> = f_rows.iter().map(|r| r.1.clone()).collect();
    assert_eq!(e_dims, f_dims, "the same tiled shapes as the family type");

    let net = net.map(&mut Jitter);
    let path = MambaSsdPath::mamba3_default();
    let x = input(2, 4, &device);
    let (whole, _) = net.forward(x.clone(), None, path, None, None);
    let mut caches = None;
    for t in 0..4 {
        let (y, c) = net.step(x.clone().narrow(1, t, 1).squeeze_dim(1), caches, None);
        caches = Some(c);
        let d = max_rel_diff(y, whole.clone().narrow(1, t, 1).squeeze_dim(1));
        assert!(d < dtype_tol(TOL), "position {t} differs by {d}");
    }
}

fn vocab_tokens(batch: usize, seq: usize, device: &Device) -> Tensor<3, Int> {
    let v: Vec<i64> = (0..batch * seq)
        .flat_map(|i| [(i as i64 * 7 + 3) % VOCAB as i64, (i % 2) as i64])
        .collect();
    Tensor::<1, Int>::from_ints(v.as_slice(), device).reshape([batch, seq, 2])
}

/// A packed forward equals each segment alone.
fn packed_matches_alone(block: MambaBlockConfig, path: MambaSsdPath, align: usize) {
    let device = test_device();
    let shape = VocabShape::new(VOCAB, NetworkShape::new(2));
    let net: MambaVocabNet = shape.init(block, &device);
    let lens = [[5usize, 9], [3, 7]];
    let width = 3 * align + 9usize.next_multiple_of(align);
    let mut starts = vec![vec![0usize; 2]; 2];
    for (b, row) in lens.iter().enumerate() {
        starts[b][1] = row[0].next_multiple_of(align);
    }
    let mut ids = vec![0i64; 2 * width * 2];
    let pair = |j: usize| [((j * 7 + 3) % VOCAB) as i64, (j % 2) as i64];
    for (b, row) in lens.iter().enumerate() {
        for (s, &len) in row.iter().enumerate() {
            for j in 0..len {
                let at = (b * width + starts[b][s] + j) * 2;
                ids[at..at + 2].copy_from_slice(&pair(j));
            }
        }
    }
    let ids = Tensor::<1, Int>::from_ints(ids.as_slice(), &device).reshape([2, width, 2]);
    let layout = Packed::from_starts(&starts, 0, width, &device);
    let (logits, _) = net.forward_packed(ids, None, path.clone(), &layout);
    for (b, row) in lens.iter().enumerate() {
        for (s, &len) in row.iter().enumerate() {
            let v: Vec<i64> = (0..len).flat_map(pair).collect();
            let alone = Tensor::<1, Int>::from_ints(v.as_slice(), &device).reshape([1, len, 2]);
            let (alone, _) = net.forward(alone, None, path.clone(), None, None);
            let packed = logits.clone().narrow(0, b, 1).narrow(1, starts[b][s], len);
            let d = max_rel_diff(packed, alone);
            assert!(d < dtype_tol(TOL), "row {b}, segment {s} differs by {d}");
        }
    }
}

#[test]
#[cfg(feature = "mamba3")]
fn packed_mamba3() {
    packed_matches_alone(
        MambaBlockConfig::Mamba3(m3()),
        MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(Some(4))),
        4,
    );
}

#[test]
#[cfg(feature = "mamba2")]
fn packed_mamba2() {
    packed_matches_alone(
        MambaBlockConfig::Mamba2(m2()),
        MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(Some(4))),
        4,
    );
}

#[test]
#[should_panic(expected = "takes no packed rows")]
#[cfg(feature = "mamba1")]
fn packed_mamba1_panics() {
    packed_matches_alone(MambaBlockConfig::Mamba1(m1()), MambaSsdPath::Mamba1, 4);
}

#[test]
fn vocab_forward_and_step() {
    let device = test_device().autodiff();
    for (tag, block, path) in families() {
        let net: MambaVocabNet = VocabShape::new(VOCAB, NetworkShape::new(2)).init(block, &device);
        let x = vocab_tokens(2, 6, &device);
        let (whole, _) = net.forward(x.clone(), None, path, None, None);
        let mut caches = None;
        for t in 0..6 {
            let (y, c) = net.step(x.clone().narrow(1, t, 1).squeeze_dim(1), caches, None);
            caches = Some(c);
            let d = max_rel_diff(y, whole.clone().narrow(1, t, 1).squeeze_dim(1));
            assert!(d < dtype_tol(TOL), "{tag}: position {t} differs by {d}");
        }
    }
}

#[test]
fn bidi_forward_and_gradients() {
    let device = test_device().autodiff();
    for (tag, block, path) in families() {
        let shape = BidiShape::new(4, OutputMergeConfig::cat_linear(4));
        let net: MambaBidiLayers = shape.init(block, &device);
        let x = Tensor::<3>::random([2, 6, D], Distribution::Normal(0.0, 1.0), &device);
        let (y, caches) = net.forward(x, None, path, None, None);
        assert_eq!(y.dims(), [2, 6, D], "{tag}");
        assert!(caches.slot_count() > 0, "{tag}");
        let grads = y.sum().backward();
        let (some, all) = grad_count(&net, &grads);
        assert!(some > 0, "{tag}: {some} of {all}");
    }
}

fn hnet(stages: Vec<MambaBlockConfig>, main: MambaBlockConfig) -> MambaHNetVocabNetConfig {
    let stack = || NetworkShape::new(2);
    let shape = HNetShape::new(stages.iter().map(|_| HNetStageShape::new(stack(), stack())).collect(), stack());
    MambaHNetVocabNetConfig::new(HNetVocabShape::new(shape, VOCAB).with_missing_lm_head(true), stages, main)
}

/// Stages of one family, a main stack of another, with one type.
#[test]
#[cfg(all(feature = "mamba2", feature = "mamba3"))]
fn hnet_mixes_stage_and_main_family() {
    let device = test_device().autodiff();
    let wide = || MambaBlockConfig::Mamba3(m3());
    let narrow = || {
        MambaBlockConfig::Mamba2(
            Mamba2Config::new(8)
                .with_expand(2)
                .with_per_head_dim(4)
                .with_state_rank(8)
                .with_ngroups(1)
                .with_conv_kernel(4),
        )
    };
    let config = hnet(vec![narrow()], wide());
    let net: MambaHNetVocabNet = config.init(&device);
    let x = vocab_tokens(2, 8, &device);
    let paths = (MambaSsdPath::mamba2_default(), MambaSsdPath::mamba3_default());
    let (whole, _, routing) = net.forward(x.clone(), None, paths.clone(), None);

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
#[cfg(feature = "mamba3")]
fn muon_plan_matches_the_family_type() {
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

/// The trainer of the examples, for each family.
#[test]
fn trainer_runs() {
    fn loss(model: &MambaLatentNet, x: Tensor<3>) -> (Tensor<1>, Tensor<3>) {
        // The SSD path follows the family of the first block.
        let path = match &model.layers.real_layers[0].block {
            #[cfg(feature = "mamba1")]
            MambaBlock::Mamba1(_) => MambaSsdPath::Mamba1,
            #[cfg(feature = "mamba2")]
            MambaBlock::Mamba2(_) => MambaSsdPath::mamba2_default(),
            #[cfg(feature = "mamba3")]
            MambaBlock::Mamba3(_) => MambaSsdPath::mamba3_default(),
        };
        let (y, _) = model.forward(x.autodiff(), None, path, None, None);
        (y.clone().sum().unsqueeze(), y.inner())
    }
    let device = test_device().autodiff();
    for (_tag, block, _path) in families() {
        let config = MambaLatentNetConfig::new(latent_shape(2), block);
        let net = config.init(&device);
        let optimizer = OptimizerConfig {
            fallback: FallbackConfig::AdamW(AdamWConfig::new()),
            muon: None,
        };
        let mut optim = optimizer.init(&config.muon_plan());
        let mut trainer = Trainer::new(net, loss, &optimizer, false);
        let x = input(2, 6, trainer.device());
        let (l, _) = trainer.train_step(x, &mut optim, 1e-3);
        assert_eq!(l.dims(), [1]);
    }
}

/// A captured step over the flat caches. On flex it replays eagerly, so this
/// checks the bounds and the write-back.
#[test]
#[cfg(feature = "mamba3")]
fn captured_step_over_flat_caches() {
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
    // Safety: the step reads nothing but its arguments and `net`, borrowed.
    let mut captured = unsafe { CapturedStep::capture(&device, xs[1].clone(), caches, |x, c| net.step(x, Some(c), None)) };
    for k in 1..4 {
        let y = captured.step(xs[k].clone()).clone();
        let d = max_abs_diff(y, eager[k].clone());
        assert_eq!(d, 0.0, "step {k} differs by {d}");
    }
}

// ---------------------------------------------------------------------------
// The mismatch rules (Q3) and the pierce of a test (Q1)
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "ssd_path family does not match the block")]
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
/// and at a `step` with no cache (the first block sizes every zero cache). Only
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

    let d_model: Tensor<2> = Tensor::zeros([2, D], &device);
    let first = net.layers.real_layers[0].block.zero_caches_2d(&d_model, 1).into_slots();
    let second = net.layers.real_layers[1].block.zero_caches_2d(&d_model, 1).into_slots();
    let slots = first.into_iter().chain(second).collect();
    let (y, _) = net.step(x0, Some(MambaCaches::from_slots(slots)), None);
    assert_eq!(y.dims(), [2, 5]);
}

/// What the handmade tests of the examples do, without the network-level match.
#[test]
#[cfg(feature = "mamba3")]
fn a_test_reaches_network_and_block_fields() {
    let device = test_device();
    let mut net: MambaLatentNet = latent_shape(1).init(MambaBlockConfig::Mamba3(m3()), &device);
    let [a, b] = net.in_proj.weight.dims();
    net.in_proj.weight = Param::from_tensor(Tensor::ones([a, b], &device));
    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(layer.norm.gamma.dims(), &device));
    let MambaBlock::Mamba3(block) = &mut layer.block else { unreachable!("a Mamba-3 stack") };
    let dims = block.in_proj.weight.dims();
    block.in_proj.weight = Param::from_tensor(Tensor::zeros(dims, &device));
    let (y, _) = net.forward(input(2, 4, &device), None, MambaSsdPath::mamba3_default(), None, None);
    assert_eq!(y.dims(), [2, 4, 5]);
}
