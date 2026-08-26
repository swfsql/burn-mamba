//! Muon plan/segmentation tests: the specs describe the real weights, the
//! groups only ever select rank-2 tensors, and per-block stepping agrees with
//! the stock optimizers.

use crate::prelude::*;
use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::{AdamWConfig, MuonConfig, Optimizer, RecordState, StateSink, StateSource};
use burn::prelude::*;
use burn_stack::optim::segmented::{BlockState, Segmented, SegmentedState};
use burn_stack::utils::test_helpers::max_abs_diff;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// One visited parameter: its module path, shape and id.
struct ParamRow {
    path: String,
    dims: Vec<usize>,
    id: ParamId,
}

#[derive(Default)]
struct Collect {
    path: Vec<String>,
    rows: Vec<ParamRow>,
}

impl ModuleVisitor for Collect {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.rows.push(ParamRow {
            path: self.path.join("."),
            dims: param.val().shape().dims::<D>().to_vec(),
            id: param.id,
        });
    }
}

fn params_of(module: &impl Module) -> Vec<ParamRow> {
    let mut collect = Collect::default();
    module.visit(&mut collect);
    collect.rows
}

/// The rows a spec's parameter group selects.
fn selected<'a>(spec: &ProjSpec, rows: &'a [ParamRow]) -> Vec<&'a ParamRow> {
    let group = spec.param_group();
    rows.iter()
        .filter(|row| group.matches(&row.id, Some(row.path.as_str())))
        .collect()
}

fn adamw() -> AdamWConfig {
    AdamWConfig::new()
}

fn muon() -> MuonConfig {
    muon_config(1e-4)
}

// ---------------------------------------------------------------------------
// The specs describe the weights that actually exist
// ---------------------------------------------------------------------------

/// Every spec must select at least one parameter, every selected parameter must
/// be rank 2 (Muon panics otherwise), and the segment widths must sum to the
/// weight's output width.
fn assert_plan_fits(plan: &MuonPlan, rows: &[ParamRow], expect_min: usize) {
    for spec in &plan.specs {
        let hits = selected(spec, rows);
        assert!(
            hits.len() >= expect_min,
            "spec {:?} selected {} params, expected at least {expect_min}",
            spec.path,
            hits.len()
        );
        for row in hits {
            assert_eq!(
                row.dims.len(),
                2,
                "spec {:?} selected the non-2D parameter {:?} {:?}",
                spec.path,
                row.path,
                row.dims
            );
            assert_eq!(
                row.dims[1],
                spec.width(),
                "spec {:?} segment widths do not sum to {:?}'s output width",
                spec.path,
                row.path
            );
        }
    }
}

/// No plan may ever select an embedding, a head, or the network's own
/// input/output projection.
fn assert_plan_excludes_boundaries(plan: &MuonPlan, rows: &[ParamRow]) {
    let boundary = |path: &str| {
        let leaf = path.rsplit('.').nth(1).unwrap_or("");
        matches!(leaf, "embedding" | "lm_head" | "class_tokens_emb")
            // the network's own projections sit at the model root, i.e. their
            // path has no block container above them
            || (matches!(leaf, "in_proj" | "out_proj")
                && !burn_stack::optim::BLOCK_CONTAINERS.iter().any(|c| path.contains(c)))
    };
    for spec in &plan.specs {
        for row in selected(spec, rows) {
            assert!(
                !boundary(&row.path),
                "spec {:?} selected the boundary parameter {:?}",
                spec.path,
                row.path
            );
        }
    }
}

#[cfg(feature = "mamba3")]
#[test]
fn mamba3_plan_fits_the_model() {
    mamba3_plan_fits(crate::mamba3::rotation::RotationKind::Complex2D);
}

/// `Real1D` drops the `in_proj`'s rotation columns entirely, so the plan has to
/// drop the matching segment — a zero-width one would not sum to the weight.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_real1d_plan_fits_the_model() {
    mamba3_plan_fits(crate::mamba3::rotation::RotationKind::Real1D);
}

#[cfg(feature = "mamba3")]
fn mamba3_plan_fits(rotation: crate::mamba3::rotation::RotationKind) {
    use crate::prelude::*;
    let device = Device::default();
    let block = Mamba3Config::new(32)
        .with_state_rank(16)
        .with_expand(2)
        .with_per_head_dim(16)
        .with_rope_fraction(1.0)
        .with_rotation(rotation);
    let config = MambaLatentNetConfig::Mamba3 {
        input_size: 3,
        output_size: 5,
        final_norm: true,
        n_real_layers: 2,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: block,
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: burn_stack::modules::ResidualsConfig::Standard,
        mlp: Some(burn_stack::modules::GatedMlpConfig::new(32, 64).with_multiple_of(32)),
    };
    let rows = params_of(&config.init(&device));
    let plan = config.muon_plan();

    // in_proj, out_proj, fc1, fc2.
    assert_eq!(plan.specs.len(), 4);
    assert_plan_fits(&plan, &rows, 2);
    assert_plan_excludes_boundaries(&plan, &rows);
}

#[cfg(feature = "mamba2")]
#[test]
fn mamba2_plan_fits_the_model() {
    use crate::prelude::*;
    let device = Device::default();
    let config = MambaVocabNetConfig::Mamba2 {
        vocab_size: 20,
        pad_vocab_size_multiple: 1,
        n_real_layers: 2,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: Mamba2Config::new(32)
            .with_state_rank(16)
            .with_expand(2)
            .with_per_head_dim(16),
        missing_lm_head: true,
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: burn_stack::modules::ResidualsConfig::Standard,
        mlp: None,
    };
    let rows = params_of(&config.init(&device));
    let plan = config.muon_plan();

    assert_eq!(plan.specs.len(), 2);
    assert_plan_fits(&plan, &rows, 2);
    assert_plan_excludes_boundaries(&plan, &rows);
}

#[cfg(feature = "mamba1")]
#[test]
fn mamba1_plan_fits_the_model() {
    use crate::prelude::*;
    let device = Device::default();
    let config = MambaLatentNetConfig::Mamba1 {
        input_size: 3,
        output_size: 5,
        final_norm: false,
        n_real_layers: 2,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: Mamba1Config::new(32).with_state_rank(16),
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: burn_stack::modules::ResidualsConfig::Standard,
        mlp: None,
    };
    let rows = params_of(&config.init(&device));
    let plan = config.muon_plan();

    // in_proj, x_proj, out_proj.
    assert_eq!(plan.specs.len(), 3);
    assert_plan_fits(&plan, &rows, 2);
    assert_plan_excludes_boundaries(&plan, &rows);
}

/// A bidirectional stack stores its blocks under `straight_block`/`reverse_block`
/// rather than `block`; the plan must still find them (both of them).
#[cfg(feature = "mamba3")]
#[test]
fn bidi_plan_fits_the_model() {
    use crate::unified::MambaBidiLayersConfig;
    use burn_stack::modules::bidi::OutputMergeConfig;
    use crate::prelude::*;
    let device = Device::default();
    let config = MambaBidiLayersConfig::Mamba3 {
        n_real_layers: 2,
        n_virtual_layers: None,
        mamba_block: Mamba3Config::new(32)
            .with_state_rank(16)
            .with_expand(2)
            .with_per_head_dim(16),
        ignore_first_residual: false,
        ignore_last_residual: false,
        outputs_merge: OutputMergeConfig::cat_linear(2),
        class_latents: Vec::new(),
        residuals: burn_stack::modules::ResidualsConfig::Standard,
    };
    let rows = params_of(&config.init(&device));
    let plan = config.muon_plan();

    // Both directions of the single pair are selected by the block specs.
    let in_proj = plan
        .specs
        .iter()
        .find(|s| s.path == "in_proj.weight")
        .expect("in_proj spec");
    assert_eq!(selected(in_proj, &rows).len(), 2);

    assert_plan_fits(&plan, &rows, 1);
    assert_plan_excludes_boundaries(&plan, &rows);
}

/// The Muon-owned share of a Mamba-3 `in_proj`: the Δ/A/λ channels stay behind.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_scalar_channels_stay_on_adamw() {
    use crate::prelude::*;
    let config = Mamba3Config::new(64).with_state_rank(32).with_expand(2);
    let specs = config.muon_projections();
    let in_proj = &specs[0];
    assert_eq!(in_proj.width(), config.d_in_proj());

    let scalar: usize = in_proj
        .segments
        .iter()
        .filter(|s| !s.muon)
        .map(|s| s.width)
        .sum();
    assert_eq!(scalar, 3 * config.nheads());
    assert!(!in_proj.is_whole_muon());
}

// ---------------------------------------------------------------------------
// Segmented stepping
// ---------------------------------------------------------------------------

fn rand_2d(shape: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::random(shape, burn::tensor::Distribution::Default, device)
}

/// AdamW is elementwise, so splitting a weight into blocks and running AdamW on
/// each must reproduce plain AdamW on the whole weight — the property that lets
/// the non-Muon channels of a fused projection keep exactly the update they had.
#[test]
fn all_adamw_blocks_equal_plain_adamw() {
    let device = Device::default();
    let spec = ProjSpec::path(
        "w",
        vec![
            ProjSegment::adamw("a", 3),
            ProjSegment::adamw("b", 5),
            ProjSegment::adamw("c", 4),
        ],
    );
    let segmented = Segmented::new(&spec, muon().build(), adamw().build(), 1);
    let plain = adamw().build();

    let mut w_seg = rand_2d([6, 12], &device);
    let mut w_ref = w_seg.clone();
    let (mut s_seg, mut s_ref) = (None, None);

    for _ in 0..3 {
        let grad = rand_2d([6, 12], &device);
        let (w, s) = segmented.step(1e-2, w_seg, grad.clone(), s_seg);
        (w_seg, s_seg) = (w, s);
        let (w, s) = plain.step(1e-2, w_ref, grad, s_ref);
        (w_ref, s_ref) = (w, s);
    }

    let diff = max_abs_diff(w_seg, w_ref);
    assert!(diff < 1e-6, "segmented AdamW diverged from plain AdamW: {diff}");
}

/// A single all-Muon block is just Muon.
#[test]
fn one_muon_block_equals_plain_muon() {
    let device = Device::default();
    let spec = ProjSpec::path_whole("w", 12);
    let segmented = Segmented::new(&spec, muon().build(), adamw().build(), 1);
    let plain = muon().build();

    let mut w_seg = rand_2d([6, 12], &device);
    let mut w_ref = w_seg.clone();
    let (mut s_seg, mut s_ref) = (None, None);

    for _ in 0..3 {
        let grad = rand_2d([6, 12], &device);
        let (w, s) = segmented.step(1e-2, w_seg, grad.clone(), s_seg);
        (w_seg, s_seg) = (w, s);
        let (w, s) = plain.step(1e-2, w_ref, grad, s_ref);
        (w_ref, s_ref) = (w, s);
    }

    let diff = max_abs_diff(w_seg, w_ref);
    assert!(diff < 1e-6, "one-block Segmented diverged from Muon: {diff}");
}

/// Per-block orthogonalisation is *not* the same as orthogonalising the fused
/// matrix — the whole reason the plan carries the column seams.
#[test]
fn per_block_muon_differs_from_fused_muon() {
    let device = Device::default();
    let spec = ProjSpec::path(
        "w",
        vec![ProjSegment::muon("a", 6), ProjSegment::muon("b", 6)],
    );
    let segmented = Segmented::new(&spec, muon().build(), adamw().build(), 1);
    let fused = muon().build();

    let w = rand_2d([6, 12], &device);
    let grad = rand_2d([6, 12], &device);
    let (w_seg, _) = segmented.step(1e-2, w.clone(), grad.clone(), None);
    let (w_fused, _) = fused.step(1e-2, w, grad, None);

    assert!(max_abs_diff(w_seg, w_fused) > 1e-6);
}

/// The hand-written record round-trips, keeping each block on its own optimizer.
#[test]
fn segmented_state_round_trips() {
    let device = Device::default();
    let spec = ProjSpec::path(
        "w",
        vec![
            ProjSegment::muon("a", 6),
            ProjSegment::adamw("b", 4),
            ProjSegment::muon("c", 2),
        ],
    );
    let segmented = Segmented::new(&spec, muon().build(), adamw().build(), 1);

    let w = rand_2d([6, 12], &device);
    let grad = rand_2d([6, 12], &device);
    let (w, state) = segmented.step(1e-2, w, grad.clone(), None);
    let state = state.expect("state");

    let mut sink = StateSink::default();
    state.state_flatten("p", &mut sink);
    let mut source = StateSource::new(sink.scalars.into_iter().collect());
    for (name, data) in sink.tensors {
        source.insert_tensor(name, data);
    }
    let reloaded = SegmentedState::<2>::state_unflatten("p", &mut source, &device)
        .expect("reloaded state");
    assert_eq!(reloaded.blocks.len(), 3);
    assert!(matches!(reloaded.blocks[0], BlockState::Muon(_)));
    assert!(matches!(reloaded.blocks[1], BlockState::AdamW(_)));
    assert!(matches!(reloaded.blocks[2], BlockState::Muon(_)));

    // Continuing from the reloaded state matches continuing from the live one.
    let (a, _) = segmented.step(1e-2, w.clone(), grad.clone(), Some(state));
    let (b, _) = segmented.step(1e-2, w, grad, Some(reloaded));
    assert!(max_abs_diff(a, b) < 1e-6);
}

/// Building the module optimizer wires one group per Muon-owning spec, on top of
/// the AdamW fallback.
#[cfg(feature = "mamba3")]
#[test]
fn build_assembles_groups_without_panicking() {
    use crate::prelude::*;
    let device = Device::default().autodiff();
    let config = MambaLatentNetConfig::Mamba3 {
        input_size: 2,
        output_size: 2,
        final_norm: false,
        n_real_layers: 1,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: Mamba3Config::new(16)
            .with_state_rank(16)
            .with_expand(2)
            .with_per_head_dim(8),
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: burn_stack::modules::ResidualsConfig::Standard,
        mlp: None,
    };
    let model = config.init(&device);
    let mut optim = config.muon_plan().build(&adamw(), &muon());

    // One real step over the whole model: every parameter goes through its
    // group's optimizer, so a mis-targeted group (a 1-D or 3-D tensor handed to
    // Muon) would panic here.
    let x = Tensor::<3>::random([1, 8, 2], burn::tensor::Distribution::Default, &device);
    let (y, _) = model.forward(x, None, MambaSsdPath::mamba3_default(), None);
    let grads = burn::optim::GradientsParams::from_grads(y.sum().backward(), &model);
    let _model = optim.step(1e-3, model, grads);
}

/// The whole [`ModuleOptimizer`](burn::optim::ModuleOptimizer) round-trips: on
/// reload each parameter is matched back to its group, and the per-block states
/// land on the same optimizers, so training continues unchanged.
#[cfg(feature = "mamba3")]
#[test]
fn module_optimizer_state_round_trips_through_a_record() {
    use crate::prelude::*;
    let device = Device::default().autodiff();
    let config = MambaLatentNetConfig::Mamba3 {
        input_size: 2,
        output_size: 2,
        final_norm: false,
        n_real_layers: 1,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: Mamba3Config::new(16)
            .with_state_rank(16)
            .with_expand(2)
            .with_per_head_dim(8),
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: burn_stack::modules::ResidualsConfig::Standard,
        mlp: None,
    };
    let plan = config.muon_plan();
    let model = config.init(&device);
    let x = Tensor::<3>::random([1, 8, 2], burn::tensor::Distribution::Default, &device);

    // The same (deterministic) gradient drives every step below.
    let step = |model: MambaLatentNet, optim: &mut burn::optim::ModuleOptimizer| {
        let (y, _) = model.forward(x.clone(), None, MambaSsdPath::mamba3_default(), None);
        let grads = burn::optim::GradientsParams::from_grads(y.sum().backward(), &model);
        optim.step(1e-3, model, grads)
    };
    let in_proj = |m: &MambaLatentNet| match m {
        MambaLatentNet::Mamba3(net) => net.layers.real_layers[0].block.in_proj.weight.val(),
        _ => panic!("expected a Mamba-3 network"),
    };

    let mut live = plan.build(&adamw(), &muon());
    let model = step(model, &mut live);

    let mut reloaded = plan
        .build(&adamw(), &muon())
        .from_bytes(live.into_bytes().expect("serialize"))
        .expect("deserialize");
    let mut fresh = plan.build(&adamw(), &muon());

    // Same second step from the reloaded state vs. from no state at all.
    let from_record = step(model.clone(), &mut reloaded);
    let from_scratch = step(model, &mut fresh);

    // A second step *with* momentum differs from one without — i.e. the record
    // really carried the per-block state — and the reload did not panic on a
    // state landing in the wrong group.
    assert!(max_abs_diff(in_proj(&from_record), in_proj(&from_scratch)) > 1e-9);
}

/// [`MuonPlan::describe`] reports one line per parameter, and the Muon share
/// matches the plan.
#[cfg(feature = "mamba3")]
#[test]
fn describe_reports_every_parameter() {
    use crate::prelude::*;
    let device = Device::default();
    let config = MambaLatentNetConfig::Mamba3 {
        input_size: 1,
        output_size: 10,
        final_norm: false,
        n_real_layers: 2,
        n_virtual_layers: None,
        grad_horizon: None,
        mamba_block: Mamba3Config::new(16)
            .with_state_rank(16)
            .with_expand(2)
            .with_per_head_dim(8),
        class_tokens: Vec::new(),
        class_latents: Vec::new(),
        ignore_first_residual: false,
        ignore_last_residual: false,
        residuals: burn_stack::modules::ResidualsConfig::Standard,
        mlp: None,
    };
    let model = config.init(&device);
    let report = config.muon_plan().describe(&model);

    let rows = params_of(&model).len();
    assert_eq!(report.lines().count(), rows + 1, "one line per param + summary");
    // Both layers' in_proj show their column segments, Δ/A/λ marked as AdamW's.
    assert_eq!(report.matches("dt:").count(), 2);
    assert!(report.contains("on muon:"));
}
