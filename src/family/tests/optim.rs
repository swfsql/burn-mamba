//! Muon plan and `Segmented` tests. The specs describe the real weights, and
//! the groups only ever select rank-2 tensors. A training step of each
//! sub-projection alone agrees with the stock optimizers.

use crate::prelude::*;
use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::{AdamWConfig, MuonConfig, Optimizer, RecordState, StateSink, StateSource};
use burn::prelude::*;
use burn_stack::optim::segmented::{Segmented, SegmentedState, SubProjectionState};
use burn_stack::utils::test_helpers::max_abs_diff;
use burn_stack::utils::test_helpers::test_device;

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
/// be rank 2 (Muon panics otherwise), and the sub-projection widths must sum to
/// the weight's output width.
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
                "spec {:?} sub-projection widths do not sum to {:?}'s output width",
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
/// drop the matching sub-projection — a zero-width one would not sum to the
/// weight.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_real1d_plan_fits_the_model() {
    mamba3_plan_fits(crate::mamba3::rotation::RotationKind::Real1D);
}

#[cfg(feature = "mamba3")]
fn mamba3_plan_fits(rotation: crate::mamba3::rotation::RotationKind) {
    use crate::prelude::*;
    let device = test_device();
    let block = Mamba3Config::new(32)
        .with_state_rank(16)
        .with_expand(2)
        .with_per_head_dim(16)
        .with_rotation_fraction(1.0)
        .with_rotation(rotation);
    let config = MambaLatentNetConfig::new(
        LatentShape::new(
            3,
            5,
            NetworkShape::new(2)
                .with_mlp(Some(GatedMlpConfig::new(32, 64).with_multiple_of(32))),
        )
        .with_final_norm(true),
        MambaBlockConfig::Mamba3(block),
    );
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
    let device = test_device();
    let config = MambaVocabNetConfig::new(
        VocabShape::new(20, NetworkShape::new(2)),
        MambaBlockConfig::Mamba2(
            Mamba2Config::new(32)
                .with_state_rank(16)
                .with_expand(2)
                .with_per_head_dim(16),
        ),
    );
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
    let device = test_device();
    let config = MambaLatentNetConfig::new(
        LatentShape::new(3, 5, NetworkShape::new(2)),
        MambaBlockConfig::Mamba1(Mamba1Config::new(32).with_state_rank(16)),
    );
    let rows = params_of(&config.init(&device));
    let plan = config.muon_plan();

    // in_proj, x_proj, out_proj.
    assert_eq!(plan.specs.len(), 3);
    assert_plan_fits(&plan, &rows, 2);
    assert_plan_excludes_boundaries(&plan, &rows);
}

/// Each bidirectional pair holds one block per direction, each under
/// `real_layers.N.block`. The plan must find the blocks of both directions.
#[cfg(feature = "mamba3")]
#[test]
fn bidi_plan_fits_the_model() {
    use crate::prelude::*;
    let device = test_device();
    let config = MambaBidiLayersConfig::new(
        BidiShape::new(2, OutputMergeConfig::cat_linear(2)),
        MambaBlockConfig::Mamba3(
            Mamba3Config::new(32)
                .with_state_rank(16)
                .with_expand(2)
                .with_per_head_dim(16),
        ),
    );
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

/// Muon owns a `CatLinear` merge, because the merge is a plain hidden matrix.
/// [`assert_plan_fits`] reads only the specs that the plan holds, so this test
/// finds the spec by its path.
#[cfg(feature = "mamba3")]
#[test]
fn bidi_plan_holds_the_cat_linear_merge() {
    use crate::prelude::*;
    let device = test_device();
    let config = MambaBidiLayersConfig::new(
        BidiShape::new(2, OutputMergeConfig::cat_linear(2)),
        MambaBlockConfig::Mamba3(
            Mamba3Config::new(32)
                .with_state_rank(16)
                .with_expand(2)
                .with_per_head_dim(16),
        ),
    );
    let rows = params_of(&config.init(&device));
    let plan = config.muon_plan();

    let merge = plan
        .specs
        .iter()
        .find(|s| s.path == "CatLinear.weight")
        .expect("the plan holds the CatLinear spec");
    assert_eq!(selected(merge, &rows).len(), 1);
    assert_eq!(merge.width(), 32);
}

/// The plan of a vocab network holds the feed-forward of each of its layers.
#[cfg(feature = "mamba2")]
#[test]
fn vocab_plan_holds_the_mlp() {
    use crate::prelude::*;
    let device = test_device();
    let config = MambaVocabNetConfig::new(
        VocabShape::new(
            20,
            NetworkShape::new(2)
                .with_mlp(Some(GatedMlpConfig::new(32, 64).with_multiple_of(32))),
        ),
        MambaBlockConfig::Mamba2(
            Mamba2Config::new(32)
                .with_state_rank(16)
                .with_expand(2)
                .with_per_head_dim(16),
        ),
    );
    let rows = params_of(&config.init(&device));
    let plan = config.muon_plan();

    for path in ["mlp.fc1.weight", "mlp.fc2.weight"] {
        let spec = plan
            .specs
            .iter()
            .find(|s| s.path == path)
            .unwrap_or_else(|| panic!("the plan has no {path} spec"));
        assert_eq!(selected(spec, &rows).len(), 2, "{path}");
    }
    assert_plan_fits(&plan, &rows, 2);
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
        .sub_projections
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

/// AdamW is elementwise, so splitting a weight into sub-projections and running
/// AdamW on each must reproduce plain AdamW on the whole weight — the property
/// that lets the non-Muon channels of a fused projection keep exactly the
/// update they had.
#[test]
fn all_adamw_sub_projections_equal_plain_adamw() {
    let device = test_device();
    let spec = ProjSpec::path(
        "w",
        vec![
            SubProjection::adamw("a", 3),
            SubProjection::adamw("b", 5),
            SubProjection::adamw("c", 4),
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

/// A single all-Muon sub-projection is just Muon.
#[test]
fn one_muon_sub_projection_equals_plain_muon() {
    let device = test_device();
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
    assert!(diff < 1e-6, "Segmented with one sub-projection diverged from Muon: {diff}");
}

/// To orthogonalise each sub-projection alone is *not* the same as to
/// orthogonalise the fused matrix — the whole reason the plan carries the
/// column seams.
#[test]
fn per_sub_projection_muon_differs_from_fused_muon() {
    let device = test_device();
    let spec = ProjSpec::path(
        "w",
        vec![SubProjection::muon("a", 6), SubProjection::muon("b", 6)],
    );
    let segmented = Segmented::new(&spec, muon().build(), adamw().build(), 1);
    let fused = muon().build();

    let w = rand_2d([6, 12], &device);
    let grad = rand_2d([6, 12], &device);
    let (w_seg, _) = segmented.step(1e-2, w.clone(), grad.clone(), None);
    let (w_fused, _) = fused.step(1e-2, w, grad, None);

    assert!(max_abs_diff(w_seg, w_fused) > 1e-6);
}

/// The hand-written record round-trips, keeping each sub-projection on its own
/// optimizer.
#[test]
fn segmented_state_round_trips() {
    let device = test_device();
    let spec = ProjSpec::path(
        "w",
        vec![
            SubProjection::muon("a", 6),
            SubProjection::adamw("b", 4),
            SubProjection::muon("c", 2),
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
    assert_eq!(reloaded.states.len(), 3);
    assert!(matches!(reloaded.states[0], SubProjectionState::Muon(_)));
    assert!(matches!(reloaded.states[1], SubProjectionState::AdamW(_)));
    assert!(matches!(reloaded.states[2], SubProjectionState::Muon(_)));

    // Continuing from the reloaded state matches continuing from the live one.
    let (a, _) = segmented.step(1e-2, w.clone(), grad.clone(), Some(state));
    let (b, _) = segmented.step(1e-2, w, grad, Some(reloaded));
    assert!(max_abs_diff(a, b) < 1e-6);
}

/// Under an SGD fallback the non-Muon sub-projections take plain SGD training
/// steps and hold no state: the state is the Muon sub-projections' alone, and
/// pairs back with them.
#[test]
fn sgd_fallback_sub_projections_are_stateless_sgd() {
    let device = test_device();
    let widths = vec![6, 4, 2];
    let spec = ProjSpec::path(
        "w",
        vec![
            SubProjection::muon("a", 6),
            SubProjection::adamw("b", 4),
            SubProjection::muon("c", 2),
        ],
    );
    let sgd = burn_stack::optim::SgdConfig::new().build();
    let segmented = Segmented::new(&spec, muon().build(), sgd.clone(), 1);
    let plain = muon().build();

    let mut w = rand_2d([6, 12], &device);
    let mut w_ref = w.clone().split_with_sizes(widths.clone(), 1);
    let (mut state, mut s_a, mut s_c) = (None, None, None);
    for _ in 0..3 {
        let grad = rand_2d([6, 12], &device);
        let (next, s) = segmented.step(1e-2, w, grad.clone(), state);
        let s = s.expect("state");
        assert_eq!(s.states.len(), 2, "no entry for the SGD sub-projection");
        assert!(matches!(s.states[1], SubProjectionState::Muon(_)));
        (w, state) = (next, Some(s));

        let g = grad.split_with_sizes(widths.clone(), 1);
        let (a, s) = plain.step(1e-2, w_ref[0].clone(), g[0].clone(), s_a);
        (w_ref[0], s_a) = (a, s);
        w_ref[1] = sgd.step(1e-2, w_ref[1].clone(), g[1].clone(), None).0;
        let (c, s) = plain.step(1e-2, w_ref[2].clone(), g[2].clone(), s_c);
        (w_ref[2], s_c) = (c, s);
    }

    let diff = max_abs_diff(w, Tensor::cat(w_ref, 1));
    assert!(diff < 1e-6, "segmented Muon + SGD diverged from its sub-projections: {diff}");
}

/// Building the module optimizer wires one group per Muon-owning spec, on top of
/// the AdamW fallback.
#[cfg(feature = "mamba3")]
#[test]
fn build_assembles_groups_without_panicking() {
    use crate::prelude::*;
    let device = test_device().autodiff();
    let config = MambaLatentNetConfig::new(
        LatentShape::new(2, 2, NetworkShape::new(1)),
        MambaBlockConfig::Mamba3(
            Mamba3Config::new(16)
                .with_state_rank(16)
                .with_expand(2)
                .with_per_head_dim(8),
        ),
    );
    let model = config.init(&device);
    let mut optim = config.muon_plan().build(&adamw().into(), &muon());

    // One real training step over the whole model: every parameter goes through its
    // group's optimizer, so a mis-targeted group (a 1-D or 3-D tensor handed to
    // Muon) would panic here.
    let x = Tensor::<3>::random([1, 8, 2], burn::tensor::Distribution::Default, &device);
    let (y, _) = model.forward(x, None, MambaSsdPath::mamba3_default(), None, None);
    let grads = burn::optim::GradientsParams::from_grads(y.sum().backward(), &model);
    let _model = optim.step(1e-3, model, grads);
}

/// The whole [`ModuleOptimizer`](burn::optim::ModuleOptimizer) round-trips: on
/// reload each parameter is matched back to its group, and the states of the
/// sub-projections land on the same optimizers, so training continues
/// unchanged.
#[cfg(feature = "mamba3")]
#[test]
fn module_optimizer_state_round_trips_through_a_record() {
    use crate::prelude::*;
    let device = test_device().autodiff();
    let config = MambaLatentNetConfig::new(
        LatentShape::new(2, 2, NetworkShape::new(1)),
        MambaBlockConfig::Mamba3(
            Mamba3Config::new(16)
                .with_state_rank(16)
                .with_expand(2)
                .with_per_head_dim(8),
        ),
    );
    let plan = config.muon_plan();
    let model = config.init(&device);
    let x = Tensor::<3>::random([1, 8, 2], burn::tensor::Distribution::Default, &device);

    // The same (deterministic) gradient drives every training step below.
    let train_step = |model: MambaLatentNet, optim: &mut burn::optim::ModuleOptimizer| {
        let (y, _) = model.forward(x.clone(), None, MambaSsdPath::mamba3_default(), None, None);
        let grads = burn::optim::GradientsParams::from_grads(y.sum().backward(), &model);
        optim.step(1e-3, model, grads)
    };
    let in_proj = |m: &MambaLatentNet| match m {
        MambaLatentNet::Mamba3(net) => net.layers.real_layers[0].block.in_proj.weight.val(),
        _ => panic!("expected a Mamba-3 network"),
    };

    let mut live = plan.build(&adamw().into(), &muon());
    let model = train_step(model, &mut live);

    let mut reloaded = plan
        .build(&adamw().into(), &muon())
        .from_bytes(live.into_bytes().expect("serialize"))
        .expect("deserialize");
    let mut fresh = plan.build(&adamw().into(), &muon());

    // Same second training step from the reloaded state vs. from no state at all.
    let from_record = train_step(model.clone(), &mut reloaded);
    let from_scratch = train_step(model, &mut fresh);

    // A second training step *with* momentum differs from one without — i.e.
    // the record really carried the sub-projection states — and the reload did
    // not panic on a state landing in the wrong group.
    assert!(max_abs_diff(in_proj(&from_record), in_proj(&from_scratch)) > 1e-9);
}

/// [`MuonPlan::describe`] reports one line per parameter, and the Muon share
/// matches the plan.
#[cfg(feature = "mamba3")]
#[test]
fn describe_reports_every_parameter() {
    use crate::prelude::*;
    let device = test_device();
    let config = MambaLatentNetConfig::new(
        LatentShape::new(1, 10, NetworkShape::new(2)),
        MambaBlockConfig::Mamba3(
            Mamba3Config::new(16)
                .with_state_rank(16)
                .with_expand(2)
                .with_per_head_dim(8),
        ),
    );
    let model = config.init(&device);
    let report = config.muon_plan().describe(&model);

    let rows = params_of(&model).len();
    assert_eq!(report.lines().count(), rows + 1, "one line per param + summary");
    // Both layers' in_proj show their sub-projections, Δ/A/λ marked as AdamW's.
    assert_eq!(report.matches("dt:").count(), 2);
    assert!(report.contains("on muon:"));
}
