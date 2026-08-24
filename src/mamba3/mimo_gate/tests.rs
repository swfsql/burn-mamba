use super::*;
use crate::mamba3::prelude::*;
use crate::utils::test_helpers::max_abs_diff;
use burn::module::Param;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

const MIMO_RANK: usize = 3;

fn base_config() -> Mamba3Config {
    Mamba3Config::new(32)
        .with_state_rank(8)
        .with_expand(2)
        .with_per_head_dim(8)
        .with_mimo_rank(MIMO_RANK)
}

fn gated_config(write: Option<GateKind>, read: Option<GateKind>) -> Mamba3Config {
    base_config().with_mimo_mix(MimoMix::Gated { write, read })
}

/// Both arms, both squashings — the widest configuration.
fn both_arms() -> Mamba3Config {
    gated_config(
        Some(GateKind::Competitive),
        Some(GateKind::Independent),
    )
}

fn input(batch: usize, seq: usize, cfg: &Mamba3Config, device: &Device) -> Tensor<3> {
    Tensor::random(
        [batch, seq, cfg.d_model],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

/// The same block with its gate removed — the [`MimoMix::Sum`] twin, sharing
/// every other parameter (so the comparison isolates the gate, not the RNG).
fn sum_twin(model: &Mamba3) -> Mamba3 {
    let mut twin = model.clone();
    twin.mimo_gate = None;
    twin
}

/// Fill an arm's query and bias with random values — a "trained" gate.
fn randomize(arm: MimoGateArm, device: &Device) -> MimoGateArm {
    let dist = Distribution::Normal(0.0, 1.0);
    MimoGateArm {
        query_hf: Param::from_tensor(Tensor::random(arm.query_hf.dims(), dist, device)),
        bias_hm: Param::from_tensor(Tensor::random(arm.bias_hm.dims(), dist, device)),
        ..arm
    }
}

fn randomize_gate(model: Mamba3, device: &Device) -> Mamba3 {
    let gate = model.mimo_gate.expect("a gated block");
    Mamba3 {
        mimo_gate: Some(MimoGate {
            write: gate.write.map(|arm| randomize(arm, device)),
            read: gate.read.map(|arm| randomize(arm, device)),
        }),
        ..model
    }
}

// ---------------------------------------------------------------------------
// The weights themselves
// ---------------------------------------------------------------------------

/// Zero query + zero bias ⇒ mean-one weights for **both** kinds: `2σ(0) = 1`
/// and `M·softmax(0) = 1`. This is what makes a fresh gated block identical to
/// `MimoMix::Sum`.
#[test]
fn weights_are_one_at_init() {
    let device: Device = Default::default();
    let (nheads, width) = (4, 8);
    for kind in [GateKind::Independent, GateKind::Competitive] {
        let arm = MimoGateArmConfig::new(nheads, MIMO_RANK, width, kind).init(&device);
        let content = Tensor::<4>::random(
            [2, MIMO_RANK, nheads, width],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let w = arm.weights(content);
        assert_eq!([2, MIMO_RANK, nheads, 1], w.dims());
        let d = max_abs_diff(w, Tensor::<4>::ones([2, MIMO_RANK, nheads, 1], &device));
        assert!(d < 1e-6, "{kind:?}: weights at init differ from 1 by {d:.3e}");
    }
}

/// A competitive arm conserves the total weight (`Σₘ wₘ = M`) whatever the
/// query — that is the "ranks compete for a fixed budget" property.
#[test]
fn competitive_weights_conserve_their_budget() {
    let device: Device = Default::default();
    let (nheads, width) = (4, 8);
    let arm = randomize(
        MimoGateArmConfig::new(nheads, MIMO_RANK, width, GateKind::Competitive).init(&device),
        &device,
    );
    let content = Tensor::<4>::random(
        [2, MIMO_RANK, nheads, width],
        Distribution::Normal(0.0, 3.0),
        &device,
    );
    let total: Tensor<4> = arm.weights(content).sum_dim(1);
    let d = max_abs_diff(total, Tensor::<4>::full([2, 1, nheads, 1], MIMO_RANK as f64, &device));
    assert!(d < 1e-5, "Σₘ wₘ deviates from mimo_rank by {d:.3e}");
}

/// The score is RMS-normalised, so scaling the content leaves the weights
/// alone — magnitude cannot shout down the query's direction.
#[test]
fn weights_are_scale_invariant() {
    let device: Device = Default::default();
    let (nheads, width) = (4, 8);
    let arm = randomize(
        MimoGateArmConfig::new(nheads, MIMO_RANK, width, GateKind::Independent).init(&device),
        &device,
    );
    let content = Tensor::<4>::random(
        [2, MIMO_RANK, nheads, width],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let d = max_abs_diff(arm.weights(content.clone()), arm.weights(content * 7.0));
    assert!(d < 1e-4, "weights moved by {d:.3e} under a pure rescale");
}

// ---------------------------------------------------------------------------
// Identity at init, and non-triviality once trained
// ---------------------------------------------------------------------------

/// A freshly initialised gated block is **bit-exact** with its `MimoMix::Sum`
/// twin, on both pathways and on `step` — enabling the feature perturbs no
/// initialisation and no loaded checkpoint's forward.
#[test]
fn gated_equals_sum_at_init() {
    let device: Device = Default::default();
    let (batch, seq) = (2, 6);
    for cfg in [
        gated_config(Some(GateKind::Independent), None),
        gated_config(Some(GateKind::Competitive), None),
        gated_config(None, Some(GateKind::Independent)),
        gated_config(None, Some(GateKind::Competitive)),
        both_arms(),
    ] {
        let model = cfg.init(&device);
        let plain = sum_twin(&model);
        let x = input(batch, seq, &cfg, &device);
        let path = Mamba3SsdPath::default();

        let (gated_out, _) = model.forward_double_ssd(x.clone(), None, &path);
        let (sum_out, _) = plain.forward_double_ssd(x.clone(), None, &path);
        let d = max_abs_diff(gated_out, sum_out);
        assert!(d < 1e-6, "{:?}: double-ssd forward differs by {d:.3e}", cfg.mimo_mix);

        let (gated_out, _) = model.forward_single_ssd(x.clone(), None, &path);
        let (sum_out, _) = plain.forward_single_ssd(x.clone(), None, &path);
        let d = max_abs_diff(gated_out, sum_out);
        assert!(d < 1e-6, "{:?}: single-ssd forward differs by {d:.3e}", cfg.mimo_mix);

        let token = x.narrow(1, 0, 1).squeeze_dim::<2>(1);
        let (gated_out, _) = model.step(token.clone(), None);
        let (sum_out, _) = plain.step(token.clone(), None);
        let d = max_abs_diff(gated_out, sum_out);
        assert!(d < 1e-6, "{:?}: step differs by {d:.3e}", cfg.mimo_mix);

        let d = max_abs_diff(model.step_infinite(token.clone()), plain.step_infinite(token));
        assert!(d < 1e-6, "{:?}: step_infinite differs by {d:.3e}", cfg.mimo_mix);
    }
}

/// …and once the gate parameters move, it is no longer the same function.
/// Guards against the whole feature being a silent no-op (each arm alone, so a
/// dead arm cannot hide behind the other).
#[test]
fn trained_gate_changes_the_output() {
    let device: Device = Default::default();
    let (batch, seq) = (2, 6);
    for cfg in [
        gated_config(Some(GateKind::Independent), None),
        gated_config(Some(GateKind::Competitive), None),
        gated_config(None, Some(GateKind::Independent)),
        gated_config(None, Some(GateKind::Competitive)),
    ] {
        let model = cfg.init(&device);
        let plain = sum_twin(&model);
        let model = randomize_gate(model, &device);
        let x = input(batch, seq, &cfg, &device);
        let path = Mamba3SsdPath::default();

        let (gated_out, _) = model.forward_double_ssd(x.clone(), None, &path);
        let (sum_out, _) = plain.forward_double_ssd(x, None, &path);
        let d = max_abs_diff(gated_out, sum_out);
        assert!(
            d > 1e-3,
            "{:?}: a trained gate left the output unchanged (max abs diff {d:.3e})",
            cfg.mimo_mix
        );
    }
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

/// Every gate parameter is reachable from the loss on both pathways. The write
/// arm's query starts at zero, so this also checks the score's gradient does
/// not vanish there (`σ'(0)`, `softmax'` at uniform, and the RMS-normalised
/// content are all nonzero).
#[test]
fn gate_parameters_receive_gradients() {
    let device: Device = Default::default();
    let cfg = both_arms();
    let model = cfg.init(&device.clone().autodiff());
    let x = Tensor::from_inner(input(2, 6, &cfg, &device));
    let path = Mamba3SsdPath::default();

    for single_ssd in [false, true] {
        let (out, _) = if single_ssd {
            let (o, c) = model.forward_single_ssd(x.clone(), None, &path);
            (o, c.ssm_bhpr)
        } else {
            let (o, c) = model.forward_double_ssd(x.clone(), None, &path);
            (o, c.ssm_bhpr)
        };
        let head = Tensor::from_inner(Tensor::<3>::random(
            out.dims(),
            Distribution::Normal(0.0, 1.0),
            &device,
        ));
        let grads = (out * head).sum().backward();

        let gate = model.mimo_gate.as_ref().expect("a gated block");
        for (name, arm) in [("write", &gate.write), ("read", &gate.read)] {
            let arm = arm.as_ref().expect("both arms are configured");
            let dq = arm.query_hf.val().grad(&grads).expect("query gradient");
            let db = arm.bias_hm.val().grad(&grads).expect("bias gradient");
            for (what, g) in [("query", dq), ("bias", db)] {
                let mag = g.abs().max().into_scalar::<f32>();
                assert!(
                    mag > 1e-8,
                    "{name} {what}: gradient vanished (max |g| = {mag:.3e}, single_ssd = {single_ssd})"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration errors
// ---------------------------------------------------------------------------

/// A SISO block has one rank; there is nothing to mix, so asking for a gate is
/// an error rather than a silent no-op.
#[test]
#[should_panic(expected = "mimo_rank > 1")]
fn gated_siso_is_rejected() {
    let device: Device = Default::default();
    base_config()
        .with_mimo_rank(1)
        .with_mimo_mix(MimoMix::Gated {
            write: Some(GateKind::Independent),
            read: None,
        })
        .init(&device);
}

/// `Gated` with neither direction is just `Sum` spelled confusingly.
#[test]
#[should_panic(expected = "neither direction")]
fn gated_with_no_arm_is_rejected() {
    let device: Device = Default::default();
    base_config()
        .with_mimo_mix(MimoMix::Gated {
            write: None,
            read: None,
        })
        .init(&device);
}

// ---------------------------------------------------------------------------
// A *trained* gate rides correctly through every execution mode
// ---------------------------------------------------------------------------
//
// The identity-at-init tests above cannot see a mis-wired gate: at init every
// weight is 1, so a gate applied at the wrong index (or dropped from the
// shifted β-term, the boundary seed or the cache) still matches `Sum`. These
// run a **randomised** gate through the equivalences the block is built on —
// double-ssd ≡ single-ssd, and `forward` ≡ `step` unrolled — on values *and*
// gradients, which is what pins the gate to the right token and rank.

/// A gated block with random (non-identity) gate parameters.
fn trained_model(cfg: &Mamba3Config, device: &Device) -> Mamba3 {
    randomize_gate(cfg.init(device), device)
}

/// The shared setup for the equivalence tests: a trained-gate block on the
/// autodiff device, one input, and the two fixed random loss heads (one for the
/// output, one for the final state).
struct Fixture {
    model: Mamba3,
    x: Tensor<3>,
    head: Tensor<3>,
    state_head: Tensor<4>,
    seq: usize,
}

fn fixture() -> Fixture {
    let device: Device = Default::default();
    let cfg = both_arms();
    let (batch, seq) = (2, 5);
    Fixture {
        model: trained_model(&cfg, &device.clone().autodiff()),
        x: Tensor::from_inner(input(batch, seq, &cfg, &device)),
        head: Tensor::from_inner(input(batch, seq, &cfg, &device)),
        state_head: Tensor::from_inner(Tensor::<4>::random(
            [batch, cfg.nheads(), cfg.per_head_dim, cfg.state_rank],
            Distribution::Normal(0.0, 1.0),
            &device,
        )),
        seq,
    }
}

fn flat<const D: usize>(t: Tensor<D>) -> Tensor<1> {
    t.flatten(0, D - 1)
}

/// One forward+backward: the flattened output, the flattened final SSM state,
/// and the flattened gradient of every parameter the gate can reach.
fn run_with_grads(
    model: &Mamba3,
    x: Tensor<3>,
    head: Tensor<3>,
    state_head: Tensor<4>,
    run: impl FnOnce(&Mamba3, Tensor<3>) -> (Tensor<3>, Tensor<4>),
) -> (Tensor<1>, Tensor<1>, Vec<(&'static str, Tensor<1>)>) {
    let (out, state) = run(model, x);
    let (out_flat, state_flat) = (flat(out.clone().inner()), flat(state.clone().inner()));
    // Couple the loss to both the output and the final state, so a gate that is
    // right in the readout but wrong in the cache still shows up.
    let loss = (out * head).sum() + (state * state_head).sum();
    let grads = loss.backward();

    let gate = model.mimo_gate.as_ref().expect("a gated block");
    let mut out_grads: Vec<(&'static str, Tensor<1>)> = vec![
        ("in_proj", flat(model.in_proj.weight.val().grad(&grads).expect("in_proj"))),
        ("out_proj", flat(model.out_proj.weight.val().grad(&grads).expect("out_proj"))),
        ("b_bias", flat(model.b_bias_hmr.val().grad(&grads).expect("b_bias"))),
        ("c_bias", flat(model.c_bias_hmr.val().grad(&grads).expect("c_bias"))),
        ("dt_bias", flat(model.dt_bias_h.val().grad(&grads).expect("dt_bias"))),
    ];
    for (name, param) in [
        ("mimo_x", &model.mimo_x_hmp),
        ("mimo_z", &model.mimo_z_hmp),
        ("mimo_o", &model.mimo_o_hmp),
    ] {
        let param = param.as_ref().expect("a MIMO block");
        out_grads.push((name, flat(param.val().grad(&grads).expect(name))));
    }
    for (query, bias, arm) in [
        ("write.query", "write.bias", &gate.write),
        ("read.query", "read.bias", &gate.read),
    ] {
        if let Some(arm) = arm {
            out_grads.push((query, flat(arm.query_hf.val().grad(&grads).expect(query))));
            out_grads.push((bias, flat(arm.bias_hm.val().grad(&grads).expect(bias))));
        }
    }
    (out_flat, state_flat, out_grads)
}

fn assert_runs_match(
    label: &str,
    a: (Tensor<1>, Tensor<1>, Vec<(&'static str, Tensor<1>)>),
    b: (Tensor<1>, Tensor<1>, Vec<(&'static str, Tensor<1>)>),
    compare_state: bool,
    tol: f32,
) {
    let d = max_abs_diff(a.0, b.0);
    assert!(d < tol, "{label}: outputs differ by {d:.3e}");
    if compare_state {
        let d = max_abs_diff(a.1, b.1);
        assert!(d < tol, "{label}: final SSM states differ by {d:.3e}");
    }
    for ((name, ga), (_, gb)) in a.2.into_iter().zip(b.2) {
        let d = max_abs_diff(ga, gb);
        assert!(d < tol * 10.0, "{label}: d{name} differs by {d:.3e}");
    }
}

/// The two SSD pathways stay exactly equivalent under a trained write gate:
/// the gate must reach single-ssd's in-kernel `scale·B` **and** its same-step
/// γ correction the same way it reaches double-ssd's γ- and β-SSD calls.
#[test]
fn pathways_agree_under_a_trained_gate() {
    let f = fixture();
    let path = Mamba3SsdPath::Minimal(Some(4));

    let (p, h, sh) = (path.clone(), f.head.clone(), f.state_head.clone());
    let double = run_with_grads(&f.model, f.x.clone(), h, sh, |m, x| {
        let (out, cache) = m.forward_double_ssd(x, None, &p);
        (out, cache.ssm_bhpr)
    });
    let single = run_with_grads(&f.model, f.x, f.head, f.state_head, |m, x| {
        let (out, cache) = m.forward_single_ssd(x, None, &path);
        (out, cache.ssm_bhpr)
    });
    // The single-ssd accumulator has different mid-sequence semantics, so only
    // the output and the gradients are comparable here.
    assert_runs_match("double vs single (trained gate)", double, single, false, 1e-4);
}

/// `forward` ≡ `step` unrolled under a trained gate — the check that the gate
/// is cached with its own token, so the shifted β-term (and single-ssd's
/// boundary-β seed) reads the *previous* token's weights, not the current
/// one's.
fn forward_matches_step_under_a_trained_gate(single_ssd: bool) {
    let f = fixture();
    let (model, seq) = (&f.model, f.seq);
    let path = Mamba3SsdPath::Minimal(Some(4));

    let (p, h, sh) = (path, f.head.clone(), f.state_head.clone());
    let forward = run_with_grads(model, f.x.clone(), h, sh, |m, x| {
        if single_ssd {
            let (out, cache) = m.forward_single_ssd(x, None, &p);
            (out, cache.ssm_bhpr)
        } else {
            let (out, cache) = m.forward_double_ssd(x, None, &p);
            (out, cache.ssm_bhpr)
        }
    });
    let stepped = run_with_grads(model, f.x, f.head, f.state_head, |m, x| {
        let mut outs = Vec::with_capacity(seq);
        if single_ssd {
            let mut cache = None;
            for t in 0..seq {
                let (out, c) = m.step_single_ssd(x.clone().narrow(1, t, 1).squeeze_dim(1), cache);
                cache = Some(c);
                outs.push(out);
            }
            (Tensor::stack(outs, 1), cache.unwrap().ssm_bhpr)
        } else {
            let mut cache = None;
            for t in 0..seq {
                let (out, c) = m.step_double_ssd(x.clone().narrow(1, t, 1).squeeze_dim(1), cache);
                cache = Some(c);
                outs.push(out);
            }
            (Tensor::stack(outs, 1), cache.unwrap().ssm_bhpr)
        }
    });
    let label = if single_ssd { "single-ssd" } else { "double-ssd" };
    assert_runs_match(
        &format!("{label} forward vs step (trained gate)"),
        forward,
        stepped,
        true,
        1e-4,
    );
}

#[test]
fn double_ssd_forward_matches_step_under_a_trained_gate() {
    forward_matches_step_under_a_trained_gate(false);
}

#[test]
fn single_ssd_forward_matches_step_under_a_trained_gate() {
    forward_matches_step_under_a_trained_gate(true);
}

/// The plain end-to-end equivalence on the **public** API: `Mamba3::forward`
/// against `Mamba3::step` unrolled token by token (a missing cache defaults to
/// the single-ssd pathway), compared on the output, the final SSM state and the
/// parameter gradients. The gate is trained, so it is part of what is compared.
#[test]
fn forward_matches_step() {
    let f = fixture();
    let seq = f.seq;
    let ssm = |cache: Mamba3Cache| {
        cache
            .single_ssd()
            .expect("a missing cache defaults to single-ssd")
            .ssm_bhpr
    };

    let (h, sh) = (f.head.clone(), f.state_head.clone());
    let forward = run_with_grads(&f.model, f.x.clone(), h, sh, |m, x| {
        let (out, cache) = m.forward(x, None, Mamba3SsdPath::default());
        (out, ssm(cache))
    });
    let stepped = run_with_grads(&f.model, f.x, f.head, f.state_head, |m, x| {
        let mut cache = None;
        let mut outs = Vec::with_capacity(seq);
        for t in 0..seq {
            let (out, new_cache) = m.step(x.clone().narrow(1, t, 1).squeeze_dim(1), cache);
            cache = Some(new_cache);
            outs.push(out);
        }
        (Tensor::stack(outs, 1), ssm(cache.expect("one step at least")))
    });
    assert_runs_match("forward vs step", forward, stepped, true, 1e-4);
}

/// `step_infinite`'s closed form still lands on the unrolled fixed point with a
/// trained gate: under a constant token the gate is a constant per-(rank, head)
/// scale on `B`, which commutes with the resolvent.
#[test]
fn step_infinite_matches_unroll_under_a_trained_gate() {
    let device: Device = Default::default();
    // A healthy decay, so a few hundred steps reach the fixed point (matching
    // `step_constant::tests::decaying`).
    let cfg = both_arms().with_a_floor(1.0).with_dt_limit((0.05, 5.0));
    let model = trained_model(&cfg, &device);
    let x = Tensor::<2>::random([2, cfg.d_model], Distribution::Normal(0.0, 1.0), &device);

    let mut cache = None;
    let mut out = None;
    for _ in 0..300 {
        let (o, c) = model.step(x.clone(), cache);
        out = Some(o);
        cache = Some(c);
    }
    let d = max_abs_diff(out.unwrap(), model.step_infinite(x));
    assert!(d < 1e-3, "step_infinite vs 300 unrolled steps: {d:.6}");
}
