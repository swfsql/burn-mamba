//! Tests for [`Layers::grad_horizon`] — the truncated-BPTT cut through a
//! (virtual-)layer stack.
//!
//! The cut is **silent when it goes wrong**: values and gradients stay correct
//! whether or not the prefix actually escaped the autodiff graph, and only the
//! memory saving disappears. So these tests assert the two mechanisms directly
//! rather than inferring them from numbers:
//!
//! * **gradient reachability** into the prefix — whether a loss built on what the
//!   prefix produced can still reach the stack input, which is exactly what a
//!   tracked tensor leaking *into* the prefix would restore.
//!
//!   Note this cannot use [`Tensor::is_require_grad`]: it matches only
//!   `Requirement::Grad`, i.e. leaves that retain a gradient, so it reports
//!   `false` for every intermediate (`Requirement::GradInBackward`) whether or
//!   not that intermediate is in the graph. Reachability has no such blind spot.
//! * absence of a gradient on prefix-only parameters — catches
//!   [`detach_params`](crate::utils::detach_params) failing to reach a nested
//!   `Param`, which is a live hazard because `Module::map` is a no-op on plain
//!   `Tensor` fields and only recurses through module-typed ones.
//!
//! Plus the two numeric guarantees: `K >= n_virtual` must reproduce the untouched
//! stack exactly, and under weight sharing a shared weight must collect the
//! gradient of its *tracked applications only*.

use super::*;
use crate::mamba3::prelude::{Mamba3Caches, Mamba3Config, Mamba3SsdPath};
use crate::modules::LayersBuilder;
use crate::utils::Schedule;
use crate::utils::test_helpers::max_abs_diff;
use burn::tensor::Distribution;

type Device = burn::prelude::Device;

fn block_config(d_model: usize) -> Mamba3Config {
    Mamba3Config::new(d_model)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .with_ngroups(1)
        .with_mimo_rank(1)
        .with_rope_fraction(Some(0.5))
}

const D_MODEL: usize = 16;
const BATCH: usize = 2;
const SEQ: usize = 6;

fn path() -> Mamba3SsdPath {
    Mamba3SsdPath::Minimal(Some(4))
}

fn input(device: &Device) -> Tensor<3> {
    Tensor::<3>::random(
        [BATCH, SEQ, D_MODEL],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

/// Sum of every tensor in a single-SSD cache slot, as one scalar to build a loss
/// from.
///
/// Reaches into the concrete cache rather than going through `CacheStack` so that
/// *all* of the slot's tensors take part — a leak through any single one of them
/// (the rotation accumulator is the easiest to forget) must show up.
fn slot_flat(caches: &Mamba3Caches, i: usize) -> Vec<Tensor<1>> {
    use crate::mamba3::prelude::RotationState;
    let c = match caches {
        Mamba3Caches::SingleSsd(cs) => &cs.caches[i],
        Mamba3Caches::DoubleSsd(_) => panic!("expected the default single-ssd caches"),
    };
    vec![
        c.ssm_bhpr.clone().reshape([-1]),
        c.k_state_bmhr.clone().reshape([-1]),
        c.v_state_bhp.clone().reshape([-1]),
        match &c.rotation {
            RotationState::Angle(t) => t.clone().reshape([-1]),
            RotationState::Quaternion(t) => t.clone().reshape([-1]),
        },
    ]
}

/// Every cache tensor of every slot, flattened, in a fixed order.
fn all_slots_flat(caches: &Mamba3Caches, n: usize) -> Vec<Tensor<1>> {
    (0..n).flat_map(|i| slot_flat(caches, i)).collect()
}

fn slot_sum(caches: &Mamba3Caches, i: usize) -> Tensor<1> {
    use crate::mamba3::prelude::RotationState;
    let c = match caches {
        Mamba3Caches::SingleSsd(cs) => &cs.caches[i],
        Mamba3Caches::DoubleSsd(_) => panic!("expected the default single-ssd caches"),
    };
    let rot = match &c.rotation {
        RotationState::Angle(t) => t.clone().sum(),
        RotationState::Quaternion(t) => t.clone().sum(),
    };
    c.ssm_bhpr.clone().sum() + c.k_state_bmhr.clone().sum() + c.v_state_bhp.clone().sum() + rot
}

// ===========================================================================
// 1. `K >= n_virtual` is the untouched stack
// ===========================================================================

/// A horizon covering every layer must reproduce `grad_horizon: None` on values
/// **and** on parameter gradients — i.e. the field cannot perturb the default
/// path, and `cut == 0` really does take the untouched branch.
#[test]
fn full_horizon_matches_no_horizon() {
    let device = Device::default().autodiff();
    let n = 4;

    let plain = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);
    let mut full = <Layers<_> as Clone>::clone(&plain);
    full.grad_horizon = Some(n);

    let x = input(&device);

    let (y_a, _) = plain.forward(x.clone(), None, path(), None);
    let (y_b, _) = full.forward(x.clone(), None, path(), None);
    assert!(max_abs_diff(y_a.clone(), y_b.clone()) < 1e-6, "outputs differ");

    let g_a = y_a.sum().backward();
    let g_b = y_b.sum().backward();
    for i in 0..n {
        let w = &plain.real_layers[i].mamba_block.in_proj.weight;
        let d_a = w.val().grad(&g_a).expect("baseline grad");
        let d_b = w.val().grad(&g_b).expect("full-horizon grad");
        assert!(
            max_abs_diff(d_a, d_b) < 1e-6,
            "layer {i} in_proj grad differs under a full horizon",
        );
    }
}

// ===========================================================================
// 2. Nothing tracked leaks into the prefix
// ===========================================================================

/// Nothing the no-grad prefix produced may lead back to what fed it.
///
/// The probe is built from the **prefix's own caches only** — deliberately not
/// from the stack output, because the tracked suffix consumes the carried caches
/// of the slots above the cut, and gradient reaching those is correct
/// cross-segment BPTT rather than a leak. An independent `anchor` leaf keeps the
/// graph non-empty, so `backward` has something to walk even when the prefix
/// contributes nothing at all.
///
/// Two inputs are probed: the stack input, which enters only through the prefix,
/// and — carrying caches out of an earlier tracked run — the input that produced
/// those caches, which is what reaches back in if `cache_to_inner` is skipped.
#[test]
fn nothing_tracked_reaches_the_prefix() {
    let device = Device::default().autodiff();
    let (n, k) = (4, 2);
    let cut = n - k;

    let mut layers = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);

    // A tracked leaf unrelated to the stack, so the probe always has a graph.
    let anchor = || Tensor::<1>::zeros([1], &device).require_grad();

    // ---- baseline: without a horizon the probe does reach the input --------
    let x0 = input(&device).require_grad();
    let (_, caches0) = layers.forward(x0.clone(), None, path(), None);
    let a0 = anchor();
    let probe0 = (0..cut).fold(a0.clone().sum(), |acc, i| acc + slot_sum(&caches0, i));
    let g0 = probe0.backward();
    assert!(a0.grad(&g0).is_some(), "the anchor should always be reachable");
    assert!(
        x0.grad(&g0).is_some(),
        "probe is vacuous: the input does not reach it even without a horizon",
    );

    // A second tracked run, left un-backwarded: `backward` consumes the graph it
    // walks, so the caches carried into the horizon run below must come from a
    // pass whose graph is still intact — otherwise the leak this is meant to
    // catch would be invisible for the wrong reason.
    let x_carry = input(&device).require_grad();
    let (_, caches_carry) = layers.forward(x_carry.clone(), None, path(), None);

    // ---- with a horizon, from zero caches ---------------------------------
    layers.grad_horizon = Some(k);
    let x1 = input(&device).require_grad();
    let (_, caches1) = layers.forward(x1.clone(), None, path(), None);
    let a1 = anchor();
    let probe1 = (0..cut).fold(a1.clone().sum(), |acc, i| acc + slot_sum(&caches1, i));
    let g1 = probe1.backward();
    assert!(a1.grad(&g1).is_some(), "the anchor should always be reachable");
    assert!(
        x1.grad(&g1).is_none(),
        "the stack input still reaches the no-grad prefix",
    );

    // ---- with a horizon, carrying caches out of a tracked run -------------
    let x2 = input(&device).require_grad();
    let (_, caches2) = layers.forward(x2.clone(), Some(caches_carry), path(), None);
    let a2 = anchor();
    let probe2 = (0..cut).fold(a2.clone().sum(), |acc, i| acc + slot_sum(&caches2, i));
    let g2 = probe2.backward();
    assert!(a2.grad(&g2).is_some(), "the anchor should always be reachable");
    assert!(
        x2.grad(&g2).is_none(),
        "the stack input still reaches the no-grad prefix (carried caches)",
    );
    assert!(
        x_carry.grad(&g2).is_none(),
        "the carried caches did not come down to the inner backend — the earlier \
         input is reachable through the prefix",
    );
}

// ===========================================================================
// 3. Prefix parameters receive no gradient
// ===========================================================================

/// With one weight set per virtual layer, a layer below the cut must receive
/// **no** gradient at all and a layer above it must receive one.
///
/// This is what discriminates a correct prefix from one that silently left a
/// parameter behind: a missed parameter still takes part in the graph and comes
/// back with `Some(grad)`. Probing several parameters per layer — at different
/// nesting depths, including the rank-1 and rank-3 ones — is deliberate, since
/// each is reached through a different path.

#[test]
fn prefix_parameters_get_no_gradient() {
    let device = Device::default().autodiff();
    let (n, k) = (4, 2);
    let cut = n - k;

    let mut layers = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);
    layers.grad_horizon = Some(k);

    let (y, _) = layers.forward(input(&device), None, path(), None);
    let grads = y.sum().backward();

    for i in 0..n {
        let l = &layers.real_layers[i];
        let probes: Vec<(&str, bool)> = vec![
            ("norm.gamma", l.norm.gamma.val().grad(&grads).is_some()),
            (
                "mamba_block.in_proj.weight",
                l.mamba_block.in_proj.weight.val().grad(&grads).is_some(),
            ),
            (
                "mamba_block.dt_bias_h",
                l.mamba_block.dt_bias_h.val().grad(&grads).is_some(),
            ),
            (
                "mamba_block.d_h",
                l.mamba_block.d_h.val().grad(&grads).is_some(),
            ),
            (
                "mamba_block.b_bias_hmr",
                l.mamba_block.b_bias_hmr.val().grad(&grads).is_some(),
            ),
            (
                "mamba_block.b_norm.gamma",
                l.mamba_block.b_norm.gamma.val().grad(&grads).is_some(),
            ),
        ];
        for (name, has_grad) in probes {
            if i < cut {
                assert!(
                    !has_grad,
                    "layer {i} (below the cut) got a gradient on {name} — \
                     detach_params did not reach it",
                );
            } else {
                assert!(has_grad, "layer {i} (above the cut) lost its {name} gradient");
            }
        }
    }
}

// ===========================================================================
// 4. Shared weights collect their tracked applications only
// ===========================================================================

/// Under weight sharing the same real layer runs on both sides of the cut. Its
/// gradient must equal that of the tracked applications alone — which is the
/// semantics that makes TRM-style recursion over a shared net work at all.
///
/// The reference is built independently: an inner-backend 4-virtual stack over
/// the same two weight sets produces the prefix activation, and a plain
/// 2-virtual stack over those same weight sets consumes it. Cyclic scheduling
/// makes the virtual→real maps line up (`0,1,0,1` then `0,1`, against the
/// 6-virtual stack's `0,1,0,1,0,1`), and cloning preserves `ParamId` and tensor
/// identity, so gradients on the clones are the gradients on the original.
#[test]
fn shared_weight_grad_counts_tracked_applications_only() {
    let device = Device::default().autodiff();
    let (n_real, n_virtual, k) = (2, 6, 2);

    let mut layers = LayersBuilder::new(n_real, block_config(D_MODEL))
        .with_n_virtual_layers(Some((n_virtual, Schedule::Cyclic)))
        .init(&device);
    layers.grad_horizon = Some(k);

    let x = input(&device);
    let (y, _) = layers.forward(x.clone(), None, path(), None);
    let grads = y.clone().sum().backward();

    // Reference: inner-backend 4-virtual prefix, then a tracked 2-virtual
    // suffix, both over the very same weight sets.
    let mut prefix = <Layers<_> as Clone>::clone(&layers);
    prefix.n_virtual_layers = Some((n_virtual - k, Schedule::Cyclic));
    prefix.grad_horizon = None;
    let prefix: Layers<_> = burn::module::AutodiffModule::valid(&prefix);

    let mut suffix = <Layers<_> as Clone>::clone(&layers);
    suffix.n_virtual_layers = Some((k, Schedule::Cyclic));
    suffix.grad_horizon = None;

    let (h, _) = prefix.forward(x.inner(), None, path(), None);
    let (y_ref, _) = suffix.forward(Tensor::from_inner(h), None, path(), None);
    assert!(
        max_abs_diff(y.clone(), y_ref.clone()) < 1e-5,
        "the cut stack and the hand-split reference disagree on values",
    );

    let grads_ref = y_ref.sum().backward();
    for r in 0..n_real {
        let w = &layers.real_layers[r].mamba_block.in_proj.weight;
        let got = w.val().grad(&grads).expect("shared weight grad");
        let want = w
            .val()
            .grad(&grads_ref)
            .expect("shared weight grad (reference)");
        assert!(
            max_abs_diff(got, want) < 1e-5,
            "real layer {r} did not collect exactly its tracked applications",
        );
    }
}

// ===========================================================================
// 5. Inert off the autodiff backend
// ===========================================================================

/// A horizon set on a model running without autodiff must be inert, not fatal.
///
/// The mechanism is `Tensor::inner` / `AutodiffModule::valid`, and both **panic**
/// on something that is already off the autodiff backend — unlike `detach`, which
/// is a documented no-op there. So the cut is guarded on
/// `Device::is_autodiff`, and inference with a horizon left set in the config has
/// to come out identical to no horizon at all.
#[test]
fn horizon_is_inert_without_autodiff() {
    let device = Device::default();
    assert!(!device.is_autodiff(), "this test needs a plain device");
    let n = 4;

    let plain = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);
    let mut with_horizon = <Layers<_> as Clone>::clone(&plain);
    with_horizon.grad_horizon = Some(2);

    let x = input(&device);
    let (y_a, _) = plain.forward(x.clone(), None, path(), None);
    let (y_b, _) = with_horizon.forward(x, None, path(), None);
    assert!(
        max_abs_diff(y_a, y_b) < 1e-6,
        "a horizon changed the result on a non-autodiff device",
    );
}

// ===========================================================================
// 6. Chunked forward equals one long forward
// ===========================================================================

/// Splitting a sequence into two `forward` calls, carrying the first call's final
/// caches into the second, must equal one call over the whole sequence — on the
/// output, on every final cache tensor, **and** on the parameter gradients.
///
/// The gradient half is the load-bearing one for `grad_horizon`. The caches
/// handed between calls straddle the cut: the slots below it come back lifted out
/// of the inner backend and must carry **no** gradient, while the slots above it
/// stay tracked and must carry gradient *across the boundary* — otherwise the
/// split version silently trains on a truncated history. An implementation that
/// detached the whole cache set (rather than only the prefix's share) still gets
/// identical outputs and caches here, and is caught only by the gradients.
///
/// The loss weights the output and every final cache tensor by a fixed random
/// head, so a sign or ordering error cannot cancel out the way a plain `sum`
/// would let it.
fn run_chunked_parity(horizon: Option<usize>) {
    let device = Device::default().autodiff();
    let (n_real, n_virtual) = (2, 6);
    // Split on a chunk boundary (`path()` uses chunk_len 4) so the two runs do
    // the same arithmetic in the same order and any mismatch is structural.
    let (head_len, tail_len) = (4, 4);

    let mut layers = LayersBuilder::new(n_real, block_config(D_MODEL))
        .with_n_virtual_layers(Some((n_virtual, Schedule::Cyclic)))
        .init(&device);
    layers.grad_horizon = horizon;

    let x = Tensor::<3>::random(
        [BATCH, head_len + tail_len, D_MODEL],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // ---- one long call ----------------------------------------------------
    let (y_full, caches_full) = layers.forward(x.clone(), None, path(), None);

    // Fixed random heads, shaped from the full run and reused by both, so the
    // two losses are the same functional of the same quantities.
    let normal = Distribution::Normal(0.0, 1.0);
    let y_head = Tensor::<3>::random(y_full.dims(), normal, &device);
    let flat_full = all_slots_flat(&caches_full, n_virtual);
    let cache_heads: Vec<Tensor<1>> = flat_full
        .iter()
        .map(|t| Tensor::<1>::random(t.dims(), normal, &device))
        .collect();

    let loss = |y: Tensor<3>, flat: &[Tensor<1>]| {
        flat.iter().zip(&cache_heads).fold(
            (y * y_head.clone()).sum(),
            |acc, (t, h)| acc + (t.clone() * h.clone()).sum(),
        )
    };
    let grads_full = loss(y_full.clone(), &flat_full).backward();

    // ---- two chunked calls, carrying the cache ----------------------------
    let head = x.clone().slice([0..BATCH, 0..head_len]);
    let tail = x.slice([0..BATCH, head_len..head_len + tail_len]);
    let (y1, mid) = layers.forward(head, None, path(), None);
    let (y2, caches_split) = layers.forward(tail, Some(mid), path(), None);
    let y_split = Tensor::cat(vec![y1, y2], 1);
    let flat_split = all_slots_flat(&caches_split, n_virtual);

    assert!(
        max_abs_diff(y_full, y_split.clone()) < 1e-5,
        "chunked output differs from the single call (horizon {horizon:?})",
    );
    for (i, (a, b)) in flat_full.iter().zip(&flat_split).enumerate() {
        assert!(
            max_abs_diff(a.clone(), b.clone()) < 1e-5,
            "chunked final cache tensor {i} differs (horizon {horizon:?})",
        );
    }

    let grads_split = loss(y_split, &flat_split).backward();
    for r in 0..n_real {
        let l = &layers.real_layers[r];
        let probes: Vec<(&str, &burn::module::Param<Tensor<2>>)> =
            vec![("mamba_block.in_proj.weight", &l.mamba_block.in_proj.weight)];
        for (name, p) in probes {
            let a = p.val().grad(&grads_full).expect("single-call grad");
            let b = p.val().grad(&grads_split).expect("chunked grad");
            assert!(
                max_abs_diff(a, b) < 1e-4,
                "real layer {r}: {name} gradient differs between the chunked and \
                 the single call (horizon {horizon:?}) — the cache handed between \
                 calls is not carrying gradient the way one continuous call does",
            );
        }
        for (name, p) in [
            ("norm.gamma", &l.norm.gamma),
            ("mamba_block.dt_bias_h", &l.mamba_block.dt_bias_h),
            ("mamba_block.d_h", &l.mamba_block.d_h),
        ] {
            let a = p.val().grad(&grads_full).expect("single-call grad");
            let b = p.val().grad(&grads_split).expect("chunked grad");
            assert!(
                max_abs_diff(a, b) < 1e-4,
                "real layer {r}: {name} gradient differs between the chunked and \
                 the single call (horizon {horizon:?})",
            );
        }
    }
}

/// Control: the parity must already hold without a horizon, so a failure under
/// one is attributable to the cut rather than to the harness.
#[test]
fn chunked_forward_matches_single_forward() {
    run_chunked_parity(None);
}

/// The same parity with the stack cut: the prefix caches round-trip through the
/// inner backend between calls, the suffix caches carry gradient across the
/// boundary, and the result must be indistinguishable from one long call.
#[test]
fn chunked_forward_matches_single_forward_under_horizon() {
    run_chunked_parity(Some(2));
}

// ===========================================================================
// 7. step / prime honour the same cut as forward
// ===========================================================================

/// `step` unrolled token-by-token must equal one `forward` over the same
/// sequence — outputs, final caches, and parameter gradients — with the horizon
/// set on both.
///
/// This is the guarantee the crate already makes for the uncut stack, extended
/// across the cut: `step`'s cascade walks *depth* per token while `forward` walks
/// *sequence* per layer, so the two only agree if the cut falls on the same
/// virtual layers in both, and if what crosses it is lowered and lifted at the
/// same boundary. A `step` that ignored `grad_horizon` (or cut at a different
/// index) still matches on values — every path computes the same numbers — and
/// diverges only on the gradients.
fn run_step_parity(horizon: Option<usize>) {
    let device = Device::default().autodiff();
    let (n_real, n_virtual, seq) = (2, 6, 4);

    let mut layers = LayersBuilder::new(n_real, block_config(D_MODEL))
        .with_n_virtual_layers(Some((n_virtual, Schedule::Cyclic)))
        .init(&device);
    layers.grad_horizon = horizon;

    let x = Tensor::<3>::random(
        [BATCH, seq, D_MODEL],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    // ---- one forward ------------------------------------------------------
    let (y_fwd, caches_fwd) = layers.forward(x.clone(), None, path(), None);

    let normal = Distribution::Normal(0.0, 1.0);
    let y_head = Tensor::<3>::random(y_fwd.dims(), normal, &device);
    let flat_fwd = all_slots_flat(&caches_fwd, n_virtual);
    let cache_heads: Vec<Tensor<1>> = flat_fwd
        .iter()
        .map(|t| Tensor::<1>::random(t.dims(), normal, &device))
        .collect();
    let loss = |y: Tensor<3>, flat: &[Tensor<1>]| {
        flat.iter().zip(&cache_heads).fold(
            (y * y_head.clone()).sum(),
            |acc, (t, h)| acc + (t.clone() * h.clone()).sum(),
        )
    };
    let grads_fwd = loss(y_fwd.clone(), &flat_fwd).backward();

    // ---- the same sequence, one token at a time ---------------------------
    let mut caches = None;
    let mut outs: Vec<Tensor<3>> = Vec::with_capacity(seq);
    for t in 0..seq {
        let token = x.clone().slice([0..BATCH, t..t + 1]).squeeze::<2>();
        let (out, c) = layers.step(token, caches, None);
        outs.push(out.unsqueeze_dim::<3>(1));
        caches = Some(c);
    }
    let y_step = Tensor::cat(outs, 1);
    let caches_step = caches.expect("at least one step ran");
    let flat_step = all_slots_flat(&caches_step, n_virtual);

    assert!(
        max_abs_diff(y_fwd, y_step.clone()) < 1e-5,
        "stepped output differs from forward (horizon {horizon:?})",
    );
    for (i, (a, b)) in flat_fwd.iter().zip(&flat_step).enumerate() {
        assert!(
            max_abs_diff(a.clone(), b.clone()) < 1e-5,
            "stepped final cache tensor {i} differs (horizon {horizon:?})",
        );
    }

    let grads_step = loss(y_step, &flat_step).backward();
    for r in 0..n_real {
        let l = &layers.real_layers[r];
        let a = l
            .mamba_block
            .in_proj
            .weight
            .val()
            .grad(&grads_fwd)
            .expect("forward grad");
        let b = l
            .mamba_block
            .in_proj
            .weight
            .val()
            .grad(&grads_step)
            .expect("step grad");
        assert!(
            max_abs_diff(a, b) < 1e-4,
            "real layer {r}: in_proj gradient differs between step and forward \
             (horizon {horizon:?}) — the two do not cut at the same layers",
        );
        for (name, p) in [
            ("norm.gamma", &l.norm.gamma),
            ("mamba_block.dt_bias_h", &l.mamba_block.dt_bias_h),
            ("mamba_block.d_h", &l.mamba_block.d_h),
        ] {
            let a = p.val().grad(&grads_fwd).expect("forward grad");
            let b = p.val().grad(&grads_step).expect("step grad");
            assert!(
                max_abs_diff(a, b) < 1e-4,
                "real layer {r}: {name} gradient differs between step and forward \
                 (horizon {horizon:?})",
            );
        }
    }
}

/// Control: step/forward parity already holds without a horizon.
#[test]
fn step_matches_forward() {
    run_step_parity(None);
}

/// The same parity with the stack cut, which is what pins `step`'s cascade to
/// `forward`'s cut.
#[test]
fn step_matches_forward_under_horizon() {
    run_step_parity(Some(2));
}

/// Below the cut `step` must leave no gradient behind either — the same
/// assertion as for `forward`, but reached through the cascade, which walks depth
/// per token rather than sequence per layer.
#[test]
fn stepped_prefix_parameters_get_no_gradient() {
    let device = Device::default().autodiff();
    let (n, k) = (4, 2);
    let cut = n - k;

    let mut layers = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);
    layers.grad_horizon = Some(k);

    let x = Tensor::<2>::random([BATCH, D_MODEL], Distribution::Normal(0.0, 1.0), &device);
    let (out, _) = layers.step(x, None, None);
    let grads = out.sum().backward();

    for i in 0..n {
        let l = &layers.real_layers[i];
        for (name, has) in [
            ("norm.gamma", l.norm.gamma.val().grad(&grads).is_some()),
            (
                "mamba_block.in_proj.weight",
                l.mamba_block.in_proj.weight.val().grad(&grads).is_some(),
            ),
            (
                "mamba_block.b_bias_hmr",
                l.mamba_block.b_bias_hmr.val().grad(&grads).is_some(),
            ),
        ] {
            if i < cut {
                assert!(!has, "stepped layer {i} (below the cut) got a {name} gradient");
            } else {
                assert!(has, "stepped layer {i} (above the cut) lost its {name} gradient");
            }
        }
    }
}

/// `prime` under a cut, in both directions across the boundary.
///
/// `prime` opens with no user token at all, so the two interesting shapes are a
/// latent **above** the cut (every layer below it runs on an empty stream and is
/// skipped entirely) and one **below** it (the emitted row is created on the
/// inner backend and has to be lifted at the boundary before any tracked layer
/// touches it — without that lift the tracked half meets an inner tensor and
/// Burn panics on a backend mismatch).
///
/// Either way the primed output is produced by tracked layers, so it must carry
/// a gradient to those and to nothing below the cut.
fn run_prime_cut(latent_layer: usize) {
    use crate::utils::class::init_class_emb;
    use crate::utils::{ClassCursors, ClassLatent};

    let device = Device::default().autodiff();
    let (n, k) = (4, 2);
    let cut = n - k;

    let mut layers = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);
    layers.grad_horizon = Some(k);
    layers.real_layers[latent_layer].class_latents = vec![ClassLatent::Start];
    layers.real_layers[latent_layer].class_latents_emb = init_class_emb(1, D_MODEL, &device);

    let mut class = ClassCursors::new(SEQ);
    let (primed, _caches) = layers.prime(BATCH, None, Some(&mut class));
    let primed = primed.expect("a latent was waiting");

    // It came out of a tracked layer, so it must still carry a gradient back to
    // that layer's parameters — and to nothing below the cut.
    let grads = primed.sum().backward();
    for i in 0..n {
        let has = layers.real_layers[i]
            .mamba_block
            .in_proj
            .weight
            .val()
            .grad(&grads)
            .is_some();
        if i < cut {
            assert!(
                !has,
                "primed layer {i} (below the cut) got a gradient \
                 (latent on layer {latent_layer})",
            );
        }
    }
    assert!(
        layers.real_layers[n - 1]
            .mamba_block
            .in_proj
            .weight
            .val()
            .grad(&grads)
            .is_some(),
        "the top layer lost its gradient (latent on layer {latent_layer})",
    );
    // The latent itself trains from either side of the cut — below it that is
    // the cascade's ghost row doing the work.
    assert!(
        layers.real_layers[latent_layer]
            .class_latents_emb
            .as_ref()
            .unwrap()
            .val()
            .grad(&grads)
            .is_some(),
        "the class latent on layer {latent_layer} never trains",
    );
}

/// Latent above the cut: every layer below it is skipped on an empty stream.
#[test]
fn prime_under_a_cut_with_a_latent_above_it() {
    run_prime_cut(3);
}

/// Latent below the cut: the row is created on the inner backend and has to
/// cross the boundary before any tracked layer sees it.
#[test]
fn prime_under_a_cut_with_a_latent_below_it() {
    run_prime_cut(0);
}

// ===========================================================================
// 8. The cut keeps the input's gradient (straight-through)
// ===========================================================================

/// A cut must not sever the gradient to whatever produced the stack input.
///
/// `x` enters `Layers` once, at the bottom, and rides the residual stream upward
/// — so a bottom cut would otherwise sever its only path and a network's
/// `in_proj` (or a vocab net's embedding) would **never train**, silently. TRM
/// and HRM never hit this because they re-inject `x` at every recursion; this
/// stack reads it once, so the boundary re-attaches it by straight-through.
///
/// Checked on a whole `LatentNetwork`, since `in_proj` never training is the
/// consequence that actually matters.
#[test]
fn boundary_weights_keep_their_gradient_under_a_cut() {
    use crate::modules::network::LatentNetworkBuilder;

    let device = Device::default().autodiff();
    let n = 4;
    for horizon in [None, Some(4), Some(2), Some(0)] {
        let mut lb = LayersBuilder::new(n, block_config(D_MODEL));
        lb.grad_horizon = horizon;
        let net = LatentNetworkBuilder {
            input_size: 3,
            layers: lb,
            output_size: 3,
            final_norm: false,
            class_tokens: Vec::new(),
        }
        .init(&device);

        let x = Tensor::<3>::random([BATCH, SEQ, 3], Distribution::Normal(0.0, 1.0), &device)
            .require_grad();
        let (y, _) = net.forward(x.clone(), None, path(), None);
        let grads = y.sum().backward();

        assert!(
            x.grad(&grads).is_some(),
            "the network input lost its gradient (horizon {horizon:?})",
        );
        assert!(
            net.in_proj.weight.val().grad(&grads).is_some(),
            "in_proj would never train (horizon {horizon:?})",
        );
        assert!(
            net.out_proj.weight.val().grad(&grads).is_some(),
            "out_proj lost its gradient (horizon {horizon:?})",
        );
    }
}

/// The same for `step`: the carry rides the cascade too, so a stack trained by
/// stepping keeps its boundary weights alive.
#[test]
fn stepped_input_keeps_its_gradient_under_a_cut() {
    let device = Device::default().autodiff();
    let (n, k) = (4, 2);

    let mut layers = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);
    layers.grad_horizon = Some(k);

    let x = Tensor::<2>::random([BATCH, D_MODEL], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();
    let (out, _) = layers.step(x.clone(), None, None);
    assert!(
        x.grad(&out.sum().backward()).is_some(),
        "step severed the input's gradient across the cut",
    );
}

/// The straight-through adds a tensor that is **exactly zero**, so a cut may
/// change gradients and must change nothing else.
///
/// Run across the whole range including `Some(0)` (nothing differentiated at
/// all), which is the case that skips the in-loop boundary entirely.
#[test]
fn a_cut_changes_gradients_only() {
    let device = Device::default().autodiff();
    let n = 4;

    let plain = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);
    let x = input(&device);
    let (want, want_caches) = plain.forward(x.clone(), None, path(), None);
    let want_flat = all_slots_flat(&want_caches, n);

    for horizon in [Some(4), Some(3), Some(1), Some(0)] {
        let mut cut = <Layers<_> as Clone>::clone(&plain);
        cut.grad_horizon = horizon;
        let (got, got_caches) = cut.forward(x.clone(), None, path(), None);
        assert!(
            max_abs_diff(want.clone(), got) < 1e-6,
            "horizon {horizon:?} changed the output",
        );
        for (i, (a, b)) in want_flat.iter().zip(&all_slots_flat(&got_caches, n)).enumerate() {
            assert!(
                max_abs_diff(a.clone(), b.clone()) < 1e-6,
                "horizon {horizon:?} changed final cache tensor {i}",
            );
        }
    }
}

/// Class latents on a layer **below** the cut lengthen the sequence, so the
/// straight-through carry has to grow with it — taking a **ghost** row (value
/// zero, but tracked) at each latent position.
///
/// Two things ride on that. The lengths must agree or the boundary add fails
/// outright; and the ghost is what keeps the latent itself trainable, since it is
/// a learnable *input row* rather than part of the layer's transform. The layer's
/// actual weights stay undifferentiated, which is what the last assertion
/// separates.
#[test]
fn the_carry_tracks_class_latents_spliced_below_the_cut() {
    use crate::utils::class::init_class_emb;
    use crate::utils::{ClassCursors, ClassLatent};

    let device = Device::default().autodiff();
    let (n, k) = (4, 2);

    let mut layers = LayersBuilder::new(n, block_config(D_MODEL)).init(&device);
    layers.grad_horizon = Some(k);
    // Two prefix layers each splice a row, so the boundary sequence is longer
    // than the input by two.
    for i in 0..(n - k) {
        layers.real_layers[i].class_latents = vec![ClassLatent::Start];
        layers.real_layers[i].class_latents_emb = init_class_emb(1, D_MODEL, &device);
    }

    let x = input(&device).require_grad();
    let mut class = ClassCursors::new(SEQ);
    let (y, _) = layers.forward(x.clone(), None, path(), Some(&mut class));
    assert_eq!(y.dims()[1], SEQ + (n - k), "each prefix layer splices one row");

    let grads = y.sum().backward();
    assert!(
        x.grad(&grads).is_some(),
        "the input lost its gradient with latents spliced below the cut",
    );
    for i in 0..(n - k) {
        let l = &layers.real_layers[i];
        assert!(
            l.class_latents_emb
                .as_ref()
                .unwrap()
                .val()
                .grad(&grads)
                .is_some(),
            "prefix layer {i}'s class latent is a learnable input row and must \
             train — a dead parameter here would be silent",
        );
        assert!(
            l.mamba_block.in_proj.weight.val().grad(&grads).is_none(),
            "prefix layer {i}'s transform must stay undifferentiated",
        );
    }
}

/// With an **identity prefix**, the straight-through is not an approximation and
/// the cut must reproduce full backpropagation on the input's gradient exactly.
///
/// Zeroing a prefix layer's `out_proj` makes its Jacobian w.r.t. its input
/// exactly zero, so the residual stream carries the input through untouched and
/// the true prefix Jacobian *is* `I` — precisely what the carry asserts. Values
/// are unaffected by the cut either way, so `dL/dx_boundary` is the same in both
/// runs and the two gradients must agree to float noise.
///
/// This is what pins the MultiGate correction. Under `Standard` there is one
/// carrier and correcting the token is the whole story, so the test would pass
/// with or without it; under `MultiGate` the streams carry the residual, and a
/// correction applied only to the pooled token leaves `dL/ds` out of the input's
/// gradient and the two runs diverge.
fn run_identity_prefix_exactness(residuals: crate::modules::ResidualsConfig) {
    use crate::modules::Residuals;

    let device = Device::default().autodiff();
    let (n, k) = (4, 2);
    let cut = n - k;

    let mut base = LayersBuilder::new(n, block_config(D_MODEL))
        .with_residuals(residuals)
        .init(&device);
    // Prefix layers become exact identities on the residual stream: a zero
    // `out_proj` weight makes their Jacobian w.r.t. their input exactly zero, so
    // the stream passes through untouched (the value may shift by a constant —
    // only the Jacobian matters here).
    for i in 0..cut {
        let w = &mut base.real_layers[i].mamba_block.out_proj.weight;
        *w = w.clone().map(|t| t.zeros_like());
    }
    // Under MultiGate the residual is in the streams, so the prefix is an
    // identity only if its gates carry rather than mix. Biasing *only* the
    // prefix's gates shut leaves the suffix mixing normally — otherwise the whole
    // stack would be near-identity and the comparison would hold vacuously.
    if let Residuals::MultiGate(mg) = &mut base.residuals {
        for i in 0..cut {
            let b = &mut mg.layers[i].b_beta;
            *b = b.clone().map(|t| t.zeros_like() - 20.0);
        }
    }

    let x = input(&device);
    let head = Tensor::<3>::random(
        [BATCH, SEQ, D_MODEL],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    // The *same* weights both times — only the horizon differs.
    let grad_of = |horizon| {
        let mut layers = <Layers<_> as Clone>::clone(&base);
        layers.grad_horizon = horizon;
        let xr = x.clone().require_grad();
        let (y, _) = layers.forward(xr.clone(), None, path(), None);
        let grads = (y * head.clone()).sum().backward();
        xr.grad(&grads).expect("input gradient")
    };

    let full = grad_of(None);
    let cut_grad = grad_of(Some(k));
    let diff = max_abs_diff(full.clone(), cut_grad);
    // Guard against a vacuous pass: if the stack were near-identity throughout,
    // the gradient would be ~1 everywhere and any wiring would "agree".
    let scale = full.abs().max().into_scalar::<f32>();
    assert!(
        scale > 1e-3,
        "degenerate fixture: the input gradient is ~0, so the comparison proves \
         nothing (max |grad| = {scale})",
    );
    assert!(
        diff < 1e-5 * scale.max(1.0),
        "with an identity prefix the cut must reproduce full backprop on the \
         input's gradient — a carrier is missing its identity path \
         (max abs diff {diff}, grad scale {scale})",
    );
}

/// One carrier: correcting the token is the whole story.
#[test]
fn identity_prefix_is_exact_under_standard_residuals() {
    run_identity_prefix_exactness(crate::modules::ResidualsConfig::Standard);
}

/// Several carriers: the streams hold the residual, so they need the carry too.
/// `n_stream: 1` with a strongly carry-biased gate keeps the prefix an identity
/// (no accumulation phase, gates closed), which is what makes the comparison
/// exact rather than merely close.
#[test]
fn identity_prefix_is_exact_under_multi_gate_residuals() {
    run_identity_prefix_exactness(crate::modules::ResidualsConfig::MultiGate {
        n_stream: 1,
        init_bias: -20.0,
        init_bias_step: 0.0,
        per_virtual_layer: false,
    });
}

/// The input keeps its gradient under `MultiGate` residuals too.
///
/// Complements the exactness test above with the ordinary configuration —
/// several streams, default gates, so the accumulation phase runs — where the
/// carry has to reach both the pooled token and the stream set.
#[test]
fn multi_gate_input_keeps_its_gradient_under_a_cut() {
    use crate::modules::ResidualsConfig;

    let device = Device::default().autodiff();
    let n = 4;
    for horizon in [None, Some(2)] {
        let mut layers = LayersBuilder::new(n, block_config(D_MODEL))
            .with_residuals(ResidualsConfig::MultiGate {
                n_stream: 3,
                init_bias: 0.0,
                init_bias_step: 0.0,
                per_virtual_layer: false,
            })
            .init(&device);
        layers.grad_horizon = horizon;

        let x = input(&device).require_grad();
        let (y, _) = layers.forward(x.clone(), None, path(), None);
        assert!(
            x.grad(&y.sum().backward()).is_some(),
            "MultiGate severed the input's gradient (horizon {horizon:?})",
        );
    }
}

// ===========================================================================
// Memory probe (ignored by default)
// ===========================================================================

/// Peak-RSS probe for the open question of whether the horizon actually saves
/// memory here, and how much the *untracked bookkeeping* costs.
///
/// Untracked ops are still registered in the graph — Burn keeps an
/// `UntrackedOpsStep` per op so a memory-bound op can still retrieve an untracked
/// parent — but with **unit state**, so no activation is retained. This measures
/// what that leaves behind. Peak RSS is process-wide, so each configuration must
/// be its own process: run it as
///
/// ```bash
/// for nv in 8 32 64; do for k in none 2; do
///   BURN_MAMBA_N_VIRTUAL=$nv BURN_MAMBA_GRAD_HORIZON=$k \
///     cargo test --lib grad_horizon_memory -- --ignored --nocapture
/// done; done
/// ```
///
/// `BURN_MAMBA_DETACHED=1` swaps in the detach-only prefix (the mechanism that
/// does not work) and `BURN_MAMBA_PLAIN=1` drops autodiff entirely, for the two
/// reference curves.
#[test]
#[ignore = "peak-RSS probe: one process per configuration, see the doc comment"]
fn grad_horizon_memory_probe() {
    // Sized so activations, not the process baseline, dominate peak RSS.
    let (n_real, d_model, batch, seq) = (2, 128, 4, 512);
    let n_virtual: usize = std::env::var("BURN_MAMBA_N_VIRTUAL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let horizon = match std::env::var("BURN_MAMBA_GRAD_HORIZON").as_deref() {
        Ok("none") | Err(_) => None,
        Ok(k) => Some(k.parse().expect("BURN_MAMBA_GRAD_HORIZON: 'none' or an integer")),
    };

    // `plain` is the control: the same forward with no autodiff device at all
    // and no backward, so whatever RSS it uses is the forward working set rather
    // than anything the graph retains.
    let plain = std::env::var("BURN_MAMBA_PLAIN").is_ok();
    let device = if plain {
        Device::default()
    } else {
        Device::default().autodiff()
    };
    let mut layers = LayersBuilder::new(n_real, block_config(d_model))
        .with_n_virtual_layers(Some((n_virtual, Schedule::Cyclic)))
        .init(&device);
    layers.grad_horizon = horizon;

    let x = Tensor::<3>::random(
        [batch, seq, d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let x = if plain { x } else { x.require_grad() };
    // `detached` is the mechanism `grad_horizon` does **not** use: a
    // parameter-detached prefix on the same (autodiff) device. Kept as the
    // contrast that justifies the inner-backend one.
    let detached = std::env::var("BURN_MAMBA_DETACHED").is_ok();
    let y = if detached {
        let k = horizon.expect("BURN_MAMBA_DETACHED needs a horizon");
        let mut prefix = <Layers<_> as Clone>::clone(&layers);
        prefix.n_virtual_layers = Some((n_virtual - k, Schedule::Cyclic));
        prefix.grad_horizon = None;
        let prefix = crate::utils::detach_params(prefix);

        let mut suffix = <Layers<_> as Clone>::clone(&layers);
        suffix.n_virtual_layers = Some((k, Schedule::Cyclic));
        suffix.grad_horizon = None;

        let (h, _) = prefix.forward(x.detach(), None, Mamba3SsdPath::default(), None);
        let (y, _) = suffix.forward(h, None, Mamba3SsdPath::default(), None);
        y
    } else {
        let (y, _) = layers.forward(x, None, Mamba3SsdPath::default(), None);
        y
    };
    if plain {
        let _ = y.sum();
    } else {
        let _ = y.sum().backward();
    }

    let hwm = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .map(|l| l.trim_start_matches("VmHWM:").trim().to_string())
        })
        .unwrap_or_else(|| "unavailable".into());
    eprintln!(
        "grad_horizon = {horizon:?}{}  |  n_virtual = {n_virtual}, d_model = {d_model}, \
         batch = {batch}, seq = {seq}  |  peak RSS = {hwm}",
        if plain {
            " (plain, no autodiff)"
        } else if detached {
            " (detached-params prefix)"
        } else {
            " (inner-backend prefix, as grad_horizon runs it)"
        }
    );
}
