//! The three claims this rung rests on, measured — including the one that makes
//! it the ladder's modest rung.
//!
//! 1. A **hand-built** block solves the task exactly: the Kalman gate *is* the
//!    filter the labels come from, so the block computes the optimal estimate
//!    rather than approximating it.
//! 2. The **same block with `κ = 0`** — the scaled member, whose decay is
//!    projected — cannot, at any gap decay on a grid: a geometric discount is
//!    not a hyperbolic one.
//! 3. The **honest bound**: a stock block may spend a second head on a count and
//!    let its per-token gate cross-multiply, deciding `v·n − Σ > 0` without ever
//!    dividing. That is a linear state with a nonlinear readout, and it comes
//!    within a few points. The Kalman gate's exactness is real; the *decision*
//!    gap it buys on this task is small, and this test is what says so.

use crate::dataset::{
    FAMILIES, GAP, MARGIN, NUM_SYMBOLS, NUM_VALUES, Q_GAP, SEQ_LENGTH, labels, posterior, task,
    value,
};
use crate::model::{NUM_CLASSES, model_config};
use crate::shared::data::{IGNORE, TallyDataset};
use crate::shared::handmade::{accuracy, affine_channels, memoryless_ceiling, param, softplus_inv};
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

type Device = burn::prelude::Device;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// The gate's `κ`, so that `q = Δ·exp(r)` and `r` alone names the doubt.
const KAPPA: f64 = 1.0;
/// The estimator's `Δ` on a gap: evidence worth (almost) nothing, which is what
/// lets the same token carry doubt worth `Q_GAP`.
const DELTA_GAP: f64 = 1e-4;
/// `r` where no time passes: `q = Δ·e^r` underflows to zero.
const R_QUIET: f64 = -40.0;
/// A `Δ` small enough that a head's state never leaves zero.
const DELTA_OFF: f64 = 1e-12;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// `Â` for a held state: `A = −softplus(Â)` lands under the block's `a_floor`.
const A_HOLD_RAW: f64 = -20.0;
/// Class-logit gain.
const OUT_GAIN: f64 = 3.0;

/// The symbol embeddings: the 33 values on a circle of radius `√2` at height
/// `1` (so the value is an affine read), the gap at the far pole. Norm `√3`, so
/// the layer's pre-`RmsNorm` passes them through.
fn embeddings() -> Vec<Vec<f64>> {
    let r = 2.0f64.sqrt();
    let mut out: Vec<Vec<f64>> = (0..NUM_VALUES)
        .map(|i| {
            let cos = value(i).unwrap() / 2.0;
            let sin = (1.0 - cos * cos).max(0.0).sqrt();
            vec![r * cos, r * sin, 1.0]
        })
        .collect();
    out.push(vec![0.0, 0.0, -(3.0f64.sqrt())]);
    out
}

/// Which arm to build.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// The rung's own construction: the gate computes the discount.
    Kalman,
    /// The join at `κ = 0`: the same block, the same read of `η/Λ`, but the
    /// decay is projected — the gap discounts geometrically by `alpha_gap`.
    Scaled { alpha_gap: f64 },
}

fn handmade(device: &Device, arm: Arm) -> MambaLatentNet {
    let cfg = model_config(true);
    let mut model = crate::common::model::ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("tally-drift configures the Mamba-3 variant")
    };
    let embeddings = embeddings();
    let values: Vec<f64> = (0..NUM_SYMBOLS).map(|s| value(s).unwrap_or(0.0)).collect();
    let per_symbol = |f: &dyn Fn(usize) -> f64| (0..NUM_SYMBOLS).map(f).collect::<Vec<f64>>();

    net.in_proj.weight = param(&embeddings.concat(), [NUM_SYMBOLS, 3], device);
    net.in_proj.bias = Some(param(&[0.0; 3], [3], device));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([3]), device));
    let block = &mut layer.block;

    // Channel order: `[z(3) | x(3) | B(1) | C(1) | Δ(3) | A(3) | r(3)]` — the
    // last three are the Kalman gate's projected noise.
    let hold = vec![A_HOLD_RAW; NUM_SYMBOLS];
    let off = vec![softplus_inv(DELTA_OFF); NUM_SYMBOLS];
    let mut targets: Vec<Vec<f64>> = vec![vec![Z_PRE; NUM_SYMBOLS]; 3]; // z
    // head 0 writes the value (nothing on a gap); head 1 is the reference.
    targets.push(per_symbol(&|s| values[s]));
    targets.push(vec![1.0; NUM_SYMBOLS]);
    targets.push(vec![0.0; NUM_SYMBOLS]);
    targets.push(vec![1.0; NUM_SYMBOLS]); // B
    targets.push(vec![1.0; NUM_SYMBOLS]); // C
    // Δ: one unit of evidence per value, almost none on a gap.
    targets.push(per_symbol(&|s| {
        softplus_inv(if s == GAP { DELTA_GAP } else { 1.0 })
    }));
    targets.push(off.clone()); // Δ, head 1
    targets.push(off); // Δ, head 2
    match arm {
        Arm::Kalman => {
            targets.push(hold.clone()); // A, head 0 — a held sum
        }
        Arm::Scaled { alpha_gap } => {
            // The discount has to come from the decay instead: `α = exp(Δ·A)`
            // with the gap's own `Δ`.
            targets.push(per_symbol(&|s| {
                if s == GAP {
                    softplus_inv((-alpha_gap.max(1e-9).ln() / DELTA_GAP).max(1e-9))
                } else {
                    A_HOLD_RAW
                }
            }));
        }
    }
    targets.push(hold.clone()); // A, head 1
    targets.push(hold); // A, head 2
    // `r`: doubt worth `Q_GAP` on a gap, none anywhere else.
    targets.push(per_symbol(&|s| {
        if s == GAP {
            (Q_GAP / (KAPPA * DELTA_GAP)).ln()
        } else {
            R_QUIET
        }
    }));
    targets.push(vec![R_QUIET; NUM_SYMBOLS]); // r, head 1
    targets.push(vec![R_QUIET; NUM_SYMBOLS]); // r, head 2

    let (weight, bias) = affine_channels(&embeddings, &targets, device);
    block.in_proj.weight = weight;
    block.in_proj.bias = Some(bias);

    block.dt_bias_h = param(&[0.0; 3], [3], device);
    block.d_h = param(&[1.0, 1.0, 0.0], [3], device);
    block.b_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([1]), device));
    block.c_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([1]), device));
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([3, 1, 1]), device));
    // `C = −1` on the estimator: its estimate is subtracted from the value.
    block.c_bias_hmr = param(&[-2.0, 0.0, 0.0], [3, 1, 1], device);
    // `κ`: one, or zero — the join where the block is stock's decay again.
    block.kalman_log_kappa_h = Some(param(
        &match arm {
            Arm::Kalman => [KAPPA.ln(), R_QUIET, R_QUIET],
            Arm::Scaled { .. } => [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
        },
        [3],
        device,
    ));
    // `ω = 1` on the estimator: it reads `η/Λ`, the estimate, not the sum.
    block.kalman_read_h = Some(param(&[1.0, 0.0, 0.0], [3], device));

    block.out_proj.weight = param(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [3, 3],
        device,
    );
    block.out_proj.bias = Some(param(&[0.0; 3], [3], device));

    let norm_f = net.norm_f.as_mut().expect("final_norm is on");
    norm_f.gamma = Param::from_tensor(Tensor::ones(Shape::new([3]), device));
    net.out_proj.weight = param(
        &[-OUT_GAIN, OUT_GAIN, 0.0, 0.0, 0.0, 0.0],
        [3, NUM_CLASSES],
        device,
    );
    net.out_proj.bias = Some(param(&[0.0; NUM_CLASSES], [NUM_CLASSES], device));
    model
}

fn worst_family(model: &MambaLatentNet, count: usize, device: &Device) -> (f64, Vec<f64>) {
    let task = task();
    let accs: Vec<f64> = FAMILIES
        .iter()
        .map(|(_, generator)| accuracy(model, &task, *generator, count, 0xE7A1, device))
        .collect();
    (accs.iter().cloned().fold(1.0, f64::min), accs)
}

// ---------------------------------------------------------------------------
// 1. the hand-built solution
// ---------------------------------------------------------------------------

/// The gate's recurrence **is** the labels' filter: `Δ = 1` and `q ≈ 0` at a
/// value, `Δ ≈ 0` and `q = Q_GAP` at a gap, read at `ω = 1`. No fitting.
#[test]
fn handmade_gate_solves_every_family() {
    let device = Device::default();
    let model = handmade(&device, Arm::Kalman);
    println!("hand-built Kalman block ({} params):", model.num_params());
    let (worst, accs) = worst_family(&model, 256, &device);
    for ((name, _), acc) in FAMILIES.iter().zip(&accs) {
        println!("  {name:<9} {:6.2}%", 100.0 * acc);
    }
    assert!(worst > 0.995, "hand-built solution is not exact: {worst}");
}

// ---------------------------------------------------------------------------
// 2. the join at κ = 0 does not
// ---------------------------------------------------------------------------

/// The same block at `κ = 0` — the scaled member, reading the same `η/Λ` — with
/// the gap's discount swept over a grid of decays. This is the ablation the
/// crate's own structure makes available: one parameter changed, nothing else.
#[test]
fn no_projected_decay_solves_the_task() {
    let device = Device::default();
    println!("scaled member (κ = 0), by gap decay:");
    let header: String = FAMILIES.iter().map(|(n, _)| format!("  {n:>7}")).collect();
    println!("     α_gap {header}     worst");
    let mut best_worst = 0.0f64;
    for alpha_gap in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.2, 0.1, 0.01] {
        let model = handmade(&device, Arm::Scaled { alpha_gap });
        let (worst, accs) = worst_family(&model, 128, &device);
        let cells: String = accs.iter().map(|a| format!("  {:6.2}%", 100.0 * a)).collect();
        println!("  {alpha_gap:7.4} {cells}  {:6.2}%", 100.0 * worst);
        best_worst = best_worst.max(worst);
    }
    println!("best worst-family accuracy over the sweep: {:.2}%", 100.0 * best_worst);
    // ~97½%, against the gate's 100. A projected decay gets *close* here and
    // the bound says so: the rung is an exactness claim, not a chasm.
    assert!(
        best_worst < 0.98,
        "a projected decay reached {best_worst:.4} — the task does not need the gate"
    );
}

// ---------------------------------------------------------------------------
// 3. the honest bound: a linear state that cross-multiplies
// ---------------------------------------------------------------------------

/// What a stock block can still do without dividing: hold a **sum** and a
/// **count** in two heads, each with its own decay on a gap, and let the
/// per-token gate form `v·n − Σ`, whose sign is the comparison. Swept over both
/// gap decays and the value decay, in f64 — a *lower* bound on stock, since it
/// is one family of readouts rather than all of them.
///
/// It lands a few points under the gate. That is the rung's honest size: the
/// Kalman gate is exact where this is not, but a decision only needs the ratio
/// cross-multiplied, and a geometric discount sits close to a hyperbolic one
/// over the range of gap lengths a 32-token sequence holds.
#[test]
fn a_cross_multiplying_linear_state_comes_close() {
    let task = task();
    let decays = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
    let value_decays = [1.0, 0.99, 0.97, 0.95, 0.9];
    let mut knife = 0.0f64;
    println!("best linear (sum, count) + gate, per family:");
    for (name, generator) in FAMILIES {
        let items = TallyDataset::new(&task, *generator, 256, 0xE7A1).items();
        let mut best = (0.0f64, (0.0, 0.0, 0.0));
        for a_s in decays {
            for a_c in decays {
                for b in value_decays {
                    let (mut hit, mut all) = (0usize, 0usize);
                    for item in &items {
                        let (mut sum, mut count) = (0.0f64, 0.0f64);
                        for (&s, &target) in item.symbols.iter().zip(&item.targets) {
                            match value(s) {
                                None => {
                                    sum *= a_s;
                                    count *= a_c;
                                }
                                Some(v) => {
                                    sum = b * sum + v;
                                    count = b * count + 1.0;
                                }
                            }
                            if target == IGNORE {
                                continue;
                            }
                            let y = value(s).unwrap() * count - sum;
                            all += 1;
                            hit += usize::from((y > 0.0) == (target == crate::dataset::ABOVE));
                        }
                    }
                    let acc = hit as f64 / all.max(1) as f64;
                    if acc > best.0 {
                        best = (acc, (a_s, a_c, b));
                    }
                }
            }
        }
        println!(
            "  {name:<9} {:6.2}%   at (gap sum {:.2}, gap count {:.2}, value {:.2})",
            100.0 * best.0,
            best.1.0,
            best.1.1,
            best.1.2
        );
        if *name == "knife" {
            knife = best.0;
        }
    }
    // On the ordinary families it is within a point of the gate; the `knife`
    // family — where the probe sits at the estimate's edge — is the one that
    // separates them, and even there by about two points.
    assert!(
        knife < 0.99,
        "the linear state matched the gate on knife ({knife:.4}) — this rung has no content"
    );
}

/// The task's own numbers: the memoryless ceiling, and how often a gap actually
/// changes the answer (the share of scored positions where ignoring the gaps
/// entirely — a plain running mean — disagrees with the filter).
#[test]
fn the_gaps_are_what_the_task_tests() {
    let task = task();
    println!("  family     memoryless   gaps matter");
    for (name, generator) in FAMILIES {
        let ceiling = memoryless_ceiling(&task, *generator, 512, 0xE7A1);
        let mut disagree = (0u64, 0u64);
        for item in TallyDataset::new(&task, *generator, 512, 0xE7A1).items() {
            let filtered = posterior(&item.symbols);
            let (mut sum, mut count) = (0.0f64, 0.0f64);
            for ((&s, &target), mean) in item.symbols.iter().zip(&item.targets).zip(filtered) {
                if let Some(v) = value(s) {
                    sum += v;
                    count += 1.0;
                    if target != IGNORE {
                        disagree.1 += 1;
                        disagree.0 += u64::from((v > sum / count) != (v > mean));
                    }
                }
            }
        }
        println!(
            "  {name:<10} {:8.2}%   {:8.2}%",
            100.0 * ceiling,
            100.0 * disagree.0 as f64 / disagree.1 as f64
        );
        assert!(ceiling < 0.85, "{name}: the current symbol nearly gives the answer");
    }

    // The labels: a gap is unscored, and so is a near-tie — including the first
    // value of a sequence, which *is* its own posterior mean (an empty prior
    // holds no evidence to be above or below).
    let symbols = vec![8, GAP, 0, 8];
    let out = labels(&symbols);
    assert_eq!(out[0], IGNORE);
    assert_eq!(out[1], IGNORE);
    assert_eq!(out[2], crate::dataset::BELOW); // −2 against a prior at +2
    assert_eq!(out[3], crate::dataset::ABOVE);
}
