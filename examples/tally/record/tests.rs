//! The four claims this rung rests on, measured.
//!
//! 1. A **hand-built** block solves the task exactly. Every weight is in closed
//!    form, and the state of the plant is unused: one tropical register per
//!    head is the whole solution, at any alphabet size.
//! 2. The **same block without the register** cannot, for any decaying
//!    comparison on a four-dimensional grid (decay, write offset,
//!    current-token gain, threshold). The best that it has is an average,
//!    which is *below* a maximum, and the `edge` family is built on exactly
//!    that.
//! 3. A maximum **is** a sum in the exponential domain, and that arm needs no
//!    register. But it needs the whole span of the alphabet inside one channel:
//!    at least `e^{45.8}` at twelve values and 64 tokens.
//!    [`the_exponential_domain_arm_costs_range`] measures what f32 does with
//!    such a span (it uses `S = 12`, so `e^{132}`). That is why this rung has
//!    twelve values, not six, where a trained stock block solves it (~100%).
//! 4. The width closes the third route, the latches. At `d_model = 3`, a block
//!    can make one function of the symbol an affine channel (the tests report
//!    the residuals), but not the twelve threshold indicators that one latch
//!    per value needs.

use crate::dataset::{FAMILIES, NUM_SYMBOLS, NUM_VALUES, RESET, labels, task, value};
use crate::model::{NUM_CLASSES, model_config};
use crate::shared::handmade::{accuracy, affine_channels, memoryless_ceiling, param, softplus_inv};
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

type Device = burn::prelude::Device;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `S`: the units of the register per value step. The soft maximum exceeds the
/// hard maximum by at most `ln(T + 1)` (≈ 4.2 at `T = 64`), in *register*
/// units. So a value step must be worth more than that plus the decision
/// margin.
const SCALE: f64 = 12.0;
/// The decision offset, half of `ln 2`. A record clears `−n·e^(−S)` from below,
/// and a tie or a lower value is at `−ln 2` or below.
const THETA: f64 = std::f64::consts::LN_2 / 2.0;
/// `a`/`b` on an `R`: enough to put the register below the level of any value.
const RESET_DROP: f64 = 1000.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// `Â` for a held state: `|A| = softplus(Â)` is below the `a_floor` of the
/// block.
const A_HOLD_RAW: f64 = -20.0;
/// A `Δ` small enough that the state of the plant stays at zero.
const DELTA_OFF: f64 = 1e-12;
/// Class-logit gain.
const OUT_GAIN: f64 = 3.0;

/// The symbol embeddings: the values on a circle of radius `√2` at height `1`,
/// and the reset at the far pole. Every embedding has norm `√3`, so the
/// pre-`RmsNorm` of the layer (`γ = 1`) does not change it.
///
/// **Which** coordinate spreads the values is the choice of the arm, and it is
/// the interesting part. An embedding is a free 13×3 map, so a block can make
/// **one** function of the symbol an affine channel, and only about three
/// independent ones. That denies the latch route (twelve indicators) at this
/// width. The arm of the register spreads the values by the value
/// (`cos ∝ v`), and the exponential arm by `u(v)`. A *trained* model picks for
/// itself, and it picks the second (see the trained row in the README).
fn embeddings(spread: &dyn Fn(usize) -> f64) -> Vec<Vec<f64>> {
    let r = 2.0f64.sqrt();
    let mut out: Vec<Vec<f64>> = (0..NUM_VALUES)
        .map(|i| {
            let cos = spread(i).clamp(-1.0, 1.0);
            let sin = (1.0 - cos * cos).max(0.0).sqrt();
            vec![r * cos, r * sin, 1.0]
        })
        .collect();
    out.push(vec![0.0, 0.0, -(3.0f64.sqrt())]);
    out
}

/// The value-spread embedding: `cos ∝ v` over the whole alphabet, so the value
/// is an affine read.
fn by_value() -> Vec<Vec<f64>> {
    let mid = (NUM_VALUES as f64 + 1.0) / 2.0;
    let half = (NUM_VALUES as f64 - 1.0) / 2.0;
    embeddings(&|i| ((i as f64 + 1.0) - mid) / half)
}

/// The exponential-spread embedding: `cos ∝ u(v)`, so `u` is the affine read
/// and the value is not.
fn by_exp() -> Vec<Vec<f64>> {
    embeddings(&|i| 2.0 * exp_write(i as f64 + 1.0, i) - 1.0)
}

/// `u(v) = exp(S·(v − v_max))`. One value step is worth `e^S ≈ 1.6·10⁵`, more
/// than a whole sequence of smaller values (`T = 64`), so a sum of these is a
/// maximum. It never overflows, because the largest is 1.
fn exp_write(v: f64, symbol: usize) -> f64 {
    if symbol == RESET {
        0.0
    } else {
        (SCALE * (v - NUM_VALUES as f64)).exp()
    }
}

/// Which arm to build.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// The construction of the rung: a running maximum in one register.
    Register,
    /// The ablation: no register. Head 0 accumulates a decaying average of the
    /// values, compared against the current value.
    Stock {
        alpha: f64,
        offset: f64,
        gain: f64,
        theta: f64,
    },
    /// The ablation that **works**: no register. Head 0 holds the plain sum of
    /// `u(v) = exp(S·(v − v_max))`, read as `u(vₜ) − hₜ₋₁`. That is a running
    /// maximum in the exponential domain, because one value step is worth more
    /// than a whole sequence of smaller values.
    ExpWrite,
}

fn handmade(device: &Device, arm: Arm) -> MambaLatentNet {
    let cfg = model_config(matches!(arm, Arm::Register));
    let mut model = crate::common::model::ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("tally-record configures the Mamba-3 variant")
    };
    let embeddings = match arm {
        Arm::ExpWrite => by_exp(),
        _ => by_value(),
    };
    let values: Vec<f64> = (0..NUM_SYMBOLS)
        .map(|s| value(s).unwrap_or(0) as f64)
        .collect();

    // ── network in_proj: one-hot → the symbol embedding ──────────────────────
    net.in_proj.weight = param(&embeddings.concat(), [NUM_SYMBOLS, 3], device);
    net.in_proj.bias = Some(param(&[0.0; 3], [3], device));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([3]), device));
    let block = &mut layer.block;

    // ── block in_proj ────────────────────────────────────────────────────────
    // Channel order: `[z(3) | x(3) | B(1) | C(1) | Δ(3) | A(3) | a(3) | b(3)]`.
    // The last six belong to the registers, and the ablation arms do not have
    // them.
    let per_symbol = |f: &dyn Fn(usize) -> f64| (0..NUM_SYMBOLS).map(f).collect::<Vec<f64>>();
    let off = vec![softplus_inv(DELTA_OFF); NUM_SYMBOLS];
    let hold = vec![A_HOLD_RAW; NUM_SYMBOLS];
    let mut targets: Vec<Vec<f64>> = vec![vec![Z_PRE; NUM_SYMBOLS]; 3]; // z, three heads
    match arm {
        Arm::Register => {
            // Head 0 reads `S·v + θ` against its register. Head 1 is the
            // reference. Head 2 is inert.
            targets.push(per_symbol(&|s| SCALE * values[s] + THETA));
            targets.push(vec![1.0; NUM_SYMBOLS]);
            targets.push(vec![0.0; NUM_SYMBOLS]);
        }
        Arm::Stock {
            offset, gain, ..
        } => {
            // Head 0 accumulates `v + offset` (and nothing on a reset). Head 1
            // is the reference. Head 2 carries the term of the current value.
            targets.push(per_symbol(&|s| {
                if s == RESET { 0.0 } else { values[s] + offset }
            }));
            targets.push(vec![1.0; NUM_SYMBOLS]);
            targets.push(per_symbol(&|s| gain * values[s]));
        }
        Arm::ExpWrite => {
            // Head 0 writes `u(v)`. Head 1 is the reference. Head 2 is inert.
            targets.push(per_symbol(&|s| exp_write(values[s], s)));
            targets.push(vec![1.0; NUM_SYMBOLS]);
            targets.push(vec![0.0; NUM_SYMBOLS]);
        }
    }
    targets.push(vec![1.0; NUM_SYMBOLS]); // B
    targets.push(vec![1.0; NUM_SYMBOLS]); // C
    match arm {
        Arm::Register => {
            targets.extend([off.clone(), off.clone(), off.clone()]); // Δ: the plant never writes
            targets.extend([hold.clone(), hold.clone(), hold.clone()]); // A
            // The register: `a = 0` holds the maximum, `b = S·v` offers this
            // value, and a reset drops both far below every value.
            targets.push(per_symbol(&|s| if s == RESET { -RESET_DROP } else { 0.0 }));
            targets.push(vec![0.0; NUM_SYMBOLS]); // a, head 1
            targets.push(vec![0.0; NUM_SYMBOLS]); // a, head 2
            targets.push(per_symbol(&|s| {
                if s == RESET { -RESET_DROP } else { SCALE * values[s] }
            }));
            targets.push(vec![0.0; NUM_SYMBOLS]); // b, head 1
            targets.push(vec![0.0; NUM_SYMBOLS]); // b, head 2
        }
        Arm::Stock { alpha, .. } => {
            targets.push(vec![softplus_inv(1.0); NUM_SYMBOLS]); // Δ, head 0 = 1
            targets.push(off.clone()); // Δ, head 1
            targets.push(off); // Δ, head 2
            targets.push(per_symbol(&|s| {
                if s == RESET {
                    softplus_inv(20.0) // a reset wipes the average
                } else {
                    // `α → 0` is "keep only the current token". The clamp keeps
                    // `−ln α` finite there.
                    softplus_inv((-alpha.max(1e-6).ln()).max(1e-9))
                }
            }));
            targets.extend([hold.clone(), hold]); // A, heads 1 and 2
        }
        Arm::ExpWrite => {
            targets.push(vec![softplus_inv(1.0); NUM_SYMBOLS]); // Δ, head 0 = 1
            targets.push(off.clone()); // Δ, head 1
            targets.push(off); // Δ, head 2
            // The sum is held (`α ≈ 1`) and wiped on a reset. A decay of one is
            // the *only* growth rate of the plant, and that decides all that
            // this arm can and cannot do.
            targets.push(per_symbol(&|s| {
                if s == RESET { softplus_inv(20.0) } else { A_HOLD_RAW }
            }));
            targets.extend([hold.clone(), hold]); // A, heads 1 and 2
        }
    }
    let (weight, bias) = affine_channels(&embeddings, &targets, device);
    block.in_proj.weight = weight;
    block.in_proj.bias = Some(bias);

    // ── per-head scalars ─────────────────────────────────────────────────────
    block.dt_bias_h = param(&[0.0; 3], [3], device);
    block.d_h = match arm {
        Arm::Register => param(&[1.0, 1.0, 0.0], [3], device),
        // Head 0 reads only its state (`C = −1` below). Head 2 is its skip.
        Arm::Stock { .. } => param(&[0.0, 1.0, 1.0], [3], device),
        // `y = −hₜ + 2·u(vₜ) = u(vₜ) − hₜ₋₁`, the state carrying its own write.
        Arm::ExpWrite => param(&[2.0, 1.0, 0.0], [3], device),
    };
    block.b_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([1]), device));
    block.c_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([1]), device));
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([3, 1, 1]), device));
    // `C = −1` on head 0 in the ablation arms: its state is *subtracted* from
    // the current value. QK-norm pins |C| = 1, so the sign is in the bias.
    block.c_bias_hmr = match arm {
        Arm::Register => Param::from_tensor(Tensor::zeros(Shape::new([3, 1, 1]), device)),
        Arm::Stock { .. } | Arm::ExpWrite => param(&[-2.0, 0.0, 0.0], [3, 1, 1], device),
    };
    if let Some(readout) = block.tropical_readout_hp.as_mut() {
        // `y += c·e`: head 0 subtracts its register from `S·v + θ`.
        *readout = param(&[-1.0, 0.0, 0.0], [3, 1], device);
    }

    // ── the two projections ──────────────────────────────────────────────────
    // The `out_proj` of the block keeps head 0 on dim 0 and the reference on
    // dim 1. In the `Stock` arm, the current-value term of head 2 joins head 0
    // on dim 0.
    let out_w = match arm {
        Arm::Register | Arm::ExpWrite => vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        Arm::Stock { .. } => vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    };
    block.out_proj.weight = param(&out_w, [3, 3], device);
    block.out_proj.bias = Some(match arm {
        Arm::Register | Arm::ExpWrite => param(&[0.0; 3], [3], device),
        Arm::Stock { theta, .. } => param(&[theta, 0.0, 0.0], [3], device),
    });

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

/// Every weight is in closed form. There is no training. `Δ = 1e-12` in every
/// head, so the SSM state stays at zero: the register gives the answer.
#[test]
fn handmade_register_solves_every_family() {
    let device = Device::default();
    let model = handmade(&device, Arm::Register);
    println!("hand-built register block ({} params):", model.num_params());
    let (worst, accs) = worst_family(&model, 256, &device);
    for ((name, _), acc) in FAMILIES.iter().zip(&accs) {
        println!("  {name:<8} {:6.2}%", 100.0 * acc);
    }
    assert!(worst > 0.995, "hand-built solution is not exact: {worst}");
}

// ---------------------------------------------------------------------------
// 2. no decaying comparison reaches it
// ---------------------------------------------------------------------------

/// The same block **without** the register, and with head 0 as a decaying
/// average of the values instead, compared against the current value:
/// `y = gain·v − h + θ`, `h = α·h + (v + offset)`, and a reset wipes `h`.
///
/// The test sweeps the whole grid and prints only the best point per decay. So
/// the number bounds the ablated *architecture*, not one fitting of it.
#[test]
fn no_decaying_comparison_solves_the_task() {
    let device = Device::default();
    let alphas = [1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0];
    let offsets = [-2.0, -1.0, 0.0, 1.0];
    let gains = [0.5, 1.0, 2.0, 4.0];
    let thetas = [-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0];

    println!("stock sweep (register off), best per decay:");
    let header: String = FAMILIES.iter().map(|(n, _)| format!("  {n:>7}")).collect();
    println!("        α  offset   gain      θ {header}     worst");
    let mut best_worst = 0.0f64;
    for alpha in alphas {
        let mut best = (0.0f64, vec![0.0; FAMILIES.len()], 0.0, 0.0, 0.0);
        for offset in offsets {
            for gain in gains {
                for theta in thetas {
                    let model = handmade(
                        &device,
                        Arm::Stock {
                            alpha,
                            offset,
                            gain,
                            theta,
                        },
                    );
                    let (worst, accs) = worst_family(&model, 64, &device);
                    if worst > best.0 {
                        best = (worst, accs, offset, gain, theta);
                    }
                }
            }
        }
        let cells: String = best.1.iter().map(|a| format!("  {:6.2}%", 100.0 * a)).collect();
        println!(
            "  {alpha:7.4} {:7.2} {:6.2} {:6.2} {cells}  {:6.2}%",
            best.2, best.3, best.4, 100.0 * best.0
        );
        best_worst = best_worst.max(best.0);
    }
    println!("best worst-family accuracy over the whole sweep: {:.2}%", 100.0 * best_worst);
    assert!(
        best_worst < 0.95,
        "a decaying comparison reached {best_worst:.4} — the task does not need the register"
    );
}

// ---------------------------------------------------------------------------
// 2b. what the register does *not* buy
// ---------------------------------------------------------------------------

/// The ablation that works, and the reason why this rung is a coda, not a
/// wall: a **maximum is a sum in the exponential domain**. A plain linear state
/// that holds `Σ exp(S·vₛ)` with a decay of one *is* the `log Σ exp(S·vₛ)` of
/// the register. The comparison that a record needs, `u(vₜ) > hₜ₋₁`, is linear
/// in that domain. So the logarithm is never taken, and the register is not
/// needed.
///
/// `scripts/kalman/gate_as_positive_system.py` §5.4 checks the identity in
/// f64. This test measures what it costs **in f32**, and that makes the rung.
///
/// The arm writes `u(v) = exp(S·(v − v_max))` through an embedding coordinate.
/// At six values and 32 tokens, that span is `e^{20}`, and a trained stock
/// block solves the task. At twelve values and 64 tokens, it is at least
/// `e^{45.8}`. The lower values then collapse onto one embedding point, and the
/// arm is far from exact. More heads do not help: every channel is an affine
/// read of the *same* embedding, so they share the resolution of one mantissa.
///
/// The register has that **range** over it: it carries `S·v` where the sum
/// carries `e^{S·v}`. In `tally-depth`, it also has **growth**: `a > 0` is a
/// decay above one, which the `α = exp(Δ·A) ≤ 1` of the plant cannot express.
#[test]
fn the_exponential_domain_arm_costs_range() {
    let device = Device::default();
    let model = handmade(&device, Arm::ExpWrite);
    println!("hand-built exponential-write block (no register), in f32:");
    let (worst, accs) = worst_family(&model, 256, &device);
    for ((name, _), acc) in FAMILIES.iter().zip(&accs) {
        println!("  {name:<8} {:6.2}%", 100.0 * acc);
    }
    assert!(
        worst < 0.9,
        "f32 held the whole exponential span ({worst}) — the range caveat is wrong"
    );
}

// ---------------------------------------------------------------------------
// 3. the task, and what the width does and does not deny
// ---------------------------------------------------------------------------

/// The memoryless ceiling, and what `d_model = 3` actually rules out.
///
/// An embedding is a free map, so the block can make **one** function of the
/// symbol an affine channel (the value, or `u(v)`, or another), but not twelve
/// independent ones. The residuals below are against the *value-spread*
/// embedding. The value fits exactly, and every threshold indicator misses. So
/// a latch construction (one per value, selected by the gate) does not fit at
/// this width. The exponential route needs only one channel, and
/// [`the_exponential_domain_arm_costs_range`] takes it.
#[test]
fn the_width_denies_the_latches_but_not_one_nonlinear_channel() {
    let task = task();
    for (name, generator) in FAMILIES {
        let ceiling = memoryless_ceiling(&task, *generator, 512, 0xE7A1);
        println!("  {name:<8} memoryless ceiling {:6.2}%", 100.0 * ceiling);
        assert!(ceiling < 0.85, "{name}: the current symbol nearly gives the answer");
    }

    // Least squares of each candidate channel on the embeddings, with the
    // residual reported: zero for the value, non-zero for every indicator.
    let embeddings = by_value();
    let rows: Vec<Vec<f64>> = embeddings
        .iter()
        .map(|e| e.iter().copied().chain([1.0]).collect())
        .collect();
    let residual = |target: &[f64]| -> f64 {
        // normal equations, the same solve `affine_channels` uses
        let n = rows[0].len();
        let mut a = vec![vec![0.0; n]; n];
        let mut y = vec![0.0; n];
        for (row, &r) in rows.iter().zip(target) {
            for i in 0..n {
                y[i] += row[i] * r;
                for j in 0..n {
                    a[i][j] += row[i] * row[j];
                }
            }
        }
        for col in 0..n {
            let piv = (col..n)
                .max_by(|&p, &q| a[p][col].abs().partial_cmp(&a[q][col].abs()).unwrap())
                .unwrap();
            a.swap(col, piv);
            y.swap(col, piv);
            for row in 0..n {
                if row != col && a[col][col].abs() > 1e-12 {
                    let f = a[row][col] / a[col][col];
                    let pivot = a[col].clone();
                    for (k, entry) in a[row].iter_mut().enumerate().skip(col) {
                        *entry -= f * pivot[k];
                    }
                    y[row] -= f * y[col];
                }
            }
        }
        let w: Vec<f64> = (0..n).map(|i| y[i] / a[i][i]).collect();
        rows.iter()
            .zip(target)
            .map(|(row, &r)| (row.iter().zip(&w).map(|(x, w)| x * w).sum::<f64>() - r).abs())
            .fold(0.0, f64::max)
    };

    let value_channel: Vec<f64> = (0..NUM_SYMBOLS)
        .map(|s| value(s).unwrap_or(0) as f64)
        .collect();
    println!("  channel 'v'      residual {:.2e}", residual(&value_channel));
    assert!(residual(&value_channel) < 1e-9, "the value must be affine in the embedding");

    for k in 2..=NUM_VALUES {
        let latch: Vec<f64> = (0..NUM_SYMBOLS)
            .map(|s| f64::from(value(s).unwrap_or(0) >= k as i64))
            .collect();
        let r = residual(&latch);
        println!("  channel '[v ≥ {k}]' residual {r:.3}");
        assert!(r > 0.05, "a latch for {k} fits at this width — the rung is not about the register");
    }

    // The labels themselves: ties are not records, and a reset restarts.
    let symbols = vec![2, 0, 3, 3, 1, RESET, 1, 4];
    assert_eq!(
        labels(&symbols),
        vec![1, 0, 1, 0, 0, crate::shared::data::IGNORE, 1, 1]
    );
}
