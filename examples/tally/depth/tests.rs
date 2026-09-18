//! The three claims this rung rests on, measured.
//!
//! 1. A **hand-built** block solves the task exactly — every weight written
//!    down in closed form, nothing fitted, and the plant's state unused: one
//!    tropical register per head is the whole solution.
//! 2. The **same block with the register removed** cannot, for any selective
//!    gate on a grid: a linear recurrence has no floor, and a multiplicative
//!    decrement (the trick the handoff's toy found stock using) never reaches
//!    zero, so it degrades exactly where the depth returns to it.
//! 3. The task is not solvable from the current symbol, and the *unclamped*
//!    sum — what any linear state computes — is misled where the floor bites.

use crate::dataset::{
    CLOSE, EMPTY, FAMILIES, INSIDE, NUM_SYMBOLS, OPEN, RESET, SEQ_LENGTH, gen_random, labels, task,
    unclamped,
};
use crate::model::{NUM_CLASSES, model_config};
use crate::shared::data::{Generator, IGNORE, TallyDataset};
use crate::shared::handmade::{accuracy, affine_channels, memoryless_ceiling, param, softplus_inv};
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

type Device = burn::prelude::Device;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `S`: the register's units per bracket. The soft `max` overshoots the hard
/// one by at most `ln(T + 1)` (≈ 3.5 at `T = 32`), so a bracket has to be worth
/// more than twice that for the threshold at `S/2` to separate depth 0 from
/// depth 1 — see [`crate::shared`]'s log-sum-exp bound.
const SCALE: f64 = 12.0;
/// The decision threshold, `S/2`.
const THETA: f64 = SCALE / 2.0;
/// `a` on a `R`: enough to push the register back onto its floor.
const RESET_DROP: f64 = 1000.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// `Â` for a held state: `A = −softplus(Â)` lands under the block's `a_floor`.
const A_HOLD_RAW: f64 = -20.0;
/// A `Δ` small enough that the plant's state never leaves zero.
const DELTA_OFF: f64 = 1e-12;
/// Class-logit gain.
const OUT_GAIN: f64 = 3.0;

/// The three symbol embeddings, each of norm `√2` so the layer's pre-`RmsNorm`
/// passes them through unchanged. Indexed by [`OPEN`] / [`CLOSE`] / [`RESET`].
const EMBED: [[f64; 2]; NUM_SYMBOLS] = [
    [std::f64::consts::SQRT_2, 0.0],
    [0.0, std::f64::consts::SQRT_2],
    [-1.0, -1.0],
];

/// Which arm to build: the register, or the same block with it removed and a
/// selective gate in its place.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// The rung's own construction: one tropical register, no plant state.
    Register,
    /// The ablation: no register; head 0 is a selective-decay accumulator
    /// (`α` on `)`, a hold on `(`, a wipe on `R`, write `w` on `)`), read
    /// against a threshold.
    Stock {
        alpha_close: f64,
        w_close: f64,
        theta: f64,
    },
}

// ---------------------------------------------------------------------------
// the hand-built model
// ---------------------------------------------------------------------------

fn handmade(device: &Device, arm: Arm) -> MambaLatentNet {
    let cfg = model_config(matches!(arm, Arm::Register));
    let mut model = crate::common::model::ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("tally-depth configures the Mamba-3 variant")
    };
    let embeddings: Vec<Vec<f64>> = EMBED.iter().map(|e| e.to_vec()).collect();

    // ── network in_proj: one-hot → the symbol embedding ──────────────────────
    net.in_proj.weight = param(&EMBED.concat(), [NUM_SYMBOLS, 2], device);
    net.in_proj.bias = Some(param(&[0.0, 0.0], [2], device));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([2]), device));
    let block = &mut layer.block;

    // ── block in_proj: one affine functional per channel ─────────────────────
    // Channel order (see `Mamba3::in_proj`): `[z(2) | x(2) | B(1) | C(1) |
    // Δ(2) | A(2) | a(2) | b(2)]` — no λ (`Trapezoid::None`), no rotation
    // (`Real1D`), no noise (`Gain::Projected`); the last four are the tropical
    // register's, and are absent entirely in the ablation arm.
    let hold = [A_HOLD_RAW; NUM_SYMBOLS];
    let off = [softplus_inv(DELTA_OFF); NUM_SYMBOLS];
    let mut targets: Vec<Vec<f64>> = vec![
        vec![Z_PRE; NUM_SYMBOLS], // z, head 0
        vec![Z_PRE; NUM_SYMBOLS], // z, head 1
    ];
    match arm {
        Arm::Register => {
            targets.push(vec![-THETA; NUM_SYMBOLS]); // x, head 0 — the threshold
            targets.push(vec![1.0; NUM_SYMBOLS]); // x, head 1 — the reference
            targets.push(vec![1.0; NUM_SYMBOLS]); // B
            targets.push(vec![1.0; NUM_SYMBOLS]); // C
            targets.push(off.to_vec()); // Δ, head 0 — the plant never writes
            targets.push(off.to_vec()); // Δ, head 1
            targets.push(hold.to_vec()); // A, head 0
            targets.push(hold.to_vec()); // A, head 1
            // The register: `a = ±S` on the brackets and a drop on `R`, while
            // `b` is the floor the token itself guarantees — `S` after a `(`
            // (the depth is at least one), `0` after a `)` or an `R`. So `c` is
            // `S ×` the clamped depth at every position, including the first,
            // where the carry is still "the max of nothing".
            targets.push(vec![SCALE, -SCALE, -RESET_DROP]); // a, head 0
            targets.push(vec![0.0; NUM_SYMBOLS]); // a, head 1
            targets.push(vec![SCALE, 0.0, 0.0]); // b, head 0
            targets.push(vec![0.0; NUM_SYMBOLS]); // b, head 1
        }
        Arm::Stock {
            alpha_close,
            w_close,
            ..
        } => {
            // head 0 accumulates `+1` per `(` and `w_close` per `)`, decaying by
            // `alpha_close` on a `)` and holding on a `(`; `R` wipes.
            targets.push(vec![1.0, w_close, 0.0]); // x, head 0 — the write
            targets.push(vec![1.0; NUM_SYMBOLS]); // x, head 1 — the reference
            targets.push(vec![1.0; NUM_SYMBOLS]); // B
            targets.push(vec![1.0; NUM_SYMBOLS]); // C
            targets.push(vec![softplus_inv(1.0); NUM_SYMBOLS]); // Δ, head 0 = 1
            targets.push(off.to_vec()); // Δ, head 1
            targets.push(vec![
                A_HOLD_RAW,                                      // `(`: hold
                softplus_inv((-alpha_close.max(1e-6).ln()).max(1e-9)), // `)`: decay by α
                softplus_inv(20.0),                              // `R`: wipe
            ]);
            targets.push(hold.to_vec()); // A, head 1
        }
    }
    let (weight, bias) = affine_channels(&embeddings, &targets, device);
    block.in_proj.weight = weight;
    block.in_proj.bias = Some(bias);

    // ── per-head scalars ─────────────────────────────────────────────────────
    block.dt_bias_h = param(&[0.0, 0.0], [2], device);
    block.d_h = match arm {
        // head 0 reads `−θ` through its skip, head 1 *is* its skip.
        Arm::Register => param(&[1.0, 1.0], [2], device),
        // head 0 reads the state alone; the threshold rides `out_proj`'s bias.
        Arm::Stock { .. } => param(&[0.0, 1.0], [2], device),
    };
    // A scalar state has nothing to normalise against: QK-norm pins |B| = |C| = 1.
    block.b_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([1]), device));
    block.c_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([1]), device));
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([2, 1, 1]), device));
    block.c_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([2, 1, 1]), device));
    if let Some(readout) = block.tropical_readout_hp.as_mut() {
        // `y += c·e`: head 0 reads its register, head 1 does not.
        *readout = param(&[1.0, 0.0], [2, 1], device);
    }

    // ── the two projections ──────────────────────────────────────────────────
    block.out_proj.weight = param(&[1.0, 0.0, 0.0, 1.0], [2, 2], device);
    block.out_proj.bias = Some(match arm {
        Arm::Register => param(&[0.0, 0.0], [2], device),
        Arm::Stock { theta, .. } => param(&[theta, 0.0], [2], device),
    });

    let norm_f = net.norm_f.as_mut().expect("final_norm is on");
    norm_f.gamma = Param::from_tensor(Tensor::ones(Shape::new([2]), device));
    net.out_proj.weight = param(
        &[-OUT_GAIN, OUT_GAIN, 0.0, 0.0],
        [2, NUM_CLASSES],
        device,
    );
    net.out_proj.bias = Some(param(&[0.0, 0.0], [NUM_CLASSES], device));
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

/// Every weight in closed form; no training anywhere. The plant's `Δ` is
/// `1e-12` in both heads, so the SSM state is zero throughout: what answers is
/// the register.
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
// 2. no selective gate reaches it
// ---------------------------------------------------------------------------

/// The same block with the register **removed** (`Tropical::None`, the one
/// changed knob) and head 0 turned into a selective accumulator instead: sweep
/// its decay on `)`, its write on `)` and the readout threshold, and report the
/// best any of them do.
///
/// A decay of 1 is the unclamped sum, which the `floor` family defeats; a decay
/// below 1 imitates the clamp by never crossing zero, which the `deep` family
/// defeats (after a deep excursion the decayed surplus is indistinguishable
/// from an empty stack).
#[test]
fn no_stock_gate_solves_the_task() {
    let device = Device::default();
    let alphas = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3];
    let writes = [-1.0, -0.5, 0.0];
    let thetas = [-2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5];

    println!("stock sweep (register off), best per decay:");
    println!("      α_)   write      θ   random    floor     deep    worst");
    let mut best_worst = 0.0f64;
    for alpha_close in alphas {
        let mut best = (0.0f64, vec![0.0; FAMILIES.len()], 0.0, 0.0);
        for w_close in writes {
            for theta in thetas {
                let model = handmade(
                    &device,
                    Arm::Stock {
                        alpha_close,
                        w_close,
                        theta,
                    },
                );
                let (worst, accs) = worst_family(&model, 96, &device);
                if worst > best.0 {
                    best = (worst, accs, w_close, theta);
                }
            }
        }
        println!(
            "  {alpha_close:7.4} {:7.2} {:6.2}  {:6.2}%  {:6.2}%  {:6.2}%  {:6.2}%",
            best.2,
            best.3,
            100.0 * best.1[0],
            100.0 * best.1[1],
            100.0 * best.1[2],
            100.0 * best.0
        );
        best_worst = best_worst.max(best.0);
    }
    println!("best worst-family accuracy over the whole sweep: {:.2}%", 100.0 * best_worst);
    assert!(
        best_worst < 0.95,
        "a selective gate reached {best_worst:.4} — the task does not need the register"
    );
}

// ---------------------------------------------------------------------------
// 3. the task itself
// ---------------------------------------------------------------------------

/// The memoryless ceiling — the best a model that sees only the current symbol
/// can do — and the share of scored positions where the **unclamped** sum (what
/// a linear state holds) disagrees with the clamped depth. The second number is
/// what the `floor` family is for.
#[test]
fn the_floor_is_what_the_task_tests() {
    let task = task();
    println!("  family     memoryless   sum ≠ clamp");
    for (name, generator) in FAMILIES {
        let ceiling = memoryless_ceiling(&task, *generator, 512, 0xE7A1);
        let mut misled = (0u64, 0u64);
        for item in TallyDataset::new(&task, *generator, 512, 0xE7A1).items() {
            let sum = unclamped(&item.symbols);
            for ((_, &c), &u) in item.symbols.iter().zip(&item.targets).zip(&sum) {
                if c == IGNORE {
                    continue;
                }
                misled.1 += 1;
                misled.0 += u64::from((u > 0) != (c == INSIDE));
            }
        }
        println!(
            "  {name:<10} {:8.2}%   {:8.2}%",
            100.0 * ceiling,
            100.0 * misled.0 as f64 / misled.1 as f64
        );
        assert!(ceiling < 0.8, "{name}: the current symbol nearly gives the answer");
        if *name == "floor" {
            assert!(
                misled.0 as f64 / misled.1 as f64 > 0.2,
                "the floor family does not actually mislead a running sum"
            );
        }
    }
    // The labels themselves: a `)` at depth 0 stays at depth 0.
    let symbols = vec![CLOSE, OPEN, CLOSE, CLOSE, OPEN, RESET, CLOSE];
    assert_eq!(
        labels(&symbols),
        vec![EMPTY, IGNORE, EMPTY, EMPTY, IGNORE, IGNORE, EMPTY]
    );
}
