//! The two claims this example rests on, measured.
//!
//! 1. A **hand-built** Mamba-2 block solves the task exactly — no fitting, every
//!    weight written down in closed form from the unrolled recurrence.
//! 2. The **same block with a non-selective decay** cannot, for *any* decay and
//!    *any* readout gain. That is what makes the task a Mamba-2 task rather than
//!    a plain linear-SSM one.
//!
//! The construction is the one derived in [`crate::model`]: head 0 is the ballot
//! box (`Δ₀` tiny on `±` so `ᾱ₀ ≈ 1`, huge on `RESET` so `ᾱ₀ ≈ 0`), head 1 is a
//! constant reference, and the gated RMSNorm turns the pair into a direction the
//! two-class head reads off.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    Family, IGNORE, MINUS, NUM_CLASSES, NUM_SYMBOLS, PLUS, RESET, ResetMajorityDataset, SEQ_LENGTH,
    labels, one_hot,
};
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `Δ₀` on a `±` symbol: with `A₀ = ln(HOLD_ALPHA)/Δ_HOLD` this makes the decay
/// `HOLD_ALPHA` per step, i.e. an (essentially) unweighted running sum.
const DELTA_HOLD: f64 = 1e-5;
/// `Δ₀` on a `RESET`: `Δ_WIPE·|A₀| = 20`, so `ᾱ₀ = e⁻²⁰` erases the past.
const DELTA_WIPE: f64 = 20.0;
/// The per-step decay the ballot box holds at.
const HOLD_ALPHA: f64 = 0.99999; // = exp(-1e-5)
/// `Δ₁`, small enough that head 1's state never leaves its `D₁·x₁` reference.
const DELTA_REF: f64 = 1e-12;
/// `x₀(±) = ±V`. Bounded by silu's floor (`min silu = -0.2785`), which is what
/// lets the two votes be exactly symmetric.
const V: f64 = 0.2;
/// The shared `B` and `C` magnitudes: `C·(Δ_HOLD·B·V) = 2` per vote.
const BC_MAG: f64 = 1000.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// Class-logit gain on the vote axis.
const OUT_GAIN: f64 = 3.0;

// ---------------------------------------------------------------------------
// scalar helpers
// ---------------------------------------------------------------------------

fn silu(t: f64) -> f64 {
    t / (1.0 + (-t).exp())
}

/// Inverse of `silu` on the branch containing 0 (`t > -1.2785`).
fn silu_inv(v: f64) -> f64 {
    assert!(v > -0.2784, "silu bottoms out at -0.2785, cannot reach {v}");
    let (mut lo, mut hi) = (-1.2785f64, v.max(0.0) + 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if silu(mid) < v { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// Inverse of `softplus`, stable for tiny `v`.
fn softplus_inv(v: f64) -> f64 {
    v.exp_m1().ln()
}

/// Solve the 3×3 system `M·w = rhs` by Gaussian elimination with partial pivoting.
fn solve3(mut m: [[f64; 3]; 3], mut rhs: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let piv = (col..3)
            .max_by(|&a, &b| m[a][col].abs().partial_cmp(&m[b][col].abs()).unwrap())
            .unwrap();
        m.swap(col, piv);
        rhs.swap(col, piv);
        assert!(m[col][col].abs() > 1e-12, "singular symbol embedding");
        for row in 0..3 {
            if row == col {
                continue;
            }
            let f = m[row][col] / m[col][col];
            for k in col..3 {
                m[row][k] -= f * m[col][k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    [rhs[0] / m[0][0], rhs[1] / m[1][1], rhs[2] / m[2][2]]
}

fn t1<const D: usize>(v: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<1>::from_floats(f.as_slice(), device).reshape(shape)
}

// ---------------------------------------------------------------------------
// the hand-built model
// ---------------------------------------------------------------------------

/// The three symbol embeddings, each of norm `√2` so the layer's pre-`RmsNorm`
/// (`γ = 1`) passes them through unchanged. Indexed by [`MINUS`]/[`PLUS`]/[`RESET`].
const EMBED: [[f64; 2]; NUM_SYMBOLS] = [
    [std::f64::consts::SQRT_2, 0.0],
    [0.0, std::f64::consts::SQRT_2],
    [-1.0, -1.0],
];

/// Build the block by hand.
///
/// `alpha` is the ballot box's per-step decay on a `±` symbol. When `selective`
/// is false the `RESET` symbol gets that same decay instead of `e⁻²⁰` — the
/// **only** difference, which turns the block into a fixed-decay (LTI) SSM.
/// `gain` scales the `C` readout, i.e. the vote-to-reference ratio.
fn handmade(device: &Device, selective: bool, alpha: f64, gain: f64) -> MambaLatentNet {
    let cfg = crate::model::model_config();
    let mut model = ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba2(net) = &mut model else {
        unreachable!("reset-majority configures the Mamba-2 variant")
    };

    // ── network in_proj: one-hot → the symbol embedding ──────────────────────
    net.in_proj.weight = Param::from_tensor(t1(&EMBED.concat(), [NUM_SYMBOLS, 2], device));
    net.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([2]), device)));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([2]), device));
    let block = &mut layer.mamba_block;

    // ── block in_proj: one affine functional per channel ─────────────────────
    // Channel order is `[z(2) | x(2) B(1) C(1) | dt_raw(2)]`; each entry is the
    // channel's target value at (MINUS, PLUS, RESET), *before* silu/softplus.
    let hold = softplus_inv(DELTA_HOLD);
    let dt0 = if selective {
        [hold, hold, softplus_inv(DELTA_WIPE)]
    } else {
        [hold; 3]
    };
    let bc = silu_inv(BC_MAG);
    let targets: [[f64; 3]; 8] = [
        [Z_PRE; 3],                              // z, head 0
        [Z_PRE; 3],                              // z, head 1
        [silu_inv(-V), silu_inv(V), 0.0],        // x, head 0 — the ballot
        [silu_inv(1.0); 3],                      // x, head 1 — the reference
        [bc; 3],                                 // B (shared)
        [silu_inv(BC_MAG * gain); 3],            // C (shared)
        dt0,                                     // dt_raw, head 0
        [softplus_inv(DELTA_REF); 3],            // dt_raw, head 1
    ];
    let rows = [
        [EMBED[MINUS][0], EMBED[MINUS][1], 1.0],
        [EMBED[PLUS][0], EMBED[PLUS][1], 1.0],
        [EMBED[RESET][0], EMBED[RESET][1], 1.0],
    ];
    let mut w = vec![0.0f64; 2 * 8];
    let mut b = vec![0.0f64; 8];
    for (ch, target) in targets.iter().enumerate() {
        let [w0, w1, bias] = solve3(rows, *target);
        w[ch] = w0; // weight is [d_model, out]: row i, column ch
        w[8 + ch] = w1;
        b[ch] = bias;
    }
    block.in_proj.weight = Param::from_tensor(t1(&w, [2, 8], device));
    block.in_proj.bias = Some(Param::from_tensor(t1(&b, [8], device)));

    // ── conv1d (kernel 1): the identity ──────────────────────────────────────
    let conv_dim = 4; // d_inner + 2·ngroups·state_rank
    block.conv1d.weight = Param::from_tensor(Tensor::ones(Shape::new([conv_dim, 1, 1]), device));
    block.conv1d.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([conv_dim]),
        device,
    )));

    // ── Δ bias, A, D ─────────────────────────────────────────────────────────
    block.dt_bias_h = Param::from_tensor(Tensor::zeros(Shape::new([2]), device));
    // A_h = -exp(a_log_h); head 0 is set so exp(Δ_HOLD·A₀) = alpha exactly.
    let a0 = -alpha.ln() / DELTA_HOLD;
    block.a_log_h = Param::from_tensor(t1(&[a0.ln(), 0.0], [2], device));
    // D₀ = 0 (head 0 reads the state alone), D₁ = 1 (head 1 *is* its skip).
    block.d_h = Param::from_tensor(t1(&[0.0, 1.0], [2], device));

    // ── output norm and the two projections ──────────────────────────────────
    block.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([2]), device));
    block.out_proj.weight = Param::from_tensor(t1(&[1.0, 0.0, 0.0, 1.0], [2, 2], device));
    block.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([2]), device)));

    // logits [NEG, POS] = [-g·o₀, +g·o₀]; `ignore_last_residual` means `o` is
    // all the head sees. `o₁` (the reference axis) enters only through the
    // normalisation, which is what keeps the margin proportional to the vote.
    net.out_proj.weight = Param::from_tensor(t1(
        &[-OUT_GAIN, OUT_GAIN, 0.0, 0.0],
        [2, NUM_CLASSES],
        device,
    ));
    net.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([NUM_CLASSES]),
        device,
    )));

    model
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

/// Per-position accuracy of `model` on `count` sequences of one family.
fn accuracy(model: &MambaLatentNet, family: Family, count: usize, device: &Device) -> f64 {
    let items: Vec<_> = ResetMajorityDataset::new(count, SEQ_LENGTH, family, 0xE7A1)
        .iter()
        .map(|i| i.expect("dataset item"))
        .collect();
    let inputs = Tensor::stack(
        items
            .iter()
            .map(|i| one_hot(&i.symbols, device))
            .collect::<Vec<_>>(),
        0,
    );
    let (out, _c) = model.forward(
        inputs,
        None,
        MambaSsdPath::Mamba2(Mamba2SsdPath::Minimal(None)),
        None,
    );
    let n = count * SEQ_LENGTH;
    let pred = out
        .reshape([n, NUM_CLASSES])
        .argmax(1)
        .reshape([n])
        .into_data()
        .try_to_vec::<i32>()
        .unwrap();
    let want: Vec<i64> = items.iter().flat_map(|i| i.targets.clone()).collect();
    // zero-vote positions have no sign to report and are not scored
    let scored: Vec<(i64, i64)> = pred
        .iter()
        .zip(&want)
        .filter(|(_, t)| **t != IGNORE)
        .map(|(p, t)| (i64::from(*p), *t))
        .collect();
    let hits = scored.iter().filter(|(p, t)| p == t).count();
    hits as f64 / scored.len() as f64
}

const FAMILIES: [(&str, Family); 3] = [
    ("random", Family::Random),
    ("long-prefix", Family::LongPrefix),
    ("long-suffix", Family::LongSuffix),
];

// ---------------------------------------------------------------------------
// 1. the hand-built solution
// ---------------------------------------------------------------------------

/// Every weight written down in closed form; no training anywhere.
#[test]
fn handmade_block_solves_every_family() {
    let device = Device::default();
    let model = handmade(&device, true, HOLD_ALPHA, 1.0);
    println!("hand-built selective block ({} params):", model.num_params());
    let mut worst = 1.0f64;
    for (name, family) in FAMILIES {
        let acc = accuracy(&model, family, 256, &device);
        println!("  {name:<12} {:6.2}%", 100.0 * acc);
        worst = worst.min(acc);
    }
    assert!(worst > 0.995, "hand-built solution is not exact: {worst}");
}

// ---------------------------------------------------------------------------
// 2. no fixed decay reaches it
// ---------------------------------------------------------------------------

/// Sweep the ballot box's decay and readout gain with `RESET`'s selectivity
/// switched **off** — the one changed knob — and report the best any of them do.
///
/// Both adversarial families are unreachable at once, and from opposite sides: a
/// decay near 1 leaks the pre-reset run into `long-prefix`, a decay away from 1
/// lets the late block outvote the early one in `long-suffix`.
#[test]
fn no_fixed_decay_solves_the_task() {
    let device = Device::default();
    let alphas = [
        1.0 - 1e-5,
        0.99,
        0.95,
        0.9,
        0.8,
        0.7,
        0.5,
        0.3,
        0.1,
        0.01,
    ];
    let gains = [0.1f64, 0.3, 1.0, 3.0, 10.0, 30.0];

    println!("fixed-decay sweep (best gain per α), accuracy per family:");
    println!("      ᾱ    random  long-prefix  long-suffix     worst");
    let mut best_worst = 0.0f64;
    for alpha in alphas {
        // the readout gain is re-fitted per α, so this is the *best case* for a
        // fixed decay, not one arbitrary calibration of it.
        let mut best = (0.0f64, [0.0f64; 3]);
        for gain in gains {
            let model = handmade(&device, false, alpha, gain);
            let accs: Vec<f64> = FAMILIES
                .iter()
                .map(|(_, f)| accuracy(&model, *f, 128, &device))
                .collect();
            let worst = accs.iter().cloned().fold(1.0f64, f64::min);
            if worst > best.0 {
                best = (worst, [accs[0], accs[1], accs[2]]);
            }
        }
        println!(
            "  {alpha:7.5}  {:6.2}%     {:6.2}%      {:6.2}%    {:6.2}%",
            100.0 * best.1[0],
            100.0 * best.1[1],
            100.0 * best.1[2],
            100.0 * best.0
        );
        best_worst = best_worst.max(best.0);
    }
    println!("best worst-family accuracy over the whole sweep: {:.2}%", 100.0 * best_worst);
    assert!(
        best_worst < 0.9,
        "a fixed decay reached {best_worst:.4} — the task does not need selectivity"
    );
}

// ---------------------------------------------------------------------------
// 3. the task itself
// ---------------------------------------------------------------------------

/// The **memoryless ceiling**: the best a model that sees only the current
/// symbol can do. No residual, embedding, or `conv_kernel = 1` window gets past
/// it, and the hand-built block above is at 100%.
///
/// It is not chance — a `+` really does more often sit on a positive count, and
/// the long same-sign runs in `long-prefix` sharpen that — but it is nowhere
/// near solving the task, which is the point.
#[test]
fn memoryless_ceiling_is_far_below_the_state() {
    let mut overall = [[0u64; NUM_CLASSES]; NUM_SYMBOLS];
    for (name, family) in FAMILIES {
        let mut tally = [[0u64; NUM_CLASSES]; NUM_SYMBOLS];
        for item in ResetMajorityDataset::new(512, SEQ_LENGTH, family, 0xE7A1).iter() {
            let item = item.expect("dataset item");
            for (&s, &c) in item.symbols.iter().zip(&labels(&item.symbols)) {
                if c == IGNORE {
                    continue; // unscored: no sign to report
                }
                tally[s][c as usize] += 1;
                overall[s][c as usize] += 1;
            }
        }
        println!("  {name:<12} best per-symbol table: {:6.2}%", 100.0 * table_ceiling(&tally));
    }
    let names = ["-", "+", "R"];
    for (s, row) in overall.iter().enumerate() {
        println!("  symbol {}: neg {:>6}  pos {:>6}", names[s], row[0], row[1]);
    }
    let ceiling = table_ceiling(&overall);
    println!(
        "memoryless ceiling {:.2}%  (chance {:.2}%, hand-built block 100%)",
        100.0 * ceiling,
        100.0 / NUM_CLASSES as f64
    );
    assert!(
        ceiling < 0.8,
        "the current symbol nearly gives the answer: {ceiling}"
    );
}

/// Accuracy of the best per-symbol lookup table implied by a tally.
fn table_ceiling(tally: &[[u64; NUM_CLASSES]; NUM_SYMBOLS]) -> f64 {
    let total: u64 = tally.iter().flatten().sum();
    let best: u64 = tally
        .iter()
        .map(|row| row.iter().max().copied().unwrap_or(0))
        .sum();
    best as f64 / total as f64
}

