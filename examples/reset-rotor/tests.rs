//! The three claims this example rests on, measured.
//!
//! 1. A **hand-built** Mamba-3 block solves the task exactly — no fitting, every
//!    weight written down in closed form from the unrolled recurrence.
//! 2. The **same block with an input-independent rotation** cannot, for *any*
//!    per-step angle: a fixed rotation measures the position since the reset,
//!    not the turns taken.
//! 3. **No real state can**, for any decay: with the rotation switched off the
//!    block is a Mamba-2-strength selective SSM, and the best readout its head
//!    can express — the count axis cut into three intervals — cannot report a
//!    residue that alternates along that axis.
//!
//! (2) and (3) are not grid sweeps of the readout: for each ablated block the
//! **best readout it admits** is computed exactly from its own outputs — every
//! 3-interval cut of the scalar channel, every 3-sector cut of the output plane.
//! What is swept is the one knob the ablation leaves free (the angle, the
//! decay).
//!
//! The construction is the one derived in [`crate::model`]: `R` writes the
//! rotor's zero detent into the state at the current phase, `±` turn the phase
//! by one detent, and the two heads read the accumulated turn on two axes a
//! quarter turn apart.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    Family, MINUS, MODULUS, NUM_CLASSES, NUM_SYMBOLS, PLUS, RESET, ResetRotorDataset, SEQ_LENGTH,
    labels, one_hot, steps_since_reset, turns,
};
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// One detent, in radians: the per-step rotation a `±` symbol applies.
const DETENT: f64 = 2.0 * std::f64::consts::PI / MODULUS as f64;
/// The block's per-step rotation bound, in radians per unit `Δ`
/// (`Mamba3Config::rotation_range · π`, at its default of 2): an angle `θ` is
/// asked for as `ϑ = atanh(θ / MAX_ANGLE)`, which for a detent is comfortably
/// inside `tanh`'s slope.
const MAX_ANGLE: f64 = 2.0 * std::f64::consts::PI;
/// `Δ` for every head and every symbol. Fixed at 1 so the per-step angle is
/// `π·tanh(ϑ)` outright and `γ = λ·Δ = 1` writes `B` unscaled.
const DELTA: f64 = 1.0;
/// `Â` on a `±` symbol. `A = −softplus(Â)` and the block floors `|A|` at its
/// `a_floor` (`1e-4`), so any value this negative leaves the decay at
/// `ᾱ = exp(−1e-4)` — the flattest hold the block allows.
const A_HOLD_RAW: f64 = -20.0;
/// `−A` on a `RESET`: `ᾱ = e⁻²⁰` erases what the state held.
const A_WIPE: f64 = 20.0;
/// `λ̂`, large enough that `λ = σ(λ̂) ≈ 1`: the trapezoid's left-endpoint weight
/// `β = (1−λ)Δᾱ` vanishes and only the current token is written.
const LAMBDA_RAW: f64 = 20.0;
/// `x(R) = 1` — the write. `x(±) = 0` exactly (`silu(0) = 0`), so a turn writes
/// nothing and only advances the phase.
const X_WRITE: f64 = 1.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// Class-logit gain on the two readout axes.
const OUT_GAIN: f64 = 3.0;
/// `x₀(±) = ±V` for the rotation-free counter of claim (3). Bounded by silu's
/// floor (`min silu = -0.2785`), which is what lets the two turns be exactly
/// symmetric.
const V: f64 = 0.2;

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
            let pivot = m[col];
            for (k, entry) in m[row].iter_mut().enumerate().skip(col) {
                *entry -= f * pivot[k];
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
// the hand-built models
// ---------------------------------------------------------------------------

/// The three symbol embeddings, each of norm `√2` so the layer's pre-`RmsNorm`
/// (`γ = 1`) passes them through unchanged. Indexed by [`MINUS`]/[`PLUS`]/[`RESET`].
const EMBED: [[f64; 2]; NUM_SYMBOLS] = [
    [std::f64::consts::SQRT_2, 0.0],
    [0.0, std::f64::consts::SQRT_2],
    [-1.0, -1.0],
];

/// What drives the block's rotation.
#[derive(Clone, Copy, Debug)]
enum Turn {
    /// The real thing: `ϑ` reads the symbol, so a `±` turns the state by one
    /// detent and `R` does not turn it at all.
    Selective,
    /// The ablation: `ϑ` is a constant, so **every** symbol turns the state by
    /// the same angle — vanilla RoPE, a phase that counts positions.
    Fixed(f64),
}

/// What the network's head reads out.
#[derive(Clone, Copy, Debug)]
enum Head {
    /// The phase decoder: logit `j` ∝ `cos(φ − 2πj/MODULUS)`.
    Decoder,
    /// Pass the block's two output axes through unchanged, so a test can search
    /// over every readout the head could have expressed.
    Probe,
}

/// The model config with the rotation switched off (`RotationKind::Real1D`
/// makes the transition real), everything else identical. The block then
/// projects no rotation channels at all, so its `in_proj` is one column
/// narrower than the rotating model's.
fn config_without_rotation() -> MambaLatentNetConfig {
    let mut cfg = crate::model::model_config();
    let MambaLatentNetConfig::Mamba3 { mamba_block, .. } = &mut cfg else {
        unreachable!("reset-rotor configures the Mamba-3 variant")
    };
    *mamba_block = mamba_block
        .clone()
        .with_rotation(burn_mamba::mamba3::prelude::RotationKind::Real1D);
    cfg
}

/// Everything the two hand-built blocks share: the network in-projection, the
/// layer norm, the (identity) block out-projection, and the class head.
///
/// `channels` gives, for each of the block's `d_in_proj` channels, its target
/// value at (`MINUS`, `PLUS`, `RESET`) *before* the channel's own
/// activation — the in-projection is then solved for exactly, three symbols
/// through a 2-D token plus a bias.
fn build(
    device: &Device,
    cfg: &MambaLatentNetConfig,
    channels: &[[f64; 3]],
    c_bias: [[f64; 2]; 2],
    d_h: [f64; 2],
    head: Head,
) -> MambaLatentNet {
    let mut model = ModelConfigExt::init(cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("reset-rotor configures the Mamba-3 variant")
    };

    // ── network in_proj: one-hot → the symbol embedding ──────────────────────
    net.in_proj.weight = Param::from_tensor(t1(&EMBED.concat(), [NUM_SYMBOLS, 2], device));
    net.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([2]), device)));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([2]), device));
    let block = &mut layer.mamba_block;

    // ── block in_proj: one affine functional per channel ─────────────────────
    let rows = [
        [EMBED[MINUS][0], EMBED[MINUS][1], 1.0],
        [EMBED[PLUS][0], EMBED[PLUS][1], 1.0],
        [EMBED[RESET][0], EMBED[RESET][1], 1.0],
    ];
    let n_ch = channels.len();
    let mut w = vec![0.0f64; 2 * n_ch];
    let mut b = vec![0.0f64; n_ch];
    for (ch, target) in channels.iter().enumerate() {
        let [w0, w1, bias] = solve3(rows, *target);
        w[ch] = w0; // weight is [d_model, out]: row i, column ch
        w[n_ch + ch] = w1;
        b[ch] = bias;
    }
    block.in_proj.weight = Param::from_tensor(t1(&w, [2, n_ch], device));
    block.in_proj.bias = Some(Param::from_tensor(t1(&b, [n_ch], device)));

    // ── Δ bias, D, QK-norm scales, B/C biases ────────────────────────────────
    // Δ and A are entirely data-dependent here, so the bias is zero.
    block.dt_bias_h = Param::from_tensor(Tensor::zeros(Shape::new([2]), device));
    block.d_h = Param::from_tensor(t1(&d_h, [2], device));
    block.b_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([2]), device));
    block.c_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([2]), device));
    // B is the same unit vector for both heads; C differs per head through this
    // bias alone — that is what puts the two readouts a quarter turn apart.
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([2, 1, 2]), device));
    block.c_bias_hmr = Param::from_tensor(t1(&c_bias.concat(), [2, 1, 2], device));

    // ── block out-projection: the identity ───────────────────────────────────
    block.out_proj.weight = Param::from_tensor(t1(&[1.0, 0.0, 0.0, 1.0], [2, 2], device));
    block.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([2]), device)));

    // ── the class head ───────────────────────────────────────────────────────
    // `ignore_last_residual` means the block's output is all the head sees.
    let w_out: Vec<f64> = match head {
        // logit_j ∝ cos(φ − ψ_j): the phase decoder, ψ_j = 2πj/MODULUS.
        Head::Decoder => {
            let psi = |j: usize| DETENT * j as f64;
            let mut v = Vec::with_capacity(2 * NUM_CLASSES);
            v.extend((0..NUM_CLASSES).map(|j| OUT_GAIN * psi(j).cos()));
            v.extend((0..NUM_CLASSES).map(|j| -OUT_GAIN * psi(j).sin()));
            v
        }
        // the two axes, verbatim
        Head::Probe => {
            let mut v = vec![0.0; 2 * NUM_CLASSES];
            v[0] = 1.0;
            v[NUM_CLASSES + 1] = 1.0;
            v
        }
    };
    net.out_proj.weight = Param::from_tensor(t1(&w_out, [2, NUM_CLASSES], device));
    net.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([NUM_CLASSES]),
        device,
    )));

    model
}

/// The rotor block: `R` writes the zero detent, `±` turn the phase, the two
/// heads read `cos` and `−sin` of the turn since that write.
fn handmade_rotor(device: &Device, turn: Turn, head: Head) -> MambaLatentNet {
    let cfg = crate::model::model_config();
    // ϑ, shared by both heads and scaled by each head's Δ (= 1 here): the
    // per-step angle is π·tanh(ϑ).
    let theta = match turn {
        Turn::Selective => {
            let t = (DETENT / MAX_ANGLE).atanh();
            [-t, t, 0.0] // −one detent, +one detent, and R does not turn
        }
        Turn::Fixed(omega) => [(omega / MAX_ANGLE).atanh(); 3],
    };
    // Channel order: [z(2) | x(2) | B_raw(2) | C_raw(2) | dt(2) | A(2) | λ(2) | ϑ(1)].
    let channels: Vec<[f64; 3]> = vec![
        [Z_PRE; 3],                                                    // z, head 0
        [Z_PRE; 3],                                                    // z, head 1
        [0.0, 0.0, silu_inv(X_WRITE)],                                 // x, head 0
        [0.0, 0.0, silu_inv(X_WRITE)],                                 // x, head 1
        [1.0; 3],                                                      // B, axis 0
        [0.0; 3],                                                      // B, axis 1
        [1.0; 3],                                                      // C, axis 0
        [0.0; 3],                                                      // C, axis 1
        [softplus_inv(DELTA); 3],                                      // Δ, head 0
        [softplus_inv(DELTA); 3],                                      // Δ, head 1
        [A_HOLD_RAW, A_HOLD_RAW, softplus_inv(A_WIPE)],              // A, head 0
        [A_HOLD_RAW, A_HOLD_RAW, softplus_inv(A_WIPE)],              // A, head 1
        [LAMBDA_RAW; 3],                                               // λ, head 0
        [LAMBDA_RAW; 3],                                               // λ, head 1
        theta,                                                         // ϑ
    ];
    // C after QK-norm is (√2, 0) for both heads; head 1's bias turns it a
    // quarter turn, to (0, √2).
    let c_bias = [[0.0, 0.0], [-std::f64::consts::SQRT_2, std::f64::consts::SQRT_2]];
    build(device, &cfg, &channels, c_bias, [0.0, 0.0], head)
}

/// The rotation-free counter: the strongest **real** state this block can hold.
/// Head 0 accumulates `±V` per turn (decaying by `alpha` per step, wiped by
/// `R`); head 1 is a constant reference. That is a Mamba-2-strength selective
/// SSM — [`crate::model`]'s block with its complex transition removed.
fn handmade_counter(device: &Device, alpha: f64) -> MambaLatentNet {
    let cfg = config_without_rotation();
    // ᾱ = exp(Δ·A) with Δ = 1 ⇒ A = ln(alpha).
    let a_hold = -alpha.ln();
    let channels: Vec<[f64; 3]> = vec![
        [Z_PRE; 3],                                             // z, head 0
        [Z_PRE; 3],                                             // z, head 1
        [silu_inv(-V), silu_inv(V), 0.0],                       // x, head 0 — the turn
        [silu_inv(1.0); 3],                                     // x, head 1 — the reference
        [1.0; 3],                                               // B, axis 0
        [0.0; 3],                                               // B, axis 1
        [1.0; 3],                                               // C, axis 0
        [0.0; 3],                                               // C, axis 1
        [softplus_inv(DELTA); 3],                               // Δ, head 0
        [softplus_inv(1e-6); 3],                                // Δ, head 1 — never writes
        [softplus_inv(a_hold), softplus_inv(a_hold), softplus_inv(A_WIPE)], // A, head 0
        [A_HOLD_RAW; 3],                                        // A, head 1
        [LAMBDA_RAW; 3],                                        // λ, head 0
        [LAMBDA_RAW; 3],                                        // λ, head 1
        // No ϑ channel: `Real1D` projects none.
    ];
    // D₀ = 0 (head 0 reads the state alone), D₁ = 1 (head 1 *is* its skip).
    build(
        device,
        &cfg,
        &channels,
        [[0.0, 0.0], [0.0, 0.0]],
        [0.0, 1.0],
        Head::Probe,
    )
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

const FAMILIES: [(&str, Family); 3] = [
    ("random", Family::Random),
    ("drift", Family::Drift),
    ("balanced", Family::Balanced),
];

/// Run `model` over `count` sequences of one family; return the per-position
/// output channels and the targets.
fn run(
    model: &MambaLatentNet,
    family: Family,
    count: usize,
    device: &Device,
) -> (Vec<[f64; NUM_CLASSES]>, Vec<i64>) {
    let items: Vec<_> = ResetRotorDataset::new(count, SEQ_LENGTH, family, 0xE7A1)
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
        MambaSsdPath::Mamba3(Mamba3SsdPath::Minimal(None)),
        None,
    );
    let n = count * SEQ_LENGTH;
    let flat = out
        .reshape([n, NUM_CLASSES])
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let channels = flat
        .chunks_exact(NUM_CLASSES)
        .map(|c| std::array::from_fn(|i| c[i] as f64))
        .collect();
    let targets = items.iter().flat_map(|i| i.targets.clone()).collect();
    (channels, targets)
}

/// Per-position accuracy of the model's own head (argmax over the class logits).
fn accuracy(model: &MambaLatentNet, family: Family, count: usize, device: &Device) -> f64 {
    let (channels, targets) = run(model, family, count, device);
    let hits = channels
        .iter()
        .zip(&targets)
        .filter(|(logits, t)| {
            let pred = (0..NUM_CLASSES)
                .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap())
                .unwrap();
            pred as i64 == **t
        })
        .count();
    hits as f64 / targets.len() as f64
}

/// Accuracy of the best lookup table from a discrete code to a class.
fn best_lookup(codes: &[usize], targets: &[i64], num_codes: usize) -> f64 {
    let mut tally = vec![[0u64; NUM_CLASSES]; num_codes];
    for (&c, &t) in codes.iter().zip(targets) {
        tally[c][t as usize] += 1;
    }
    let total: u64 = tally.iter().flatten().sum();
    let best: u64 = tally.iter().map(|row| *row.iter().max().unwrap()).sum();
    best as f64 / total as f64
}

/// Bin `values` into `bins` equal-width buckets over their observed range.
fn bin(values: &[f64], bins: usize) -> Vec<usize> {
    let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(1e-12);
    values
        .iter()
        .map(|v| ((((v - lo) / span) * bins as f64) as usize).min(bins - 1))
        .collect()
}

/// Per-class counts per bin, as a prefix sum: `p[i][c]` = items in bins `< i`.
fn prefix(codes: &[usize], targets: &[i64], bins: usize) -> Vec<[u64; NUM_CLASSES]> {
    let mut p = vec![[0u64; NUM_CLASSES]; bins + 1];
    for (&c, &t) in codes.iter().zip(targets) {
        p[c + 1][t as usize] += 1;
    }
    for i in 1..=bins {
        let prev = p[i - 1];
        for (acc, add) in p[i].iter_mut().zip(&prev) {
            *acc += add;
        }
    }
    p
}

/// The best accuracy any readout of a **scalar** channel (plus a constant one)
/// can reach: the argmax of `NUM_CLASSES` affine functions of a scalar cuts the
/// axis into at most that many intervals, so this maximises over every such cut
/// and every class assignment.
fn best_interval_accuracy(values: &[f64], targets: &[i64]) -> f64 {
    const BINS: usize = 256;
    let p = prefix(&bin(values, BINS), targets, BINS);
    let seg = |a: usize, b: usize| -> u64 {
        (0..NUM_CLASSES)
            .map(|c| p[b][c] - p[a][c])
            .max()
            .unwrap_or(0)
    };
    let total = targets.len() as f64;
    let mut best = 0u64;
    for i in 0..=BINS {
        for j in i..=BINS {
            best = best.max(seg(0, i) + seg(i, j) + seg(j, BINS));
        }
    }
    best as f64 / total
}

/// The best accuracy any readout of the block's output **direction** can reach:
/// the argmax of `NUM_CLASSES` linear functions of a 2-D vector cuts the plane
/// into that many sectors, so this maximises over every triple of sector
/// boundaries and every class assignment.
fn best_sector_accuracy(channels: &[[f64; NUM_CLASSES]], targets: &[i64]) -> f64 {
    const BINS: usize = 60;
    let two_pi = 2.0 * std::f64::consts::PI;
    let angles: Vec<f64> = channels
        .iter()
        .map(|o| o[1].atan2(o[0]).rem_euclid(two_pi))
        .collect();
    let codes: Vec<usize> = angles
        .iter()
        .map(|a| (((a / two_pi) * BINS as f64) as usize).min(BINS - 1))
        .collect();
    // circular prefix over two laps, so an arc may wrap past bin 0
    let mut p = vec![[0u64; NUM_CLASSES]; 2 * BINS + 1];
    for (&c, &t) in codes.iter().zip(targets) {
        p[c + 1][t as usize] += 1;
        p[c + BINS + 1][t as usize] += 1;
    }
    for i in 1..=2 * BINS {
        let prev = p[i - 1];
        for (acc, add) in p[i].iter_mut().zip(&prev) {
            *acc += add;
        }
    }
    let seg = |a: usize, b: usize| -> u64 {
        (0..NUM_CLASSES)
            .map(|c| p[b][c] - p[a][c])
            .max()
            .unwrap_or(0)
    };
    let mut best = 0u64;
    for i in 0..BINS {
        for j in i + 1..=i + BINS {
            for k in j..=i + BINS {
                best = best.max(seg(i, j) + seg(j, k) + seg(k, i + BINS));
            }
        }
    }
    best as f64 / targets.len() as f64
}

// ---------------------------------------------------------------------------
// 1. the hand-built solution
// ---------------------------------------------------------------------------

/// Every weight written down in closed form; no training anywhere.
#[test]
fn handmade_block_solves_every_family() {
    let device = Device::default();
    let model = handmade_rotor(&device, Turn::Selective, Head::Decoder);
    println!(
        "hand-built rotating block ({} params):",
        model.num_params()
    );
    let mut worst = 1.0f64;
    for (name, family) in FAMILIES {
        let acc = accuracy(&model, family, 256, &device);
        println!("  {name:<10} {:6.2}%", 100.0 * acc);
        worst = worst.min(acc);
    }
    assert!(worst > 0.995, "hand-built solution is not exact: {worst}");
}

// ---------------------------------------------------------------------------
// 2. no fixed rotation reaches it
// ---------------------------------------------------------------------------

/// Sweep the per-step angle with `ϑ`'s **data dependence switched off** — the
/// one changed knob — and give each angle the best readout its output plane
/// admits.
///
/// A fixed rotation is vanilla RoPE: the phase it accumulates between the reset
/// and the read is `ω · (positions since the reset)`, which says nothing about
/// how many turns those positions contained. `balanced` is built to show that
/// directly — its turn count is decorrelated from the position — and `drift`,
/// where a fixed rotation comes closest to being right (most steps really do
/// turn the same way), is no better off once the few opposite ones have pushed
/// the count off the position by more than a detent.
#[test]
fn no_fixed_rotation_solves_the_task() {
    let device = Device::default();
    let omegas: Vec<f64> = (1..=12).map(|k| k as f64 * std::f64::consts::PI / 12.0).collect();

    println!("fixed-rotation sweep (best readout per ω), accuracy per family:");
    println!("      ω/π    random     drift   balanced      worst");
    let mut best_worst = 0.0f64;
    for omega in omegas {
        let model = handmade_rotor(&device, Turn::Fixed(omega), Head::Probe);
        let accs: Vec<f64> = FAMILIES
            .iter()
            .map(|(_, f)| {
                let (channels, targets) = run(&model, *f, 128, &device);
                best_sector_accuracy(&channels, &targets)
            })
            .collect();
        let worst = accs.iter().cloned().fold(1.0f64, f64::min);
        println!(
            "  {:7.4}  {:6.2}%   {:6.2}%    {:6.2}%    {:6.2}%",
            omega / std::f64::consts::PI,
            100.0 * accs[0],
            100.0 * accs[1],
            100.0 * accs[2],
            100.0 * worst
        );
        best_worst = best_worst.max(worst);
    }
    println!(
        "best worst-family accuracy over the whole sweep: {:.2}%",
        100.0 * best_worst
    );
    assert!(
        best_worst < 0.6,
        "a fixed rotation reached {best_worst:.4} — the task does not need a data-dependent one"
    );
}

// ---------------------------------------------------------------------------
// 3. no real state reaches it
// ---------------------------------------------------------------------------

/// The rotation switched off entirely, leaving a Mamba-2-strength selective SSM
/// that holds the turn count. Sweep its decay, and give each decay the best
/// readout its scalar channel admits.
///
/// The label is *periodic* in the count while any such readout is not: three
/// intervals, three residues, and `drift` runs the count through sixty-odd
/// values. Even `balanced`, which keeps it inside `±9`, is a dozen detents too
/// wide. What is left is `random`, where frequent resets keep the count near
/// zero often enough to lift the readout to ~59% — and no further.
#[test]
fn no_real_state_solves_the_task() {
    let device = Device::default();
    let alphas = [1.0 - 1e-4, 0.99, 0.95, 0.9, 0.8, 0.5, 0.2];

    println!("rotation-free sweep (best readout per ᾱ), accuracy per family:");
    println!("      ᾱ      random     drift   balanced      worst");
    let mut best_worst = 0.0f64;
    for alpha in alphas {
        let model = handmade_counter(&device, alpha);
        let accs: Vec<f64> = FAMILIES
            .iter()
            .map(|(_, f)| {
                let (channels, targets) = run(&model, *f, 128, &device);
                let values: Vec<f64> = channels.iter().map(|o| o[0]).collect();
                best_interval_accuracy(&values, &targets)
            })
            .collect();
        let worst = accs.iter().cloned().fold(1.0f64, f64::min);
        println!(
            "  {alpha:7.5}  {:6.2}%   {:6.2}%    {:6.2}%    {:6.2}%",
            100.0 * accs[0],
            100.0 * accs[1],
            100.0 * accs[2],
            100.0 * worst
        );
        best_worst = best_worst.max(worst);
    }
    println!(
        "best worst-family accuracy over the whole sweep: {:.2}%",
        100.0 * best_worst
    );
    assert!(
        best_worst < 0.6,
        "a real state reached {best_worst:.4} — the task does not need a complex transition"
    );
}

// ---------------------------------------------------------------------------
// 4. the task itself
// ---------------------------------------------------------------------------

/// Two ceilings that need no model at all.
///
/// - **memoryless**: the best a predictor that sees only the current symbol can
///   do. `R` is free (the rotor is at detent 0 whenever it appears); everything
///   else is chance.
/// - **positional**: the best a predictor that sees the current symbol *and how
///   many steps have passed since the last reset* can do. That bounds every
///   block which, like the one swept above, writes its state at the reset and
///   reads what has accumulated since: under an input-independent rotation both
///   the phase and the decay of what it reads are functions of exactly that
///   number. It is still nowhere near the block.
#[test]
fn memoryless_and_positional_ceilings() {
    let mut sym = [[0u64; NUM_CLASSES]; NUM_SYMBOLS];
    for (name, family) in FAMILIES {
        let (mut codes_sym, mut codes_pos, mut targets) = (vec![], vec![], vec![]);
        let (mut lo, mut hi) = (0i64, 0i64);
        for item in ResetRotorDataset::new(512, SEQ_LENGTH, family, 0xE7A1).iter() {
            let item = item.expect("dataset item");
            let steps = steps_since_reset(&item.symbols);
            for c in turns(&item.symbols) {
                lo = lo.min(c);
                hi = hi.max(c);
            }
            for ((&s, &k), &t) in item
                .symbols
                .iter()
                .zip(&steps)
                .zip(&labels(&item.symbols))
            {
                sym[s][t as usize] += 1;
                codes_sym.push(s);
                codes_pos.push(s * SEQ_LENGTH + k as usize);
                targets.push(t);
            }
        }
        // the turn count's excursion: how many detents a readout that cuts the
        // count axis into `NUM_CLASSES` intervals would have to cover
        println!(
            "  {name:<10} memoryless {:6.2}%   positional {:6.2}%   turns in [{lo}, {hi}]",
            100.0 * best_lookup(&codes_sym, &targets, NUM_SYMBOLS),
            100.0 * best_lookup(&codes_pos, &targets, NUM_SYMBOLS * SEQ_LENGTH),
        );
    }
    let names = ["-", "+", "R"];
    for (s, row) in sym.iter().enumerate() {
        println!("  symbol {}: {:?}", names[s], row);
    }
    let total: u64 = sym.iter().flatten().sum();
    let ceiling = sym.iter().map(|r| *r.iter().max().unwrap()).sum::<u64>() as f64 / total as f64;
    println!(
        "memoryless ceiling {:.2}%  (chance {:.2}%, hand-built block 100%)",
        100.0 * ceiling,
        100.0 / NUM_CLASSES as f64
    );
    assert!(
        ceiling < 0.6,
        "the current symbol nearly gives the answer: {ceiling}"
    );
}
