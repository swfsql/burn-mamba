//! The four claims this example rests on, measured.
//!
//! 1. A **hand-built `micro_steps = 2` block solves the task exactly** — no
//!    fitting, every weight written down in closed form. It is `reset-spinor`'s
//!    construction with one micro-step per symbol: the token's two symbols are
//!    two recurrence steps, so its transition is the group product.
//! 2. The **same construction at `micro_steps = 1`** — the one config change —
//!    does not, and cannot be repaired by tuning: a single step's rotation
//!    generator is an affine functional of the token, so the two symbols'
//!    generators can only *add*. Measured through the identical head and
//!    through the **best readout its output admits**, and swept over the
//!    generator scale so the number bounds the construction rather than one
//!    setting of it.
//! 3. That obstruction is **exact, not statistical**
//!    ([`one_step_generators_add_and_cannot_reach_k`]): with the hold symbol in
//!    the alphabet a token can carry one turn alone, which pins each slot's
//!    generator to the axis of the unit it turns by; every sum of two of those
//!    lies in their plane, and `exp` of a vector in a plane has **no component**
//!    along the axis orthogonal to it — which is exactly where the pair's
//!    product (`ij = k`, and the five others) lives.
//! 4. **No order-blind model can** either, so the gap is not something the
//!    counts could have covered.
//!
//! The readouts in (2) and (4) are lookup tables **fitted on one split and
//! scored on another**, so they are ceilings a model could actually reach.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    EVAL_LENGTHS, Family, HOLD, INPUT_SIZE, NUM_CLASSES, NUM_SYMBOLS, PAIR, ProductDataset, RESET,
    SEQ_LENGTH, TURN_I, TURN_J, TURN_K, apply, counts_since_reset, labels, quaternion, two_hot,
};
use crate::model::D_MODEL;
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `Δ` for every head and every micro-step. Fixed at 1 so the rotation
/// generator is `2π·tanh(‖ϑ‖)` outright and `γ = λ·Δ = 1` writes `B` unscaled.
const DELTA: f64 = 1.0;
/// `‖ϑ‖` for a symbol that turns — a half-turn, i.e. the unit quaternion `i` (or
/// `j`) up to `cos(π/2) ≈ 4e-8`.
///
/// The block bounds one step to `rotation_range · π · Δ`, and the default range
/// of 2 makes a half-turn `tanh(‖ϑ‖) = 1/2`: interior, where `tanh` is steep and
/// the gradient is alive.
const TURN_RAW: f64 = 0.5493061443340549; // atanh(1/2)
/// `Â` on a symbol that does not reset. `A = −softplus(Â)`, floored at the
/// block's `a_floor` (`1e-4`), so this is the flattest hold the block allows.
const A_HOLD_RAW: f64 = -20.0;
/// `−A` on a [`RESET`]: `ᾱ = e⁻²⁰` erases what the state held.
const A_WIPE: f64 = 20.0;
/// `λ̂`, large enough that `λ = σ(λ̂) ≈ 1`: the trapezoid's left-endpoint weight
/// `β = (1−λ)Δᾱ` vanishes and only the current micro-step is written.
const LAMBDA_RAW: f64 = 20.0;
/// `x(R) = 1` — the write. Every other symbol writes nothing (`silu(0) = 0`).
const X_WRITE: f64 = 1.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// Class-logit gain on the four readout axes.
const OUT_GAIN: f64 = 3.0;

/// `state_rank`, and the number of heads: one per quaternion component.
const N: usize = 4;
/// `per_head_dim` — the construction uses each head's **first** value channel.
const P: usize = 2;
/// `d_inner = nheads · per_head_dim`.
const D_INNER: usize = N * P;

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

fn t1<const D: usize>(v: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<1>::from_floats(f.as_slice(), device).reshape(shape)
}

// ---------------------------------------------------------------------------
// the hand-built model
// ---------------------------------------------------------------------------

/// One in-projection channel, as the values it must take **before** its own
/// activation — one per symbol, per slot.
///
/// The two slots occupy disjoint halves of `d_model`, each holding its five
/// symbols on a regular 4-simplex (see [`embedding`]), so every channel of the
/// block's in-projection realises `a_vals[a] + b_vals[b]` exactly, in closed
/// form. That additivity is not a convenience — it is the whole constraint a
/// single step is under, and why [`channels_single`] has to fold both symbols
/// into one number where [`channels_product`] does not.
#[derive(Clone, Copy)]
struct Ch {
    a: [f64; NUM_SYMBOLS],
    b: [f64; NUM_SYMBOLS],
}

impl Ch {
    /// The same value at every token, split evenly across the two slots.
    fn konst(v: f64) -> Self {
        Ch {
            a: [v / 2.0; NUM_SYMBOLS],
            b: [v / 2.0; NUM_SYMBOLS],
        }
    }
    /// A channel that reads **one** slot: micro-step `j` reads slot `j`.
    fn slot(slot: usize, vals: [f64; NUM_SYMBOLS]) -> Self {
        let zero = [0.0; NUM_SYMBOLS];
        match slot {
            0 => Ch { a: vals, b: zero },
            _ => Ch { a: zero, b: vals },
        }
    }
    /// A channel that reads both slots — the only shape available at `u = 1`.
    fn both(a: [f64; NUM_SYMBOLS], b: [f64; NUM_SYMBOLS]) -> Self {
        Ch { a, b }
    }
}

/// What the network's head reads out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    /// The nearest-element decoder: logit `g` ∝ `⟨q, g⟩` over the eight
    /// elements of `Q₈`.
    Decoder,
    /// Pass the block's four output axes through as the first four logits, so a
    /// test can search over every readout the head could have expressed.
    Probe,
}

/// The value a per-symbol channel takes: `turns` on `i` / `j` / `k`, `hold` on
/// the hold, `reset` on the reset.
fn per_symbol(turns: [f64; 3], hold: f64, reset: f64) -> [f64; NUM_SYMBOLS] {
    [turns[0], turns[1], turns[2], hold, reset]
}

/// The `u = 2` channels: **one micro-step per symbol**, each one a plain
/// `reset-spinor` step over the slot it reads.
///
/// `x` writes only on a reset, `A` wipes only on a reset, `B` is the group's
/// unit, and the rotation is a half-turn about `x̂` for `i`, `ŷ` for `j` and
/// `ẑ` for `k` — the axes of the three units. `HOLD` does nothing at all: no
/// write, no wipe, no turn.
fn channels_product(u: usize) -> Vec<Ch> {
    assert_eq!(
        PAIR, u,
        "the hand-built product solution is one step per symbol"
    );
    let mut chs = Vec::new();
    // z — per token
    chs.extend([Ch::konst(Z_PRE); D_INNER]);
    // x·u — head h's first value channel writes on a reset; the second is unused
    for j in 0..u {
        for _h in 0..N {
            chs.push(Ch::slot(
                j,
                per_symbol([0.0; 3], 0.0, silu_inv(X_WRITE)),
            ));
            chs.push(Ch::konst(0.0));
        }
    }
    // B·u — the group's unit, (1,0,0,0); QK-norm scales it to (2,0,0,0)
    for _j in 0..u {
        chs.extend((0..N).map(|r| Ch::konst(f64::from(r == 0))));
    }
    // C — per token, the same unit; the per-head bias moves head h onto e_h
    chs.extend((0..N).map(|r| Ch::konst(f64::from(r == 0))));
    // Δ·u, A·u, λ·u — per head
    for _j in 0..u {
        chs.extend([Ch::konst(softplus_inv(DELTA)); N]);
    }
    for j in 0..u {
        chs.extend(
            [Ch::slot(
                j,
                per_symbol([A_HOLD_RAW; 3], A_HOLD_RAW, softplus_inv(A_WIPE)),
            ); N],
        );
    }
    for _j in 0..u {
        chs.extend([Ch::konst(LAMBDA_RAW); N]);
    }
    // rotation·u — three generator channels per head, one per axis: `i` turns
    // about x̂, `j` about ŷ, `k` about ẑ, and no two of them commute.
    for j in 0..u {
        for _h in 0..N {
            for axis in 0..3 {
                let mut turns = [0.0; 3];
                turns[axis] = TURN_RAW;
                chs.push(Ch::slot(j, per_symbol(turns, 0.0, 0.0)));
            }
        }
    }
    chs
}

/// The `u = 1` channels: the same construction with **one** step per token, and
/// every knob a single step still has set as favourably as it can be.
///
/// Four of the five jobs survive the fold, because they only ever need to know
/// *whether* a slot holds a reset, which is additive:
///
/// - `A` wipes when **either** slot resets (`+40 / −20`, so any single `R`
///   clears the sum);
/// - `x` writes on the same condition (`+10 / −5`);
/// - `B` writes the element the token ends on — the identity when slot `b`
///   resets or holds, and `q_b` when slot `a` reset and slot `b` turned, which
///   *compensates* for the turn the write can no longer sit before. This is
///   strictly better than the naive fold and is what makes every
///   reset-containing token exact.
///
/// The fifth is the rotation, and it is the one that cannot survive: the
/// generator is `gen_scale · (v_a + w_b)`, a **sum**, where the token needs the
/// **product** `exp(w_b) ⊗ exp(v_a)`. `gen_scale` is what the caller sweeps —
/// no scale can fix it (see [`one_step_generators_add_and_cannot_reach_k`]),
/// but sweeping it means the measured number bounds the construction rather
/// than one setting of it.
fn channels_single(gen_scale: f64) -> Vec<Ch> {
    let g = TURN_RAW * gen_scale;
    // write iff either slot resets; wipe on the same condition
    let write = per_symbol([-5.0; 3], -5.0, 10.0);
    let wipe = per_symbol([-20.0; 3], -20.0, 40.0);
    let mut chs = Vec::new();
    // z — as before
    chs.extend([Ch::konst(Z_PRE); D_INNER]);
    // x — write iff either slot resets
    for _h in 0..N {
        chs.push(Ch::both(write, write));
        chs.push(Ch::konst(0.0));
    }
    // B — the element the token ends on (see the doc comment): the identity
    // when slot b holds or resets, and q_b when it turns
    let none = [0.0; NUM_SYMBOLS];
    chs.push(Ch::both(none, per_symbol([0.0; 3], 1.0, 1.0))); // w ← 1
    for unit in 0..3 {
        let mut turns = [0.0; 3];
        turns[unit] = 1.0;
        chs.push(Ch::both(none, per_symbol(turns, 0.0, 0.0))); // x/y/z ← q_b
    }
    // C — per token, the unit
    chs.extend((0..N).map(|r| Ch::konst(f64::from(r == 0))));
    // Δ, A, λ
    chs.extend([Ch::konst(softplus_inv(DELTA)); N]);
    chs.extend([Ch::both(wipe, wipe); N]);
    chs.extend([Ch::konst(LAMBDA_RAW); N]);
    // rotation — the sum of the two slots' generators
    for _h in 0..N {
        for axis in 0..3 {
            let mut turns = [0.0; 3];
            turns[axis] = g;
            let vals = per_symbol(turns, 0.0, 0.0);
            chs.push(Ch::both(vals, vals));
        }
    }
    chs
}

/// Half of `d_model`, i.e. the dimensions one slot owns.
const HALF: usize = D_MODEL / PAIR;
/// The radius the simplex is scaled to, so that a token — one vertex per slot —
/// has RMS exactly 1 over `d_model` and the layer's pre-`RmsNorm` (`γ = 1`)
/// passes it through unchanged.
const RHO: f64 = 2.0;

/// The five symbols of one slot, as the vertices of a regular 4-simplex of norm
/// [`RHO`] in that slot's [`HALF`] dimensions.
///
/// Built from the Helmert basis of `{x ∈ R⁵ : Σx = 0}`, so the vertices are
/// equidistant, sum to zero, and are affinely independent — which is what makes
/// an arbitrary per-symbol value an affine functional of the embedding, and the
/// weight below a closed form rather than a linear solve.
fn simplex() -> [[f64; HALF]; NUM_SYMBOLS] {
    let mut v = [[0.0f64; HALF]; NUM_SYMBOLS];
    for (col, row) in v.iter_mut().enumerate() {
        for (k, coord) in row.iter_mut().enumerate() {
            let k = k + 1; // Helmert vectors are 1-indexed
            let norm = (k * (k + 1)) as f64;
            *coord = if col < k {
                1.0 / norm.sqrt()
            } else if col == k {
                -(k as f64) / norm.sqrt()
            } else {
                0.0
            };
        }
        // the projection has norm √(1 − 1/5); scale it to RHO
        let scale = RHO / (1.0 - 1.0 / NUM_SYMBOLS as f64).sqrt();
        for coord in row.iter_mut() {
            *coord *= scale;
        }
    }
    v
}

/// The weight (in one slot's dimensions) and the offset that realise the
/// per-symbol values `vals` on the simplex.
///
/// For a regular simplex with `Σ v_s = 0`, `‖v_s‖ = ρ` and `v_s·v_r = −ρ²/4`,
/// the functional `w·v_s + mean` reproduces `vals` at
/// `w = 4/(5ρ²) · Σ_s (vals_s − mean) v_s`.
fn functional(vals: &[f64; NUM_SYMBOLS]) -> ([f64; HALF], f64) {
    let mean = vals.iter().sum::<f64>() / NUM_SYMBOLS as f64;
    let c = 4.0 / (NUM_SYMBOLS as f64 * RHO * RHO);
    let vertices = simplex();
    let mut w = [0.0f64; HALF];
    for (s, v) in vertices.iter().enumerate() {
        for (d, coord) in v.iter().enumerate() {
            w[d] += c * (vals[s] - mean) * coord;
        }
    }
    (w, mean)
}

/// The network's input projection: each slot's one-hot symbol maps to its
/// vertex of that slot's simplex, inside that slot's half of `d_model`.
fn embedding(device: &Device) -> Tensor<2> {
    let vertices = simplex();
    let mut w = vec![0.0f64; INPUT_SIZE * D_MODEL];
    for slot in 0..PAIR {
        for s in 0..NUM_SYMBOLS {
            let row = slot * NUM_SYMBOLS + s;
            for d in 0..HALF {
                w[row * D_MODEL + slot * HALF + d] = vertices[s][d];
            }
        }
    }
    t1(&w, [INPUT_SIZE, D_MODEL], device)
}

/// Build the block by hand at the given `micro_steps`.
fn handmade(device: &Device, micro_steps: usize, head: Head, gen_scale: f64) -> MambaLatentNet {
    let cfg = crate::model::model_config(micro_steps);
    let MambaLatentNetConfig::Mamba3 { mamba_block, .. } = &cfg else {
        unreachable!("spinor-product configures the Mamba-3 variant")
    };
    let mamba_block = mamba_block.clone();
    let mut model = ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("spinor-product configures the Mamba-3 variant")
    };

    // ── network in_proj: two one-hot slots → the token embedding ─────────────
    net.in_proj.weight = Param::from_tensor(embedding(device));
    net.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([D_MODEL]),
        device,
    )));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([D_MODEL]), device));
    let block = &mut layer.block;

    // ── block in_proj: one affine functional per channel ─────────────────────
    // Channel order: [z | x·u | B·u | C | Δ·u | A·u | λ·u | ϑ·u], with every
    // per-micro-step segment laid out micro-step by micro-step.
    let channels = if micro_steps == 1 {
        channels_single(gen_scale)
    } else {
        channels_product(micro_steps)
    };
    let n_ch = channels.len();
    assert_eq!(
        mamba_block.d_in_proj(),
        n_ch,
        "the hand-built channels must tile the fused in-projection"
    );
    let mut w = vec![0.0f64; D_MODEL * n_ch];
    let mut bias = vec![0.0f64; n_ch];
    for (ch, spec) in channels.iter().enumerate() {
        // each slot's values are an affine functional of its own simplex; the
        // two offsets share the channel's single bias, which is all they need
        let (w_a, off_a) = functional(&spec.a);
        let (w_b, off_b) = functional(&spec.b);
        for d in 0..HALF {
            w[d * n_ch + ch] = w_a[d];
            w[(HALF + d) * n_ch + ch] = w_b[d];
        }
        bias[ch] = off_a + off_b;
    }
    block.in_proj.weight = Param::from_tensor(t1(&w, [D_MODEL, n_ch], device));
    block.in_proj.bias = Some(Param::from_tensor(t1(&bias, [n_ch], device)));

    // ── Δ bias, D, QK-norm scales, B/C biases ────────────────────────────────
    // Δ and A are entirely data-dependent here, so the bias is zero.
    block.dt_bias_h = Param::from_tensor(Tensor::zeros(Shape::new([N]), device));
    block.d_h = Param::from_tensor(Tensor::zeros(Shape::new([N]), device));
    block.b_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([N]), device));
    block.c_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([N]), device));
    // QK-Norm over `state_rank = 4` maps (1,0,0,0) to (2,0,0,0), so C starts at
    // twice the identity quaternion for every head; head h's bias moves it to
    // twice the h-th basis quaternion — the only per-head weight in the whole
    // construction, and what makes the four heads read four components of one
    // state.
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([N, 1, N]), device));
    let c_bias: Vec<f64> = (0..N)
        .flat_map(|h| (0..N).map(move |r| 2.0 * (f64::from(r == h) - f64::from(r == 0))))
        .collect();
    block.c_bias_hmr = Param::from_tensor(t1(&c_bias, [N, 1, N], device));

    // ── block out-projection: head h's first value channel → axis h ──────────
    let mut sel = vec![0.0f64; D_INNER * D_MODEL];
    for h in 0..N {
        sel[(h * P) * D_MODEL + h] = 1.0;
    }
    block.out_proj.weight = Param::from_tensor(t1(&sel, [D_INNER, D_MODEL], device));
    block.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([D_MODEL]),
        device,
    )));

    // ── the class head: nearest group element ────────────────────────────────
    // logit_g ∝ ⟨q, g⟩ over the eight elements of Q₈. `ignore_last_residual`
    // means the block's output is all the head sees.
    let mut w_out = vec![0.0f64; D_MODEL * NUM_CLASSES];
    match head {
        Head::Decoder => {
            for class in 0..NUM_CLASSES {
                let q = quaternion(class as i64);
                for (r, qr) in q.iter().enumerate() {
                    w_out[r * NUM_CLASSES + class] = OUT_GAIN * qr;
                }
            }
        }
        // the four axes, verbatim, in the first four logits
        Head::Probe => {
            for r in 0..N {
                w_out[r * NUM_CLASSES + r] = 1.0;
            }
        }
    }
    net.out_proj.weight = Param::from_tensor(t1(&w_out, [D_MODEL, NUM_CLASSES], device));
    net.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([NUM_CLASSES]),
        device,
    )));

    model
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

const FAMILIES: [(&str, Family); 3] = [
    ("random", Family::Random),
    ("shuffle", Family::Shuffle),
    ("runs", Family::Runs),
];

/// Seed of the split a lookup table is **fitted** on.
const FIT: u64 = 0x51D3;
/// Seed of the split everything is **scored** on.
const EVAL: u64 = 0xE7A1;

/// Run `model` over `count` sequences of one family at `length` tokens; return
/// the per-token output channels and the targets.
fn run(
    model: &MambaLatentNet,
    family: Family,
    count: usize,
    length: usize,
    seed: u64,
    device: &Device,
) -> (Vec<[f64; NUM_CLASSES]>, Vec<i64>) {
    let items: Vec<_> = ProductDataset::new(count, length, family, seed)
        .iter()
        .map(|i| i.expect("dataset item"))
        .collect();
    let inputs = Tensor::stack(
        items
            .iter()
            .map(|i| two_hot(&i.symbols, device))
            .collect::<Vec<_>>(),
        0,
    );
    let (out, _c) = model.forward(
        inputs,
        None,
        MambaSsdPath::Mamba3(Mamba3SsdPath::Minimal(None)),
        None,
    );
    let n = count * length;
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

/// Per-token accuracy of the model's own head (argmax over the class logits).
fn accuracy(
    model: &MambaLatentNet,
    family: Family,
    count: usize,
    length: usize,
    device: &Device,
) -> f64 {
    let (channels, targets) = run(model, family, count, length, EVAL, device);
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

/// Accuracy of the best lookup table from a discrete code to a class — fitted on
/// one split, scored on another, so it is a ceiling a model could actually reach
/// rather than a memorised answer key.
///
/// Codes unseen while fitting fall back to the fit split's majority class.
fn best_lookup(fit: (&[usize], &[i64]), eval: (&[usize], &[i64]), num_codes: usize) -> f64 {
    let mut tally = vec![[0u64; NUM_CLASSES]; num_codes];
    let mut overall = [0u64; NUM_CLASSES];
    for (&c, &t) in fit.0.iter().zip(fit.1) {
        tally[c][t as usize] += 1;
        overall[t as usize] += 1;
    }
    let argmax = |row: &[u64; NUM_CLASSES]| {
        (0..NUM_CLASSES).max_by_key(|&c| row[c]).expect("non-empty") as i64
    };
    let fallback = argmax(&overall);
    let table: Vec<i64> = tally
        .iter()
        .map(|row| {
            if row.iter().sum::<u64>() == 0 {
                fallback
            } else {
                argmax(row)
            }
        })
        .collect();
    let hits = eval
        .0
        .iter()
        .zip(eval.1)
        .filter(|&(&c, &t)| table[c] == t)
        .count();
    hits as f64 / eval.1.len() as f64
}

/// Quantise each output channel into `LEVELS` equal-width bins over the range
/// observed on the fit split, and pack the four into one code — a finite
/// partition of the block's output space, so the best table over it dominates
/// every linear head the model could have carried.
const LEVELS: usize = 5;
const NUM_OUTPUT_CODES: usize = LEVELS * LEVELS * LEVELS * LEVELS;

fn output_codes(channels: &[[f64; NUM_CLASSES]], range: &[(f64, f64); N]) -> Vec<usize> {
    channels
        .iter()
        .map(|o| {
            (0..N).fold(0, |code, r| {
                let (lo, hi) = range[r];
                let span = (hi - lo).max(1e-12);
                let level = ((((o[r] - lo) / span) * LEVELS as f64) as usize).min(LEVELS - 1);
                code * LEVELS + level
            })
        })
        .collect()
}

fn channel_range(channels: &[[f64; NUM_CLASSES]]) -> [(f64, f64); N] {
    std::array::from_fn(|r| {
        channels
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), o| {
                (lo.min(o[r]), hi.max(o[r]))
            })
    })
}

/// The best table over a fine partition of the block's output space.
fn best_readout(probe: &MambaLatentNet, family: Family, device: &Device) -> f64 {
    let (fit_ch, fit_t) = run(probe, family, 256, SEQ_LENGTH, FIT, device);
    let (eval_ch, eval_t) = run(probe, family, 128, SEQ_LENGTH, EVAL, device);
    let range = channel_range(&fit_ch);
    best_lookup(
        (&output_codes(&fit_ch, &range), &fit_t),
        (&output_codes(&eval_ch, &range), &eval_t),
        NUM_OUTPUT_CODES,
    )
}

/// Collect, per token, the codes an order-blind model could key on — the token
/// itself (a pair of symbols), and the `(#i, #j)` counts since the reset — with
/// the targets.
fn codes(family: Family, count: usize, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let (mut tok, mut cnt, mut targets) = (vec![], vec![], vec![]);
    for item in ProductDataset::new(count, SEQ_LENGTH, family, seed).iter() {
        let item = item.expect("dataset item");
        for ((pair, counts), &t) in item
            .symbols
            .chunks_exact(PAIR)
            .zip(counts_since_reset(&item.symbols))
            .zip(&item.targets)
        {
            tok.push(pair[0] * NUM_SYMBOLS + pair[1]);
            cnt.push(counts.iter().fold(0, |code, &n| code * COUNT_BASE + n as usize));
            targets.push(t);
        }
    }
    (tok, cnt, targets)
}

/// Number of distinct token codes: an ordered pair of symbols.
const NUM_TOKEN_CODES: usize = NUM_SYMBOLS * NUM_SYMBOLS;
/// One more than the largest count a sequence can reach.
const COUNT_BASE: usize = PAIR * SEQ_LENGTH + 1;
/// Number of distinct `(#i, #j, #k)` codes [`codes`] can emit.
const NUM_COUNT_CODES: usize = COUNT_BASE * COUNT_BASE * COUNT_BASE;

// ---------------------------------------------------------------------------
// 1. the hand-built solution
// ---------------------------------------------------------------------------

/// Every weight written down in closed form; no training anywhere. One
/// micro-step per symbol, so the token's transition is the group product.
///
/// Reported at every length in [`EVAL_LENGTHS`], because that is what
/// "composes the token" means: the block is a group tracker, so the word may be
/// as long as you like. An approximation of the same construction is not, which
/// is what the long column measures for everything else.
#[test]
fn handmade_product_block_solves_every_family() {
    let device = Device::default();
    let model = handmade(&device, PAIR, Head::Decoder, 1.0);
    println!(
        "hand-built micro_steps = {PAIR} block ({} params):",
        model.num_params()
    );
    let mut worst = 1.0f64;
    for length in EVAL_LENGTHS {
        print!("  {length:>3} tokens:");
        for (name, family) in FAMILIES {
            let acc = accuracy(&model, family, 64, length, &device);
            print!("   {name} {:6.2}%", 100.0 * acc);
            worst = worst.min(acc);
        }
        println!();
    }
    assert!(worst > 0.995, "hand-built solution is not exact: {worst}");
}

// ---------------------------------------------------------------------------
// 2. the same construction at micro_steps = 1
// ---------------------------------------------------------------------------

/// The identical construction with `micro_steps = 1` — the one config change —
/// swept over the generator scale and reported two ways.
///
/// Everything that only needs to know *whether* a slot resets survives the fold
/// (see [`channels_single`]); the rotation does not, because a single step's
/// generator is affine in the token and the group is not. The second column
/// hands the block the best table over a fine partition of its whole output
/// space, fitted on a separate split, so the number bounds every readout the
/// head could have expressed and not just the decoder this one carries.
#[test]
fn one_step_cannot_compose_a_token() {
    let device = Device::default();
    println!("the same construction at micro_steps = 1, swept over the generator scale:");
    println!("      scale    random    shuffle       runs");
    let scales = [0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5];
    let mut best = [0.0f64; 3];
    let mut best_scale = 0.0;
    for scale in scales {
        let decoder = handmade(&device, 1, Head::Decoder, scale);
        let accs: Vec<f64> = FAMILIES
            .iter()
            .map(|(_, f)| accuracy(&decoder, *f, 128, SEQ_LENGTH, &device))
            .collect();
        println!(
            "      {scale:>5.2}   {:6.2}%    {:6.2}%    {:6.2}%",
            100.0 * accs[0],
            100.0 * accs[1],
            100.0 * accs[2],
        );
        if accs.iter().sum::<f64>() > best.iter().sum::<f64>() {
            best_scale = scale;
        }
        for (n, a) in accs.iter().enumerate() {
            best[n] = best[n].max(*a);
        }
    }
    let probe = handmade(&device, 1, Head::Probe, best_scale);
    println!("  best over the sweep, against the best readout of the state (scale {best_scale}):");
    println!("      family    same head   best readout");
    let mut ceiling = 0.0f64;
    for (n, (name, family)) in FAMILIES.iter().enumerate() {
        let readout = best_readout(&probe, *family, &device);
        println!(
            "  {name:<9}      {:6.2}%         {:6.2}%",
            100.0 * best[n],
            100.0 * readout
        );
        ceiling = ceiling.max(readout);
    }
    println!(
        "best readout of the one-step state, over the families: {:.2}%",
        100.0 * ceiling
    );
    assert!(
        ceiling < 0.9,
        "the one-step twin reached {ceiling:.4} — the task does not need micro_steps"
    );
}

// ---------------------------------------------------------------------------
// 3. why no setting of it would have worked
// ---------------------------------------------------------------------------

/// Quaternion exponential: `exp(v) = cos‖v‖ + sin‖v‖ · v̂`, the map the block's
/// rotation generator goes through.
fn quat_exp(v: [f64; 3]) -> [f64; 4] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let (c, s) = (n.cos(), n.sin() / n);
    [c, s * v[0], s * v[1], s * v[2]]
}

/// The Hamilton product `p ⊗ q`.
fn quat_mul(p: [f64; 4], q: [f64; 4]) -> [f64; 4] {
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
        p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
    ]
}

/// The generators inside the block's bound (`‖ϑ‖ < rotation_range·π = 2π`) whose
/// exponential is the unit `axis` — `cos‖ϑ‖ = 0`, so `‖ϑ‖ ∈ {π/2, 3π/2}` with
/// the axis following the sign of `sin‖ϑ‖`. There are exactly two.
fn generators_of(axis: usize) -> [[f64; 3]; 2] {
    let mut small = [0.0; 3];
    let mut large = [0.0; 3];
    small[axis] = std::f64::consts::FRAC_PI_2;
    large[axis] = -3.0 * std::f64::consts::FRAC_PI_2;
    [small, large]
}

/// The obstruction, exactly.
///
/// At `micro_steps = 1` the rotation generator is an affine functional of the
/// token, and a token is two one-hot slots, so `ϑ(a, b) = v_a + w_b` for some
/// per-slot vectors — whatever the weights, the scale, or the squash, which only
/// rescale `‖ϑ‖` and leave its **direction** alone.
///
/// The hold symbol pins those vectors down: the token `(i, .)` must turn by `i`,
/// so `exp(v_i) = i`, and the only generators with that image inside the block's
/// bound lie along `±x̂` — the axis of `i`. Likewise `w_j` lies along `±ŷ`.
/// Every sum of the two therefore lies in the `xy`-plane, and `exp` of a vector
/// in a plane has **zero** component along the orthogonal axis — while the token
/// `(i, j)` needs `j·i = −k`, which is nothing *but* that component. The same
/// holds for all six ordered pairs of distinct units, since each composes to
/// `±` the third one.
///
/// Two micro-steps have no such constraint: they apply `exp(w_b) ⊗ exp(v_a)`,
/// which is the token's element on the nose.
#[test]
fn one_step_generators_add_and_cannot_reach_k() {
    let mut closest = f64::INFINITY;
    for a in 0..3 {
        for b in 0..3 {
            if a == b {
                continue;
            }
            // the third axis: the one `q_b · q_a` turns about
            let third = 3 - a - b;
            for v in generators_of(a) {
                for w in generators_of(b) {
                    let (qa, qb) = (quat_exp(v), quat_exp(w));
                    // two micro-steps: the product, exact
                    let product = quat_mul(qb, qa);
                    assert!(
                        product[1 + third].abs() > 1.0 - 1e-9,
                        "the pair composes to ± the third unit"
                    );
                    // one micro-step: the sum, whatever scale it is read at
                    for n in 0..=40 {
                        let s = f64::from(n) / 10.0;
                        let sum =
                            quat_exp([s * (v[0] + w[0]), s * (v[1] + w[1]), s * (v[2] + w[2])]);
                        assert_eq!(
                            0.0,
                            sum[1 + third],
                            "a generator in a plane cannot reach the axis orthogonal to it"
                        );
                        closest = closest.min(dist(sum, product));
                    }
                }
            }
        }
    }
    println!(
        "closest a single step gets to a two-turn token: |Δq| = {closest:.4} \
         (0 for micro_steps = 2)"
    );
    assert!(closest > 1.0, "a single step got within {closest} of it");
}

fn dist(p: [f64; 4], q: [f64; 4]) -> f64 {
    (0..4).map(|n| (p[n] - q[n]).powi(2)).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// 4. the ceiling for everything order-blind
// ---------------------------------------------------------------------------

/// The best predictors that see only the current token, or only the symbol
/// **counts** since the reset. `Q₈`'s commutator subgroup is `{±1}`, so the
/// counts fix the element up to a sign and the table has to guess wherever both
/// occur — which is what `shuffle` is built to make constant.
#[test]
fn counts_ceiling_is_the_order_blind_limit() {
    println!("order-blind ceilings, accuracy per family:");
    println!("      family    memoryless   by (#i,#j)   ambiguous");
    let mut best = 0.0f64;
    for (name, family) in FAMILIES {
        let (fit_tok, fit_cnt, fit_t) = codes(family, 2048, FIT);
        let (eval_tok, eval_cnt, eval_t) = codes(family, 512, EVAL);
        // how much of the mass sits on a (counts) cell that carries both signs
        let mut tally = vec![[0u64; NUM_CLASSES]; NUM_COUNT_CODES];
        for (&c, &t) in fit_cnt.iter().zip(&fit_t) {
            tally[c][t as usize] += 1;
        }
        let (mut split, mut total) = (0u64, 0u64);
        for row in &tally {
            let sum: u64 = row.iter().sum();
            total += sum;
            if row.iter().filter(|n| **n > 0).count() > 1 {
                split += sum;
            }
        }
        let by_counts = best_lookup((&fit_cnt, &fit_t), (&eval_cnt, &eval_t), NUM_COUNT_CODES);
        println!(
            "  {name:<9}    {:6.2}%      {:6.2}%      {:6.2}%",
            100.0 * best_lookup((&fit_tok, &fit_t), (&eval_tok, &eval_t), NUM_TOKEN_CODES),
            100.0 * by_counts,
            100.0 * split as f64 / total as f64,
        );
        best = best.max(by_counts);
    }
    println!(
        "best over the families: {:.2}%  (chance {:.2}%, hand-built block 100%)",
        100.0 * best,
        100.0 / NUM_CLASSES as f64
    );
    assert!(
        best < 0.85,
        "the symbol counts nearly give the answer: {best}"
    );
}

// ---------------------------------------------------------------------------
// 5. the dataset
// ---------------------------------------------------------------------------

/// The labels really are the `Q₈` word problem read two symbols at a time: a
/// token composes its pair in order, the pairs compose with each other, `HOLD`
/// does nothing and a reset restarts the word wherever in a token it lands.
#[test]
fn labels_are_the_paired_quaternion_word_problem() {
    // one token, two turns: (i, j) is j·i and (j, i) is i·j — and they differ
    let ij = labels(&[RESET, HOLD, TURN_I, TURN_J]);
    let ji = labels(&[RESET, HOLD, TURN_J, TURN_I]);
    assert_eq!(quaternion(ij[1]), [0.0, 0.0, 0.0, -1.0], "j·i = −k");
    assert_eq!(quaternion(ji[1]), [0.0, 0.0, 0.0, 1.0], "i·j = k");

    // the third unit is a generator like the others, and no two commute
    let ik = labels(&[RESET, HOLD, TURN_I, TURN_K]);
    let ki = labels(&[RESET, HOLD, TURN_K, TURN_I]);
    assert_eq!(quaternion(ik[1]), [0.0, 0.0, 1.0, 0.0], "k·i = j");
    assert_eq!(quaternion(ki[1]), [0.0, 0.0, -1.0, 0.0], "i·k = −j");

    // the hold is the identity, and a reset in the second slot still resets
    assert_eq!(labels(&[RESET, HOLD])[0], 0, "R then hold is the identity");
    assert_eq!(
        labels(&[RESET, TURN_I, TURN_J, RESET])[1],
        0,
        "a reset in the second slot restarts the word"
    );
    assert_eq!(
        labels(&[RESET, TURN_I, RESET, TURN_J])[1],
        labels(&[RESET, TURN_J, HOLD, HOLD])[0],
        "a reset in the first slot leaves the second symbol alone"
    );

    // order four, and consistency with the symbol-at-a-time reading
    let powers = labels(&[RESET, HOLD, TURN_I, TURN_I, TURN_I, TURN_I]);
    assert_eq!(quaternion(powers[1]), [-1.0, 0.0, 0.0, 0.0], "i² = −1");
    assert_eq!(powers[2], 0, "i⁴ = 1");

    // every element is reachable, and a reset lands on the identity when it is
    // the token's last symbol
    let mut seen = [false; NUM_CLASSES];
    for item in ProductDataset::new(64, SEQ_LENGTH, Family::Mixed, 7).iter() {
        let item = item.expect("dataset item");
        // the labels are the running product of the stream, sampled per token
        let mut state = 0i64;
        for (t, (pair, &c)) in item.symbols.chunks_exact(PAIR).zip(&item.targets).enumerate() {
            for &s in pair {
                state = apply(s, state);
            }
            assert_eq!(state, c, "token {t} label");
            seen[c as usize] = true;
            if pair[PAIR - 1] == RESET {
                assert_eq!(c, 0, "a token ending in a reset lands on the identity");
            }
        }
    }
    assert!(seen.iter().all(|s| *s), "some group element never occurs");
}
