//! The claims this example rests on, measured.
//!
//! 1. A **hand-built `Rotor4D`** Mamba-3 block solves the task exactly — no
//!    fitting, every weight written down in closed form. The block's rotation is
//!    set to conjugation (`p = q`), which is `SO(3)`, and the three swaps become
//!    three half-turns about three axes `60°` apart.
//! 2. The **same construction one rung down** (`Quaternion4D`, one enum knob)
//!    does not. Not because it forgets: a left-isoclinic state carries the
//!    *double cover* `2D₃`, which has **more** information than the label. The
//!    two lifts `±W` of one permutation are **antipodal** state vectors with the
//!    same target, and that is what a linear readout cannot merge — measured
//!    three ways: through the identical head, through the best table over its
//!    output space (which *does* recover it, and is the point), and by how
//!    completely the two lifts cancel.
//! 3. **No order-blind model can** — the ceiling from the symbol counts — and
//!    in particular not the sign character `(−1)^(#s+#t)`, which is all a
//!    *homomorphism* `S₃ → SU(2)` can carry, since `SU(2)` has exactly one
//!    element of order two and `S₃` has three.
//!
//! The readouts in (2) and (3) are lookup tables **fitted on one split and
//! scored on another**, so they are ceilings a model could actually reach, not
//! memorised labels.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    Family, NUM_CLASSES, NUM_SYMBOLS, PERMS, REF_POINT, RESET, ResetSwapDataset, SEQ_LENGTH,
    SWAP_S, SWAP_T, class_of, compose, counts_since_reset, labels, one_hot, point, quat_mul,
    swap_axis, symbol_perm, symbol_quat,
};
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `Δ` for every head and every symbol. Fixed at 1 so the rotation generator is
/// `range·π·tanh(ϑ)` outright and `γ = λ·Δ = 1` writes `B` unscaled.
const DELTA: f64 = 1.0;
/// `‖ϑ‖` for a **half-turn** — the same constant as `reset-spinor`, and for the
/// same reason: the block bounds one step to `rotation_range · π · Δ` and the
/// default range of 2 puts a half-turn at `tanh(‖ϑ‖) = 1/2`, interior, where the
/// gradient is alive.
///
/// A half-turn is all this task ever needs: a transposition has order two, and
/// under conjugation the *rotation* it induces is a `180°` turn about its axis.
const TURN_RAW: f64 = 0.5493061443340549; // atanh(1/2)
/// `Â` on a swap. `A = −softplus(Â)`, floored at the block's `a_floor` (`1e-4`),
/// so this is the flattest hold the block allows.
const A_HOLD_RAW: f64 = -20.0;
/// `−A` on a `RESET`: `ᾱ = e⁻²⁰` erases what the state held.
const A_WIPE: f64 = 20.0;
/// `λ̂`, large enough that `λ = σ(λ̂) ≈ 1`: the trapezoid's left-endpoint weight
/// `β = (1−λ)Δᾱ` vanishes and only the current token is written.
const LAMBDA_RAW: f64 = 20.0;
/// `x(R) = 1` — the write. `x(s) = x(t) = 0` exactly (`silu(0) = 0`), so a swap
/// writes nothing and only turns the state.
const X_WRITE: f64 = 1.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// Class-logit gain on the readout axes.
const OUT_GAIN: f64 = 3.0;

/// `d_model`, `state_rank` and `nheads` all equal 4 here — one head per state
/// component.
const N: usize = 4;

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

/// The three symbol embeddings: **orthogonal** vectors of norm 2, indexed by
/// [`SWAP_S`] / [`SWAP_T`] / [`RESET`].
///
/// Norm 2 in `d_model = 4` is what the layer's pre-`RmsNorm` (`γ = 1`) passes
/// through unchanged, and orthogonality is what turns the block's in-projection
/// into a lookup table: a channel that must take the values `(v_s, v_t, v_R)`
/// gets the weight `(v_s/2, v_t/2, v_R/2, 0)` and no bias.
const EMBED: [[f64; N]; NUM_SYMBOLS] = [
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
];

/// What the network's head reads out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    /// The nearest-point decoder: logit `g` ∝ `⟨state, point(g)⟩` over the six
    /// permutations.
    Decoder,
    /// Pass the block's four output axes through as the first four logits, so a
    /// test can search over every readout the head could have expressed.
    Probe,
}

/// Build the block by hand for the given rotation.
///
/// The three rotations differ only in what the rotation channels mean:
///
/// - [`RotationKind::Rotor4D`] takes six per head — a left and a right scaled
///   axis. Setting them **equal** makes the step `v ↦ q v q̄`, conjugation, i.e.
///   an honest `SO(3)` rotation by `180°` about [`swap_axis`]. This is the one
///   that works.
/// - [`RotationKind::Quaternion4D`] takes three — the left factor alone, so the
///   step is `v ↦ q v`, and `q² = −1` rather than `1`: the state runs in the
///   double cover.
/// - [`RotationKind::Complex2D`] takes two, one angle per state pair, so `s` and
///   `t` become half-turns of the two pairs — which commute.
fn handmade(device: &Device, rotation: RotationKind, head: Head) -> MambaLatentNet {
    let cfg = crate::model::model_config(rotation);
    let mut model = ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("reset-swap configures the Mamba-3 variant")
    };

    // ── network in_proj: one-hot → the symbol embedding ──────────────────────
    net.in_proj.weight = Param::from_tensor(t1(&EMBED.concat(), [NUM_SYMBOLS, N], device));
    net.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([N]), device)));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([N]), device));
    let block = &mut layer.mamba_block;

    // ── block in_proj: one affine functional per channel ─────────────────────
    // Channel order: [z(4) | x(4) | B_raw(4) | C_raw(4) | Δ(4) | A(4) | λ(4) | ϑ(2, 12 or 24)].
    // Each entry is the channel's value at (SWAP_S, SWAP_T, RESET), *before* its
    // own activation.
    let mut channels: Vec<[f64; NUM_SYMBOLS]> = Vec::new();
    channels.extend([[Z_PRE; NUM_SYMBOLS]; N]); // z
    channels.extend([[0.0, 0.0, silu_inv(X_WRITE)]; N]); // x — only R writes
    channels.extend(b_channels(rotation)); // B_raw — the vector the write stores
    channels.extend(basis_channels()); // C_raw (the per-head bias splits it)
    channels.extend([[softplus_inv(DELTA); NUM_SYMBOLS]; N]); // Δ
    channels.extend([[A_HOLD_RAW, A_HOLD_RAW, softplus_inv(A_WIPE)]; N]); // A
    channels.extend([[LAMBDA_RAW; NUM_SYMBOLS]; N]); // λ
    channels.extend(rotation_channels(rotation));

    let n_ch = channels.len();
    let mut w = vec![0.0f64; N * n_ch];
    for (ch, target) in channels.iter().enumerate() {
        // orthogonal embeddings of norm 2 ⇒ the weight is the target, halved
        for (s, t) in target.iter().enumerate() {
            w[s * n_ch + ch] = t / 2.0;
        }
    }
    block.in_proj.weight = Param::from_tensor(t1(&w, [N, n_ch], device));
    block.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([n_ch]),
        device,
    )));

    // ── Δ bias, D, QK-norm scales, B/C biases ────────────────────────────────
    // Δ and A are entirely data-dependent here, so the bias is zero.
    block.dt_bias_h = Param::from_tensor(Tensor::zeros(Shape::new([N]), device));
    block.d_h = Param::from_tensor(Tensor::zeros(Shape::new([N]), device));
    block.b_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([N]), device));
    block.c_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([N]), device));
    // QK-Norm rescales B and C but cannot turn them; head h's bias moves C from
    // the shared (1,0,0,0) to twice the h-th basis vector, so the four heads read
    // the four components of one state. It is the only per-head weight here.
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([N, 1, N]), device));
    let c_bias: Vec<f64> = (0..N)
        .flat_map(|h| (0..N).map(move |r| 2.0 * (f64::from(r == h) - f64::from(r == 0))))
        .collect();
    block.c_bias_hmr = Param::from_tensor(t1(&c_bias, [N, 1, N], device));

    // ── block out-projection: the identity ───────────────────────────────────
    let eye: Vec<f64> = (0..N * N).map(|n| f64::from(n / N == n % N)).collect();
    block.out_proj.weight = Param::from_tensor(t1(&eye, [N, N], device));
    block.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([N]), device)));

    // ── the class head: nearest orbit point ──────────────────────────────────
    // logit_g ∝ ⟨state, point(g)⟩ over the six permutations. `ignore_last_residual`
    // means the block's output is all the head sees.
    let mut w_out = vec![0.0f64; N * NUM_CLASSES];
    match head {
        Head::Decoder => {
            for class in 0..NUM_CLASSES {
                let p = point(class as i64);
                for (r, pr) in p.iter().enumerate() {
                    w_out[r * NUM_CLASSES + class] = OUT_GAIN * pr;
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
    net.out_proj.weight = Param::from_tensor(t1(&w_out, [N, NUM_CLASSES], device));
    net.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([NUM_CLASSES]),
        device,
    )));

    model
}

/// The rotation channels, per [`RotationKind`]. Every entry is the channel's
/// value at `(SWAP_S, SWAP_T, RESET)`.
fn rotation_channels(rotation: RotationKind) -> Vec<[f64; NUM_SYMBOLS]> {
    // The scaled axis of each swap: direction = the axis, magnitude = a half-turn.
    let axis = |k: usize| {
        [
            TURN_RAW * swap_axis(SWAP_S)[k],
            TURN_RAW * swap_axis(SWAP_T)[k],
            0.0,
        ]
    };
    match rotation {
        // Left and right generators **equal** ⇒ v ↦ q v q̄, conjugation, SO(3).
        // Channels are laid out [head][left | right][x, y, z].
        RotationKind::Rotor4D => (0..N)
            .flat_map(|_| [axis(0), axis(1), axis(2), axis(0), axis(1), axis(2)])
            .collect(),
        // The left factor alone ⇒ v ↦ q v, whose square is −1: the double cover.
        RotationKind::Quaternion4D => (0..N).flat_map(|_| [axis(0), axis(1), axis(2)]).collect(),
        // one angle per state pair: `s` turns pair 0 by π, `t` turns pair 1 by π
        RotationKind::Complex2D => vec![[TURN_RAW, 0.0, 0.0], [0.0, TURN_RAW, 0.0]],
    }
}

/// The four `C` channels, carrying `(1, 0, 0, 0)` for every symbol; the per-head
/// bias then moves head `h` to twice the `h`-th basis vector.
fn basis_channels() -> Vec<[f64; NUM_SYMBOLS]> {
    (0..N).map(|r| [f64::from(r == 0); NUM_SYMBOLS]).collect()
}

/// The four `B` channels — the vector `R` writes into the state.
///
/// For the two quaternion rotations it is [`REF_POINT`], a point of the
/// rotation's imaginary 3-space lying on none of the group's axes, so its orbit
/// is six distinct points.
///
/// The abelian twin gets `(1, 0, 1, 0)` instead — one unit in **each** rotated
/// pair, so its state carries both parities rather than one. As in
/// `reset-spinor`, that is the fairest analogue rather than a detail.
fn b_channels(rotation: RotationKind) -> Vec<[f64; NUM_SYMBOLS]> {
    (0..N)
        .map(|r| match rotation {
            RotationKind::Complex2D => [f64::from(r % 2 == 0); NUM_SYMBOLS],
            _ => [REF_POINT[r]; NUM_SYMBOLS],
        })
        .collect()
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

const FAMILIES: [(&str, Family); 3] = [
    ("random", Family::Random),
    ("shuffle", Family::Shuffle),
    ("runs", Family::Runs),
];

/// Run `model` over `count` sequences of one family; return the per-position
/// output channels and the targets.
fn run(
    model: &MambaLatentNet,
    family: Family,
    count: usize,
    seed: u64,
    device: &Device,
) -> (Vec<[f64; NUM_CLASSES]>, Vec<i64>) {
    let items: Vec<_> = ResetSwapDataset::new(count, SEQ_LENGTH, family, seed)
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
    let (channels, targets) = run(model, family, count, EVAL, device);
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

/// Seed of the split a lookup table is **fitted** on.
const FIT: u64 = 0x51D3;
/// Seed of the split everything is **scored** on.
const EVAL: u64 = 0xE7A1;

/// Accuracy of the best lookup table from a discrete code to a class — fitted
/// on one split, scored on another, so it is a ceiling a model could actually
/// reach rather than a memorised answer key.
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

/// Accuracy of the **best linear readout** of the block's output — a softmax
/// regression on the four output channels plus a bias, fitted on one split and
/// scored on another.
///
/// This is the honest version of "what a head like this example's could do":
/// the example's own head is one linear map, and this searches over all of them.
/// It is the column where the left-isoclinic twin's antipodal pairs bite, and
/// the one the [`best_lookup`] over [`output_codes`] deliberately dominates
/// (a table is not linear, and that gap is the finding).
fn best_linear_readout(
    fit: (&[[f64; NUM_CLASSES]], &[i64]),
    eval: (&[[f64; NUM_CLASSES]], &[i64]),
) -> f64 {
    // Standardise by the global RMS so one learning rate fits every block.
    let rms = {
        let n = (fit.0.len() * N) as f64;
        let sq: f64 = fit
            .0
            .iter()
            .flat_map(|o| o[..N].iter().map(|v| v * v))
            .sum();
        (sq / n).sqrt().max(1e-12)
    };
    let feats = |o: &[f64; NUM_CLASSES]| {
        let mut f = [0.0f64; N + 1];
        for r in 0..N {
            f[r] = o[r] / rms;
        }
        f[N] = 1.0; // bias
        f
    };
    let mut w = [[0.0f64; N + 1]; NUM_CLASSES];
    let lr = 1.0;
    for _ in 0..1200 {
        let mut grad = [[0.0f64; N + 1]; NUM_CLASSES];
        for (o, &t) in fit.0.iter().zip(fit.1) {
            let f = feats(o);
            let logits: [f64; NUM_CLASSES] =
                std::array::from_fn(|c| (0..=N).map(|k| w[c][k] * f[k]).sum());
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: [f64; NUM_CLASSES] = std::array::from_fn(|c| (logits[c] - max).exp());
            let sum: f64 = exp.iter().sum();
            for c in 0..NUM_CLASSES {
                let d = exp[c] / sum - f64::from(c as i64 == t);
                for k in 0..=N {
                    grad[c][k] += d * f[k];
                }
            }
        }
        let scale = lr / fit.0.len() as f64;
        for c in 0..NUM_CLASSES {
            for k in 0..=N {
                w[c][k] -= scale * grad[c][k];
            }
        }
    }
    let hits = eval
        .0
        .iter()
        .zip(eval.1)
        .filter(|&(o, &t)| {
            let f = feats(o);
            let pred = (0..NUM_CLASSES)
                .max_by(|&a, &b| {
                    let la: f64 = (0..=N).map(|k| w[a][k] * f[k]).sum();
                    let lb: f64 = (0..=N).map(|k| w[b][k] * f[k]).sum();
                    la.partial_cmp(&lb).unwrap()
                })
                .unwrap();
            pred as i64 == t
        })
        .count();
    hits as f64 / eval.1.len() as f64
}

/// Quantise each output channel into `LEVELS` equal-width bins over the range
/// observed on the fit split, and pack the four into one code — a finite
/// partition of the block's output space, so the best table over it dominates
/// every readout the model could have carried, linear or not.
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

/// Collect, for one family, the per-position codes an order-blind model could
/// key on — the symbol, the `(#s, #t)` counts since the reset, and the parity
/// `(#s + #t) mod 2` — with the targets.
fn codes(
    family: Family,
    count: usize,
    seed: u64,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<i64>) {
    let (mut sym, mut cnt, mut par, mut targets) = (vec![], vec![], vec![], vec![]);
    for item in ResetSwapDataset::new(count, SEQ_LENGTH, family, seed).iter() {
        let item = item.expect("dataset item");
        for ((&s, (a, b)), &t) in item
            .symbols
            .iter()
            .zip(counts_since_reset(&item.symbols))
            .zip(&labels(&item.symbols))
        {
            sym.push(s);
            cnt.push(a as usize * SEQ_LENGTH + b as usize);
            par.push(((a + b) % 2) as usize);
            targets.push(t);
        }
    }
    (sym, cnt, par, targets)
}

/// Number of distinct `(#s, #t)` codes [`codes`] can emit.
const NUM_COUNT_CODES: usize = SEQ_LENGTH * SEQ_LENGTH;

// ---------------------------------------------------------------------------
// 1. the hand-built solution
// ---------------------------------------------------------------------------

/// Every weight written down in closed form; no training anywhere.
#[test]
fn handmade_rotor_solves_every_family() {
    let device = Device::default();
    let model = handmade(&device, RotationKind::Rotor4D, Head::Decoder);
    println!("hand-built SO(4) block ({} params):", model.num_params());
    let mut worst = 1.0f64;
    for (name, family) in FAMILIES {
        let acc = accuracy(&model, family, 256, &device);
        println!("  {name:<9} {:6.2}%", 100.0 * acc);
        worst = worst.min(acc);
    }
    assert!(worst > 0.995, "hand-built solution is not exact: {worst}");
}

// ---------------------------------------------------------------------------
// 2. the same construction one rung down
// ---------------------------------------------------------------------------

/// `Quaternion4D` — the left factor alone — reported three ways.
///
/// The failure here is **not** the abelian one. The left-isoclinic state carries
/// the double cover `2D₃` (order 12), which is strictly *more* than the label:
/// the best table over its output space recovers the permutation outright. What
/// it cannot do is present it to a **linear** head, because the two lifts `±W`
/// of one permutation are antipodal vectors sharing a target — so every linear
/// functional of the state takes opposite values on the same class, and averaged
/// over a class they cancel. The last column measures exactly that cancellation:
/// `‖mean output‖ / rms output` per class, which the conjugating block leaves at
/// 1 and the left-isoclinic block drives to ~0.
#[test]
fn left_isoclinic_carries_a_double_cover() {
    let device = Device::default();
    let decoder = handmade(&device, RotationKind::Quaternion4D, Head::Decoder);
    let probe = handmade(&device, RotationKind::Quaternion4D, Head::Probe);
    let rotor_probe = handmade(&device, RotationKind::Rotor4D, Head::Probe);
    println!(
        "the same construction, left-isoclinic ({} params):",
        decoder.num_params()
    );
    println!(
        "      family    same head   best linear   best table   ‖mean‖/rms   (SO(4) linear / ‖mean‖/rms)"
    );
    let mut best_head = 0.0f64;
    let mut worst_cancel = 1.0f64;
    for (name, family) in FAMILIES {
        let (fit_ch, fit_t) = run(&probe, family, 256, FIT, &device);
        let (eval_ch, eval_t) = run(&probe, family, 256, EVAL, &device);
        let range = channel_range(&fit_ch);
        let table = best_lookup(
            (&output_codes(&fit_ch, &range), &fit_t),
            (&output_codes(&eval_ch, &range), &eval_t),
            NUM_OUTPUT_CODES,
        );
        let linear = best_linear_readout((&fit_ch, &fit_t), (&eval_ch, &eval_t));
        let cancel = mean_over_rms(&eval_ch, &eval_t);
        let (rotor_fit, rotor_fit_t) = run(&rotor_probe, family, 256, FIT, &device);
        let (rotor_ch, rotor_t) = run(&rotor_probe, family, 256, EVAL, &device);
        println!(
            "  {name:<9}      {:6.2}%       {:6.2}%       {:6.2}%      {cancel:5.3}         {:6.2}% / {:5.3}",
            100.0 * accuracy(&decoder, family, 256, &device),
            100.0 * linear,
            100.0 * table,
            100.0 * best_linear_readout((&rotor_fit, &rotor_fit_t), (&rotor_ch, &rotor_t)),
            mean_over_rms(&rotor_ch, &rotor_t),
        );
        best_head = best_head.max(linear);
        worst_cancel = worst_cancel.min(cancel);
        assert!(
            table > 0.95,
            "{name}: the double cover should *contain* the answer ({table:.4})"
        );
    }
    assert!(
        best_head < 0.75,
        "the left-isoclinic twin reached {best_head:.4} through the best linear head"
    );
    assert!(
        worst_cancel < 0.2,
        "the two lifts did not cancel ({worst_cancel:.3}) — check the double cover"
    );
}

/// Per class, `‖mean of the output vectors‖ / rms of the output vectors`,
/// averaged over classes: 1 when every position of a class lands on the same
/// state, ~0 when they land on `±` the same state equally often.
fn mean_over_rms(channels: &[[f64; NUM_CLASSES]], targets: &[i64]) -> f64 {
    let mut sum = vec![[0.0f64; N]; NUM_CLASSES];
    let mut sq = vec![0.0f64; NUM_CLASSES];
    let mut n = vec![0.0f64; NUM_CLASSES];
    for (o, &t) in channels.iter().zip(targets) {
        let c = t as usize;
        n[c] += 1.0;
        for r in 0..N {
            sum[c][r] += o[r];
            sq[c] += o[r] * o[r];
        }
    }
    let mut total = 0.0;
    let mut seen = 0.0;
    for c in 0..NUM_CLASSES {
        if n[c] == 0.0 {
            continue;
        }
        let mean_norm = sum[c]
            .iter()
            .map(|v| v * v / (n[c] * n[c]))
            .sum::<f64>()
            .sqrt();
        let rms = (sq[c] / n[c]).sqrt().max(1e-12);
        total += mean_norm / rms;
        seen += 1.0;
    }
    total / seen
}

/// And the rung below that: the abelian rotation, for completeness. A `cumsum`
/// of angles is a function of the symbol counts, and `st ≠ ts`.
#[test]
fn abelian_rotation_loses_the_order() {
    let device = Device::default();
    let probe = handmade(&device, RotationKind::Complex2D, Head::Probe);
    println!("the same construction, abelian rotation:");
    let mut best = 0.0f64;
    for (name, family) in FAMILIES {
        let (fit_ch, fit_t) = run(&probe, family, 512, FIT, &device);
        let (eval_ch, eval_t) = run(&probe, family, 256, EVAL, &device);
        let range = channel_range(&fit_ch);
        let ceiling = best_lookup(
            (&output_codes(&fit_ch, &range), &fit_t),
            (&output_codes(&eval_ch, &range), &eval_t),
            NUM_OUTPUT_CODES,
        );
        println!("  {name:<9} best readout {:6.2}%", 100.0 * ceiling);
        best = best.max(ceiling);
    }
    assert!(
        best < 0.85,
        "the abelian twin reached {best:.4} — the task does not need order"
    );
}

// ---------------------------------------------------------------------------
// 3. the ceilings
// ---------------------------------------------------------------------------

/// What an order-blind model can do, and what the **sign character** alone can.
///
/// The second is the sharp one for this example: every finite subgroup of
/// `SU(2)` has exactly one element of order two, so the only homomorphism from
/// `S₃` into `SU(2)` sends the three transpositions to `−1` — the parity, and
/// nothing else. A left-isoclinic block whose rotation is a homomorphic image of
/// the word is therefore bounded by the `parity` column; escaping it (as the
/// hand-built twin does, by tracking the double cover instead) buys information
/// that is no longer linearly readable.
#[test]
fn counts_and_parity_ceilings() {
    println!("order-blind ceilings, accuracy per family:");
    println!("      family    memoryless   by parity   by (#s,#t)");
    let (mut best_counts, mut best_parity) = (0.0f64, 0.0f64);
    for (name, family) in FAMILIES {
        let (fit_sym, fit_cnt, fit_par, fit_t) = codes(family, 2048, FIT);
        let (eval_sym, eval_cnt, eval_par, eval_t) = codes(family, 512, EVAL);
        let by_counts = best_lookup((&fit_cnt, &fit_t), (&eval_cnt, &eval_t), NUM_COUNT_CODES);
        let by_parity = best_lookup((&fit_par, &fit_t), (&eval_par, &eval_t), 2);
        println!(
            "  {name:<9}    {:6.2}%     {:6.2}%      {:6.2}%",
            100.0 * best_lookup((&fit_sym, &fit_t), (&eval_sym, &eval_t), NUM_SYMBOLS),
            100.0 * by_parity,
            100.0 * by_counts,
        );
        best_counts = best_counts.max(by_counts);
        best_parity = best_parity.max(by_parity);
    }
    println!(
        "best over the families: counts {:.2}%, parity {:.2}%  (chance {:.2}%, hand-built SO(4) 100%)",
        100.0 * best_counts,
        100.0 * best_parity,
        100.0 / NUM_CLASSES as f64
    );
    assert!(
        best_counts < 0.85,
        "the symbol counts nearly give the answer: {best_counts}"
    );
    assert!(
        best_parity < 0.6,
        "the sign character nearly gives the answer: {best_parity}"
    );
}

// ---------------------------------------------------------------------------
// 4. the group itself, and the double cover
// ---------------------------------------------------------------------------

/// The dataset's labels really are the `S₃` word problem, and `S₃` really is the
/// group `SU(2)` cannot hold: **three** elements of order two.
#[test]
fn labels_are_the_symmetric_group() {
    // st ≠ ts — the reset-spinor requirement, inherited
    let st = labels(&[RESET, SWAP_T, SWAP_S]); // newest on the left: s∘t
    let ts = labels(&[RESET, SWAP_S, SWAP_T]); // t∘s
    assert_ne!(st[2], ts[2], "st and ts must differ");
    assert_eq!(st[2], 3, "s∘t = bca");
    assert_eq!(ts[2], 4, "t∘s = cab");

    // every swap is an involution, and there are three of them
    let squared = labels(&[RESET, SWAP_S, SWAP_S]);
    assert_eq!(squared[2], 0, "s² = 1");
    let involutions: Vec<i64> = (0..NUM_CLASSES as i64)
        .filter(|&c| c != 0 && class_of(compose(PERMS[c as usize], PERMS[c as usize])) == 0)
        .collect();
    assert_eq!(involutions.len(), 3, "S₃ has three involutions");

    // (s∘t)³ = 1 — the two axes are 60° apart
    let cubed = labels(&[RESET, SWAP_T, SWAP_S, SWAP_T, SWAP_S, SWAP_T, SWAP_S]);
    assert_eq!(cubed[6], 0, "(st)³ = 1");

    // every element is reachable, and the reset restarts the word
    let mut seen = [false; NUM_CLASSES];
    for item in ResetSwapDataset::new(64, SEQ_LENGTH, Family::Mixed, 7).iter() {
        let item = item.expect("dataset item");
        for (t, (&s, &c)) in item.symbols.iter().zip(&item.targets).enumerate() {
            seen[c as usize] = true;
            if s == RESET {
                assert_eq!(c, 0, "a reset lands on the identity (position {t})");
            }
        }
    }
    assert!(seen.iter().all(|s| *s), "some permutation never occurs");
}

/// The obstruction, in three lines of quaternion algebra: the lift of a swap
/// squares to `−1`, so left multiplication cannot represent an involution —
/// while conjugation by the same lift can, because `±q` conjugate identically.
#[test]
fn the_lift_of_a_swap_squares_to_minus_one() {
    for symbol in [SWAP_S, SWAP_T] {
        let q = symbol_quat(symbol);
        let q2 = quat_mul(q, q);
        assert!(
            (q2[0] + 1.0).abs() < 1e-12 && q2[1..].iter().all(|v| v.abs() < 1e-12),
            "q² should be −1 for a half-turn lift, got {q2:?}"
        );
        // …yet the *rotation* it induces is an involution: conjugating twice is
        // the identity on the whole space.
        let v = REF_POINT;
        let once = conjugate(q, v);
        let twice = conjugate(q, once);
        for (a, b) in twice.iter().zip(&v) {
            assert!((a - b).abs() < 1e-12, "conjugation should square to 1");
        }
    }

    // and the six orbit points are distinct — what makes the readout a decoder
    for a in 0..NUM_CLASSES as i64 {
        for b in 0..a {
            let (pa, pb) = (point(a), point(b));
            let d: f64 = pa.iter().zip(&pb).map(|(x, y)| (x - y) * (x - y)).sum();
            assert!(d > 0.1, "orbit points {a} and {b} coincide");
        }
    }

    // the orbit really is the group acting: point(g∘h) = g · point(h)
    for g in [SWAP_S, SWAP_T] {
        for h in 0..NUM_CLASSES as i64 {
            let composed = class_of(compose(symbol_perm(g), PERMS[h as usize]));
            let rotated = conjugate(symbol_quat(g), point(h));
            for (x, y) in rotated.iter().zip(&point(composed)) {
                assert!((x - y).abs() < 1e-9, "the action is not the composition");
            }
        }
    }
}

/// `q v q̄`, the two-sided step this example's block is built on.
fn conjugate(q: [f64; 4], v: [f64; 4]) -> [f64; 4] {
    quat_mul(quat_mul(q, v), [q[0], -q[1], -q[2], -q[3]])
}
