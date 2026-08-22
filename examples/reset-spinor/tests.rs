//! The three claims this example rests on, measured.
//!
//! 1. A **hand-built** quaternion Mamba-3 block solves the task exactly — no
//!    fitting, every weight written down in closed form from the unrolled
//!    recurrence.
//! 2. The **same construction with the abelian rotation** (`Complex2D`, one enum
//!    knob) does not. A `cumsum` of angles is a function of the symbol counts,
//!    and what its state ends up carrying is exactly the abelianisation
//!    `Q₈/{±1} ≅ Z₂×Z₂` — the commutator, which is the whole content of the
//!    task, is gone. Measured twice: through the identical head, and through the
//!    **best readout its output admits**.
//! 3. **No order-blind model can**, whatever it does with those counts: the best
//!    lookup table from `(#i, #j)` since the reset is far below the block. It
//!    bounds every block that, like this one, writes its state at the reset and
//!    reads the rotation accumulated since — under an abelian rotation that
//!    accumulation *is* a function of the two counts, whatever the angles and
//!    whatever the decoder.
//!
//! The readouts in (2) and (3) are lookup tables **fitted on one split and
//! scored on another**, so they are ceilings a model could actually reach, not
//! memorised labels.
//!
//! The construction is the one derived in [`crate::model`]: `R` writes the
//! identity quaternion into the state at the current cumulative rotation, `i`
//! and `j` turn that rotation by non-commuting half-turns, and the four heads
//! read the four components of the relative quaternion — the group element
//! itself.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    Family, NUM_CLASSES, NUM_SYMBOLS, RESET, ResetSpinorDataset, SEQ_LENGTH, TURN_I, TURN_J,
    counts_since_reset, labels, one_hot, quaternion,
};
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `Δ` for every head and every symbol. Fixed at 1 so the rotation generator is
/// `π·tanh(ϑ)` outright and `γ = λ·Δ = 1` writes `B` unscaled.
const DELTA: f64 = 1.0;
/// `ϑ` for an axis a symbol turns about — a half-turn, i.e. the unit quaternion
/// `i` (or `j`) up to `cos(π/2) ≈ 4e-8`.
///
/// The block bounds one step to `rotation_range · π · Δ` and defaults to
/// `range = 2` for the quaternion rotation, so a half-turn is
/// `tanh(‖ϑ‖) = 1/2`: interior, where `tanh` is steep and the gradient is
/// alive. (At `range = 1` the same half-turn would sit exactly on `tanh`'s
/// asymptote, reachable only by saturating a channel — a place no optimiser can
/// arrive at, since f32's `tanh` derivative there is exactly zero.)
const TURN_RAW: f64 = 0.5493061443340549; // atanh(1/2)
/// `Â` on a turn. `A = −softplus(Â)`, floored at the block's `a_floor` (`1e-4`),
/// so this is the flattest hold the block allows.
const A_HOLD_RAW: f64 = -20.0;
/// `−A` on a `RESET`: `ᾱ = e⁻²⁰` erases what the state held.
const A_WIPE: f64 = 20.0;
/// `λ̂`, large enough that `λ = σ(λ̂) ≈ 1`: the trapezoid's left-endpoint weight
/// `β = (1−λ)Δᾱ` vanishes and only the current token is written.
const LAMBDA_RAW: f64 = 20.0;
/// `x(R) = 1` — the write. `x(i) = x(j) = 0` exactly (`silu(0) = 0`), so a turn
/// writes nothing and only advances the rotation.
const X_WRITE: f64 = 1.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// Class-logit gain on the four readout axes.
const OUT_GAIN: f64 = 3.0;

/// `d_model`, `state_rank` and `nheads` all equal 4 here — one head per
/// quaternion component.
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
/// [`TURN_I`] / [`TURN_J`] / [`RESET`].
///
/// Norm 2 in `d_model = 4` is what the layer's pre-`RmsNorm` (`γ = 1`) passes
/// through unchanged, and orthogonality is what turns the block's in-projection
/// into a lookup table: a channel that must take the values `(t_i, t_j, t_R)`
/// gets the weight `(t_i/2, t_j/2, t_R/2, 0)` and no bias, with no linear system
/// to solve (`reset-majority` and `reset-rotor` need one only because three
/// symbols do not fit orthogonally into their `d_model = 2`).
const EMBED: [[f64; N]; NUM_SYMBOLS] = [
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
];

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

/// Build the block by hand for the given rotation.
///
/// The two rotations differ in what the rotation channels mean:
/// [`RotationKind::Quaternion4D`] takes three of them — a scaled rotation axis,
/// so `i` and `j` become half-turns about two orthogonal axes, which do not
/// commute — while [`RotationKind::Complex2D`] takes two, one angle per state
/// pair, so `i` and `j` become half-turns of the two pairs, which do. The only
/// other difference is the vector the write stores; see [`b_channels`] for why
/// the abelian twin gets a different (and better) one.
fn handmade(device: &Device, rotation: RotationKind, head: Head) -> MambaLatentNet {
    let cfg = crate::model::model_config(rotation);
    let mut model = ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("reset-spinor configures the Mamba-3 variant")
    };

    // ── network in_proj: one-hot → the symbol embedding ──────────────────────
    net.in_proj.weight = Param::from_tensor(t1(&EMBED.concat(), [NUM_SYMBOLS, N], device));
    net.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([N]), device)));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([N]), device));
    let block = &mut layer.mamba_block;

    // ── block in_proj: one affine functional per channel ─────────────────────
    // Channel order: [z(4) | x(4) | B_raw(4) | C_raw(4) | Δ(4) | A(4) | λ(4) | ϑ(2 or 3)].
    // Each entry is the channel's value at (TURN_I, TURN_J, RESET), *before* its
    // own activation.
    let rotation_channels: Vec<[f64; NUM_SYMBOLS]> = match rotation {
        // a scaled rotation axis: `i` turns π about x, `j` turns π about y
        RotationKind::Quaternion4D => vec![
            [TURN_RAW, 0.0, 0.0], // axis x
            [0.0, TURN_RAW, 0.0], // axis y
            [0.0, 0.0, 0.0],      // axis z — unused
        ],
        // one angle per state pair: `i` turns pair 0 by π, `j` turns pair 1 by π
        RotationKind::Complex2D => vec![[TURN_RAW, 0.0, 0.0], [0.0, TURN_RAW, 0.0]],
    };
    let mut channels: Vec<[f64; NUM_SYMBOLS]> = Vec::new();
    channels.extend([[Z_PRE; NUM_SYMBOLS]; N]); // z
    channels.extend([[0.0, 0.0, silu_inv(X_WRITE)]; N]); // x — only R writes
    channels.extend(b_channels(rotation)); // B_raw — the vector the write stores
    channels.extend(basis_channels()); // C_raw (the per-head bias splits it)
    channels.extend([[softplus_inv(DELTA); NUM_SYMBOLS]; N]); // Δ
    channels.extend([[A_HOLD_RAW, A_HOLD_RAW, softplus_inv(A_WIPE)]; N]); // A
    channels.extend([[LAMBDA_RAW; NUM_SYMBOLS]; N]); // λ
    channels.extend(rotation_channels);

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
    // QK-Norm over `state_rank = 4` maps (1,0,0,0) to (2,0,0,0), so B is twice
    // the identity quaternion for every head. C starts there too; head h's bias
    // moves it to twice the h-th basis quaternion — the only per-head weight in
    // the whole construction, and what makes the four heads read four
    // components of the same state.
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([N, 1, N]), device));
    let c_bias: Vec<f64> = (0..N)
        .flat_map(|h| (0..N).map(move |r| 2.0 * (f64::from(r == h) - f64::from(r == 0))))
        .collect();
    block.c_bias_hmr = Param::from_tensor(t1(&c_bias, [N, 1, N], device));

    // ── block out-projection: the identity ───────────────────────────────────
    let eye: Vec<f64> = (0..N * N).map(|n| f64::from(n / N == n % N)).collect();
    block.out_proj.weight = Param::from_tensor(t1(&eye, [N, N], device));
    block.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([N]), device)));

    // ── the class head: nearest group element ────────────────────────────────
    // logit_g ∝ ⟨q, g⟩ over the eight elements of Q₈. `ignore_last_residual`
    // means the block's output is all the head sees.
    let mut w_out = vec![0.0f64; N * NUM_CLASSES];
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
    net.out_proj.weight = Param::from_tensor(t1(&w_out, [N, NUM_CLASSES], device));
    net.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([NUM_CLASSES]),
        device,
    )));

    model
}

/// The four `C` channels, carrying the identity quaternion `(1, 0, 0, 0)` for
/// every symbol. QK-Norm scales it to `(2, 0, 0, 0)`; the per-head bias then
/// moves head `h` to twice the `h`-th basis quaternion.
fn basis_channels() -> Vec<[f64; NUM_SYMBOLS]> {
    (0..N)
        .map(|r| [f64::from(r == 0); NUM_SYMBOLS])
        .collect()
}

/// The four `B` channels — the vector `R` writes into the state.
///
/// For the quaternion rotation it is the group's unit, `(1, 0, 0, 0)`: the
/// state then holds the group element itself.
///
/// The abelian twin gets `(1, 0, 1, 0)` instead — one unit in **each** rotated
/// pair. This is the fairest analogue rather than a detail: `Complex2D` turns
/// the two pairs by two independent cumulative angles, and with `B = (1,0,0,0)`
/// the second pair would multiply zero, so the twin would carry one parity
/// instead of two. With a unit in both, its state carries `(#i mod 2, #j mod 2)`
/// — the entire abelianisation `Q₈/{±1}`, which is the most any sum of angles
/// can hold.
fn b_channels(rotation: RotationKind) -> Vec<[f64; NUM_SYMBOLS]> {
    (0..N)
        .map(|r| match rotation {
            RotationKind::Quaternion4D => [f64::from(r == 0); NUM_SYMBOLS],
            RotationKind::Complex2D => [f64::from(r % 2 == 0); NUM_SYMBOLS],
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
    let items: Vec<_> = ResetSpinorDataset::new(count, SEQ_LENGTH, family, seed)
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
fn best_lookup(
    fit: (&[usize], &[i64]),
    eval: (&[usize], &[i64]),
    num_codes: usize,
) -> f64 {
    let mut tally = vec![[0u64; NUM_CLASSES]; num_codes];
    let mut overall = [0u64; NUM_CLASSES];
    for (&c, &t) in fit.0.iter().zip(fit.1) {
        tally[c][t as usize] += 1;
        overall[t as usize] += 1;
    }
    let argmax = |row: &[u64; NUM_CLASSES]| {
        (0..NUM_CLASSES)
            .max_by_key(|&c| row[c])
            .expect("non-empty") as i64
    };
    let fallback = argmax(&overall);
    let table: Vec<i64> = tally
        .iter()
        .map(|row| if row.iter().sum::<u64>() == 0 { fallback } else { argmax(row) })
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
        channels.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), o| {
            (lo.min(o[r]), hi.max(o[r]))
        })
    })
}

/// Collect, for one family, the per-position codes an order-blind model could
/// key on — the symbol, and the `(#i, #j)` counts since the reset — with the
/// targets.
fn codes(family: Family, count: usize, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<i64>) {
    let (mut sym, mut cnt, mut targets) = (vec![], vec![], vec![]);
    for item in ResetSpinorDataset::new(count, SEQ_LENGTH, family, seed).iter() {
        let item = item.expect("dataset item");
        for ((&s, (a, b)), &t) in item
            .symbols
            .iter()
            .zip(counts_since_reset(&item.symbols))
            .zip(&labels(&item.symbols))
        {
            sym.push(s);
            cnt.push(a as usize * SEQ_LENGTH + b as usize);
            targets.push(t);
        }
    }
    (sym, cnt, targets)
}

/// Number of distinct `(#i, #j)` codes [`codes`] can emit.
const NUM_COUNT_CODES: usize = SEQ_LENGTH * SEQ_LENGTH;

// ---------------------------------------------------------------------------
// 1. the hand-built solution
// ---------------------------------------------------------------------------

/// Every weight written down in closed form; no training anywhere.
#[test]
fn handmade_block_solves_every_family() {
    let device = Device::default();
    let model = handmade(&device, RotationKind::Quaternion4D, Head::Decoder);
    println!(
        "hand-built quaternion block ({} params):",
        model.num_params()
    );
    let mut worst = 1.0f64;
    for (name, family) in FAMILIES {
        let acc = accuracy(&model, family, 256, &device);
        println!("  {name:<9} {:6.2}%", 100.0 * acc);
        worst = worst.min(acc);
    }
    assert!(worst > 0.995, "hand-built solution is not exact: {worst}");
}

// ---------------------------------------------------------------------------
// 2. the abelian twin of the same construction
// ---------------------------------------------------------------------------

/// The identical construction with `RotationKind::Complex2D` — the one enum
/// knob — reported two ways.
///
/// Its cumulative rotation is a `cumsum`, so the two state pairs are turned by
/// an angle proportional to `#i` and to `#j`: what reaches the head is a
/// function of the **counts**, and nothing else. That is not nothing — at the
/// abelian default (`rotation_range = 1`, so a quarter turn per symbol here) it
/// resolves each count mod 4, which on this data is nearly everything the counts
/// have to give. It is still not the answer: the second column hands the block
/// the best table over a fine partition of its whole output space (fitted on a
/// separate split), and it lands on the counts ceiling of
/// [`counts_ceiling_is_the_abelian_limit`] rather than above it. The order —
/// `ij` against `ji` — is simply not in there.
#[test]
fn abelian_rotation_loses_the_order() {
    let device = Device::default();
    let decoder = handmade(&device, RotationKind::Complex2D, Head::Decoder);
    let probe = handmade(&device, RotationKind::Complex2D, Head::Probe);
    println!(
        "the same construction, abelian rotation ({} params):",
        decoder.num_params()
    );
    println!("      family    same head   best readout");
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
        println!(
            "  {name:<9}      {:6.2}%         {:6.2}%",
            100.0 * accuracy(&decoder, family, 256, &device),
            100.0 * ceiling,
        );
        best = best.max(ceiling);
    }
    println!(
        "best readout of the abelian state, over the families: {:.2}%",
        100.0 * best
    );
    // the same bar the counts ceiling is held to: an angle-sum cannot beat it
    assert!(
        best < 0.85,
        "the abelian twin reached {best:.4} — the task does not need a non-abelian rotation"
    );
}

// ---------------------------------------------------------------------------
// 3. the ceiling for everything order-blind
// ---------------------------------------------------------------------------

/// The best predictor that sees only the symbol **counts** since the reset.
///
/// A block that writes its state at the reset and reads the rotation
/// accumulated since — this construction, under either rotation — can see no
/// more than this: an abelian accumulation is a `cumsum`, hence a linear
/// function of exactly these two numbers, whatever the angles and whatever the
/// decoder. (A block that instead wrote on every symbol could carry an
/// order-dependent sum; the trained `--rotation complex` run in the README is
/// what covers that, and it does not clear this ceiling either.) The gap the
/// counts leave is
/// the commutator: the counts fix the element only up to `{±1}`, so wherever
/// both signs occur the table has to guess. `runs` is the family where they come
/// closest to sufficing (the word is a few long blocks, so its reduction is
/// nearly pinned by how many of each symbol went by); `shuffle` is where they
/// say the least.
#[test]
fn counts_ceiling_is_the_abelian_limit() {
    println!("order-blind ceilings, accuracy per family:");
    println!("      family    memoryless   by (#i,#j)   ambiguous");
    let mut best = 0.0f64;
    for (name, family) in FAMILIES {
        let (fit_sym, fit_cnt, fit_t) = codes(family, 2048, FIT);
        let (eval_sym, eval_cnt, eval_t) = codes(family, 512, EVAL);
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
            100.0 * best_lookup((&fit_sym, &fit_t), (&eval_sym, &eval_t), NUM_SYMBOLS),
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
    assert!(best < 0.85, "the symbol counts nearly give the answer: {best}");
}

// ---------------------------------------------------------------------------
// 4. the group itself
// ---------------------------------------------------------------------------

/// The dataset's labels really are the `Q₈` word problem: the product is
/// non-commutative (`ij ≠ ji`), the generators have order 4, and `R` restarts
/// the word.
#[test]
fn labels_are_the_quaternion_group() {
    let ij = labels(&[RESET, TURN_J, TURN_I]); // newest factor on the left: i·j
    let ji = labels(&[RESET, TURN_I, TURN_J]); // j·i
    assert_ne!(ij[2], ji[2], "ij and ji must differ");
    assert_eq!(quaternion(ij[2]), [0.0, 0.0, 0.0, 1.0], "ij = k");
    assert_eq!(quaternion(ji[2]), [0.0, 0.0, 0.0, -1.0], "ji = -k");

    let powers = labels(&[RESET, TURN_I, TURN_I, TURN_I, TURN_I]);
    assert_eq!(quaternion(powers[2]), [-1.0, 0.0, 0.0, 0.0], "i² = -1");
    assert_eq!(powers[4], powers[0], "i⁴ = 1");

    // every element is reachable, and the reset restarts the word
    let mut seen = [false; NUM_CLASSES];
    for item in ResetSpinorDataset::new(64, SEQ_LENGTH, Family::Mixed, 7).iter() {
        let item = item.expect("dataset item");
        for (t, (&s, &c)) in item.symbols.iter().zip(&item.targets).enumerate() {
            seen[c as usize] = true;
            if s == RESET {
                assert_eq!(c, 0, "a reset lands on the identity (position {t})");
            }
        }
    }
    assert!(seen.iter().all(|s| *s), "some group element never occurs");
}
