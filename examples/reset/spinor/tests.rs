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
//! and `j` turn that rotation by non-commuting half-turns, and the state is then
//! the relative quaternion — the group element itself. Two heads read it, along
//! the two coordinates of a [`plane`] on which the eight elements of `Q₈` are
//! eight directions `45°` apart, so the head decodes a sector. The ceilings in
//! (2) are still measured on all four components of the state (see [`probe`]).

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
/// `π·tanh(ϑ)` outright and `γ = Δ = 1` writes `B` unscaled — the block runs at
/// `Trapezoid::None`, which spends the whole step on the current token.
const DELTA: f64 = 1.0;
/// `ϑ` for an axis a symbol turns about — a half-turn, i.e. the unit quaternion
/// `i` (or `j`) up to `cos(π/2) ≈ 4e-8`.
///
/// The block bounds one step to `rotation_range · π · Δ`, and the default range
/// of 2 makes a half-turn `tanh(‖ϑ‖) = 1/2`: interior, where `tanh` is steep
/// and the gradient is alive. (At `range = 1` the same half-turn would sit
/// exactly on `tanh`'s asymptote, reachable only by saturating a channel — a
/// place no optimiser can arrive at, since f32's `tanh` derivative there is
/// exactly zero.) The abelian twin below reads the same constant through the
/// same bound, so it too turns by a half — which is what makes what it computes
/// exactly the abelianisation.
const TURN_RAW: f64 = 0.5493061443340549; // atanh(1/2)
/// `Â` on a turn. `A = −softplus(Â)`, floored at the block's `a_floor` (`1e-4`),
/// so this is the flattest hold the block allows.
const A_HOLD_RAW: f64 = -20.0;
/// `−A` on a `RESET`: `ᾱ = e⁻²⁰` erases what the state held.
const A_WIPE: f64 = 20.0;
/// `x(R) = 1` — the write. `x(i) = x(j) = 0` exactly (`silu(0) = 0`), so a turn
/// writes nothing and only advances the rotation.
const X_WRITE: f64 = 1.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// Class-logit gain on the plane the readout is folded onto.
const OUT_GAIN: f64 = 3.0;

/// `state_rank` — the four components of the quaternion the state holds.
const RANK: usize = 4;

/// `nheads`, which at `per_head_dim = 1` is also `d_inner`.
///
/// Two, because two is what the *readout* needs: every head holds its own copy
/// of the same quaternion (same `B`, same `ᾱ`, same rotation) and differs only
/// in the `C` it reads that copy with, so `nheads` is a count of **projections**,
/// not of state. Eight group elements fit on eight directions of a plane
/// ([`plane`]), and two projections separate them; a third and a fourth would be
/// duplicates the head cannot use.
const NHEADS: usize = 2;

/// `d_model`. Two, the floor for a three-symbol alphabet: the layer's
/// pre-`RmsNorm` sends a token to the unit sphere, so a 1-D token would carry
/// only its sign. It equals `d_inner` here, so the block's `out_proj` is the
/// identity and the two heads *are* the plane.
const D_MODEL: usize = 2;

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

// ---------------------------------------------------------------------------
// the hand-built model
// ---------------------------------------------------------------------------

/// The three symbol embeddings, each of norm `√2` so the layer's pre-`RmsNorm`
/// (`γ = 1`) passes them through unchanged. Indexed by
/// [`TURN_I`] / [`TURN_J`] / [`RESET`].
///
/// Three points of `ℝ²` are affinely independent, so a channel that must take
/// the values `(t_i, t_j, t_R)` is one 3×3 [`solve3`] away — weight plus bias,
/// exactly as `reset-majority` and `reset-rotor` do it. (Orthogonal embeddings
/// would spare the solve, but they need `d_model = 3` at least, and the block is
/// the thing being minimised.)
const EMBED: [[f64; D_MODEL]; NUM_SYMBOLS] = [
    [std::f64::consts::SQRT_2, 0.0],
    [0.0, std::f64::consts::SQRT_2],
    [-1.0, -1.0],
];

/// The plane the two heads read the state on: state component `r` goes to the
/// unit vector at `r · 45°`.
///
/// The eight elements of `Q₈` are `±eᵣ`, so they land on the eight directions
/// `45°` apart: distinct, equidistant, and in convex position, which is exactly
/// what a linear eight-way head needs. The block never forms this plane as a
/// separate step — head `h` reads the state with [`c_axis`], the `h`-th
/// coordinate of this map — so the block's own `out_proj` is the identity.
fn plane(r: usize) -> [f64; D_MODEL] {
    let angle = std::f64::consts::FRAC_PI_4 * r as f64;
    [angle.cos(), angle.sin()]
}

/// [`plane`] applied to a quaternion — where a group element lands.
fn plane_point(q: [f64; RANK]) -> [f64; D_MODEL] {
    let mut p = [0.0; D_MODEL];
    for (r, qr) in q.iter().enumerate() {
        let axis = plane(r);
        for (c, pc) in p.iter_mut().enumerate() {
            *pc += qr * axis[c];
        }
    }
    p
}

/// The `C` vector head `h` reads the state with: the `h`-th coordinate of
/// [`plane`] over the four components, scaled to norm 2 — the length QK-Norm
/// hands the construction, so the two heads share one positive factor and
/// `(y₀, y₁) ∝ plane_point(q_rel)` exactly.
fn c_axis(h: usize) -> [f64; RANK] {
    let raw: [f64; RANK] = std::array::from_fn(|r| plane(r)[h]);
    let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    raw.map(|v| 2.0 * v / norm)
}

/// What the network's head reads out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    /// The nearest-element decoder: logit `g` ∝ `⟨p, plane_point(g)⟩` over the
    /// eight elements of `Q₈`, read on the plane [`plane`] puts them on.
    Decoder,
    /// Point the two heads' `C` at the two named state components instead, and
    /// pass them through as the first two logits, so a test can search over every
    /// readout the head could have expressed. Two at a time is all two heads
    /// carry, so [`probe`] runs the block twice.
    Probe([usize; 2]),
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
    net.in_proj.weight = Param::from_tensor(t1(&EMBED.concat(), [NUM_SYMBOLS, D_MODEL], device));
    net.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([D_MODEL]),
        device,
    )));

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([D_MODEL]), device));
    let block = &mut layer.block;

    // ── block in_proj: one affine functional per channel ─────────────────────
    // Channel order: [z(2) | x(2) | B_raw(4) | C_raw(4) | Δ(2) | A(2) | ϑ(2, 6 or 12)].
    // The 2s are `nheads`, the 4s `state_rank`; `Trapezoid::None` projects no λ.
    // Each entry is the channel's value at (TURN_I, TURN_J, RESET), *before* its
    // own activation.
    let rotation_channels: Vec<[f64; NUM_SYMBOLS]> = match rotation {
        // `Real1D` has no rotation to hand-build; the ladder starts above it.
        RotationKind::Real1D => unreachable!("{rotation:?} has no rotation channels"),
        // A scaled rotation axis per head: `i` turns π about x, `j` about y.
        // The block projects the generators **per head** (`nheads · 3` channels
        // here), so every head gets its own copy — this construction wants both
        // to read the same rotation, but the block no longer forces that.
        RotationKind::Quaternion4D => (0..NHEADS)
            .flat_map(|_| {
                [
                    [TURN_RAW, 0.0, 0.0], // axis x
                    [0.0, TURN_RAW, 0.0], // axis y
                    [0.0, 0.0, 0.0],      // axis z — unused
                ]
            })
            .collect(),
        // one angle per state pair: `i` turns pair 0 by π, `j` turns pair 1 by π
        RotationKind::Complex2D => vec![[TURN_RAW, 0.0, 0.0], [0.0, TURN_RAW, 0.0]],
        // `Rotor4D` is the same rotation with a second, *right* factor per head
        // and block (channels `[head][left|right][3]`). The quaternion solution
        // lives inside it at `p ≡ 1`, so the left generators are the ones above
        // and the right ones are zero.
        RotationKind::Rotor4D => (0..NHEADS)
            .flat_map(|_| {
                [
                    [TURN_RAW, 0.0, 0.0], // left: axis x
                    [0.0, TURN_RAW, 0.0], // left: axis y
                    [0.0, 0.0, 0.0],      // left: axis z — unused
                    [0.0, 0.0, 0.0],      // right: the identity, p ≡ 1
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            })
            .collect(),
    };
    let mut channels: Vec<[f64; NUM_SYMBOLS]> = Vec::new();
    channels.extend([[Z_PRE; NUM_SYMBOLS]; NHEADS]); // z
    channels.extend([[0.0, 0.0, silu_inv(X_WRITE)]; NHEADS]); // x — only R writes
    channels.extend(b_channels(rotation)); // B_raw — the vector the write stores
    channels.extend(basis_channels()); // C_raw (the per-head bias splits it)
    channels.extend([[softplus_inv(DELTA); NUM_SYMBOLS]; NHEADS]); // Δ
    channels.extend([[A_HOLD_RAW, A_HOLD_RAW, softplus_inv(A_WIPE)]; NHEADS]); // A
    channels.extend(rotation_channels);

    let rows = [
        [EMBED[TURN_I][0], EMBED[TURN_I][1], 1.0],
        [EMBED[TURN_J][0], EMBED[TURN_J][1], 1.0],
        [EMBED[RESET][0], EMBED[RESET][1], 1.0],
    ];
    let n_ch = channels.len();
    let mut w = vec![0.0f64; D_MODEL * n_ch];
    let mut b = vec![0.0f64; n_ch];
    for (ch, target) in channels.iter().enumerate() {
        // three symbols through a 2-D token plus a bias: one exact 3×3 solve
        let [w0, w1, bias] = solve3(rows, *target);
        w[ch] = w0; // weight is [d_model, out]: row d, column ch
        w[n_ch + ch] = w1;
        b[ch] = bias;
    }
    block.in_proj.weight = Param::from_tensor(t1(&w, [D_MODEL, n_ch], device));
    block.in_proj.bias = Some(Param::from_tensor(t1(&b, [n_ch], device)));

    // ── Δ bias, D, QK-norm scales, B/C biases ────────────────────────────────
    // Δ and A are entirely data-dependent here, so the bias is zero.
    block.dt_bias_h = Param::from_tensor(Tensor::zeros(Shape::new([NHEADS]), device));
    block.d_h = Param::from_tensor(Tensor::zeros(Shape::new([NHEADS]), device));
    block.b_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([RANK]), device));
    block.c_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([RANK]), device));
    // QK-Norm over `state_rank = 4` maps (1,0,0,0) to (2,0,0,0), so B is twice
    // the identity quaternion for both heads, and C starts there too. Head h's
    // bias is what moves its C — to `c_axis(h)` for the decoder, or to twice a
    // basis quaternion for a probe. It is the only per-head weight in the whole
    // construction, and it is the entire readout: two projections of one state.
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(
        Shape::new([NHEADS, 1, RANK]),
        device,
    ));
    let c_bias: Vec<f64> = (0..NHEADS)
        .flat_map(|h| {
            let target: [f64; RANK] = match head {
                Head::Decoder => c_axis(h),
                Head::Probe(axes) => std::array::from_fn(|r| 2.0 * f64::from(r == axes[h])),
            };
            (0..RANK).map(move |r| target[r] - 2.0 * f64::from(r == 0))
        })
        .collect();
    block.c_bias_hmr = Param::from_tensor(t1(&c_bias, [NHEADS, 1, RANK], device));

    // ── block out-projection: the identity ───────────────────────────────────
    // `d_inner = d_model = 2`, and the two heads already *are* the plane.
    let eye: Vec<f64> = (0..NHEADS * D_MODEL)
        .map(|n| f64::from(n / D_MODEL == n % D_MODEL))
        .collect();
    block.out_proj.weight = Param::from_tensor(t1(&eye, [NHEADS, D_MODEL], device));
    block.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(
        Shape::new([D_MODEL]),
        device,
    )));

    // ── the class head: nearest group element ────────────────────────────────
    // logit_g ∝ ⟨p, plane_point(g)⟩ over the eight elements of Q₈, which sit on
    // eight directions 45° apart. `ignore_last_residual` means the block's
    // output is all the head sees.
    let mut w_out = vec![0.0f64; D_MODEL * NUM_CLASSES];
    match head {
        Head::Decoder => {
            for class in 0..NUM_CLASSES {
                let p = plane_point(quaternion(class as i64));
                for (c, pc) in p.iter().enumerate() {
                    w_out[c * NUM_CLASSES + class] = OUT_GAIN * pc;
                }
            }
        }
        // the two selected axes, verbatim, in the first two logits
        Head::Probe(_) => {
            for c in 0..D_MODEL {
                w_out[c * NUM_CLASSES + c] = 1.0;
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

/// The four `C` channels, carrying the identity quaternion `(1, 0, 0, 0)` for
/// every symbol. QK-Norm scales it to `(2, 0, 0, 0)`; the per-head bias then
/// moves head `h` to wherever that head reads — [`c_axis`], or a basis
/// quaternion for a probe.
fn basis_channels() -> Vec<[f64; NUM_SYMBOLS]> {
    (0..RANK).map(|r| [f64::from(r == 0); NUM_SYMBOLS]).collect()
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
    (0..RANK)
        .map(|r| match rotation {
            RotationKind::Quaternion4D | RotationKind::Rotor4D => [f64::from(r == 0); NUM_SYMBOLS],
            RotationKind::Complex2D => [f64::from(r % 2 == 0); NUM_SYMBOLS],
            // `Real1D` has no rotation to hand-build; the ladder starts above it.
            RotationKind::Real1D => unreachable!("{rotation:?} has no rotation channels"),
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

/// The block's **four** output axes, for one family — the state itself, not one
/// projection of it.
///
/// `d_model = 2` lets two components through per run, so this runs the same
/// hand-built block twice, once per half. The ceiling is meant to bound every
/// readout the model could have carried, and the model's readout is a linear
/// map of these four numbers (`out_proj` then the head), so a table over a fine
/// partition of all four still dominates it.
fn probe(
    device: &Device,
    rotation: RotationKind,
    family: Family,
    count: usize,
    seed: u64,
) -> (Vec<[f64; RANK]>, Vec<i64>) {
    let (lo, targets) = run(
        &handmade(device, rotation, Head::Probe([0, 1])),
        family,
        count,
        seed,
        device,
    );
    let (hi, _) = run(
        &handmade(device, rotation, Head::Probe([2, 3])),
        family,
        count,
        seed,
        device,
    );
    let channels = lo
        .iter()
        .zip(&hi)
        .map(|(a, b)| [a[0], a[1], b[0], b[1]])
        .collect();
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

/// Quantise each output channel into `LEVELS` equal-width bins over the range
/// observed on the fit split, and pack the four into one code — a finite
/// partition of the block's output space, so the best table over it dominates
/// every linear head the model could have carried.
const LEVELS: usize = 5;
const NUM_OUTPUT_CODES: usize = LEVELS * LEVELS * LEVELS * LEVELS;

fn output_codes(channels: &[[f64; RANK]], range: &[(f64, f64); RANK]) -> Vec<usize> {
    channels
        .iter()
        .map(|o| {
            (0..RANK).fold(0, |code, r| {
                let (lo, hi) = range[r];
                let span = (hi - lo).max(1e-12);
                let level = ((((o[r] - lo) / span) * LEVELS as f64) as usize).min(LEVELS - 1);
                code * LEVELS + level
            })
        })
        .collect()
}

fn channel_range(channels: &[[f64; RANK]]) -> [(f64, f64); RANK] {
    std::array::from_fn(|r| {
        channels
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), o| {
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
/// Both rotations read the same `ϑ` through the same bound, so the abelian twin
/// turns each of its two state pairs by a half-turn per symbol: what reaches
/// the head is the pair of **parities**, which is precisely the abelianisation
/// `Q₈/{±1} ≅ Z₂×Z₂`. Half the group is folded onto the other half, and no
/// readout can unfold it — hence the second column, which hands the block the
/// best table over a fine partition of its whole output space (fitted on a
/// separate split) and still lands near the 50% that guessing the commutator's
/// sign implies.
///
/// A `cumsum` of angles is a function of the symbol counts however the angles
/// are chosen, so this is one point on a curve whose ceiling is
/// [`counts_ceiling_is_the_abelian_limit`]; what no point on it has is the
/// order.
#[test]
fn abelian_rotation_loses_the_order() {
    let device = Device::default();
    let decoder = handmade(&device, RotationKind::Complex2D, Head::Decoder);
    println!(
        "the same construction, abelian rotation ({} params):",
        decoder.num_params()
    );
    println!("      family    same head   best readout");
    let mut best = 0.0f64;
    for (name, family) in FAMILIES {
        let (fit_ch, fit_t) = probe(&device, RotationKind::Complex2D, family, 512, FIT);
        let (eval_ch, eval_t) = probe(&device, RotationKind::Complex2D, family, 256, EVAL);
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
    assert!(
        best < 0.75,
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
    assert!(
        best < 0.85,
        "the symbol counts nearly give the answer: {best}"
    );
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
