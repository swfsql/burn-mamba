//! The three claims this example rests on, measured.
//!
//! 1. A **hand-built** quaternion Mamba-3 block solves the task exactly. There
//!    is no fitting: the unrolled recurrence gives every weight in closed form.
//! 2. The **same construction with the abelian rotation** (`Complex2D`, one enum
//!    knob) does not. A `cumsum` of angles is a function of the symbol counts,
//!    and its state carries exactly the abelianisation `Q₈/{±1} ≅ Z₂×Z₂`. The
//!    commutator, which is the whole content of the task, is gone. The test
//!    measures this twice: through the identical head, and through the **best
//!    readout that its output admits**.
//! 3. **No order-blind model can**, whatever it does with those counts. The
//!    best lookup table from `(#i, #j)` since the reset is far below the block.
//!    It bounds every block that, like this one, writes its state at the reset
//!    and reads the rotation accumulated after it. Under an abelian rotation,
//!    that accumulation *is* a function of the two counts, whatever the angles
//!    and whatever the decoder.
//!
//! The readouts in (2) and (3) are lookup tables **fitted on one split and
//! scored on another**. So they are ceilings that a model can reach, not
//! memorised labels.
//!
//! The construction is the one that [`crate::model`] derives. `R` writes the
//! identity quaternion into the state at the current cumulative rotation. `i`
//! and `j` turn that rotation by non-commuting half-turns. The state is then the
//! relative quaternion: the group element itself. Two heads read it, along the
//! two coordinates of a [`plane`] on which the eight elements of `Q₈` are eight
//! directions `45°` apart, so the head decodes a sector. The ceilings in (2)
//! still use all four components of the state (see [`probe`]).

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

/// `Δ` for every head and every symbol. It is fixed at 1, so the rotation
/// generator is exactly `2π·tanh(‖ϑ‖)`, and `γ = Δ = 1` writes `B` unscaled.
/// (The block runs at `Trapezoid::None`, which gives the whole step to the
/// current token.)
const DELTA: f64 = 1.0;
/// `ϑ` for the axis that a symbol turns about: a half-turn, that is the unit
/// quaternion `i` (or `j`) up to `cos(π/2) ≈ 4e-8`.
///
/// The block bounds one step to `rotation_range · π · Δ`. At the default range
/// of 2, a half-turn is `tanh(‖ϑ‖) = 1/2`: in the interior, where `tanh` is
/// steep and the gradient is alive. (At `range = 1`, the same half-turn is
/// exactly on the asymptote of `tanh`, and only a saturated channel reaches it.
/// No optimiser can arrive there, because the f32 derivative of `tanh` there is
/// exactly zero.) The abelian twin below reads the same constant through the
/// same bound, so it also turns by a half. That makes its result exactly the
/// abelianisation.
const TURN_RAW: f64 = 0.5493061443340549; // atanh(1/2)
/// `Â` on a turn. `A = −softplus(Â)`, floored at the `a_floor` of the block
/// (`1e-4`), so this is the flattest hold that the block allows.
const A_HOLD_RAW: f64 = -20.0;
/// `−A` on a `RESET`: `ᾱ = e⁻²⁰` erases what the state held.
const A_WIPE: f64 = 20.0;
/// `x(R) = 1`: the write. `x(i) = x(j) = 0` (`x` takes no activation), so a turn
/// writes nothing and only advances the rotation.
const X_WRITE: f64 = 1.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// Class-logit gain on the plane that the readout folds onto.
const OUT_GAIN: f64 = 3.0;

/// `state_rank`: the four components of the quaternion in the state.
const RANK: usize = 4;

/// `nheads`, which at `per_head_dim = 1` is also `d_inner`.
///
/// Two, because the *readout* needs two. Every head holds its own copy of the
/// same quaternion (same `B`, same `ᾱ`, same rotation). The heads differ only in
/// the `C` with which they read that copy, so `nheads` counts **projections**,
/// not state. Eight group elements fit on eight directions of a plane
/// ([`plane`]), and two projections separate them. A third and a fourth would
/// be duplicates that the head cannot use.
const NHEADS: usize = 2;

/// `d_model`. Two, the floor for a three-symbol alphabet: the pre-`RmsNorm` of
/// the layer sends a token to the unit sphere, so a 1-D token carries only its
/// sign. It equals `d_inner` here, so the `out_proj` of the block is the
/// identity, and the two heads *are* the plane.
const D_MODEL: usize = 2;

// ---------------------------------------------------------------------------
// scalar helpers
// ---------------------------------------------------------------------------

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

/// The three symbol embeddings, each of norm `√2`, so the pre-`RmsNorm` of the
/// layer (`γ = 1`) does not change them. Indexed by
/// [`TURN_I`] / [`TURN_J`] / [`RESET`].
///
/// Three points of `ℝ²` are affinely independent. So a channel that must take
/// the values `(t_i, t_j, t_R)` needs only one 3×3 [`solve3`] (weight plus
/// bias), as in `reset-majority` and `reset-rotor`. (Orthogonal embeddings need
/// no solve, but they need `d_model = 3` at least, and this construction
/// minimises the block.)
const EMBED: [[f64; D_MODEL]; NUM_SYMBOLS] = [
    [std::f64::consts::SQRT_2, 0.0],
    [0.0, std::f64::consts::SQRT_2],
    [-1.0, -1.0],
];

/// The plane on which the two heads read the state: state component `r` goes to
/// the unit vector at `r · 45°`.
///
/// The eight elements of `Q₈` are `±eᵣ`, so they land on the eight directions
/// `45°` apart. These are distinct, equidistant, and in convex position, which
/// is exactly what a linear eight-way head needs. The block never forms this
/// plane as a separate step. Head `h` reads the state with [`c_axis`], the
/// `h`-th coordinate of this map, so the `out_proj` of the block is the
/// identity.
fn plane(r: usize) -> [f64; D_MODEL] {
    let angle = std::f64::consts::FRAC_PI_4 * r as f64;
    [angle.cos(), angle.sin()]
}

/// [`plane`] applied to a quaternion: where a group element lands.
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

/// The `C` vector with which head `h` reads the state: the `h`-th coordinate of
/// [`plane`] over the four components, scaled to norm 2. That is the length
/// that QK-Norm gives the construction. So the two heads share one positive
/// factor, and `(y₀, y₁) ∝ plane_point(q_rel)` exactly.
fn c_axis(h: usize) -> [f64; RANK] {
    let raw: [f64; RANK] = std::array::from_fn(|r| plane(r)[h]);
    let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    raw.map(|v| 2.0 * v / norm)
}

/// What the head of the network reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    /// The nearest-element decoder: logit `g` ∝ `⟨p, plane_point(g)⟩` over the
    /// eight elements of `Q₈`, read on the plane where [`plane`] puts them.
    Decoder,
    /// Point the `C` of the two heads at the two named state components
    /// instead, and pass them through as the first two logits. A test can then
    /// search over every readout that the head can express. Two heads carry
    /// only two components at a time, so [`probe`] runs the block twice.
    Probe([usize; 2]),
}

/// Build the block by hand for the given rotation.
///
/// The two rotations differ in the meaning of the rotation channels:
///
/// - [`RotationKind::Quaternion4D`] takes three per head: a scaled rotation
///   axis. So `i` and `j` become half-turns about two orthogonal axes, which do
///   not commute.
/// - [`RotationKind::Complex2D`] takes two, one angle per state pair. So `i`
///   and `j` become half-turns of the two pairs, which commute.
///
/// The only other difference is the vector that the write stores. [`b_channels`]
/// tells why the abelian twin gets a different (and better) one.
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
    // The 2s are `nheads`, and the 4s are `state_rank`. `Trapezoid::None`
    // projects no λ. Each entry is the value of the channel at
    // (TURN_I, TURN_J, RESET), *before* its activation.
    let rotation_channels: Vec<[f64; NUM_SYMBOLS]> = match rotation {
        // `Real1D` has no rotation to hand-build. The ladder starts above it.
        RotationKind::Real1D => unreachable!("{rotation:?} has no rotation channels"),
        // A scaled rotation axis per head: `i` turns π about x, `j` about y.
        // The block projects the generators **per head** (`nheads · 3` channels
        // here), so every head gets its own copy. This construction wants both
        // heads to read the same rotation, but the block does not force that.
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
    channels.extend([[0.0, 0.0, X_WRITE]; NHEADS]); // x: only R writes
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
    // the identity quaternion for both heads, and C starts there too. The bias
    // of head h moves its C: to `c_axis(h)` for the decoder, or to twice a
    // basis quaternion for a probe. It is the only per-head weight in the
    // construction, and it is the whole readout: two projections of one state.
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
    // eight directions 45° apart. With `ignore_last_residual`, the head sees
    // only the output of the block.
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
/// every symbol. QK-Norm scales it to `(2, 0, 0, 0)`. The per-head bias then
/// moves head `h` to where that head reads: [`c_axis`], or a basis quaternion
/// for a probe.
fn basis_channels() -> Vec<[f64; NUM_SYMBOLS]> {
    (0..RANK).map(|r| [f64::from(r == 0); NUM_SYMBOLS]).collect()
}

/// The four `B` channels: the vector that `R` writes into the state.
///
/// For the quaternion rotation, it is the unit of the group, `(1, 0, 0, 0)`.
/// The state then holds the group element itself.
///
/// The abelian twin gets `(1, 0, 1, 0)` instead: one unit in **each** rotated
/// pair. This is not a detail. It is the fairest analogue. `Complex2D` turns
/// the two pairs by two independent cumulative angles. With `B = (1,0,0,0)`,
/// the second pair multiplies zero, so the twin carries one parity instead of
/// two. With a unit in both pairs, its state carries `(#i mod 2, #j mod 2)`:
/// the whole abelianisation `Q₈/{±1}`, which is the most that any sum of
/// angles can hold.
fn b_channels(rotation: RotationKind) -> Vec<[f64; NUM_SYMBOLS]> {
    (0..RANK)
        .map(|r| match rotation {
            RotationKind::Quaternion4D | RotationKind::Rotor4D => [f64::from(r == 0); NUM_SYMBOLS],
            RotationKind::Complex2D => [f64::from(r % 2 == 0); NUM_SYMBOLS],
            // `Real1D` has no rotation to hand-build. The ladder starts above it.
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

/// Run `model` over `count` sequences of one family. Return the per-position
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

/// The **four** output axes of the block, for one family: the state itself, not
/// one projection of it.
///
/// `d_model = 2` lets two components through per run, so this runs the same
/// hand-built block twice, once per half. The ceiling must bound every readout
/// that the model can carry. The readout of the model is a linear map of these
/// four numbers (`out_proj`, then the head). So a table over a fine partition
/// of all four still dominates it.
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

/// Per-position accuracy of the head of the model (argmax over the class
/// logits).
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

/// Accuracy of the best lookup table from a discrete code to a class. It is
/// fitted on one split and scored on another, so it is a ceiling that a model
/// can reach, not a memorised answer key.
///
/// A code that the fit split does not contain gets the majority class of the
/// fit split.
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
/// of the fit split, and pack the four into one code. That is a finite
/// partition of the output space of the block, so the best table over it
/// dominates every linear head that the model can carry.
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

/// Collect the per-position codes that an order-blind model can key on, for
/// one family: the symbol, and the `(#i, #j)` counts since the reset. Also
/// collect the targets.
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

/// Every weight is written in closed form. There is no training.
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

/// The identical construction with `RotationKind::Complex2D` (the one enum
/// knob), reported in two ways.
///
/// Both rotations read the same `ϑ` through the same bound, so the abelian twin
/// turns each of its two state pairs by a half-turn per symbol. What reaches
/// the head is the pair of **parities**, which is exactly the abelianisation
/// `Q₈/{±1} ≅ Z₂×Z₂`. Half the group folds onto the other half, and no readout
/// can unfold it. The second column shows this. It gives the block the best
/// table over a fine partition of its whole output space (fitted on a separate
/// split), and the result is still near the 50% of a guess of the sign of the
/// commutator.
///
/// A `cumsum` of angles is a function of the symbol counts for any choice of
/// angles. So this is one point on a curve, and
/// [`counts_ceiling_is_the_abelian_limit`] is its ceiling. No point on the
/// curve has the order.
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
/// accumulated after it (this construction, under an abelian rotation) can see
/// no more than this. An abelian accumulation is a `cumsum`, so it is a linear
/// function of exactly these two numbers, whatever the angles and whatever the
/// decoder. (A block that writes on every symbol can carry an order-dependent
/// sum. The trained `--rotation complex` run in the README covers that case,
/// and it does not clear this ceiling either.)
///
/// The gap that the counts leave is the commutator. The counts fix the element
/// only up to `{±1}`, so where both signs occur, the table must guess. In
/// `runs`, the counts come closest to enough: the word is a few long blocks, so
/// the number of each symbol nearly fixes its reduction. In `shuffle`, the
/// counts tell the least.
#[test]
fn counts_ceiling_is_the_abelian_limit() {
    println!("order-blind ceilings, accuracy per family:");
    println!("      family    memoryless   by (#i,#j)   ambiguous");
    let mut best = 0.0f64;
    for (name, family) in FAMILIES {
        let (fit_sym, fit_cnt, fit_t) = codes(family, 2048, FIT);
        let (eval_sym, eval_cnt, eval_t) = codes(family, 512, EVAL);
        // The share of the mass on a (counts) cell that carries both signs.
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

/// The labels of the dataset are the `Q₈` word problem: the product is
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
