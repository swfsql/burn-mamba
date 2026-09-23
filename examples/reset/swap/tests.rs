//! The claims this example rests on, measured.
//!
//! 1. A **hand-built `Rotor4D`** Mamba-3 block solves the task exactly. There
//!    is no fitting: every weight is written in closed form. The rotation of
//!    the block is set to conjugation (`p = q`), which is `SO(3)`, and the
//!    three swaps become three half-turns about three axes `60°` apart.
//! 2. The **same construction one rung down** (`Quaternion4D`, one enum knob)
//!    does not. It does not forget: a left-isoclinic state carries the *double
//!    cover* `2D₃`, which has **more** information than the label. But the two
//!    lifts `±W` of one permutation are **antipodal** state vectors with the
//!    same target, and a linear readout cannot merge them. The test measures
//!    this in three ways: through the identical head, through the best table
//!    over its output space (which *does* recover it, and that is the point),
//!    and by how completely the two lifts cancel.
//! 3. **No order-blind model can**: the ceiling from the symbol counts. In
//!    particular, the sign character `(−1)^(#l+#r)` cannot. It is all that a
//!    *homomorphism* `S₃ → SU(2)` can carry, because `SU(2)` has exactly one
//!    element of order two, and `S₃` has three.
//! 4. The wall in (2) is the **linear head**, so it is a config choice. Behind
//!    the final `RmsNorm` of the network, a hand-built `Quaternion4D` block is
//!    exact too. This is also true in the input of that norm, when a constant
//!    is next to the state.
//!
//! The readouts in (2) and (3) are lookup tables **fitted on one split and
//! scored on another**. So they are ceilings that a model can reach, not
//! memorised labels.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    Family, NUM_CLASSES, NUM_SYMBOLS, PERMS, REF_POINT, RESET, ResetSwapDataset, SEQ_LENGTH,
    SWAP_L, SWAP_R, class_of, compose, counts_since_reset, labels, one_hot, point, quat_mul,
    swap_axis, symbol_perm, symbol_quat,
};
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

// ---------------------------------------------------------------------------
// the construction's constants
// ---------------------------------------------------------------------------

/// `Δ` for every head and every symbol. It is fixed at 1, so the rotation
/// generator is exactly `range·π·tanh(ϑ)`, and `γ = Δ = 1` writes `B`
/// unscaled. (The block runs at `Trapezoid::None`, which gives the whole step
/// to the current token.)
const DELTA: f64 = 1.0;
/// `‖ϑ‖` for a **half-turn**: the same constant as in `reset-spinor`, for the
/// same reason. The block bounds one step to `rotation_range · π · Δ`, and the
/// default range of 2 puts a half-turn at `tanh(‖ϑ‖) = 1/2`: in the interior,
/// where the gradient is alive.
///
/// This task needs only half-turns. A transposition has order two, and under
/// conjugation, the *rotation* that it induces is a `180°` turn about its axis.
const TURN_RAW: f64 = 0.5493061443340549; // atanh(1/2)
/// `Â` on a swap. `A = −softplus(Â)`, floored at the `a_floor` of the block
/// (`1e-4`), so this is the flattest hold that the block allows.
const A_HOLD_RAW: f64 = -20.0;
/// `−A` on a `RESET`: `ᾱ = e⁻²⁰` erases what the state held.
const A_WIPE: f64 = 20.0;
/// `x(R) = 1`: the write. `x(l) = x(r) = 0` (`x` takes no activation), so a swap
/// writes nothing and only turns the state.
const X_WRITE: f64 = 1.0;
/// The gate `z`, constant and positive so it never flips a sign.
const Z_PRE: f64 = 5.0;
/// Class-logit gain on the plane that the readout folds onto.
const OUT_GAIN: f64 = 3.0;

/// `state_rank`: the four components of the 4-block that the rotation turns.
const RANK: usize = 4;

/// `nheads`, which at `per_head_dim = 1` is also `d_inner`.
///
/// Two, because the *readout* needs two. Every head holds its own copy of the
/// same rotated vector (same `B`, same `ᾱ`, same rotation). The heads differ
/// only in the `C` with which they read that copy, so `nheads` counts
/// **projections**, not state. Two projections separate the six orbit points
/// (see [`PLANE_AXES`]).
const NHEADS: usize = 2;

/// `d_model`. Two, the floor for a three-symbol alphabet: the pre-`RmsNorm` of
/// the layer sends a token to the unit sphere, so a 1-D token carries only its
/// sign. It equals `d_inner` here, so the `out_proj` of the block is the
/// identity, and the two heads *are* the plane.
const D_MODEL: usize = 2;

/// The two state components that the heads read: the `x` and `y` of the
/// rotation.
///
/// Conjugation fixes the real axis, so component 0 is constant. The six orbit
/// points of [`point`] all have the *same* `z` up to its sign (a half-turn about
/// an axis of the `xy`-plane flips it). So `z` carries only the parity that the
/// counts already give. What is left is six directions of one plane, all of
/// norm `√2` and pairwise distinct. They lie on a circle, so they are in convex
/// position, which is exactly what a linear six-way head needs.
const PLANE_AXES: [usize; NHEADS] = [1, 2];

/// [`PLANE_AXES`] applied to a state vector: where a permutation lands.
fn plane_point(v: [f64; RANK]) -> [f64; D_MODEL] {
    std::array::from_fn(|c| v[PLANE_AXES[c]])
}

// ---------------------------------------------------------------------------
// scalar helpers
// ---------------------------------------------------------------------------

/// The gate activation, `silu(z) = z·σ(z)`.
fn silu(t: f64) -> f64 {
    t / (1.0 + (-t).exp())
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

/// The three symbol embeddings, each of norm `√2`, so the pre-`RmsNorm` of the
/// layer (`γ = 1`) does not change them. Indexed by
/// [`SWAP_L`] / [`SWAP_R`] / [`RESET`].
///
/// Three points of `ℝ²` are affinely independent. So a channel that must take
/// the values `(v_l, v_r, v_R)` needs only one 3×3 [`solve3`] (weight plus
/// bias), as in `reset-majority` and `reset-rotor`.
const EMBED: [[f64; D_MODEL]; NUM_SYMBOLS] = [
    [std::f64::consts::SQRT_2, 0.0],
    [0.0, std::f64::consts::SQRT_2],
    [-1.0, -1.0],
];

/// What the head of the network reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    /// The nearest-point decoder: logit `g` ∝ `⟨p, plane_point(point(g))⟩` over
    /// the six permutations, read on the plane that [`PLANE_AXES`] keeps.
    Decoder,
    /// Pass the two named state components through as the first two logits. A
    /// test can then search over every readout that the head can express.
    /// `d_model = 2` carries only two components at a time, so [`probe`] runs
    /// the block twice.
    Probe([usize; 2]),
}

/// Build the block by hand for the given rotation.
///
/// The three rotations differ only in the meaning of the rotation channels:
///
/// - [`RotationKind::Rotor4D`] takes six per head: a left and a right scaled
///   axis. When they are **equal**, the step is `v ↦ q v q̄`, conjugation, that
///   is a true `SO(3)` rotation by `180°` about [`swap_axis`]. This is the one
///   that works.
/// - [`RotationKind::Quaternion4D`] takes three: the left factor alone. So the
///   step is `v ↦ q v`, and `q² = −1`, not `1`: the state runs in the double
///   cover.
/// - [`RotationKind::Complex2D`] takes two, one angle per state pair. So `l`
///   and `r` become half-turns of the two pairs, which commute.
fn handmade(device: &Device, rotation: RotationKind, head: Head) -> MambaLatentNet {
    let cfg = crate::model::model_config(rotation);
    let mut model = ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("reset-swap configures the Mamba-3 variant")
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
    // projects no λ, so the whole mass of the step is `γ = Δ`. Each entry is
    // the value of the channel at (SWAP_L, SWAP_R, RESET), *before* its
    // activation.
    let mut channels: Vec<[f64; NUM_SYMBOLS]> = Vec::new();
    channels.extend([[Z_PRE; NUM_SYMBOLS]; NHEADS]); // z
    channels.extend([[0.0, 0.0, X_WRITE]; NHEADS]); // x: only R writes
    channels.extend(b_channels(rotation)); // B_raw — the vector the write stores
    channels.extend(basis_channels()); // C_raw (the per-head bias aims it)
    channels.extend([[softplus_inv(DELTA); NUM_SYMBOLS]; NHEADS]); // Δ
    channels.extend([[A_HOLD_RAW, A_HOLD_RAW, softplus_inv(A_WIPE)]; NHEADS]); // A
    channels.extend(rotation_channels(rotation));

    let rows = [
        [EMBED[SWAP_L][0], EMBED[SWAP_L][1], 1.0],
        [EMBED[SWAP_R][0], EMBED[SWAP_R][1], 1.0],
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
    // QK-Norm rescales B and C, but it cannot turn them. The bias of head h
    // moves C from the shared (1,0,0,0) to twice one basis vector, and that
    // aims the head at one component of the state. It is the only per-head
    // weight here, and it is the whole readout: `PLANE_AXES` for the decoder,
    // and the two components that a probe asks for.
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(
        Shape::new([NHEADS, 1, RANK]),
        device,
    ));
    let kept = match head {
        Head::Decoder => PLANE_AXES,
        Head::Probe(axes) => axes,
    };
    let c_bias: Vec<f64> = (0..NHEADS)
        .flat_map(|h| {
            (0..RANK).map(move |r| 2.0 * (f64::from(r == kept[h]) - f64::from(r == 0)))
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

    // ── the class head: nearest orbit point ──────────────────────────────────
    // logit_g ∝ ⟨p, plane_point(point(g))⟩ over the six permutations, which lie
    // on six distinct directions of one circle. With `ignore_last_residual`, the
    // head sees only the output of the block.
    let mut w_out = vec![0.0f64; D_MODEL * NUM_CLASSES];
    match head {
        Head::Decoder => {
            for class in 0..NUM_CLASSES {
                let p = plane_point(point(class as i64));
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

/// The rotation channels, per [`RotationKind`]. Every entry is the value of the
/// channel at `(SWAP_L, SWAP_R, RESET)`.
fn rotation_channels(rotation: RotationKind) -> Vec<[f64; NUM_SYMBOLS]> {
    // The scaled axis of each swap: direction = the axis, magnitude = a half-turn.
    let axis = |k: usize| {
        [
            TURN_RAW * swap_axis(SWAP_L)[k],
            TURN_RAW * swap_axis(SWAP_R)[k],
            0.0,
        ]
    };
    match rotation {
        // `Real1D` has no rotation to hand-build. The ladder starts above it.
        RotationKind::Real1D => unreachable!("{rotation:?} has no rotation channels"),
        // Left and right generators **equal** ⇒ v ↦ q v q̄, conjugation, SO(3).
        // Channels are laid out [head][left | right][x, y, z].
        RotationKind::Rotor4D => (0..NHEADS)
            .flat_map(|_| [axis(0), axis(1), axis(2), axis(0), axis(1), axis(2)])
            .collect(),
        // The left factor alone ⇒ v ↦ q v, whose square is −1: the double cover.
        RotationKind::Quaternion4D => {
            (0..NHEADS).flat_map(|_| [axis(0), axis(1), axis(2)]).collect()
        }
        // one angle per state pair: `l` turns pair 0 by π, `r` turns pair 1 by π
        RotationKind::Complex2D => vec![[TURN_RAW, 0.0, 0.0], [0.0, TURN_RAW, 0.0]],
    }
}

/// The four `C` channels, carrying `(1, 0, 0, 0)` for every symbol. The
/// per-head bias then aims head `h` at one basis vector.
fn basis_channels() -> Vec<[f64; NUM_SYMBOLS]> {
    (0..RANK).map(|r| [f64::from(r == 0); NUM_SYMBOLS]).collect()
}

/// The four `B` channels: the vector that `R` writes into the state.
///
/// For the two quaternion rotations, it is [`REF_POINT`]: a point of the
/// imaginary 3-space of the rotation, on none of the axes of the group, so its
/// orbit is six distinct points.
///
/// The abelian twin gets `(1, 0, 1, 0)` instead: one unit in **each** rotated
/// pair, so its state carries both parities, not one. As in `reset-spinor`,
/// that is not a detail but the fairest analogue.
fn b_channels(rotation: RotationKind) -> Vec<[f64; NUM_SYMBOLS]> {
    (0..RANK)
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

/// Run `model` over `count` sequences of one family. Return the per-position
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
/// hand-built block twice, once per half. Every ceiling below must bound what a
/// readout of the *state* can do. The readout of the model is a linear map of
/// these four numbers (`out_proj`, then the head). So a search over all four is
/// on the conservative side.
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

/// Accuracy of the **best linear readout** of the output of the block: a
/// softmax regression on the four output channels plus a bias, fitted on one
/// split and scored on another.
///
/// This is the fair version of "what a head like the head of this example can
/// do". The head of the example is one linear map, and this searches over all
/// of them. In this column, the antipodal pairs of the left-isoclinic twin have
/// their effect. The [`best_lookup`] over [`output_codes`] dominates this column
/// on purpose (a table is not linear, and that gap is the finding).
fn best_linear_readout(fit: (&[[f64; RANK]], &[i64]), eval: (&[[f64; RANK]], &[i64])) -> f64 {
    // Standardise by the global RMS so one learning rate fits every block.
    let rms = {
        let n = (fit.0.len() * RANK) as f64;
        let sq: f64 = fit
            .0
            .iter()
            .flat_map(|o| o[..RANK].iter().map(|v| v * v))
            .sum();
        (sq / n).sqrt().max(1e-12)
    };
    let feats = |o: &[f64; RANK]| {
        let mut f = [0.0f64; RANK + 1];
        for r in 0..RANK {
            f[r] = o[r] / rms;
        }
        f[RANK] = 1.0; // bias
        f
    };
    let mut w = [[0.0f64; RANK + 1]; NUM_CLASSES];
    let lr = 1.0;
    for _ in 0..1200 {
        let mut grad = [[0.0f64; RANK + 1]; NUM_CLASSES];
        for (o, &t) in fit.0.iter().zip(fit.1) {
            let f = feats(o);
            let logits: [f64; NUM_CLASSES] =
                std::array::from_fn(|c| (0..=RANK).map(|k| w[c][k] * f[k]).sum());
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: [f64; NUM_CLASSES] = std::array::from_fn(|c| (logits[c] - max).exp());
            let sum: f64 = exp.iter().sum();
            for c in 0..NUM_CLASSES {
                let d = exp[c] / sum - f64::from(c as i64 == t);
                for k in 0..=RANK {
                    grad[c][k] += d * f[k];
                }
            }
        }
        let scale = lr / fit.0.len() as f64;
        for c in 0..NUM_CLASSES {
            for k in 0..=RANK {
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
                    let la: f64 = (0..=RANK).map(|k| w[a][k] * f[k]).sum();
                    let lb: f64 = (0..=RANK).map(|k| w[b][k] * f[k]).sum();
                    la.partial_cmp(&lb).unwrap()
                })
                .unwrap();
            pred as i64 == t
        })
        .count();
    hits as f64 / eval.1.len() as f64
}

/// Quantise each output channel into `LEVELS` equal-width bins over the range
/// of the fit split, and pack the four into one code. That is a finite
/// partition of the output space of the block, so the best table over it
/// dominates every readout that the model can carry, linear or not.
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
/// one family: the symbol, the `(#l, #r)` counts since the reset, and the
/// parity `(#l + #r) mod 2`. Also collect the targets.
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

/// Number of distinct `(#l, #r)` codes [`codes`] can emit.
const NUM_COUNT_CODES: usize = SEQ_LENGTH * SEQ_LENGTH;

// ---------------------------------------------------------------------------
// 1. the hand-built solution
// ---------------------------------------------------------------------------

/// Every weight is written in closed form. There is no training.
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

/// `Quaternion4D` (the left factor alone), reported in three ways.
///
/// The failure here is **not** the abelian failure. The left-isoclinic state
/// carries the double cover `2D₃` (order 12), which is strictly *more* than the
/// label: the best table over its output space recovers the permutation fully.
/// But it cannot present the permutation to a **linear** head. The two lifts
/// `±W` of one permutation are antipodal vectors with the same target. So every
/// linear functional of the state takes opposite values on the same class, and
/// the average over a class cancels. The last column measures exactly that
/// cancellation: `‖mean output‖ / rms output` per class. The conjugating block
/// leaves it at 1, and the left-isoclinic block drives it to ~0.
#[test]
fn left_isoclinic_carries_a_double_cover() {
    let device = Device::default();
    let decoder = handmade(&device, RotationKind::Quaternion4D, Head::Decoder);
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
        let (fit_ch, fit_t) = probe(&device, RotationKind::Quaternion4D, family, 256, FIT);
        let (eval_ch, eval_t) = probe(&device, RotationKind::Quaternion4D, family, 256, EVAL);
        let range = channel_range(&fit_ch);
        let table = best_lookup(
            (&output_codes(&fit_ch, &range), &fit_t),
            (&output_codes(&eval_ch, &range), &eval_t),
            NUM_OUTPUT_CODES,
        );
        let linear = best_linear_readout((&fit_ch, &fit_t), (&eval_ch, &eval_t));
        let cancel = mean_over_rms(&eval_ch, &eval_t);
        let (rotor_fit, rotor_fit_t) = probe(&device, RotationKind::Rotor4D, family, 256, FIT);
        let (rotor_ch, rotor_t) = probe(&device, RotationKind::Rotor4D, family, 256, EVAL);
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
fn mean_over_rms(channels: &[[f64; RANK]], targets: &[i64]) -> f64 {
    let mut sum = vec![[0.0f64; RANK]; NUM_CLASSES];
    let mut sq = vec![0.0f64; NUM_CLASSES];
    let mut n = vec![0.0f64; NUM_CLASSES];
    for (o, &t) in channels.iter().zip(targets) {
        let c = t as usize;
        n[c] += 1.0;
        for r in 0..RANK {
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
/// of angles is a function of the symbol counts, and `lr ≠ rl`.
#[test]
fn abelian_rotation_loses_the_order() {
    let device = Device::default();
    println!("the same construction, abelian rotation:");
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

/// What an order-blind model can do, and what the **sign character** alone can
/// do.
///
/// The second is the sharp one for this example. Every finite subgroup of
/// `SU(2)` has exactly one element of order two. So the only nontrivial
/// homomorphism from `S₃` into `SU(2)` sends the three transpositions to `−1`:
/// the parity, and nothing else. So the `parity` column bounds a left-isoclinic
/// block whose rotation is a homomorphic image of the word. The hand-built twin
/// gets past this bound, because it tracks the double cover instead. But the
/// information that it gets is no longer linearly readable.
#[test]
fn counts_and_parity_ceilings() {
    println!("order-blind ceilings, accuracy per family:");
    println!("      family    memoryless   by parity   by (#l,#r)");
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

/// The labels of the dataset are the `S₃` word problem, and `S₃` is a group
/// that `SU(2)` cannot hold: it has **three** elements of order two.
#[test]
fn labels_are_the_symmetric_group() {
    // lr ≠ rl: the reset-spinor requirement, inherited
    let lr = labels(&[RESET, SWAP_R, SWAP_L]); // newest on the left: l∘r
    let rl = labels(&[RESET, SWAP_L, SWAP_R]); // r∘l
    assert_ne!(lr[2], rl[2], "lr and rl must differ");
    assert_eq!(lr[2], 3, "l∘r = cab");
    assert_eq!(rl[2], 4, "r∘l = bca");

    // every swap is an involution, and there are three of them
    let squared = labels(&[RESET, SWAP_L, SWAP_L]);
    assert_eq!(squared[2], 0, "l² = 1");
    let involutions: Vec<i64> = (0..NUM_CLASSES as i64)
        .filter(|&c| c != 0 && class_of(compose(PERMS[c as usize], PERMS[c as usize])) == 0)
        .collect();
    assert_eq!(involutions.len(), 3, "S₃ has three involutions");

    // (l∘r)³ = 1: the two axes are 60° apart
    let cubed = labels(&[RESET, SWAP_R, SWAP_L, SWAP_R, SWAP_L, SWAP_R, SWAP_L]);
    assert_eq!(cubed[6], 0, "(lr)³ = 1");

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

/// The obstruction, in three lines of quaternion algebra. The lift of a swap
/// squares to `−1`, so left multiplication cannot represent an involution.
/// Conjugation by the same lift can, because `±q` conjugate identically.
#[test]
fn the_lift_of_a_swap_squares_to_minus_one() {
    for symbol in [SWAP_L, SWAP_R] {
        let q = symbol_quat(symbol);
        let q2 = quat_mul(q, q);
        assert!(
            (q2[0] + 1.0).abs() < 1e-12 && q2[1..].iter().all(|v| v.abs() < 1e-12),
            "q² should be −1 for a half-turn lift, got {q2:?}"
        );
        // But the *rotation* that it induces is an involution: two conjugations
        // give the identity on the whole space.
        let v = REF_POINT;
        let once = conjugate(q, v);
        let twice = conjugate(q, once);
        for (a, b) in twice.iter().zip(&v) {
            assert!((a - b).abs() < 1e-12, "conjugation should square to 1");
        }
    }

    // The six orbit points are distinct, and that makes the readout a decoder.
    for a in 0..NUM_CLASSES as i64 {
        for b in 0..a {
            let (pa, pb) = (point(a), point(b));
            let d: f64 = pa.iter().zip(&pb).map(|(x, y)| (x - y) * (x - y)).sum();
            assert!(d > 0.1, "orbit points {a} and {b} coincide");
        }
    }

    // The orbit is the action of the group: point(g∘h) = g · point(h).
    for g in [SWAP_L, SWAP_R] {
        for h in 0..NUM_CLASSES as i64 {
            let composed = class_of(compose(symbol_perm(g), PERMS[h as usize]));
            let rotated = conjugate(symbol_quat(g), point(h));
            for (x, y) in rotated.iter().zip(&point(composed)) {
                assert!((x - y).abs() < 1e-9, "the action is not the composition");
            }
        }
    }
}

/// `q v q̄`: the two-sided step on which the block of this example is built.
fn conjugate(q: [f64; 4], v: [f64; 4]) -> [f64; 4] {
    quat_mul(quat_mul(q, v), [q[0], -q[1], -q[2], -q[3]])
}

// ---------------------------------------------------------------------------
// 5. what the linear head is load-bearing for
// ---------------------------------------------------------------------------

/// The left-isoclinic block **solves** the task when the final `RmsNorm` of the
/// network is on. That is why `model_config` keeps it off.
///
/// The state is still `±W ⊗ B`. One head reads `y = ⟨C, W ⊗ B⟩ = ⟨u, W⟩`, and
/// the out-proj bias of the block puts a constant `e` next to it.
/// `RmsNorm(e, y)` then has the first coordinate `e / √((e² + y²)/2)`, which is
/// **even** in `y`, so both lifts land on one point. `u` gives six distinct
/// values of `|⟨u, W⟩|` (`15°` into both planes of `2D₃`, the odd one scaled),
/// and the head cuts that one coordinate into six intervals. The same weights
/// without the norm are near chance. Behind a linear head, the outputs of every
/// class average to a point that depends only on the sign character, and three
/// classes cannot share the point of a convex region.
#[test]
fn left_isoclinic_with_final_norm() {
    let phi = 15.0f64.to_radians();
    const ODD_SCALE: f64 = 0.55;
    const E: f64 = 0.35;
    const SLOPE: f64 = 10.0;
    let device = Device::default();

    // both lifts of every class, by walking the group from the identity
    let mut lifts: Vec<Vec<[f64; RANK]>> = vec![Vec::new(); NUM_CLASSES];
    let mut frontier = vec![([1.0, 0.0, 0.0, 0.0], PERMS[0])];
    while let Some((q, p)) = frontier.pop() {
        let class = class_of(p) as usize;
        if lifts[class].iter().any(|l| l.iter().zip(&q).all(|(a, b)| (a - b).abs() < 1e-9)) {
            continue;
        }
        lifts[class].push(q);
        for s in [SWAP_L, SWAP_R] {
            frontier.push((quat_mul(symbol_quat(s), q), compose(symbol_perm(s), p)));
        }
    }
    assert!(lifts.iter().all(|l| l.len() == 2), "expected the double cover");

    // B as the block sees it (QK-normed), and C = u ⊗ B / |B|² so ⟨C, W⊗B⟩ = ⟨u, W⟩
    let rms = (REF_POINT.iter().map(|v| v * v).sum::<f64>() / RANK as f64).sqrt();
    let b = REF_POINT.map(|v| v / rms);
    let u = [phi.cos(), ODD_SCALE * phi.cos(), ODD_SCALE * phi.sin(), phi.sin()];
    let b_sq: f64 = b.iter().map(|v| v * v).sum();
    let c = quat_mul(u, b).map(|v| v / b_sq);

    // The scale of y: the gate, times the write `x(R)` as projected (x takes no
    // activation).
    let gate = silu(Z_PRE) * X_WRITE;
    let e = gate * E; // the out-proj bias, at the gated y's scale
    let n0: Vec<f64> = lifts
        .iter()
        .map(|ls| {
            let ys: Vec<f64> = ls
                .iter()
                .map(|w| gate * c.iter().zip(&quat_mul(*w, b)).map(|(x, y)| x * y).sum::<f64>())
                .collect();
            assert!((ys[0] + ys[1]).abs() < 1e-9, "lifts should be antipodal");
            e / ((e * e + ys[0] * ys[0]) / 2.0).sqrt()
        })
        .collect();
    let mut order: Vec<usize> = (0..NUM_CLASSES).collect();
    order.sort_by(|&a, &b| n0[a].partial_cmp(&n0[b]).unwrap());
    let gap = order.windows(2).map(|w| n0[w[1]] - n0[w[0]]).fold(f64::MAX, f64::min);
    println!("class n0 = {n0:.4?}, min gap {gap:.4}");

    let build = |with_norm: bool| {
        let mut model = handmade(&device, RotationKind::Quaternion4D, Head::Decoder);
        let MambaLatentNet::Mamba3(net) = &mut model else { unreachable!() };
        if with_norm {
            net.norm_f = Some(RmsNormConfig::new(D_MODEL).init(&device));
        }
        let block = &mut net.layers.real_layers[0].block;
        // The C of every head: the QK-normed (1,0,0,0) is (2,0,0,0), and the
        // bias moves it to `c`.
        let c_bias: Vec<f64> = (0..NHEADS)
            .flat_map(|_| (0..RANK).map(|r| c[r] - 2.0 * f64::from(r == 0)))
            .collect();
        block.c_bias_hmr = Param::from_tensor(t1(&c_bias, [NHEADS, 1, RANK], &device));
        // Head 0 → coordinate 1. Coordinate 0 is the constant `e`.
        let w_block = [0.0, 1.0, 0.0, 0.0];
        block.out_proj.weight = Param::from_tensor(t1(&w_block, [NHEADS, D_MODEL], &device));
        block.out_proj.bias = Some(Param::from_tensor(t1(&[e, 0.0], [D_MODEL], &device)));
        // the head: upper envelope of lines in the first normed coordinate
        let mut w_out = vec![0.0f64; D_MODEL * NUM_CLASSES];
        let mut b_out = vec![0.0f64; NUM_CLASSES];
        for (i, &class) in order.iter().enumerate() {
            w_out[class] = SLOPE * i as f64;
            if i > 0 {
                let prev = order[i - 1];
                let cut = 0.5 * (n0[prev] + n0[class]);
                b_out[class] = b_out[prev] - SLOPE * cut;
            }
        }
        net.out_proj.weight = Param::from_tensor(t1(&w_out, [D_MODEL, NUM_CLASSES], &device));
        net.out_proj.bias = Some(Param::from_tensor(t1(&b_out, [NUM_CLASSES], &device)));
        model
    };

    let normed = build(true);
    let plain = build(false);
    println!(
        "left-isoclinic, the same weights with / without norm_f ({} / {} params):",
        normed.num_params(),
        plain.num_params()
    );
    let mut worst = 1.0f64;
    let mut best_plain = 0.0f64;
    for (name, family) in FAMILIES {
        let acc = accuracy(&normed, family, 256, &device);
        let acc_plain = accuracy(&plain, family, 256, &device);
        println!("  {name:<9} {:6.2}%   {:6.2}%", 100.0 * acc, 100.0 * acc_plain);
        worst = worst.min(acc);
        best_plain = best_plain.max(acc_plain);
    }
    assert!(worst > 0.995, "left-isoclinic + final norm is not exact: {worst}");
    assert!(best_plain < 0.75, "without the norm it reached {best_plain:.4}");
}
