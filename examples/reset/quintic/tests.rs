//! The claims this example rests on, measured.
//!
//! 1. A **hand-built one-layer `Rotor4D`** block solves `A₅` exactly at
//!    `reset-swap`'s width (`d_model = 2`, two heads) — every weight in closed
//!    form. Conjugation (`p = q`) is `SO(3)`, `d` a half-turn about an edge axis
//!    of the icosahedron and `t` a third-turn about a face axis, and the two heads
//!    read two coordinates of the sixty orbit points of one written vector.
//! 2. A **hand-built two-layer** stack of the same block solves `S₅` exactly at
//!    `d_model = 3`. The first layer holds the sign `e` and hands it over on `c`;
//!    the second holds the even part `a` of `σ = sᵉ ∘ a`, turning on `c` by `c` or
//!    by `s∘c∘s` according to that sign, and tracks the sign again in a head of
//!    its own.
//! 3. The **left-isoclinic twin** of (1) carries the double cover `2I` (order
//!    120), whose two lifts of an element are antipodal: its outputs cancel per
//!    class and its head fails — `reset-swap`'s wall, at sixty classes.
//! 4. Why `S₅` takes the second layer — the facts the obstruction is made of:
//!    `A₅` is simple and perfect, `S₅`'s only proper quotient is its sign, and an
//!    odd element carries the `72°` rotation `c` to the `144°` rotation `s∘c∘s`
//!    (conjugate in `S₅`, not in `A₅`) while conjugation by rotations preserves
//!    every rotation angle.
//! 5. **No order-blind model can** — the ceilings from the symbol, the sign, and
//!    the `(#a, #b)` counts, fitted on one split and scored on another.

use crate::common::model::ModelConfigExt;
use crate::dataset::{
    DOUBLE_SWAP, EDGE_AXIS, EVAL_LENGTHS, FIVE_CYCLE, Family, Group, IDENTITY, NUM_SYMBOLS, Perm,
    QuinticDataset, REF_POINT, RESET, SEQ_LENGTH, SWAP, THREE_CYCLE, TURN_A, TURN_B, compose,
    counts_since_reset, face_axis, icosahedral_lifts, inverse, is_even, labels, lift, one_hot,
    orbit_point, parity_split, quat_conj, quat_mul, rotation_angle, rotation_vector,
};
use crate::model::{floor_width, model_config_with};
use burn::data::dataset::Dataset;
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

// ---------------------------------------------------------------------------
// the constructions' constants
// ---------------------------------------------------------------------------

/// `Δ` for every head and every input, so a step turns by `range·π·tanh(‖ϑ‖)`
/// outright and `γ = Δ = 1` writes `B` unscaled.
const DELTA: f64 = 1.0;
/// `Â` on a hold. `A = −softplus(Â)`, floored at the block's `a_floor`.
const A_HOLD_RAW: f64 = -20.0;
/// `−A` on a wipe: `ᾱ = e⁻²⁰` erases what the state held.
const A_WIPE: f64 = 20.0;
/// `x` on a write (`x` takes no activation).
const X_WRITE: f64 = 1.0;
/// The gate `z` where a head speaks — positive, so it never flips a sign.
const Z_ON: f64 = 5.0;
/// Class-logit gain.
const OUT_GAIN: f64 = 3.0;
/// The rotation bound `Mamba3Config::rotation_range`, in half-turns per unit `Δ`.
const ROTATION_RANGE: f64 = 2.0;
/// `state_rank` — one quaternion block.
const RANK: usize = 4;

// ---------------------------------------------------------------------------
// scalar helpers
// ---------------------------------------------------------------------------

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

/// Solve the square system `M·w = rhs` by Gaussian elimination with partial
/// pivoting.
fn solve(mut m: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&a, &b| m[a][col].abs().partial_cmp(&m[b][col].abs()).unwrap())
            .unwrap();
        m.swap(col, piv);
        rhs.swap(col, piv);
        assert!(m[col][col].abs() > 1e-9, "the input points are affinely dependent");
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = m[row][col] / m[col][col];
            let pivot = m[col].clone();
            for (k, entry) in m[row].iter_mut().enumerate().skip(col) {
                *entry -= f * pivot[k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    (0..n).map(|i| rhs[i] / m[i][i]).collect()
}

/// The raw in-projection generator that makes one step turn by the rotation
/// vector `v` (`angle · axis`): the block maps a raw `r` to
/// `range·π·tanh(‖r‖)·r̂`, so `r = atanh(angle/(range·π))·v̂`.
fn raw_generator(v: [f64; 3]) -> [f64; 3] {
    let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if angle < 1e-12 {
        return [0.0; 3];
    }
    let scale = (angle / (ROTATION_RANGE * std::f64::consts::PI)).atanh() / angle;
    v.map(|c| c * scale)
}

/// The per-head rotation channels for one input: `nheads` copies of the raw
/// generator of `v`, laid out `[head][left | right][x, y, z]` for `Rotor4D`
/// (both factors equal: conjugation) and `[head][x, y, z]` for `Quaternion4D`
/// (the left factor alone).
fn rotation_values(rotation: RotationKind, v: [f64; 3], nheads: usize) -> Vec<f64> {
    let g = raw_generator(v);
    let per_head: Vec<f64> = match rotation {
        RotationKind::Rotor4D => [g, g].concat(),
        RotationKind::Quaternion4D => g.to_vec(),
        other => unreachable!("{other:?} is not hand-built here"),
    };
    (0..nheads).flat_map(|_| per_head.clone()).collect()
}

/// `B` / `C` as the block sees them: QK-normed (`γ = 1`).
fn qk_norm(v: [f64; RANK]) -> [f64; RANK] {
    let rms = (v.iter().map(|x| x * x).sum::<f64>() / RANK as f64).sqrt();
    v.map(|x| x / rms)
}

// ---------------------------------------------------------------------------
// the hand-built models
// ---------------------------------------------------------------------------

/// What the network's head reads out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    /// The nearest-point decoder over the group's elements.
    Decoder,
    /// The last block's outputs, verbatim, in the first `d_model` logits.
    Probe,
}

/// One in-projection channel: its value at each of the block's input points,
/// before the channel's own activation.
type Channel = Vec<f64>;

/// Write a block's in-projection so each channel takes its listed values at the
/// listed input points — one exact affine solve per channel.
///
/// `points` are the block's inputs *after* the layer's pre-`RmsNorm` (`γ = 1`),
/// and there are exactly `d_model + 1` of them, so each channel is a square
/// solve; the solve refuses affinely dependent points.
fn write_in_proj(block: &mut Mamba3, points: &[Vec<f64>], channels: &[Channel], device: &Device) {
    let d = points[0].len();
    assert_eq!(points.len(), d + 1, "one affine solve per channel needs d_model + 1 points");
    let rows: Vec<Vec<f64>> = points
        .iter()
        .map(|p| p.iter().copied().chain([1.0]).collect())
        .collect();
    let n_ch = channels.len();
    let mut w = vec![0.0f64; d * n_ch];
    let mut b = vec![0.0f64; n_ch];
    for (ch, values) in channels.iter().enumerate() {
        assert_eq!(values.len(), points.len());
        let sol = solve(rows.clone(), values.clone());
        for k in 0..d {
            w[k * n_ch + ch] = sol[k]; // weight is [d_model, out]
        }
        b[ch] = sol[d];
    }
    assert_eq!(block.in_proj.weight.dims(), [d, n_ch], "channel count mismatch");
    block.in_proj.weight = Param::from_tensor(t1(&w, [d, n_ch], device));
    block.in_proj.bias = Some(Param::from_tensor(t1(&b, [n_ch], device)));
}

/// Everything in a block that is not its in-projection: `Δ`'s bias and `D` at
/// zero, the QK-norm scales at one, no `B` bias, `C` aimed by head at the state
/// component `aim[h]` (`C_raw = (1,0,0,0)` normalises to `(2,0,0,0)`; the bias
/// moves it to `2·e_aim`), and an identity out-projection.
fn write_block_rest(block: &mut Mamba3, aim: &[usize], device: &Device) {
    let nheads = aim.len();
    block.dt_bias_h = Param::from_tensor(Tensor::zeros(Shape::new([nheads]), device));
    block.d_h = Param::from_tensor(Tensor::zeros(Shape::new([nheads]), device));
    block.b_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([RANK]), device));
    block.c_norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([RANK]), device));
    block.b_bias_hmr = Param::from_tensor(Tensor::zeros(Shape::new([nheads, 1, RANK]), device));
    let c_bias: Vec<f64> = aim
        .iter()
        .flat_map(|&k| (0..RANK).map(move |r| 2.0 * (f64::from(r == k) - f64::from(r == 0))))
        .collect();
    block.c_bias_hmr = Param::from_tensor(t1(&c_bias, [nheads, 1, RANK], device));
    identity_out_proj(block, nheads, device);
}

fn identity_out_proj(block: &mut Mamba3, n: usize, device: &Device) {
    let eye: Vec<f64> = (0..n * n).map(|k| f64::from(k / n == k % n)).collect();
    block.out_proj.weight = Param::from_tensor(t1(&eye, [n, n], device));
    block.out_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([n]), device)));
}

/// The network's input embedding (one row per symbol, each of norm `√d_model`
/// so the layer's pre-`RmsNorm` passes it unchanged) and its class head: the
/// direction decoder over `points[class]`, or the probe.
///
/// The decoder's row for a class is its point's **unit** vector and there is no
/// bias, so `argmax ⟨y, p̂⟩` picks the class whose direction is closest to `y`'s:
/// exact whenever the points' directions are distinct, and blind to the state's
/// scale — which drifts, since every hold decays by `a_floor`.
fn write_ends(
    net: &mut burn_stack::modules::LatentNetwork<Mamba3>,
    embed: &[Vec<f64>],
    points: &[Vec<f64>],
    head: Head,
    device: &Device,
) {
    let d = embed[0].len();
    let classes = points.len();
    net.in_proj.weight = Param::from_tensor(t1(&embed.concat(), [NUM_SYMBOLS, d], device));
    net.in_proj.bias = Some(Param::from_tensor(Tensor::zeros(Shape::new([d]), device)));
    let mut w_out = vec![0.0f64; d * classes];
    let b_out = vec![0.0f64; classes];
    match head {
        Head::Decoder => {
            for (class, p) in points.iter().enumerate() {
                let norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
                for (k, pk) in p.iter().enumerate() {
                    w_out[k * classes + class] = OUT_GAIN * pk / norm;
                }
            }
        }
        Head::Probe => {
            for k in 0..d {
                w_out[k * classes + k] = 1.0;
            }
        }
    }
    net.out_proj.weight = Param::from_tensor(t1(&w_out, [d, classes], device));
    net.out_proj.bias = Some(Param::from_tensor(t1(&b_out, [classes], device)));
}

/// What a head aimed at a state component emits per unit of that component of
/// `W·REF_POINT·W*`: `C = 2·e_k`, `B` is [`REF_POINT`] QK-normed, and the write
/// `x` is gated by `silu(z)`.
fn head_gain() -> f64 {
    2.0 * X_WRITE * silu(Z_ON) * qk_norm(REF_POINT)[1] / REF_POINT[1]
}

/// The three symbols as `reset-swap`'s planar tokens, rescaled to norm `√d` (so
/// the pre-norm passes them unchanged) and padded with zeros to `d`: three
/// affinely independent points, spanning the first two axes.
fn embeddings(d: usize) -> Vec<Vec<f64>> {
    let r = (d as f64 / 2.0).sqrt();
    let planar = [[std::f64::consts::SQRT_2, 0.0], [0.0, std::f64::consts::SQRT_2], [-1.0, -1.0]];
    planar
        .iter()
        .map(|p| (0..d).map(|k| if k < 2 { r * p[k] } else { 0.0 }).collect())
        .collect()
}

/// The one-layer `A₅` block, by hand, at `reset-swap`'s width (`d_model = 2`,
/// two heads), for `Rotor4D` (the solution) or `Quaternion4D` (its
/// left-isoclinic twin: the same generators, left factor only).
///
/// `R` writes [`REF_POINT`] and wipes; `d` and `t` write nothing and turn — a
/// half-turn about [`EDGE_AXIS`] and a third-turn about [`face_axis`]. The two
/// heads read the orbit's `x` and `y`, where its sixty points stay distinct, and
/// the head is the nearest-point decoder over them.
fn handmade_a5(device: &Device, rotation: RotationKind, head: Head) -> MambaLatentNet {
    const H: usize = 2;
    let group = Group::Alternating;
    let (d, nheads) = floor_width(group);
    assert_eq!((d, nheads), (H, H), "one head per readout, per_head_dim = 1");
    let cfg = model_config_with(group, rotation, 1, d, 1, H, 1);
    let mut model = ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("reset-quintic configures the Mamba-3 variant")
    };
    let embed = embeddings(d);

    let layer = &mut net.layers.real_layers[0];
    layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([d]), device));
    let block = &mut layer.block;

    // values at (TURN_A = d, TURN_B = t, RESET)
    let mut channels: Vec<Channel> = Vec::new();
    channels.extend((0..H).map(|_| vec![Z_ON; NUM_SYMBOLS])); // z
    channels.extend((0..H).map(|_| vec![0.0, 0.0, X_WRITE])); // x — only R writes
    channels.extend((0..RANK).map(|r| vec![REF_POINT[r]; NUM_SYMBOLS])); // B_raw
    channels.extend((0..RANK).map(|r| vec![f64::from(r == 0); NUM_SYMBOLS])); // C_raw
    channels.extend((0..H).map(|_| vec![softplus_inv(DELTA); NUM_SYMBOLS])); // Δ
    channels.extend((0..H).map(|_| vec![A_HOLD_RAW, A_HOLD_RAW, softplus_inv(A_WIPE)])); // A
    let pi = std::f64::consts::PI;
    let turn_d = rotation_values(rotation, EDGE_AXIS.map(|a| pi * a), H);
    let turn_t = rotation_values(rotation, face_axis().map(|a| 2.0 * pi / 3.0 * a), H);
    channels.extend((0..turn_d.len()).map(|n| vec![turn_d[n], turn_t[n], 0.0]));
    write_in_proj(block, &embed, &channels, device);
    write_block_rest(block, &[1, 2], device);

    let decoder: Vec<Vec<f64>> = group
        .elements()
        .iter()
        .map(|&g| orbit_point(g)[..H].iter().map(|v| head_gain() * v).collect())
        .collect();
    write_ends(net, &embed, &decoder, head, device);
    model
}

/// The two-layer `S₅` stack, by hand (`Rotor4D`, `d_model = 3`, three heads).
///
/// **Layer 1 — the sign, handed over on `c`.** `s` is a half-turn about `x̂`,
/// `c` does not turn, `R` writes `ŷ` and wipes; head 0 reads `±ŷ`, gated open
/// on `c` alone (`silu(0) = 0` shuts it elsewhere), and the out-projection puts
/// it on the third stream axis. `c` itself embeds as the zero vector, so after
/// the residual and the second layer's pre-norm its four inputs are
///
/// ```text
///   s   √3·e₀        c, sign ±   ±√3·e₂        R   √3·e₁
/// ```
///
/// — affinely independent in `ℝ³`, which is all its in-projection needs, and
/// exactly these at every length: the sign arrives scaled by how far layer 1's
/// hold has decayed (`ᾱ = e^(−a_floor)` a step), and the norm removes the scale.
/// Next to a nonzero `c` embedding it would not, and the turns would drift.
///
/// **Layer 2 — the even part, and a sign of its own.** With `σ = sᵉ ∘ a`, one step
/// is `a ← sᵉ' ∘ g ∘ sᵉ · a` (`e`, `e'` the sign before and after `g`): the
/// identity on `s`, and on `c` either `c` (`e = 0`) or `s∘c∘s` (`e = 1`) — the
/// `72°` and the `144°` rotations, the pair no conjugation relates, which is why
/// the sign must be read *here*. Heads 0–1 hold `a` as the `A₅` block does;
/// head 2 repeats layer 1's half-turn on `s`, so it carries the sign itself and
/// layer 1 never has to deliver it on `s`. The output is `(orbit(a)ₓ, orbit(a)ᵧ,
/// ±REFᵧ)`, a hundred and twenty distinct points.
fn handmade_s5(device: &Device, head: Head) -> MambaLatentNet {
    const H: usize = 3;
    const K1: f64 = 1.5; // the sign's amplitude on the stream
    let group = Group::Symmetric;
    let (d, nheads) = floor_width(group);
    assert_eq!((d, nheads), (H, H), "one head per readout, per_head_dim = 1");
    let rotation = RotationKind::Rotor4D;
    let cfg = model_config_with(group, rotation, 2, d, 1, H, 1);
    let mut model = ModelConfigExt::init(&cfg, device);
    let MambaLatentNet::Mamba3(net) = &mut model else {
        unreachable!("reset-quintic configures the Mamba-3 variant")
    };
    let pi = std::f64::consts::PI;
    let half_turn = [pi, 0.0, 0.0];
    let r3 = 3f64.sqrt();
    let axis = |k: usize, scale: f64| -> Vec<f64> {
        (0..d).map(|i| scale * f64::from(i == k)).collect()
    };
    // s, c, R — c at the origin
    let embed = vec![axis(0, r3), vec![0.0; d], axis(1, r3)];

    // ── layer 1: the sign ─────────────────────────────────────────────────────
    // a fourth point off the symbols' plane; no channel is evaluated there
    let mut points1 = embed.clone();
    points1.push(axis(2, r3));
    {
        let layer = &mut net.layers.real_layers[0];
        layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([d]), device));
        let block = &mut layer.block;
        // values at (s, c, R, off-plane)
        let v = |s: f64, c: f64, r: f64| -> Channel { vec![s, c, r, 0.0] };
        let mut channels: Vec<Channel> = Vec::new();
        channels.push(v(0.0, Z_ON, 0.0)); // z, head 0: speaks on c alone
        channels.extend((1..H).map(|_| v(0.0, 0.0, 0.0))); // z, heads 1–2: never
        channels.extend((0..H).map(|_| v(0.0, 0.0, X_WRITE))); // x
        channels.extend((0..RANK).map(|r| {
            let y = f64::from(r == 2);
            v(y, y, y) // B = ŷ
        }));
        channels.extend((0..RANK).map(|r| {
            let w = f64::from(r == 0);
            v(w, w, w) // C_raw
        }));
        let dt = softplus_inv(DELTA);
        channels.extend((0..H).map(|_| v(dt, dt, dt))); // Δ
        channels.extend((0..H).map(|_| v(A_HOLD_RAW, A_HOLD_RAW, softplus_inv(A_WIPE))));
        channels.extend(rotation_values(rotation, half_turn, H).iter().map(|&g| v(g, 0.0, 0.0)));
        write_in_proj(block, &points1, &channels, device);
        write_block_rest(block, &[2; H], device);
        // head 0 emits 2·(±2)·x·silu(z) → ±K1 on the third stream axis
        let mut w = vec![0.0f64; H * d];
        w[2] = K1 / (4.0 * X_WRITE * silu(Z_ON)); // [head 0][e₂]
        block.out_proj.weight = Param::from_tensor(t1(&w, [H, d], device));
    }

    // ── layer 2: the even part, and the sign ─────────────────────────────────
    // its inputs after the pre-norm, in order s, (c,+), (c,−), R
    let points2 = vec![embed[TURN_A].clone(), axis(2, r3), axis(2, -r3), embed[RESET].clone()];
    {
        let layer = &mut net.layers.real_layers[1];
        layer.norm.gamma = Param::from_tensor(Tensor::ones(Shape::new([d]), device));
        let block = &mut layer.block;
        let mut channels: Vec<Channel> = Vec::new();
        channels.extend((0..H).map(|_| vec![Z_ON; 4])); // z
        channels.extend((0..H).map(|_| vec![0.0, 0.0, 0.0, X_WRITE])); // x — only R writes
        channels.extend((0..RANK).map(|r| vec![REF_POINT[r]; 4])); // B_raw
        channels.extend((0..RANK).map(|r| vec![f64::from(r == 0); 4])); // C_raw
        channels.extend((0..H).map(|_| vec![softplus_inv(DELTA); 4])); // Δ
        let wipe = softplus_inv(A_WIPE);
        channels.extend((0..H).map(|_| vec![A_HOLD_RAW, A_HOLD_RAW, A_HOLD_RAW, wipe])); // A
        let scs = compose(SWAP, compose(FIVE_CYCLE, SWAP));
        let even = raw_generator(rotation_vector(lift(FIVE_CYCLE)));
        let twisted = raw_generator(rotation_vector(lift(scs)));
        let flip = raw_generator(half_turn);
        for h in 0..H {
            for _factor in 0..2 {
                for k in 0..3 {
                    channels.push(match h {
                        0 | 1 => vec![0.0, even[k], twisted[k], 0.0], // a
                        _ => vec![flip[k], 0.0, 0.0, 0.0],            // the sign
                    });
                }
            }
        }
        write_in_proj(block, &points2, &channels, device);
        write_block_rest(block, &[1, 2, 2], device);
    }

    // ── the head: directions of (orbit(a)ₓ, orbit(a)ᵧ, sign·REFᵧ) ──────────────
    let decoder: Vec<Vec<f64>> = group
        .elements()
        .iter()
        .map(|&sigma| {
            let (parity, a) = parity_split(sigma);
            let o = orbit_point(a);
            let sign = if parity == 0 { 1.0 } else { -1.0 };
            vec![o[0], o[1], sign * REF_POINT[2]].iter().map(|v| head_gain() * v).collect()
        })
        .collect();
    write_ends(net, &embed, &decoder, head, device);
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

/// Run `model` over `count` sequences of `len` symbols from one family; return
/// the per-position output channels and the targets.
fn run(
    model: &MambaLatentNet,
    group: Group,
    family: Family,
    (count, len): (usize, usize),
    seed: u64,
    device: &Device,
) -> (Vec<Vec<f64>>, Vec<i64>) {
    let items: Vec<_> = QuinticDataset::new(group, count, len, family, seed)
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
    let classes = group.num_classes();
    let flat = out
        .reshape([count * len, classes])
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    let channels = flat
        .chunks_exact(classes)
        .map(|c| c.iter().map(|&v| v as f64).collect())
        .collect();
    let targets = items.iter().flat_map(|i| i.targets.clone()).collect();
    (channels, targets)
}

/// Per-position accuracy of the model's own head (argmax over the class logits)
/// on 128 sequences of `len` symbols.
fn accuracy(
    model: &MambaLatentNet,
    group: Group,
    family: Family,
    len: usize,
    device: &Device,
) -> f64 {
    let (channels, targets) = run(model, group, family, (128, len), EVAL, device);
    let hits = channels
        .iter()
        .zip(&targets)
        .filter(|(logits, t)| {
            let pred = (0..logits.len())
                .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap())
                .unwrap();
            pred as i64 == **t
        })
        .count();
    hits as f64 / targets.len() as f64
}

/// Accuracy of the best lookup table from a discrete code to a class — fitted
/// on one split, scored on another. Codes unseen while fitting fall back to the
/// fit split's majority class.
fn best_lookup(
    fit: (&[usize], &[i64]),
    eval: (&[usize], &[i64]),
    num_codes: usize,
    num_classes: usize,
) -> f64 {
    let mut tally = vec![vec![0u64; num_classes]; num_codes];
    let mut overall = vec![0u64; num_classes];
    for (&c, &t) in fit.0.iter().zip(fit.1) {
        tally[c][t as usize] += 1;
        overall[t as usize] += 1;
    }
    let argmax = |row: &[u64]| (0..num_classes).max_by_key(|&c| row[c]).unwrap() as i64;
    let fallback = argmax(&overall);
    let table: Vec<i64> = tally
        .iter()
        .map(|row| match row.iter().sum::<u64>() {
            0 => fallback,
            _ => argmax(row),
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

/// Per class, `‖mean output‖ / rms output`, averaged over classes: 1 when every
/// position of a class lands on one point, ~0 when on `±` a point equally often.
fn mean_over_rms(channels: &[Vec<f64>], targets: &[i64], width: usize, classes: usize) -> f64 {
    let mut sum = vec![vec![0.0f64; width]; classes];
    let mut sq = vec![0.0f64; classes];
    let mut n = vec![0.0f64; classes];
    for (o, &t) in channels.iter().zip(targets) {
        let c = t as usize;
        n[c] += 1.0;
        for r in 0..width {
            sum[c][r] += o[r];
            sq[c] += o[r] * o[r];
        }
    }
    let (mut total, mut seen) = (0.0, 0.0);
    for c in 0..classes {
        if n[c] == 0.0 {
            continue;
        }
        let mean = sum[c].iter().map(|v| v * v / (n[c] * n[c])).sum::<f64>().sqrt();
        total += mean / (sq[c] / n[c]).sqrt().max(1e-12);
        seen += 1.0;
    }
    total / seen
}

// ---------------------------------------------------------------------------
// 1–2. the hand-built solutions
// ---------------------------------------------------------------------------

/// Accuracy of `model` on every family at every [`EVAL_LENGTHS`] length,
/// printed as a table; returns the worst.
fn exact_everywhere(model: &MambaLatentNet, group: Group, device: &Device) -> f64 {
    let mut worst = 1.0f64;
    println!("      family    {}", EVAL_LENGTHS.map(|l| format!("{l:>4} symbols")).join("   "));
    for (name, family) in FAMILIES {
        let accs = EVAL_LENGTHS.map(|len| accuracy(model, group, family, len, device));
        println!("  {name:<9}   {}", accs.map(|a| format!("{:9.2}%", 100.0 * a)).join("    "));
        worst = accs.iter().copied().fold(worst, f64::min);
    }
    worst
}

/// `A₅` in one `Rotor4D` block: every weight in closed form, no training.
#[test]
fn handmade_a5_rotor_solves_every_family() {
    let device = Device::default();
    let model = handmade_a5(&device, RotationKind::Rotor4D, Head::Decoder);
    println!("hand-built one-layer SO(4) block on A₅ ({} params):", model.num_params());
    let worst = exact_everywhere(&model, Group::Alternating, &device);
    assert!(worst == 1.0, "hand-built A₅ solution is not exact: {worst}");
}

/// `S₅` in two `Rotor4D` layers: every weight in closed form, no training.
#[test]
fn handmade_s5_two_layers_solve_every_family() {
    let device = Device::default();
    let model = handmade_s5(&device, Head::Decoder);
    println!("hand-built two-layer SO(4) stack on S₅ ({} params):", model.num_params());
    let worst = exact_everywhere(&model, Group::Symmetric, &device);
    assert!(worst == 1.0, "hand-built S₅ solution is not exact: {worst}");
}

// ---------------------------------------------------------------------------
// 3. the left-isoclinic twin
// ---------------------------------------------------------------------------

/// `Quaternion4D` — the same `A₅` construction with the left factor alone.
///
/// Its state runs in the binary icosahedral group `2I`: `d`'s lift squares to
/// `−1`, so every element of `A₅` is reached as both `±W`, and the heads read
/// `±(W ⊗ B)`. The two lifts occur about equally often and a linear functional
/// takes opposite values on them, so each class's outputs cancel on average —
/// `‖mean‖/rms` near 0 where the conjugating block sits at 1 — and the head that
/// is exact for the conjugating block is near chance here.
#[test]
fn left_isoclinic_carries_the_binary_icosahedral_group() {
    let device = Device::default();
    let group = Group::Alternating;
    let classes = group.num_classes();
    let twin = handmade_a5(&device, RotationKind::Quaternion4D, Head::Decoder);
    println!("the same construction, left-isoclinic ({} params):", twin.num_params());
    println!("      family    same head   ‖mean‖/rms   (SO(4): same head / ‖mean‖/rms)");
    let (mut best_head, mut worst_cancel) = (0.0f64, 0.0f64);
    let rotor = handmade_a5(&device, RotationKind::Rotor4D, Head::Decoder);
    let shape = (128, SEQ_LENGTH);
    for (name, family) in FAMILIES {
        let head = accuracy(&twin, group, family, SEQ_LENGTH, &device);
        let twin_probe = handmade_a5(&device, RotationKind::Quaternion4D, Head::Probe);
        let (ch, t) = run(&twin_probe, group, family, shape, EVAL, &device);
        let cancel = mean_over_rms(&ch, &t, 2, classes);
        let rotor_probe = handmade_a5(&device, RotationKind::Rotor4D, Head::Probe);
        let (rch, rt) = run(&rotor_probe, group, family, shape, EVAL, &device);
        println!(
            "  {name:<9}     {:6.2}%      {cancel:5.3}        {:6.2}% / {:5.3}",
            100.0 * head,
            100.0 * accuracy(&rotor, group, family, SEQ_LENGTH, &device),
            mean_over_rms(&rch, &rt, 2, classes),
        );
        best_head = best_head.max(head);
        worst_cancel = worst_cancel.max(cancel);
    }
    assert!(best_head < 0.6, "the left-isoclinic twin reached {best_head:.4}");
    assert!(worst_cancel < 0.3, "the two lifts did not cancel ({worst_cancel:.3})");
}

// ---------------------------------------------------------------------------
// 4. the groups, and the obstruction
// ---------------------------------------------------------------------------

/// The dataset's labels are the two word problems, generated as claimed.
#[test]
fn labels_are_the_alternating_and_symmetric_groups() {
    let a5 = Group::Alternating;
    let s5 = Group::Symmetric;
    assert_eq!(a5.elements().len(), 60);
    assert_eq!(s5.elements().len(), 120);
    assert!(a5.elements().iter().all(|&p| is_even(p)));

    // A₅: d² = t³ = (d∘t)⁵ = 1 — the (2,3,5) triangle group, which *is* A₅
    let last = |group: Group, word: &[usize]| {
        let mut seq = vec![RESET];
        seq.extend_from_slice(word);
        *labels(group, &seq).last().unwrap()
    };
    assert_eq!(last(a5, &[TURN_A, TURN_A]), 0, "d² = 1");
    assert_eq!(last(a5, &[TURN_B; 3]), 0, "t³ = 1");
    assert_ne!(last(a5, &[TURN_B, TURN_A]), 0);
    assert_eq!(last(a5, &[TURN_B, TURN_A].repeat(5)), 0, "(d∘t)⁵ = 1");
    // S₅: s² = c⁵ = 1, and s∘c ≠ c∘s
    assert_eq!(last(s5, &[TURN_A, TURN_A]), 0, "s² = 1");
    assert_eq!(last(s5, &[TURN_B; 5]), 0, "c⁵ = 1");
    assert_ne!(last(s5, &[TURN_B, TURN_A]), last(s5, &[TURN_A, TURN_B]));

    // the streams `examples/reset/README.md` shows
    let (s, c, d, t, r) = (TURN_A, TURN_B, TURN_A, TURN_B, RESET);
    assert_eq!(labels(s5, &[r, s, c, s, c, c, r, c, s]), [0, 24, 57, 51, 82, 108, 0, 33, 9]);
    assert_eq!(labels(a5, &[r, d, t, d, t, t, r, t, d]), [0, 13, 16, 7, 32, 57, 0, 29, 38]);

    for group in [a5, s5] {
        let classes = group.num_classes();
        let mut seen = vec![false; classes];
        for item in QuinticDataset::new(group, 1024, SEQ_LENGTH, Family::Mixed, 7).iter() {
            let item = item.expect("dataset item");
            for (&s, &c) in item.symbols.iter().zip(&item.targets) {
                seen[c as usize] = true;
                if s == RESET {
                    assert_eq!(c, 0, "a reset lands on the identity");
                }
            }
        }
        assert!(seen.iter().all(|s| *s), "{group:?}: some element never occurs");
    }
    assert_eq!(a5.symbol_perm(TURN_A), DOUBLE_SWAP);
    assert_eq!(a5.symbol_perm(TURN_B), THREE_CYCLE);
}

/// The icosahedral lifts are a homomorphism up to sign, and the orbit of
/// [`REF_POINT`] is sixty points on one sphere whose `xy`-shadows point in sixty
/// distinct directions — what makes the two-head `A₅` readout a decoder.
#[test]
fn a5_is_the_icosahedral_rotation_group() {
    let elements = Group::Alternating.elements();
    let lifts = icosahedral_lifts();
    for (i, &g) in elements.iter().enumerate() {
        for (j, &h) in elements.iter().enumerate() {
            let gh = Group::Alternating.class_of(compose(g, h)) as usize;
            let prod = quat_mul(lifts[i], lifts[j]);
            let same = |s: f64| prod.iter().zip(&lifts[gh]).all(|(a, b)| (a - s * b).abs() < 1e-9);
            assert!(same(1.0) || same(-1.0), "lift(g∘h) ≠ ±lift(g)⊗lift(h)");
        }
    }
    let radius = REF_POINT.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut directions: Vec<f64> = elements
        .iter()
        .map(|&g| {
            let p = orbit_point(g);
            let norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - radius).abs() < 1e-9, "the orbit is not on one sphere");
            p[1].atan2(p[0]).to_degrees()
        })
        .collect();
    directions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let wrap = directions[0] + 360.0 - directions[directions.len() - 1];
    let gap = directions.windows(2).map(|w| w[1] - w[0]).fold(wrap, f64::min);
    println!("closest pair of xy-directions: {gap:.2}°");
    assert!(gap > 1.0, "two orbit points share a direction on the readout plane ({gap}°)");

    // the generators turn by what they should, about the axes they should
    let angles = [DOUBLE_SWAP, THREE_CYCLE, compose(DOUBLE_SWAP, THREE_CYCLE)].map(rotation_angle);
    println!("d, t, d∘t turn by {angles:.1?} degrees");
    for (got, want) in angles.iter().zip([180.0, 120.0, 72.0]) {
        assert!((got - want).abs() < 1e-6);
    }
}

/// Why `S₅` needs the second layer, as four checkable facts.
///
/// A layer's transition is a scalar times a block-diagonal rotation, so what it
/// can follow of a group *exactly* is a quotient of a group of such rotations.
///
/// 1. `A₅` is **simple** and **perfect**: an abelian transition follows none of
///    it, and it cannot be split across a stack of solvable ones.
/// 2. `S₅`'s normal subgroups are `1`, `A₅`, `S₅`, so its only proper quotient is
///    the sign: a layer either follows all of `S₅` or at most its sign.
/// 3. An odd element carries the five-cycle `c` to `s∘c∘s`, which is conjugate
///    to `c` in `S₅` but **not** in `A₅` — the outer automorphism — and in the
///    icosahedral rotations the two turn by `72°` and `144°`.
/// 4. Conjugation by a rotation never changes a rotation angle: in a 4-block
///    `(q, p)` conjugated by `(u, w)` keeps both `Re q` and `Re p`. So the `A₅`
///    inside any group of block rotations meets only inner automorphisms, and a
///    transposition has nowhere to act — it would take the reflection that swaps
///    `q` and `p`, which no Mamba-3 transition is.
#[test]
fn s5_needs_a_reflection_one_layer_does_not_have() {
    let a5 = Group::Alternating.elements();
    let s5 = Group::Symmetric.elements();

    // 1. A₅ simple and perfect
    let normals_a5 = normal_subgroup_orders(&a5);
    assert_eq!(normals_a5, vec![1, 60], "A₅ should be simple");
    let commutators: Vec<Perm> = a5
        .iter()
        .flat_map(|&g| {
            a5.iter()
                .map(move |&h| compose(compose(g, h), compose(inverse(g), inverse(h))))
        })
        .collect();
    assert_eq!(closure(&commutators).len(), 60, "A₅ should be perfect");

    // 2. S₅'s normal subgroups
    assert_eq!(normal_subgroup_orders(&s5), vec![1, 60, 120]);

    // 3. the outer automorphism moves a fifth-turn to two fifths
    let c = FIVE_CYCLE;
    let scs = compose(SWAP, compose(c, SWAP));
    let conj_in = |group: &[Perm]| group.iter().any(|&g| compose(g, compose(c, inverse(g))) == scs);
    assert!(conj_in(&s5) && !conj_in(&a5), "c ~ s∘c∘s in S₅ only");
    let (ac, ascs) = (rotation_angle(c), rotation_angle(scs));
    println!("c turns by {ac:.1}°, s∘c∘s by {ascs:.1}°");
    assert!((ac - 72.0).abs() < 1e-6 && (ascs - 144.0).abs() < 1e-6);

    // 4. two-sided conjugation keeps both factors' real parts
    let (q, p) = (lift(c), lift(scs));
    for (u, w) in [(lift(THREE_CYCLE), lift(DOUBLE_SWAP)), (lift(scs), lift(IDENTITY))] {
        let q2 = quat_mul(quat_mul(u, q), quat_conj(u));
        let p2 = quat_mul(quat_mul(w, p), quat_conj(w));
        assert!((q2[0] - q[0]).abs() < 1e-12 && (p2[0] - p[0]).abs() < 1e-12);
    }
    assert!((q[0].abs() - p[0].abs()).abs() > 0.4, "the two lifts' real parts should differ");
}

/// Close a set of permutations under composition.
fn closure(generators: &[Perm]) -> Vec<Perm> {
    let mut seen = vec![IDENTITY];
    let mut frontier = vec![IDENTITY];
    while let Some(x) = frontier.pop() {
        for &g in generators {
            let y = compose(g, x);
            if !seen.contains(&y) {
                seen.push(y);
                frontier.push(y);
            }
        }
    }
    seen
}

/// The orders of a group's normal subgroups: unions of conjugacy classes that
/// contain the identity and are closed under composition.
fn normal_subgroup_orders(group: &[Perm]) -> Vec<usize> {
    let mut classes: Vec<Vec<Perm>> = Vec::new();
    for &x in group {
        if classes.iter().any(|c| c.contains(&x)) {
            continue;
        }
        let mut class: Vec<Perm> =
            group.iter().map(|&g| compose(g, compose(x, inverse(g)))).collect();
        class.sort();
        class.dedup();
        classes.push(class);
    }
    let others: Vec<&Vec<Perm>> = classes.iter().filter(|c| !c.contains(&IDENTITY)).collect();
    let mut orders = Vec::new();
    for mask in 0u32..(1 << others.len()) {
        let mut set = vec![IDENTITY];
        for (k, c) in others.iter().enumerate() {
            if mask & (1 << k) != 0 {
                set.extend(c.iter().copied());
            }
        }
        let closed = set
            .iter()
            .all(|&a| set.iter().all(|&b| set.contains(&compose(a, b))));
        if closed {
            orders.push(set.len());
        }
    }
    orders.sort();
    orders.dedup();
    orders
}

// ---------------------------------------------------------------------------
// a tool: what a trained model's first layer turns by
// ---------------------------------------------------------------------------

/// Read a trained `Rotor4D` model's **first** layer off its weights: for every
/// symbol and head, the step `Δ`, the hold `ᾱ = exp(Δ·A)` and the angles the
/// left and right factors turn by; then, per head, the angle between the two
/// turns' axes in each factor. The icosahedral generators read `180°`/`120°`
/// with axes `20.9°` or `69.1°` apart (or their supplements).
///
/// ```text
/// QUINTIC_ARTIFACTS=<artifacts dir> cargo test --release --example reset-quintic \
///     learned_rotations -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn learned_rotations() {
    let dir = std::env::var("QUINTIC_ARTIFACTS").expect("set QUINTIC_ARTIFACTS");
    let dir = std::path::PathBuf::from(dir);
    let device = Device::default();
    let cfg: MambaLatentNetConfig =
        crate::common::cli::load_model_config(&dir.join("model_config.json")).unwrap();
    let model: MambaLatentNet = crate::common::cli::load_model(&dir, &cfg, &device).unwrap();
    let MambaLatentNet::Mamba3(net) = &model else {
        unreachable!("reset-quintic configures the Mamba-3 variant")
    };
    let to_vec = |t: Tensor<1>| -> Vec<f64> {
        t.into_data().try_to_vec::<f32>().unwrap().iter().map(|&v| f64::from(v)).collect()
    };
    let layer = &net.layers.real_layers[0];
    let block = &layer.block;
    let [d, n_ch] = block.in_proj.weight.dims();
    let nheads = block.d_h.dims()[0];
    let w = to_vec(block.in_proj.weight.val().reshape([d * n_ch]));
    let b = to_vec(block.in_proj.bias.as_ref().unwrap().val());
    let dt_bias = to_vec(block.dt_bias_h.val());
    let gamma = to_vec(layer.norm.gamma.val());
    let embed_w = to_vec(net.in_proj.weight.val().reshape([NUM_SYMBOLS * d]));
    let embed_b = to_vec(net.in_proj.bias.as_ref().unwrap().val());
    // [z(d_inner) | x(d_inner) | B(4m) | C(4m) | Δ(h) | A(h) | rotation(6h)]
    let d_inner = block.out_proj.weight.dims()[0];
    let mimo_rank = (n_ch - 2 * d_inner - 8 * nheads) / (2 * RANK);
    let off_dt = 2 * d_inner + 2 * RANK * mimo_rank;
    let (off_a, off_rot) = (off_dt + nheads, off_dt + 2 * nheads);
    let norm = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let angle_between = |u: [f64; 3], v: [f64; 3]| {
        let cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (norm(u) * norm(v)).max(1e-12);
        cos.clamp(-1.0, 1.0).acos().to_degrees()
    };
    println!("first layer: d_model {d}, {nheads} heads, mimo_rank {mimo_rank}");
    // gens[symbol][head][factor] — the rotation vector one step turns by
    let mut gens = vec![vec![[[0.0f64; 3]; 2]; nheads]; NUM_SYMBOLS];
    for s in 0..NUM_SYMBOLS {
        let token: Vec<f64> = (0..d).map(|k| embed_w[s * d + k] + embed_b[k]).collect();
        let rms = (token.iter().map(|v| v * v).sum::<f64>() / d as f64 + 1e-5).sqrt();
        let x: Vec<f64> = (0..d).map(|k| token[k] / rms * gamma[k]).collect();
        let ch: Vec<f64> = (0..n_ch)
            .map(|c| (0..d).map(|k| w[k * n_ch + c] * x[k]).sum::<f64>() + b[c])
            .collect();
        for h in 0..nheads {
            let softplus = |v: f64| v.exp().ln_1p();
            let dt = softplus(ch[off_dt + h] + dt_bias[h]);
            let a = -softplus(ch[off_a + h]).max(1e-4);
            let mut line = format!("  symbol {s} head {h}: Δ {dt:.3}  ᾱ {:.4}", (dt * a).exp());
            for (f, name) in ["left", "right"].iter().enumerate() {
                let r: [f64; 3] = std::array::from_fn(|k| ch[off_rot + 6 * h + 3 * f + k]);
                let angle = dt * ROTATION_RANGE * std::f64::consts::PI * norm(r).tanh();
                gens[s][h][f] = r.map(|v| angle * v / norm(r).max(1e-12));
                line += &format!("  {name} {:7.2}°", angle.to_degrees());
            }
            println!("{line}");
        }
    }
    for h in 0..nheads {
        println!(
            "  head {h}: axes of symbols 0 and 1 are {:.1}° apart (left), {:.1}° (right)",
            angle_between(gens[0][h][0], gens[1][h][0]),
            angle_between(gens[0][h][1], gens[1][h][1]),
        );
    }
}

// ---------------------------------------------------------------------------
// 5. the ceilings
// ---------------------------------------------------------------------------

/// What an order-blind model can do: the best table from the symbol, from the
/// sign `(−1)^#odd` (for `S₅`, everything a layer can follow *homomorphically*),
/// and from the `(#a, #b)` counts since the reset.
#[test]
fn counts_and_sign_ceilings() {
    for group in [Group::Alternating, Group::Symmetric] {
        let classes = group.num_classes();
        println!("{group:?}: order-blind ceilings (chance {:.2}%)", 100.0 / classes as f64);
        println!("      family    memoryless   by sign   by (#a,#b)");
        let mut best = 0.0f64;
        for (name, family) in FAMILIES {
            let (fit_sym, fit_cnt, fit_sign, fit_t) = codes(group, family, 2048, FIT);
            let (eval_sym, eval_cnt, eval_sign, eval_t) = codes(group, family, 512, EVAL);
            let by_counts = best_lookup(
                (&fit_cnt, &fit_t),
                (&eval_cnt, &eval_t),
                SEQ_LENGTH * SEQ_LENGTH,
                classes,
            );
            // `A₅` has no sign: every element is even
            let by_sign = match group {
                Group::Alternating => "     —".to_string(),
                Group::Symmetric => format!(
                    "{:6.2}%",
                    100.0 * best_lookup((&fit_sign, &fit_t), (&eval_sign, &eval_t), 2, classes)
                ),
            };
            println!(
                "  {name:<9}    {:6.2}%     {by_sign}    {:6.2}%",
                100.0 * best_lookup((&fit_sym, &fit_t), (&eval_sym, &eval_t), NUM_SYMBOLS, classes),
                100.0 * by_counts,
            );
            best = best.max(by_counts);
        }
        assert!(best < 0.85, "{group:?}: the counts nearly give the answer ({best:.4})");
    }
}

/// Per position: the symbol, the `(#a, #b)` counts since the reset, the sign.
fn codes(
    group: Group,
    family: Family,
    count: usize,
    seed: u64,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<i64>) {
    let (mut sym, mut cnt, mut sign, mut targets) = (vec![], vec![], vec![], vec![]);
    for item in QuinticDataset::new(group, count, SEQ_LENGTH, family, seed).iter() {
        let item = item.expect("dataset item");
        for ((&s, (a, b)), &t) in item
            .symbols
            .iter()
            .zip(counts_since_reset(&item.symbols))
            .zip(&labels(group, &item.symbols))
        {
            sym.push(s);
            cnt.push(a as usize * SEQ_LENGTH + b as usize);
            let odd = match group {
                Group::Alternating => 0,
                Group::Symmetric => a % 2,
            };
            sign.push(odd as usize);
            targets.push(t);
        }
    }
    (sym, cnt, sign, targets)
}
