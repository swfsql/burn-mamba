//! Scalar helpers for the hand-built blocks and for measuring them.
//!
//! The constructions all follow `reset-majority`'s recipe: pick one embedding
//! per symbol on the sphere the layer's pre-`RmsNorm` leaves unchanged, then
//! solve every in-projection channel as an affine functional of that embedding
//! from its target value on each symbol.

use super::data::{Generator, IGNORE, NUM_CLASSES, TallyBatcher, TallyDataset, Task};
use super::training::ssd_path as training_ssd_path;
use burn::data::dataloader::batcher::Batcher;
use burn::module::Param;
use burn::prelude::*;
use burn_mamba::prelude::*;

/// Inverse of `softplus`, stable at both ends: `ln(eᵛ − 1)` underflows for tiny
/// `v` unless written with `exp_m1`, and overflows for large `v`, where
/// `softplus` is the identity to well under f32's resolution.
pub fn softplus_inv(v: f64) -> f64 {
    if v > 30.0 { v } else { v.exp_m1().ln() }
}

/// A tensor from `f64`s.
pub fn t1<const D: usize>(v: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::<1>::from_floats(f.as_slice(), device).reshape(shape)
}

/// A parameter from `f64`s.
pub fn param<const D: usize>(v: &[f64], shape: [usize; D], device: &Device) -> Param<Tensor<D>> {
    Param::from_tensor(t1(v, shape, device))
}

/// Least-squares solve of `rows · w = rhs` (normal equations, Gaussian
/// elimination with partial pivoting), asserting the fit is **exact**: a
/// construction whose channel is not an affine function of the embedding is a
/// bug in the construction, not an approximation to accept.
pub fn solve_affine(rows: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = rows[0].len();
    let mut a = vec![vec![0.0; n]; n];
    let mut y = vec![0.0; n];
    for (row, &r) in rows.iter().zip(rhs) {
        for i in 0..n {
            y[i] += row[i] * r;
            for j in 0..n {
                a[i][j] += row[i] * row[j];
            }
        }
    }
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&p, &q| a[p][col].abs().partial_cmp(&a[q][col].abs()).unwrap())
            .unwrap();
        a.swap(col, piv);
        y.swap(col, piv);
        assert!(a[col][col].abs() > 1e-12, "singular symbol embedding");
        for row in 0..n {
            if row != col {
                let f = a[row][col] / a[col][col];
                let pivot = a[col].clone();
                for (k, entry) in a[row].iter_mut().enumerate().skip(col) {
                    *entry -= f * pivot[k];
                }
                y[row] -= f * y[col];
            }
        }
    }
    let w: Vec<f64> = (0..n).map(|i| y[i] / a[i][i]).collect();
    for (row, &r) in rows.iter().zip(rhs) {
        let fit: f64 = row.iter().zip(&w).map(|(x, w)| x * w).sum();
        let scale = 1.0 + r.abs();
        assert!((fit - r).abs() < 1e-6 * scale, "channel is not affine in the embedding: {fit} vs {r}");
    }
    w
}

/// An affine in-projection `d_model → channels` from per-channel targets on
/// each symbol: `targets[ch][symbol]`. Returns `Linear`'s `(weight, bias)` as
/// `[d_model, channels]` and `[channels]`.
pub fn affine_channels(
    embeddings: &[Vec<f64>],
    targets: &[Vec<f64>],
    device: &Device,
) -> (Param<Tensor<2>>, Param<Tensor<1>>) {
    let d = embeddings[0].len();
    let rows: Vec<Vec<f64>> = embeddings
        .iter()
        .map(|e| e.iter().copied().chain([1.0]).collect())
        .collect();
    let n_ch = targets.len();
    let mut w = vec![0.0; d * n_ch];
    let mut b = vec![0.0; n_ch];
    for (ch, target) in targets.iter().enumerate() {
        assert_eq!(target.len(), embeddings.len(), "one target per symbol");
        let sol = solve_affine(&rows, target);
        for i in 0..d {
            w[i * n_ch + ch] = sol[i];
        }
        b[ch] = sol[d];
    }
    (param(&w, [d, n_ch], device), param(&b, [n_ch], device))
}

/// Per-position predicted classes (row-major over `batch × seq`).
pub fn predictions(model: &MambaLatentNet, inputs: &Tensor<3>) -> Vec<i64> {
    let [batch, seq, _] = inputs.dims();
    let (out, _) = model.forward(inputs.clone(), None, training_ssd_path(), None, None);
    out.reshape([batch * seq, NUM_CLASSES])
        .argmax(1)
        .reshape([batch * seq])
        .into_data()
        .try_to_vec::<i32>()
        .unwrap()
        .into_iter()
        .map(i64::from)
        .collect()
}

/// Accuracy over the scored positions of `count` sequences of one generator.
pub fn accuracy(
    model: &MambaLatentNet,
    task: &Task,
    generator: Generator,
    count: usize,
    seed: u64,
    device: &Device,
) -> f64 {
    let items = TallyDataset::new(task, generator, count, seed).items();
    let inputs = TallyBatcher {
        num_symbols: task.num_symbols,
    }
    .batch(items.clone(), device)
    .inputs;
    let pred = predictions(model, &inputs);
    let want: Vec<i64> = items.iter().flat_map(|i| i.targets.clone()).collect();
    scored_accuracy(&pred, &want)
}

/// Fraction of `pred == want` over the positions `want` scores.
pub fn scored_accuracy(pred: &[i64], want: &[i64]) -> f64 {
    let (hit, all) = pred
        .iter()
        .zip(want)
        .filter(|(_, t)| **t != IGNORE)
        .fold((0usize, 0usize), |(h, a), (p, t)| (h + usize::from(p == t), a + 1));
    hit as f64 / all.max(1) as f64
}

/// The best per-symbol lookup table's accuracy: the memoryless ceiling.
pub fn memoryless_ceiling(task: &Task, generator: Generator, count: usize, seed: u64) -> f64 {
    let mut tally = vec![[0u64; NUM_CLASSES]; task.num_symbols];
    for item in TallyDataset::new(task, generator, count, seed).items() {
        for (&s, &c) in item.symbols.iter().zip(&item.targets) {
            if c != IGNORE {
                tally[s][c as usize] += 1;
            }
        }
    }
    let total: u64 = tally.iter().flatten().sum();
    let best: u64 = tally.iter().map(|row| *row.iter().max().unwrap()).sum();
    best as f64 / total.max(1) as f64
}
