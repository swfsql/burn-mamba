//! Modular-addition dataset, generalized to `k` summands: all `pᵏ` sequences
//! `(x₁, …, x_k)` labelled `(Σ xᵢ) mod p`, deterministically split into
//! train/test **by sequence** with a seeded `ChaCha8Rng` (stable across
//! platforms and `rand` versions).
//!
//! `k = 2` is the literature-standard grokking task. `k > 2` is the SSM-native
//! arm: with one layer (and `conv_kernel = 1`) the final position's only
//! access to `x₁…x_{k−1}` is through the recurrent state, so state-side
//! diagnostics become load-bearing by construction. Pick `(p, k)` with
//! `pᵏ` small enough to enumerate (asserted) and, for full-batch training,
//! comparable to the `p = 97, k = 2` case (e.g. `p = 11, k = 4`).

use burn::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

/// Hard cap on `pᵏ` — everything is enumerated in memory.
const ENUMERATION_CAP: usize = 2_000_000;

/// One side of the train/test split: `n` token sequences and their labels.
pub struct Split {
    /// The modulus `p` (= vocab size = number of classes).
    pub p: usize,
    /// Number of summands (sequence length).
    pub k: usize,
    /// Token sequences, row-major flat `[n · k]`.
    pub seqs: Vec<i32>,
    /// Labels `(Σ xᵢ) mod p`, aligned with rows of `seqs`.
    pub labels: Vec<i32>,
}

impl Split {
    fn new(p: usize, k: usize, seqs: Vec<i32>) -> Self {
        assert_eq!(seqs.len() % k, 0);
        let labels = seqs
            .chunks_exact(k)
            .map(|row| row.iter().map(|x| *x as i64).sum::<i64>() as i32 % p as i32)
            .collect();
        Split { p, k, seqs, labels }
    }

    /// Number of examples.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the split holds no examples.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// The first `n` examples as their own split (for sample displays).
    pub fn head(&self, n: usize) -> Split {
        let n = n.min(self.len());
        Split {
            p: self.p,
            k: self.k,
            seqs: self.seqs[..n * self.k].to_vec(),
            labels: self.labels[..n].to_vec(),
        }
    }

    /// Token IDs as an Int tensor `[n, k]`.
    pub fn inputs_tensor(&self, device: &Device) -> Tensor<2, Int> {
        Tensor::<1, Int>::from_ints(self.seqs.as_slice(), device).reshape([self.len(), self.k])
    }

    /// Labels as an Int tensor `[n]`.
    pub fn labels_tensor(&self, device: &Device) -> Tensor<1, Int> {
        Tensor::from_ints(self.labels.as_slice(), device)
    }

    /// One-hot float targets `[n, p]` for the cross-entropy loss. Built as
    /// floats directly (an `Int one_hot → float` round-trip would land on the
    /// plain backend even for an autodiff `device`).
    pub fn targets_tensor(&self, device: &Device) -> Tensor<2> {
        let mut flat = vec![0.0f32; self.len() * self.p];
        for (i, &label) in self.labels.iter().enumerate() {
            flat[i * self.p + label as usize] = 1.0;
        }
        Tensor::<1>::from_floats(flat.as_slice(), device).reshape([self.len(), self.p])
    }
}

/// `pᵏ`, asserting the enumeration cap.
fn space_size(p: usize, k: usize) -> usize {
    let total = p.checked_pow(k as u32).expect("p^k overflows usize");
    assert!(
        total <= ENUMERATION_CAP,
        "p^k = {total} exceeds the enumeration cap ({ENUMERATION_CAP}); pick smaller p or k"
    );
    total
}

/// Materialize sequence `index ∈ [0, pᵏ)` as its `k` base-`p` digits
/// (most-significant first), appended to `out`.
fn push_digits(index: usize, p: usize, k: usize, out: &mut Vec<i32>) {
    let mut rem = index;
    let start = out.len();
    out.resize(start + k, 0);
    for j in (0..k).rev() {
        out[start + j] = (rem % p) as i32;
        rem /= p;
    }
}

/// Enumerate all `pᵏ` sequences, shuffle them with `ChaCha8Rng(split_seed)`,
/// and return `(train, test)` where train takes the first
/// `round(train_fraction·pᵏ)` sequences (the splits are disjoint by sequence).
pub fn build(p: usize, k: usize, train_fraction: f64, split_seed: u64) -> (Split, Split) {
    let total = space_size(p, k);
    let mut indices: Vec<usize> = (0..total).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(split_seed);
    indices.shuffle(&mut rng);

    let n_train = (total as f64 * train_fraction).round() as usize;
    assert!(
        n_train >= 1 && n_train < total,
        "train_fraction {train_fraction} must leave both splits non-empty"
    );
    let mut train_seqs = Vec::with_capacity(n_train * k);
    let mut test_seqs = Vec::with_capacity((total - n_train) * k);
    for (i, &index) in indices.iter().enumerate() {
        let out = if i < n_train { &mut train_seqs } else { &mut test_seqs };
        push_digits(index, p, k, out);
    }
    (Split::new(p, k, train_seqs), Split::new(p, k, test_seqs))
}

/// The diagnostic eval set: all `pᵏ` sequences when they fit in `max_n`
/// (the PR estimator wants everything; `tr Σ̂²` is upward-biased at small
/// sample counts), otherwise a deterministic `ChaCha8Rng(seed)` sample of
/// `max_n` distinct sequences.
pub fn diagnostic_set(p: usize, k: usize, max_n: usize, seed: u64) -> Split {
    let total = space_size(p, k);
    let mut seqs = Vec::with_capacity(total.min(max_n) * k);
    if total <= max_n {
        for index in 0..total {
            push_digits(index, p, k, &mut seqs);
        }
    } else {
        let mut indices: Vec<usize> = (0..total).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        for &index in indices.iter().take(max_n) {
            push_digits(index, p, k, &mut seqs);
        }
    }
    Split::new(p, k, seqs)
}
