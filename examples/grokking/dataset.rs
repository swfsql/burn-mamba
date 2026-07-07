//! Modular-addition dataset: all `p²` pairs `(a, b)` labelled `(a + b) mod p`,
//! deterministically split into train/test **by pair** with a seeded
//! `ChaCha8Rng` (stable across platforms and `rand` versions).

use burn::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

/// One side of the train/test split: `n` token pairs and their labels.
pub struct Split {
    /// The modulus `p` (= vocab size = number of classes).
    pub p: usize,
    /// Token pairs `[a, b]`, one per example.
    pub pairs: Vec<[i32; 2]>,
    /// Labels `(a + b) mod p`, aligned with `pairs`.
    pub labels: Vec<i32>,
}

impl Split {
    fn new(p: usize, pairs: Vec<[i32; 2]>) -> Self {
        let labels = pairs.iter().map(|[a, b]| (a + b) % p as i32).collect();
        Split { p, pairs, labels }
    }

    /// Number of examples.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the split holds no examples.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Token IDs as an Int tensor `[n, 2]`.
    pub fn inputs_tensor(&self, device: &Device) -> Tensor<2, Int> {
        let flat: Vec<i32> = self.pairs.iter().flatten().copied().collect();
        Tensor::<1, Int>::from_ints(flat.as_slice(), device).reshape([self.len(), 2])
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

/// Enumerate all `p²` pairs, shuffle them with `ChaCha8Rng(split_seed)`, and
/// return `(train, test)` where train takes the first
/// `round(train_fraction·p²)` pairs (the splits are disjoint by pair).
pub fn build(p: usize, train_fraction: f64, split_seed: u64) -> (Split, Split) {
    let mut pairs: Vec<[i32; 2]> = (0..p as i32)
        .flat_map(|a| (0..p as i32).map(move |b| [a, b]))
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(split_seed);
    pairs.shuffle(&mut rng);

    let n_train = ((p * p) as f64 * train_fraction).round() as usize;
    assert!(
        n_train >= 1 && n_train < p * p,
        "train_fraction {train_fraction} must leave both splits non-empty"
    );
    let test_pairs = pairs.split_off(n_train);
    (Split::new(p, pairs), Split::new(p, test_pairs))
}
