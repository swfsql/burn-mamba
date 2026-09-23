//! The symbol-stream dataset every rung draws from.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, DatasetError, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};

/// Placeholder target for a position whose class is not a function of the
/// history (a reset, a tie, a symbol that fixes the answer by itself). Dropped
/// from the loss and the accuracy.
pub const IGNORE: i64 = -1;
/// Number of output classes on every rung.
pub const NUM_CLASSES: usize = 2;
/// Number of training sequences.
pub const NUM_TRAIN: usize = 4096;
/// Number of evaluation sequences per family.
pub const NUM_EVAL: usize = 512;
/// Dataset RNG seed for the training split.
pub const TRAIN_SEED: u64 = 0xC0FFEE;
/// Dataset RNG seed for the evaluation splits (distinct from training).
pub const EVAL_SEED: u64 = 0xBEEF;

/// A sequence generator.
pub type Generator = fn(&mut Rng, usize) -> Vec<usize>;

/// One rung's task: its alphabet, labels and families.
pub struct Task {
    /// Input alphabet size.
    pub num_symbols: usize,
    /// Length of every generated sequence.
    pub seq_length: usize,
    /// Per-position targets implied by a symbol sequence.
    pub labels: fn(&[usize]) -> Vec<i64>,
    /// The training mixture.
    pub train: Generator,
    /// The evaluation families, reported separately.
    pub families: &'static [(&'static str, Generator)],
    /// One character per symbol, for printing samples.
    pub glyphs: &'static [char],
    /// One character per class, for printing samples.
    pub class_glyphs: [char; NUM_CLASSES],
    /// Class names, for per-class accuracy.
    pub class_names: [&'static str; NUM_CLASSES],
}

/// SplitMix64 — a small deterministic RNG so splits reproduce exactly.
pub struct Rng(pub u64);

impl Rng {
    /// Next raw 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A standard normal (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        let u = self.unit().max(1e-300);
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}

/// One generated sequence and its per-position targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TallyItem {
    /// Input symbols.
    pub symbols: Vec<usize>,
    /// Per-position target class, or [`IGNORE`].
    pub targets: Vec<i64>,
}

/// An in-memory dataset of generated [`TallyItem`]s.
pub struct TallyDataset {
    dataset: InMemDataset<TallyItem>,
}

impl TallyDataset {
    /// `count` sequences of one generator, seeded deterministically.
    pub fn new(task: &Task, generator: Generator, count: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let items = (0..count)
            .map(|_| {
                let symbols = generator(&mut rng, task.seq_length);
                assert_eq!(symbols.len(), task.seq_length);
                let targets = (task.labels)(&symbols);
                TallyItem { symbols, targets }
            })
            .collect();
        Self {
            dataset: InMemDataset::new(items),
        }
    }

    /// The items, in order.
    pub fn items(&self) -> Vec<TallyItem> {
        self.iter().map(|i| i.expect("dataset item")).collect()
    }
}

impl Dataset<TallyItem> for TallyDataset {
    fn get(&self, index: usize) -> Result<TallyItem, DatasetError> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// One-hot encode a symbol sequence into `[seq, num_symbols]`.
pub fn one_hot(symbols: &[usize], num_symbols: usize, device: &Device) -> Tensor<2> {
    let mut buf = vec![0.0f32; symbols.len() * num_symbols];
    for (t, &s) in symbols.iter().enumerate() {
        buf[t * num_symbols + s] = 1.0;
    }
    Tensor::<1>::from_floats(buf.as_slice(), device).reshape([symbols.len(), num_symbols])
}

/// Collates [`TallyItem`]s into a [`TallyBatch`].
#[derive(Clone, Debug)]
pub struct TallyBatcher {
    /// Input alphabet size.
    pub num_symbols: usize,
}

/// A batch of one-hot symbol sequences and their targets.
#[derive(Clone, Debug)]
pub struct TallyBatch {
    /// `[batch, seq, num_symbols]`.
    pub inputs: Tensor<3>,
    /// `[batch, seq]`, [`IGNORE`] where unscored.
    pub targets: Tensor<2, Int>,
}

impl TallyBatch {
    /// The batch, built by a dataloader worker on the host, moved to `device`
    /// by the thread that steps the model (see `device::loader_device`).
    pub fn to_device(self, device: &Device) -> Self {
        use crate::common::device::{batch_float, batch_int};
        Self {
            inputs: batch_float(self.inputs, device),
            targets: batch_int(self.targets, device),
        }
    }
}

impl Batcher<TallyItem, TallyBatch> for TallyBatcher {
    fn batch(&self, items: Vec<TallyItem>, device: &Device) -> TallyBatch {
        let inputs = items
            .iter()
            .map(|item| one_hot(&item.symbols, self.num_symbols, device))
            .collect();
        let targets = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_ints(item.targets.as_slice(), device))
            .collect();
        TallyBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
