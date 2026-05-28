//! Synthetic fibonacci-like dataset: each item is a short sequence where every
//! value is (roughly) the sum of the previous two plus Gaussian noise, and the
//! target is the next value. Used to give the fibonacci example a learnable
//! recurrence to fit.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, InMemDataset},
};
use burn::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Number of sequences in the generated dataset.
pub const NUM_SEQUENCES: usize = 1000;
/// Length of each generated sequence.
pub const SEQ_LENGTH: usize = 10;
/// Standard deviation of the additive Gaussian noise.
pub const NOISE_LEVEL: f32 = 0.1;

/// One sequence plus its next-value target.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SequenceDatasetItem {
    /// The input sequence.
    pub sequence: Vec<f32>,
    /// The value following the sequence (the regression target).
    pub target: f32,
}

impl SequenceDatasetItem {
    pub fn new(seq_length: usize, noise_level: f32) -> Self {
        // Start with two random numbers between 0 and 1
        let lower = rand::rng().random::<f32>();
        let upper = rand::rng().random::<f32>();
        let mut seq = vec![lower, upper];

        // Generate sequence
        for _i in 0..seq_length {
            // Next number is sum of previous two plus noise
            let normal = Normal::new(0.0, noise_level).unwrap();
            let normal: f32 = normal.sample(&mut rand::rng());
            let next_val = seq[seq.len() - 2] + seq[seq.len() - 1] + normal;
            seq.push(next_val);
        }

        Self {
            // Convert to sequence and target
            sequence: seq[0..seq.len() - 1].to_vec(), // All but last
            target: seq[seq.len() - 1],               // Last value
        }
    }
}

/// An in-memory dataset of randomly generated [`SequenceDatasetItem`]s.
pub struct SequenceDataset {
    dataset: InMemDataset<SequenceDatasetItem>,
}

impl SequenceDataset {
    /// Generate `num_sequences` random sequences of the given length and noise.
    pub fn new(num_sequences: usize, seq_length: usize, noise_level: f32) -> Self {
        let mut items = vec![];
        for _i in 0..num_sequences {
            items.push(SequenceDatasetItem::new(seq_length, noise_level));
        }
        let dataset = InMemDataset::new(items);

        Self { dataset }
    }
}

impl Dataset<SequenceDatasetItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`SequenceDatasetItem`]s into a [`SequenceBatch`].
#[derive(Clone, Debug, Default)]
pub struct SequenceBatcher {}

/// A batch of sequences and their regression targets.
#[derive(Clone, Debug)]
pub struct SequenceBatch {
    /// Input sequences, shape `[batch_size, seq_length, 1]`.
    pub sequences: Tensor<3>,
    /// Targets, shape `[batch_size, 1]`.
    pub targets: Tensor<2>,
}

impl Batcher<SequenceDatasetItem, SequenceBatch> for SequenceBatcher {
    fn batch(&self, items: Vec<SequenceDatasetItem>, device: &Device) -> SequenceBatch {
        let mut sequences: Vec<Tensor<2>> = Vec::new();

        for item in items.iter() {
            let seq_tensor = Tensor::<1>::from_floats(item.sequence.as_slice(), device);
            // Add feature dimension, the input_size is 1 implicitly. We can change the input_size here with some operations
            sequences.push(seq_tensor.unsqueeze_dims(&[-1]));
        }
        let sequences = Tensor::stack(sequences, 0);

        let targets = items
            .iter()
            .map(|item| Tensor::<1>::from_floats([item.target], device))
            .collect();
        let targets = Tensor::stack(targets, 0);

        SequenceBatch { sequences, targets }
    }
}
