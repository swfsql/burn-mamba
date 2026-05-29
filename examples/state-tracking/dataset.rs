//! The `A₅` state-tracking dataset.
//!
//! Each item is a leading **reference token** followed by a random sequence of
//! `A₅` generators (a 5-cycle and a 3-cycle); the per-position target is the
//! *running product* so far — the index of the cumulative permutation in the
//! enumerated `A₅` (a 60-way classification at every step). Predicting it
//! requires tracking composition in the non-solvable group `A₅`, which is the
//! capability the quaternion rotation demonstrates (see the crate-level docs in
//! `main.rs`).
//!
//! ## Why the reference token
//!
//! The Mamba-3 rotation rotates both `B` and `C`, so the SSD readout at position
//! `t` for a key at position `i` sees only the **relative** rotation
//! `Rₜ⋯Rᵢ₊₁ = Pₜ Pᵢ⁻¹` (RoPE-style), never the **absolute** running product
//! `Pₜ = Rₜ⋯R₁` that the task asks for. Prepending a fixed reference symbol at
//! position 0 (whose rotation the model learns to be the identity, `P₀ = I`)
//! anchors the readout: its contribution `Cₜᵀ Pₜ B₀ x₀` carries the absolute
//! product. Without it both rotations are confined to relative information and
//! collapse at the same shallow depth, hiding the quaternion's advantage.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Number of generator symbols (a 5-cycle and a 3-cycle) that generate `A₅`.
///
/// Note on difficulty: with `g` generators the input-prefix space grows as
/// `g^depth`, so an eval prefix is likely *already seen in training* up to a
/// "memorisation frontier" of ≈ `log_g(NUM_TRAIN)`. Below that depth even the
/// abelian model scores by memorising; the genuine abelian-vs-non-abelian gap
/// only appears *beyond* it (where composition is required). At this tiny
/// single-layer scale the gap is modest — see the README.
pub const NUM_GENERATORS: usize = 2;
/// Input alphabet size: the generators plus one leading reference token.
pub const NUM_SYMBOLS: usize = NUM_GENERATORS + 1;
/// Input channel of the reference (BOS / identity-anchor) token.
pub const REFERENCE_SYMBOL: usize = NUM_GENERATORS;
/// Number of `A₅` elements, i.e. the number of output classes.
pub const NUM_CLASSES: usize = 60;
/// Number of generators per sequence; the full token sequence is one longer (the
/// leading reference token), i.e. `SEQ_LENGTH + 1`.
pub const SEQ_LENGTH: usize = 12;
/// Number of training sequences.
pub const NUM_TRAIN: usize = 512;
/// Number of evaluation sequences.
pub const NUM_EVAL: usize = 128;

/// Dataset RNG seed for the training split.
pub const TRAIN_SEED: u64 = 0xC0FFEE;
/// Dataset RNG seed for the evaluation split (distinct from training).
pub const EVAL_SEED: u64 = 0xBEEF;

// ---------------------------------------------------------------------------
// A₅ group: enumerate the 60 even permutations of {0,..,4} and compose them.
// ---------------------------------------------------------------------------

/// The `A₅` generators used as the input alphabet: a 5-cycle and a 3-cycle
/// (both even permutations, and together they generate `A₅`).
pub fn generators() -> [[usize; 5]; NUM_GENERATORS] {
    [
        [1, 2, 3, 4, 0], // 5-cycle (0 1 2 3 4)
        [1, 2, 0, 3, 4], // 3-cycle (0 1 2)
    ]
}

/// `n!` permutations of `[0,1,2,3,4]`, keeping the even ones (sign `+1`), sorted
/// so the class indices are stable across runs.
pub fn even_permutations() -> Vec<[usize; 5]> {
    let mut perms = Vec::new();
    let mut p = [0usize, 1, 2, 3, 4];
    permute(&mut p, 0, &mut perms);
    perms.retain(|p| parity_even(p));
    perms.sort();
    perms
}

fn permute(p: &mut [usize; 5], k: usize, out: &mut Vec<[usize; 5]>) {
    if k == 5 {
        out.push(*p);
        return;
    }
    for i in k..5 {
        p.swap(k, i);
        permute(p, k + 1, out);
        p.swap(k, i);
    }
}

/// A permutation is even when its number of inversions is even.
fn parity_even(p: &[usize; 5]) -> bool {
    let mut inv = 0;
    for i in 0..5 {
        for j in (i + 1)..5 {
            if p[i] > p[j] {
                inv += 1;
            }
        }
    }
    inv % 2 == 0
}

/// `(a ∘ b)[i] = a[b[i]]` — apply `b`, then `a`.
fn compose(a: &[usize; 5], b: &[usize; 5]) -> [usize; 5] {
    let mut r = [0usize; 5];
    for i in 0..5 {
        r[i] = a[b[i]];
    }
    r
}

/// Tiny deterministic RNG (SplitMix64) so the dataset is reproducible without
/// pulling in an external dependency.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Dataset / batcher
// ---------------------------------------------------------------------------

/// One generated sequence: the input symbol at each position and the matching
/// per-position target class (the running product's `A₅` index). Position 0 is
/// always the [`REFERENCE_SYMBOL`] (identity anchor); positions `1..=SEQ_LENGTH`
/// are generators.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateTrackingItem {
    /// Input symbol at each position (`0..NUM_SYMBOLS`), length `SEQ_LENGTH + 1`.
    pub symbols: Vec<usize>,
    /// Target `A₅`-element class at each position, length `SEQ_LENGTH + 1`.
    pub targets: Vec<i64>,
}

/// An in-memory dataset of randomly generated [`StateTrackingItem`]s.
pub struct StateTrackingDataset {
    dataset: InMemDataset<StateTrackingItem>,
}

impl StateTrackingDataset {
    /// Generate `num_sequences` random sequences (a leading reference token then
    /// `seq_length` generators) and their running-product targets, seeded
    /// deterministically by `seed`.
    pub fn new(num_sequences: usize, seq_length: usize, seed: u64) -> Self {
        let perms = even_permutations();
        assert_eq!(perms.len(), NUM_CLASSES, "A₅ has 60 elements");
        let index_of: HashMap<[usize; 5], usize> =
            perms.iter().enumerate().map(|(i, p)| (*p, i)).collect();
        let generators = generators();
        let identity = [0usize, 1, 2, 3, 4];
        let identity_class = index_of[&identity] as i64;

        let mut rng = Lcg(seed);
        let items = (0..num_sequences)
            .map(|_| {
                let mut state = identity;
                let mut symbols = Vec::with_capacity(seq_length + 1);
                let mut targets = Vec::with_capacity(seq_length + 1);
                // position 0: the reference token anchors the readout at identity.
                symbols.push(REFERENCE_SYMBOL);
                targets.push(identity_class);
                for _ in 0..seq_length {
                    let g = rng.below(NUM_GENERATORS);
                    state = compose(&generators[g], &state); // Pₜ = g ∘ Pₜ₋₁
                    symbols.push(g);
                    targets.push(index_of[&state] as i64);
                }
                StateTrackingItem { symbols, targets }
            })
            .collect();

        Self {
            dataset: InMemDataset::new(items),
        }
    }
}

impl Dataset<StateTrackingItem> for StateTrackingDataset {
    fn get(&self, index: usize) -> Option<StateTrackingItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`StateTrackingItem`]s into a [`StateTrackingBatch`], building the
/// one-hot symbol inputs.
#[derive(Clone, Debug, Default)]
pub struct StateTrackingBatcher {}

/// A batch of one-hot symbol sequences and their per-position target classes.
#[derive(Clone, Debug)]
pub struct StateTrackingBatch {
    /// One-hot input symbol at each position, `[batch, seq, NUM_SYMBOLS]`.
    pub inputs: Tensor<3>,
    /// Per-position target class, `[batch, seq]`.
    pub targets: Tensor<2, Int>,
}

impl Batcher<StateTrackingItem, StateTrackingBatch> for StateTrackingBatcher {
    fn batch(&self, items: Vec<StateTrackingItem>, device: &Device) -> StateTrackingBatch {
        let mut inputs: Vec<Tensor<2>> = Vec::with_capacity(items.len());
        let mut targets: Vec<Tensor<1, Int>> = Vec::with_capacity(items.len());

        for item in items.iter() {
            let seq = item.symbols.len();
            // one-hot encode the symbol at each position → [seq, NUM_SYMBOLS]
            let mut one_hot = vec![0.0f32; seq * NUM_SYMBOLS];
            for (t, &s) in item.symbols.iter().enumerate() {
                one_hot[t * NUM_SYMBOLS + s] = 1.0;
            }
            inputs.push(
                Tensor::<1>::from_floats(one_hot.as_slice(), device).reshape([seq, NUM_SYMBOLS]),
            );
            targets.push(Tensor::<1, Int>::from_ints(item.targets.as_slice(), device));
        }

        StateTrackingBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
