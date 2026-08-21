//! The reset-rotor dataset: the same three-symbol stream as `reset-majority`,
//! read as a **rotor with three detents**. `+` turns it one detent forward, `-`
//! one back, `R` snaps it to detent 0; the per-position target is the detent the
//! rotor is on — the running turn count since the last reset, **mod 3**.
//!
//! ```text
//!   symbols   R  +  +  +  -  +  +  R  -  -
//!   turns     0  1  2  3  2  3  4  0 -1 -2
//!   target    0  1  2  0  2  0  1  0  2  1
//! ```
//!
//! Every position is scored, and every sequence opens with an `R` — the reset
//! that anchors the rotor (see [`crate::model`]: it is the token that gives the
//! block its phase reference).
//!
//! Three properties make this the task a single **Mamba-3** block is for:
//!
//! - **It needs the SSM state.** The lookback is unbounded and the label is not
//!   a function of the current symbol. Mamba-3 has no short convolution at all,
//!   so the recurrent state is the model's only memory.
//! - **It needs the decay to be selective** — the same requirement
//!   `reset-majority` isolates: `R` must erase the past outright while the turns
//!   after it stay unweighted.
//! - **And it needs the transition to be *complex*.** The label is a *periodic*
//!   function of the turn count, and a real state with non-negative eigenvalues
//!   (every Mamba-1/Mamba-2 state, and Mamba-3 with the rotation switched off)
//!   can only hold that count, not its residue: a linear readout cuts the count
//!   axis into at most `NUM_CLASSES` intervals, while the answer alternates
//!   along it.
//! - **And the turn has to be data-dependent.** A fixed per-step angle is
//!   vanilla RoPE: the phase it accumulates between the reset and the read
//!   measures *positions*, not turns.
//!
//! The two adversarial families close the two shortcuts those leave open —
//! [`Family::Drift`] drives the count out of any three-interval readout's
//! reach, [`Family::Balanced`] decorrelates it from the position — while
//! [`Family::Random`] is the mixture where both are partly available.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, DatasetError, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};

/// Input symbol: turn the rotor one detent back.
pub const MINUS: usize = 0;
/// Input symbol: turn the rotor one detent forward.
pub const PLUS: usize = 1;
/// Input symbol: snap the rotor back to detent 0 (the selective-forget token).
pub const RESET: usize = 2;
/// Input alphabet size.
pub const NUM_SYMBOLS: usize = 3;

/// Number of detents — the modulus of the readout.
pub const MODULUS: i64 = 3;
/// Number of output classes: one per detent.
pub const NUM_CLASSES: usize = MODULUS as usize;

/// Length of every generated sequence.
pub const SEQ_LENGTH: usize = 32;
/// Number of training sequences.
pub const NUM_TRAIN: usize = 4096;
/// Number of evaluation sequences (per family).
pub const NUM_EVAL: usize = 512;

/// Dataset RNG seed for the training split.
pub const TRAIN_SEED: u64 = 0xC0FFEE;
/// Dataset RNG seed for the evaluation splits (distinct from training).
pub const EVAL_SEED: u64 = 0xBEEF;

/// The running turn count since the last reset, at every position.
///
/// `RESET` clears it; `PLUS` / `MINUS` move it by one. The value is the count
/// *after* consuming the symbol at that position.
pub fn turns(symbols: &[usize]) -> Vec<i64> {
    let mut count: i64 = 0;
    symbols
        .iter()
        .map(|&s| {
            match s {
                RESET => count = 0,
                PLUS => count += 1,
                MINUS => count -= 1,
                _ => panic!("symbol out of alphabet: {s}"),
            }
            count
        })
        .collect()
}

/// Positions elapsed since the last reset (`0` at the reset itself).
///
/// This is everything a **fixed** (input-independent) rotation can see: its
/// phase is `ω · steps_since_reset`. See `tests.rs`.
pub fn steps_since_reset(symbols: &[usize]) -> Vec<i64> {
    let mut steps: i64 = 0;
    symbols
        .iter()
        .map(|&s| {
            if s == RESET {
                steps = 0;
            } else {
                steps += 1;
            }
            steps
        })
        .collect()
}

/// The per-position targets implied by a symbol sequence: the detent index,
/// i.e. [`turns`] reduced mod [`MODULUS`].
pub fn labels(symbols: &[usize]) -> Vec<i64> {
    turns(symbols)
        .into_iter()
        .map(|c| c.rem_euclid(MODULUS))
        .collect()
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Which generator a split draws from. Every family opens with a `RESET`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Independent symbols: `RESET` with probability ~⅛, otherwise `±` evenly.
    Random,
    /// One reset, then a strongly biased walk: the turn count runs out to `±31`,
    /// sweeping the detents over and over at large magnitude. This is what a
    /// **real** state cannot follow — holding the count is easy, but no readout
    /// that cuts the count axis into three intervals can report a residue that
    /// alternates across thirty of them.
    Drift,
    /// One reset, then a shuffled bag of equally many `+` and `-`: the count
    /// stays inside `±9` while the position marches on, and the order is
    /// random, so the two are decorrelated. This is what a **positional**
    /// phase — a fixed rotation, or anything else keyed to the steps since the
    /// reset — cannot follow.
    Balanced,
    /// The training mixture: half [`Self::Random`], a quarter of each
    /// adversarial family.
    Mixed,
}

/// SplitMix64 — a small deterministic RNG so splits reproduce exactly.
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

fn gen_random(rng: &mut Lcg, len: usize) -> Vec<usize> {
    let mut out = vec![RESET];
    out.extend((1..len).map(|_| match rng.below(8) {
        0 => RESET,
        n if n % 2 == 0 => PLUS,
        _ => MINUS,
    }));
    out
}

fn gen_drift(rng: &mut Lcg, len: usize) -> Vec<usize> {
    let (run, opp) = if rng.below(2) == 0 {
        (PLUS, MINUS)
    } else {
        (MINUS, PLUS)
    };
    // 7/8 of the steps go the same way, so the count leaves the range any
    // three-interval readout could cover — but the sequence is not simply
    // "position since the reset" either.
    let mut out = vec![RESET];
    out.extend((1..len).map(|_| if rng.below(8) == 0 { opp } else { run }));
    out
}

fn gen_balanced(rng: &mut Lcg, len: usize) -> Vec<usize> {
    // Equally many `+` and `-` (a bridge walk), shuffled: the count stays small
    // and says nothing about how far the sequence has advanced.
    let tail = len - 1;
    let mut bag: Vec<usize> = (0..tail)
        .map(|i| if i * 2 < tail { PLUS } else { MINUS })
        .collect();
    for i in (1..bag.len()).rev() {
        bag.swap(i, rng.below(i + 1));
    }
    let mut out = vec![RESET];
    out.extend(bag);
    out
}

/// Generate one sequence of the given family.
pub fn generate(family: Family, rng_state: &mut u64, len: usize) -> Vec<usize> {
    assert!(len >= 8, "a sequence needs room for a reset and a walk");
    let mut rng = Lcg(*rng_state);
    let out = match family {
        Family::Random => gen_random(&mut rng, len),
        Family::Drift => gen_drift(&mut rng, len),
        Family::Balanced => gen_balanced(&mut rng, len),
        Family::Mixed => match rng.below(4) {
            0 => gen_drift(&mut rng, len),
            1 => gen_balanced(&mut rng, len),
            _ => gen_random(&mut rng, len),
        },
    };
    *rng_state = rng.0;
    out
}

// ---------------------------------------------------------------------------
// Dataset / batcher
// ---------------------------------------------------------------------------

/// One generated sequence and its per-position target detent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResetRotorItem {
    /// Input symbols, one of [`MINUS`] / [`PLUS`] / [`RESET`].
    pub symbols: Vec<usize>,
    /// Per-position target detent, in `0..`[`MODULUS`].
    pub targets: Vec<i64>,
}

/// An in-memory dataset of generated [`ResetRotorItem`]s.
pub struct ResetRotorDataset {
    dataset: InMemDataset<ResetRotorItem>,
}

impl ResetRotorDataset {
    /// Generate `num_sequences` sequences of one family, seeded deterministically.
    pub fn new(num_sequences: usize, seq_length: usize, family: Family, seed: u64) -> Self {
        let mut state = seed;
        let items = (0..num_sequences)
            .map(|_| {
                let symbols = generate(family, &mut state, seq_length);
                let targets = labels(&symbols);
                ResetRotorItem { symbols, targets }
            })
            .collect();
        Self {
            dataset: InMemDataset::new(items),
        }
    }
}

impl Dataset<ResetRotorItem> for ResetRotorDataset {
    fn get(&self, index: usize) -> Result<ResetRotorItem, DatasetError> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`ResetRotorItem`]s into a [`ResetRotorBatch`], one-hotting the
/// symbols.
#[derive(Clone, Debug, Default)]
pub struct ResetRotorBatcher {}

/// A batch of one-hot symbol sequences and their per-position target detents.
#[derive(Clone, Debug)]
pub struct ResetRotorBatch {
    /// One-hot input symbol at each position, `[batch, seq, NUM_SYMBOLS]`.
    pub inputs: Tensor<3>,
    /// Per-position target detent, `[batch, seq]`.
    pub targets: Tensor<2, Int>,
}

/// One-hot encode a symbol sequence into `[seq, NUM_SYMBOLS]`.
pub fn one_hot(symbols: &[usize], device: &Device) -> Tensor<2> {
    let mut buf = vec![0.0f32; symbols.len() * NUM_SYMBOLS];
    for (t, &s) in symbols.iter().enumerate() {
        buf[t * NUM_SYMBOLS + s] = 1.0;
    }
    Tensor::<1>::from_floats(buf.as_slice(), device).reshape([symbols.len(), NUM_SYMBOLS])
}

impl Batcher<ResetRotorItem, ResetRotorBatch> for ResetRotorBatcher {
    fn batch(&self, items: Vec<ResetRotorItem>, device: &Device) -> ResetRotorBatch {
        let inputs: Vec<Tensor<2>> = items
            .iter()
            .map(|item| one_hot(&item.symbols, device))
            .collect();
        let targets: Vec<Tensor<1, Int>> = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_ints(item.targets.as_slice(), device))
            .collect();
        ResetRotorBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
