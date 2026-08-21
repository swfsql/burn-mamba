//! The reset-majority dataset: a three-symbol stream whose per-position target
//! is the **sign of the running vote since the last reset**.
//!
//! ```text
//!   symbols   −  +  +  −  +  R  −  −  +  −
//!   count     -1  0  1  0  1  0 -1 -2 -1 -2
//!   target   Neg  .  Pos  .  Pos  .  Neg Neg Neg Neg
//! ```
//!
//! Positions where the vote is exactly zero (every reset, and every tie) have no
//! sign to report and are **not scored** — see [`IGNORE`].
//!
//! Two properties make this the task a single Mamba-2 block is for:
//!
//! - **It needs the SSM state.** The lookback is unbounded (a reset may be
//!   arbitrarily far back) and the answer is not a function of the last symbol.
//!   With `conv_kernel = 1` the recurrent state is the model's *only* memory.
//! - **It needs the state to be *selective*.** A fixed decay `ᾱ` cannot both
//!   erase a reset's past outright and keep an unweighted vote afterwards.
//!   [`Family::LongPrefix`] and [`Family::LongSuffix`] are the two adversarial
//!   halves that pin that down: the first buries a 1-vote majority behind a long
//!   pre-reset run (any `ᾱ` near 1 leaks it through), the second decides a long
//!   post-reset vote on its *early* tokens (any small `ᾱ` votes with the recent
//!   ones instead). See `tests.rs` for the sweep.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, DatasetError, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};

/// Input symbol: subtract one vote.
pub const MINUS: usize = 0;
/// Input symbol: add one vote.
pub const PLUS: usize = 1;
/// Input symbol: clear the vote (the selective-forget token).
pub const RESET: usize = 2;
/// Input alphabet size.
pub const NUM_SYMBOLS: usize = 3;

/// Target class: the running vote is negative.
pub const NEG: i64 = 0;
/// Target class: the running vote is positive.
pub const POS: i64 = 1;
/// Number of output classes.
pub const NUM_CLASSES: usize = 2;

/// Placeholder target for a position with **no sign to report** — the vote is
/// exactly zero, which every reset and every tie produces.
///
/// Those positions are dropped from the loss and from the accuracy (the batcher
/// emits [`ResetMajorityBatch::scored`] for the rest). Asking for them as a
/// third class instead is a much harder objective for no gain in what the task
/// tests: it turns a *sign* readout into an exact-zero detector, and the model
/// spends its capacity calibrating a band rather than holding a vote.
pub const IGNORE: i64 = -1;

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

/// The per-position targets implied by a symbol sequence.
///
/// `RESET` clears the running count; `PLUS` / `MINUS` move it by one. The label
/// is the count's sign *after* consuming the symbol at that position, or
/// [`IGNORE`] when the count is zero.
pub fn labels(symbols: &[usize]) -> Vec<i64> {
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
            match count.cmp(&0) {
                std::cmp::Ordering::Less => NEG,
                std::cmp::Ordering::Equal => IGNORE,
                std::cmp::Ordering::Greater => POS,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Which generator a split draws from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Independent symbols: `RESET` with probability ~⅛, otherwise `±` evenly.
    Random,
    /// A long same-sign run, one `RESET`, then a majority of **one vote** the
    /// other way. Defeats any decay close to 1 (the buried run leaks through).
    LongPrefix,
    /// An early `RESET`, then `b+1` votes one way followed by `b` the other, so
    /// the majority is decided by the *oldest* post-reset tokens. Defeats any
    /// decay far from 1 (the recent block outvotes them).
    LongSuffix,
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
    (0..len)
        .map(|_| match rng.below(8) {
            0 => RESET,
            n if n % 2 == 0 => PLUS,
            _ => MINUS,
        })
        .collect()
}

fn gen_long_prefix(rng: &mut Lcg, len: usize) -> Vec<usize> {
    assert!(len >= 8, "LongPrefix needs room for a run, a reset and a vote");
    // `j` odd ⇒ the post-reset vote is decided by exactly one ballot.
    let j = 5 + 2 * rng.below((len / 4).max(1));
    let m = len - 1 - j;
    let (run, opp) = if rng.below(2) == 0 {
        (PLUS, MINUS)
    } else {
        (MINUS, PLUS)
    };
    let mut out = vec![run; m];
    out.push(RESET);
    // The opposing ballots come first, so the vote is `opp` at *every* tail
    // position, not just the last one: a decay that leaks the pre-reset run
    // through is then wrong on the whole tail rather than half of it.
    out.extend(std::iter::repeat_n(opp, j / 2 + 1));
    out.extend(std::iter::repeat_n(run, j / 2));
    out
}

fn gen_long_suffix(rng: &mut Lcg, len: usize) -> Vec<usize> {
    assert!(len >= 8, "LongSuffix needs room for a reset and two blocks");
    let head = 1 + rng.below(3);
    let mut out: Vec<usize> = (0..head)
        .map(|_| if rng.below(2) == 0 { PLUS } else { MINUS })
        .collect();
    out.push(RESET);
    // `b + 1` early votes one way, then `b` late votes the other: the majority
    // is one ballot wide and sits at the *far* end of the post-reset window.
    let j = len - out.len();
    let b = (j - 1) / 2;
    let (early, late) = if rng.below(2) == 0 {
        (PLUS, MINUS)
    } else {
        (MINUS, PLUS)
    };
    out.extend(std::iter::repeat_n(early, j - b));
    out.extend(std::iter::repeat_n(late, b));
    out
}

/// Generate one sequence of the given family.
pub fn generate(family: Family, rng_state: &mut u64, len: usize) -> Vec<usize> {
    let mut rng = Lcg(*rng_state);
    let out = match family {
        Family::Random => gen_random(&mut rng, len),
        Family::LongPrefix => gen_long_prefix(&mut rng, len),
        Family::LongSuffix => gen_long_suffix(&mut rng, len),
        Family::Mixed => match rng.below(4) {
            0 => gen_long_prefix(&mut rng, len),
            1 => gen_long_suffix(&mut rng, len),
            _ => gen_random(&mut rng, len),
        },
    };
    *rng_state = rng.0;
    out
}

// ---------------------------------------------------------------------------
// Dataset / batcher
// ---------------------------------------------------------------------------

/// One generated sequence and its per-position target class.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResetMajorityItem {
    /// Input symbols, one of [`MINUS`] / [`PLUS`] / [`RESET`].
    pub symbols: Vec<usize>,
    /// Per-position target class ([`NEG`] / [`ZERO`] / [`POS`]).
    pub targets: Vec<i64>,
}

/// An in-memory dataset of generated [`ResetMajorityItem`]s.
pub struct ResetMajorityDataset {
    dataset: InMemDataset<ResetMajorityItem>,
}

impl ResetMajorityDataset {
    /// Generate `num_sequences` sequences of one family, seeded deterministically.
    pub fn new(num_sequences: usize, seq_length: usize, family: Family, seed: u64) -> Self {
        let mut state = seed;
        let items = (0..num_sequences)
            .map(|_| {
                let symbols = generate(family, &mut state, seq_length);
                let targets = labels(&symbols);
                ResetMajorityItem { symbols, targets }
            })
            .collect();
        Self {
            dataset: InMemDataset::new(items),
        }
    }
}

impl Dataset<ResetMajorityItem> for ResetMajorityDataset {
    fn get(&self, index: usize) -> Result<ResetMajorityItem, DatasetError> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`ResetMajorityItem`]s into a [`ResetMajorityBatch`], one-hotting
/// the symbols.
#[derive(Clone, Debug, Default)]
pub struct ResetMajorityBatcher {}

/// A batch of one-hot symbol sequences and their per-position target classes.
#[derive(Clone, Debug)]
pub struct ResetMajorityBatch {
    /// One-hot input symbol at each position, `[batch, seq, NUM_SYMBOLS]`.
    pub inputs: Tensor<3>,
    /// Per-position target class, `[batch, seq]`; [`IGNORE`] where the vote is
    /// zero.
    pub targets: Tensor<2, Int>,
    /// Flat indices (row-major over `batch × seq`) of the positions that carry a
    /// sign — everything but the [`IGNORE`]s.
    pub scored: Tensor<1, Int>,
}

/// One-hot encode a symbol sequence into `[seq, NUM_SYMBOLS]`.
pub fn one_hot(symbols: &[usize], device: &Device) -> Tensor<2> {
    let mut buf = vec![0.0f32; symbols.len() * NUM_SYMBOLS];
    for (t, &s) in symbols.iter().enumerate() {
        buf[t * NUM_SYMBOLS + s] = 1.0;
    }
    Tensor::<1>::from_floats(buf.as_slice(), device).reshape([symbols.len(), NUM_SYMBOLS])
}

impl Batcher<ResetMajorityItem, ResetMajorityBatch> for ResetMajorityBatcher {
    fn batch(&self, items: Vec<ResetMajorityItem>, device: &Device) -> ResetMajorityBatch {
        let inputs: Vec<Tensor<2>> = items
            .iter()
            .map(|item| one_hot(&item.symbols, device))
            .collect();
        let targets: Vec<Tensor<1, Int>> = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_ints(item.targets.as_slice(), device))
            .collect();
        let seq = items[0].symbols.len();
        let scored: Vec<i32> = items
            .iter()
            .enumerate()
            .flat_map(|(b, item)| {
                item.targets
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c != IGNORE)
                    .map(move |(t, _)| (b * seq + t) as i32)
            })
            .collect();
        ResetMajorityBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
            scored: Tensor::<1, Int>::from_ints(scored.as_slice(), device),
        }
    }
}
