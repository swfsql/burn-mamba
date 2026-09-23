//! The reset-majority dataset: a three-symbol stream. The target at each
//! position is the **sign of the running vote since the last reset**.
//!
//! ```text
//!   symbols    −    +    +    −    +    R    −    −    +    −
//!   count     -1    0    1    0    1    0   -1   -2   -1   -2
//!   target   Neg    .  Pos    .  Pos    .  Neg  Neg  Neg  Neg
//! ```
//!
//! A position where the vote is exactly zero (every reset, and every tie) has
//! no sign to report, so it is **not scored** (see [`IGNORE`]).
//!
//! Two properties make this a task for a single selective-decay block:
//!
//! - **It needs the SSM state.** The lookback has no bound (a reset can be
//!   arbitrarily far back), and the answer is not a function of the last
//!   symbol. Mamba-3 has no short convolution, so the recurrent state is the
//!   *only* memory of the model.
//! - **It needs the state to be *selective*.** A fixed decay `ᾱ` cannot both
//!   erase the past at a reset and keep an unweighted vote after it.
//!   [`Family::LongPrefix`] and [`Family::LongSuffix`] are the two adversarial
//!   halves that show this. `LongPrefix` puts a 1-vote majority after a long
//!   pre-reset run, and any `ᾱ` near 1 leaks the run into the vote.
//!   `LongSuffix` decides a long post-reset vote on its *early* tokens, and any
//!   small `ᾱ` votes with the recent tokens instead. `tests.rs` has the sweep.

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

/// Placeholder target for a position with **no sign to report**: the vote is
/// exactly zero, as at every reset and every tie.
///
/// The loss and the accuracy (`training::forward_classification`) mask these
/// positions. A third class for them is a much harder objective, and it adds
/// nothing to what the task tests. It turns a *sign* readout into an
/// exact-zero detector, and the model spends its capacity to calibrate a band
/// instead of holding a vote.
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
/// `RESET` clears the running count. `PLUS` / `MINUS` move it by one. The label
/// is the sign of the count *after* the symbol at that position, or [`IGNORE`]
/// when the count is zero.
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
    /// Independent symbols: `RESET` with probability ⅛, `PLUS` ⅜ and `MINUS`
    /// ½.
    Random,
    /// A long same-sign run, one `RESET`, then a majority of **one vote** the
    /// other way. It defeats any decay close to 1 (the old run leaks through).
    LongPrefix,
    /// An early `RESET`, then `j − b` votes one way and `b = ⌊(j − 1)/2⌋` the
    /// other way, for `j` post-reset tokens. So the *oldest* post-reset tokens
    /// decide the majority. It defeats any decay far from 1 (the recent block
    /// outvotes the old one).
    LongSuffix,
    /// The training mixture: half [`Self::Random`], a quarter of each
    /// adversarial family.
    Mixed,
}

/// SplitMix64: a small deterministic RNG, so the splits reproduce exactly.
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
    // position, not only the last one. A decay that leaks the pre-reset run
    // is then wrong on the whole tail, not on half of it.
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
    // `j − b` early votes one way, then `b` late votes the other way. The
    // majority is one ballot wide at odd `j` and two at even `j`, and it sits
    // at the *far* end of the post-reset window.
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
    /// Per-position target class ([`NEG`] / [`POS`] / [`IGNORE`]).
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
    /// Per-position target class, `[batch, seq]`. It is [`IGNORE`] where the vote
    /// is zero.
    pub targets: Tensor<2, Int>,
}

impl ResetMajorityBatch {
    /// Moves the batch to `device`. A dataloader worker builds the batch on the
    /// host, and the thread that steps the model moves it (see
    /// `device::loader_device`).
    pub fn to_device(self, device: &Device) -> Self {
        use crate::common::device::{batch_float, batch_int};
        Self {
            inputs: batch_float(self.inputs, device),
            targets: batch_int(self.targets, device),
        }
    }
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
        ResetMajorityBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
