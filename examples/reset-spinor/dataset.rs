//! The reset-spinor dataset: a three-symbol stream whose per-position target is
//! the **running product in the quaternion group `Q₈`** since the last reset.
//!
//! `i` and `j` each multiply the accumulated state **on the left**; `R` resets it
//! to `1`. The eight states are `±1, ±i, ±j, ±k`:
//!
//! ```text
//!   symbols   R   i   j   i   j   R   j   i   i
//!   state     1   i   k  -j  -1   1   j  -k  -i
//!   target    0   1   3   6   4   0   2   7   5
//! ```
//!
//! `Q₈` is the smallest non-abelian group of unit quaternions, and that is the
//! whole point: `ij = k` but `ji = −k`, so **how many** `i`s and `j`s have gone
//! by never determines the answer — only their **order** does. Formally the
//! commutator subgroup is `{±1}`, so the symbol counts pin the state down to a
//! sign and no further ([`counts_since_reset`] is what the ceilings in
//! `tests.rs` are computed over).
//!
//! Three properties make this the task a **quaternion** Mamba-3 block is for:
//!
//! - **It needs the SSM state.** The lookback is unbounded and the label is not
//!   a function of the current symbol; Mamba-3 has no short convolution, so the
//!   recurrent state is the model's only memory.
//! - **It needs the state to turn.** The label is periodic in every direction —
//!   `i⁴ = 1` — which no real, non-negative-eigenvalue state can report (the
//!   argument of `reset-rotor`, one rung down).
//! - **And it needs the turns to *not commute*.** An abelian (`Complex2D`)
//!   rotation accumulates a **sum** of angles, and a sum forgets the order; the
//!   quaternion accumulates an ordered **product**, which is exactly `Q₈`'s
//!   composition. [`Family::Shuffle`] pins that down.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, DatasetError, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};

/// Input symbol: multiply the state by `i` (on the left).
pub const TURN_I: usize = 0;
/// Input symbol: multiply the state by `j` (on the left).
pub const TURN_J: usize = 1;
/// Input symbol: reset the state to `1` (the selective-forget token).
pub const RESET: usize = 2;
/// Input alphabet size.
pub const NUM_SYMBOLS: usize = 3;

/// Number of output classes: the eight elements of `Q₈`.
pub const NUM_CLASSES: usize = 8;

/// Class index of the identity element `1`.
pub const IDENTITY: i64 = 0;

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

// ---------------------------------------------------------------------------
// The group
// ---------------------------------------------------------------------------

// A class index is `unit + 4·negative`: `0..4` are `1, i, j, k` and `4..8` are
// their negatives. The two tables below are the Hamilton product on the units
// (`i·j = k`, `j·i = −k`, `i² = j² = k² = −1`), split into the resulting unit
// and its sign.
const UNIT_MUL: [[usize; 4]; 4] = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]];
const UNIT_SIGN: [[i32; 4]; 4] = [[1, 1, 1, 1], [1, -1, 1, -1], [1, -1, -1, 1], [1, 1, -1, -1]];

/// Left-multiply the class `state` by the unit quaternion `unit` (`0..4`).
pub fn left_mul(unit: usize, state: i64) -> i64 {
    let (s_unit, s_neg) = ((state % 4) as usize, state >= 4);
    let neg = (UNIT_SIGN[unit][s_unit] < 0) ^ s_neg;
    UNIT_MUL[unit][s_unit] as i64 + if neg { 4 } else { 0 }
}

/// The class as a unit quaternion `(w, x, y, z)` — the vector the block's own
/// state holds, and the column the classifier head reads it with.
pub fn quaternion(class: i64) -> [f64; 4] {
    let mut q = [0.0; 4];
    q[(class % 4) as usize] = if class >= 4 { -1.0 } else { 1.0 };
    q
}

/// The per-position targets implied by a symbol sequence: the running product
/// since the last `RESET`, as a class index.
///
/// The newest factor multiplies **on the left**, matching the block's own
/// cumulative rotation (`Pₜ = qₜ ⊗ ⋯ ⊗ q₁`).
pub fn labels(symbols: &[usize]) -> Vec<i64> {
    let mut state = IDENTITY;
    symbols
        .iter()
        .map(|&s| {
            state = match s {
                TURN_I => left_mul(1, state),
                TURN_J => left_mul(2, state),
                RESET => IDENTITY,
                _ => panic!("symbol out of alphabet: {s}"),
            };
            state
        })
        .collect()
}

/// How many `i`s and `j`s have gone by since the last reset, at every position.
///
/// This is everything an **abelian** transition can carry: a cumulative sum of
/// per-symbol angles is a linear function of exactly these two numbers. It
/// determines the answer only up to a sign — see the module docs.
pub fn counts_since_reset(symbols: &[usize]) -> Vec<(i64, i64)> {
    let (mut a, mut b) = (0, 0);
    symbols
        .iter()
        .map(|&s| {
            match s {
                TURN_I => a += 1,
                TURN_J => b += 1,
                RESET => (a, b) = (0, 0),
                _ => panic!("symbol out of alphabet: {s}"),
            }
            (a, b)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Which generator a split draws from. Every family opens with a `RESET`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Independent symbols: `RESET` with probability ~⅛, otherwise `i` / `j`
    /// evenly. Resets are frequent, so many positions carry a **short** word —
    /// and a short word is often pinned down by its symbol counts alone.
    Random,
    /// One reset, then a shuffled bag of equally many `i`s and `j`s: the counts
    /// are fixed by construction and only the **order** varies. This is where an
    /// abelian state has nothing left to read.
    Shuffle,
    /// One reset, then long runs of one symbol (`i…i j…j i…i`). The word is
    /// still non-commutative, but grouping it into blocks is the case where the
    /// counts come closest to determining the product — the family every
    /// order-blind model does best on.
    Runs,
    /// The training mixture: half [`Self::Random`], a quarter of each of the
    /// other two.
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
        n if n % 2 == 0 => TURN_I,
        _ => TURN_J,
    }));
    out
}

fn gen_shuffle(rng: &mut Lcg, len: usize) -> Vec<usize> {
    let tail = len - 1;
    let mut bag: Vec<usize> = (0..tail)
        .map(|n| if n * 2 < tail { TURN_I } else { TURN_J })
        .collect();
    for n in (1..bag.len()).rev() {
        bag.swap(n, rng.below(n + 1));
    }
    let mut out = vec![RESET];
    out.extend(bag);
    out
}

fn gen_runs(rng: &mut Lcg, len: usize) -> Vec<usize> {
    let mut out = vec![RESET];
    let mut symbol = if rng.below(2) == 0 { TURN_I } else { TURN_J };
    while out.len() < len {
        let run = 3 + rng.below(6);
        for _ in 0..run.min(len - out.len()) {
            out.push(symbol);
        }
        symbol = if symbol == TURN_I { TURN_J } else { TURN_I };
    }
    out
}

/// Generate one sequence of the given family.
pub fn generate(family: Family, rng_state: &mut u64, len: usize) -> Vec<usize> {
    assert!(len >= 8, "a sequence needs room for a reset and a word");
    let mut rng = Lcg(*rng_state);
    let out = match family {
        Family::Random => gen_random(&mut rng, len),
        Family::Shuffle => gen_shuffle(&mut rng, len),
        Family::Runs => gen_runs(&mut rng, len),
        Family::Mixed => match rng.below(4) {
            0 => gen_shuffle(&mut rng, len),
            1 => gen_runs(&mut rng, len),
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
pub struct ResetSpinorItem {
    /// Input symbols, one of [`TURN_I`] / [`TURN_J`] / [`RESET`].
    pub symbols: Vec<usize>,
    /// Per-position target class, in `0..`[`NUM_CLASSES`].
    pub targets: Vec<i64>,
}

/// An in-memory dataset of generated [`ResetSpinorItem`]s.
pub struct ResetSpinorDataset {
    dataset: InMemDataset<ResetSpinorItem>,
}

impl ResetSpinorDataset {
    /// Generate `num_sequences` sequences of one family, seeded deterministically.
    pub fn new(num_sequences: usize, seq_length: usize, family: Family, seed: u64) -> Self {
        let mut state = seed;
        let items = (0..num_sequences)
            .map(|_| {
                let symbols = generate(family, &mut state, seq_length);
                let targets = labels(&symbols);
                ResetSpinorItem { symbols, targets }
            })
            .collect();
        Self {
            dataset: InMemDataset::new(items),
        }
    }
}

impl Dataset<ResetSpinorItem> for ResetSpinorDataset {
    fn get(&self, index: usize) -> Result<ResetSpinorItem, DatasetError> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`ResetSpinorItem`]s into a [`ResetSpinorBatch`], one-hotting the
/// symbols.
#[derive(Clone, Debug, Default)]
pub struct ResetSpinorBatcher {}

/// A batch of one-hot symbol sequences and their per-position target classes.
#[derive(Clone, Debug)]
pub struct ResetSpinorBatch {
    /// One-hot input symbol at each position, `[batch, seq, NUM_SYMBOLS]`.
    pub inputs: Tensor<3>,
    /// Per-position target class, `[batch, seq]`.
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

impl Batcher<ResetSpinorItem, ResetSpinorBatch> for ResetSpinorBatcher {
    fn batch(&self, items: Vec<ResetSpinorItem>, device: &Device) -> ResetSpinorBatch {
        let inputs: Vec<Tensor<2>> = items
            .iter()
            .map(|item| one_hot(&item.symbols, device))
            .collect();
        let targets: Vec<Tensor<1, Int>> = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_ints(item.targets.as_slice(), device))
            .collect();
        ResetSpinorBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
