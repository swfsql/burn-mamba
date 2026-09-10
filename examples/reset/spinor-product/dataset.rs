//! The `spinor-product` dataset: `reset-spinor`'s `Q₈` stream, read **two
//! symbols per token**.
//!
//! Every token carries an ordered *pair* of symbols from `i` / `j` / `k` /
//! `.` (hold) / `R` (reset); the per-token target is the running product in the
//! quaternion group `Q₈` after **both** of them have been applied, newest factor
//! on the left:
//!
//! ```text
//!   token       R.      ij      .k      jk      ii      k.
//!   state      1  1    i -k   -k  1    j -i    1  i    j  j
//!   target        1       -k       1      -i       i       j
//! ```
//!
//! (`state` is the group element after each of the token's two symbols; the
//! target is the second of them.)
//!
//! The stream is [`reset-spinor`](../README.md#reset-spinor)'s, over the three units
//! rather than two and with a hold, and the same length in *symbols* — only the
//! packing changes. That is the whole example: what a token asks the recurrence
//! for is now a **product of two group elements**, and a Mamba-3 step applies
//! **one** rotation whose generator is an affine function of the token. At
//! `micro_steps = 2` a token is two steps and the product is exact; at
//! `micro_steps = 1` the two generators can only **add**, and `exp(v + w)` is
//! not `exp(w)·exp(v)` unless they commute.
//!
//! Two properties of the alphabet make that airtight rather than merely likely:
//!
//! - the **hold** lets a token carry one turn alone, which pins each slot's
//!   generator to the axis of the unit it turns by (`±x̂`, `±ŷ`, `±ẑ`);
//! - the **third unit** keeps the two-turn tokens non-abelian. Over `i`/`j`
//!   alone every two-turn token composes into `⟨k⟩ ≅ Z₄`, which commutes — one
//!   rotation per token would then be enough, and a `micro_steps = 1` model
//!   solves the task.
//!
//! Together they leave a token like `(i, j)` asking for a turn about `ẑ` from a
//! generator confined to the `xy`-plane. See `tests.rs`
//! (`one_step_generators_add_and_cannot_reach_k`).

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
/// Input symbol: multiply the state by `k` (on the left).
///
/// The **third** unit is what keeps a token's pair non-abelian: with two turn
/// symbols every two-turn token composes into `⟨k⟩ ≅ Z₄`, which commutes, and
/// one rotation per token would be enough. With three, `(i,j) ↦ k` and
/// `(i,k) ↦ j` do not commute, so composing the pair is the whole job.
pub const TURN_K: usize = 2;
/// Input symbol: do nothing — the group's identity, and the reason a single
/// step provably cannot compose a token (see the module docs).
pub const HOLD: usize = 3;
/// Input symbol: reset the state to `1` (the selective-forget symbol).
pub const RESET: usize = 4;
/// Input alphabet size, per slot.
pub const NUM_SYMBOLS: usize = 5;
/// Symbols per token — the dial this example is about, mirrored in
/// `Mamba3Config::micro_steps`.
pub const PAIR: usize = 2;
/// Width of one input token: two one-hot slots.
pub const INPUT_SIZE: usize = PAIR * NUM_SYMBOLS;

/// Number of output classes: the eight elements of `Q₈`.
pub const NUM_CLASSES: usize = 8;

/// Class index of the identity element `1`.
pub const IDENTITY: i64 = 0;

/// Length of every **training** sequence, in tokens (so `2 · SEQ_LENGTH`
/// symbols).
///
/// Long on purpose: a block that composes each token exactly holds the word for
/// as long as you like, while one that only approximates the composition
/// compounds its error with every token. At a handful of tokens an
/// approximation still scores well; over a word this long it does not.
pub const SEQ_LENGTH: usize = 32;

/// The lengths inference reports, in tokens: the trained one, and three times
/// it.
///
/// A block that composes each token exactly is a group tracker and does not
/// care; one that approximates the composition compounds its error with every
/// token, so the long column is where an approximation separates from a
/// solution. Nothing about the model is length-specific — the recurrence just
/// runs longer.
pub const EVAL_LENGTHS: [usize; 2] = [SEQ_LENGTH, 3 * SEQ_LENGTH];
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

/// Apply one symbol to a class index.
pub fn apply(symbol: usize, state: i64) -> i64 {
    match symbol {
        TURN_I => left_mul(1, state),
        TURN_J => left_mul(2, state),
        TURN_K => left_mul(3, state),
        HOLD => state,
        RESET => IDENTITY,
        _ => panic!("symbol out of alphabet: {symbol}"),
    }
}

/// The three turn symbols, in the order their unit quaternions index.
pub const TURNS: [usize; 3] = [TURN_I, TURN_J, TURN_K];

/// The per-**token** targets implied by a symbol stream: the running product
/// since the last [`RESET`], sampled after every second symbol.
pub fn labels(symbols: &[usize]) -> Vec<i64> {
    assert_eq!(0, symbols.len() % PAIR, "the stream is read in pairs");
    let mut state = IDENTITY;
    symbols
        .chunks_exact(PAIR)
        .map(|pair| {
            for &s in pair {
                state = apply(s, state);
            }
            state
        })
        .collect()
}

/// How many `i`s, `j`s and `k`s have gone by since the last reset, at the end of
/// every token.
///
/// This is everything an order-blind model can key on, and the ceiling
/// `tests.rs` computes over: `Q₈`'s commutator subgroup is `{±1}`, so the counts
/// pin the element down to a sign and no further.
pub fn counts_since_reset(symbols: &[usize]) -> Vec<[i64; 3]> {
    let mut counts = [0i64; 3];
    symbols
        .chunks_exact(PAIR)
        .map(|pair| {
            for &s in pair {
                match s {
                    TURN_I | TURN_J | TURN_K => counts[s] += 1,
                    HOLD => (),
                    RESET => counts = [0; 3],
                    _ => panic!("symbol out of alphabet: {s}"),
                }
            }
            counts
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Which generator a split draws from. Every sequence opens with a [`RESET`].
///
/// Every family mixes **holds** into the word, so single-turn tokens (`(i, .)`)
/// occur beside two-turn ones (`(i, j)`). That mixture is deliberate: the holds
/// pin each slot's rotation generator to the axis of the unit it turns by, and
/// the two-turn tokens then ask for a composition no sum of those axes can
/// reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Independent symbols: `RESET` with probability ~1/16, `HOLD` ~¼,
    /// otherwise `i` / `j` / `k` evenly. Resets still happen inside the
    /// sequence, so this is the family with the **shortest** words — and a short
    /// word is the one a decaying trace can carry without any group structure.
    Random,
    /// One reset, then a shuffled bag: equally many `i`s, `j`s and `k`s plus a
    /// quarter holds. The counts are fixed by construction and the word runs the
    /// whole sequence, so only the **order** varies. This is the family with
    /// nothing in it but composition.
    Shuffle,
    /// One reset, then runs of one symbol, holds included (`i…i . . k…k`). The
    /// word is still non-commutative, but grouping it into blocks is the case
    /// where the counts come closest to determining the product.
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
    out.extend((1..len).map(|_| match rng.below(16) {
        0 => RESET,
        1..=4 => HOLD,
        n => TURNS[n as usize % 3],
    }));
    out
}

fn gen_shuffle(rng: &mut Lcg, len: usize) -> Vec<usize> {
    let tail = len - 1;
    let mut bag: Vec<usize> = (0..tail)
        .map(|n| if n % 4 == 3 { HOLD } else { TURNS[n % 3] })
        .collect();
    for n in (1..bag.len()).rev() {
        bag.swap(n, rng.below(n + 1));
    }
    let mut out = vec![RESET];
    out.extend(bag);
    out
}

fn gen_runs(rng: &mut Lcg, len: usize) -> Vec<usize> {
    const BLOCKS: [usize; 4] = [TURN_I, TURN_J, TURN_K, HOLD];
    let mut out = vec![RESET];
    let mut block = rng.below(BLOCKS.len());
    while out.len() < len {
        let run = 2 + rng.below(5);
        for _ in 0..run.min(len - out.len()) {
            out.push(BLOCKS[block]);
        }
        // a different block each time, so a run never merges with the next
        block = (block + 1 + rng.below(BLOCKS.len() - 1)) % BLOCKS.len();
    }
    out
}

/// Generate one symbol stream (`PAIR · tokens` symbols) of the given family.
pub fn generate(family: Family, rng_state: &mut u64, tokens: usize) -> Vec<usize> {
    let len = tokens * PAIR;
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

/// One generated stream and its per-token target class.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductItem {
    /// The symbol stream, `PAIR · tokens` long, read two per token.
    pub symbols: Vec<usize>,
    /// Per-token target class, in `0..`[`NUM_CLASSES`].
    pub targets: Vec<i64>,
}

/// An in-memory dataset of generated [`ProductItem`]s.
pub struct ProductDataset {
    dataset: InMemDataset<ProductItem>,
}

impl ProductDataset {
    /// Generate `num_sequences` sequences of one family, seeded deterministically.
    pub fn new(num_sequences: usize, tokens: usize, family: Family, seed: u64) -> Self {
        let mut state = seed;
        let items = (0..num_sequences)
            .map(|_| {
                let symbols = generate(family, &mut state, tokens);
                let targets = labels(&symbols);
                ProductItem { symbols, targets }
            })
            .collect();
        Self {
            dataset: InMemDataset::new(items),
        }
    }
}

impl Dataset<ProductItem> for ProductDataset {
    fn get(&self, index: usize) -> Result<ProductItem, DatasetError> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`ProductItem`]s into a [`ProductBatch`], two-hotting each token.
#[derive(Clone, Debug, Default)]
pub struct ProductBatcher {}

/// A batch of two-hot token sequences and their per-token target classes.
#[derive(Clone, Debug)]
pub struct ProductBatch {
    /// Two-hot input token at each position, `[batch, tokens, INPUT_SIZE]`.
    pub inputs: Tensor<3>,
    /// Per-token target class, `[batch, tokens]`.
    pub targets: Tensor<2, Int>,
}

/// Two-hot encode a symbol stream into `[tokens, INPUT_SIZE]`: one one-hot slot
/// per symbol of the pair, concatenated.
pub fn two_hot(symbols: &[usize], device: &Device) -> Tensor<2> {
    let tokens = symbols.len() / PAIR;
    let mut buf = vec![0.0f32; tokens * INPUT_SIZE];
    for (t, pair) in symbols.chunks_exact(PAIR).enumerate() {
        for (slot, &s) in pair.iter().enumerate() {
            buf[t * INPUT_SIZE + slot * NUM_SYMBOLS + s] = 1.0;
        }
    }
    Tensor::<1>::from_floats(buf.as_slice(), device).reshape([tokens, INPUT_SIZE])
}

impl Batcher<ProductItem, ProductBatch> for ProductBatcher {
    fn batch(&self, items: Vec<ProductItem>, device: &Device) -> ProductBatch {
        let inputs: Vec<Tensor<2>> = items
            .iter()
            .map(|item| two_hot(&item.symbols, device))
            .collect();
        let targets: Vec<Tensor<1, Int>> = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_ints(item.targets.as_slice(), device))
            .collect();
        ProductBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
