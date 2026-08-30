//! The reset-swap dataset: a three-symbol stream whose per-position target is
//! the **running permutation of three items** since the last reset — the word
//! problem in the symmetric group `S₃`.
//!
//! `s` swaps the first two items, `t` swaps the last two, and `R` restores the
//! original order:
//!
//! ```text
//!   symbols   R   s   t   s   t   R   t   s   s
//!   order    abc bac bca cba acb abc acb cab bca
//!   target    0   2   3   5   1   0   1   4   3
//! ```
//!
//! `S₃` is the **smallest non-abelian group**, so — exactly as in `reset-spinor`
//! — how many `s`s and `t`s went by never decides the answer: `st ≠ ts`. What
//! makes it the *next* rung is a second property, which `Q₈` does not have:
//!
//! > `S₃` has **three** elements of order two (the three swaps), and every
//! > finite subgroup of `SU(2)` has exactly **one** (`−1`, the unique element of
//! > order 2 in the unit quaternions).
//!
//! So `S₃` does not embed in `SU(2)` at all. A left-isoclinic
//! ([`Quaternion4D`](burn_mamba::prelude::RotationKind::Quaternion4D)) state can
//! only carry the **double cover** `2D₃` (order 12) — the element *and* a
//! spurious sign — where the two lifts `±W` of one permutation are **antipodal**
//! state vectors that no linear readout can merge. Two-sided
//! ([`Rotor4D`](burn_mamba::prelude::RotationKind::Rotor4D)) the block reaches
//! `SO(3) ⊂ SO(4)` by conjugation `v ↦ q v q̄`, where `±q` act *identically* and
//! the three swaps are three honest half-turns about three different axes. The
//! group itself is then the state.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, DatasetError, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};

/// Input symbol: swap the first two items (the transposition `(0 1)`).
pub const SWAP_S: usize = 0;
/// Input symbol: swap the last two items (the transposition `(1 2)`).
pub const SWAP_T: usize = 1;
/// Input symbol: restore the original order (the selective-forget token).
pub const RESET: usize = 2;
/// Input alphabet size.
pub const NUM_SYMBOLS: usize = 3;

/// Number of output classes: the six elements of `S₃`.
pub const NUM_CLASSES: usize = 6;

/// Class index of the identity permutation `abc`.
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

/// The six permutations of three items, in class-index order. `PERMS[c][i]` is
/// the item sitting at position `i`, so class 2 (`[1, 0, 2]`) reads `bac`.
pub const PERMS: [[usize; 3]; NUM_CLASSES] = [
    [0, 1, 2], // 0: abc — identity
    [0, 2, 1], // 1: acb — the swap `t`
    [1, 0, 2], // 2: bac — the swap `s`
    [1, 2, 0], // 3: bca — a 3-cycle (`s∘t`)
    [2, 0, 1], // 4: cab — the other 3-cycle (`t∘s`)
    [2, 1, 0], // 5: cba — the third swap (`s∘t∘s`)
];

/// Class index of a permutation array.
pub fn class_of(perm: [usize; 3]) -> i64 {
    PERMS
        .iter()
        .position(|p| *p == perm)
        .expect("not a permutation of three items") as i64
}

/// Compose two permutations: `compose(p, q)` applies `q` first, then `p`.
pub fn compose(p: [usize; 3], q: [usize; 3]) -> [usize; 3] {
    [p[q[0]], p[q[1]], p[q[2]]]
}

/// The permutation a symbol applies. `RESET` returns the identity, but note it
/// *replaces* the state rather than composing with it (see [`labels`]).
pub fn symbol_perm(symbol: usize) -> [usize; 3] {
    match symbol {
        SWAP_S => PERMS[2],
        SWAP_T => PERMS[1],
        RESET => PERMS[0],
        _ => panic!("symbol out of alphabet: {symbol}"),
    }
}

/// The per-position targets implied by a symbol sequence: the running
/// composition since the last `RESET`, as a class index.
///
/// The newest swap composes **on the left**, matching the block's own
/// cumulative rotation (`Pₜ = qₜ ⊗ ⋯ ⊗ q₁`).
pub fn labels(symbols: &[usize]) -> Vec<i64> {
    let mut state = PERMS[IDENTITY as usize];
    symbols
        .iter()
        .map(|&s| {
            state = match s {
                RESET => PERMS[IDENTITY as usize],
                turn => compose(symbol_perm(turn), state),
            };
            class_of(state)
        })
        .collect()
}

/// How many `s`s and `t`s have gone by since the last reset, at every position.
///
/// This is everything an **abelian** transition can carry, and — because the
/// parity `(#s + #t) mod 2` is the sign character — it also bounds everything a
/// left-isoclinic one can carry *linearly*: the only homomorphism from `S₃` into
/// `SU(2)` sends the odd permutations to `−1` and nothing else, so the state is
/// `±1` times a constant. See the module docs.
pub fn counts_since_reset(symbols: &[usize]) -> Vec<(i64, i64)> {
    let (mut a, mut b) = (0, 0);
    symbols
        .iter()
        .map(|&s| {
            match s {
                SWAP_S => a += 1,
                SWAP_T => b += 1,
                RESET => (a, b) = (0, 0),
                _ => panic!("symbol out of alphabet: {s}"),
            }
            (a, b)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The group as rotations — what the block's state actually holds
// ---------------------------------------------------------------------------

/// Hamilton product of two quaternions `(w, x, y, z)`.
pub fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Quaternion conjugate `q* = (w, −x, −y, −z)`.
pub fn quat_conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// The **axis** each swap turns about, as a unit 3-vector.
///
/// A transposition is an order-2 element, so it must be a **half-turn**; two
/// half-turns about axes `θ` apart compose to a rotation by `2θ`, and `s∘t` has
/// order 3, so the axes sit `60°` apart. That is the whole embedding
/// `S₃ ≅ D₃ ⊂ SO(3)`.
pub fn swap_axis(symbol: usize) -> [f64; 3] {
    const H: f64 = 0.866_025_403_784_438_6; // sin 60°
    match symbol {
        SWAP_S => [1.0, 0.0, 0.0],
        SWAP_T => [0.5, H, 0.0],
        RESET => [0.0, 0.0, 0.0],
        _ => panic!("symbol out of alphabet: {symbol}"),
    }
}

/// The unit quaternion lifting a symbol's rotation: a half-turn about
/// [`swap_axis`] is the **pure** quaternion `(0, û)`, and `RESET` is the
/// identity `(1, 0, 0, 0)`.
///
/// Note `(0, û)² = −1`, not `1`: the lift of a swap has order **four**. That is
/// the double cover, and the reason a left-isoclinic state cannot be the group.
pub fn symbol_quat(symbol: usize) -> [f64; 4] {
    let u = swap_axis(symbol);
    match symbol {
        RESET => [1.0, 0.0, 0.0, 0.0],
        _ => [0.0, u[0], u[1], u[2]],
    }
}

/// A word for each class, newest factor first — the symbols whose composition
/// (left-multiplying) reaches that permutation from the identity.
const WORDS: [&[usize]; NUM_CLASSES] = [
    &[],
    &[SWAP_T],
    &[SWAP_S],
    &[SWAP_S, SWAP_T],
    &[SWAP_T, SWAP_S],
    &[SWAP_S, SWAP_T, SWAP_S],
];

/// The reference vector the construction writes into the state: a point in the
/// rotation's imaginary 3-space whose orbit under the group is **six distinct
/// points** (it lies on none of the group's axes).
pub const REF_POINT: [f64; 4] = [0.0, 1.0, 1.0, 1.0];

/// Where the class `class` carries [`REF_POINT`] — i.e. `W v W*` for the
/// element's lift `W`.
///
/// These six vectors are what the block's four heads read out, and the columns
/// of the example's classifier head: `logit_g = ⟨state, point(g)⟩` is a
/// nearest-point decoder, exact because the orbit is six distinct points and the
/// action is orthogonal. Both lifts `±W` give the *same* point — that is what
/// conjugation buys and left multiplication does not.
pub fn point(class: i64) -> [f64; 4] {
    let w = WORDS[class as usize]
        .iter()
        .fold([1.0, 0.0, 0.0, 0.0], |acc, &s| {
            quat_mul(acc, symbol_quat(s))
        });
    quat_mul(quat_mul(w, REF_POINT), quat_conj(w))
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// Which generator a split draws from. Every family opens with a `RESET`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Independent symbols: `RESET` with probability ~⅛, otherwise `s` / `t`
    /// evenly. Resets are frequent, so many positions carry a **short** word —
    /// and a short word is often pinned down by its symbol counts alone.
    Random,
    /// One reset, then a shuffled bag of equally many `s`s and `t`s: the counts
    /// are fixed by construction and only the **order** varies.
    Shuffle,
    /// One reset, then long runs of one symbol. A run is nearly wasted motion
    /// (`s² = 1`, so a run only alternates), which makes this the family where
    /// the counts come closest to determining the answer.
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
        n if n % 2 == 0 => SWAP_S,
        _ => SWAP_T,
    }));
    out
}

fn gen_shuffle(rng: &mut Lcg, len: usize) -> Vec<usize> {
    let tail = len - 1;
    let mut bag: Vec<usize> = (0..tail)
        .map(|n| if n * 2 < tail { SWAP_S } else { SWAP_T })
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
    let mut symbol = if rng.below(2) == 0 { SWAP_S } else { SWAP_T };
    while out.len() < len {
        let run = 3 + rng.below(6);
        for _ in 0..run.min(len - out.len()) {
            out.push(symbol);
        }
        symbol = if symbol == SWAP_S { SWAP_T } else { SWAP_S };
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
pub struct ResetSwapItem {
    /// Input symbols, one of [`SWAP_S`] / [`SWAP_T`] / [`RESET`].
    pub symbols: Vec<usize>,
    /// Per-position target class, in `0..`[`NUM_CLASSES`].
    pub targets: Vec<i64>,
}

/// An in-memory dataset of generated [`ResetSwapItem`]s.
pub struct ResetSwapDataset {
    dataset: InMemDataset<ResetSwapItem>,
}

impl ResetSwapDataset {
    /// Generate `num_sequences` sequences of one family, seeded deterministically.
    pub fn new(num_sequences: usize, seq_length: usize, family: Family, seed: u64) -> Self {
        let mut state = seed;
        let items = (0..num_sequences)
            .map(|_| {
                let symbols = generate(family, &mut state, seq_length);
                let targets = labels(&symbols);
                ResetSwapItem { symbols, targets }
            })
            .collect();
        Self {
            dataset: InMemDataset::new(items),
        }
    }
}

impl Dataset<ResetSwapItem> for ResetSwapDataset {
    fn get(&self, index: usize) -> Result<ResetSwapItem, DatasetError> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Collates [`ResetSwapItem`]s into a [`ResetSwapBatch`], one-hotting the
/// symbols.
#[derive(Clone, Debug, Default)]
pub struct ResetSwapBatcher {}

/// A batch of one-hot symbol sequences and their per-position target classes.
#[derive(Clone, Debug)]
pub struct ResetSwapBatch {
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

impl Batcher<ResetSwapItem, ResetSwapBatch> for ResetSwapBatcher {
    fn batch(&self, items: Vec<ResetSwapItem>, device: &Device) -> ResetSwapBatch {
        let inputs: Vec<Tensor<2>> = items
            .iter()
            .map(|item| one_hot(&item.symbols, device))
            .collect();
        let targets: Vec<Tensor<1, Int>> = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_ints(item.targets.as_slice(), device))
            .collect();
        ResetSwapBatch {
            inputs: Tensor::stack(inputs, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
