//! The reset-swap dataset: a three-symbol stream. The target at each position
//! is the **running permutation of three items** since the last reset: the
//! word problem in the symmetric group `S₃`.
//!
//! `l` swaps the left pair of positions, `r` swaps the right pair, and `R`
//! restores the original order:
//!
//! ```text
//!   symbols   R   l   r   l   r   R   r   l   l
//!   order    abc bac bca cba cab abc acb cab acb
//!   target    0   2   4   5   3   0   1   3   1
//! ```
//!
//! The classes are `abc, acb, bac, cab, bca, cba`, in that order (see
//! [`PERMS`]).
//!
//! `S₃` is the **smallest non-abelian group**. So, as in `reset-spinor`, the
//! number of `l`s and `r`s never decides the answer: `lr ≠ rl`. A second
//! property makes it the *next* rung, and `Q₈` does not have it:
//!
//! > `S₃` has **three** elements of order two (the three swaps). Every finite
//! > subgroup of `SU(2)` has exactly **one** (`−1`, the unique element of order
//! > 2 in the unit quaternions).
//!
//! So `S₃` does not embed in `SU(2)` at all. A left-isoclinic
//! ([`Quaternion4D`](burn_mamba::prelude::RotationKind::Quaternion4D)) state can
//! carry only the **double cover** `2D₃` (order 12): the element *and* a
//! spurious sign. The two lifts `±W` of one permutation are then **antipodal**
//! state vectors, and no linear readout can merge them. A two-sided
//! ([`Rotor4D`](burn_mamba::prelude::RotationKind::Rotor4D)) block reaches
//! `SO(3) ⊂ SO(4)` by conjugation `v ↦ q v q̄`. There, `±q` act *identically*,
//! and the three swaps are three true half-turns about three different axes.
//! The group itself is then the state.

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{Dataset, DatasetError, InMemDataset},
};
use burn::prelude::*;
use burn::tensor::Int;
use serde::{Deserialize, Serialize};

/// Input symbol `l`: swap the left pair of positions (the transposition
/// `(0 1)`).
pub const SWAP_L: usize = 0;
/// Input symbol `r`: swap the right pair of positions (the transposition
/// `(1 2)`).
pub const SWAP_R: usize = 1;
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

/// The six permutations of three items, in class-index order. `PERMS[c][x]` is
/// the position of item `x`. So class 3 (`[1, 2, 0]`) puts `a` at position 1,
/// `b` at 2 and `c` at 0: it reads `cab`.
///
/// A swap `τ` of two positions moves the item at position `p` to `τ(p)`. So
/// it maps the array `PERMS[c]` to `τ ∘ PERMS[c]`, and [`labels`] composes
/// each new swap on the left.
pub const PERMS: [[usize; 3]; NUM_CLASSES] = [
    [0, 1, 2], // 0: abc, the identity
    [0, 2, 1], // 1: acb, the swap `r`
    [1, 0, 2], // 2: bac, the swap `l`
    [1, 2, 0], // 3: cab, a 3-cycle (`l∘r`: `r`, then `l`)
    [2, 0, 1], // 4: bca, the other 3-cycle (`r∘l`: `l`, then `r`)
    [2, 1, 0], // 5: cba, the third swap (`l∘r∘l`)
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

/// The permutation that a symbol applies. `RESET` returns the identity, but it
/// *replaces* the state instead of composing with it (see [`labels`]).
pub fn symbol_perm(symbol: usize) -> [usize; 3] {
    match symbol {
        SWAP_L => PERMS[2],
        SWAP_R => PERMS[1],
        RESET => PERMS[0],
        _ => panic!("symbol out of alphabet: {symbol}"),
    }
}

/// The per-position targets of a symbol sequence: the running composition
/// since the last `RESET`, as a class index.
///
/// The newest swap composes **on the left** of the item-to-position array (see
/// [`PERMS`]), like the cumulative rotation of the block
/// (`Pₜ = qₜ ⊗ ⋯ ⊗ q₁`).
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

/// The number of `l`s and `r`s since the last reset, at every position.
///
/// This is everything that an **abelian** transition can carry. The parity
/// `(#l + #r) mod 2` is the sign character, so the counts also bound
/// everything that a left-isoclinic transition can carry *linearly*. The only
/// nontrivial homomorphism from `S₃` into `SU(2)` sends the odd permutations
/// to `−1` and the even ones to `1`. So the state is `±1` times a constant.
/// See the module docs.
pub fn counts_since_reset(symbols: &[usize]) -> Vec<(i64, i64)> {
    let (mut a, mut b) = (0, 0);
    symbols
        .iter()
        .map(|&s| {
            match s {
                SWAP_L => a += 1,
                SWAP_R => b += 1,
                RESET => (a, b) = (0, 0),
                _ => panic!("symbol out of alphabet: {s}"),
            }
            (a, b)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The group as rotations: what the state of the block holds
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
/// A transposition is an order-2 element, so it must be a **half-turn**. Two
/// half-turns about axes `θ` apart compose to a rotation by `2θ`. `l∘r` has
/// order 3, so the axes are `60°` apart. That is the whole embedding
/// `S₃ ≅ D₃ ⊂ SO(3)`.
pub fn swap_axis(symbol: usize) -> [f64; 3] {
    const H: f64 = 0.866_025_403_784_438_6; // sin 60°
    match symbol {
        SWAP_L => [1.0, 0.0, 0.0],
        SWAP_R => [0.5, H, 0.0],
        RESET => [0.0, 0.0, 0.0],
        _ => panic!("symbol out of alphabet: {symbol}"),
    }
}

/// The unit quaternion that lifts the rotation of a symbol. A half-turn about
/// [`swap_axis`] is the **pure** quaternion `(0, û)`, and `RESET` is the
/// identity `(1, 0, 0, 0)`.
///
/// Note that `(0, û)² = −1`, not `1`: the lift of a swap has order **four**.
/// That is the double cover, and it is why a left-isoclinic state cannot be
/// the group.
pub fn symbol_quat(symbol: usize) -> [f64; 4] {
    let u = swap_axis(symbol);
    match symbol {
        RESET => [1.0, 0.0, 0.0, 0.0],
        _ => [0.0, u[0], u[1], u[2]],
    }
}

/// A word for each class, newest factor first: the symbols whose composition
/// (by left multiplication) reaches that permutation from the identity.
const WORDS: [&[usize]; NUM_CLASSES] = [
    &[],
    &[SWAP_R],
    &[SWAP_L],
    &[SWAP_L, SWAP_R],
    &[SWAP_R, SWAP_L],
    &[SWAP_L, SWAP_R, SWAP_L],
];

/// The reference vector that the construction writes into the state. It is a
/// point in the imaginary 3-space of the rotation, and its orbit under the
/// group is **six distinct points** (it lies on none of the axes of the
/// group).
pub const REF_POINT: [f64; 4] = [0.0, 1.0, 1.0, 1.0];

/// Where the class `class` carries [`REF_POINT`]: `W v W*` for the lift `W` of
/// the element.
///
/// The two heads of the block read the `x` and `y` of these six vectors. In
/// `tests.rs`, those shadows are the columns of the classifier head:
/// `logit_g ∝ ⟨p, plane_point(point(g))⟩` is a nearest-point decoder. It is
/// exact because the six shadows lie on one circle, in six distinct
/// directions. Both lifts `±W` give the *same* point. Conjugation gives this,
/// and left multiplication does not.
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
    /// Independent symbols after the first `RESET`: `RESET` with probability
    /// ⅛, `l` ⅜ and `r` ½. Resets are frequent, so many positions carry a
    /// **short** word, and the symbol counts alone often determine a short
    /// word.
    Random,
    /// One reset, then a shuffled bag of `l`s and `r`s in equal numbers (one
    /// extra `l` when the tail length is odd). The construction fixes the
    /// counts, and only the **order** varies.
    Shuffle,
    /// One reset, then alternating runs of 3 to 8 copies of one symbol. A run
    /// is nearly wasted motion (`l² = 1`, so a run only alternates between two
    /// elements). So in this family, the counts come closest to determining the
    /// answer.
    Runs,
    /// The training mixture: half [`Self::Random`], a quarter of each of the
    /// other two.
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
    let mut out = vec![RESET];
    out.extend((1..len).map(|_| match rng.below(8) {
        0 => RESET,
        n if n % 2 == 0 => SWAP_L,
        _ => SWAP_R,
    }));
    out
}

fn gen_shuffle(rng: &mut Lcg, len: usize) -> Vec<usize> {
    let tail = len - 1;
    let mut bag: Vec<usize> = (0..tail)
        .map(|n| if n * 2 < tail { SWAP_L } else { SWAP_R })
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
    let mut symbol = if rng.below(2) == 0 { SWAP_L } else { SWAP_R };
    while out.len() < len {
        let run = 3 + rng.below(6);
        for _ in 0..run.min(len - out.len()) {
            out.push(symbol);
        }
        symbol = if symbol == SWAP_L { SWAP_R } else { SWAP_L };
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
    /// Input symbols, one of [`SWAP_L`] / [`SWAP_R`] / [`RESET`].
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

impl ResetSwapBatch {
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
