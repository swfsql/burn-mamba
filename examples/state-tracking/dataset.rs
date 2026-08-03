//! The `A₅` word-problem dataset, in the **grokking protocol**: all
//! `NUM_GENERATORS^SEQ_LEN` generator words are enumerated, deterministically
//! split into train/test **by word** with a seeded `ChaCha8Rng` (stable across
//! platforms and `rand` versions), and the supervised target is **only the
//! final running product** — a `NUM_CLASSES`-way classification read at the
//! last position, exactly like the modular-addition example (mod-p addition
//! *is* this task on the cyclic group; here the group is the non-solvable
//! `A₅`, the smallest group whose word problem is `NC¹`-complete by
//! Barrington's theorem).
//!
//! The per-position running products are still generated, but as an **eval
//! probe** (how deep does composition hold), never as supervision.
//!
//! ## Vocabulary layout
//!
//! One shared vocabulary serves both sides: ids `0..NUM_GENERATORS` are the
//! generator symbols, [`ANCHOR_SYMBOL`] is the leading anchor token, and the
//! `A₅` element classes live at `CLASS_BASE..VOCAB_SIZE` (they never occur as
//! inputs). Keeping the input symbols first lets the shared weight
//! diagnostics read the input-alphabet embedding rows directly.
//!
//! ## Why the anchor token
//!
//! The Mamba-3 rotation rotates both `B` and `C`, so the SSD readout at
//! position `t` for a key at position `i` sees only the **relative** rotation
//! `Rₜ⋯Rᵢ₊₁ = Pₜ Pᵢ⁻¹` (RoPE-style), never the **absolute** running product
//! `Pₜ = Rₜ⋯R₁` the task asks for. A fixed anchor symbol at position 0
//! (rotation learned to identity, `P₀ = I`) anchors the readout: its
//! contribution `Cₜᵀ Pₜ B₀ x₀` carries the absolute product.

use burn::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

/// Number of generator symbols (a 5-cycle and a 3-cycle) that generate `A₅`.
pub const NUM_GENERATORS: usize = 2;
/// Input alphabet size: the generators plus the leading anchor token.
pub const NUM_SYMBOLS: usize = NUM_GENERATORS + 1;
/// Token id of the anchor (BOS / identity-reference) symbol.
pub const ANCHOR_SYMBOL: usize = NUM_GENERATORS;
/// Number of `A₅` elements, i.e. the number of output classes.
pub const NUM_CLASSES: usize = 60;
/// Vocabulary id of `A₅` class `c` is `CLASS_BASE + c`.
pub const CLASS_BASE: usize = NUM_SYMBOLS;
/// One shared vocabulary: input symbols first, then the element classes.
pub const VOCAB_SIZE: usize = NUM_SYMBOLS + NUM_CLASSES;

/// Hard cap on `NUM_GENERATORS^seq_len` — everything is enumerated in memory.
const ENUMERATION_CAP: usize = 1 << 20;

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

/// `5!` permutations of `[0,1,2,3,4]`, keeping the even ones (sign `+1`),
/// sorted so the class indices are stable across runs.
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

/// Class indexer over [`even_permutations`], shared by the split builders.
struct Group {
    generators: [[usize; 5]; NUM_GENERATORS],
    index_of: HashMap<[usize; 5], usize>,
    identity_class: i32,
}

impl Group {
    fn new() -> Self {
        let perms = even_permutations();
        assert_eq!(perms.len(), NUM_CLASSES, "A₅ has 60 elements");
        let index_of: HashMap<[usize; 5], usize> =
            perms.iter().enumerate().map(|(i, p)| (*p, i)).collect();
        let identity = [0usize, 1, 2, 3, 4];
        let identity_class = index_of[&identity] as i32;
        Group { generators: generators(), index_of, identity_class }
    }

    /// Materialize word `index ∈ [0, g^seq_len)` (its base-`g` digits,
    /// most-significant first) via [`Group::push_digits`].
    fn push_word(
        &self,
        index: usize,
        seq_len: usize,
        seqs: &mut Vec<i32>,
        pos_targets: &mut Vec<i32>,
    ) -> i32 {
        // digits, most-significant first (as the grokking dataset does)
        let mut digits = vec![0usize; seq_len];
        let mut rem = index;
        for j in (0..seq_len).rev() {
            digits[j] = rem % NUM_GENERATORS;
            rem /= NUM_GENERATORS;
        }
        self.push_digits(&digits, seqs, pos_targets)
    }

    /// Materialize a generator word as anchor-led token ids plus the
    /// per-position running-product classes, appended to the flat buffers;
    /// returns the final-product class.
    fn push_digits(
        &self,
        digits: &[usize],
        seqs: &mut Vec<i32>,
        pos_targets: &mut Vec<i32>,
    ) -> i32 {
        seqs.push(ANCHOR_SYMBOL as i32);
        pos_targets.push(self.identity_class);
        let mut state = [0usize, 1, 2, 3, 4];
        let mut label = self.identity_class;
        for &g in digits {
            state = compose(&self.generators[g], &state); // Pₜ = g ∘ Pₜ₋₁
            label = self.index_of[&state] as i32;
            seqs.push(g as i32);
            pos_targets.push(label);
        }
        label
    }
}

// ---------------------------------------------------------------------------
// Splits (full-batch tensors, as in the grokking example)
// ---------------------------------------------------------------------------

/// One side of the train/test split: `n` anchor-led token sequences, their
/// final-product labels, and the per-position running products (eval probe).
pub struct Split {
    /// Generators per word; the token sequence is one longer (the anchor).
    pub seq_len: usize,
    /// Token ids, row-major flat `[n · (seq_len + 1)]`.
    pub seqs: Vec<i32>,
    /// Final running-product class (`0..NUM_CLASSES`), aligned with rows.
    pub labels: Vec<i32>,
    /// Running-product class at every position, flat `[n · (seq_len + 1)]`.
    pub pos_targets: Vec<i32>,
}

impl Split {
    /// Number of examples.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether the split holds no examples.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Tokens per sequence (`seq_len + 1`, the anchor included).
    pub fn tokens(&self) -> usize {
        self.seq_len + 1
    }

    /// The first `n` examples as their own split (for sample displays).
    pub fn head(&self, n: usize) -> Split {
        let n = n.min(self.len());
        let s = self.tokens();
        Split {
            seq_len: self.seq_len,
            seqs: self.seqs[..n * s].to_vec(),
            labels: self.labels[..n].to_vec(),
            pos_targets: self.pos_targets[..n * s].to_vec(),
        }
    }

    /// Token ids as an Int tensor `[n, seq_len + 1]`.
    pub fn inputs_tensor(&self, device: &Device) -> Tensor<2, Int> {
        Tensor::<1, Int>::from_ints(self.seqs.as_slice(), device)
            .reshape([self.len(), self.tokens()])
    }

    /// Final-product class labels as an Int tensor `[n]` (`0..NUM_CLASSES`).
    pub fn labels_tensor(&self, device: &Device) -> Tensor<1, Int> {
        Tensor::from_ints(self.labels.as_slice(), device)
    }

    /// Per-position running-product classes as an Int tensor
    /// `[n, seq_len + 1]` (`0..NUM_CLASSES`) — the depth probe's targets.
    pub fn pos_targets_tensor(&self, device: &Device) -> Tensor<2, Int> {
        Tensor::<1, Int>::from_ints(self.pos_targets.as_slice(), device)
            .reshape([self.len(), self.tokens()])
    }

    /// One-hot float targets `[n, VOCAB_SIZE]` for the cross-entropy loss:
    /// mass at vocab id `CLASS_BASE + label` (the class region; the model
    /// learns to suppress the symbol logits).
    pub fn targets_tensor(&self, device: &Device) -> Tensor<2> {
        let mut flat = vec![0.0f32; self.len() * VOCAB_SIZE];
        for (i, &label) in self.labels.iter().enumerate() {
            flat[i * VOCAB_SIZE + CLASS_BASE + label as usize] = 1.0;
        }
        Tensor::<1>::from_floats(flat.as_slice(), device).reshape([self.len(), VOCAB_SIZE])
    }

    /// Dense one-hot targets `[n · (seq_len + 1), VOCAB_SIZE]` over **every**
    /// position (row-major over `(word, position)`) — the frontier mode's
    /// per-position loss targets, mass at `CLASS_BASE + running class`.
    pub fn pos_targets_onehot(&self, device: &Device) -> Tensor<2> {
        let rows = self.pos_targets.len();
        let mut flat = vec![0.0f32; rows * VOCAB_SIZE];
        for (i, &class) in self.pos_targets.iter().enumerate() {
            flat[i * VOCAB_SIZE + CLASS_BASE + class as usize] = 1.0;
        }
        Tensor::<1>::from_floats(flat.as_slice(), device).reshape([rows, VOCAB_SIZE])
    }
}

/// `NUM_GENERATORS^seq_len`, asserting the enumeration cap.
fn space_size(seq_len: usize) -> usize {
    let total = NUM_GENERATORS
        .checked_pow(seq_len as u32)
        .expect("g^seq_len overflows usize");
    assert!(
        total <= ENUMERATION_CAP,
        "g^seq_len = {total} exceeds the enumeration cap ({ENUMERATION_CAP}); pick a smaller seq_len"
    );
    total
}

/// Enumerate all `NUM_GENERATORS^seq_len` words, shuffle them with
/// `ChaCha8Rng(split_seed)`, and return `(train, test)` where train takes the
/// first `round(train_fraction·total)` words (the splits are disjoint by word).
pub fn build(seq_len: usize, train_fraction: f64, split_seed: u64) -> (Split, Split) {
    let total = space_size(seq_len);
    let mut indices: Vec<usize> = (0..total).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(split_seed);
    indices.shuffle(&mut rng);

    let n_train = (total as f64 * train_fraction).round() as usize;
    assert!(
        n_train >= 1 && n_train < total,
        "train_fraction {train_fraction} must leave both splits non-empty"
    );
    let group = Group::new();
    let mut splits = [
        Split { seq_len, seqs: Vec::new(), labels: Vec::new(), pos_targets: Vec::new() },
        Split { seq_len, seqs: Vec::new(), labels: Vec::new(), pos_targets: Vec::new() },
    ];
    for (i, &index) in indices.iter().enumerate() {
        let split = &mut splits[usize::from(i >= n_train)];
        let label = group.push_word(index, seq_len, &mut split.seqs, &mut split.pos_targets);
        split.labels.push(label);
    }
    let [train, test] = splits;
    (train, test)
}

/// `num` freshly sampled random words of length `seq_len` (uniform i.i.d.
/// generators, seeded) — the **frontier mode**'s data source: every batch is
/// new, so memorization is impossible and no train/test split exists; the
/// enumeration cap does not apply.
pub fn sample_split(num: usize, seq_len: usize, seed: u64) -> Split {
    use rand::Rng as _;
    let group = Group::new();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut split = Split { seq_len, seqs: Vec::new(), labels: Vec::new(), pos_targets: Vec::new() };
    let mut digits = vec![0usize; seq_len];
    for _ in 0..num {
        for d in digits.iter_mut() {
            *d = rng.random_range(0..NUM_GENERATORS);
        }
        let label = group.push_digits(&digits, &mut split.seqs, &mut split.pos_targets);
        split.labels.push(label);
    }
    split
}

/// The diagnostic eval set: all words when they fit in `max_n` (the PR
/// estimator wants everything), otherwise a deterministic `ChaCha8Rng(seed)`
/// sample of `max_n` distinct words.
pub fn diagnostic_set(seq_len: usize, max_n: usize, seed: u64) -> Split {
    let total = space_size(seq_len);
    let group = Group::new();
    let mut split = Split { seq_len, seqs: Vec::new(), labels: Vec::new(), pos_targets: Vec::new() };
    let mut push = |index: usize| {
        let label = group.push_word(index, seq_len, &mut split.seqs, &mut split.pos_targets);
        split.labels.push(label);
    };
    if total <= max_n {
        (0..total).for_each(&mut push);
    } else {
        let mut indices: Vec<usize> = (0..total).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        indices.iter().take(max_n).for_each(|&i| push(i));
    }
    split
}
