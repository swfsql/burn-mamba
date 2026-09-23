//! The tally-record stream: twelve value symbols `1`…`12` and a reset `R`. The
//! target at each position is **whether this value is a new maximum since the
//! last `R`** (strictly above every earlier value).
//!
//! ```text
//!   symbols   3  1  4  4  2  R  2  5  1  6
//!   max       3  3  4  4  4  -  2  5  5  6
//!   target    +  -  +  -  -  .  +  +  -  +
//! ```
//!
//! A running maximum is `cₜ = max(cₜ₋₁, vₜ)`: the (max, +) recursion with
//! `a = 0`, `b = v`. So one tropical register holds it, whatever the alphabet.
//! A linear state can hold it only as a sum in the exponential domain, and that
//! route runs out of range (see the crate docs). Otherwise, the best that a
//! linear state can do is to compare against a decaying average. [`gen_plateau`]
//! defeats that: it sets the maximum early, and then it never comes near it
//! again.
//!
//! The *width* argument is why the values are read from a circle, not one-hot.
//! A linear state can hold one **latch** per threshold ("was a value ≥ k
//! seen?"), and the per-token gate can select the latch that the current value
//! names. So at `d_model ≥ 12`, a stock block solves this exactly. At
//! `d_model = 3`, it cannot. Every in-projection channel is an affine function
//! of a 3-D embedding. The twelve indicators `[v ≥ k]` are not affine in one,
//! but the `b = S·v` of the register is.

use crate::shared::data::{Generator, IGNORE, Rng};

/// Number of distinct values (`1`…`12`). Symbol `i` carries value `i + 1`.
pub const NUM_VALUES: usize = 12;
/// Input symbol: clear the running maximum.
pub const RESET: usize = NUM_VALUES;
/// Input alphabet size.
pub const NUM_SYMBOLS: usize = NUM_VALUES + 1;

/// Target class: not a new maximum.
pub const HELD: i64 = 0;
/// Target class: a new maximum since the last `R`.
pub const RECORD: i64 = 1;

/// Length of every generated sequence.
pub const SEQ_LENGTH: usize = 64;

/// The value a symbol carries (`RESET` carries none).
pub fn value(symbol: usize) -> Option<i64> {
    (symbol < NUM_VALUES).then(|| symbol as i64 + 1)
}

/// Per-position targets: at every value, whether it exceeds the running maximum
/// since the last reset. [`IGNORE`] at a reset.
pub fn labels(symbols: &[usize]) -> Vec<i64> {
    let mut max = i64::MIN;
    symbols
        .iter()
        .map(|&s| match value(s) {
            None => {
                max = i64::MIN;
                IGNORE
            }
            Some(v) => {
                let record = v > max;
                max = max.max(v);
                if record { RECORD } else { HELD }
            }
        })
        .collect()
}

/// Independent symbols: `R` with probability ⅙, otherwise a uniform value.
pub fn gen_random(rng: &mut Rng, len: usize) -> Vec<usize> {
    (0..len)
        .map(|_| {
            if rng.below(6) == 0 {
                RESET
            } else {
                rng.below(NUM_VALUES)
            }
        })
        .collect()
}

/// **The plateau family.** A high value early, then a run of 4 to 10 values
/// that never exceed it (ties occur, and they are *not* records), then a climb
/// of one or more records above it. Any comparison against a decaying average
/// drifts down to the run, and then it calls the ordinary values of the run
/// records.
pub fn gen_plateau(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        out.push(RESET);
        let top = rng.range(2, NUM_VALUES - 2); // room to be beaten later
        out.push(top - 1);
        for _ in 0..rng.range(4, 10) {
            // below the plateau, and sometimes exactly on it (a tie, not a record)
            out.push(rng.below(top));
        }
        for v in top..rng.range(top + 1, NUM_VALUES) {
            out.push(v); // the records that beat it, one at a time
        }
    }
    out.truncate(len);
    out
}

/// **The climb family.** A staircase. Each new maximum occurs exactly once as a
/// record, with values at or below it between the records. So the whole task is
/// to tell a record from a tie, and the answer changes every few tokens.
pub fn gen_climb(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        out.push(RESET);
        let mut max = 0usize;
        while max < NUM_VALUES && out.len() < len {
            if rng.below(2) == 0 && max > 0 {
                out.push(rng.below(max)); // at or below the maximum
            } else {
                max += 1;
                out.push(max - 1); // the record
            }
        }
    }
    out.truncate(len);
    out
}

/// **The bands family.** Each segment stays in a narrow band of four values of
/// the alphabet (the bottom, the middle or the top). So neighbouring values
/// decide a record, wherever the band is.
///
/// This is the range adversary. The other way to hold a maximum is a sum of
/// `exp(S·v)`. It needs `e^S` to exceed the sequence length, *and* the span of
/// the whole alphabet to fit in the precision of one channel. An arm that uses
/// its resolution near the top of the alphabet is then blind at the bottom, and
/// this family scores both ends equally.
pub fn gen_bands(rng: &mut Rng, len: usize) -> Vec<usize> {
    let width = 4;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        out.push(RESET);
        let base = rng.below(NUM_VALUES - width + 1);
        for _ in 0..rng.range(4, 12) {
            out.push(base + rng.below(width));
        }
    }
    out.truncate(len);
    out
}

/// **The edge family.** Every token is within one step of the running maximum:
/// `v ∈ {max − 1, max, max + 1}`. So a record is exactly `v = max + 1`, and
/// only the maximum decides *every scored position*. A large value alone does
/// not, and that is what carries the other families.
///
/// It is the analogue of the knife burst of `tally-drift`, for the same reason.
/// A comparison against a decaying **average** is below the maximum. So it
/// calls both `max` and `max + 1` a record, and it is wrong on a third of the
/// stream, however it is tuned.
pub fn gen_edge(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        out.push(RESET);
        // start in the lower half, so the walk has room to climb
        let mut max: usize = 1 + rng.below(NUM_VALUES / 2);
        out.push(max - 1); // the first value of the segment, a record by definition
        for _ in 0..rng.range(6, 14) {
            // `max − 1`, `max` or `max + 1`, clipped to the alphabet
            let v = (max + rng.below(3)).saturating_sub(1).clamp(1, NUM_VALUES);
            out.push(v - 1); // symbol id of value `v`
            max = max.max(v);
        }
    }
    out.truncate(len);
    out
}

/// The training mixture: a third [`gen_random`], and a sixth for each
/// adversarial family.
pub fn gen_mixed(rng: &mut Rng, len: usize) -> Vec<usize> {
    match rng.below(6) {
        0 => gen_plateau(rng, len),
        1 => gen_climb(rng, len),
        2 => gen_bands(rng, len),
        3 => gen_edge(rng, len),
        _ => gen_random(rng, len),
    }
}

/// The evaluation families, reported separately.
pub const FAMILIES: &[(&str, Generator)] = &[
    ("random", gen_random as Generator),
    ("plateau", gen_plateau as Generator),
    ("climb", gen_climb as Generator),
    ("bands", gen_bands as Generator),
    ("edge", gen_edge as Generator),
];

/// The [`Task`](crate::shared::Task) of this rung.
pub fn task() -> crate::shared::Task {
    crate::shared::Task {
        num_symbols: NUM_SYMBOLS,
        seq_length: SEQ_LENGTH,
        labels,
        train: gen_mixed as Generator,
        families: FAMILIES,
        glyphs: &['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'R'],
        class_glyphs: ['-', '+'],
        class_names: ["held", "record"],
    }
}
