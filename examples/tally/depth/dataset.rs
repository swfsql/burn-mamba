//! The tally-depth stream: `(` / `)` / `R`, whose per-position target is
//! **whether the bracket depth is still positive** — with a `)` at depth 0
//! ignored rather than going negative.
//!
//! ```text
//!   symbols   (  (  )  )  )  (  )  R  )  (  )
//!   depth     1  2  1  0  0  1  0  0  0  1  0
//!   target    .  .  in out out  .  out  .  out  .  out
//! ```
//!
//! Only `)` positions are scored: after a `(` the depth is at least one
//! whatever the history, so that target is one bit of the token class and a
//! model can collect it without any memory at all (the mistake the handoff's
//! own toy made first). `R` is unscored for the same reason.
//!
//! The recursion is Lindley's, `cₜ = max(cₜ₋₁ + aₜ, 0)`: linear in the
//! (max, +) semiring, so a tropical register computes it exactly — and no
//! linear recurrence does, because the floor is where the two semirings differ.
//! [`gen_floor`] is the family built on that difference: the **unclamped** sum
//! is misled at a third of the scored positions there.

use crate::shared::data::{Generator, IGNORE, Rng};

/// Input symbol: open a bracket (depth + 1).
pub const OPEN: usize = 0;
/// Input symbol: close a bracket (depth − 1, floored at 0).
pub const CLOSE: usize = 1;
/// Input symbol: reset the depth to 0.
pub const RESET: usize = 2;
/// Input alphabet size.
pub const NUM_SYMBOLS: usize = 3;

/// Target class: the depth is 0 after this `)` — it closed nothing.
pub const EMPTY: i64 = 0;
/// Target class: the depth is still positive after this `)`.
pub const INSIDE: i64 = 1;

/// Length of every generated sequence.
pub const SEQ_LENGTH: usize = 32;

/// Per-position targets: the clamped depth's sign at every `)`, [`IGNORE`]
/// elsewhere.
pub fn labels(symbols: &[usize]) -> Vec<i64> {
    let mut depth: i64 = 0;
    symbols
        .iter()
        .map(|&s| match s {
            RESET => {
                depth = 0;
                IGNORE
            }
            OPEN => {
                depth += 1;
                IGNORE
            }
            CLOSE => {
                depth = (depth - 1).max(0);
                if depth > 0 { INSIDE } else { EMPTY }
            }
            _ => panic!("symbol out of alphabet: {s}"),
        })
        .collect()
}

/// The depth a plain (unclamped) running sum would report — what a linear
/// recurrence computes, and what [`gen_floor`] misleads.
pub fn unclamped(symbols: &[usize]) -> Vec<i64> {
    let mut sum: i64 = 0;
    symbols
        .iter()
        .map(|&s| {
            match s {
                RESET => sum = 0,
                OPEN => sum += 1,
                CLOSE => sum -= 1,
                _ => unreachable!(),
            }
            sum
        })
        .collect()
}

/// Independent symbols: `R` with probability ⅛, otherwise `(` / `)` evenly.
pub fn gen_random(rng: &mut Rng, len: usize) -> Vec<usize> {
    (0..len)
        .map(|_| match rng.below(8) {
            0 => RESET,
            n if n % 2 == 0 => OPEN,
            _ => CLOSE,
        })
        .collect()
}

/// **The floor family.** A run of `)` that goes past zero (ignored there, so the
/// clamp forgets it while a running sum does not), then a matching `(`…`)`
/// group: the clamp says `inside` until the group's last close, the sum — still
/// in the red from the surplus — says `empty` throughout.
///
/// The group is never longer than the surplus, which is what keeps the sum
/// wrong for the whole of it, and both classes present at every position.
pub fn gen_floor(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        if rng.below(8) == 0 {
            out.push(RESET);
        }
        let surplus = rng.range(2, 5);
        for _ in 0..surplus {
            out.push(CLOSE);
        }
        let opens = rng.range(2, surplus.max(2));
        for _ in 0..opens {
            out.push(OPEN);
        }
        for _ in 0..opens {
            out.push(CLOSE);
        }
    }
    out.truncate(len);
    out
}

/// **The deep family.** A long `(` run, then a long `)` run that crosses zero
/// exactly, then short alternations: telling depth 1 from depth 0 after a deep
/// excursion is what a multiplicative decrement (which never reaches zero)
/// cannot do.
pub fn gen_deep(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let depth = rng.range(4, 12);
        for _ in 0..depth {
            out.push(OPEN);
        }
        for _ in 0..depth + rng.below(3) {
            out.push(CLOSE);
        }
        for _ in 0..rng.below(3) {
            out.push(OPEN);
            out.push(CLOSE);
        }
        if rng.below(3) == 0 {
            out.push(RESET);
        }
    }
    out.truncate(len);
    out
}

/// The training mixture: half [`gen_random`], a quarter of each adversarial
/// family.
pub fn gen_mixed(rng: &mut Rng, len: usize) -> Vec<usize> {
    match rng.below(4) {
        0 => gen_floor(rng, len),
        1 => gen_deep(rng, len),
        _ => gen_random(rng, len),
    }
}

/// The evaluation families, reported separately.
pub const FAMILIES: &[(&str, Generator)] = &[
    ("random", gen_random as Generator),
    ("floor", gen_floor as Generator),
    ("deep", gen_deep as Generator),
];

/// This rung's [`Task`](crate::shared::Task).
pub fn task() -> crate::shared::Task {
    crate::shared::Task {
        num_symbols: NUM_SYMBOLS,
        seq_length: SEQ_LENGTH,
        labels,
        train: gen_mixed as Generator,
        families: FAMILIES,
        glyphs: &['(', ')', 'R'],
        class_glyphs: ['o', 'i'],
        class_names: ["empty", "inside"],
    }
}
