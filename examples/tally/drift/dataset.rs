//! The tally-drift stream: 33 value symbols on a grid from `−2` to `+2` in
//! steps of `1/8`, and a gap `~`. The target at each position is **whether this
//! value is above the running estimate of the level**. The estimate *ages*
//! across gaps, and it already includes the value that it is compared to.
//!
//! ```text
//!   symbols   +1   +1   +2   ~    ~    -1   +1
//!   estimate  1.0  1.0  1.33 1.33 1.33 0.0  0.36
//!   target     .    .    +    .    .    -    +
//! ```
//!
//! (`Q_GAP = 0.5`: the two gaps shrink the weight of three values to `0.75`, so
//! the `-1` pulls the estimate to `0`. A value on its estimate is unscored.)
//!
//! The level is a random walk that moves only during gaps, so a gap is elapsed
//! time: it adds doubt without adding evidence. The optimal estimate is the
//! information-form filter of [`posterior`]. Its whole content is **how much
//! the past is still worth** after `k` gaps:
//!
//! ```text
//!   Λ ← Λ / (1 + k·q·Λ)          the prior's weight, in votes
//! ```
//!
//! That discount is hyperbolic in `k` and **saturating** in `Λ`. After a long
//! run, the past is worth `1/(k·q)` votes, however long the run was. A linear
//! recurrence can discount only geometrically (`αᵏ·Λ`, a product of the two).
//! That is why the Kalman gate computes this. It is also why the difference
//! between the arms is a few points, not a large margin: over a narrow range
//! of `k`, the two curves are close. See `examples/tally/README.md`.

use crate::shared::data::{Generator, IGNORE, Rng};

/// Number of distinct values. Symbol `i` carries value [`value`]`(i)`.
///
/// The grid is fine (an eighth of a unit) on purpose. The rung is about *how
/// good* an estimate is, so a value must be able to sit right next to an
/// estimate. [`gen_knife`] puts values there.
pub const NUM_VALUES: usize = 33;
/// Input symbol: a gap. Time passes, and nothing is observed.
pub const GAP: usize = NUM_VALUES;
/// Input alphabet size.
pub const NUM_SYMBOLS: usize = NUM_VALUES + 1;

/// Target class: this value is below the current estimate.
pub const BELOW: i64 = 0;
/// Target class: this value is above it.
pub const ABOVE: i64 = 1;

/// Length of every generated sequence.
pub const SEQ_LENGTH: usize = 32;

/// The process noise one gap token injects (`q` in [`posterior`]).
pub const Q_GAP: f64 = 0.5;
/// How far the level walks per gap token.
pub const JUMP: f64 = 0.8;
/// Positions closer than this to the estimate are unscored: a near-tie tests
/// calibration, not memory. It is well below the probe offset of
/// [`gen_knife`], but a probe can still land inside it. The aim is quantised
/// onto the `1/8` value grid, and that moves it by up to `1/16`.
pub const MARGIN: f64 = 0.005;

/// The distance between the probes of [`gen_knife`] and the estimate that they
/// test. It is small enough that an estimator must be *right*, not only close:
/// after a gap, the error of a geometric discount is larger than this.
pub const PROBE_OFFSET: f64 = 0.06;

/// The value a symbol carries (`GAP` carries none): `−2, −1.875, … , +2`.
pub fn value(symbol: usize) -> Option<f64> {
    (symbol < NUM_VALUES).then(|| symbol as f64 * 0.125 - 2.0)
}

/// The nearest symbol to a real value (clipped to the grid).
pub fn quantize(x: f64) -> usize {
    (((x + 2.0) * 8.0).round() as i64).clamp(0, NUM_VALUES as i64 - 1) as usize
}

/// The information-form filter that gives the labels: evidence `m = 1` at a
/// value, and process noise `q` at a gap.
///
/// It returns the posterior mean **after** each position (the estimate against
/// which the value at that position is compared).
pub fn posterior(symbols: &[usize]) -> Vec<f64> {
    let (mut lambda, mut eta) = (0.0f64, 0.0f64);
    symbols
        .iter()
        .map(|&s| {
            match value(s) {
                None => {
                    let d = 1.0 / (1.0 + Q_GAP * lambda);
                    lambda *= d;
                    eta *= d;
                }
                Some(v) => {
                    lambda += 1.0;
                    eta += v;
                }
            }
            if lambda > 0.0 { eta / lambda } else { 0.0 }
        })
        .collect()
}

/// Per-position targets: at every value, whether it is above the posterior
/// mean. [`IGNORE`] at a gap and within [`MARGIN`] of the estimate.
pub fn labels(symbols: &[usize]) -> Vec<i64> {
    let estimate = posterior(symbols);
    symbols
        .iter()
        .zip(estimate)
        .map(|(&s, mean)| match value(s) {
            None => IGNORE,
            Some(v) if (v - mean).abs() < MARGIN => IGNORE,
            Some(v) => {
                if v > mean { ABOVE } else { BELOW }
            }
        })
        .collect()
}

/// Draw one observation of `level`, quantised onto the value grid.
fn observe(rng: &mut Rng, level: f64) -> usize {
    quantize(level + rng.normal())
}

/// Runs of values separated by runs of gaps, both of random length.
pub fn gen_random(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len);
    let mut level = rng.normal();
    while out.len() < len {
        for _ in 0..rng.range(1, 8) {
            out.push(observe(rng, level));
        }
        let k = rng.range(1, 4);
        for _ in 0..k {
            out.push(GAP);
        }
        level += JUMP * (k as f64).sqrt() * rng.normal();
    }
    out.truncate(len);
    out
}

/// **The straddle family.** A run long enough to make the estimate confident,
/// then a gap of random length, then one to three probes. The probes compare
/// against the *discounted* weight of the prior, and that is where a geometric
/// discount and a hyperbolic discount differ.
pub fn gen_straddle(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len);
    let mut level = rng.normal();
    while out.len() < len {
        for _ in 0..rng.range(4, 14) {
            out.push(observe(rng, level));
        }
        let k = rng.range(1, 8);
        for _ in 0..k {
            out.push(GAP);
        }
        level += JUMP * (k as f64).sqrt() * rng.normal();
        for _ in 0..rng.range(1, 3) {
            out.push(observe(rng, level));
        }
    }
    out.truncate(len);
    out
}

/// **The knife family.** A run, a gap, then a *fresh* observation, and only
/// then a burst of probes at the edge of the estimate.
///
/// The order is the whole construction. A gap scales `η` and `Λ` by the same
/// factor. So it changes how much the past is **worth**, but not the estimate
/// itself: every arm agrees right after a gap. The disagreement appears when
/// new evidence is blended in. `S = (w·S_prev + x)/(w + 1)` moves by an amount
/// that only the weight `w` of the prior sets. The Kalman gate computes that
/// weight, and a projected decay only approximates it. The probes then land
/// between the two answers.
pub fn gen_knife(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(len);
    let (mut lambda, mut eta) = (0.0f64, 0.0f64);
    let mut level = rng.normal();
    // The same filter as [`posterior`], kept online so that a probe can be
    // aimed.
    let observe_into = |out: &mut Vec<usize>, symbol: usize, lambda: &mut f64, eta: &mut f64| {
        match value(symbol) {
            None => {
                let d = 1.0 / (1.0 + Q_GAP * *lambda);
                *lambda *= d;
                *eta *= d;
            }
            Some(v) => {
                *lambda += 1.0;
                *eta += v;
            }
        }
        out.push(symbol);
    };
    while out.len() < len {
        // A short run to build a prior, whose weight after `k` gaps is
        // `Λ/(1 + k·q·Λ)` against a geometric `αᵏ·Λ`.
        for _ in 0..rng.range(3, 6) {
            observe_into(&mut out, quantize(level + rng.normal()), &mut lambda, &mut eta);
        }
        let k = rng.range(1, 5);
        for _ in 0..k {
            observe_into(&mut out, GAP, &mut lambda, &mut eta);
        }
        level += JUMP * (k as f64).sqrt() * rng.normal();
        // Fresh evidence first: the weight of the prior decides how far it
        // moves the estimate, so this is where the arms differ.
        observe_into(&mut out, quantize(level + rng.normal()), &mut lambda, &mut eta);
        // Then a **burst** of probes, each at the edge of the current estimate
        // (every probe is itself evidence, so the next probe is aimed again).
        // The burst makes the *scored* positions the positions that the
        // estimate decides. An ordinary observation lands far enough from the
        // mean that every arm agrees on it, and a family of those measures
        // nothing.
        for _ in 0..rng.range(4, 8) {
            let mean = if lambda > 0.0 { eta / lambda } else { 0.0 };
            let side = if rng.below(2) == 0 { PROBE_OFFSET } else { -PROBE_OFFSET };
            observe_into(&mut out, quantize(mean + side), &mut lambda, &mut eta);
        }
    }
    out.truncate(len);
    out
}

/// **The chain family.** Short runs and single gaps throughout, so the discount
/// of the prior compounds many times over one sequence.
pub fn gen_chain(rng: &mut Rng, len: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(len);
    let mut level = rng.normal();
    while out.len() < len {
        for _ in 0..rng.range(1, 4) {
            out.push(observe(rng, level));
        }
        out.push(GAP);
        level += JUMP * rng.normal();
    }
    out.truncate(len);
    out
}

/// The training mixture: a quarter of each family.
pub fn gen_mixed(rng: &mut Rng, len: usize) -> Vec<usize> {
    match rng.below(4) {
        0 => gen_straddle(rng, len),
        1 => gen_chain(rng, len),
        2 => gen_knife(rng, len),
        _ => gen_random(rng, len),
    }
}

/// The evaluation families, reported separately.
pub const FAMILIES: &[(&str, Generator)] = &[
    ("random", gen_random as Generator),
    ("straddle", gen_straddle as Generator),
    ("chain", gen_chain as Generator),
    ("knife", gen_knife as Generator),
];

/// The [`Task`](crate::shared::Task) of this rung.
pub fn task() -> crate::shared::Task {
    crate::shared::Task {
        num_symbols: NUM_SYMBOLS,
        seq_length: SEQ_LENGTH,
        labels,
        train: gen_mixed as Generator,
        families: FAMILIES,
        glyphs: &[
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', '~',
        ],
        class_glyphs: ['-', '+'],
        class_names: ["below", "above"],
    }
}
