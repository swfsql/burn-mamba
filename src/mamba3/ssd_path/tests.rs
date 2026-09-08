//! The chunk-length schedule. Pure arithmetic — no tensors, no backend.
//!
//! What is being pinned is that the two dials which widen a chunk without
//! appearing in it are treated as the *different* widenings they are:
//! `mimo_rank` fuses onto both of a chunk's axes and so divides it out, while
//! `micro_steps` widens only the write axis and so subdivides a chunk of
//! unchanged folded width. See [`Mamba3SsdPath::optimal_chunk_len`] and
//! `info/architecture-deltas.md` §8.

use super::*;

/// The SISO block's schedule is unchanged: `√(N·P)` on the 32 grid, capped.
#[test]
fn siso_schedule_is_the_square_root_rule() {
    let opt = |r, p| Mamba3SsdPath::optimal_chunk_len(r, p, 1, 1);
    assert_eq!(opt(128, 64), 96); // isqrt(8192) = 90
    assert_eq!(opt(64, 64), 64);
    assert_eq!(opt(128, 128), 128);
    assert_eq!(opt(16, 64), 32);
    assert_eq!(opt(4096, 4096), 512); // the ceiling
    assert_eq!(opt(1, 1), 32); // the floor
}

/// The fused axis is `chunk_len · mimo_rank`, so the rank divides the chunk.
#[test]
fn mimo_rank_divides_the_chunk() {
    let opt = |m| Mamba3SsdPath::optimal_chunk_len(128, 64, m, 1);
    assert_eq!(opt(1), 96);
    assert_eq!(opt(2), 64); // 90/2 = 45 -> 64
    assert_eq!(opt(4), 32); // 90/4 = 23 -> 32, and the floor takes over
    assert_eq!(opt(8), 32);
}

/// `micro_steps` subdivides the chunk instead of shortening it: the folded
/// width stays put (to within one token) and the token count carries the `u`.
#[test]
fn micro_steps_subdivides_the_chunk() {
    let base = Mamba3SsdPath::optimal_chunk_len(128, 64, 1, 1);
    assert_eq!(base, 96);
    for u in [1, 2, 3, 4, 5, 8, 16] {
        let folded = Mamba3SsdPath::optimal_chunk_len(128, 64, 1, u);
        assert!(
            (base..base + u).contains(&folded),
            "u = {u} moved the folded chunk off {base} to {folded}",
        );
        assert_eq!(
            Mamba3SsdPath::chunk_tokens(folded, u),
            base.div_ceil(u),
            "u = {u} should divide the *token* count, not the folded width",
        );
    }
}

/// `mimo_rank` and `micro_steps` are not interchangeable: `m` divides the fused
/// read axis, `u` only splits a chunk into more tokens' worth of writes.
#[test]
fn the_two_dials_are_not_the_same_widening() {
    // Score elements per token, up to the shared `batch · sequence · heads`:
    // `chunk_tokens · micro_steps · mimo_rank²` = `chunk_len · m²`.
    let score = |m: usize, u: usize| Mamba3SsdPath::optimal_chunk_len(128, 64, m, u) * m * m;
    let siso = score(1, 1);
    for u in [1, 2, 4, 8] {
        assert!(
            score(1, u) <= siso + u,
            "micro_steps should leave the score flat, not grow it",
        );
    }
    assert!(score(2, 1) > siso / 2, "mimo_rank costs `m`, not nothing");
}

/// Every value the schedule can produce is a usable chunk: a whole number of
/// tokens, never zero, never far above the ceiling — for any dial combination.
#[test]
fn the_schedule_stays_on_the_grid() {
    for state_rank in [1, 4, 16, 64, 128, 256, 4096] {
        for per_head_dim in [1, 32, 64, 128] {
            for mimo_rank in [1, 2, 3, 4, 8] {
                for micro_steps in [1, 2, 3, 5, 8] {
                    let n = Mamba3SsdPath::optimal_chunk_len(
                        state_rank,
                        per_head_dim,
                        mimo_rank,
                        micro_steps,
                    );
                    assert_eq!(n % micro_steps, 0, "{n} is not a whole number of tokens");
                    assert!(Mamba3SsdPath::chunk_tokens(n, micro_steps) >= 1);
                    // The rounding to a multiple of `u` can overshoot the ceiling
                    // by less than one token, and only there.
                    assert!((32..512 + micro_steps).contains(&n), "{n} is out of range");
                    if micro_steps == 1 {
                        assert_eq!(n % 32, 0, "{n} is off the 32 grid at u = 1");
                    }
                }
            }
        }
    }
}

/// Widening a dial never widens the chunk: the divisor is monotone.
#[test]
fn the_schedule_is_monotone_in_the_fold() {
    let mut previous = usize::MAX;
    for fold in 1..=16 {
        let n = Mamba3SsdPath::optimal_chunk_len(256, 128, fold, 1);
        assert!(n <= previous, "fold {fold} widened the chunk to {n}");
        previous = n;
    }
}

/// An explicit chunk length wins over the schedule, at every dial setting.
#[test]
fn an_explicit_chunk_len_is_not_overridden() {
    let path = Mamba3SsdPath::SerialRecalculated(Some(256));
    assert_eq!(path.chunk_len(), Some(256));
    assert_eq!(Mamba3SsdPath::default().chunk_len(), None);
}
