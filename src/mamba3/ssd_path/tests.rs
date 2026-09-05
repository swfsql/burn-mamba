//! The chunk-length schedule. Pure arithmetic — no tensors, no backend.
//!
//! What is being pinned is that the two dials which widen a chunk without
//! appearing in it (`mimo_rank` fuses onto the chunk axis, `micro_steps` folds
//! into the sequence axis) divide it back out, and that the 32 grid survives the
//! division. See [`Mamba3SsdPath::optimal_chunk_len`] and
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

/// `micro_steps` folds into the sequence axis and divides it the same way.
#[test]
fn micro_steps_divides_the_chunk_the_same_way() {
    for (m, u) in [(1, 2), (2, 1), (1, 4), (4, 1), (2, 2), (1, 8), (8, 1)] {
        assert_eq!(
            Mamba3SsdPath::optimal_chunk_len(128, 64, m, u),
            Mamba3SsdPath::optimal_chunk_len(128, 64, 1, m * u),
            "only the product m·u is supposed to matter (m = {m}, u = {u})",
        );
    }
}

/// Every value the schedule can produce is a usable chunk: on the 32 grid,
/// never zero, never above the ceiling — for any dial combination.
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
                    assert_eq!(n % 32, 0, "{n} is off the 32 grid");
                    assert!((32..=512).contains(&n), "{n} is outside 32..=512");
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
