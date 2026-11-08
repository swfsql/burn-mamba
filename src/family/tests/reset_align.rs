//! The reset alignment of each family (`Block::reset_align`). A block accepts a
//! reset of a packed row only at a multiple of this value. Each expected value
//! is a chunk length in tokens, computed by hand from the config of the test.

use crate::prelude::*;
use burn_stack::utils::test_helpers::test_device;

/// `state_rank · per_head_dim = 32`, so `⌊√32⌋ = 5`. The next multiple of 32
/// is 32, the optimal chunk length.
#[cfg(feature = "mamba2")]
#[test]
fn a_mamba2_block_aligns_to_its_chunk() {
    let block = Mamba2Config::new(8)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .init(&test_device());
    assert_eq!(block.reset_align(&Mamba2SsdPath::SerialRecalculated(Some(4))), Some(4));
    assert_eq!(block.reset_align(&Mamba2SsdPath::Serial(Some(16))), Some(16));
    assert_eq!(block.reset_align(&Mamba2SsdPath::SerialRecalculated(None)), Some(32));
}

/// A packed forward on the `Minimal` path panics, so the block takes no packed
/// rows on that path.
#[cfg(feature = "mamba2")]
#[test]
fn a_mamba2_block_on_the_minimal_path_takes_no_packed_rows() {
    let block = Mamba2Config::new(8)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .init(&test_device());
    assert_eq!(block.reset_align(&Mamba2SsdPath::Minimal(Some(4))), None);
    assert_eq!(block.reset_align(&Mamba2SsdPath::Minimal(None)), None);
}

/// A chunk length counts write positions, and a token has `micro_steps` write
/// positions. The optimal chunk length is 32, as above. At 3 micro-steps the
/// length grows to 33, the next multiple of 3.
#[cfg(feature = "mamba3")]
#[test]
fn a_mamba3_block_aligns_to_its_chunk_in_tokens() {
    let block = |micro_steps| {
        Mamba3Config::new(8)
            .with_expand(2)
            .with_per_head_dim(4)
            .with_state_rank(8)
            .with_micro_steps(micro_steps)
            .with_tap_pattern(TapPattern::Vertical)
            .init(&test_device())
    };
    let with_chunk = Mamba3SsdPath::SerialRecalculated;
    assert_eq!(block(1).reset_align(&with_chunk(Some(4))), Some(4));
    assert_eq!(block(2).reset_align(&with_chunk(Some(8))), Some(4));
    // The next multiple of 2 after 7 is 8.
    assert_eq!(block(2).reset_align(&with_chunk(Some(7))), Some(4));
    assert_eq!(block(1).reset_align(&with_chunk(None)), Some(32));
    assert_eq!(block(2).reset_align(&with_chunk(None)), Some(16));
    assert_eq!(block(3).reset_align(&with_chunk(None)), Some(11));
}

/// A packed forward on the `Minimal` path panics, so the block takes no packed
/// rows on that path.
#[cfg(feature = "mamba3")]
#[test]
fn a_mamba3_block_on_the_minimal_path_takes_no_packed_rows() {
    let block = Mamba3Config::new(8)
        .with_expand(2)
        .with_per_head_dim(4)
        .with_state_rank(8)
        .init(&test_device());
    assert_eq!(block.reset_align(&Mamba3SsdPath::Minimal(Some(4))), None);
    assert_eq!(block.reset_align(&Mamba3SsdPath::Minimal(None)), None);
}

/// Mamba-1 has no chunks, so it takes no packed rows.
#[cfg(feature = "mamba1")]
#[test]
fn a_mamba1_block_takes_no_packed_rows() {
    let block = Mamba1Config::new(8).with_state_rank(8).init(&test_device());
    assert_eq!(block.reset_align(&()), None);
}
