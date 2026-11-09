use super::*;

/// The chunk-local pass of the recompute backward costs two iterations at any
/// chunk count. It never holds more of the chunk axis than the forward K5.
#[test]
fn the_backward_walk_is_two_iterations() {
    for nchunks in [1usize, 2, 3, 4, 5, 16, 31, 32, 1024] {
        let group = backward_chunk_group(nchunks);
        assert!((1..=nchunks).contains(&group), "{group} is not a group of {nchunks}");
        assert!(2 * group <= nchunks + 1, "{group} of {nchunks} is over half the axis");
        assert!(nchunks.div_ceil(group) <= 2, "{nchunks} in groups of {group} needs a third pass");
    }
}
