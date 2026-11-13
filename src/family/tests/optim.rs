//! The Muon specs that only the Mamba math sets.
//!
//! The block contracts of burn-stack check each spec against the weights of
//! the block (`block_contract.rs`). These tests check two properties that no
//! contract can state:
//!
//! - the count of the specs of each family,
//! - the channels of a fused projection that stay on the fallback optimizer.

use crate::prelude::*;

/// The Muon plan of a Mamba network config holds a fixed count of specs for
/// each family. The network of this test has no feed-forward, no `CatLinear`
/// merge and no untied parameter. So the plan holds only the specs of the
/// block:
///
/// - Mamba-3: `in_proj`, `out_proj`.
/// - Mamba-2: `in_proj`, `out_proj`.
/// - Mamba-1: `in_proj`, `x_proj`, `out_proj`.
///
/// A change that adds a spec to a family, or removes one, fails this test.
///
/// Each family goes through a different network config. So the test reaches
/// the `muon_plan` of three network configs, and the arm of each family in
/// `MambaBlockConfig::muon_projections`. No block contract reaches these
/// lines, because a contract takes a block config.
#[test]
fn each_family_gives_a_fixed_count_of_muon_specs() {
    let networks: Vec<(&str, MuonPlan, usize)> = vec![
        #[cfg(feature = "mamba3")]
        (
            "Mamba-3 in a `MambaBidiLayersConfig`",
            MambaBidiLayersConfig::new(
                BidiShape::new(1, OutputMergeConfig::mean(1)),
                MambaBlockConfig::Mamba3(
                    Mamba3Config::new(32)
                        .with_state_rank(16)
                        .with_expand(2)
                        .with_per_head_dim(16),
                ),
            )
            .muon_plan(),
            2,
        ),
        #[cfg(feature = "mamba2")]
        (
            "Mamba-2 in a `MambaVocabNetConfig`",
            MambaVocabNetConfig::new(
                VocabShape::new(20, NetworkShape::new(1)),
                MambaBlockConfig::Mamba2(
                    Mamba2Config::new(32)
                        .with_state_rank(16)
                        .with_expand(2)
                        .with_per_head_dim(16),
                ),
            )
            .muon_plan(),
            2,
        ),
        #[cfg(feature = "mamba1")]
        (
            "Mamba-1 in a `MambaLatentNetConfig`",
            MambaLatentNetConfig::new(
                LatentShape::new(3, 5, NetworkShape::new(1)),
                MambaBlockConfig::Mamba1(Mamba1Config::new(32).with_state_rank(16)),
            )
            .muon_plan(),
            3,
        ),
    ];
    for (network, plan, count) in networks {
        let paths: Vec<&str> = plan.specs.iter().map(|spec| spec.path.as_str()).collect();
        assert_eq!(plan.specs.len(), count, "The plan of {network} has the Muon specs {paths:?}.");
    }
}

/// The Muon-owned share of a Mamba-3 `in_proj`: the Δ/A/λ channels stay behind.
#[cfg(feature = "mamba3")]
#[test]
fn mamba3_scalar_channels_stay_on_adamw() {
    let config = Mamba3Config::new(64).with_state_rank(32).with_expand(2);
    let specs = config.muon_projections();
    let in_proj = &specs[0];
    assert_eq!(in_proj.width(), config.d_in_proj());

    let scalar: usize = in_proj
        .sub_projections
        .iter()
        .filter(|s| !s.muon)
        .map(|s| s.width)
        .sum();
    assert_eq!(scalar, 3 * config.nheads());
    assert!(!in_proj.is_whole_muon());
}
