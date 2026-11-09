//! # Mamba-3 — In-Projection Layout
//!
//! One table holds the channel groups of the in-projection. Each row gives the
//! key, the name, the width, the axis, the optimizer and the place of one
//! group. The widths of the in-projection, the Muon plan and the split read
//! this table. A new optional group needs one [`GroupKey`] variant, one row,
//! and one field of [`InProjStreams`](crate::mamba3::plant_frame::InProjStreams)
//! with its `take` line.
//!
//! The doc of [`Mamba3::in_proj`](crate::mamba3::mamba3::Mamba3::in_proj) gives
//! the layout in full.
//!
//! Burn has no zero-width tensors, and `split_with_sizes` drops a part of zero
//! length. So the table lists only the groups that the block projects, and each
//! row has a width above zero. Each row then has exactly one part of the split.

use crate::mamba3::positive::{Gain, Tropical};
use crate::mamba3::product::unfold_micro_bs;
use crate::mamba3::tap_pattern::TapPattern;
use burn::prelude::*;

/// The key of a channel group. There is one variant for each group, in layout
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupKey {
    Z,
    X,
    BRaw,
    CRaw,
    DdDt,
    DdARaw,
    LambdaRaw,
    MuRaw,
    Rotation,
    KalmanNoise,
    TropicalA,
    TropicalB,
}

/// The axis of the stream of a channel group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupAxis {
    /// One sub-projection. The stream is on the read axis.
    Token,
    /// One sub-projection for each micro-step. The stream is on the write axis.
    MicroStep,
}

impl GroupAxis {
    /// The number of sub-projections of a group on this axis.
    fn sub_projections(self, micro_steps: usize) -> usize {
        match self {
            Self::Token => 1,
            Self::MicroStep => micro_steps,
        }
    }
}

/// The optimizer of a channel group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "optim"), allow(dead_code))]
pub(crate) enum GroupOptimizer {
    /// Muon orthogonalises each sub-projection of the group.
    Muon,
    /// The group stays on the fallback optimizer.
    Fallback,
}

/// The weight that holds a channel group under
/// [`Mamba3Untied::InProjTail`](crate::mamba3::mamba3::Mamba3Untied::InProjTail).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupPlace {
    /// `in_proj`.
    InProj,
    /// `in_proj_tail`.
    InProjTail,
}

/// One row of the table.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(feature = "optim"), allow(dead_code))]
pub(crate) struct ChannelGroup {
    pub key: GroupKey,
    pub name: &'static str,
    /// The channels of one micro-step, or of the token for a per-token group.
    pub width: usize,
    pub axis: GroupAxis,
    pub optimizer: GroupOptimizer,
    pub place: GroupPlace,
}

/// The settings of a block that set the table. `Mamba3Config` and `Mamba3` both
/// hold them.
pub(crate) struct InProjSettings {
    pub d_inner: usize,
    pub nheads: usize,
    /// `ngroups · state_rank · mimo_rank`.
    pub bc_width: usize,
    pub num_rotation_channels: usize,
    pub micro_steps: usize,
    pub tap_pattern: TapPattern,
    pub gain: Gain,
    pub tropical: Tropical,
}

/// The channel groups that a block projects, in layout order.
pub(crate) struct InProjLayout {
    groups: Vec<ChannelGroup>,
    micro_steps: usize,
}

impl InProjLayout {
    /// The table of a block with these settings.
    ///
    /// # Panics
    /// If a group that the block projects has a width of zero.
    pub(crate) fn new(settings: InProjSettings) -> Self {
        use GroupAxis::{MicroStep, Token};
        use GroupKey::*;
        use GroupOptimizer::{Fallback, Muon};
        use GroupPlace::{InProj, InProjTail};
        let InProjSettings {
            d_inner,
            nheads,
            bc_width,
            num_rotation_channels,
            micro_steps,
            tap_pattern,
            gain,
            tropical,
        } = settings;

        // The first column is the presence condition of the group.
        #[rustfmt::skip]
        let rows = [
            (true,                                  Z,           "z",            d_inner,               Token,     Muon,     InProj),
            (true,                                  X,           "x",            d_inner,               MicroStep, Muon,     InProj),
            (true,                                  BRaw,        "b_raw",        bc_width,              MicroStep, Muon,     InProj),
            (true,                                  CRaw,        "c_raw",        bc_width,              Token,     Muon,     InProj),
            (true,                                  DdDt,        "dd_dt",        nheads,                MicroStep, Fallback, InProjTail),
            (true,                                  DdARaw,      "dd_a_raw",     nheads,                MicroStep, Fallback, InProjTail),
            (tap_pattern.has_beta_tap(),            LambdaRaw,   "lambda_raw",   nheads,                MicroStep, Fallback, InProjTail),
            (tap_pattern.has_two_taps(micro_steps), MuRaw,       "mu_raw",       nheads,                MicroStep, Fallback, InProjTail),
            (num_rotation_channels > 0,             Rotation,    "rotation",     num_rotation_channels, MicroStep, Muon,     InProjTail),
            (gain.projects_noise(),                 KalmanNoise, "kalman_noise", nheads,                MicroStep, Fallback, InProjTail),
            (tropical.is_on(),                      TropicalA,   "tropical_a",   nheads,                MicroStep, Fallback, InProjTail),
            (tropical.is_on(),                      TropicalB,   "tropical_b",   nheads,                MicroStep, Fallback, InProjTail),
        ];
        let groups: Vec<ChannelGroup> = rows
            .into_iter()
            .filter(|row| row.0)
            .map(|(_, key, name, width, axis, optimizer, place)| ChannelGroup {
                key,
                name,
                width,
                axis,
                optimizer,
                place,
            })
            .collect();
        for group in &groups {
            assert!(group.width > 0, "the channel group {} has a width of zero", group.name);
        }
        Self { groups, micro_steps }
    }

    /// The channels of `group` over all of its sub-projections.
    fn channels(&self, group: &ChannelGroup) -> usize {
        group.width * group.axis.sub_projections(self.micro_steps)
    }

    /// The output width of the in-projection.
    pub(crate) fn d_in_proj(&self) -> usize {
        self.groups.iter().map(|group| self.channels(group)).sum()
    }

    /// The output width of the groups at [`GroupPlace::InProjTail`].
    pub(crate) fn d_in_proj_tail(&self) -> usize {
        self.groups
            .iter()
            .filter(|group| group.place == GroupPlace::InProjTail)
            .map(|group| self.channels(group))
            .sum()
    }

    /// The channels of the group `key`, or `0` when the block does not project
    /// it.
    #[cfg(all(test, feature = "_dev-test"))]
    pub(crate) fn channels_of(&self, key: GroupKey) -> usize {
        self.groups
            .iter()
            .find(|group| group.key == key)
            .map_or(0, |group| self.channels(group))
    }

    /// The sub-projections of the groups at `place`, one for each micro-step
    /// of a per-micro-step group.
    #[cfg(feature = "optim")]
    pub(crate) fn sub_projections(
        &self,
        place: GroupPlace,
    ) -> Vec<burn_stack::optim::SubProjection> {
        use burn_stack::optim::SubProjection;
        self.groups
            .iter()
            .filter(|group| group.place == place)
            .flat_map(|group| {
                let sub = match group.optimizer {
                    GroupOptimizer::Muon => SubProjection::muon(group.name, group.width),
                    GroupOptimizer::Fallback => SubProjection::adamw(group.name, group.width),
                };
                std::iter::repeat_n(sub, group.axis.sub_projections(self.micro_steps))
            })
            .collect()
    }

    /// Split the output of the in-projection into the stream of each group.
    ///
    /// The stream of a per-micro-step group is on the write axis. See
    /// [`crate::mamba3::product`].
    ///
    /// # Shapes
    /// - `proj_btd` : `[batch, tokens, d_in_proj]`
    pub(crate) fn split(&self, proj_btd: Tensor<3>) -> InProjParts {
        let sizes = self.groups.iter().map(|group| self.channels(group)).collect();
        let parts = proj_btd
            .split_with_sizes(sizes, 2)
            .into_iter()
            .zip(&self.groups)
            .map(|(part, group)| {
                let stream = match group.axis {
                    GroupAxis::Token => part,
                    GroupAxis::MicroStep => unfold_micro_bs(part, self.micro_steps),
                };
                (group.key, stream)
            })
            .collect();
        InProjParts { parts }
    }
}

/// The streams of the channel groups of one in-projection output.
pub(crate) struct InProjParts {
    parts: Vec<(GroupKey, Tensor<3>)>,
}

impl InProjParts {
    /// Remove the stream of `key`. `None` when the block does not project the
    /// group.
    pub(crate) fn take(&mut self, key: GroupKey) -> Option<Tensor<3>> {
        let at = self.parts.iter().position(|(part_key, _)| *part_key == key)?;
        Some(self.parts.swap_remove(at).1)
    }

    /// Remove the stream of `key`, a group that each block projects.
    ///
    /// # Panics
    /// If the table has no group `key`.
    pub(crate) fn take_fixed(&mut self, key: GroupKey) -> Tensor<3> {
        self.take(key).unwrap_or_else(|| panic!("the in-projection has no channel group {key:?}"))
    }
}
