//! Column layout of the fused projection weights — the data the Muon parameter
//! groups are built from.
//!
//! Every family fuses several independent linear maps into one `Linear`, so a
//! block's weight tensor is a *concatenation* of matrices along its output
//! (column) axis. Muon orthogonalises a whole matrix at once, so it must be told
//! where those seams are; a [`ProjSpec`] is that description, and each family
//! config builds one per fused weight (`Mamba3Config::muon_projections` and
//! friends), right next to the `d_in_proj()` that sizes them.

use burn::module::ParamGroup;

/// The field names a [`MambaBlock`](crate::modules::MambaBlock) is stored under.
///
/// [`Layer`](crate::modules::Layer) holds one (`mamba_block`);
/// [`BidiLayerPair`](crate::modules::bidi::BidiLayerPair) holds a pair. Matching
/// a block weight under any of them makes a [`ProjSpec`] independent of the
/// container — one plan covers a plain stack, a virtual-layer stack, and a
/// bidirectional stack alike, including hand-written models built from these
/// pieces.
pub const BLOCK_CONTAINERS: [&str; 3] = ["mamba_block.", "straight_block.", "reverse_block."];

/// Where in the module tree a [`ProjSpec`]'s path is anchored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjScope {
    /// A weight of the SSM block: the path is matched under each of
    /// [`BLOCK_CONTAINERS`].
    Block,
    /// Any other weight: the path is matched as a plain substring of the
    /// parameter path (the layer MLP, a bidirectional merge, …).
    Path,
}

/// One contiguous column block of a fused projection weight.
///
/// `width` is the number of *columns* the block owns (Burn's `Linear` weight is
/// `[d_input, d_output]`, so the fused axis is dim 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjSegment {
    /// Name of the sub-projection, for diagnostics (`"z"`, `"b"`, `"dt"`, …).
    pub name: &'static str,
    /// Number of columns this segment spans.
    pub width: usize,
    /// Whether Muon owns this segment (`false` ⇒ it stays on the fallback
    /// optimizer, i.e. AdamW).
    pub muon: bool,
}

impl ProjSegment {
    /// A segment Muon orthogonalises on its own.
    pub fn muon(name: &'static str, width: usize) -> Self {
        Self { name, width, muon: true }
    }

    /// A segment left to the fallback optimizer.
    ///
    /// Used for the channels that produce *per-head scalars* rather than a
    /// feature vector (Δ, `A`, `λ`): orthogonalising a `[d_model, nheads]` slab
    /// would force the heads' step-size directions to be mutually orthogonal,
    /// which is a constraint on a gain, not on a linear map. Same reasoning as
    /// the usual "no Muon on biases, norm gains or embeddings" rule.
    pub fn adamw(name: &'static str, width: usize) -> Self {
        Self { name, width, muon: false }
    }
}

/// One 2-D weight tensor and how its columns split into independent maps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjSpec {
    /// Trailing part of the parameter path identifying the weight, e.g.
    /// `"in_proj.weight"`, anchored per [`Self::scope`].
    pub path: String,
    /// Where `path` is anchored.
    pub scope: ProjScope,
    /// The column blocks, in order; their widths must sum to the weight's output
    /// width.
    pub segments: Vec<ProjSegment>,
}

impl ProjSpec {
    /// A fused weight of the SSM block.
    pub fn block(path: impl Into<String>, segments: Vec<ProjSegment>) -> Self {
        Self { path: path.into(), scope: ProjScope::Block, segments }
    }

    /// An unfused weight of the SSM block, Muon owns it in full.
    pub fn block_whole(path: impl Into<String>, width: usize) -> Self {
        Self::block(path, vec![ProjSegment::muon("all", width)])
    }

    /// A fused weight matched by plain path substring.
    pub fn path(path: impl Into<String>, segments: Vec<ProjSegment>) -> Self {
        Self { path: path.into(), scope: ProjScope::Path, segments }
    }

    /// An unfused weight matched by plain path substring, Muon owns it in full.
    pub fn path_whole(path: impl Into<String>, width: usize) -> Self {
        Self::path(path, vec![ProjSegment::muon("all", width)])
    }

    /// Total output width (sum of the segment widths).
    pub fn width(&self) -> usize {
        self.segments.iter().map(|s| s.width).sum()
    }

    /// Whether any segment is Muon's.
    pub fn has_muon(&self) -> bool {
        self.segments.iter().any(|s| s.muon)
    }

    /// Whether Muon owns the whole tensor as a single block (so stock
    /// [`Muon`](burn::optim::Muon) applies directly, no splitting needed).
    pub fn is_whole_muon(&self) -> bool {
        self.segments.len() == 1 && self.segments[0].muon
    }

    /// The path substrings this spec matches (one per [`BLOCK_CONTAINERS`]
    /// entry for [`ProjScope::Block`], otherwise the path itself).
    pub fn predicates(&self) -> Vec<String> {
        match self.scope {
            ProjScope::Block => BLOCK_CONTAINERS
                .iter()
                .map(|container| format!("{container}{}", self.path))
                .collect(),
            ProjScope::Path => vec![self.path.clone()],
        }
    }

    /// The parameter group selecting this weight (OR over [`Self::predicates`]).
    pub fn param_group(&self) -> ParamGroup {
        ParamGroup::from_any_predicates(self.predicates())
    }
}
