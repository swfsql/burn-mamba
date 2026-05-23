//! # Mamba-3 Forward Pathway Selection
//!
//! A runtime knob — analogous to [`Mamba3SsdPath`] — that selects which
//! full-sequence algorithm the layer stack / network runs:
//!
//! - [`Mamba3Pathway::DoubleSsd`] — the original two-SSD decomposition
//!   ([`Mamba3Layers::forward`] / [`Mamba3Network::forward`]). Carries a
//!   [`Mamba3SsdPath`] for the inner SSD algorithm + chunk length.
//! - [`Mamba3Pathway::Trap`] — the merged-form single-pass trapezoidal
//!   algorithm ([`Mamba3Layers::forward2`] / [`Mamba3Network::forward2`]).
//!   Carries a [`Mamba3TrapSsdPath`].
//!
//! The two pathways use **different cache types** (`Mamba3Caches` vs
//! `Mamba3MergedCaches`) because the stored SSM accumulator has different
//! semantics — see [`crate::mamba3::cache_v2`]. To keep a single dispatch entry
//! point, caches cross the pathway boundary wrapped in [`Mamba3AnyCaches`].
//!
//! ## Why a wrapper enum instead of one cache type?
//!
//! `forward`'s cache holds the original trapezoidal hidden state `h`; `forward2`'s
//! holds the merged accumulator `h'`. Feeding one into the other mid-sequence
//! silently corrupts state, so they are distinct types. [`Mamba3AnyCaches`]
//! tags which variant is held; [`Mamba3Layers::forward_pathway`] /
//! [`Mamba3Network::forward_pathway`] assert the tag matches the chosen pathway.
//!
//! ## Decode
//!
//! There is no pathway knob for `step`: token-by-token decoding always uses the
//! recurrent form ([`Mamba3Caches`]). When continuing decode after a `Trap`
//! prefill, the merged accumulator would first need converting back to the
//! original form; that conversion is intentionally not provided here.

use crate::mamba3::prelude::*;
use burn::prelude::*;

/// Selects the full-sequence forward algorithm (the "pathway"), bundling the
/// pathway choice with its corresponding SSD-path selector.
///
/// Mirrors how [`Mamba3SsdPath`] selects the inner SSD algorithm: pass it to
/// [`Mamba3Layers::forward_pathway`] / [`Mamba3Network::forward_pathway`].
#[derive(Debug, Clone)]
pub enum Mamba3Pathway {
    /// Original two-SSD (β-term + γ-term) decomposition.
    DoubleSsd(Mamba3SsdPath),
    /// Merged-form single-pass trapezoidal SSD.
    Trap(Mamba3TrapSsdPath),
}

impl Default for Mamba3Pathway {
    fn default() -> Self {
        Mamba3Pathway::DoubleSsd(Mamba3SsdPath::default())
    }
}

/// A pathway-tagged bundle of per-layer caches, so a single dispatch entry can
/// accept / return either cache family.
#[derive(Debug)]
pub enum Mamba3AnyCaches<B: Backend> {
    /// Caches for the [`Mamba3Pathway::DoubleSsd`] pathway.
    DoubleSsd(Mamba3Caches<B>),
    /// Caches for the [`Mamba3Pathway::Trap`] pathway.
    Trap(Mamba3MergedCaches<B>),
}

impl<B: Backend> Mamba3AnyCaches<B> {
    /// Unwrap the double-SSD caches, panicking if this holds trap caches.
    pub fn expect_double_ssd(self) -> Mamba3Caches<B> {
        match self {
            Mamba3AnyCaches::DoubleSsd(c) => c,
            Mamba3AnyCaches::Trap(_) => {
                panic!("expected DoubleSsd caches but found Trap caches")
            }
        }
    }

    /// Unwrap the trap caches, panicking if this holds double-SSD caches.
    pub fn expect_trap(self) -> Mamba3MergedCaches<B> {
        match self {
            Mamba3AnyCaches::Trap(c) => c,
            Mamba3AnyCaches::DoubleSsd(_) => {
                panic!("expected Trap caches but found DoubleSsd caches")
            }
        }
    }
}

impl<B: Backend + Mamba3BackendExt + Mamba3TrapBackendExt> Mamba3Layers<B> {
    /// Run the layer stack under the selected [`Mamba3Pathway`].
    ///
    /// Dispatches to [`Self::forward`] ([`Mamba3Pathway::DoubleSsd`]) or
    /// [`Self::forward2`] ([`Mamba3Pathway::Trap`]) and wraps the resulting
    /// caches in [`Mamba3AnyCaches`]. When `caches` is `Some`, its tag must
    /// match the chosen pathway (otherwise it panics — mixing accumulator
    /// semantics is a bug).
    ///
    /// # Arguments
    /// - `x` — input tensor, shape `[batch, sequence, d_model]`
    /// - `caches` — optional pathway-tagged caches (`None` → zero-init)
    /// - `pathway` — the algorithm selection knob
    pub fn forward_pathway(
        &self,
        x: Tensor<B, 3>,
        caches: Option<Mamba3AnyCaches<B>>,
        pathway: Mamba3Pathway,
    ) -> (Tensor<B, 3>, Mamba3AnyCaches<B>) {
        match pathway {
            Mamba3Pathway::DoubleSsd(ssd_path) => {
                let caches = caches.map(Mamba3AnyCaches::expect_double_ssd);
                let (out, caches) = self.forward(x, caches, ssd_path);
                (out, Mamba3AnyCaches::DoubleSsd(caches))
            }
            Mamba3Pathway::Trap(trap_path) => {
                let caches = caches.map(Mamba3AnyCaches::expect_trap);
                let (out, caches) = self.forward2(x, caches, trap_path);
                (out, Mamba3AnyCaches::Trap(caches))
            }
        }
    }
}

impl<B: Backend + Mamba3BackendExt + Mamba3TrapBackendExt> Mamba3Network<B> {
    /// Run the full network under the selected [`Mamba3Pathway`].
    ///
    /// Dispatches to [`Self::forward`] ([`Mamba3Pathway::DoubleSsd`]) or
    /// [`Self::forward2`] ([`Mamba3Pathway::Trap`]); see
    /// [`Mamba3Layers::forward_pathway`] for the cache-tag rules.
    ///
    /// # Arguments
    /// - `x` — integer token IDs, shape `[batch, sequence]`
    /// - `caches` — optional pathway-tagged caches (`None` → zero-init)
    /// - `pathway` — the algorithm selection knob
    pub fn forward_pathway(
        &self,
        x: Tensor<B, 2, Int>,
        caches: Option<Mamba3AnyCaches<B>>,
        pathway: Mamba3Pathway,
    ) -> (Tensor<B, 3>, Mamba3AnyCaches<B>) {
        match pathway {
            Mamba3Pathway::DoubleSsd(ssd_path) => {
                let caches = caches.map(Mamba3AnyCaches::expect_double_ssd);
                let (out, caches) = self.forward(x, caches, ssd_path);
                (out, Mamba3AnyCaches::DoubleSsd(caches))
            }
            Mamba3Pathway::Trap(trap_path) => {
                let caches = caches.map(Mamba3AnyCaches::expect_trap);
                let (out, caches) = self.forward2(x, caches, trap_path);
                (out, Mamba3AnyCaches::Trap(caches))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — the pathway knob dispatches to forward / forward2 and the two
// pathways agree (they are reformulations of the same recurrence).
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "backend-flex"))]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, Flex};
    use burn::tensor::{Distribution, Int};

    type InnerB = Flex;
    type B = Autodiff<InnerB>;
    type Device = <InnerB as burn::tensor::backend::BackendTypes>::Device;

    fn small_block() -> Mamba3Config {
        Mamba3Config::new(32)
            .with_state_rank(8)
            .with_expand(2)
            .with_per_head_dim(8)
    }

    fn small_network_cfg() -> Mamba3NetworkConfig {
        Mamba3NetworkConfig::new(2, 64, 16, small_block(), false)
    }

    /// The DoubleSsd and Trap pathways are reformulations of the same
    /// recurrence, so `forward_pathway` outputs must agree across the two
    /// knob settings (network level).
    #[test]
    fn network_pathway_double_matches_trap() {
        let device: Device = Default::default();
        let cfg = small_network_cfg();
        let model = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 6;
        let tokens = Tensor::<InnerB, 2, Int>::random(
            [batch, seq_len],
            Distribution::Uniform(0.0, cfg.vocab_size as f64),
            &device,
        );
        let tokens = Tensor::<B, 2, Int>::from_inner(tokens);

        let (out_double, caches_double) = model.forward_pathway(
            tokens.clone(),
            None,
            Mamba3Pathway::DoubleSsd(Mamba3SsdPath::Minimal(Some(4))),
        );
        let (out_trap, caches_trap) = model.forward_pathway(
            tokens,
            None,
            Mamba3Pathway::Trap(Mamba3TrapSsdPath::Minimal(Some(4))),
        );

        // Tags must match the chosen pathway.
        assert!(matches!(caches_double, Mamba3AnyCaches::DoubleSsd(_)));
        assert!(matches!(caches_trap, Mamba3AnyCaches::Trap(_)));

        let diff = (out_double.inner() - out_trap.inner())
            .abs()
            .max()
            .into_scalar();
        assert!(
            diff < 1e-4,
            "pathway double vs trap (network) max abs diff = {diff:.6} (expected < 1e-4)"
        );
    }

    /// Same agreement check at the layer-stack level, also exercising a
    /// continued (cache-carrying) call through the knob.
    #[test]
    fn layers_pathway_split_matches_full() {
        let device: Device = Default::default();
        let cfg = Mamba3LayersConfig::new(2, small_block());
        let layers = cfg.init::<B>(&device);

        let batch = 2;
        let seq_len = 6;
        let split = 2;
        let d_model = cfg.mamba_block.d_model;

        let x = Tensor::<InnerB, 3>::random(
            [batch, seq_len, d_model],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let x = Tensor::<B, 3>::from_inner(x);

        let trap = || Mamba3Pathway::Trap(Mamba3TrapSsdPath::SerialRecalculated(Some(4)));

        // Full pass.
        let (out_full, _) = layers.forward_pathway(x.clone(), None, trap());

        // Split prefill then continue with carried caches.
        let prefix = x.clone().narrow(1, 0, split);
        let suffix = x.narrow(1, split, seq_len - split);
        let (out_prefix, caches) = layers.forward_pathway(prefix, None, trap());
        let (out_suffix, _) = layers.forward_pathway(suffix, Some(caches), trap());
        let out_split = Tensor::cat(vec![out_prefix, out_suffix], 1);

        let diff = (out_full.inner() - out_split.inner())
            .abs()
            .max()
            .into_scalar();
        assert!(
            diff < 1e-4,
            "pathway split vs full (layers) max abs diff = {diff:.6} (expected < 1e-4)"
        );
    }

    /// Feeding a cache whose tag mismatches the chosen pathway must panic
    /// (mixing accumulator semantics is a bug).
    #[test]
    #[should_panic(expected = "expected Trap caches")]
    fn pathway_cache_tag_mismatch_panics() {
        let device: Device = Default::default();
        let cfg = Mamba3LayersConfig::new(2, small_block());
        let layers = cfg.init::<B>(&device);

        let x = Tensor::<B, 3>::from_inner(Tensor::<InnerB, 3>::random(
            [2, 4, cfg.mamba_block.d_model],
            Distribution::Normal(0.0, 1.0),
            &device,
        ));

        // Produce DoubleSsd caches, then wrongly feed them to a Trap call.
        let (_, double_caches) = layers.forward_pathway(
            x.clone(),
            None,
            Mamba3Pathway::DoubleSsd(Mamba3SsdPath::Minimal(Some(4))),
        );
        let _ = layers.forward_pathway(
            x,
            Some(double_caches),
            Mamba3Pathway::Trap(Mamba3TrapSsdPath::Minimal(Some(4))),
        );
    }
}
