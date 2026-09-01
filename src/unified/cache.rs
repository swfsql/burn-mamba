//! Runtime-tagged cache collection, and where each family plugs into the
//! block-generic stack: `impl Block`, `impl BlockConfig`, `impl CacheStack`.

use burn::prelude::*;
use burn_stack::modules::{Block, BlockConfig, CacheStack};


/// Runtime-tagged caches: one variant per family, matching
/// [`MambaLatentNet`](crate::unified::MambaLatentNet).
///
/// This is plain runtime state (not a `Module`): caches are threaded through
/// `forward`/`step`, never recorded or optimised. (`Mamba3Caches` is itself a
/// non-`Module` enum, so a `Module` derive here would not even apply.)
#[derive(Debug, Clone)]
pub enum MambaCaches {
    /// Mamba-1 caches.
    #[cfg(feature = "mamba1")]
    Mamba1(crate::mamba1::prelude::Mamba1Caches),
    /// Mamba-2 caches.
    #[cfg(feature = "mamba2")]
    Mamba2(crate::mamba2::prelude::Mamba2Caches),
    /// Mamba-3 caches.
    #[cfg(feature = "mamba3")]
    Mamba3(crate::mamba3::prelude::Mamba3Caches),
}

impl MambaCaches {
    /// Round-trip every slot of the tagged family through the inner backend:
    /// same values, no graph.
    ///
    /// What [`CacheStack::detach`] is, dispatched over the runtime tag — the
    /// enum cannot implement [`CacheStack`] itself (its slot type would have to
    /// be a fourth enum), and a caller carrying a cache across a gradient
    /// boundary (truncated BPTT: the `tiny-stories` window loop) holds the enum,
    /// not the family type.
    ///
    /// # Panics
    /// The caches must be on an autodiff device; see
    /// [`CacheStack::cache_to_inner`].
    pub fn detach(self) -> Self {
        match self {
            #[cfg(feature = "mamba1")]
            Self::Mamba1(caches) => Self::Mamba1(caches.detach()),
            #[cfg(feature = "mamba2")]
            Self::Mamba2(caches) => Self::Mamba2(caches.detach()),
            #[cfg(feature = "mamba3")]
            Self::Mamba3(caches) => Self::Mamba3(caches.detach()),
        }
    }
}

// ===========================================================================
// Per-family impls
// ===========================================================================

#[cfg(feature = "mamba2")]
mod impl_mamba2 {
    use super::*;
    use crate::mamba2::prelude::{
        Mamba2, Mamba2Cache, Mamba2CacheConfig, Mamba2Caches, Mamba2CachesConfig, Mamba2Config,
        Mamba2SsdPath,
    };

    impl CacheStack for Mamba2Caches {
        type Cache = Mamba2Cache;
        fn slot_count(&self) -> usize {
            self.caches.len()
        }
        fn into_slots(self) -> Vec<Option<Mamba2Cache>> {
            self.caches.into_iter().map(Some).collect()
        }
        fn from_slots(slots: Vec<Option<Mamba2Cache>>) -> Self {
            Self {
                caches: slots.into_iter().map(Option::unwrap).collect(),
            }
        }
        fn cache_to_inner(c: Mamba2Cache) -> Mamba2Cache {
            Mamba2Cache {
                conv_bvk: c.conv_bvk.inner(),
                ssm_bhpr: c.ssm_bhpr.inner(),
            }
        }
        fn cache_from_inner(c: Mamba2Cache) -> Mamba2Cache {
            Mamba2Cache {
                conv_bvk: Tensor::from_inner(c.conv_bvk),
                ssm_bhpr: Tensor::from_inner(c.ssm_bhpr),
            }
        }
    }

    impl Block for Mamba2 {
        type Cache = Mamba2Cache;
        type Caches = Mamba2Caches;
        type Options = Mamba2SsdPath;

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba2Cache>,
            options: Mamba2SsdPath,
        ) -> (Tensor<3>, Mamba2Cache) {
            self.forward(x, cache, options)
        }
        fn block_step(&self, x: Tensor<2>, cache: Option<Mamba2Cache>) -> (Tensor<2>, Mamba2Cache) {
            self.step(x, cache)
        }
        fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Mamba2Caches {
            let [batch, _seq, _d] = x.dims();
            self.make_zero(batch, n_virtual, &x.device())
        }
        fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Mamba2Caches {
            let [batch, _d] = x.dims();
            self.make_zero(batch, n_virtual, &x.device())
        }
    }

    impl Mamba2 {
        fn make_zero(&self, batch: usize, n_virtual: usize, device: &Device) -> Mamba2Caches {
            let [conv_dim, _, conv_kernel] = self.conv1d.weight.dims();
            Mamba2CachesConfig::new(
                n_virtual,
                Mamba2CacheConfig {
                    batch,
                    state_rank: self.state_rank,
                    conv_kernel,
                    conv_dim,
                    per_head_dim: self.per_head_dim(),
                    nheads: self.nheads(),
                },
            )
            .init(device)
        }
    }

    impl BlockConfig for Mamba2Config {
        type Block = Mamba2;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba2 {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
            self.muon_projections()
        }
    }
}

#[cfg(feature = "mamba3")]
mod impl_mamba3 {
    use super::*;
    use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCache;
    use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Caches, Mamba3Config, Mamba3SsdPath};
    use crate::mamba3::single_ssd::prelude::{
        Mamba3SingleSsdCache, Mamba3SingleSsdCacheConfig, Mamba3SingleSsdCaches,
        Mamba3SingleSsdCachesConfig,
    };

    /// Zero single-ssd caches sized from a `[batch, sequence, d_model]` input.
    /// (A missing cache defaults to the single-ssd pathway — ≈½ the SSD memory
    /// of double-ssd — for either rotation kind.)
    fn zero_single_ssd_caches(
        mamba_block: &Mamba3,
        batch: usize,
        n_virtual: usize,
        device: &Device,
    ) -> Mamba3SingleSsdCaches {
        Mamba3SingleSsdCachesConfig::new(
            n_virtual,
            Mamba3SingleSsdCacheConfig {
                batch,
                state_rank: mamba_block.state_rank,
                num_rope_angles: mamba_block.num_rope_angles,
                per_head_dim: mamba_block.per_head_dim(),
                nheads: mamba_block.nheads(),
                mimo_rank: mamba_block.mimo_rank,
                rotation: mamba_block.rotation,
                num_quat_blocks: mamba_block.num_quat_blocks,
            },
        )
        .init(device)
    }

    impl CacheStack for Mamba3Caches {
        type Cache = Mamba3Cache;
        fn slot_count(&self) -> usize {
            self.caches_len()
        }
        fn into_slots(self) -> Vec<Option<Mamba3Cache>> {
            self.into_options()
        }
        fn from_slots(slots: Vec<Option<Mamba3Cache>>) -> Self {
            Self::from_options(slots)
        }
        fn cache_to_inner(c: Mamba3Cache) -> Mamba3Cache {
            use crate::mamba3::prelude::RotationState;
            fn rot(r: RotationState) -> RotationState {
                match r {
                    RotationState::Real(u) => RotationState::Real(u),
                    RotationState::Angle(t) => RotationState::Angle(t.inner()),
                    RotationState::Quaternion(t) => RotationState::Quaternion(t.inner()),
                    RotationState::Rotor(t) => RotationState::Rotor(t.inner()),
                }
            }
            match c {
                Mamba3Cache::DoubleSsd(c) => Mamba3Cache::DoubleSsd(Mamba3DoubleSsdCache {
                    ssm_bhpr: c.ssm_bhpr.inner(),
                    k_state_bmhr: c.k_state_bmhr.inner(),
                    v_state_bhp: c.v_state_bhp.inner(),
                    rotation: rot(c.rotation),
                }),
                Mamba3Cache::SingleSsd(c) => Mamba3Cache::SingleSsd(Mamba3SingleSsdCache {
                    ssm_bhpr: c.ssm_bhpr.inner(),
                    k_state_bmhr: c.k_state_bmhr.inner(),
                    v_state_bhp: c.v_state_bhp.inner(),
                    rotation: rot(c.rotation),
                }),
            }
        }
        fn cache_from_inner(c: Mamba3Cache) -> Mamba3Cache {
            use crate::mamba3::prelude::RotationState;
            fn rot(r: RotationState) -> RotationState {
                match r {
                    RotationState::Real(u) => RotationState::Real(u),
                    RotationState::Angle(t) => RotationState::Angle(Tensor::from_inner(t)),
                    RotationState::Quaternion(t) => {
                        RotationState::Quaternion(Tensor::from_inner(t))
                    }
                    RotationState::Rotor(t) => RotationState::Rotor(Tensor::from_inner(t)),
                }
            }
            match c {
                Mamba3Cache::DoubleSsd(c) => Mamba3Cache::DoubleSsd(Mamba3DoubleSsdCache {
                    ssm_bhpr: Tensor::from_inner(c.ssm_bhpr),
                    k_state_bmhr: Tensor::from_inner(c.k_state_bmhr),
                    v_state_bhp: Tensor::from_inner(c.v_state_bhp),
                    rotation: rot(c.rotation),
                }),
                Mamba3Cache::SingleSsd(c) => Mamba3Cache::SingleSsd(Mamba3SingleSsdCache {
                    ssm_bhpr: Tensor::from_inner(c.ssm_bhpr),
                    k_state_bmhr: Tensor::from_inner(c.k_state_bmhr),
                    v_state_bhp: Tensor::from_inner(c.v_state_bhp),
                    rotation: rot(c.rotation),
                }),
            }
        }
    }

    impl Block for Mamba3 {
        type Cache = Mamba3Cache;
        type Caches = Mamba3Caches;
        type Options = Mamba3SsdPath;

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba3Cache>,
            options: Mamba3SsdPath,
        ) -> (Tensor<3>, Mamba3Cache) {
            self.forward(x, cache, options)
        }
        fn block_step(&self, x: Tensor<2>, cache: Option<Mamba3Cache>) -> (Tensor<2>, Mamba3Cache) {
            self.step(x, cache)
        }
        fn block_step_infinite(&self, x: Tensor<2>) -> Tensor<2> {
            self.step_infinite(x)
        }
        fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Mamba3Caches {
            let [batch, _seq, _d] = x.dims();
            zero_single_ssd_caches(self, batch, n_virtual, &x.device()).into()
        }
        fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Mamba3Caches {
            let [batch, _d] = x.dims();
            zero_single_ssd_caches(self, batch, n_virtual, &x.device()).into()
        }
    }

    impl BlockConfig for Mamba3Config {
        type Block = Mamba3;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba3 {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
            self.muon_projections()
        }
    }
}

#[cfg(feature = "mamba1")]
mod impl_mamba1 {
    use super::*;
    use crate::mamba1::prelude::{
        Mamba1, Mamba1Cache, Mamba1CacheConfig, Mamba1Caches, Mamba1CachesConfig, Mamba1Config,
    };

    impl CacheStack for Mamba1Caches {
        type Cache = Mamba1Cache;
        fn slot_count(&self) -> usize {
            self.caches.len()
        }
        fn into_slots(self) -> Vec<Option<Mamba1Cache>> {
            self.caches.into_iter().map(Some).collect()
        }
        fn from_slots(slots: Vec<Option<Mamba1Cache>>) -> Self {
            Self {
                caches: slots.into_iter().map(Option::unwrap).collect(),
            }
        }
        fn cache_to_inner(c: Mamba1Cache) -> Mamba1Cache {
            Mamba1Cache {
                conv_bik: c.conv_bik.inner(),
                ssm_bir: c.ssm_bir.inner(),
            }
        }
        fn cache_from_inner(c: Mamba1Cache) -> Mamba1Cache {
            Mamba1Cache {
                conv_bik: Tensor::from_inner(c.conv_bik),
                ssm_bir: Tensor::from_inner(c.ssm_bir),
            }
        }
    }

    impl Block for Mamba1 {
        type Cache = Mamba1Cache;
        type Caches = Mamba1Caches;
        /// Mamba-1 has no SSD chunking, so there is no path selector.
        type Options = ();

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba1Cache>,
            _options: (),
        ) -> (Tensor<3>, Mamba1Cache) {
            self.forward(x, cache)
        }
        fn block_step(&self, x: Tensor<2>, cache: Option<Mamba1Cache>) -> (Tensor<2>, Mamba1Cache) {
            self.step(x, cache)
        }
        fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Mamba1Caches {
            let [batch, _seq, _d] = x.dims();
            self.make_zero(batch, n_virtual, &x.device())
        }
        fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Mamba1Caches {
            let [batch, _d] = x.dims();
            self.make_zero(batch, n_virtual, &x.device())
        }
    }

    impl Mamba1 {
        fn cache_config(&self, batch: usize) -> Mamba1CacheConfig {
            let [d_inner, state_rank] = self.a_log.dims();
            let [_, _, conv_kernel] = self.conv1d.weight.dims();
            Mamba1CacheConfig::new(batch, d_inner)
                .with_state_rank(state_rank)
                .with_conv_kernel(conv_kernel)
        }
        fn make_zero(&self, batch: usize, n_virtual: usize, device: &Device) -> Mamba1Caches {
            Mamba1CachesConfig::new(n_virtual, self.cache_config(batch)).init(device)
        }
    }

    impl BlockConfig for Mamba1Config {
        type Block = Mamba1;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba1 {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
            self.muon_projections()
        }
    }
}
