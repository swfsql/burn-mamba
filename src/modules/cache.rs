use crate::prelude::*;
use burn::prelude::*;

// ===========================================================================
// Unifying enums: one runtime + one serializable Config across all families
// ===========================================================================

/// The uniform interface a per-network cache collection exposes for the generic
/// [`Layers`] loop. The existing `Mamba{1,2,3}Caches` already provide it (under
/// `caches_len`/`into_options`/`from_options`).
pub trait CacheStack: Sized {
    /// The per-layer cache element.
    type Cache;
    /// Number of per-(virtual-)layer slots.
    fn slot_count(&self) -> usize;
    /// Move each slot into an `Option` so the loop can `take` without cloning.
    fn into_slots(self) -> Vec<Option<Self::Cache>>;
    /// Inverse of [`Self::into_slots`].
    fn from_slots(slots: Vec<Option<Self::Cache>>) -> Self;
    /// Move one cache slot **to** the inner (non-autodiff) backend.
    ///
    /// Needed by [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon),
    /// whose no-grad prefix runs on the inner backend: a cache carried in from a
    /// tracked segment has to come down with it, both so the prefix builds no
    /// graph and because Burn's dispatch cannot mix backends within one op.
    ///
    /// Spelled out per family rather than derived, because
    /// [`Module::map`] is a **no-op on plain `Tensor`
    /// fields** (Burn implements `Module for Tensor` as a constant) and caches
    /// hold bare tensors, not `Param`s — a `Module`-based conversion would
    /// silently skip every one of them.
    ///
    /// # Panics
    /// `Tensor::inner` panics on a tensor that is already off the autodiff
    /// backend, so the caller must have checked
    /// [`Device::is_autodiff`](burn::prelude::Device::is_autodiff) first.
    fn cache_to_inner(cache: Self::Cache) -> Self::Cache;

    /// Lift one cache slot back **from** the inner backend, as a fresh graph
    /// root. The inverse of [`Self::cache_to_inner`]; see its notes.
    fn cache_from_inner(cache: Self::Cache) -> Self::Cache;
}

/// Runtime-tagged caches: one variant per family, matching [`MambaLatentNet`].
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

    impl MambaBlock for Mamba2 {
        type Cache = Mamba2Cache;
        type Caches = Mamba2Caches;
        type SsdPath = Mamba2SsdPath;

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba2Cache>,
            ssd_path: Mamba2SsdPath,
        ) -> (Tensor<3>, Mamba2Cache) {
            self.forward(x, cache, ssd_path)
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

    impl MambaBlockConfig for Mamba2Config {
        type Block = Mamba2;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba2 {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<crate::optim::ProjSpec> {
            self.muon_projections()
        }
    }
}

#[cfg(feature = "mamba3")]
mod impl_mamba3 {
    use super::*;
    use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Caches, Mamba3Config, Mamba3SsdPath};
    use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCache;
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
                    RotationState::Angle(t) => RotationState::Angle(t.inner()),
                    RotationState::Quaternion(t) => RotationState::Quaternion(t.inner()),
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
                    RotationState::Angle(t) => RotationState::Angle(Tensor::from_inner(t)),
                    RotationState::Quaternion(t) => {
                        RotationState::Quaternion(Tensor::from_inner(t))
                    }
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

    impl MambaBlock for Mamba3 {
        type Cache = Mamba3Cache;
        type Caches = Mamba3Caches;
        type SsdPath = Mamba3SsdPath;

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba3Cache>,
            ssd_path: Mamba3SsdPath,
        ) -> (Tensor<3>, Mamba3Cache) {
            self.forward(x, cache, ssd_path)
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

    impl MambaBlockConfig for Mamba3Config {
        type Block = Mamba3;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba3 {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<crate::optim::ProjSpec> {
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

    impl MambaBlock for Mamba1 {
        type Cache = Mamba1Cache;
        type Caches = Mamba1Caches;
        /// Mamba-1 has no SSD chunking, so there is no path selector.
        type SsdPath = ();

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba1Cache>,
            _ssd_path: (),
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

    impl MambaBlockConfig for Mamba1Config {
        type Block = Mamba1;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, device: &Device) -> Mamba1 {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<crate::optim::ProjSpec> {
            self.muon_projections()
        }
    }
}
