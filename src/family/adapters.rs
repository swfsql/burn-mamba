//! The place where each family connects to the block-generic stack: `impl
//! Block`, `impl BlockConfig`, `impl CacheStack`.
//!
//! Each family implements `CacheStack::cache_to_inner`/`cache_from_inner` by
//! hand. `Module::map` does nothing on plain `Tensor` fields, which is all a
//! cache holds, so a `Module`-based conversion would silently skip every field.
//! `cache_from_inner` lifts each tensor with `burn_stack::modules::lift`, not a
//! bare `Tensor::from_inner` (see `lift`).

use burn::prelude::*;
use burn_stack::modules::{Block, BlockConfig, CacheStack, lift};
use burn_stack::utils::UntiedParam;

#[cfg(feature = "mamba1")]
mod impl_mamba1 {
    use super::*;
    use crate::mamba1::prelude::{Mamba1, Mamba1Cache, Mamba1Caches, Mamba1Config};

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
        fn cache_from_inner(c: Mamba1Cache, device: &Device) -> Mamba1Cache {
            Mamba1Cache {
                conv_bik: lift(c.conv_bik, device),
                ssm_bir: lift(c.ssm_bir, device),
            }
        }
        fn cache_device(c: &Mamba1Cache) -> Device {
            c.ssm_bir.device()
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
            pad: Option<Tensor<2, Bool>>,
        ) -> (Tensor<3>, Mamba1Cache) {
            self.forward(x, cache, pad)
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
        fn untied_params(&self) -> Vec<UntiedParam> {
            self.untied_params()
        }
    }

    impl Mamba1 {
        fn make_zero(&self, batch: usize, n_virtual: usize, device: &Device) -> Mamba1Caches {
            let caches = (0..n_virtual).map(|_| self.zero_cache(batch, device)).collect();
            Mamba1Caches { caches }
        }
    }

    impl BlockConfig for Mamba1Config {
        type Block = Mamba1;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, n_applications: usize, device: &Device) -> Mamba1 {
            self.init_applications(n_applications, device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
            self.muon_projections()
        }
    }
}

#[cfg(feature = "mamba2")]
mod impl_mamba2 {
    use super::*;
    use crate::mamba2::prelude::{Mamba2, Mamba2Cache, Mamba2Caches, Mamba2Config, Mamba2SsdPath};

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
        fn cache_from_inner(c: Mamba2Cache, device: &Device) -> Mamba2Cache {
            Mamba2Cache {
                conv_bvk: lift(c.conv_bvk, device),
                ssm_bhpr: lift(c.ssm_bhpr, device),
            }
        }
        fn cache_device(c: &Mamba2Cache) -> Device {
            c.ssm_bhpr.device()
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
            pad: Option<Tensor<2, Bool>>,
        ) -> (Tensor<3>, Mamba2Cache) {
            self.forward(x, cache, options, pad)
        }
        /// A reset only at a chunk start (see [`crate::packing`]).
        fn block_forward_packed(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba2Cache>,
            options: Mamba2SsdPath,
            reset: Tensor<2, Bool>,
        ) -> (Tensor<3>, Mamba2Cache) {
            self.forward_packed(x, cache, None, options, None, Some(reset))
        }
        /// The chunk length of `options`, in tokens. `None` for
        /// [`Mamba2SsdPath::Minimal`], which takes no resets.
        fn reset_align(&self, options: &Mamba2SsdPath) -> Option<usize> {
            match options {
                Mamba2SsdPath::Minimal(_) => None,
                _ => Some(options.chunk_len_or_optimal(self.state_rank, self.per_head_dim())),
            }
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
        fn untied_params(&self) -> Vec<UntiedParam> {
            self.untied_params()
        }
    }

    impl Mamba2 {
        fn make_zero(&self, batch: usize, n_virtual: usize, device: &Device) -> Mamba2Caches {
            let caches = (0..n_virtual).map(|_| self.zero_cache(batch, device)).collect();
            Mamba2Caches { caches }
        }
    }

    impl BlockConfig for Mamba2Config {
        type Block = Mamba2;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, n_applications: usize, device: &Device) -> Mamba2 {
            self.init_applications(n_applications, device)
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
    use crate::mamba3::single_ssd::prelude::{Mamba3SingleSsdCache, Mamba3SingleSsdCaches};

    /// Zero single-ssd caches for a batch of `batch`. (A missing cache selects
    /// the single-ssd pathway, with ≈½ the SSD memory of double-ssd, for every
    /// rotation kind.)
    fn zero_single_ssd_caches(
        mamba_block: &Mamba3,
        batch: usize,
        n_virtual: usize,
        device: &Device,
    ) -> Mamba3SingleSsdCaches {
        let caches = (0..n_virtual).map(|_| mamba_block.zero_cache(batch, device)).collect();
        Mamba3SingleSsdCaches { caches }
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
            use crate::mamba3::prelude::{Mamba3FrameCache, RotationState, TapFifo};
            fn rot(r: RotationState) -> RotationState {
                match r {
                    RotationState::Real(u) => RotationState::Real(u),
                    RotationState::Angle(t) => RotationState::Angle(t.inner()),
                    RotationState::Quaternion(t) => RotationState::Quaternion(t.inner()),
                    RotationState::Rotor(t) => RotationState::Rotor(t.inner()),
                }
            }
            fn fifo(f: TapFifo) -> TapFifo {
                TapFifo { b_bumhr: f.b_bumhr.inner(), x_buhp: f.x_buhp.inner() }
            }
            fn frame(f: Mamba3FrameCache) -> Mamba3FrameCache {
                Mamba3FrameCache {
                    tap_fifo: f.tap_fifo.map(fifo),
                    rotation: rot(f.rotation),
                    log_precision_bh: f.log_precision_bh.map(|t| t.inner()),
                    tropical_bh: f.tropical_bh.map(|t| t.inner()),
                }
            }
            match c {
                Mamba3Cache::DoubleSsd(c) => Mamba3Cache::DoubleSsd(Mamba3DoubleSsdCache {
                    ssm_bhpr: c.ssm_bhpr.inner(),
                    frame: frame(c.frame),
                }),
                Mamba3Cache::SingleSsd(c) => Mamba3Cache::SingleSsd(Mamba3SingleSsdCache {
                    ssm_bhpr: c.ssm_bhpr.inner(),
                    frame: frame(c.frame),
                }),
            }
        }
        fn cache_from_inner(c: Mamba3Cache, device: &Device) -> Mamba3Cache {
            use crate::mamba3::prelude::{Mamba3FrameCache, RotationState, TapFifo};
            fn rot(r: RotationState, device: &Device) -> RotationState {
                match r {
                    RotationState::Real(u) => RotationState::Real(u),
                    RotationState::Angle(t) => RotationState::Angle(lift(t, device)),
                    RotationState::Quaternion(t) => RotationState::Quaternion(lift(t, device)),
                    RotationState::Rotor(t) => RotationState::Rotor(lift(t, device)),
                }
            }
            fn fifo(f: TapFifo, device: &Device) -> TapFifo {
                TapFifo { b_bumhr: lift(f.b_bumhr, device), x_buhp: lift(f.x_buhp, device) }
            }
            fn frame(f: Mamba3FrameCache, device: &Device) -> Mamba3FrameCache {
                Mamba3FrameCache {
                    tap_fifo: f.tap_fifo.map(|f| fifo(f, device)),
                    rotation: rot(f.rotation, device),
                    log_precision_bh: f.log_precision_bh.map(|t| lift(t, device)),
                    tropical_bh: f.tropical_bh.map(|t| lift(t, device)),
                }
            }
            match c {
                Mamba3Cache::DoubleSsd(c) => Mamba3Cache::DoubleSsd(Mamba3DoubleSsdCache {
                    ssm_bhpr: lift(c.ssm_bhpr, device),
                    frame: frame(c.frame, device),
                }),
                Mamba3Cache::SingleSsd(c) => Mamba3Cache::SingleSsd(Mamba3SingleSsdCache {
                    ssm_bhpr: lift(c.ssm_bhpr, device),
                    frame: frame(c.frame, device),
                }),
            }
        }
        fn cache_device(c: &Mamba3Cache) -> Device {
            match c {
                Mamba3Cache::DoubleSsd(c) => c.ssm_bhpr.device(),
                Mamba3Cache::SingleSsd(c) => c.ssm_bhpr.device(),
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
            pad: Option<Tensor<2, Bool>>,
        ) -> (Tensor<3>, Mamba3Cache) {
            self.forward(x, cache, options, pad)
        }
        /// A reset only at a chunk start: every `chunk_len / micro_steps`
        /// tokens (see [`crate::packing`]).
        fn block_forward_packed(
            &self,
            x: Tensor<3>,
            cache: Option<Mamba3Cache>,
            options: Mamba3SsdPath,
            reset: Tensor<2, Bool>,
        ) -> (Tensor<3>, Mamba3Cache) {
            self.forward_packed(x, cache, None, options, None, Some(reset))
        }
        /// The chunk length of `options`, in tokens: `chunk_len / micro_steps`.
        /// `None` for [`Mamba3SsdPath::Minimal`], which takes no resets.
        fn reset_align(&self, options: &Mamba3SsdPath) -> Option<usize> {
            match options {
                Mamba3SsdPath::Minimal(_) => None,
                _ => Some(crate::standard_ssd::chunk_tokens(options.chunk_len_or_optimal(self), self.micro_steps)),
            }
        }
        fn block_step(&self, x: Tensor<2>, cache: Option<Mamba3Cache>) -> (Tensor<2>, Mamba3Cache) {
            self.step(x, cache)
        }
        fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> Mamba3Caches {
            let [batch, _seq, _d] = x.dims();
            zero_single_ssd_caches(self, batch, n_virtual, &x.device()).into()
        }
        fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> Mamba3Caches {
            let [batch, _d] = x.dims();
            zero_single_ssd_caches(self, batch, n_virtual, &x.device()).into()
        }
        fn untied_params(&self) -> Vec<UntiedParam> {
            self.untied_params()
        }
    }

    impl BlockConfig for Mamba3Config {
        type Block = Mamba3;
        fn d_model(&self) -> usize {
            self.d_model
        }
        fn init_block(&self, n_applications: usize, device: &Device) -> Mamba3 {
            self.init_applications(n_applications, device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
            self.muon_projections()
        }
    }
}
