//! The place where each family connects to the block-generic stack: `impl
//! Block`, `impl BlockConfig`, `impl CacheStack`.
//!
//! Each `CacheStack` impl holds only the slot logic of its family. The backend
//! hop of a cache follows from its cache traversal in
//! [`crate::family::cache_traversal`].
//!
//! Three trait methods have the name of an inherent method of their family:
//! `zero_cache`, `untied_params` and `muon_projections`. Each one calls that
//! inherent method by its type path, for example
//! `Mamba1::zero_cache(self, batch, device)` and
//! `Mamba1Config::muon_projections(self)`. A call on `self` selects the same
//! method, but it reads like an endless recursion.

use burn::prelude::*;
use burn_stack::modules::{Block, BlockConfig, CacheStack};
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
        fn zero_cache(&self, batch: usize, device: &Device) -> Mamba1Cache {
            Mamba1::zero_cache(self, batch, device)
        }
        fn untied_params(&self) -> Vec<UntiedParam> {
            Mamba1::untied_params(self)
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
            Mamba1Config::muon_projections(self)
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
        fn zero_cache(&self, batch: usize, device: &Device) -> Mamba2Cache {
            Mamba2::zero_cache(self, batch, device)
        }
        fn untied_params(&self) -> Vec<UntiedParam> {
            Mamba2::untied_params(self)
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
            Mamba2Config::muon_projections(self)
        }
    }
}

#[cfg(feature = "mamba3")]
mod impl_mamba3 {
    use super::*;
    use crate::mamba3::double_ssd::prelude::Mamba3DoubleSsdCaches;
    use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Caches, Mamba3Config, Mamba3SsdPath};
    use crate::mamba3::single_ssd::prelude::Mamba3SingleSsdCaches;

    impl CacheStack for Mamba3Caches {
        type Cache = Mamba3Cache;
        /// The number of slots, for both pathways.
        fn slot_count(&self) -> usize {
            match self {
                Self::DoubleSsd(caches) => caches.caches.len(),
                Self::SingleSsd(caches) => caches.caches.len(),
            }
        }
        fn into_slots(self) -> Vec<Option<Mamba3Cache>> {
            match self {
                Self::DoubleSsd(caches) => caches
                    .caches
                    .into_iter()
                    .map(Mamba3Cache::from)
                    .map(Some)
                    .collect(),
                Self::SingleSsd(caches) => caches
                    .caches
                    .into_iter()
                    .map(Mamba3Cache::from)
                    .map(Some)
                    .collect(),
            }
        }
        /// The first slot gives the pathway. An empty list gives single-SSD
        /// caches.
        ///
        /// # Panics
        ///
        /// If a slot is empty, or if the slots mix the two pathways.
        fn from_slots(slots: Vec<Option<Mamba3Cache>>) -> Self {
            let caches: Vec<Mamba3Cache> = slots.into_iter().map(Option::unwrap).collect();
            if matches!(caches.first(), Some(Mamba3Cache::DoubleSsd(_))) {
                Mamba3DoubleSsdCaches {
                    caches: caches
                        .into_iter()
                        .map(Mamba3Cache::double_ssd)
                        .map(Option::unwrap)
                        .collect(),
                }
                .into()
            } else {
                Mamba3SingleSsdCaches {
                    caches: caches
                        .into_iter()
                        .map(Mamba3Cache::single_ssd)
                        .map(Option::unwrap)
                        .collect(),
                }
                .into()
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
        /// A single-SSD cache, because a missing cache selects that pathway.
        /// It holds about half the SSD memory of a double-SSD cache, for every
        /// rotation kind.
        fn zero_cache(&self, batch: usize, device: &Device) -> Mamba3Cache {
            Mamba3::zero_cache(self, batch, device).into()
        }
        fn untied_params(&self) -> Vec<UntiedParam> {
            Mamba3::untied_params(self)
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
            Mamba3Config::muon_projections(self)
        }
    }
}
