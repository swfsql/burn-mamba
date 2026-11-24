//! The place where each family connects to the block-generic stack: `impl
//! Block` and `impl BlockConfig`.
//!
//! A block always gets a cache. Each `Block` call gives `Some(cache)` to the
//! family API. The backend hop of a cache follows from its cache traversal in
//! [`crate::family::cache_traversal`].
//!
//! Three trait methods have the name of an inherent method of their family:
//! `zero_cache`, `untied_params` and `muon_projections`. Each one calls that
//! inherent method by its type path, for example
//! `Mamba1::zero_cache(self, batch, device)` and
//! `Mamba1Config::muon_projections(self)`. A call on `self` selects the same
//! method, but it reads like an endless recursion.

use burn::prelude::*;
use burn_stack::modules::{Block, BlockConfig};
use burn_stack::utils::UntiedParam;

#[cfg(feature = "mamba1")]
mod impl_mamba1 {
    use super::*;
    use crate::mamba1::prelude::{Mamba1, Mamba1Cache, Mamba1Config};

    impl Block for Mamba1 {
        type Cache = Mamba1Cache;
        /// Mamba-1 has no SSD chunking, so it has no scan options.
        type Options = ();

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Mamba1Cache,
            _options: (),
            pad: Option<Tensor<2, Bool>>,
        ) -> (Tensor<3>, Mamba1Cache) {
            self.forward(x, Some(cache), pad)
        }
        fn block_step(&self, x: Tensor<2>, cache: Mamba1Cache) -> (Tensor<2>, Mamba1Cache) {
            self.step(x, Some(cache))
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
        fn init_block(&self, device: &Device) -> Mamba1 {
            self.init(device)
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
    use crate::mamba2::prelude::{Mamba2, Mamba2Cache, Mamba2Config, Mamba2ScanOptions};
    use crate::ssd_path::SsdPath;

    impl Block for Mamba2 {
        type Cache = Mamba2Cache;
        type Options = Mamba2ScanOptions;

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Mamba2Cache,
            options: Mamba2ScanOptions,
            pad: Option<Tensor<2, Bool>>,
        ) -> (Tensor<3>, Mamba2Cache) {
            self.forward(x, Some(cache), options, pad)
        }
        /// A reset only at a chunk start (see [`crate::packing`]).
        fn block_forward_packed(
            &self,
            x: Tensor<3>,
            cache: Mamba2Cache,
            origin: Mamba2Cache,
            options: Mamba2ScanOptions,
            reset: Tensor<2, Bool>,
        ) -> (Tensor<3>, Mamba2Cache) {
            self.forward_packed(x, Some(cache), Some(origin), options, None, Some(reset))
        }
        /// The chunk length of `options`, in tokens. `None` for
        /// [`SsdPath::Minimal`], which takes no resets.
        fn reset_align(&self, options: &Mamba2ScanOptions) -> Option<usize> {
            if options.ssd_path == SsdPath::Minimal {
                None
            } else {
                Some(options.chunk_len_or_optimal(self.state_rank, self.per_head_dim()))
            }
        }
        fn block_step(&self, x: Tensor<2>, cache: Mamba2Cache) -> (Tensor<2>, Mamba2Cache) {
            self.step(x, Some(cache))
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
        fn init_block(&self, device: &Device) -> Mamba2 {
            self.init(device)
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
    use crate::mamba3::prelude::{Mamba3, Mamba3Cache, Mamba3Config, Mamba3ScanOptions};
    use crate::ssd_path::SsdPath;

    impl Block for Mamba3 {
        type Cache = Mamba3Cache;
        type Options = Mamba3ScanOptions;

        fn block_forward(
            &self,
            x: Tensor<3>,
            cache: Mamba3Cache,
            options: Mamba3ScanOptions,
            pad: Option<Tensor<2, Bool>>,
        ) -> (Tensor<3>, Mamba3Cache) {
            self.forward(x, Some(cache), options, pad)
        }
        /// A reset only at a chunk start: every `chunk_len / micro_steps`
        /// tokens (see [`crate::packing`]).
        fn block_forward_packed(
            &self,
            x: Tensor<3>,
            cache: Mamba3Cache,
            origin: Mamba3Cache,
            options: Mamba3ScanOptions,
            reset: Tensor<2, Bool>,
        ) -> (Tensor<3>, Mamba3Cache) {
            self.forward_packed(x, Some(cache), Some(origin), options, None, Some(reset))
        }
        /// The chunk length of `options`, in tokens: `chunk_len / micro_steps`.
        /// `None` for [`SsdPath::Minimal`], which takes no resets.
        fn reset_align(&self, options: &Mamba3ScanOptions) -> Option<usize> {
            if options.ssd_path == SsdPath::Minimal {
                None
            } else {
                Some(crate::standard_ssd::chunk_tokens(options.chunk_len_or_optimal(self), self.micro_steps))
            }
        }
        fn block_step(&self, x: Tensor<2>, cache: Mamba3Cache) -> (Tensor<2>, Mamba3Cache) {
            self.step(x, Some(cache))
        }
        fn zero_cache(&self, batch: usize, device: &Device) -> Mamba3Cache {
            Mamba3::zero_cache(self, batch, device)
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
        fn init_block(&self, device: &Device) -> Mamba3 {
            self.init(device)
        }
        #[cfg(feature = "optim")]
        fn muon_projections(&self) -> Vec<burn_stack::optim::ProjSpec> {
            Mamba3Config::muon_projections(self)
        }
    }
}
