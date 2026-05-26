pub mod double_ssd;
pub mod single_ssd;

pub mod bidi;
pub mod cache;
pub(crate) mod helpers;
pub mod layer;
pub mod mamba3;
pub mod network;
pub mod ssd_path;

use crate::mamba3::double_ssd::prelude::*;
use crate::mamba3::single_ssd::prelude::*;
use burn::tensor::backend::Backend;

pub trait Mamba3BackendExt:
    burn::tensor::backend::Backend + Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt
{
}

crate::decl_ssd_autodiff_backend_ext!(
    Mamba3AutodiffBackendExt,
    Mamba3BackendExt,
    Mamba3DoubleSsdAutodiffBackendExt,
    Mamba3SingleSsdAutodiffBackendExt
);
crate::impl_ssd_backend_ext_for_burn_backends!(Mamba3BackendExt);

#[cfg(feature = "autodiff")]
pub mod backwards {
    use super::*;
    use burn::backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};

    impl<B: Backend + Mamba3DoubleSsdBackendExt + Mamba3SingleSsdBackendExt, C: CheckpointStrategy>
        Mamba3BackendExt for Autodiff<B, C>
    {
    }
}

pub mod prelude {
    #[cfg(feature = "autodiff")]
    pub use super::Mamba3AutodiffBackendExt;
    pub use super::Mamba3BackendExt;
    use super::*;

    pub use cache::{Mamba3Cache, Mamba3Caches};
    pub use layer::{Mamba3Layer, Mamba3LayerConfig, Mamba3Layers, Mamba3LayersConfig};
    pub use mamba3::{Mamba3, Mamba3Config};
    pub use network::{Mamba3Network, Mamba3NetworkConfig};
    pub use ssd_path::Mamba3SsdPath;
}
